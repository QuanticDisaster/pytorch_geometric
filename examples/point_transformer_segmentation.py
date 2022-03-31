import os.path as osp

import torch
import torch.nn.functional as F
from point_transformer_classification import (MLP, TransformerBlock,
                                              TransitionDown)
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_cluster import knn_graph

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.utils import intersection_and_union as i_and_u


class TransitionUp(torch.nn.Module):
    '''
        Reduce features dimensionnality and interpolate back to higher
        resolution and cardinality
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = MLP([in_channels, out_channels])
        self.mlp = MLP([out_channels, out_channels])

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=3,
                                         batch_x=batch_sub, batch_y=batch)

        x = self.mlp(x) + x_interpolated

        return x


class TransitionSummit(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp_sub = Seq(Lin(in_channels, in_channels), ReLU())
        self.mlp = MLP([2 * in_channels, in_channels])

    def forward(self, x, batch=None):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        # compute the mean of features batch_wise
        x_mean = global_mean_pool(x, batch=batch)
        x_mean = self.mlp_sub(x_mean)  # (batchs, features)

        # reshape back to (N_points, features)
        counts = batch.unique(return_counts=True)[1]
        x_mean = torch.cat(
            [x_mean[i].repeat(counts[i], 1) for i in range(x_mean.shape[0])],
            dim=0)

        # transform features
        x = self.mlp(torch.cat((x, x_mean), 1))
        return x


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], bias=False)

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
        )

        blocks = [1, 2, 3, 5, 2]

        # backbone layers
        self.encoders = torch.nn.ModuleList()
        n = len(dim_model) - 1
        for i in range(0, n):

            # Add Transition Down block followed by a Point Transformer block
            self.encoders.append(
                Seq(
                    TransitionDown(in_channels=dim_model[i],
                                   out_channels=dim_model[i + 1], k=self.k),
                    *[
                        TransformerBlock(in_channels=dim_model[i + 1],
                                         out_channels=dim_model[i + 1])
                        for k in range(blocks[1:][i])
                    ]))

        # summit layers
        self.mlp_summit = TransitionSummit(dim_model[-1])

        self.transformer_summit = Seq(*[
            TransformerBlock(
                in_channels=dim_model[-1],
                out_channels=dim_model[-1],
            ) for i in range(1)
        ])

        self.decoders = torch.nn.ModuleList()
        for i in range(0, n):
            # Add Transition Up block followed by Point Transformer block
            self.decoders.append(
                Seq(
                    TransitionUp(in_channels=dim_model[n - i],
                                 out_channels=dim_model[n - i - 1]),
                    *[
                        TransformerBlock(in_channels=dim_model[n - i - 1],
                                         out_channels=dim_model[n - i - 1])
                        for k in range(1)
                    ]))

        # class score computation
        self.mlp_output = Seq(MLP([dim_model[0], dim_model[0]]),
                              Lin(dim_model[0], out_channels))

    def forward(self, x, pos, batch=None):
        # add dummy features in case there is none
        x = pos if x is None else torch.cat((pos, x), 1)

        out_x = []
        out_pos = []
        out_batch = []
        edges_index = []

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
        x = self.transformer_input(x, pos, edge_index)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)
        edges_index.append(edge_index)

        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.encoders)):

            x, pos, batch = self.encoders[i][0](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
            for layer in self.encoders[i][1:]:
                x = layer(x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)
            edges_index.append(edge_index)

        # summit
        x = self.mlp_summit(x, batch=batch)
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
        for layer in self.transformer_summit:
            x = layer(x, pos, edge_index)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.encoders)
        for i in range(n):
            x = self.decoders[i][0](x=out_x[-i - 2], x_sub=x,
                                    pos=out_pos[-i - 2],
                                    pos_sub=out_pos[-i - 1],
                                    batch_sub=out_batch[-i - 1],
                                    batch=out_batch[-i - 2])

            edge_index = edges_index[-i - 2]

            for layer in self.decoders[i][1:]:
                x = layer(x, out_pos[-i - 2], edge_index)

        # Class score
        out = self.mlp_output(x)

        return F.log_softmax(out, dim=-1)


def train():
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 10 == 0:
            print(f'[{i+1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} '
                  f'Train Acc: {correct_nodes / total_nodes:.4f}')
            total_loss = correct_nodes = total_nodes = 0


def test(loader):
    model.eval()

    y_mask = loader.dataset.y_mask
    ious = [[] for _ in range(len(loader.dataset.categories))]

    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.pos, data.batch).argmax(dim=1)

        i, u = i_and_u(pred, data.y, loader.dataset.num_classes, data.batch)
        iou = i.cpu().to(torch.float) / u.cpu().to(torch.float)
        iou[torch.isnan(iou)] = 1

        # Find and filter the relevant classes for each category.
        for iou, category in zip(iou.unbind(), data.category.unbind()):
            ious[category.item()].append(iou[y_mask[category]])

    # Compute mean IoU.
    ious = [torch.stack(iou).mean(0).mean(0) for iou in ious]
    return torch.tensor(ious).mean().item()


if __name__ == '__main__':
    category = 'Airplane'  # Pass in `None` to train on all categories.
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                    'ShapeNet')
    transform = T.Compose([
        T.RandomTranslate(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2),
    ])
    pre_transform = T.NormalizeScale()
    train_dataset = ShapeNet(path, category, split='trainval',
                             transform=transform, pre_transform=pre_transform)
    test_dataset = ShapeNet(path, category, split='test',
                            pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(3 + 3, train_dataset.num_classes,
                dim_model=[32, 64, 128, 256, 512], k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                                gamma=0.5)

    for epoch in range(1, 100):
        train()
        iou = test(test_loader)
        print(f'Epoch: {epoch:03d}, Test IoU: {iou:.4f}')
        scheduler.step()