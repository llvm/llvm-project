# 'mesh' Dialect

The `mesh` dialect contains a set of attributes, operations and interfaces that
are useful for representing sharding and communication on a device mesh
cluster.

[TOC]

## Collective Communication Operations
There are a number of operations in the Mesh dialect to facilitate
communication between devices in a mesh.
It is assumed that the user is familiar with collective operations.
[Wikipedia](https://en.wikipedia.org/wiki/Collective_operation) has a good
explanation.
The main addition is that the collectives in this dialect have mesh
semantics.
The operation attributes `mesh` and `mesh_axes` specifies a set of device mesh
axes that partition the devices into disjoint groups.
The collective operation is performed between devices in the same group.
Devices that have the same coordinates outside of axes `mesh_axes` are in the
same group.
For example if we have a device mesh of size `2x3x4x5` and the partition mesh
axes set is `{0, 1}` then devices are partitioned into the groups
`{ { (i, j, k, m) | 0<=i<2, 0<=j<3 } | 0<=k<4, 0<=m<5 }`.
Devices (1, 0, 2, 3) and (1, 1, 2, 3) will be in the same group.
Device (1, 0, 2, 4) will be in another group.

## Operations

[include "Dialects/MeshOps.md"]

## Attributes

[include "Dialects/MeshAttributes.md"]
