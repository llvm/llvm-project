# 'shard' Dialect

The `shard` dialect contains a set of attributes, operations and interfaces that
are useful for representing sharding and communication on a device grid
cluster.

[TOC]

## Collective Communication Operations
There are a number of operations in the Shard dialect to facilitate
communication between devices in a grid.
It is assumed that the user is familiar with collective operations.
[Wikipedia](https://en.wikipedia.org/wiki/Collective_operation) has a good
explanation.
The main addition is that the collectives in this dialect have grid
semantics.

### Device groups
The operation attributes `grid` and `grid_axes` specifies a list of device grid
axes that partition the devices into disjoint groups.
The collective operation is performed between devices in the same group.
Devices that have the same coordinates outside of axes `grid_axes` are in the
same group.
A group is described by its multi-index along the axes outside of `grid_axes`.
For example if we have a device grid of size `2x3x4x5` and the partition grid
axes list is `[0, 1]` then devices are partitioned into the groups
`{ { (i, j, k, m) | 0<=i<2, 0<=j<3 } | 0<=k<4, 0<=m<5 }`.
The device groups would be `{ (k, m) | 0<=k<4, 0<=m<5 }`.
Devices (1, 0, 2, 3) and (1, 1, 2, 3) will be in the same group.
Device (1, 0, 2, 4) will be in another group.
Some collective operations like all-to-all and all-gather care about the
order of devices.
The order of device in a device group is induced by the order of axes in
`grid_axes`.
The axes are ordered from outer to inner.
If we have an axis list `[3, 1]` then device `(i, 1, k, 0)` will precede
both devices `(i, 0, k, 1)` and `(i, 2, k, 0)`.

### In-group Device
Some operations like `broadcast`, `scatter` and `send` specify devices in each
device-group.
These devices are represented with their multi-index over the grid axes that
are not constant within a device group.
These are the axes specified by `grid_axes` attribute.

For Example on a 3D grid an operation with `grid_axes = [0, 2]` would specify
an in-group device with `(i, j)`. Then for each group with index `g` on the
second axis, the in-group device would be `(i, g, j)`.
### Purity
Collectives that involve the whole device group to perform a single operation
are pure. The exceptions are `send` and `recv`.

There is an assumption that the execution is SPMD.
Not only that each process runs the same program, but that at the point of
execution of a collective operation, all processes are in a coherent state.
All compiler transformations must be consistent.
Collective operations in the IR that may correspond to the same runtime
collective operation must be transformed in a consistent manner.
For example if a collective operation is optimized out, than it must also
not appear in any path of execution on any process.

Having the operations as `Pure` implies that if an interpreter is to execute
the IR containing the `grid` collectives, all processes would execute the same
line when they reach a pure collective operation.
This requirement stems from the need to be compatible with general optimization
passes like dead code and common sub-expression elimination.

## Operations

[include "Dialects/ShardOps.md"]

## Attributes

[include "Dialects/ShardAttrs.md"]
