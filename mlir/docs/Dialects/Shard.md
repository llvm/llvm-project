# 'shard' Dialect

The 'shard' dialect defines a set of attributes, operations, and interfaces for
working with tensor sharding and device communication.

It’s inspired by [GSPMD](*General and Scalable Parallelization for ML Computation Graphs*).

Originally, the dialect was called `mesh`, but it was renamed to better reflect
what it actually does.

[TOC]

## Collective Communication Operations

The 'shard' dialect includes several collective operations that help coordinate
communication between devices arranged in a grid.

If you’re not already familiar with collective operations, [this Wikipedia
article](https://en.wikipedia.org/wiki/Collective_operation) is a good starting
point.

Unlike traditional collectives that are defined in terms of message-passing
between explicit buffers on each process, the collectives in this dialect work
at a higher level. They’re defined in terms of how data moves across the
dimensions of a tensor, and the participating processes are inferred from how
the tensor is sharded - not specified manually.

### Device Groups

Collective operations run within groups of devices, which are defined
using the `grid` and `grid_axes` attributes. These describe
how the full device grid is sliced into smaller groups.

Devices that have the same coordinates *outside* the listed `grid_axes` belong
to the same group.

Example: Say your device grid is shaped `2×3×4×5`, and you set
`grid_axes = [0, 1]`. This splits the grid into groups by fixing axes 2 and 3. You’d get groups like:

```
{ { (i, j, k, m) | 0 ≤ i < 2, 0 ≤ j < 3 } | 0 ≤ k < 4, 0 ≤ m < 5 }
```

So the groups are identified by the coordinates `(k, m)`, and devices like
`(1, 0, 2, 3)` and `(1, 1, 2, 3)` are in the same group. But `(1, 0, 2, 4)`
is in a different group.

For some collectives (like `all-to-all`), the order of devices in the group
matters. The device order is based on the order of axes in `grid_axes`, from
outermost to innermost.

Example: If `grid_axes = [3, 1]`, then device `(i, 1, k, 0)` comes before
`(i, 0, k, 1)` and `(i, 2, k, 0)`.

### In-group Devices

Some operations (like `broadcast`, `scatter`, and `send`) refer to a specific
device within each group. These in-group devices are identified using their
coordinates over the axes listed in `grid_axes`.

Example: In a 3D grid with `grid_axes = [0, 2]`, an in-group device is specified
as `(i, j)`. If a group is fixed at coordinate `g` on axis 1, then the full
device index would be `(i, g, j)`.

### Purity and Execution Model

Collective operations involve all devices in a group (e.g. `all-gather`,
`all-to-all`) and are considered pure. Operations like `send` and `recv` are not
collective and are not pure.

The execution model assumes SPMD (Single Program, Multiple Data):

* Every process runs the same program.
* At any collective operation, all processes are in sync.

This means compiler optimizations must treat collective ops carefully. For
example, if a collective is removed during optimization, it must be removed from
*every* path and *every* process that would have participated - otherwise, you’ll
get undefined behavior at runtime.

Marking these ops as pure also helps with standard compiler passes like dead
code elimination and common subexpression elimination. It ensures that when the
program is executed, all devices hit the same line of code at the same time
during collectives and so avoid dead-locks.

## Operations

[include "Dialects/ShardOps.md"]

## Attributes

[include "Dialects/ShardAttrs.md"]
