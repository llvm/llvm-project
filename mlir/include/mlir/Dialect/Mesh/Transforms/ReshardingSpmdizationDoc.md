# Resharding Spmdization Examples

Reshard `2x3` tensor from sharding `[[0, 1]]` to sharding `[[0, 1]]` on a `2x3` mesh.

unsharded `2x3` tensor
```
11 12 13
21 22 23
```

sharded on a `2x3` mesh

sharding = `[[0, 1]]`

mesh contents:

```
mesh axis 1
----------->
+----+----+----+ mesh axis 0 |
| 11 | 12 | 13 |             |
+----+----+----+             |
| 21 | 22 | 23 |             |
+----+----+----+             ↓
```

Transform into
sharding = `[[1, 0]]`
```
mesh axis 1
----------->
+----+----+----+ mesh axis 0 |
| 11 | 13 | 22 |             |
+----+----+----+             |
| 12 | 21 | 23 |             |
+----+----+----+             ↓
```
Algorithm:
Swap contents on devices that have the same linear index in the 2 shardings.

--------------------------------------------------------------

Reshard `2x3` tensor from sharding `[[0, 1]]` to sharding `[[1]]` on a `2x3` mesh.

unsharded `2x3` tensor
```
11 12 13
21 22 23
```

sharded on a `2x3` mesh

sharding = `[[0, 1]]`

mesh contents:
```
mesh axis 1
----------->
+----+----+----+ mesh axis 0 |
| 11 | 12 | 13 |             |
+----+----+----+             |
| 21 | 22 | 23 |             |
+----+----+----+             ↓
```

Transform into
sharding = `[[1]]`
```
mesh axis 1
----------->
+----+----+----+ mesh axis 0 |
| 11 | 12 | 13 |             |
| 21 | 22 | 23 |             |
+----+----+----+             |
| 11 | 12 | 13 |             |
| 21 | 22 | 23 |             |
+----+----+----+             ↓
```
Algorithm:
All-gather along mesh axis 0.

--------------------------------------------------------------

Reshard `4x6` tensor from sharding `[[], [0, 1]]` to sharding `[[], [0]]` on a `2x3` mesh.

unsharded `4x6` tensor
```
11 12 13 14 15 16
21 22 23 24 25 26
```

sharded on a `2x3` mesh

sharding = `[[], [0, 1]]`

mesh contents:
```
mesh axis 1
----------->
+----+----+----+ mesh axis 0 |
| 11 | 12 | 13 |             |
| 21 | 22 | 23 |             |
+----+----+----+             |
| 14 | 15 | 16 |             |
| 24 | 25 | 26 |             |
+----+----+----+             ↓
```
Transform into
sharding = `[[], [0]]`
```
mesh axis 1
----------->
+----------+----------+ mesh axis 0 |
| 11 12 13 | 11 12 13 |             |
| 21 22 23 | 21 22 23 |             |
+----------+----------+             |
| 14 15 16 | 14 15 16 |             |
| 24 25 26 | 24 25 26 |             |
+----------+----------+             ↓
```
Algorithm:
All-gather along mesh axis 1.

--------------------------------------------------------------

Reshard `4x8` tensor from sharding `[[0], [1, 2]]` to sharding `[[0], [2]]` on a `2x2x2` mesh.

unsharded `4x8` tensor
```
11 12 13 14 15 16 17 18
21 22 23 24 25 26 27 28
31 32 33 34 35 36 37 38
41 42 43 44 45 46 47 48
```
sharded on a `2x2x2` mesh

sharding = `[[0], [1, 2]]`

mesh contents:
```
mesh axis 2
----------->
+-------+-------+ mesh axis 1 | mesh axis 0 |
| 11 12 | 13 14 |             |             |
| 21 22 | 23 24 |             |             |
+-------+-------+             |             |
| 15 16 | 17 18 |             |             |
| 25 26 | 27 28 |             |             |
+-------+-------+             ↓             |
+-------+-------+                           |
| 31 32 | 33 34 |                           |
| 41 42 | 43 44 |                           |
+-------+-------+                           |
| 35 36 | 37 38 |                           |
| 45 46 | 47 48 |                           |
+-------+-------+                           ↓
```
Transform into
sharding = `[[0], [2]]`
```
mesh axis 2
----------->
+-------------+-------------+ mesh axis 1 | mesh axis 0 |
| 11 12 13 14 | 15 16 17 18 |             |             |
| 21 22 23 24 | 25 26 27 28 |             |             |
+-------------+-------------+             |             |
| 11 12 13 14 | 15 16 17 18 |             |             |
| 21 22 23 24 | 25 26 27 28 |             |             |
+-------------+-------------+             ↓             |
+-------------+-------------+                           |
| 31 32 33 34 | 35 36 37 38 |                           |
| 41 42 43 44 | 45 46 47 48 |                           |
+-------------+-------------+                           |
| 31 32 33 34 | 35 36 37 38 |                           |
| 41 42 43 44 | 45 46 47 48 |                           |
+-------------+-------------+                           ↓
```
Algorithm:

Can't be done with just an all-gather along mesh axis 1.
Can be handled by multiple resharding transformations
`[[0], [1, 2]] -> [[0], [2, 1]] -> [[0], [2]]`

--------------------------------------------------------------

Reshard `6x6` tensor from sharding `[[0], [1]]` to sharding `[[1], [0]]` on a `2x3` mesh.

unsharded `6x6` tensor
```
11 12 13 14 15 16
21 22 23 24 25 26
31 32 33 34 35 36
41 42 43 44 45 46
51 52 53 54 55 56
61 62 63 64 65 66
```
sharded on a `2x3` mesh

sharding = `[[0], [1]]`
```
mesh axis 1
----------->
+-------+-------+-------+ mesh axis 0 |
| 11 12 | 13 14 | 15 16 |             |
| 21 22 | 23 24 | 25 26 |             |
| 31 32 | 33 34 | 35 36 |             |
+-------+-------+-------+             |
| 41 42 | 43 44 | 45 46 |             |
| 51 52 | 53 54 | 55 56 |             |
| 61 62 | 63 64 | 65 66 |             |
+-------+-------+-------+             ↓
```
transform to
sharding = `[[1], [0]]`
```
mesh axis 1
----------->
+----------+----------+----------+ mesh axis 0 |
| 11 12 13 | 31 32 33 | 51 52 53 |             |
| 21 22 23 | 41 42 43 | 61 62 63 |             |
+----------+----------+----------+             |
| 14 15 16 | 34 35 36 | 54 55 56 |             |
| 24 25 26 | 44 45 46 | 64 65 66 |             |
+----------+----------+----------+             ↓

mesh axis 0
----------->
+----------+----------+ mesh axis 1 |
| 11 12 13 | 14 15 16 |             |
| 21 22 23 | 24 25 26 |             |
+----------+----------+             |
| 31 32 33 | 34 35 36 |             |
| 41 42 43 | 44 45 46 |             |
+----------+----------+             |
| 51 52 53 | 54 55 56 |             |
| 61 62 63 | 64 65 66 |             |
+----------+----------+             ↓
```
Algorithm: TODO

--------------------------------------------------------------

Reshard `6x6` tensor from sharding `[[0], [1]]` to sharding `[[1], [0]]` on a `2x6` mesh.

unsharded 6x6 tensor
```
11 12 13 14 15 16
21 22 23 24 25 26
31 32 33 34 35 36
41 42 43 44 45 46
51 52 53 54 55 56
61 62 63 64 65 66
```
shard on `2x6` mesh

sharding = `[[0], [1]]`
```
mesh axis 1
----------->
+----+----+----+----+----+----+ mesh axis 0 |
| 11 | 12 | 13 ‖ 14 | 15 | 16 |             |
| 21 | 22 | 23 ‖ 24 | 23 | 26 |             |
| 31 | 32 | 33 ‖ 34 | 35 | 36 |             |
+----+----+----+----+----+----+             |
| 41 | 42 | 43 ‖ 44 | 45 | 46 |             |
| 51 | 52 | 53 ‖ 54 | 55 | 56 |             |
| 61 | 62 | 63 ‖ 64 | 65 | 66 |             |
+----+----+----+----+----+----+             ↓
```
transform to
sharding = `[[1], [0]]`
```
mesh axis 0
----------->
+----------+----------+ mesh axis 1 |
| 11 12 13 | 14 15 16 |             |
+----------+----------+             |
| 21 22 23 | 24 25 26 |             |
+----------+----------+             |
| 31 32 33 | 34 35 36 |             |
+==========+==========+             |
| 41 42 43 | 44 45 46 |             |
+----------+----------+             |
| 51 52 53 | 54 55 56 |             |
+----------+----------+             |
| 61 62 63 | 64 65 66 |             |
+----------+----------+             ↓
```
Algorithm: TODO

--------------------------------------------------------------

Reshard KxL tensor from `[[0], [1]]` to `[[1], [0]]` on `MxN` mesh.

`M x N` mesh.
`K x L` tensor `t`.
`d(m, n)` the tensor on device `(m, n)`.

sharding = `[[0], [1]]`
Tensor shard s on each device has size `(K ceildiv M, L ceildiv N)`.
```
d(m, n)[k, l] -> t[m * (K ceildiv M) + k, n * (L ceildiv N) + l]
```
substitute
```
i <- m * (K ceildiv M) + k
j <- n * (L ceildiv N) + l
```
```
m -> i floordiv (K ceildiv M)
n -> j floordiv (L ceildiv N)
k -> i % (K ceildiv M)
l -> j % (L ceildiv N)
```
For the inverse map we get
```
t[i, j] -> d(
    i floordiv (K ceildiv M), j floordiv (L ceildiv N)
)[
    i % (K ceildiv M), j % (L ceildiv N)
]
```
Check:
```
i = 13, j = 17, M = 3, N = 4, K = 16, L = 23
t[13, 17] = d(
    13 floordiv (16 ceildiv 3),
    17 floordiv (23 ceilvid 4)
)[
    13 % (16 ceildiv 3),
    17 % (23 ceilvid 4)
]
= d(
    13 floordiv 6,
    17 floordiv 6
)[
    13 % 6,
    17 % 6
]
= d(2, 2)[1, 5]
= t[
    2 * (16 ceildiv 3) + 1,
    2 * (23 ceildiv 4) + 5
]
= t[
    2 * 6 + 1,
    2 * 6 + 5
]
= t[13, 17]
```

sharding = `[[1], [0]]`
Tensor shard s on each device has size `(K ceildiv N, L ceildiv M)`.
```
d(m, n)[k, l] -> t[n * (K ceildiv N) + k, m * (L ceildiv M) + l]
```
substitute
```
i <- n * (K ceildiv N) + k
j <- m * (L ceildiv M) + l
```
```
m -> j floordiv (L ceildiv M)
n -> i floordiv (K ceildiv N)
k -> i % (K ceildiv N)
l -> j % (L ceildiv M)
```
For the inverse map we get
```
t[i, j] -> d(
    j floordiv (L ceildiv M), i floordiv (K ceildiv N)
)[
    i % (K ceildiv N), j % (L ceildiv M)
]
```
Check:
```
i = 9, j = 19, M = 5, N = 2, K = 27, L = 14
t[9, 19] = d(
    19 floordiv (14 ceildiv 5),
    9 floordiv (27 ceildiv 2)
)[
    9 % (27 ceildiv 2),
    19 % (14 ceildiv 5)
]
= d(
    19 floordiv 3,
    9 floordiv 14
)[
    9 % 14
    19 % 3
]
= d(6, 0)[9, 1]
= t[
    0 * (27 ceildiv 2) + 9,
    6 * (14 ceildiv 5) + 1
]
= t[
    0 * 14 + 9,
    6 * 3 + 1
]
= t[9, 19]
```
sharding = `[[0], [1]]`
```
d(m, n)[k, l] -> t[m * (K ceildiv M) + k, n * (L ceildiv N) + l]
t[i, j] -> d(i floordiv (K ceildiv M), j floordiv (L ceildiv N))[i % (K ceildiv M), j % (L ceildiv N)]
```
sharding = `[[1], [0]]`
```
d(m, n)[k, l] -> t[n * (K ceildiv N) + k, m * (L ceildiv M) + l]
t[i, j] -> d(j floordiv (L ceildiv M), i floordiv (K ceildiv N))[i % (K ceildiv N), j % (L ceildiv M)]
```
sharding `[[0], [1]] -> [[1], [0]]`
`d1(m, n)` the tensor on device `(m, n)` for sharding sharding `[[0], [1]]`.
`d2(m, n)` the tensor on device `(m, n)` for sharding sharding `[[1], [0]]`.
```
d1(m, n)[k, l] ->
t[m * (K ceildiv M) + k, n * (L ceildiv N) + l] ->
d2(
    (m * (L ceildiv M) + l) floordiv (L ceildiv M),
    (n * (K ceildiv N) + k) floordiv (K ceildiv N)
)[
    (n * (K ceildiv N) + k) % (K ceildiv N),
    (m * (L ceildiv M) + l) % (L ceildiv M)
]
= d2(p, q)[u, v]
```
We want to copy the the data between devices in slices/tiles.
What are the source/target tile coordinates?
For a fixed `(m, n, p, q)` what is the range of `(k, l, u, v)`?
TODO

--------------------------------------------------------------

Reshard `KxL` tensor from sharding `[[0], [1]]` to sharding `[[1], [0]]` on a `2x3` mesh.

Device placement on a `2x3` mesh
```
11 12 13  <- devices
21 22 23
```
sharding `[[0], [1]]`
```
tensor axis 1
----------->
+----+----+----+ tensor axis 0 |
| 11 | 12 | 13 |               |
+----+----+----+               |
| 21 | 22 | 23 |               |
+----+----+----+               ↓
```
transform to
sharding `[[1], [0]]`
```
tensor axis 1
----------->
+----+----+ tensor axis 0 |
| 11 | 21 |               |
+----+----+               |
| 12 | 22 |               |
+----+----+               |
| 13 | 23 |               |
+----+----+               ↓
```
```
+-----------------+--------+--------+-----------------+
|                 |                 |                 |
+                 +                 +                 +
|       11        |        12       |        13       |
+                 +                 +                 +
|                 |                 |                 |
+-----------------+--------+--------+-----------------+
|                 |                 |                 |
+                 +                 +                 +
|       21        |        22       |        23       |
+                 +                 +                 +
|                 |                 |                 |
+-----------------+--------+--------+-----------------+

+-----------------+--------+--------+-----------------+
|                          |                          |
+          11              +               21         +
|                          |                          |
+-----------------+--------+--------+-----------------+
|                          |                          |
+          12              +               22         +
|                          |                          |
+-----------------+--------+--------+-----------------+
|                          |                          |
+          13              +               23         +
|                          |                          |
+-----------------+--------+--------+-----------------+

+-----------------+--------+--------+-----------------+
|                 |        |        |                 |
+     11  11      + 12  11 + 12  21 +     13  21      +
|                 |        |        |                 |
+-----------------+--------+--------+-----------------+
|     11  12      | 12  12 | 12  22 |     13  22      |
+-----------------+--------+--------+-----------------+
|     21  12      | 22  12 | 22  22 |     23  22      |
+-----------------+--------+--------+-----------------+
|                 |        |        |                 |
+     21  13      + 22  13 + 22  23 +     23  23      +
|                 |        |        |                 |
+-----------------+--------+--------+-----------------+
```
If `S` and `T` are the source and target shard sizes along some tensor axis.
Then we have a period of `(S*T)/gcd(S, T)`. Then the cut pattern repeats.
TODO

--------------------------------------------------------------

Reshard `6x6` tensor from sharding `[[0], []]` to sharding `[[], [0]]` on a `3` mesh.

unsharded `6x6` tensor
```
11 12 13 14 15 16
21 22 23 24 25 26
31 32 33 34 35 36
41 42 43 44 45 46
51 52 53 54 55 56
61 62 63 64 65 66
```
sharded on a `3` mesh

sharding = `[[0], []]`
```
+-------------------+ mesh axis 0 |
| 11 12 13 14 15 16 |             |
| 21 22 23 24 25 26 |             |
+-------------------+             |
| 31 32 33 34 35 36 |             |
| 41 42 43 44 45 46 |             |
+-------------------+             |
| 51 52 53 54 55 56 |             |
| 61 62 63 64 65 66 |             |
+-------------------+             ↓
```
transform to
sharding = `[[], [0]]`
```
mesh axis 0
----------->
+-------+-------+-------+
| 11 12 | 13 14 | 15 16 |
| 21 22 | 23 24 | 25 26 |
| 31 32 | 33 34 | 35 36 |
| 41 42 | 43 44 | 45 46 |
| 51 52 | 53 54 | 55 56 |
| 61 62 | 63 64 | 65 66 |
+-------+-------+-------+
```
Algorithm:
```mlir
%1 = all_to_all %0 on @mesh mesh_axes = [0] split_axis = 1 concat_axis = 0 : tensor<2x6xi8> -> tensor<6x2xi8>
```
--------------------------------------------------------------

Reshard `4x4` tensor from sharding `[[0], [1, 2]]` to sharding `[[0, 1], [2]]` on a `2x2x2` mesh.

unsharded `4x4` tensor
```
11 12 13 14
21 22 23 24
31 32 33 34
41 42 43 44
```
sharded on a `2x2x2` mesh

sharding = `[[0], [1, 2]]`
```
mesh axis 2
----------->
+----+----+ mesh axis 1 | mesh axis 0 |
| 11 | 12 |             |             |
| 21 | 22 |             |             |
+----+----+             |             |
| 13 | 14 |             |             |
| 23 | 24 |             |             |
+----+----+             ↓             |
+----+----+                           |
| 31 | 32 |                           |
| 41 | 42 |                           |
+----+----+                           |
| 33 | 34 |                           |
| 43 | 44 |                           |
+----+----+                           ↓
```
transform to
sharding = `[[0, 1], [2]]`
```
mesh axis 2
----------->
+-------+-------+ mesh axis 1 | mesh axis 0 |
| 11 12 | 13 41 |             |             |
+-------+-------+             |             |
| 21 22 | 23 24 |             |             |
+-------+-------+             ↓             |
+-------+-------+                           |
| 31 32 | 33 34 |                           |
+-------+-------+                           |
| 41 42 | 43 44 |                           |
+-------+-------+                           ↓
```
Algorithm:
```mlir
%1 = all_to_all %0 on @mesh mesh_axes = [2] split_axis = 1 concat_axis = 0 : tensor<2x1xi8> -> tensor<1x2xi8>
```
is not enough.

Can be decomposed into
```
[[0], [1, 2]] -> [[0], [2, 1]] -> [[0, 1], [2]]
```

## Decomposition into basis of reshardings

We can decompose each resharding into a sequence of basis reshardings.
It is not communication efficient in terms of minimizing the data communicated
between devices.
An efficient approach would be more complicated to implement.
Each device has to receive at most as much data as the size of its target
sharding tensor.

--------------------------------------------------------------

Basis:

*   From replicate to split.
    ```
    [[]] -> [[1]]
    ```
    Extract slices without communication.

* From split to replicate.
    ```
    [[0]] -> [[]]
    [[0, 1]] -> [[1]]
    ```
    All-gather along mesh axis 0.

*   Swap mesh axes order when assigned to the same tensor axis.
    ```
    [[0, 1]] -> [[1, 0]]
    ```
    Swap contents on devices with the same linear index.

*   Move mesh axis to different tensor dimension.
    ```
    [[0], []] -> [[], [0]]
    ```
    All-to-all.

--------------------------------------------------------------

Example decomposition of
```
[[0], [1]] -> [[1], [0]]
```
into
```
[[0], [1]] -> all-gather along mesh axis 1    ->
[[0], []]  -> all-to-all along mesh axis 0    ->
[[], [0]]  -> extract slice along mesh axis 1 ->
[[1], [0]]
```

--------------------------------------------------------------

Example decomposition of
```
[[3, 2], [], [0, 1]] -> [[0], [1, 2], []]
```
into
```
[[3, 2], [], [0, 1]] -> all-to-all along mesh axis 1 ->
[[3, 2], [1], [0]]   -> all-to-all along mesh axis 2 ->
[[3], [1, 2], [0]]   -> all-gather along mesh axis 3 ->
[[], [1, 2], [0]]    -> all-to-all along mesh axis 0 ->
[[0], [1, 2], []]
```
