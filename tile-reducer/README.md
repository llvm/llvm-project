# TileReducer

Out-of-tree MLIR compiler for a tile-level reduction language. Built against
an existing LLVM/MLIR tree; do not use the Tensor dialect.

## Milestone 1

Standalone project: `tr` dialect registration, `tr-opt`, CMake, lit smoke test.

```bash
cmake -G Ninja -S tile-reducer -B tile-reducer/build \
  -DMLIR_DIR=/Users/ionut/work/llvm/build/build-xcode/lib/cmake/mlir
cmake --build tile-reducer/build --target tr-opt check-tr
```

## Milestone 8

`--convert-tr-to-linalg` lowers `tr.reduce_sum` / `tr.add` / `tr.constant` to
Linalg over MemRefs. Row, column, and full reductions use `linalg.generic`
with `arith.addf` and iterator types `parallel,reduction`,
`reduction,parallel`, and `reduction,reduction`. No Tensor dialect.

## Milestone 9

The same pass realizes `tr.load` as `memref.subview` of the input
(offset = tile coordinate × tile size). A 128×128 tile is not allocated.
A 128-element alloca is used only for the accumulator / reduce destination.
`tr.store` is a subview of the output plus `memref.copy`.

## Milestone 10

`--tr-tile-linalg=tile-sizes=M,K` tiles `linalg.generic` reductions with
`scf::tileUsingSCF`. Representative sizes: 128×128, 64×128, 32×128. Outer
`scf.for` is the parallel (row) dimension; inner is the reduction (K)
dimension. No GPU thread mapping.
