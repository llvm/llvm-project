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
