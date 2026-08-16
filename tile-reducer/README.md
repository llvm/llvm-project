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

## Milestone 11

Transform dialect schedules in `transform/row_sum_schedule.mlir` and
`transform/column_sum_schedule.mlir`. Payload IR is the computation;
transform IR is the schedule. `transform.structured.match` +
`transform.structured.tile_using_for` tile the Linalg reduction.

## Milestone 12

Named schedule `@row_sum_schedule` is a public symbol. It includes private
`@tile_row_reduction` via `transform.include` (`SymbolRefAttr` lookup in a
`SymbolTable` with `transform.with_named_sequence`). Entry point selection
uses `--transform-interpreter=entry-point=row_sum_schedule`. A missing
symbol is a hard lookup failure.

## Milestone 13

`transform.tr.map_row_reduction` is a custom Transform extension. It takes a
handle, checks the payload is a row-reduction `linalg.generic`, and annotates
`tr.warps_per_block=8`, `tr.warp_size=32`, `tr.elements_per_lane=4`. Wrong
payload ops produce a silenceable diagnostic.

## Milestone 14

`--tr-index-to-affine` raises affine index arithmetic to `affine.apply`:
`programId * 128`, `programId * 128 + localRow`, and
`kt * 128 + lane + j * 32`. A product of two SSA values stays `arith.muli`.
`affine.for` is used for constant-bound local-row walks, not for Linalg.
