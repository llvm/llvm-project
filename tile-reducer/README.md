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

## Milestone 15

`--tr-lower-affine` applies upstream `populateAffineToStdConversionPatterns`:
`affine.for` becomes `scf.for`, `affine.apply` becomes `arith.muli` /
`arith.addi` on `index`.

## Milestone 16

`GPUTargetInfo` records warp size, SM / register / shared-memory
capacities, and the baseline launch (256 threads, 8 warps).
`--tr-annotate-gpu-target` writes those fields as `tr.target.*` module
attributes. They are target properties, not source semantics.

## Milestone 17

`--convert-tr-row-sum-to-gpu` maps a 128×128 logical tile onto 256
threads / 8 warps. Warp `w` walks rows `w, w+8, …, 120`. Lane `L` owns
columns `L, L+32, L+64, L+96`. Four register values are summed, then
`gpu.subgroup_reduce`. `tr.program_id` stays a logical program instance.

## Milestone 18

The same pass fuses `tr.load` → `tr.reduce_sum(axis=1)` into coalesced
`memref.load`s, per-lane `arith.addf`, and a subgroup reduce. No 128×128
temporary and no shared memory.

## Milestone 19

The source `arith.divui` trip count is replaced by `arith.ceildivui`.
Out-of-bounds columns in the last tile contribute zero via `scf.if`.
The kernel is K-dynamic and covers
`K ∈ {1,31,32,33,127,128,129,255,256,257}`. The lane-then-warp tree
reassociates the K-sum; TileReducer treats row-sum as reassociative.

## Milestone 20

`--tr-emit-gpu-kernels` builds `gpu.module @tr_kernels` with a SymbolTable:
lookup of an existing module, insertion of `gpu.func` symbols, uniqueness
when a name is taken, and nested `SymbolRefAttr` on
`gpu.launch_func @tr_kernels::@row_sum_kernel`. A missing kernel symbol is
a hard lookup failure.

## Milestone 21

Full `MxK → scalar` uses two kernels, `@full_sum_stage1` and
`@full_sum_stage2`. Stage 1: thread-local sum, warp reduce, `smem[warp]`,
barrier, block reduce, one partial per block. Stage 2 reduces the
partials. No unordered FP atomics.

## Milestone 22

Column reduction on row-major input: coalesced global loads into a
128×128 workgroup memref, `gpu.barrier`, then one thread per column.
128 is already a multiple of the warp size, so a padded 128×132 layout
is not used. Direct strided `in[row, col]` would break coalescing.

## Milestone 23

`ReductionSchedule` plus a roofline cost model:
`T ~= max(T_compute, T_memory) + T_sync + T_launch + T_tail`.
`--tr-estimate-reduction-cost` records occupancy, coalescing, register /
smem pressure, and grid saturation. Not cycle-exact.

## Milestone 24

`--tr-emit-gpu-kernels=k-splits=N` refines one logical row program into
N physical blocks (`gpu.block_id y`) plus `@row_sum_splitk_stage2`.
`tr.program_id` is still `gpu.block_id x`. Used when M is small and K is
huge (e.g. M=1, K=1e8).

## Milestone 25

`--tr-autotune-reduction` enumerates a bounded legal space, prunes
analytically, and caches the winner by
`kind|axis|dtype|tile|shape-bucket|arch|compiler`. It does not tune
every exact shape.

## Milestone 26

Async / double-buffering is measured after the baseline. Row-sum
intensity is too low; extra smem and registers do not pay off, so the
winner keeps `asyncDepth = 0`.

## Milestone 27

Boundary matrix `M ∈ {1,31,32,127,128,129}`,
`K ∈ {0,1,31,32,33,127,128,129,255,256,257}` for row, column, and full.
Property: `full_sum(A) ~= sum(row_sum(A)) ~= sum(column_sum(A))`.
`--tr-bench-report` records latency, GB/s, threads/block, registers,
smem, occupancy, and kernel count.

## Milestone 28

`--tr-split-host-device` records the host/device cut after kernels are
outlined. Host `func.func` keeps `gpu.launch_func` (`tr.role = "host"`).
Device `gpu.func` symbols live in `gpu.module @tr_kernels` with
`#nvvm.target<chip = "sm_80">` (`tr.role = "device"`). A missing kernel
symbol is a hard lookup failure. No computation is rewritten.

## Milestone 29

Device: `--tr-lower-device-to-nvvm` turns `gpu.func` into NVVM / LLVM
dialect (`nvvm.read.ptx.sreg.*`, `nvvm.shfl.sync`, `llvm.func`).
Host: `--tr-lower-host-to-llvm` uses upstream `gpu-to-llvm` so launch
ops become LLVM-typed runtime/launch IR. `--tr-emit-device-llvmir`
translates the device LLVM dialect to actual LLVM IR (`.ll`). Those
are not the same representation.
