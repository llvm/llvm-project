//===- TileReducerPasses.td - TileReducer passes -----------*- tablegen -*-===//

#ifndef TILE_REDUCER_PASSES
#define TILE_REDUCER_PASSES

include "mlir/Pass/PassBase.td"

def FoldTRAddZero : Pass<"tr-fold-add-zero", "::mlir::func::FuncOp"> {
  let summary = "Fold tr.add(x, zero) to x.";
  let dependentDialects = ["::mlir::tr::TileReducerDialect"];
}

def RecognizeLoadReduce : Pass<"tr-recognize-load-reduce", "::mlir::func::FuncOp"> {
  let summary = "Mark tr.load -> tr.reduce_sum pairs for later fusion.";
  let description = [{
    When a reduce_sum's input is a single-use load of the same element type,
    set `tr.load_reduce` on the reduce. No IR shape change; this is a
    scheduling hint consumed by later fusion/GPU lowering.
  }];
  let dependentDialects = ["::mlir::tr::TileReducerDialect"];
}

def AnnotateReductionPlan : Pass<"tr-annotate-reduction-plan", "::mlir::func::FuncOp"> {
  let summary = "Consume analyses and annotate a reduction strategy candidate.";
  let description = [{
    Walks ReduceSumOps, reads Bounds/Layout/Reduction analyses, and sets
    `tr.plan` to one of `row`, `column`, or `full`. Does not rewrite payload
    computation.
  }];
  let dependentDialects = ["::mlir::tr::TileReducerDialect"];
}

def ConvertTRBuffersToMemRef : Pass<"convert-tr-buffers-to-memref", "::mlir::ModuleOp"> {
  let summary = "Dialect-convert !tr.buffer to memref. Leaves !tr.tile in place.";
  let description = [{
    Uses ConversionTarget, TypeConverter, and OpConversionPattern.
    `!tr.buffer<MxKxf32>` becomes `memref<?x?xf32>`. Function signatures
    are rewritten. `tr.dim` becomes `memref.dim`. Tile ops stay.
  }];
  let dependentDialects = [
    "::mlir::tr::TileReducerDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def ConvertTRForBoundsToArith : Pass<"tr-for-bounds-to-arith", "::mlir::func::FuncOp"> {
  let summary = "Materialize tr.for integer bounds as arith.constant index values.";
  let dependentDialects = [
    "::mlir::tr::TileReducerDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def TileTRLinalg : Pass<"tr-tile-linalg", "::mlir::func::FuncOp"> {
  let summary = "Tile Linalg reductions with scf.for. No GPU mapping.";
  let description = [{
    Uses the TilingInterface (`scf::tileUsingSCF`) on `linalg.generic`
    reductions. Representative tile sizes: 128x128, 64x128, 32x128.
    Parallel and reduction dimensions are identified from iterator_types.
    Does not map loops to GPU threads.
  }];
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect"
  ];
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Tile sizes per loop dimension (0 = do not tile)">
  ];
}

def ConvertTRIndexToAffine : Pass<"tr-index-to-affine", "::mlir::func::FuncOp"> {
  let summary = "Raise affine index arithmetic to affine.apply. Leave non-affine ops alone.";
  let description = [{
    `arith.muli` of an index by a constant and `arith.addi` of index values
    become `affine.apply`. Used for genuinely affine relations such as

      rowBase   = programId * 128
      globalRow = programId * 128 + localRow
      globalCol = kt * 128 + lane + j * 32

    Product of two SSA values is not affine and is left as `arith.muli`.
    `affine.for` is used only for constant-bound loops whose body is affine;
    `scf.for` around Linalg is not rewritten.
  }];
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def LowerTRAffine : Pass<"tr-lower-affine", "::mlir::func::FuncOp"> {
  let summary = "Lower affine.for/apply to scf.for and arith using upstream patterns.";
  let description = [{
    Applies `populateAffineToStdConversionPatterns`:
      affine.for   -> scf.for
      affine.apply -> arith.addi / arith.muli on index
  }];
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

def AnnotateGPUTarget : Pass<"tr-annotate-gpu-target", "::mlir::ModuleOp"> {
  let summary = "Attach GPUTargetInfo fields as module attributes.";
  let description = [{
    Records warp size, the baseline 256-thread / 8-warp launch, and A100-like
    SM / register / shared-memory capacities. These are target properties,
    not TileReducer source semantics.
  }];
}

def ConvertTRRowSumToGPU : Pass<"convert-tr-row-sum-to-gpu", "::mlir::func::FuncOp"> {
  let summary = "Lower the canonical row-sum to GPU dialect. No 128x128 temp.";
  let description = [{
    Fuses `tr.load` -> `tr.reduce_sum(axis=1)` into coalesced global loads,
    per-lane register accumulation (4 elements), and `gpu.subgroup_reduce`.

    Physical map for a 128x128 logical tile and a 256-thread block:
      8 warps; warp w walks rows w, w+8, ..., 120
      lane L owns columns L, L+32, L+64, L+96

    `tr.program_id` stays a logical program instance (not threadIdx).
    No shared memory. Tails use `arith.ceildivui` and OOB zeros.
  }];
  let dependentDialects = [
    "::mlir::tr::TileReducerDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::func::FuncDialect"
  ];
}

def EmitTRGPUKernels : Pass<"tr-emit-gpu-kernels", "::mlir::ModuleOp"> {
  let summary = "Outline reductions into gpu.module @tr_kernels and launch by symbol.";
  let description = [{
    Uses SymbolTable to look up or insert `gpu.module @tr_kernels`, insert
    unique `gpu.func` kernel symbols, and reference them from the host with
    a nested SymbolRefAttr on `gpu.launch_func` (`@tr_kernels::@name`).

    Row sum: `@row_sum_kernel` (no shared memory).
    Full sum: `@full_sum_stage1` + `@full_sum_stage2` (smem block reduce).
    Column sum: `@column_sum_kernel` (smem staging + barrier).

    `k-splits > 1` refines one logical row program into many physical
    blocks (`gpu.block_id y`) plus a second reduction stage.
  }];
  let dependentDialects = [
    "::mlir::tr::TileReducerDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::func::FuncDialect"
  ];
  let options = [
    Option<"kSplits", "k-splits", "int", /*default=*/"1",
           "Physical K-splits per logical row (1 = one block per program_id)">
  ];
}

def EstimateTRReductionCost : Pass<"tr-estimate-reduction-cost", "::mlir::ModuleOp"> {
  let summary = "Annotate the baseline ReductionSchedule cost model.";
  let description = [{
    T ~= max(T_compute, T_memory) + T_sync + T_launch + T_tail.
    Includes occupancy, coalescing, register/smem pressure, and grid
    saturation. Not cycle-exact.
  }];
  let options = [
    Option<"problemM", "rows", "int64_t", /*default=*/"0",
           "Problem rows (0 = use 1024 as a representative bucket)">,
    Option<"problemK", "cols", "int64_t", /*default=*/"0",
           "Problem cols (0 = use 1024 as a representative bucket)">
  ];
}

def AutotuneTRReduction : Pass<"tr-autotune-reduction", "::mlir::ModuleOp"> {
  let summary = "Enumerate legal schedules, prune, rank, and cache by shape bucket.";
  let description = [{
    Cache key: kind, axis, dtype, tile shape, shape bucket, arch, compiler.
    Does not tune every exact M×K. Autotune runs only after a correct
    baseline exists.
  }];
  let options = [
    Option<"problemM", "rows", "int64_t", /*default=*/"0",
           "Problem rows for the shape bucket">,
    Option<"problemK", "cols", "int64_t", /*default=*/"0",
           "Problem cols for the shape bucket">
  ];
}

def TRBenchReport : Pass<"tr-bench-report", "::mlir::ModuleOp"> {
  let summary = "Record latency, GB/s, occupancy, and kernel-count estimates.";
  let options = [
    Option<"problemM", "rows", "int64_t", /*default=*/"0", "Problem rows">,
    Option<"problemK", "cols", "int64_t", /*default=*/"0", "Problem cols">
  ];
}

def LowerTRDeviceToNVVM : Pass<"tr-lower-device-to-nvvm", "::mlir::ModuleOp"> {
  let summary = "Lower gpu.module payload to NVVM / LLVM dialect.";
  let description = [{
    Device path only. f32 `gpu.subgroup_reduce` becomes butterfly
    `gpu.shuffle`, then upstream `convert-gpu-to-nvvm` produces
    `nvvm.read.ptx.sreg.*`, `nvvm.shfl.sync`, and `llvm.func` kernels.
    Host `func.func` / `gpu.launch_func` are left alone.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::NVVM::NVVMDialect",
    "::mlir::LLVM::LLVMDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

def LowerTRHostToLLVM : Pass<"tr-lower-host-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Lower host launch/runtime ops toward the LLVM dialect.";
  let description = [{
    Host path only. Uses upstream `gpu-to-llvm` so `gpu.launch_func`
    keeps the MLIR GPU runtime ABI (`mgpu*` when a binary is present;
    legalized launch_func + LLVM types otherwise). Does not invent a
    proprietary runtime.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::LLVM::LLVMDialect",
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

def EmitTRDeviceLLVMIR : Pass<"tr-emit-device-llvmir", "::mlir::ModuleOp"> {
  let summary = "Translate device LLVM dialect to actual LLVM IR (.ll).";
  let description = [{
    MLIR LLVM dialect is not LLVM IR. This pass calls
    `translateModuleToLLVMIR` on the lowered `gpu.module` kernels and
    writes a `.ll` file. The MLIR module is unchanged.
  }];
  let options = [
    Option<"outputPath", "output", "std::string", /*default=*/[{""}],
           "Path for the device .ll (empty = stderr)">
  ];
}

def SplitTRHostDevice : Pass<"tr-split-host-device", "::mlir::ModuleOp"> {
  let summary = "Make the GPU host/device split explicit. No computation rewrite.";
  let description = [{
    A GPU program is not one linear function. After `tr-emit-gpu-kernels`:

      host:  func.func + gpu.launch_func   (runtime / launch)
      device: gpu.module @tr_kernels / gpu.func  (payload)

    This pass records that split: attaches `#nvvm.target` (A100 sm_80) to
    each device module, sets `tr.role = "host"` / `"device"`, and verifies
    every `gpu.launch_func` resolves through the SymbolTable. It does not
    lower ops; later passes own GPU->NVVM and host runtime lowering.
  }];
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::NVVM::NVVMDialect",
    "::mlir::func::FuncDialect"
  ];
}

def ConvertTRToLinalg : Pass<"convert-tr-to-linalg", "::mlir::func::FuncOp"> {
  let summary = "Lower tile compute to Linalg over MemRefs. Load becomes a subview.";
  let description = [{
    `!tr.tile` becomes a memref of the same static shape. Reductions become
    `linalg.generic` with `arith.addf`/`arith.addi` and the iterator types
    required by Milestone 8:

      row    (axis = 1 on rank 2): parallel, reduction
      column (axis = 0 on rank 2): reduction, parallel
      full   (rank 1, or fused 2D->scalar): reduction [, reduction]

    `tr.load` is a `memref.subview` of the input (tile coord * tile size).
    No `memref.alloc` of the loaded tile. A small alloca for an accumulator
    or reduce destination is allowed. `tr.store` is a subview plus copy.
    `tr.for` stays; only its iter_arg types change.
  }];
  let dependentDialects = [
    "::mlir::tr::TileReducerDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect"
  ];
}


#endif // TILE_REDUCER_PASSES
