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
