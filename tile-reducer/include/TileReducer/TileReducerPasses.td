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

def ConvertTRForBoundsToArith : Pass<"tr-for-bounds-to-arith", "::mlir::func::FuncOp"> {
  let summary = "Materialize tr.for integer bounds as arith.constant index values.";
  let dependentDialects = [
    "::mlir::tr::TileReducerDialect",
    "::mlir::arith::ArithDialect"
  ];
}


#endif // TILE_REDUCER_PASSES
