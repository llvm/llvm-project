//===-- Passes.td - Linalg pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_PASSES
#define MLIR_DIALECT_LINALG_PASSES

include "mlir/Pass/PassBase.td"

def ConvertElementwiseToLinalg : Pass<"convert-elementwise-to-linalg", ""> {
  let summary = "Convert ElementwiseMappable ops to linalg";
  let description = [{
    Convert ops with the `ElementwiseMappable` trait to linalg parallel loops.

    This pass only converts ops that operate on ranked tensors. It can be
    run on op which contains linalg ops (most commonly a
    FunctionOpInterface op).
  }];
  let constructor = "mlir::createConvertElementwiseToLinalgPass()";
  let dependentDialects = ["linalg::LinalgDialect", "memref::MemRefDialect"];
}

def LinalgFoldUnitExtentDims : Pass<"linalg-fold-unit-extent-dims", ""> {
  let summary = "Remove unit-extent dimension in Linalg ops on tensors";
  let constructor = "mlir::createLinalgFoldUnitExtentDimsPass()";
  let options = [
    Option<"foldOneTripLoopsOnly", "fold-one-trip-loops-only", "bool",
            /*default=*/"false",
           "Only folds the one-trip loops from Linalg ops on tensors "
           "(for testing purposes only)">
  ];
  let dependentDialects = [
    "linalg::LinalgDialect", "AffineDialect", "memref::MemRefDialect"
  ];
}

def LinalgElementwiseOpFusion : Pass<"linalg-fuse-elementwise-ops"> {
  let summary = "Fuse elementwise operations on tensors";
  let constructor = "mlir::createLinalgElementwiseOpFusionPass()";
  let dependentDialects = [
    "AffineDialect", "linalg::LinalgDialect", "memref::MemRefDialect"
  ];
}

def LinalgNamedOpConversion: Pass<"linalg-named-op-conversion"> {
  let summary = "Convert from one named linalg op to another.";
  let constructor = "mlir::createLinalgNamedOpConversionPass()";
  let dependentDialects = ["linalg::LinalgDialect", "tensor::TensorDialect"];
}

def LinalgInlineScalarOperands : Pass<"linalg-inline-scalar-operands", "func::FuncOp"> {
  let summary = "Inline scalar operands into linalg generic ops";
  let constructor = "mlir::createLinalgInlineScalarOperandsPass()";
  let dependentDialects = [
    "linalg::LinalgDialect"
  ];
}

def LinalgLowerToAffineLoops : Pass<"convert-linalg-to-affine-loops", "func::FuncOp"> {
  let summary = "Lower the operations from the linalg dialect into affine "
                "loops";
  let constructor = "mlir::createConvertLinalgToAffineLoopsPass()";
  let dependentDialects = [
    "AffineDialect", "linalg::LinalgDialect", "memref::MemRefDialect"];
}

def LinalgLowerToLoops : Pass<"convert-linalg-to-loops", "func::FuncOp"> {
  let summary = "Lower the operations from the linalg dialect into loops";
  let constructor = "mlir::createConvertLinalgToLoopsPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "AffineDialect"
  ];
}

def LinalgLowerToParallelLoops
    : Pass<"convert-linalg-to-parallel-loops", "func::FuncOp"> {
  let summary = "Lower the operations from the linalg dialect into parallel "
                "loops";
  let constructor = "mlir::createConvertLinalgToParallelLoopsPass()";
  let dependentDialects = [
    "AffineDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def LinalgBufferize : Pass<"linalg-bufferize", "func::FuncOp"> {
  let summary = "Bufferize the linalg dialect";
  let constructor = "mlir::createLinalgBufferizePass()";
  let dependentDialects = [
    "AffineDialect",
    "bufferization::BufferizationDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
  ];
}

def LinalgTilingPass : Pass<"linalg-tile", "func::FuncOp"> {
  let summary = "Tile operations in the linalg dialect";
  let constructor = "mlir::createLinalgTilingPass()";
  let dependentDialects = [
    "AffineDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes">,
    Option<"loopType", "loop-type", "std::string", /*default=*/"\"for\"",
           "Specify the type of loops to generate: for, parallel">
  ];
}

def LinalgGeneralization : Pass<"linalg-generalize-named-ops", "func::FuncOp"> {
  let summary = "Convert named ops into generic ops";
  let constructor = "mlir::createLinalgGeneralizationPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
}

def LinalgDetensorize : Pass<"linalg-detensorize", ""> {
  let summary = "Detensorize linalg ops";
  let constructor = "mlir::createLinalgDetensorizePass()";
  let dependentDialects = [];

  let description = [{
    Detensoring is the process through which a tensor value is converted to one
    or potentially more primitive value(s). During this process, operations with
    such detensored operands are also converted to an equivalent form that works
    on primitives.

    The detensoring process is driven by linalg-on-tensor ops. In particular, a
    linalg-on-tensor op is checked to see whether *all* its operands can be
    detensored. If so, those operands are converted to their primitive
    counterparts and the linalg op is replaced by an equivalent op that takes
    those new primitive values as operands. Therefore, detensoring an op can be
    divided into 2 main logical phases:

    1. Detect/match an op that can be detensored.
    2. Detensor the operands of the op and replace it with a primitive
       equivalent.

    In addition to detensoring individual ops, this pass detensors internal
    control flow inside a function. All blocks except for the entry block are
    detensored by converting their arguments whenever possible.

    This can be run on any FunctionOpInterface op and must not be
    run on others. This is because it performs specific legalization of the
    blocks that make up the body, which it assumes has is a FunctionOpInterface.
  }];
  let options = [
    Option<"aggressiveMode", "aggressive-mode", "bool", /*default=*/"false",
           "Detensorize all ops that qualify for detensoring along with branch"
           " operands and basic-block arguments.">

  ];
}

def LinalgStrategyTilePass
    : Pass<"linalg-strategy-tile-pass", "func::FuncOp"> {
  let summary = "Configurable pass to apply pattern-based linalg tiling.";
  let constructor = "mlir::createLinalgStrategyTilePass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
    Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"",
      "Which linalg op within the func is the anchor to latch on.">,
  ];
}

def LinalgStrategyRemoveMarkersPass
    : Pass<"linalg-strategy-remove-markers-pass", "func::FuncOp"> {
  let summary = "Cleanup pass that drops markers.";
  let constructor = "mlir::createLinalgStrategyRemoveMarkersPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
  ];
}

#endif // MLIR_DIALECT_LINALG_PASSES
