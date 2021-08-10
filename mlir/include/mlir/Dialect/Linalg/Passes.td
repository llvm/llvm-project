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

def ConvertElementwiseToLinalg : FunctionPass<"convert-elementwise-to-linalg"> {
  let summary = "Convert ElementwiseMappable ops to linalg";
  let description = [{
    Convert ops with the `ElementwiseMappable` trait to linalg parallel loops.

    This pass only converts ops that operate on ranked tensors.
  }];
  let constructor = "mlir::createConvertElementwiseToLinalgPass()";
  let dependentDialects = ["linalg::LinalgDialect", "memref::MemRefDialect"];
}

def LinalgComprehensiveModuleBufferize :
    Pass<"linalg-comprehensive-module-bufferize", "ModuleOp"> {
  let summary = "Bufferize (tensor into memref) for a Module.";
  let description = [{
    This pass implements a cross-dialect bufferization approach and performs an
    analysis to determine which op operands and results may be bufferized in the
    same buffers. The analysis is performed on topologically sorted CallOp and
    FuncOp within a module. It provides analyses and bufferization across
    function boundaries. Within a function boundary, the analysis is performed
    on SSA use-def chains starting from function operands that are annotated
    with the 'inplaceable' attribute.
  }];
  let options = [
    Option<"testAnalysisOnly", "test-analysis-only", "bool",
            /*default=*/"false",
           "Only runs inplaceability analysis (for testing purposes only)">
  ];
  let constructor = "mlir::createLinalgComprehensiveModuleBufferizePass()";
}

def LinalgFoldUnitExtentDims : FunctionPass<"linalg-fold-unit-extent-dims"> {
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
  let options = [
    Option<"allowFoldingUnitDimReshapes", "allow-folding-unit-dim-reshapes",
           "bool", /*default=*/"false",
           "Allow fusing linalg.tensor_reshape ops that performs unit "
           "dimension collapsing">
  ];
  let dependentDialects = [
    "AffineDialect", "linalg::LinalgDialect", "memref::MemRefDialect"
  ];
}

def LinalgFoldReshapeOpsByLinearization :
  Pass<"linalg-fold-reshape-ops-by-linearization"> {
  let summary = "Fold TensorReshapeOps with generic/indexed generic ops by "
                "linearization";
  let constructor = "mlir::createFoldReshapeOpsByLinearizationPass()";
  let options = [
    Option<"allowFoldingUnitDimReshapes", "allow-folding-unit-dim-reshapes",
           "bool", /*default=*/"false",
           "Allow fusing linalg.tensor_reshape ops that performs unit "
           "dimension collapsing">
  ];
  let dependentDialects = ["AffineDialect", "memref::MemRefDialect"];
}

def LinalgLowerTiledLoopsToSCF
    : FunctionPass<"convert-linalg-tiled-loops-to-scf"> {
  let summary = "Lower linalg tiled loops to SCF loops and parallel loops";
  let constructor = "mlir::createConvertLinalgTiledLoopsToSCFPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "AffineDialect"
  ];
}

def LinalgInlineScalarOperands : FunctionPass<"linalg-inline-scalar-operands"> {
  let summary = "Inline scalar operands into linalg generic ops";
  let constructor = "mlir::createLinalgInlineScalarOperandsPass()";
  let dependentDialects = [
    "linalg::LinalgDialect"
  ];
}

def LinalgLowerToAffineLoops : FunctionPass<"convert-linalg-to-affine-loops"> {
  let summary = "Lower the operations from the linalg dialect into affine "
                "loops";
  let constructor = "mlir::createConvertLinalgToAffineLoopsPass()";
  let dependentDialects = [
    "AffineDialect", "linalg::LinalgDialect", "memref::MemRefDialect"];
}

def LinalgLowerToLoops : FunctionPass<"convert-linalg-to-loops"> {
  let summary = "Lower the operations from the linalg dialect into loops";
  let constructor = "mlir::createConvertLinalgToLoopsPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "AffineDialect"
  ];
}

def LinalgLowerToParallelLoops
    : FunctionPass<"convert-linalg-to-parallel-loops"> {
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

def LinalgBufferize : Pass<"linalg-bufferize", "FuncOp"> {
  let summary = "Bufferize the linalg dialect";
  let constructor = "mlir::createLinalgBufferizePass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "AffineDialect",
    "memref::MemRefDialect"
  ];
}

def LinalgPromotion : FunctionPass<"linalg-promote-subviews"> {
  let summary = "Promote subview ops to local buffers";
  let constructor = "mlir::createLinalgPromotionPass()";
  let options = [
    Option<"dynamicBuffers", "test-promote-dynamic", "bool",
           /*default=*/"false", "Test generation of dynamic promoted buffers">,
    Option<"useAlloca", "test-use-alloca", "bool",
           /*default=*/"false", "Test generation of alloca'ed buffers.">
  ];
  let dependentDialects = ["linalg::LinalgDialect"];
}

def LinalgTiling : FunctionPass<"linalg-tile"> {
  let summary = "Tile operations in the linalg dialect";
  let constructor = "mlir::createLinalgTilingPass()";
  let dependentDialects = [
    "AffineDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
  let options = [
    ListOption<"tileSizes", "linalg-tile-sizes", "int64_t", "Tile sizes",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
}

def LinalgTilingToParallelLoops
    : FunctionPass<"linalg-tile-to-parallel-loops"> {
  let summary = "Tile operations in the linalg dialect to parallel loops";
  let constructor = "mlir::createLinalgTilingToParallelLoopsPass()";
  let options = [
    ListOption<"tileSizes", "linalg-tile-sizes", "int64_t", "Tile sizes",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
  let dependentDialects = [
    "AffineDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def LinalgTilingToTiledLoops
    : FunctionPass<"linalg-tile-to-tiled-loop"> {
  let summary = "Tile operations in the linalg dialect to linalg.tiled_loop";
  let constructor = "mlir::createLinalgTilingToTiledLoopPass()";
  let options = [
    ListOption<"tileSizes", "linalg-tile-sizes", "int64_t", "Tile sizes",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
    ListOption<"distributionTypes", "linalg-distribution-types", "std::string",
               "DistributionTypes",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
  let dependentDialects = [
    "AffineDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def LinalgGeneralization : FunctionPass<"linalg-generalize-named-ops"> {
  let summary = "Convert named ops into generic ops";
  let constructor = "mlir::createLinalgGeneralizationPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
}

def LinalgDetensorize : FunctionPass<"linalg-detensorize"> {
  let summary = "Detensorize linalg ops";
  let constructor = "mlir::createLinalgDetensorizePass()";
  let dependentDialects = [];

  let description = [{
    Detensoring is the process through which a tensor value is convereted to one
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
  }];
}

#endif // MLIR_DIALECT_LINALG_PASSES
