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
           "Only runs inplaceability analysis (for testing purposes only)">,
    Option<"printConflicts", "print-conflicts", "bool",
            /*default=*/"false",
           "Annotates IR with RaW conflicts. Requires test-analysis-only.">,
    Option<"allowReturnAllocs", "allow-return-allocs", "bool",
            /*default=*/"false",
           "Allows returning/yielding new allocations from a block.">,
    Option<"allowUnknownOps", "allow-unknown-ops", "bool",
           /*default=*/"false",
           "Allows unknown (not bufferizable) ops in the input IR.">,
    Option<"alwaysAliasingWithDest", "always-aliasing-with-dest", "bool",
            /*default=*/"true",
            "Tensor OpResult cannot bufferize inplace OpOperands other than "
            "out or dest OpOperands (if the op has a notion of such operands)">,
    Option<"useAlloca", "use-alloca", "bool",
           /*default=*/"false",
           "Use stack allocations for memrefs (for testing purposes only)">,
    Option<"fullyDynamicLayoutMaps", "fully-dynamic-layout-maps", "bool",
           /*default=*/"true",
           "Generate MemRef types with dynamic offset+strides by default.">,
    Option<"analysisFuzzerSeed", "analysis-fuzzer-seed", "unsigned",
           /*default=*/"0",
           "Analyze ops in random order with a given seed (fuzzer)">,
    Option<"initTensorElimination", "init-tensor-elimination", "bool",
            /*default=*/"false",
           "(Experimental) Try to eliminate init_tensor operations that are "
           "anchored at an insert_slice op">,
    Option<"createDeallocs", "create-deallocs", "bool", /*default=*/"true",
           "Specify if buffers should be deallocated. For compatibility with "
           "core bufferization passes.">,
  ];
  let constructor = "mlir::createLinalgComprehensiveModuleBufferizePass()";
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

def LinalgPromotion : Pass<"linalg-promote-subviews", "func::FuncOp"> {
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

def LinalgTiling : Pass<"linalg-tile", "func::FuncOp"> {
  let summary = "Tile operations in the linalg dialect";
  let constructor = "mlir::createLinalgTilingPass()";
  let dependentDialects = [
    "AffineDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes",
               "llvm::cl::ZeroOrMore">,
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

def LinalgStrategyTileAndFusePass
    : Pass<"linalg-strategy-tile-and-fuse-pass", "func::FuncOp"> {
  let summary = "Configurable pass to apply pattern-based tiling and fusion.";
  let constructor = "mlir::createLinalgStrategyTileAndFusePass()";
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
    Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"",
      "Which linalg op within the func is the anchor to latch on.">,
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

def LinalgStrategyPadPass
    : Pass<"linalg-strategy-pad-pass", "func::FuncOp"> {
  let summary = "Configurable pass to apply padding and hoisting.";
  let constructor = "mlir::createLinalgStrategyPadPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
    Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"",
      "Which linalg op within the func is the anchor to latch on.">,
  ];
}

def LinalgStrategyPromotePass
    : Pass<"linalg-strategy-promote-pass", "func::FuncOp"> {
  let summary = "Configurable pass to apply pattern-based linalg promotion.";
  let constructor = "mlir::createLinalgStrategyPromotePass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
    Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"",
      "Which linalg op within the func is the anchor to latch on.">,
  ];
}

def LinalgStrategyGeneralizePass
    : Pass<"linalg-strategy-generalize-pass", "func::FuncOp"> {
  let summary = "Configurable pass to apply pattern-based generalization.";
  let constructor = "mlir::createLinalgStrategyGeneralizePass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
    Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"",
      "Which linalg op within the func is the anchor to latch on.">,
  ];
}

// TODO: if/when we need finer control add an anchorOp option.
def LinalgStrategyDecomposePass
    : Pass<"linalg-strategy-decompose-pass", "func::FuncOp"> {
  let summary = "Configurable pass to apply pattern-based generalization.";
  let constructor = "mlir::createLinalgStrategyDecomposePass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
  ];
}

def LinalgStrategyInterchangePass
    : Pass<"linalg-strategy-interchange-pass", "func::FuncOp"> {
  let summary = "Configurable pass to apply pattern-based iterator interchange.";
  let constructor = "mlir::createLinalgStrategyInterchangePass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
  ];
}

def LinalgStrategyVectorizePass
    : Pass<"linalg-strategy-vectorize-pass", "func::FuncOp"> {
  let summary = "Configurable pass to apply pattern-based linalg vectorization.";
  let constructor = "mlir::createLinalgStrategyVectorizePass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
    Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"",
      "Which linalg op within the func is the anchor to latch on.">,
    Option<"vectorizePadding", "vectorize-padding", "bool", "false",
      "Enable vectorization of padding ops.">,
  ];
}

def LinalgStrategyEnablePass
    : Pass<"linalg-strategy-enable-pass", "func::FuncOp"> {
  let summary = "Configurable pass to enable the application of other "
    "pattern-based linalg passes.";
  let constructor = "mlir::createLinalgStrategyEnablePass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
  ];
}

def LinalgStrategyLowerVectorsPass
    : Pass<"linalg-strategy-lower-vectors-pass", "func::FuncOp"> {
  let summary = "Configurable pass to lower vector operations.";
  let constructor = "mlir::createLinalgStrategyLowerVectorsPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
  let options = [
    Option<"anchorFuncName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
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
