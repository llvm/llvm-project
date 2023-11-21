//===- SparseTensorPasses.cpp - Pass for autogen sparse tensor code -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SPARSEREINTERPRETMAP
#define GEN_PASS_DEF_PRESPARSIFICATIONREWRITE
#define GEN_PASS_DEF_SPARSIFICATIONPASS
#define GEN_PASS_DEF_LOWERSPARSEOPSTOFOREACH
#define GEN_PASS_DEF_LOWERFOREACHTOSCF
#define GEN_PASS_DEF_SPARSETENSORCONVERSIONPASS
#define GEN_PASS_DEF_SPARSETENSORCODEGEN
#define GEN_PASS_DEF_SPARSEBUFFERREWRITE
#define GEN_PASS_DEF_SPARSEVECTORIZATION
#define GEN_PASS_DEF_SPARSEGPUCODEGEN
#define GEN_PASS_DEF_STAGESPARSEOPERATIONS
#define GEN_PASS_DEF_STORAGESPECIFIERTOLLVM
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

//===----------------------------------------------------------------------===//
// Passes implementation.
//===----------------------------------------------------------------------===//

struct SparseReinterpretMap
    : public impl::SparseReinterpretMapBase<SparseReinterpretMap> {
  SparseReinterpretMap() = default;
  SparseReinterpretMap(const SparseReinterpretMap &pass) = default;
  SparseReinterpretMap(const SparseReinterpretMapOptions &options) {
    scope = options.scope;
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateSparseReinterpretMap(patterns, scope);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct PreSparsificationRewritePass
    : public impl::PreSparsificationRewriteBase<PreSparsificationRewritePass> {
  PreSparsificationRewritePass() = default;
  PreSparsificationRewritePass(const PreSparsificationRewritePass &pass) =
      default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populatePreSparsificationRewriting(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct SparsificationPass
    : public impl::SparsificationPassBase<SparsificationPass> {
  SparsificationPass() = default;
  SparsificationPass(const SparsificationPass &pass) = default;
  SparsificationPass(const SparsificationOptions &options) {
    parallelization = options.parallelizationStrategy;
    enableRuntimeLibrary = options.enableRuntimeLibrary;
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    // Translate strategy flags to strategy options.
    SparsificationOptions options(parallelization, enableRuntimeLibrary);
    // Apply sparsification and cleanup rewriting.
    RewritePatternSet patterns(ctx);
    populateSparsificationPatterns(patterns, options);
    scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct StageSparseOperationsPass
    : public impl::StageSparseOperationsBase<StageSparseOperationsPass> {
  StageSparseOperationsPass() = default;
  StageSparseOperationsPass(const StageSparseOperationsPass &pass) = default;
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateStageSparseOperationsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct LowerSparseOpsToForeachPass
    : public impl::LowerSparseOpsToForeachBase<LowerSparseOpsToForeachPass> {
  LowerSparseOpsToForeachPass() = default;
  LowerSparseOpsToForeachPass(const LowerSparseOpsToForeachPass &pass) =
      default;
  LowerSparseOpsToForeachPass(bool enableRT, bool convert) {
    enableRuntimeLibrary = enableRT;
    enableConvert = convert;
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateLowerSparseOpsToForeachPatterns(patterns, enableRuntimeLibrary,
                                            enableConvert);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct LowerForeachToSCFPass
    : public impl::LowerForeachToSCFBase<LowerForeachToSCFPass> {
  LowerForeachToSCFPass() = default;
  LowerForeachToSCFPass(const LowerForeachToSCFPass &pass) = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateLowerForeachToSCFPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct SparseTensorConversionPass
    : public impl::SparseTensorConversionPassBase<SparseTensorConversionPass> {
  SparseTensorConversionPass() = default;
  SparseTensorConversionPass(const SparseTensorConversionPass &pass) = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    SparseTensorTypeToPtrConverter converter;
    ConversionTarget target(*ctx);
    // Everything in the sparse dialect must go!
    target.addIllegalDialect<SparseTensorDialect>();
    // All dynamic rules below accept new function, call, return, and various
    // tensor and bufferization operations as legal output of the rewriting
    // provided that all sparse tensor types have been fully rewritten.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return converter.isSignatureLegal(op.getCalleeType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return converter.isLegal(op.getOperandTypes());
    });
    target.addDynamicallyLegalOp<tensor::DimOp>([&](tensor::DimOp op) {
      return converter.isLegal(op.getOperandTypes());
    });
    target.addDynamicallyLegalOp<tensor::CastOp>([&](tensor::CastOp op) {
      return converter.isLegal(op.getSource().getType()) &&
             converter.isLegal(op.getDest().getType());
    });
    target.addDynamicallyLegalOp<tensor::ExpandShapeOp>(
        [&](tensor::ExpandShapeOp op) {
          return converter.isLegal(op.getSrc().getType()) &&
                 converter.isLegal(op.getResult().getType());
        });
    target.addDynamicallyLegalOp<tensor::CollapseShapeOp>(
        [&](tensor::CollapseShapeOp op) {
          return converter.isLegal(op.getSrc().getType()) &&
                 converter.isLegal(op.getResult().getType());
        });
    target.addDynamicallyLegalOp<bufferization::AllocTensorOp>(
        [&](bufferization::AllocTensorOp op) {
          return converter.isLegal(op.getType());
        });
    target.addDynamicallyLegalOp<bufferization::DeallocTensorOp>(
        [&](bufferization::DeallocTensorOp op) {
          return converter.isLegal(op.getTensor().getType());
        });
    // The following operations and dialects may be introduced by the
    // rewriting rules, and are therefore marked as legal.
    target.addLegalOp<complex::ConstantOp, complex::NotEqualOp, linalg::FillOp,
                      linalg::YieldOp, tensor::ExtractOp,
                      tensor::FromElementsOp>();
    target.addLegalDialect<
        arith::ArithDialect, bufferization::BufferizationDialect,
        LLVM::LLVMDialect, memref::MemRefDialect, scf::SCFDialect>();

    // Populate with rules and apply rewriting rules.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    populateSparseTensorConversionPatterns(converter, patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

struct SparseTensorCodegenPass
    : public impl::SparseTensorCodegenBase<SparseTensorCodegenPass> {
  SparseTensorCodegenPass() = default;
  SparseTensorCodegenPass(const SparseTensorCodegenPass &pass) = default;
  SparseTensorCodegenPass(bool createDeallocs, bool enableInit) {
    createSparseDeallocs = createDeallocs;
    enableBufferInitialization = enableInit;
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    SparseTensorTypeToBufferConverter converter;
    ConversionTarget target(*ctx);
    // Most ops in the sparse dialect must go!
    target.addIllegalDialect<SparseTensorDialect>();
    target.addLegalOp<SortOp>();
    target.addLegalOp<PushBackOp>();
    // Storage specifier outlives sparse tensor pipeline.
    target.addLegalOp<GetStorageSpecifierOp>();
    target.addLegalOp<SetStorageSpecifierOp>();
    target.addLegalOp<StorageSpecifierInitOp>();
    // Note that tensor::FromElementsOp might be yield after lowering unpack.
    target.addLegalOp<tensor::FromElementsOp>();
    // All dynamic rules below accept new function, call, return, and
    // various tensor and bufferization operations as legal output of the
    // rewriting provided that all sparse tensor types have been fully
    // rewritten.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return converter.isSignatureLegal(op.getCalleeType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return converter.isLegal(op.getOperandTypes());
    });
    target.addDynamicallyLegalOp<bufferization::AllocTensorOp>(
        [&](bufferization::AllocTensorOp op) {
          return converter.isLegal(op.getType());
        });
    target.addDynamicallyLegalOp<bufferization::DeallocTensorOp>(
        [&](bufferization::DeallocTensorOp op) {
          return converter.isLegal(op.getTensor().getType());
        });
    // The following operations and dialects may be introduced by the
    // codegen rules, and are therefore marked as legal.
    target.addLegalOp<linalg::FillOp>();
    target.addLegalDialect<
        arith::ArithDialect, bufferization::BufferizationDialect,
        complex::ComplexDialect, memref::MemRefDialect, scf::SCFDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    // Populate with rules and apply rewriting rules.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    populateSparseTensorCodegenPatterns(
        converter, patterns, createSparseDeallocs, enableBufferInitialization);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

struct SparseBufferRewritePass
    : public impl::SparseBufferRewriteBase<SparseBufferRewritePass> {
  SparseBufferRewritePass() = default;
  SparseBufferRewritePass(const SparseBufferRewritePass &pass) = default;
  SparseBufferRewritePass(bool enableInit) {
    enableBufferInitialization = enableInit;
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateSparseBufferRewriting(patterns, enableBufferInitialization);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct SparseVectorizationPass
    : public impl::SparseVectorizationBase<SparseVectorizationPass> {
  SparseVectorizationPass() = default;
  SparseVectorizationPass(const SparseVectorizationPass &pass) = default;
  SparseVectorizationPass(unsigned vl, bool vla, bool sidx32) {
    vectorLength = vl;
    enableVLAVectorization = vla;
    enableSIMDIndex32 = sidx32;
  }

  void runOnOperation() override {
    if (vectorLength == 0)
      return signalPassFailure();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateSparseVectorizationPatterns(
        patterns, vectorLength, enableVLAVectorization, enableSIMDIndex32);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct SparseGPUCodegenPass
    : public impl::SparseGPUCodegenBase<SparseGPUCodegenPass> {
  SparseGPUCodegenPass() = default;
  SparseGPUCodegenPass(const SparseGPUCodegenPass &pass) = default;
  SparseGPUCodegenPass(unsigned nT, bool enableRT) {
    numThreads = nT;
    enableRuntimeLibrary = enableRT;
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    if (numThreads == 0)
      populateSparseGPULibgenPatterns(patterns, enableRuntimeLibrary);
    else
      populateSparseGPUCodegenPatterns(patterns, numThreads);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct StorageSpecifierToLLVMPass
    : public impl::StorageSpecifierToLLVMBase<StorageSpecifierToLLVMPass> {
  StorageSpecifierToLLVMPass() = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    ConversionTarget target(*ctx);
    RewritePatternSet patterns(ctx);
    StorageSpecifierToLLVMTypeConverter converter;

    // All ops in the sparse dialect must go!
    target.addIllegalDialect<SparseTensorDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return converter.isSignatureLegal(op.getCalleeType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return converter.isLegal(op.getOperandTypes());
    });
    target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect>();

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    populateStorageSpecifierToLLVMPatterns(converter, patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass creation methods.
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::createSparseReinterpretMapPass() {
  return std::make_unique<SparseReinterpretMap>();
}

std::unique_ptr<Pass>
mlir::createSparseReinterpretMapPass(ReinterpretMapScope scope) {
  SparseReinterpretMapOptions options;
  options.scope = scope;
  return std::make_unique<SparseReinterpretMap>(options);
}

std::unique_ptr<Pass> mlir::createPreSparsificationRewritePass() {
  return std::make_unique<PreSparsificationRewritePass>();
}

std::unique_ptr<Pass> mlir::createSparsificationPass() {
  return std::make_unique<SparsificationPass>();
}

std::unique_ptr<Pass>
mlir::createSparsificationPass(const SparsificationOptions &options) {
  return std::make_unique<SparsificationPass>(options);
}

std::unique_ptr<Pass> mlir::createStageSparseOperationsPass() {
  return std::make_unique<StageSparseOperationsPass>();
}

std::unique_ptr<Pass> mlir::createLowerSparseOpsToForeachPass() {
  return std::make_unique<LowerSparseOpsToForeachPass>();
}

std::unique_ptr<Pass>
mlir::createLowerSparseOpsToForeachPass(bool enableRT, bool enableConvert) {
  return std::make_unique<LowerSparseOpsToForeachPass>(enableRT, enableConvert);
}

std::unique_ptr<Pass> mlir::createLowerForeachToSCFPass() {
  return std::make_unique<LowerForeachToSCFPass>();
}

std::unique_ptr<Pass> mlir::createSparseTensorConversionPass() {
  return std::make_unique<SparseTensorConversionPass>();
}

std::unique_ptr<Pass> mlir::createSparseTensorCodegenPass() {
  return std::make_unique<SparseTensorCodegenPass>();
}

std::unique_ptr<Pass>
mlir::createSparseTensorCodegenPass(bool createSparseDeallocs,
                                    bool enableBufferInitialization) {
  return std::make_unique<SparseTensorCodegenPass>(createSparseDeallocs,
                                                   enableBufferInitialization);
}

std::unique_ptr<Pass> mlir::createSparseBufferRewritePass() {
  return std::make_unique<SparseBufferRewritePass>();
}

std::unique_ptr<Pass>
mlir::createSparseBufferRewritePass(bool enableBufferInitialization) {
  return std::make_unique<SparseBufferRewritePass>(enableBufferInitialization);
}

std::unique_ptr<Pass> mlir::createSparseVectorizationPass() {
  return std::make_unique<SparseVectorizationPass>();
}

std::unique_ptr<Pass>
mlir::createSparseVectorizationPass(unsigned vectorLength,
                                    bool enableVLAVectorization,
                                    bool enableSIMDIndex32) {
  return std::make_unique<SparseVectorizationPass>(
      vectorLength, enableVLAVectorization, enableSIMDIndex32);
}

std::unique_ptr<Pass> mlir::createSparseGPUCodegenPass() {
  return std::make_unique<SparseGPUCodegenPass>();
}

std::unique_ptr<Pass> mlir::createSparseGPUCodegenPass(unsigned numThreads,
                                                       bool enableRT) {
  return std::make_unique<SparseGPUCodegenPass>(numThreads, enableRT);
}

std::unique_ptr<Pass> mlir::createStorageSpecifierToLLVMPass() {
  return std::make_unique<StorageSpecifierToLLVMPass>();
}
