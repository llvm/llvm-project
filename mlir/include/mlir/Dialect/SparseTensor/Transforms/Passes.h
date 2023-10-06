//===- Passes.h - Sparse tensor pass entry points ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all sparse tensor passes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

//===----------------------------------------------------------------------===//
// Include the generated pass header (which needs some early definitions).
//===----------------------------------------------------------------------===//

namespace mlir {

namespace bufferization {
struct OneShotBufferizationOptions;
} // namespace bufferization

/// Defines a parallelization strategy. Any independent loop is a candidate
/// for parallelization. The loop is made parallel if (1) allowed by the
/// strategy (e.g., AnyStorageOuterLoop considers either a dense or sparse
/// outermost loop only), and (2) the generated code is an actual for-loop
/// (and not a co-iterating while-loop).
enum class SparseParallelizationStrategy {
  kNone,
  kDenseOuterLoop,
  kAnyStorageOuterLoop,
  kDenseAnyLoop,
  kAnyStorageAnyLoop
};

/// Defines data movement strategy between host and device for GPU.
// TODO : Zero copy is disabled due to correctness bugs (tracker #64316)
enum class GPUDataTransferStrategy { kRegularDMA, kZeroCopy, kPinnedDMA };

#define GEN_PASS_DECL
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// The PreSparsificationRewriting pass.
//===----------------------------------------------------------------------===//

void populatePreSparsificationRewriting(RewritePatternSet &patterns);

std::unique_ptr<Pass> createPreSparsificationRewritePass();

//===----------------------------------------------------------------------===//
// The Sparsification pass.
//===----------------------------------------------------------------------===//

/// Options for the Sparsification pass.
struct SparsificationOptions {
  SparsificationOptions(SparseParallelizationStrategy p,
                        GPUDataTransferStrategy t, bool idxReduc,
                        bool gpuLibgen, bool enableRT)
      : parallelizationStrategy(p), gpuDataTransferStrategy(t),
        enableIndexReduction(idxReduc), enableGPULibgen(gpuLibgen),
        enableRuntimeLibrary(enableRT) {}
  SparsificationOptions()
      : SparsificationOptions(SparseParallelizationStrategy::kNone,
                              GPUDataTransferStrategy::kRegularDMA, false,
                              false, true) {}
  SparseParallelizationStrategy parallelizationStrategy;
  GPUDataTransferStrategy gpuDataTransferStrategy;
  bool enableIndexReduction;
  bool enableGPULibgen;
  bool enableRuntimeLibrary;
};

/// Sets up sparsification rewriting rules with the given options.
void populateSparsificationPatterns(
    RewritePatternSet &patterns,
    const SparsificationOptions &options = SparsificationOptions());

std::unique_ptr<Pass> createSparsificationPass();
std::unique_ptr<Pass>
createSparsificationPass(const SparsificationOptions &options);

//===----------------------------------------------------------------------===//
// The StageSparseOperations pass.
//===----------------------------------------------------------------------===//

/// Sets up StageSparseOperation rewriting rules.
void populateStageSparseOperationsPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createStageSparseOperationsPass();

//===----------------------------------------------------------------------===//
// The PostSparsificationRewriting pass.
//===----------------------------------------------------------------------===//

void populatePostSparsificationRewriting(RewritePatternSet &patterns,
                                         bool enableRT, bool enableForeach,
                                         bool enableConvert);

std::unique_ptr<Pass> createPostSparsificationRewritePass();
std::unique_ptr<Pass>
createPostSparsificationRewritePass(bool enableRT, bool enableForeach = true,
                                    bool enableConvert = true);

//===----------------------------------------------------------------------===//
// The SparseTensorConversion pass.
//===----------------------------------------------------------------------===//

/// Sparse tensor type converter into an opaque pointer.
class SparseTensorTypeToPtrConverter : public TypeConverter {
public:
  SparseTensorTypeToPtrConverter();
};

/// Defines a strategy for implementing sparse-to-sparse conversion.
/// `kAuto` leaves it up to the compiler to automatically determine
/// the method used.  `kViaCOO` converts the source tensor to COO and
/// then converts the COO to the target format.  `kDirect` converts
/// directly via the algorithm in <https://arxiv.org/abs/2001.02609>;
/// however, beware that there are many formats not supported by this
/// conversion method.
enum class SparseToSparseConversionStrategy { kAuto, kViaCOO, kDirect };

/// Converts command-line sparse2sparse flag to the strategy enum.
SparseToSparseConversionStrategy sparseToSparseConversionStrategy(int32_t flag);

/// SparseTensorConversion options.
struct SparseTensorConversionOptions {
  SparseTensorConversionOptions(SparseToSparseConversionStrategy s2s)
      : sparseToSparseStrategy(s2s) {}
  SparseTensorConversionOptions()
      : SparseTensorConversionOptions(SparseToSparseConversionStrategy::kAuto) {
  }
  SparseToSparseConversionStrategy sparseToSparseStrategy;
};

/// Sets up sparse tensor conversion rules.
void populateSparseTensorConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    const SparseTensorConversionOptions &options =
        SparseTensorConversionOptions());

std::unique_ptr<Pass> createSparseTensorConversionPass();
std::unique_ptr<Pass>
createSparseTensorConversionPass(const SparseTensorConversionOptions &options);

//===----------------------------------------------------------------------===//
// The SparseTensorCodegen pass.
//===----------------------------------------------------------------------===//

/// Sparse tensor type converter into an actual buffer.
class SparseTensorTypeToBufferConverter : public TypeConverter {
public:
  SparseTensorTypeToBufferConverter();
};

/// Sets up sparse tensor codegen rules.
void populateSparseTensorCodegenPatterns(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         bool createSparseDeallocs,
                                         bool enableBufferInitialization);

std::unique_ptr<Pass> createSparseTensorCodegenPass();
std::unique_ptr<Pass>
createSparseTensorCodegenPass(bool createSparseDeallocs,
                              bool enableBufferInitialization);

//===----------------------------------------------------------------------===//
// The SparseBufferRewrite pass.
//===----------------------------------------------------------------------===//

void populateSparseBufferRewriting(RewritePatternSet &patterns,
                                   bool enableBufferInitialization);

std::unique_ptr<Pass> createSparseBufferRewritePass();
std::unique_ptr<Pass>
createSparseBufferRewritePass(bool enableBufferInitialization);

//===----------------------------------------------------------------------===//
// The SparseVectorization pass.
//===----------------------------------------------------------------------===//

void populateSparseVectorizationPatterns(RewritePatternSet &patterns,
                                         unsigned vectorLength,
                                         bool enableVLAVectorization,
                                         bool enableSIMDIndex32);

std::unique_ptr<Pass> createSparseVectorizationPass();
std::unique_ptr<Pass> createSparseVectorizationPass(unsigned vectorLength,
                                                    bool enableVLAVectorization,
                                                    bool enableSIMDIndex32);

//===----------------------------------------------------------------------===//
// The SparseGPU pass.
//===----------------------------------------------------------------------===//

void populateSparseGPUCodegenPatterns(RewritePatternSet &patterns,
                                      unsigned numThreads);

void populateSparseGPULibgenPatterns(RewritePatternSet &patterns, bool enableRT,
                                     GPUDataTransferStrategy gpuDataTransfer);

std::unique_ptr<Pass> createSparseGPUCodegenPass();
std::unique_ptr<Pass> createSparseGPUCodegenPass(unsigned numThreads);

//===----------------------------------------------------------------------===//
// The SparseStorageSpecifierToLLVM pass.
//===----------------------------------------------------------------------===//

class StorageSpecifierToLLVMTypeConverter : public TypeConverter {
public:
  StorageSpecifierToLLVMTypeConverter();
};

void populateStorageSpecifierToLLVMPatterns(TypeConverter &converter,
                                            RewritePatternSet &patterns);
std::unique_ptr<Pass> createStorageSpecifierToLLVMPass();

//===----------------------------------------------------------------------===//
// The mini-pipeline for sparsification and bufferization.
//===----------------------------------------------------------------------===//

bufferization::OneShotBufferizationOptions
getBufferizationOptionsForSparsification(bool analysisOnly);

std::unique_ptr<Pass> createSparsificationAndBufferizationPass();

std::unique_ptr<Pass> createSparsificationAndBufferizationPass(
    const bufferization::OneShotBufferizationOptions &bufferizationOptions,
    const SparsificationOptions &sparsificationOptions,
    const SparseTensorConversionOptions &sparseTensorConversionOptions,
    bool createSparseDeallocs, bool enableRuntimeLibrary,
    bool enableBufferInitialization, unsigned vectorLength,
    bool enableVLAVectorization, bool enableSIMDIndex32);

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES_H_
