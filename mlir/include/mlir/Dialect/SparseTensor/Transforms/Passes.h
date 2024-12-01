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
#include "mlir/Transforms/OneToNTypeConversion.h"

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

/// Defines a scope for reinterpret map pass.
enum class ReinterpretMapScope {
  kAll,           // reinterprets all applicable operations
  kGenericOnly,   // reinterprets only linalg.generic
  kExceptGeneric, // reinterprets operation other than linalg.generic
};

/// Defines a scope for reinterpret map pass.
enum class SparseEmitStrategy {
  kFunctional,     // generate fully inlined (and functional) sparse iteration
  kSparseIterator, // generate (experimental) loop using sparse iterator.
  kDebugInterface, // generate only place-holder for sparse iteration
};

#define GEN_PASS_DECL
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// The SparseAssembler pass.
//===----------------------------------------------------------------------===//

void populateSparseAssembler(RewritePatternSet &patterns, bool directOut);

std::unique_ptr<Pass> createSparseAssembler();
std::unique_ptr<Pass> createSparseAssembler(bool directOut);

//===----------------------------------------------------------------------===//
// The SparseReinterpretMap pass.
//===----------------------------------------------------------------------===//

void populateSparseReinterpretMap(RewritePatternSet &patterns,
                                  ReinterpretMapScope scope);

std::unique_ptr<Pass> createSparseReinterpretMapPass();
std::unique_ptr<Pass> createSparseReinterpretMapPass(ReinterpretMapScope scope);

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
  SparsificationOptions(SparseParallelizationStrategy p, SparseEmitStrategy d,
                        bool enableRT)
      : parallelizationStrategy(p), sparseEmitStrategy(d),
        enableRuntimeLibrary(enableRT) {}

  SparsificationOptions(SparseParallelizationStrategy p, bool enableRT)
      : SparsificationOptions(p, SparseEmitStrategy::kFunctional, enableRT) {}

  SparsificationOptions()
      : SparsificationOptions(SparseParallelizationStrategy::kNone,
                              SparseEmitStrategy::kFunctional, true) {}

  SparseParallelizationStrategy parallelizationStrategy;
  SparseEmitStrategy sparseEmitStrategy;
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
// The LowerSparseOpsToForeach pass.
//===----------------------------------------------------------------------===//

void populateLowerSparseOpsToForeachPatterns(RewritePatternSet &patterns,
                                             bool enableRT, bool enableConvert);

std::unique_ptr<Pass> createLowerSparseOpsToForeachPass();
std::unique_ptr<Pass> createLowerSparseOpsToForeachPass(bool enableRT,
                                                        bool enableConvert);

//===----------------------------------------------------------------------===//
// The LowerForeachToSCF pass.
//===----------------------------------------------------------------------===//

void populateLowerForeachToSCFPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createLowerForeachToSCFPass();

//===----------------------------------------------------------------------===//
// The LowerSparseIterationToSCF pass.
//===----------------------------------------------------------------------===//

/// Type converter for iter_space and iterator.
struct SparseIterationTypeConverter : public TypeConverter {
  SparseIterationTypeConverter();
};

void populateLowerSparseIterationToSCFPatterns(const TypeConverter &converter,
                                               RewritePatternSet &patterns);

std::unique_ptr<Pass> createLowerSparseIterationToSCFPass();

//===----------------------------------------------------------------------===//
// The SparseTensorConversion pass.
//===----------------------------------------------------------------------===//

/// Sparse tensor type converter into an opaque pointer.
class SparseTensorTypeToPtrConverter : public TypeConverter {
public:
  SparseTensorTypeToPtrConverter();
};

/// Sets up sparse tensor conversion rules.
void populateSparseTensorConversionPatterns(const TypeConverter &typeConverter,
                                            RewritePatternSet &patterns);

std::unique_ptr<Pass> createSparseTensorConversionPass();

//===----------------------------------------------------------------------===//
// The SparseTensorCodegen pass.
//===----------------------------------------------------------------------===//

/// Sparse tensor type converter into an actual buffer.
class SparseTensorTypeToBufferConverter : public TypeConverter {
public:
  SparseTensorTypeToBufferConverter();
};

/// Sets up sparse tensor codegen rules.
void populateSparseTensorCodegenPatterns(const TypeConverter &typeConverter,
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

void populateSparseGPULibgenPatterns(RewritePatternSet &patterns,
                                     bool enableRT);

std::unique_ptr<Pass> createSparseGPUCodegenPass();
std::unique_ptr<Pass> createSparseGPUCodegenPass(unsigned numThreads,
                                                 bool enableRT);

//===----------------------------------------------------------------------===//
// The SparseStorageSpecifierToLLVM pass.
//===----------------------------------------------------------------------===//

class StorageSpecifierToLLVMTypeConverter : public TypeConverter {
public:
  StorageSpecifierToLLVMTypeConverter();
};

void populateStorageSpecifierToLLVMPatterns(const TypeConverter &converter,
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
    bool createSparseDeallocs, bool enableRuntimeLibrary,
    bool enableBufferInitialization, unsigned vectorLength,
    bool enableVLAVectorization, bool enableSIMDIndex32, bool enableGPULibgen,
    SparseEmitStrategy emitStrategy,
    SparseParallelizationStrategy parallelizationStrategy);

//===----------------------------------------------------------------------===//
// Sparse Iteration Transform Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createSparseSpaceCollapsePass();

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES_H_
