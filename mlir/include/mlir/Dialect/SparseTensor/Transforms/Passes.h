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
// In general, this file takes the approach of keeping "mechanism" (the
// actual steps of applying a transformation) completely separate from
// "policy" (heuristics for when and where to apply transformations).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace bufferization {
struct OneShotBufferizationOptions;
} // namespace bufferization

#define GEN_PASS_DECL_SPARSIFICATIONPASS
#define GEN_PASS_DECL_SPARSETENSORCONVERSIONPASS
#define GEN_PASS_DECL_SPARSETENSORCODEGEN
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// The Sparsification pass.
//===----------------------------------------------------------------------===//

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
  // TODO: support reduction parallelization too?
};

/// Converts command-line parallelization flag to the strategy enum.
SparseParallelizationStrategy sparseParallelizationStrategy(int32_t flag);

/// Defines a vectorization strategy. Any inner loop is a candidate (full SIMD
/// for parallel loops and horizontal SIMD for reduction loops). A loop is
/// actually vectorized if (1) allowed by the strategy, and (2) the emitted
/// code is an actual for-loop (and not a co-iterating while-loop).
enum class SparseVectorizationStrategy {
  kNone,
  kDenseInnerLoop,
  kAnyStorageInnerLoop
};

/// Converts command-line vectorization flag to the strategy enum.
SparseVectorizationStrategy sparseVectorizationStrategy(int32_t flag);

/// Options for the Sparsification pass.
struct SparsificationOptions {
  SparsificationOptions(SparseParallelizationStrategy p,
                        SparseVectorizationStrategy v, unsigned vl, bool e,
                        bool vla)
      : parallelizationStrategy(p), vectorizationStrategy(v), vectorLength(vl),
        enableSIMDIndex32(e), enableVLAVectorization(vla) {}
  SparsificationOptions()
      : SparsificationOptions(SparseParallelizationStrategy::kNone,
                              SparseVectorizationStrategy::kNone, 1u, false,
                              false) {}
  SparseParallelizationStrategy parallelizationStrategy;
  SparseVectorizationStrategy vectorizationStrategy;
  unsigned vectorLength;
  bool enableSIMDIndex32;
  bool enableVLAVectorization;
};

/// Sets up sparsification rewriting rules with the given options.
void populateSparsificationPatterns(
    RewritePatternSet &patterns,
    const SparsificationOptions &options = SparsificationOptions());

std::unique_ptr<Pass> createSparsificationPass();
std::unique_ptr<Pass>
createSparsificationPass(const SparsificationOptions &options);

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
///
/// The presence of the `kAuto` option violates our usual goal of keeping
/// policy completely separated from mechanism.  The reason it exists is
/// because (at present) this strategy can only be specified on a per-file
/// basis.  To see why this is a problem, note that `kDirect` cannot
/// support certain conversions; so if there is no `kAuto` setting,
/// then whenever a file contains a single non-`kDirect`-able conversion
/// the user would be forced to use `kViaCOO` for all conversions in
/// that file!  In the future, instead of using this enum as a `Pass`
/// option, we could instead move it to being an attribute on the
/// conversion op; at which point `kAuto` would no longer be necessary.
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

/// Sets up sparse tensor conversion rules.
void populateSparseTensorCodegenPatterns(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns);

std::unique_ptr<Pass> createSparseTensorCodegenPass();

//===----------------------------------------------------------------------===//
// The SparseTensorStorageExpansion pass.
//===----------------------------------------------------------------------===//

/// Sparse tensor storage type converter from compound to expanded form.
class SparseTensorStorageTupleExpander : public TypeConverter {
public:
  SparseTensorStorageTupleExpander();
};

/// Sets up sparse tensor storage expansion rules.
void populateSparseTensorStorageExpansionPatterns(TypeConverter &typeConverter,
                                                  RewritePatternSet &patterns);

std::unique_ptr<Pass> createSparseTensorStorageExpansionPass();

//===----------------------------------------------------------------------===//
// Other rewriting rules and passes.
//===----------------------------------------------------------------------===//

void populateSparseTensorRewriting(RewritePatternSet &patterns);

std::unique_ptr<Pass> createDenseBufferizationPass(
    const bufferization::OneShotBufferizationOptions &options);

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_PASSES_H_
