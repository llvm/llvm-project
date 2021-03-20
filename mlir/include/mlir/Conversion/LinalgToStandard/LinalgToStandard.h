//===- LinalgToStandard.h - Utils to convert from the linalg dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LINALGTOSTANDARD_LINALGTOSTANDARD_H_
#define MLIR_CONVERSION_LINALGTOSTANDARD_LINALGTOSTANDARD_H_

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;

namespace linalg {

//===----------------------------------------------------------------------===//
// Patterns to convert a LinalgOp to std.call @external library implementation.
//===----------------------------------------------------------------------===//
// These patterns are exposed individually because they are expected to be
// typically used individually.

// Create a new call to the type-canonicalized `LinalgOp::getLibraryCallName()`
// function. The implementation of the function can be either in the same module
// or in an externally linked library.
// This is a generic entry point for all LinalgOp, except for CopyOp and
// IndexedGenericOp, for which omre specialized patterns are provided.
class LinalgOpToLibraryCallRewrite : public RewritePattern {
public:
  LinalgOpToLibraryCallRewrite()
      : RewritePattern(/*benefit=*/1, MatchAnyOpTypeTag()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

/// Rewrite pattern specialization for CopyOp, kicks in when both input and
/// output permutations are left unspecified or are the identity.
class CopyOpToLibraryCallRewrite : public OpRewritePattern<CopyOp> {
public:
  using OpRewritePattern<CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CopyOp op,
                                PatternRewriter &rewriter) const override;
};

/// Rewrite CopyOp with permutations into a sequence of TransposeOp and
/// permutation-free CopyOp. This interplays with TransposeOpConversion and
/// LinalgConversion<CopyOp> to create a path to the LLVM dialect.
class CopyTransposeRewrite : public OpRewritePattern<CopyOp> {
public:
  using OpRewritePattern<CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CopyOp op,
                                PatternRewriter &rewriter) const override;
};

/// Conversion pattern specialization for IndexedGenericOp, has special handling
/// for the extra index operands.
class IndexedGenericOpToLibraryCallRewrite
    : public OpRewritePattern<IndexedGenericOp> {
public:
  using OpRewritePattern<IndexedGenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IndexedGenericOp op,
                                PatternRewriter &rewriter) const override;
};

/// Populate the given list with patterns that convert from Linalg to Standard.
void populateLinalgToStandardConversionPatterns(
    OwningRewritePatternList &patterns);

} // namespace linalg

/// Create a pass to convert Linalg operations to the Standard dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertLinalgToStandardPass();

} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOSTANDARD_LINALGTOSTANDARD_H_
