//===- TypeConsistency.h - Rewrites to improve type consistency -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Set of rewrites to improve the coherency of types within an LLVM dialect
// program. This will adjust operations around a given pointer so they interpret
// its pointee type as consistently as possible.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_TYPECONSISTENCY_H
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_TYPECONSISTENCY_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace LLVM {

#define GEN_PASS_DECL_LLVMTYPECONSISTENCY
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"

/// Creates a pass that adjusts operations operating on pointers so they
/// interpret pointee types as consistently as possible.
std::unique_ptr<Pass> createTypeConsistencyPass();

/// Canonicalizes GEPs of which the base type and the pointer's type hint do not
/// match. This is done by replacing the original GEP into a GEP with the type
/// hint as a base type when an element of the hinted type aligns with the
/// original GEP.
class CanonicalizeAlignedGep : public OpRewritePattern<GEPOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GEPOp gep,
                                PatternRewriter &rewriter) const override;
};

/// Splits stores which write into multiple adjacent elements of an aggregate
/// through a pointer. Currently, integers and vector are split and stores
/// are generated for every element being stored to in a type-consistent manner.
/// This is done on a best-effort basis.
class SplitStores : public OpRewritePattern<StoreOp> {
  unsigned maxVectorSplitSize;

public:
  SplitStores(MLIRContext *context, unsigned maxVectorSplitSize)
      : OpRewritePattern(context), maxVectorSplitSize(maxVectorSplitSize) {}

  LogicalResult matchAndRewrite(StoreOp store,
                                PatternRewriter &rewrite) const override;
};

/// Splits GEPs with more than two indices into multiple GEPs with exactly
/// two indices. The created GEPs are then guaranteed to index into only
/// one aggregate at a time.
class SplitGEP : public OpRewritePattern<GEPOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GEPOp gepOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_TYPECONSISTENCY_H
