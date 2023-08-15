//===- BufferDeallocationSimplification.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for optimizing `bufferization.dealloc` operations
// that requires more analysis than what can be supported by regular
// canonicalization patterns.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_BUFFERDEALLOCATIONSIMPLIFICATION
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir;
using namespace mlir::bufferization;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static LogicalResult updateDeallocIfChanged(DeallocOp deallocOp,
                                            ValueRange memrefs,
                                            ValueRange conditions,
                                            PatternRewriter &rewriter) {
  if (deallocOp.getMemrefs() == memrefs &&
      deallocOp.getConditions() == conditions)
    return failure();

  rewriter.updateRootInPlace(deallocOp, [&]() {
    deallocOp.getMemrefsMutable().assign(memrefs);
    deallocOp.getConditionsMutable().assign(conditions);
  });
  return success();
}

/// Checks if 'memref' may or must alias a MemRef in 'memrefList'. It is often a
/// requirement of optimization patterns that there cannot be any aliasing
/// memref in order to perform the desired simplification. The 'allowSelfAlias'
/// argument indicates whether 'memref' may be present in 'memrefList' which
/// makes this helper function applicable to situations where we already know
/// that 'memref' is in the list but also when we don't want it in the list.
static bool potentiallyAliasesMemref(AliasAnalysis &analysis,
                                     ValueRange memrefList, Value memref,
                                     bool allowSelfAlias) {
  for (auto mr : memrefList) {
    if (allowSelfAlias && mr == memref)
      continue;
    if (!analysis.alias(mr, memref).isNo())
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Remove values from the `memref` operand list that are also present in the
/// `retained` list since they will always alias and thus never actually be
/// deallocated. However, we also need to be certain that no other value in the
/// `retained` list can alias, for which we use a static alias analysis. This is
/// necessary because the `dealloc` operation is defined to return one `i1`
/// value per memref in the `retained` list which represents the disjunction of
/// the condition values corresponding to all aliasing values in the `memref`
/// list. In particular, this means that if there is some value R in the
/// `retained` list which aliases with a value M in the `memref` list (but can
/// only be staticaly determined to may-alias) and M is also present in the
/// `retained` list, then it would be illegal to remove M because the result
/// corresponding to R would be computed incorrectly afterwards.
/// Because we require an alias analysis, this pattern cannot be applied as a
/// regular canonicalization pattern.
///
/// Example:
/// ```mlir
/// %0:3 = bufferization.dealloc (%m0 : ...) if (%cond0)
///                     retain (%m0, %r0, %r1 : ...)
/// ```
/// is canonicalized to
/// ```mlir
/// // bufferization.dealloc without memrefs and conditions returns %false for
/// // every retained value
/// %0:3 = bufferization.dealloc retain (%m0, %r0, %r1 : ...)
/// %1 = arith.ori %0#0, %cond0 : i1
/// // replace %0#0 with %1
/// ```
/// given that `%r0` and `%r1` may not alias with `%m0`.
struct DeallocRemoveDeallocMemrefsContainedInRetained
    : public OpRewritePattern<DeallocOp> {
  DeallocRemoveDeallocMemrefsContainedInRetained(MLIRContext *context,
                                                 AliasAnalysis &aliasAnalysis)
      : OpRewritePattern<DeallocOp>(context), aliasAnalysis(aliasAnalysis) {}

  LogicalResult matchAndRewrite(DeallocOp deallocOp,
                                PatternRewriter &rewriter) const override {
    // Unique memrefs to be deallocated.
    DenseMap<Value, unsigned> retained;
    for (auto [i, ret] : llvm::enumerate(deallocOp.getRetained()))
      retained[ret] = i;

    // There must not be any duplicates in the retain list anymore because we
    // would miss updating one of the result values otherwise.
    if (retained.size() != deallocOp.getRetained().size())
      return failure();

    SmallVector<Value> newMemrefs, newConditions;
    for (auto memrefAndCond :
         llvm::zip(deallocOp.getMemrefs(), deallocOp.getConditions())) {
      Value memref = std::get<0>(memrefAndCond);
      Value cond = std::get<1>(memrefAndCond);

      auto replaceResultsIfNoInvalidAliasing = [&](Value memref) -> bool {
        Value retainedMemref = deallocOp.getRetained()[retained[memref]];
        // The current memref must not have a may-alias relation to any retained
        // memref, and exactly one must-alias relation.
        // TODO: it is possible to extend this pattern to allow an arbitrary
        // number of must-alias relations as long as there is no may-alias. If
        // it's no-alias, then just proceed (only supported case as of now), if
        // it's must-alias, we also need to update the condition for that alias.
        if (llvm::all_of(deallocOp.getRetained(), [&](Value mr) {
              return aliasAnalysis.alias(mr, memref).isNo() ||
                     mr == retainedMemref;
            })) {
          rewriter.setInsertionPointAfter(deallocOp);
          auto orOp = rewriter.create<arith::OrIOp>(
              deallocOp.getLoc(),
              deallocOp.getUpdatedConditions()[retained[memref]], cond);
          rewriter.replaceAllUsesExcept(
              deallocOp.getUpdatedConditions()[retained[memref]],
              orOp.getResult(), orOp);
          return true;
        }
        return false;
      };

      if (retained.contains(memref) &&
          replaceResultsIfNoInvalidAliasing(memref))
        continue;

      auto extractOp = memref.getDefiningOp<memref::ExtractStridedMetadataOp>();
      if (extractOp && retained.contains(extractOp.getOperand()) &&
          replaceResultsIfNoInvalidAliasing(extractOp.getOperand()))
        continue;

      newMemrefs.push_back(memref);
      newConditions.push_back(cond);
    }

    // Return failure if we don't change anything such that we don't run into an
    // infinite loop of pattern applications.
    return updateDeallocIfChanged(deallocOp, newMemrefs, newConditions,
                                  rewriter);
  }

private:
  AliasAnalysis &aliasAnalysis;
};

/// Remove memrefs from the `retained` list which are guaranteed to not alias
/// any memref in the `memrefs` list. The corresponding result value can be
/// replaced with `false` in that case according to the operation description.
///
/// Example:
/// ```mlir
/// %0:2 = bufferization.dealloc (%m : memref<2xi32>) if (%cond)
///                       retain (%r0, %r1 : memref<2xi32>, memref<2xi32>)
/// return %0#0, %0#1
/// ```
/// can be canonicalized to the following given that `%r0` and `%r1` do not
/// alias `%m`:
/// ```mlir
/// bufferization.dealloc (%m : memref<2xi32>) if (%cond)
/// return %false, %false
/// ```
struct RemoveRetainedMemrefsGuaranteedToNotAlias
    : public OpRewritePattern<DeallocOp> {
  RemoveRetainedMemrefsGuaranteedToNotAlias(MLIRContext *context,
                                            AliasAnalysis &aliasAnalysis)
      : OpRewritePattern<DeallocOp>(context), aliasAnalysis(aliasAnalysis) {}

  LogicalResult matchAndRewrite(DeallocOp deallocOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> newRetainedMemrefs, replacements;
    Value falseValue;
    auto getOrCreateFalse = [&]() -> Value {
      if (!falseValue)
        falseValue = rewriter.create<arith::ConstantOp>(
            deallocOp.getLoc(), rewriter.getBoolAttr(false));
      return falseValue;
    };

    for (auto retainedMemref : deallocOp.getRetained()) {
      if (potentiallyAliasesMemref(aliasAnalysis, deallocOp.getMemrefs(),
                                   retainedMemref, false)) {
        newRetainedMemrefs.push_back(retainedMemref);
        replacements.push_back({});
        continue;
      }

      replacements.push_back(getOrCreateFalse());
    }

    if (newRetainedMemrefs.size() == deallocOp.getRetained().size())
      return failure();

    auto newDeallocOp = rewriter.create<DeallocOp>(
        deallocOp.getLoc(), deallocOp.getMemrefs(), deallocOp.getConditions(),
        newRetainedMemrefs);
    int i = 0;
    for (auto &repl : replacements) {
      if (!repl)
        repl = newDeallocOp.getUpdatedConditions()[i++];
    }

    rewriter.replaceOp(deallocOp, replacements);
    return success();
  }

private:
  AliasAnalysis &aliasAnalysis;
};

/// Split off memrefs to separate dealloc operations to reduce the number of
/// runtime checks required and enable further canonicalization of the new and
/// simpler dealloc operations. A memref can be split off if it is guaranteed to
/// not alias with any other memref in the `memref` operand list.  The results
/// of the old and the new dealloc operation have to be combined by computing
/// the element-wise disjunction of them.
///
/// Example:
/// ```mlir
/// %0:2 = bufferization.dealloc (%m0, %m1 : memref<2xi32>, memref<2xi32>)
///                           if (%cond0, %cond1)
///                       retain (%r0, %r1 : memref<2xi32>, memref<2xi32>)
/// return %0#0, %0#1
/// ```
/// Given that `%m0` is guaranteed to never alias with `%m1`, the above IR is
/// canonicalized to the following, thus reducing the number of runtime alias
/// checks by 1 and potentially enabling further canonicalization of the new
/// split-up dealloc operations.
/// ```mlir
/// %0:2 = bufferization.dealloc (%m0 : memref<2xi32>) if (%cond0)
///                       retain (%r0, %r1 : memref<2xi32>, memref<2xi32>)
/// %1:2 = bufferization.dealloc (%m1 : memref<2xi32>) if (%cond1)
///                       retain (%r0, %r1 : memref<2xi32>, memref<2xi32>)
/// %2 = arith.ori %0#0, %1#0
/// %3 = arith.ori %0#1, %1#1
/// return %2, %3
/// ```
struct SplitDeallocWhenNotAliasingAnyOther
    : public OpRewritePattern<DeallocOp> {
  SplitDeallocWhenNotAliasingAnyOther(MLIRContext *context,
                                      AliasAnalysis &aliasAnalysis)
      : OpRewritePattern<DeallocOp>(context), aliasAnalysis(aliasAnalysis) {}

  LogicalResult matchAndRewrite(DeallocOp deallocOp,
                                PatternRewriter &rewriter) const override {
    if (deallocOp.getMemrefs().size() <= 1)
      return failure();

    SmallVector<Value> newMemrefs, newConditions, replacements;
    DenseSet<Operation *> exceptedUsers;
    replacements = deallocOp.getUpdatedConditions();
    for (auto [memref, cond] :
         llvm::zip(deallocOp.getMemrefs(), deallocOp.getConditions())) {
      if (potentiallyAliasesMemref(aliasAnalysis, deallocOp.getMemrefs(),
                                   memref, true)) {
        newMemrefs.push_back(memref);
        newConditions.push_back(cond);
        continue;
      }

      auto newDeallocOp = rewriter.create<DeallocOp>(
          deallocOp.getLoc(), memref, cond, deallocOp.getRetained());
      replacements = SmallVector<Value>(llvm::map_range(
          llvm::zip(replacements, newDeallocOp.getUpdatedConditions()),
          [&](auto replAndNew) -> Value {
            auto orOp = rewriter.create<arith::OrIOp>(deallocOp.getLoc(),
                                                      std::get<0>(replAndNew),
                                                      std::get<1>(replAndNew));
            exceptedUsers.insert(orOp);
            return orOp.getResult();
          }));
    }

    if (newMemrefs.size() == deallocOp.getMemrefs().size())
      return failure();

    rewriter.replaceUsesWithIf(deallocOp.getUpdatedConditions(), replacements,
                               [&](OpOperand &operand) {
                                 return !exceptedUsers.contains(
                                     operand.getOwner());
                               });
    return updateDeallocIfChanged(deallocOp, newMemrefs, newConditions,
                                  rewriter);
  }

private:
  AliasAnalysis &aliasAnalysis;
};

} // namespace

//===----------------------------------------------------------------------===//
// BufferDeallocationSimplificationPass
//===----------------------------------------------------------------------===//

namespace {

/// The actual buffer deallocation pass that inserts and moves dealloc nodes
/// into the right positions. Furthermore, it inserts additional clones if
/// necessary. It uses the algorithm described at the top of the file.
struct BufferDeallocationSimplificationPass
    : public bufferization::impl::BufferDeallocationSimplificationBase<
          BufferDeallocationSimplificationPass> {
  void runOnOperation() override {
    AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
    RewritePatternSet patterns(&getContext());
    patterns.add<DeallocRemoveDeallocMemrefsContainedInRetained,
                 RemoveRetainedMemrefsGuaranteedToNotAlias,
                 SplitDeallocWhenNotAliasingAnyOther>(&getContext(),
                                                      aliasAnalysis);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::bufferization::createBufferDeallocationSimplificationPass() {
  return std::make_unique<BufferDeallocationSimplificationPass>();
}
