//===- Rewrite.cpp - C API for Rewrite Patterns ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Rewrite.h"

#include "mlir-c/Transforms.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PDLPatternMatch.h.inc"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
/// RewriterBase API inherited from OpBuilder
//===----------------------------------------------------------------------===//

MlirContext mlirRewriterBaseGetContext(MlirRewriterBase rewriter) {
  return wrap(unwrap(rewriter)->getContext());
}

//===----------------------------------------------------------------------===//
/// Insertion points methods
//===----------------------------------------------------------------------===//

void mlirRewriterBaseClearInsertionPoint(MlirRewriterBase rewriter) {
  unwrap(rewriter)->clearInsertionPoint();
}

void mlirRewriterBaseSetInsertionPointBefore(MlirRewriterBase rewriter,
                                             MlirOperation op) {
  unwrap(rewriter)->setInsertionPoint(unwrap(op));
}

void mlirRewriterBaseSetInsertionPointAfter(MlirRewriterBase rewriter,
                                            MlirOperation op) {
  unwrap(rewriter)->setInsertionPointAfter(unwrap(op));
}

void mlirRewriterBaseSetInsertionPointAfterValue(MlirRewriterBase rewriter,
                                                 MlirValue value) {
  unwrap(rewriter)->setInsertionPointAfterValue(unwrap(value));
}

void mlirRewriterBaseSetInsertionPointToStart(MlirRewriterBase rewriter,
                                              MlirBlock block) {
  unwrap(rewriter)->setInsertionPointToStart(unwrap(block));
}

void mlirRewriterBaseSetInsertionPointToEnd(MlirRewriterBase rewriter,
                                            MlirBlock block) {
  unwrap(rewriter)->setInsertionPointToEnd(unwrap(block));
}

MlirBlock mlirRewriterBaseGetInsertionBlock(MlirRewriterBase rewriter) {
  return wrap(unwrap(rewriter)->getInsertionBlock());
}

MlirBlock mlirRewriterBaseGetBlock(MlirRewriterBase rewriter) {
  return wrap(unwrap(rewriter)->getBlock());
}

MlirOperation
mlirRewriterBaseGetOperationAfterInsertion(MlirRewriterBase rewriter) {
  mlir::RewriterBase *base = unwrap(rewriter);
  mlir::Block *block = base->getInsertionBlock();
  mlir::Block::iterator it = base->getInsertionPoint();
  if (it == block->end())
    return {nullptr};

  return wrap(std::addressof(*it));
}

//===----------------------------------------------------------------------===//
/// Block and operation creation/insertion/cloning
//===----------------------------------------------------------------------===//

MlirBlock mlirRewriterBaseCreateBlockBefore(MlirRewriterBase rewriter,
                                            MlirBlock insertBefore,
                                            intptr_t nArgTypes,
                                            MlirType const *argTypes,
                                            MlirLocation const *locations) {
  SmallVector<Type, 4> args;
  ArrayRef<Type> unwrappedArgs = unwrapList(nArgTypes, argTypes, args);
  SmallVector<Location, 4> locs;
  ArrayRef<Location> unwrappedLocs = unwrapList(nArgTypes, locations, locs);
  return wrap(unwrap(rewriter)->createBlock(unwrap(insertBefore), unwrappedArgs,
                                            unwrappedLocs));
}

MlirOperation mlirRewriterBaseInsert(MlirRewriterBase rewriter,
                                     MlirOperation op) {
  return wrap(unwrap(rewriter)->insert(unwrap(op)));
}

// Other methods of OpBuilder

MlirOperation mlirRewriterBaseClone(MlirRewriterBase rewriter,
                                    MlirOperation op) {
  return wrap(unwrap(rewriter)->clone(*unwrap(op)));
}

MlirOperation mlirRewriterBaseCloneWithoutRegions(MlirRewriterBase rewriter,
                                                  MlirOperation op) {
  return wrap(unwrap(rewriter)->cloneWithoutRegions(*unwrap(op)));
}

void mlirRewriterBaseCloneRegionBefore(MlirRewriterBase rewriter,
                                       MlirRegion region, MlirBlock before) {

  unwrap(rewriter)->cloneRegionBefore(*unwrap(region), unwrap(before));
}

//===----------------------------------------------------------------------===//
/// RewriterBase API
//===----------------------------------------------------------------------===//

void mlirRewriterBaseInlineRegionBefore(MlirRewriterBase rewriter,
                                        MlirRegion region, MlirBlock before) {
  unwrap(rewriter)->inlineRegionBefore(*unwrap(region), unwrap(before));
}

void mlirRewriterBaseReplaceOpWithValues(MlirRewriterBase rewriter,
                                         MlirOperation op, intptr_t nValues,
                                         MlirValue const *values) {
  SmallVector<Value, 4> vals;
  ArrayRef<Value> unwrappedVals = unwrapList(nValues, values, vals);
  unwrap(rewriter)->replaceOp(unwrap(op), unwrappedVals);
}

void mlirRewriterBaseReplaceOpWithOperation(MlirRewriterBase rewriter,
                                            MlirOperation op,
                                            MlirOperation newOp) {
  unwrap(rewriter)->replaceOp(unwrap(op), unwrap(newOp));
}

void mlirRewriterBaseEraseOp(MlirRewriterBase rewriter, MlirOperation op) {
  unwrap(rewriter)->eraseOp(unwrap(op));
}

void mlirRewriterBaseEraseBlock(MlirRewriterBase rewriter, MlirBlock block) {
  unwrap(rewriter)->eraseBlock(unwrap(block));
}

void mlirRewriterBaseInlineBlockBefore(MlirRewriterBase rewriter,
                                       MlirBlock source, MlirOperation op,
                                       intptr_t nArgValues,
                                       MlirValue const *argValues) {
  SmallVector<Value, 4> vals;
  ArrayRef<Value> unwrappedVals = unwrapList(nArgValues, argValues, vals);

  unwrap(rewriter)->inlineBlockBefore(unwrap(source), unwrap(op),
                                      unwrappedVals);
}

void mlirRewriterBaseMergeBlocks(MlirRewriterBase rewriter, MlirBlock source,
                                 MlirBlock dest, intptr_t nArgValues,
                                 MlirValue const *argValues) {
  SmallVector<Value, 4> args;
  ArrayRef<Value> unwrappedArgs = unwrapList(nArgValues, argValues, args);
  unwrap(rewriter)->mergeBlocks(unwrap(source), unwrap(dest), unwrappedArgs);
}

void mlirRewriterBaseMoveOpBefore(MlirRewriterBase rewriter, MlirOperation op,
                                  MlirOperation existingOp) {
  unwrap(rewriter)->moveOpBefore(unwrap(op), unwrap(existingOp));
}

void mlirRewriterBaseMoveOpAfter(MlirRewriterBase rewriter, MlirOperation op,
                                 MlirOperation existingOp) {
  unwrap(rewriter)->moveOpAfter(unwrap(op), unwrap(existingOp));
}

void mlirRewriterBaseMoveBlockBefore(MlirRewriterBase rewriter, MlirBlock block,
                                     MlirBlock existingBlock) {
  unwrap(rewriter)->moveBlockBefore(unwrap(block), unwrap(existingBlock));
}

void mlirRewriterBaseStartOpModification(MlirRewriterBase rewriter,
                                         MlirOperation op) {
  unwrap(rewriter)->startOpModification(unwrap(op));
}

void mlirRewriterBaseFinalizeOpModification(MlirRewriterBase rewriter,
                                            MlirOperation op) {
  unwrap(rewriter)->finalizeOpModification(unwrap(op));
}

void mlirRewriterBaseCancelOpModification(MlirRewriterBase rewriter,
                                          MlirOperation op) {
  unwrap(rewriter)->cancelOpModification(unwrap(op));
}

void mlirRewriterBaseReplaceAllUsesWith(MlirRewriterBase rewriter,
                                        MlirValue from, MlirValue to) {
  unwrap(rewriter)->replaceAllUsesWith(unwrap(from), unwrap(to));
}

void mlirRewriterBaseReplaceAllValueRangeUsesWith(MlirRewriterBase rewriter,
                                                  intptr_t nValues,
                                                  MlirValue const *from,
                                                  MlirValue const *to) {
  SmallVector<Value, 4> fromVals;
  ArrayRef<Value> unwrappedFromVals = unwrapList(nValues, from, fromVals);
  SmallVector<Value, 4> toVals;
  ArrayRef<Value> unwrappedToVals = unwrapList(nValues, to, toVals);
  unwrap(rewriter)->replaceAllUsesWith(unwrappedFromVals, unwrappedToVals);
}

void mlirRewriterBaseReplaceAllOpUsesWithValueRange(MlirRewriterBase rewriter,
                                                    MlirOperation from,
                                                    intptr_t nTo,
                                                    MlirValue const *to) {
  SmallVector<Value, 4> toVals;
  ArrayRef<Value> unwrappedToVals = unwrapList(nTo, to, toVals);
  unwrap(rewriter)->replaceAllOpUsesWith(unwrap(from), unwrappedToVals);
}

void mlirRewriterBaseReplaceAllOpUsesWithOperation(MlirRewriterBase rewriter,
                                                   MlirOperation from,
                                                   MlirOperation to) {
  unwrap(rewriter)->replaceAllOpUsesWith(unwrap(from), unwrap(to));
}

void mlirRewriterBaseReplaceOpUsesWithinBlock(MlirRewriterBase rewriter,
                                              MlirOperation op,
                                              intptr_t nNewValues,
                                              MlirValue const *newValues,
                                              MlirBlock block) {
  SmallVector<Value, 4> vals;
  ArrayRef<Value> unwrappedVals = unwrapList(nNewValues, newValues, vals);
  unwrap(rewriter)->replaceOpUsesWithinBlock(unwrap(op), unwrappedVals,
                                             unwrap(block));
}

void mlirRewriterBaseReplaceAllUsesExcept(MlirRewriterBase rewriter,
                                          MlirValue from, MlirValue to,
                                          MlirOperation exceptedUser) {
  unwrap(rewriter)->replaceAllUsesExcept(unwrap(from), unwrap(to),
                                         unwrap(exceptedUser));
}

//===----------------------------------------------------------------------===//
/// IRRewriter API
//===----------------------------------------------------------------------===//

MlirRewriterBase mlirIRRewriterCreate(MlirContext context) {
  return wrap(new IRRewriter(unwrap(context)));
}

MlirRewriterBase mlirIRRewriterCreateFromOp(MlirOperation op) {
  return wrap(new IRRewriter(unwrap(op)));
}

void mlirIRRewriterDestroy(MlirRewriterBase rewriter) {
  delete static_cast<IRRewriter *>(unwrap(rewriter));
}

//===----------------------------------------------------------------------===//
/// RewritePatternSet and FrozenRewritePatternSet API
//===----------------------------------------------------------------------===//

MlirFrozenRewritePatternSet
mlirFreezeRewritePattern(MlirRewritePatternSet set) {
  auto *m = new mlir::FrozenRewritePatternSet(std::move(*unwrap(set)));
  set.ptr = nullptr;
  return wrap(m);
}

void mlirFrozenRewritePatternSetDestroy(MlirFrozenRewritePatternSet set) {
  delete unwrap(set);
  set.ptr = nullptr;
}

MlirLogicalResult
mlirApplyPatternsAndFoldGreedily(MlirModule op,
                                 MlirFrozenRewritePatternSet patterns,
                                 MlirGreedyRewriteDriverConfig) {
  return wrap(mlir::applyPatternsGreedily(unwrap(op), *unwrap(patterns)));
}

MlirLogicalResult
mlirApplyPatternsAndFoldGreedilyWithOp(MlirOperation op,
                                       MlirFrozenRewritePatternSet patterns,
                                       MlirGreedyRewriteDriverConfig) {
  return wrap(mlir::applyPatternsGreedily(unwrap(op), *unwrap(patterns)));
}

//===----------------------------------------------------------------------===//
/// PatternRewriter API
//===----------------------------------------------------------------------===//

MlirRewriterBase mlirPatternRewriterAsBase(MlirPatternRewriter rewriter) {
  return wrap(static_cast<mlir::RewriterBase *>(unwrap(rewriter)));
}

//===----------------------------------------------------------------------===//
/// RewritePattern API
//===----------------------------------------------------------------------===//

namespace mlir {

class ExternalRewritePattern : public mlir::RewritePattern {
public:
  ExternalRewritePattern(MlirRewritePatternCallbacks callbacks, void *userData,
                         StringRef rootName, PatternBenefit benefit,
                         MLIRContext *context,
                         ArrayRef<StringRef> generatedNames)
      : RewritePattern(rootName, benefit, context, generatedNames),
        callbacks(callbacks), userData(userData) {
    if (callbacks.construct)
      callbacks.construct(userData);
  }

  ~ExternalRewritePattern() {
    if (callbacks.destruct)
      callbacks.destruct(userData);
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return unwrap(callbacks.matchAndRewrite(
        wrap(static_cast<const mlir::RewritePattern *>(this)), wrap(op),
        wrap(&rewriter), userData));
  }

private:
  MlirRewritePatternCallbacks callbacks;
  void *userData;
};

} // namespace mlir

MlirRewritePattern mlirOpRewritePatternCreate(
    MlirStringRef rootName, unsigned benefit, MlirContext context,
    MlirRewritePatternCallbacks callbacks, void *userData,
    size_t nGeneratedNames, MlirStringRef *generatedNames) {
  std::vector<mlir::StringRef> generatedNamesVec;
  generatedNamesVec.reserve(nGeneratedNames);
  for (size_t i = 0; i < nGeneratedNames; ++i) {
    generatedNamesVec.push_back(unwrap(generatedNames[i]));
  }
  return wrap(new mlir::ExternalRewritePattern(
      callbacks, userData, unwrap(rootName), PatternBenefit(benefit),
      unwrap(context), generatedNamesVec));
}

//===----------------------------------------------------------------------===//
/// RewritePatternSet API
//===----------------------------------------------------------------------===//

MlirRewritePatternSet mlirRewritePatternSetCreate(MlirContext context) {
  return wrap(new mlir::RewritePatternSet(unwrap(context)));
}

void mlirRewritePatternSetDestroy(MlirRewritePatternSet set) {
  delete unwrap(set);
}

void mlirRewritePatternSetAdd(MlirRewritePatternSet set,
                              MlirRewritePattern pattern) {
  std::unique_ptr<mlir::RewritePattern> patternPtr(
      const_cast<mlir::RewritePattern *>(unwrap(pattern)));
  pattern.ptr = nullptr;
  unwrap(set)->add(std::move(patternPtr));
}

//===----------------------------------------------------------------------===//
/// PDLPatternModule API
//===----------------------------------------------------------------------===//

#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
MlirPDLPatternModule mlirPDLPatternModuleFromModule(MlirModule op) {
  return wrap(new mlir::PDLPatternModule(
      mlir::OwningOpRef<mlir::ModuleOp>(unwrap(op))));
}

void mlirPDLPatternModuleDestroy(MlirPDLPatternModule op) {
  delete unwrap(op);
  op.ptr = nullptr;
}

MlirRewritePatternSet
mlirRewritePatternSetFromPDLPatternModule(MlirPDLPatternModule op) {
  auto *m = new mlir::RewritePatternSet(std::move(*unwrap(op)));
  op.ptr = nullptr;
  return wrap(m);
}

MlirValue mlirPDLValueAsValue(MlirPDLValue value) {
  return wrap(unwrap(value)->dyn_cast<mlir::Value>());
}

MlirType mlirPDLValueAsType(MlirPDLValue value) {
  return wrap(unwrap(value)->dyn_cast<mlir::Type>());
}

MlirOperation mlirPDLValueAsOperation(MlirPDLValue value) {
  return wrap(unwrap(value)->dyn_cast<mlir::Operation *>());
}

MlirAttribute mlirPDLValueAsAttribute(MlirPDLValue value) {
  return wrap(unwrap(value)->dyn_cast<mlir::Attribute>());
}

void mlirPDLResultListPushBackValue(MlirPDLResultList results,
                                    MlirValue value) {
  unwrap(results)->push_back(unwrap(value));
}

void mlirPDLResultListPushBackType(MlirPDLResultList results, MlirType value) {
  unwrap(results)->push_back(unwrap(value));
}

void mlirPDLResultListPushBackOperation(MlirPDLResultList results,
                                        MlirOperation value) {
  unwrap(results)->push_back(unwrap(value));
}

void mlirPDLResultListPushBackAttribute(MlirPDLResultList results,
                                        MlirAttribute value) {
  unwrap(results)->push_back(unwrap(value));
}

inline std::vector<MlirPDLValue> wrap(ArrayRef<PDLValue> values) {
  std::vector<MlirPDLValue> mlirValues;
  mlirValues.reserve(values.size());
  for (auto &value : values) {
    mlirValues.push_back(wrap(&value));
  }
  return mlirValues;
}

void mlirPDLPatternModuleRegisterRewriteFunction(
    MlirPDLPatternModule pdlModule, MlirStringRef name,
    MlirPDLRewriteFunction rewriteFn, void *userData) {
  unwrap(pdlModule)->registerRewriteFunction(
      unwrap(name),
      [userData, rewriteFn](PatternRewriter &rewriter, PDLResultList &results,
                            ArrayRef<PDLValue> values) -> LogicalResult {
        std::vector<MlirPDLValue> mlirValues = wrap(values);
        return unwrap(rewriteFn(wrap(&rewriter), wrap(&results),
                                mlirValues.size(), mlirValues.data(),
                                userData));
      });
}

void mlirPDLPatternModuleRegisterConstraintFunction(
    MlirPDLPatternModule pdlModule, MlirStringRef name,
    MlirPDLConstraintFunction constraintFn, void *userData) {
  unwrap(pdlModule)->registerConstraintFunction(
      unwrap(name),
      [userData, constraintFn](PatternRewriter &rewriter,
                               PDLResultList &results,
                               ArrayRef<PDLValue> values) -> LogicalResult {
        std::vector<MlirPDLValue> mlirValues = wrap(values);
        return unwrap(constraintFn(wrap(&rewriter), wrap(&results),
                                   mlirValues.size(), mlirValues.data(),
                                   userData));
      });
}
#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCH
