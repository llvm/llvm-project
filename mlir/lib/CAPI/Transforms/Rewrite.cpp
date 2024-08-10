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

//===----------------------------------------------------------------------===//
/// Block and operation creation/insertion/cloning

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

inline mlir::RewritePatternSet &unwrap(MlirRewritePatternSet module) {
  assert(module.ptr && "unexpected null module");
  return *(static_cast<mlir::RewritePatternSet *>(module.ptr));
}

inline MlirRewritePatternSet wrap(mlir::RewritePatternSet *module) {
  return {module};
}

inline mlir::FrozenRewritePatternSet *
unwrap(MlirFrozenRewritePatternSet module) {
  assert(module.ptr && "unexpected null module");
  return static_cast<mlir::FrozenRewritePatternSet *>(module.ptr);
}

inline MlirFrozenRewritePatternSet wrap(mlir::FrozenRewritePatternSet *module) {
  return {module};
}

MlirFrozenRewritePatternSet mlirFreezeRewritePattern(MlirRewritePatternSet op) {
  auto *m = new mlir::FrozenRewritePatternSet(std::move(unwrap(op)));
  op.ptr = nullptr;
  return wrap(m);
}

void mlirFrozenRewritePatternSetDestroy(MlirFrozenRewritePatternSet op) {
  delete unwrap(op);
  op.ptr = nullptr;
}

MlirLogicalResult
mlirApplyPatternsAndFoldGreedily(MlirModule op,
                                 MlirFrozenRewritePatternSet patterns,
                                 MlirGreedyRewriteDriverConfig) {
  return wrap(
      mlir::applyPatternsAndFoldGreedily(unwrap(op), *unwrap(patterns)));
}

//===----------------------------------------------------------------------===//
/// PDLPatternModule API
//===----------------------------------------------------------------------===//

#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
inline mlir::PDLPatternModule *unwrap(MlirPDLPatternModule module) {
  assert(module.ptr && "unexpected null module");
  return static_cast<mlir::PDLPatternModule *>(module.ptr);
}

inline MlirPDLPatternModule wrap(mlir::PDLPatternModule *module) {
  return {module};
}

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
#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCH
