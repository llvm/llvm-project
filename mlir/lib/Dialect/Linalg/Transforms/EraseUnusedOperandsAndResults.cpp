//===- EraseUnusedOperandsAndResults.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;
using namespace mlir::linalg;

/// Return `true` if the `result` of an operation `genericOp` is dead.
static bool isResultValueDead(linalg::GenericOp genericOp, OpResult result) {
  if (!result.use_empty())
    return false;
  // If out operand not used in payload, we can drop it.
  OpOperand *outputOpOperand =
      genericOp.getDpsInitOperand(result.getResultNumber());
  if (!genericOp.payloadUsesValueFromOperand(outputOpOperand))
    return true;

  // The out operand that is part of a payload can be dropped if
  // these conditions are met:
  // - Result from out operand is dead.
  // - User of arg is yield.
  // - outArg data is not being used by other outArgs.

  // Check block arg and cycle from out operand has a single use.
  BlockArgument outputArg =
      genericOp.getRegionOutputArgs()[result.getResultNumber()];
  if (!outputArg.hasOneUse())
    return false;
  Operation *argUserOp = *outputArg.user_begin();

  // Check argUser has no other use.
  if (!argUserOp->use_empty())
    return false;

  // Check that argUser is a yield.
  auto yieldOp = dyn_cast<linalg::YieldOp>(argUserOp);
  if (!yieldOp)
    return false;

  // Check outArg data is not being used by other outArgs.
  if (yieldOp.getOperand(result.getResultNumber()) != outputArg)
    return false;

  return true;
}

namespace {

struct DeduplicateAndRemoveDeadOperandsAndResults
    : public OpRewritePattern<GenericOp> {
  DeduplicateAndRemoveDeadOperandsAndResults(MLIRContext *ctx,
                                             bool removeOutputs)
      : OpRewritePattern<GenericOp>(ctx), removeOutputs(removeOutputs) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Create a map from argument position in the original op to the argument
    // position in the new op. If the argument is dropped it wont have an entry.
    SmallVector<OpOperand *> droppedOpOperands;

    // Information needed to build the new op.
    SmallVector<Value> newInputOperands, newOutputOperands;
    SmallVector<AffineMap> newIndexingMaps;

    // Gather information about duplicate input operands.
    llvm::SmallDenseMap<unsigned, unsigned> origInsToNewInsPos =
        deduplicateInputOperands(genericOp, droppedOpOperands, newInputOperands,
                                 newIndexingMaps);

    // Gather information about the dropped outputs.
    llvm::SmallDenseMap<unsigned, unsigned> origOutsToNewOutsPos =
        deduplicateOutputOperands(genericOp, droppedOpOperands,
                                  newOutputOperands, newIndexingMaps);

    // Check if there is any change to operands.
    if (newInputOperands.size() + newOutputOperands.size() ==
        genericOp->getNumOperands())
      return failure();

    // Create the new op with the body being empty.
    Location loc = genericOp.getLoc();
    SmallVector<Type> newResultTypes;
    for (Value v : newOutputOperands)
      if (isa<TensorType>(v.getType()))
        newResultTypes.push_back(v.getType());
    auto newOp = rewriter.create<GenericOp>(
        loc, newResultTypes, newInputOperands, newOutputOperands,
        rewriter.getAffineMapArrayAttr(newIndexingMaps),
        genericOp.getIteratorTypes(), genericOp.getDocAttr(),
        genericOp.getLibraryCallAttr(),
        [](OpBuilder & /*builder*/, Location /*loc*/, ValueRange /*args*/) {
          return;
        });
    // Copy over unknown attributes. They might be load bearing for some flow.
    ArrayRef<StringRef> odsAttrs = genericOp.getAttributeNames();
    for (NamedAttribute kv : genericOp->getAttrs())
      if (!llvm::is_contained(odsAttrs, kv.getName().getValue()))
        newOp->setAttr(kv.getName(), kv.getValue());

    // Fix up the payload of the canonicalized operation.
    populateOpPayload(genericOp, newOp, origInsToNewInsPos,
                      origOutsToNewOutsPos, rewriter);

    // Replace all live uses of the op.
    SmallVector<Value> replacementsVals(genericOp->getNumResults(), nullptr);
    for (const auto &result : llvm::enumerate(genericOp.getResults())) {
      auto it = origOutsToNewOutsPos.find(result.index());
      if (it == origOutsToNewOutsPos.end())
        continue;
      replacementsVals[result.index()] = newOp.getResult(it->second);
    }
    rewriter.replaceOp(genericOp, replacementsVals);
    return success();
  }

private:
  /// If unset, outputs are not modified by this pattern.
  bool removeOutputs;

  // Deduplicate input operands, and return the
  // - Mapping from operand position in the original op, to operand position in
  // the canonicalized op.
  // - The preserved input operands list (by reference).
  llvm::SmallDenseMap<unsigned, unsigned>
  deduplicateInputOperands(GenericOp genericOp,
                           SmallVector<OpOperand *> &droppedOpOperands,
                           SmallVector<Value> &newInputOperands,
                           SmallVector<AffineMap> &newIndexingMaps) const {
    llvm::SmallDenseMap<unsigned, unsigned> origToNewPos;
    llvm::SmallDenseMap<std::pair<Value, AffineMap>, unsigned> dedupedInputs;
    for (const auto &en : llvm::enumerate(genericOp.getDpsInputOperands())) {
      OpOperand *inputOpOperand = en.value();
      // Check if operand is dead and if dropping the indexing map makes the
      // loops to shape computation invalid.
      if (!genericOp.payloadUsesValueFromOperand(inputOpOperand)) {
        // Add the current operands to the list of potentially droppable
        // operands. If it cannot be dropped, this needs to be popped back.
        droppedOpOperands.push_back(inputOpOperand);
        if (genericOp.canOpOperandsBeDropped(droppedOpOperands))
          continue;
        droppedOpOperands.pop_back();
      }

      // Check if this operand is a duplicate.
      AffineMap indexingMap = genericOp.getMatchingIndexingMap(inputOpOperand);
      auto it = dedupedInputs.find(
          std::make_pair(inputOpOperand->get(), indexingMap));
      if (it != dedupedInputs.end()) {
        origToNewPos[en.index()] = it->second;
        droppedOpOperands.push_back(inputOpOperand);
        continue;
      }

      // This is a preserved argument.
      origToNewPos[en.index()] = newInputOperands.size();
      dedupedInputs[{inputOpOperand->get(), indexingMap}] =
          newInputOperands.size();
      newInputOperands.push_back(inputOpOperand->get());
      newIndexingMaps.push_back(indexingMap);
    }
    return origToNewPos;
  }

  // Deduplicate output operands, and return the
  // - Mapping from operand position in the original op, to operand position in
  // the canonicalized op.
  // - The preserved output operands list (by reference).
  llvm::SmallDenseMap<unsigned, unsigned>
  deduplicateOutputOperands(GenericOp genericOp,
                            SmallVector<OpOperand *> &droppedOpOperands,
                            SmallVector<Value> &newOutputOperands,
                            SmallVector<AffineMap> &newIndexingMaps) const {
    llvm::SmallDenseMap<unsigned, unsigned> origToNewPos;
    llvm::SmallDenseMap<std::tuple<Value, AffineMap, Value>, unsigned>
        dedupedOutpts;
    // If the op doesn't have tensor semantics or outputs should not be removed,
    // keep all the outputs as preserved.
    if (!genericOp.hasTensorSemantics() || !removeOutputs) {
      for (const auto &en : llvm::enumerate(genericOp.getDpsInitsMutable())) {
        origToNewPos[en.index()] = newOutputOperands.size();
        newOutputOperands.push_back(en.value().get());
        newIndexingMaps.push_back(
            genericOp.getMatchingIndexingMap(&en.value()));
      }
      return origToNewPos;
    }
    // Output argument can be dropped if the result has
    // - no users, and
    // - it is not used in the payload, and
    // - the corresponding indexing maps are not needed for loop bound
    //   computation.
    auto yieldOp = cast<YieldOp>(genericOp.getBody()->getTerminator());
    for (const auto &outputOpOperand :
         llvm::enumerate(genericOp.getDpsInitsMutable())) {
      OpResult result = genericOp.getTiedOpResult(&outputOpOperand.value());
      AffineMap indexingMap =
          genericOp.getMatchingIndexingMap(&outputOpOperand.value());
      auto key = std::make_tuple(outputOpOperand.value().get(), indexingMap,
                                 yieldOp->getOperand(outputOpOperand.index()));
      if (isResultValueDead(genericOp, result)) {
        // Check if the opoperand can be dropped without affecting loop
        // bound computation. Add the operand to the list of dropped op
        // operand for checking. If it cannot be dropped, need to pop the
        // value back.
        droppedOpOperands.push_back(&outputOpOperand.value());
        if (genericOp.canOpOperandsBeDropped(droppedOpOperands)) {
          continue;
        }
        droppedOpOperands.pop_back();
      }

      if (!genericOp.payloadUsesValueFromOperand(&outputOpOperand.value())) {
        // The out operand can also be dropped if it is computed redundantly
        // by another result, the conditions for that are
        // - The same operand is used as the out operand
        // - The same indexing map is used
        // - The same yield value is used.
        auto it = dedupedOutpts.find(key);
        if (it != dedupedOutpts.end()) {
          origToNewPos[outputOpOperand.index()] = it->second;
          droppedOpOperands.push_back(&outputOpOperand.value());
          continue;
        }
      }

      origToNewPos[outputOpOperand.index()] = newOutputOperands.size();
      dedupedOutpts[key] = newOutputOperands.size();
      newOutputOperands.push_back(outputOpOperand.value().get());
      newIndexingMaps.push_back(
          genericOp.getMatchingIndexingMap(&outputOpOperand.value()));
    }
    return origToNewPos;
  }

  // Populate the body of the canonicalized operation.
  void populateOpPayload(
      GenericOp genericOp, GenericOp newOp,
      const llvm::SmallDenseMap<unsigned, unsigned> &origInsToNewInsPos,
      const llvm::SmallDenseMap<unsigned, unsigned> &origOutsToNewOutsPos,
      PatternRewriter &rewriter) const {
    // Merge the body of the original op with the new op.
    Block *newOpBlock = &newOp.getRegion().front();
    assert(newOpBlock->empty() && "expected new op to have an empty payload");
    Block *origOpBlock = &genericOp.getRegion().front();
    SmallVector<Value> replacements(origOpBlock->getNumArguments(), nullptr);

    // Replace all arguments in the original op, with arguments from the
    // canonicalized op.
    auto updateReplacements =
        [&](SmallVector<OpOperand *> &origOperands,
            SmallVector<OpOperand *> &newOperands,
            const llvm::SmallDenseMap<unsigned, unsigned> &map) {
          for (const auto &origOperand : llvm::enumerate(origOperands)) {
            auto it = map.find(origOperand.index());
            if (it == map.end())
              continue;
            OpOperand *newOperand = newOperands[it->second];
            replacements[origOperand.value()->getOperandNumber()] =
                newOpBlock->getArgument(newOperand->getOperandNumber());
          }
        };

    SmallVector<OpOperand *> origInputOperands =
        genericOp.getDpsInputOperands();
    SmallVector<OpOperand *> newInputOperands = newOp.getDpsInputOperands();
    updateReplacements(origInputOperands, newInputOperands, origInsToNewInsPos);

    SmallVector<OpOperand *> origOutputOperands =
        llvm::to_vector(llvm::map_range(genericOp.getDpsInitsMutable(),
                                        [](OpOperand &o) { return &o; }));
    SmallVector<OpOperand *> newOutputOperands =
        llvm::to_vector(llvm::map_range(newOp.getDpsInitsMutable(),
                                        [](OpOperand &o) { return &o; }));
    updateReplacements(origOutputOperands, newOutputOperands,
                       origOutsToNewOutsPos);

    // Drop the unused yield args.
    if (newOp.getNumDpsInits() != genericOp.getNumDpsInits()) {
      OpBuilder::InsertionGuard g(rewriter);
      YieldOp origYieldOp = cast<YieldOp>(origOpBlock->getTerminator());
      rewriter.setInsertionPoint(origYieldOp);

      SmallVector<Value> newYieldVals(newOp.getNumDpsInits(), nullptr);
      for (const auto &yieldOpOperands :
           llvm::enumerate(origYieldOp.getValues())) {
        auto it = origOutsToNewOutsPos.find(yieldOpOperands.index());
        if (it == origOutsToNewOutsPos.end())
          continue;
        newYieldVals[it->second] = yieldOpOperands.value();
      }
      rewriter.replaceOpWithNewOp<YieldOp>(origYieldOp, newYieldVals);
    }

    rewriter.mergeBlocks(origOpBlock, newOpBlock, replacements);
  }
};

/// Remove unused cycles.
/// We can remove unused cycle within a payload of generic region
/// if these conditions are met:
/// - Result from out operand is dead.
/// - Block arg from out operand has a single use in the %cycle
/// instruction.
/// - Cycle has a single use and it is in yield.
struct RemoveUnusedCycleInGenericOp : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {

    // If the op doesnt have tensor semantics, preserve the outputs as is.
    if (!genericOp.hasTensorSemantics())
      return failure();

    bool hasRemovedCycles = false;
    // Iterate over output operands and remove any unused cycles.
    for (const auto &outputOpOperand :
         llvm::enumerate(genericOp.getDpsInits())) {

      // Check that result from out operand is dead.
      Value result = genericOp.getResult(outputOpOperand.index());
      if (!result.use_empty())
        continue;

      // Check that outputArg has one use in cycle.
      BlockArgument outputArg =
          genericOp.getRegionOutputArgs()[outputOpOperand.index()];
      if (!outputArg.hasOneUse())
        continue;

      // Check cycle has at most one use.
      Operation *cycleOp = *outputArg.user_begin();
      if (!cycleOp->hasOneUse())
        continue;

      // Check that the cycleUser is a yield.
      Operation *cycleUserOp = *cycleOp->user_begin();
      if (!isa<linalg::YieldOp>(cycleUserOp))
        continue;

      // Check that argIndex matches yieldIndex, else data is being used.
      if (cycleUserOp->getOperand(outputOpOperand.index()) !=
          cycleOp->getResult(0))
        continue;

      // Directly replace the cycle with the blockArg such that
      // Deduplicate pattern can eliminate it along with unused yield.
      rewriter.replaceOp(cycleOp, outputArg);
      rewriter.updateRootInPlace(genericOp, [] {});
      hasRemovedCycles = true;
    }

    if (hasRemovedCycles) {
      return success();
    }

    return failure();
  }
};

/// Fold uses of duplicate inputs in the body of a linalg.generic. E.g.:
/// ```
/// linalg.generic ins(%a, %b, %a, %b) outs(%a)
/// ^bb0(%in0, %in1, %in2, %in3, %out1)
/// ```
/// Assuming that all %a and %b have the same index map:
/// * All uses of %in0 and %in2 are replaced with %out1
/// * All uses of %in1 are replaced with %in3
/// This pattern can enable additional canonicalizations: In the above example,
/// %in0, %in1 and %in3 have no uses anymore and their corresponding operands
/// can be folded away. This pattern does not modify uses of output block args.
struct FoldDuplicateInputBbArgs : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Find replacement bbArgs for all input bbArg.
    DenseMap<int, int> replacements;
    for (int i = 0; i < genericOp.getNumDpsInputs(); ++i) {
      // Skip bbArgs that have no uses.
      if (genericOp.getBody()->getArgument(i).getUses().empty())
        continue;
      // Find replacement bbArg. This can be an input or an output bbArg.
      for (int j = genericOp->getNumOperands() - 1; j > i; --j) {
        if (genericOp->getOperand(i) == genericOp->getOperand(j) &&
            genericOp.getIndexingMapsArray()[i] ==
                genericOp.getIndexingMapsArray()[j]) {
          replacements[i] = j;
          break;
        }
      }
    }

    // Stop here if no replacements were found.
    if (replacements.empty())
      return failure();

    // Rewrite the op.
    rewriter.updateRootInPlace(genericOp, [&]() {
      for (auto [before, after] : replacements) {
        BlockArgument bbArg = genericOp.getBody()->getArgument(before);
        BlockArgument replacement = genericOp.getBody()->getArgument(after);
        rewriter.replaceAllUsesWith(bbArg, replacement);
      }
    });

    return success();
  }
};

} // namespace

void mlir::linalg::populateEraseUnusedOperandsAndResultsPatterns(
    RewritePatternSet &patterns) {
  patterns.insert<DeduplicateAndRemoveDeadOperandsAndResults>(
      patterns.getContext(), /*removeOutputs=*/true);
  patterns.insert<RemoveUnusedCycleInGenericOp>(patterns.getContext());
}

void mlir::linalg::populateEraseUnnecessaryInputsPatterns(
    RewritePatternSet &patterns) {
  patterns.insert<DeduplicateAndRemoveDeadOperandsAndResults>(
      patterns.getContext(), /*removeOutputs=*/false);
  patterns.insert<FoldDuplicateInputBbArgs>(patterns.getContext());
}
