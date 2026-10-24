//===- LoopCarriedComputationReuse.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements reuse of a pure loop-body computation whose affine
// accesses are translated by one iteration.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPCARRIEDCOMPUTATIONREUSE
#include "mlir/Dialect/Affine/Transforms/Passes.h.inc"
} // namespace affine
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

namespace {

constexpr unsigned maxComputationDepth = 256;

struct ReuseCandidate {
  Value earlierRoot;
  Value laterRoot;
  SmallVector<Operation *> earlierOps;
  SmallVector<Value> sources;
};

static void canonicalizeAccess(AffineLoadOp load, AffineMap &map,
                               SmallVectorImpl<Value> &operands) {
  map = load.getAffineMap();
  llvm::append_range(operands, load.getMapOperands());
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);
}

/// Return whether evaluating `earlier` one loop step later accesses exactly
/// the location accessed by `later` in the current iteration.
static bool areLoadsOneIterationApart(AffineLoadOp earlier, AffineLoadOp later,
                                      AffineForOp loop, bool &isTranslated) {
  if (earlier.getMemRef() != later.getMemRef() ||
      earlier.getType() != later.getType())
    return false;

  AffineMap earlierMap, laterMap;
  SmallVector<Value> earlierOperands, laterOperands;
  canonicalizeAccess(earlier, earlierMap, earlierOperands);
  canonicalizeAccess(later, laterMap, laterOperands);

  auto hasUnsupportedLoopLocalOperand = [&](ArrayRef<Value> operands) {
    return llvm::any_of(operands, [&](Value operand) {
      return operand != loop.getInductionVar() &&
             !loop.isDefinedOutsideOfLoop(operand);
    });
  };
  if (hasUnsupportedLoopLocalOperand(earlierOperands) ||
      hasUnsupportedLoopLocalOperand(laterOperands))
    return false;

  MLIRContext *context = loop.getContext();
  SmallVector<AffineExpr> dimReplacements;
  SmallVector<AffineExpr> symbolReplacements;
  for (unsigned i = 0; i < earlierMap.getNumDims(); ++i)
    dimReplacements.push_back(getAffineDimExpr(i, context));
  for (unsigned i = 0; i < earlierMap.getNumSymbols(); ++i)
    symbolReplacements.push_back(getAffineSymbolExpr(i, context));

  for (auto [index, operand] : llvm::enumerate(earlierOperands)) {
    if (operand != loop.getInductionVar())
      continue;
    isTranslated = true;
    if (index < earlierMap.getNumDims())
      dimReplacements[index] = dimReplacements[index] + loop.getStepAsInt();
    else
      symbolReplacements[index - earlierMap.getNumDims()] =
          symbolReplacements[index - earlierMap.getNumDims()] +
          loop.getStepAsInt();
  }

  AffineMap shiftedEarlierMap = earlierMap.replaceDimsAndSymbols(
      dimReplacements, symbolReplacements, earlierMap.getNumDims(),
      earlierMap.getNumSymbols());
  return AffineValueMap(shiftedEarlierMap, earlierOperands) ==
         AffineValueMap(laterMap, laterOperands);
}

/// Match two side-effect-free single-result DAGs. The only varying leaves are
/// affine loads separated by one loop iteration. Equal values must be defined
/// outside the loop, so an existing iter_arg cannot become a reusable context.
class ShiftedDAGMatcher {
public:
  explicit ShiftedDAGMatcher(AffineForOp loop) : loop(loop) {}

  LogicalResult match(Value earlier, Value later) {
    return matchImpl(earlier, later, /*depth=*/0);
  }

  SmallVector<Operation *> takeEarlierOps() { return std::move(earlierOps); }

  ArrayRef<Operation *> getLaterOps() const { return laterOps; }

  bool hasTranslation() const { return hasTranslatedLoad; }

  SmallVector<Value> takeSources() {
    return SmallVector<Value>(sources.begin(), sources.end());
  }

private:
  LogicalResult matchImpl(Value earlier, Value later, unsigned depth) {
    if (depth > maxComputationDepth)
      return failure();
    if (earlier == later)
      return success(loop.isDefinedOutsideOfLoop(earlier));
    if (earlier.getType() != later.getType())
      return failure();

    auto knownEarlier = earlierToLater.find(earlier);
    if (knownEarlier != earlierToLater.end())
      return success(knownEarlier->second == later);
    auto knownLater = laterToEarlier.find(later);
    if (knownLater != laterToEarlier.end())
      return success(knownLater->second == earlier);

    auto earlierLoad = earlier.getDefiningOp<AffineLoadOp>();
    auto laterLoad = later.getDefiningOp<AffineLoadOp>();
    if (earlierLoad || laterLoad) {
      if (!earlierLoad || !laterLoad || earlierLoad->getParentOp() != loop ||
          laterLoad->getParentOp() != loop ||
          !loop.isDefinedOutsideOfLoop(earlierLoad.getMemRef()) ||
          !areLoadsOneIterationApart(earlierLoad, laterLoad, loop,
                                     hasTranslatedLoad))
        return failure();
      for (Value operand : earlierLoad.getMapOperands())
        if (failed(recordAffineApplyDependencies(operand, depth + 1)))
          return failure();
      mapValues(earlier, later);
      record(earlierLoad, laterLoad);
      sources.insert(earlierLoad.getMemRef());
      return success();
    }

    Operation *earlierOp = earlier.getDefiningOp();
    Operation *laterOp = later.getDefiningOp();
    if (!earlierOp || !laterOp || earlierOp == laterOp ||
        earlierOp->getParentOp() != loop || laterOp->getParentOp() != loop ||
        earlierOp->getNumResults() != 1 || laterOp->getNumResults() != 1 ||
        earlierOp->getNumRegions() != 0 || laterOp->getNumRegions() != 0 ||
        !isMemoryEffectFree(earlierOp) || !isMemoryEffectFree(laterOp) ||
        !isSpeculatable(earlierOp) || !isSpeculatable(laterOp))
      return failure();

    mapValues(earlier, later);
    auto flags = static_cast<OperationEquivalence::Flags>(
        OperationEquivalence::IgnoreLocations |
        OperationEquivalence::IgnoreDiscardableAttrs |
        OperationEquivalence::IgnoreCommutativity);
    if (!OperationEquivalence::isEquivalentTo(
            earlierOp, laterOp,
            [&](Value earlierOperand, Value laterOperand) {
              return matchImpl(earlierOperand, laterOperand, depth + 1);
            },
            /*markEquivalent=*/nullptr, flags))
      return failure();

    record(earlierOp, laterOp);
    return success();
  }

  LogicalResult recordAffineApplyDependencies(Value value, unsigned depth) {
    if (depth > maxComputationDepth)
      return failure();
    if (value == loop.getInductionVar() || loop.isDefinedOutsideOfLoop(value))
      return success();
    auto apply = value.getDefiningOp<AffineApplyOp>();
    if (!apply || apply->getParentOp() != loop)
      return failure();
    if (seenEarlier.contains(apply))
      return success();
    for (Value operand : apply.getMapOperands())
      if (failed(recordAffineApplyDependencies(operand, depth + 1)))
        return failure();
    if (seenEarlier.insert(apply).second)
      earlierOps.push_back(apply);
    return success();
  }

  void mapValues(Value earlier, Value later) {
    earlierToLater.try_emplace(earlier, later);
    laterToEarlier.try_emplace(later, earlier);
  }

  void record(Operation *earlier, Operation *later) {
    if (seenEarlier.insert(earlier).second)
      earlierOps.push_back(earlier);
    if (seenLater.insert(later).second)
      laterOps.push_back(later);
  }

  AffineForOp loop;
  DenseMap<Value, Value> earlierToLater;
  DenseMap<Value, Value> laterToEarlier;
  llvm::SmallPtrSet<Operation *, 16> seenEarlier;
  llvm::SmallPtrSet<Operation *, 16> seenLater;
  llvm::SmallSetVector<Value, 4> sources;
  SmallVector<Operation *> earlierOps;
  SmallVector<Operation *> laterOps;
  bool hasTranslatedLoad = false;
};

static bool isSourceStable(AffineForOp loop, Value source,
                           AliasAnalysis &aliasAnalysis) {
  bool stable = true;
  (void)loop.getBody()->walk<WalkOrder::PostOrder>([&](Operation *operation) {
    // Recursive-effect operations derive their effects from nested operations
    // unless they also expose direct effects. The post-order walk has already
    // checked those nested operations.
    if (operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
        !isa<MemoryEffectOpInterface>(operation))
      return WalkResult::advance();
    if (aliasAnalysis.getModRef(operation, source).isMod()) {
      stable = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return stable;
}

/// Return whether the loop is proven to execute at least twice. Handle
/// constant bounds directly to avoid overflowing a signed bound difference.
/// Reject a negative symbolic result returned in an APInt as well.
static bool hasAtLeastTwoIterations(AffineForOp loop) {
  if (loop.hasConstantBounds()) {
    int64_t lowerBound = loop.getConstantLowerBound();
    int64_t upperBound = loop.getConstantUpperBound();
    if (upperBound <= lowerBound)
      return false;
    uint64_t span =
        static_cast<uint64_t>(upperBound) - static_cast<uint64_t>(lowerBound);
    return span > static_cast<uint64_t>(loop.getStepAsInt());
  }

  std::optional<APInt> tripCount = loop.getStaticTripCount();
  return tripCount && !tripCount->isNegative() && tripCount->ugt(1);
}

/// Return true only when the loop executes at least twice, every source is
/// stable, and moving `prologueOps` before the loop does not cross a blocking
/// operation in the first iteration.
static bool isSafeToPreload(AffineForOp loop, ValueRange sources,
                            ArrayRef<Operation *> prologueOps,
                            AliasAnalysis &aliasAnalysis) {
  if (loop.getLowerBoundMap().getNumResults() != 1 ||
      !hasAtLeastTwoIterations(loop) || sources.empty() || prologueOps.empty())
    return false;
  if (llvm::any_of(sources, [&](Value source) {
        return !isSourceStable(loop, source, aliasAnalysis);
      }))
    return false;

  Operation *root = prologueOps.back();
  if (!root || root->getParentOp() != loop)
    return false;
  llvm::SmallPtrSet<Operation *, 16> prologueSet(prologueOps.begin(),
                                                 prologueOps.end());
  for (Operation &operation : loop.getBody()->without_terminator()) {
    bool reachedRoot = &operation == root;
    if (!prologueSet.contains(&operation) &&
        !isa<AffineReadOpInterface, AffineWriteOpInterface>(operation) &&
        !isPure(&operation))
      return false;
    if (reachedRoot)
      return true;
  }
  return false;
}

static std::optional<ReuseCandidate>
findReuseCandidate(AffineForOp loop, AliasAnalysis &aliasAnalysis) {
  SmallVector<Operation *> bodyOps;
  for (Operation &operation : loop.getBody()->without_terminator())
    bodyOps.push_back(&operation);

  // Search consumers backwards so that a whole translated computation is
  // chosen before a translated subexpression inside that computation.
  for (Operation *consumer : llvm::reverse(bodyOps)) {
    for (unsigned laterIndex = 0; laterIndex < consumer->getNumOperands();
         ++laterIndex) {
      for (unsigned earlierIndex = 0; earlierIndex < consumer->getNumOperands();
           ++earlierIndex) {
        if (earlierIndex == laterIndex)
          continue;
        Value earlierRoot = consumer->getOperand(earlierIndex);
        Value laterRoot = consumer->getOperand(laterIndex);
        if (!earlierRoot.getDefiningOp() || !laterRoot.getDefiningOp() ||
            isa<AffineLoadOp>(earlierRoot.getDefiningOp()))
          continue;

        ShiftedDAGMatcher matcher(loop);
        if (failed(matcher.match(earlierRoot, laterRoot)) ||
            !matcher.hasTranslation())
          continue;
        if (llvm::is_contained(matcher.getLaterOps(),
                               earlierRoot.getDefiningOp()))
          continue;

        SmallVector<Operation *> earlierOps = matcher.takeEarlierOps();
        SmallVector<Value> sources = matcher.takeSources();
        if (earlierOps.empty() || sources.empty() ||
            !isSafeToPreload(loop, sources, earlierOps, aliasAnalysis))
          continue;

        llvm::SmallPtrSet<Operation *, 16> earlierSet(earlierOps.begin(),
                                                      earlierOps.end());
        if (llvm::any_of(earlierOps, [&](Operation *operation) {
              return llvm::any_of(operation->getOperands(), [&](Value operand) {
                if (operand == loop.getInductionVar() ||
                    loop.isDefinedOutsideOfLoop(operand))
                  return false;
                Operation *definingOp = operand.getDefiningOp();
                return !definingOp || !earlierSet.contains(definingOp);
              });
            }))
          continue;

        return ReuseCandidate{earlierRoot, laterRoot, std::move(earlierOps),
                              std::move(sources)};
      }
    }
  }
  return std::nullopt;
}

static LogicalResult materializeReuse(IRRewriter &rewriter, AffineForOp loop,
                                      ReuseCandidate candidate) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loop);

  Value lowerBound;
  if (loop.hasConstantLowerBound())
    lowerBound = arith::ConstantIndexOp::create(rewriter, loop.getLoc(),
                                                loop.getConstantLowerBound());
  else
    lowerBound =
        AffineApplyOp::create(rewriter, loop.getLoc(), loop.getLowerBoundMap(),
                              loop.getLowerBoundOperands());

  IRMapping mapping;
  mapping.map(loop.getInductionVar(), lowerBound);
  SmallVector<Operation *> clonedOps;
  clonedOps.reserve(candidate.earlierOps.size());
  for (Operation *operation : candidate.earlierOps)
    clonedOps.push_back(rewriter.clone(*operation, mapping));
  Value initial = mapping.lookup(candidate.earlierRoot);

  BlockArgument carried;
  FailureOr<LoopLikeOpInterface> replacement = loop.replaceWithAdditionalYields(
      rewriter, initial, /*replaceInitOperandUsesInLoop=*/false,
      [&](OpBuilder &, Location, ArrayRef<BlockArgument> newArguments) {
        carried = newArguments.front();
        return SmallVector<Value>{candidate.laterRoot};
      });
  if (failed(replacement)) {
    for (Operation *operation : llvm::reverse(clonedOps))
      rewriter.eraseOp(operation);
    if (lowerBound.use_empty())
      rewriter.eraseOp(lowerBound.getDefiningOp());
    return failure();
  }

  candidate.earlierRoot.replaceAllUsesWith(carried);
  for (Operation *operation : llvm::reverse(candidate.earlierOps))
    if (isOpTriviallyDead(operation))
      rewriter.eraseOp(operation);
  return success();
}

struct AffineLoopCarriedComputationReuse
    : public affine::impl::AffineLoopCarriedComputationReuseBase<
          AffineLoopCarriedComputationReuse> {
  void runOnOperation() override {
    AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
    SmallVector<AffineForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>(
        [&](AffineForOp loop) { loops.push_back(loop); });

    IRRewriter rewriter(&getContext());
    for (AffineForOp loop : loops) {
      std::optional<ReuseCandidate> candidate =
          findReuseCandidate(loop, aliasAnalysis);
      if (candidate &&
          failed(materializeReuse(rewriter, loop, std::move(*candidate)))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineLoopCarriedComputationReusePass() {
  return std::make_unique<AffineLoopCarriedComputationReuse>();
}
