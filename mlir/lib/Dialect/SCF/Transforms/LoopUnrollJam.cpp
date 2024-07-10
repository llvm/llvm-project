/*
File containing pass for loop unroll and jam transformation
 on scf dialect forOp
*/

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "../lib/Analysis/SliceAnalysis.cpp"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include <map>
namespace mlir {
#define GEN_PASS_DEF_SCFLOOPUNROLLJAM
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir
using namespace llvm;
using namespace mlir;
using namespace mlir::scf;
using namespace mlir::affine;
using scf::ForOp;

namespace {
struct ForBlockGatherer {
  SmallVector<std::pair<Block::iterator, Block::iterator>> subBlocks;
  void walk(Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        walk(block);
  }
  void walk(Block &block) {
    assert(!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>() &&
           "expected block to have a terminator");
    for (Block::iterator it = block.begin(), e = std::prev(block.end());
         it != e;) {
      Block::iterator subBlockStart = it;
      while (it != e && !isa<ForOp>(&*it))
        ++it;
      if (it != subBlockStart)
        subBlocks.emplace_back(subBlockStart, std::prev(it));
      while (it != e && isa<ForOp>(&*it))
        walk(&*it++);
    }
  }
};
struct LoopUnrollJam : public impl::SCFLoopUnrollJamBase<LoopUnrollJam> {
  explicit LoopUnrollJam(
      std::optional<unsigned> unrollJamFactor = std::nullopt) {
    if (unrollJamFactor)
      this->unrollJamFactor = *unrollJamFactor;
  }
  void runOnOperation() override {
    std::map<ForOp, int> outerLoopMap;
    std::map<ForOp, int> innerLoopMap;
    auto *op = getOperation();
    op->walk([&](ForOp forOp) {
      outerLoopMap[forOp] = 0;
      innerLoopMap[forOp] = 0;
    });
    op->walk(
        [&](ForOp forOp) { loopGatherer(forOp, outerLoopMap, innerLoopMap); });
    for (auto ele : innerLoopMap) {
      if (ele.second == 0) {
        (void)loopUnrollJamByFactor(ele.first);
      }
    }
  }

  // Obtains trip count for a given for op
  std::optional<uint64_t> getTripCount(scf::ForOp forOp) {
    auto lowerBoundConst =
        forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto upperBoundConst =
        forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    auto stepConst = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();

    if (!lowerBoundConst || !upperBoundConst || !stepConst) {
      return std::nullopt;
    }

    int64_t lowerBoundValue = lowerBoundConst.value();
    int64_t upperBoundValue = upperBoundConst.value();
    int64_t stepValue = stepConst.value();
    return (upperBoundValue - lowerBoundValue) / stepValue;
  }
  // Maps the nesting levels of the loops
  void loopNestMapper(ForOp op, std::map<ForOp, int> &outerLoopMap,
                      std::map<ForOp, int> &innerLoopMap) {
    op.getBody()->walk([&](ForOp nestedForOp) {
      outerLoopMap[op] += 1;
      innerLoopMap[nestedForOp] += 1;
    });
  }

  // Gathers loops to map their nesting levels
  void loopGatherer(Operation *op, std::map<ForOp, int> &outerLoopMap,
                    std::map<ForOp, int> &innerLoopMap) {
    op->walk([&](ForOp forOp) {
      loopNestMapper(forOp, outerLoopMap, innerLoopMap);
    });
  }

  // duplicates loop argument for unrolling and jamming
  void duplicateArgs(SmallVector<IRMapping> &mapper,
                     SmallVector<ForOp> &newInnerLoops, ForOp &forOp,
                     IRRewriter &rewriter, const SmallVector<ForOp> &innerLoops,
                     unsigned unrollJamFactor) {
    for (scf::ForOp currentForOp : innerLoops) {
      SmallVector<Value> iterOperandsList, yeildOperandsList;
      ValueRange previousIterOperands = currentForOp.getInits();
      ValueRange previousIterArgs = currentForOp.getRegionIterArgs();
      ValueRange previousYeildOperands =
          cast<YieldOp>(currentForOp.getBody()->getTerminator()).getOperands();
      for (unsigned i = unrollJamFactor - 1; i >= 1; --i) {
        iterOperandsList.append(previousIterOperands.begin(),
                                previousIterOperands.end());
        yeildOperandsList.append(previousYeildOperands.begin(),
                                 previousYeildOperands.end());
      }
      bool forOpReplaced = currentForOp == forOp;
      scf::ForOp newForOp =
          cast<scf::ForOp>(*currentForOp.replaceWithAdditionalYields(
              rewriter, iterOperandsList,
              /*replaceInitOperandUsesInLoop=*/false,
              [&](OpBuilder &b, Location loc,
                  ArrayRef<BlockArgument> newBbArgs) {
                return yeildOperandsList;
              }));
      newInnerLoops.push_back(newForOp);

      if (forOpReplaced)
        forOp = newForOp;
      ValueRange newIterArgs = newForOp.getRegionIterArgs();
      unsigned oldNumIterArgs = previousIterArgs.size();
      ValueRange newResults = newForOp.getResults();
      unsigned oldNumResults = newResults.size() / unrollJamFactor;
      for (unsigned i = unrollJamFactor - 1; i >= 1; --i) {
        for (unsigned j = 0; j < oldNumIterArgs; ++j) {
          mapper[i - 1].map(newIterArgs[j],
                            newIterArgs[i * oldNumIterArgs + j]);
          mapper[i - 1].map(newResults[j], newResults[i * oldNumResults + j]);
        }
      }
    }
  }
  // creates an updated induction variable
  Value createUpdatedInductionVar(unsigned i, Value iv, OpBuilder &builder,
                                  Value step) {
    Location loc = iv.getLoc();
    Value constantI = builder.create<arith::ConstantIndexOp>(loc, i);
    Value increment = builder.create<arith::MulIOp>(loc, constantI, step);
    Value updatedIv = builder.create<arith::AddIOp>(loc, iv, increment);
    return updatedIv;
  }
  // updates the step of a given loop
  void updateStep(ForOp forOp, IRRewriter &rewriter) {
    if (Block *prevBlock = forOp->getBlock()->getPrevNode())
      rewriter.setInsertionPointToEnd(prevBlock);
    else
      rewriter.setInsertionPoint(forOp);
    auto newStep = rewriter.createOrFold<arith::MulIOp>(
        forOp.getLoc(), forOp.getStep(),
        rewriter.createOrFold<arith::ConstantOp>(
            forOp.getLoc(), rewriter.getIndexAttr(unrollJamFactor)));
    forOp.setStep(newStep);
  }

  // performs the final unroll and jam operations, cloning and updating the
  // subblocks
  void finalUnrollJam(
      Value forOpInductionVar,
      SmallVector<std::pair<Block::iterator, Block::iterator>> &subBlocks,
      SmallVector<IRMapping> &mapper, const SmallVector<ForOp> &newInnerLoops,
      ForOp forOp) {
    for (unsigned i = unrollJamFactor - 1; i >= 1; --i) {
      for (auto &subBlock : subBlocks) {
        OpBuilder builder(subBlock.first->getBlock(),
                          std::next(subBlock.second));
        if (!forOpInductionVar.use_empty()) {
          auto updatedInductionVar = createUpdatedInductionVar(
              i, forOpInductionVar, builder, forOp.getStep());
          mapper[i - 1].map(forOpInductionVar, updatedInductionVar);
        }
        for (auto it = subBlock.first; it != std::next(subBlock.second); ++it)
          builder.clone(*it, mapper[i - 1]);
      }
      for (auto newForOp : newInnerLoops) {
        unsigned oldNumIterOperands =
            newForOp.getNumRegionIterArgs() / unrollJamFactor;
        unsigned numControlOperands = newForOp.getNumControlOperands();
        auto yieldOp = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
        unsigned oldNumYieldOperands =
            yieldOp.getNumOperands() / unrollJamFactor;
        for (unsigned j = 0; j < oldNumIterOperands; ++j) {
          newForOp.setOperand(numControlOperands + i * oldNumIterOperands + j,
                              mapper[i - 1].lookupOrDefault(
                                  newForOp.getOperand(numControlOperands + j)));
          yieldOp.setOperand(
              i * oldNumYieldOperands + j,
              mapper[i - 1].lookupOrDefault(yieldOp.getOperand(j)));
        }
      }
    }
  }

  // Gathers reduction operations in the loop
  void getReductions(ForOp forOp, SmallVectorImpl<LoopReduction> &reductions) {
    ValueRange iterArgs = forOp.getRegionIterArgs();
    unsigned numIterArgs = iterArgs.size();

    if (numIterArgs == 0) {
      return;
    }
    reductions.reserve(numIterArgs);
    for (unsigned i = 0; i < numIterArgs; ++i) {
      arith::AtomicRMWKind kind;
      if (Value value = getReduction(forOp, i, kind)) {
        reductions.emplace_back(LoopReduction{kind, i, value});
      }
    }
  }

  // Matches the reduction operation within the loop
  Value getReduction(ForOp forOp, unsigned pos, arith::AtomicRMWKind &kind) {
    SmallVector<Operation *> combinerOps;
    Value reducedVal =
        matchReduction(forOp.getRegionIterArgs(), pos, combinerOps);
    if (!reducedVal)
      return nullptr;

    if (combinerOps.size() > 1)
      return nullptr;

    Operation *combinerOp = combinerOps.back();
    std::optional<arith::AtomicRMWKind> maybeKind =
        mlir::TypeSwitch<Operation *, std::optional<arith::AtomicRMWKind>>(
            combinerOp)
            .Case([](arith::AddFOp) { return arith::AtomicRMWKind::addf; })
            .Case([](arith::MulFOp) { return arith::AtomicRMWKind::mulf; })
            .Case([](arith::AddIOp) { return arith::AtomicRMWKind::addi; })
            .Case([](arith::AndIOp) { return arith::AtomicRMWKind::andi; })
            .Case([](arith::OrIOp) { return arith::AtomicRMWKind::ori; })
            .Case([](arith::MulIOp) { return arith::AtomicRMWKind::muli; })
            .Default([](Operation *) -> std::optional<arith::AtomicRMWKind> {
              return std::nullopt;
            });
    if (!maybeKind)
      return nullptr;

    kind = *maybeKind;
    return reducedVal;
  }

  // Updates reduction operations after unrolling
  void updateReductions(ForOp forOp, IRRewriter &rewriter,
                        SmallVector<LoopReduction> &reductions) {
    rewriter.setInsertionPointAfter(forOp);
    auto loc = forOp.getLoc();
    unsigned oldNumResults = forOp.getNumResults() / unrollJamFactor;
    for (LoopReduction &reduction : reductions) {
      unsigned pos = reduction.iterArgPosition;
      Value lhs = forOp.getResult(pos);
      Value rhs;
      SmallPtrSet<Operation *, 4> newOps;
      for (unsigned i = unrollJamFactor - 1; i >= 1; --i) {
        rhs = forOp.getResult(i * oldNumResults + pos);
        lhs = arith::getReductionOp(reduction.kind, rewriter, loc, lhs, rhs);
        if (!lhs)
          return;
        Operation *op = lhs.getDefiningOp();
        newOps.insert(op);
      }
      forOp.getResult(pos).replaceAllUsesExcept(lhs, newOps);
    }
  }

  // unrolls and jams a loop by a given factor
  LogicalResult loopUnrollJamByFactor(ForOp forOp) {
    if (unrollJamFactor == 1)
      return success();
    std::optional<uint64_t> tripCount = getTripCount(forOp);
    if (unrollJamFactor > *tripCount)
      unrollJamFactor = *tripCount;
    else if (*tripCount % unrollJamFactor != 0)
      return failure();
    if (llvm::hasSingleElement(forOp.getBody()->getOperations()))
      return success();

    auto bg = ForBlockGatherer();
    bg.walk(forOp);
    auto &subBlocks = bg.subBlocks;

    SmallVector<ForOp> innerLoops;
    forOp.walk([&](ForOp innerForOp) { innerLoops.push_back(innerForOp); });

    SmallVector<LoopReduction> reductions;
    ValueRange iterArgs = forOp.getRegionIterArgs();
    unsigned numIterOperands = iterArgs.size();
    if (numIterOperands > 0)
      getReductions(forOp, reductions);

    SmallVector<IRMapping> mapper(unrollJamFactor - 1);
    IRRewriter rewriter(forOp.getContext());
    SmallVector<ForOp> newInnerLoops;
    duplicateArgs(mapper, newInnerLoops, forOp, rewriter, innerLoops,
                  unrollJamFactor);
    updateStep(forOp, rewriter);
    auto forOpInductionVar = forOp.getInductionVar();
    finalUnrollJam(forOpInductionVar, subBlocks, mapper, newInnerLoops, forOp);
    if (forOp.getNumResults() > 0) {
      updateReductions(forOp, rewriter, reductions);
    }
    (void)forOp.promoteIfSingleIteration(rewriter);
    return success();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createLoopUnrollJam(int unrollJamFactor) {
  return std::make_unique<LoopUnrollJam>(
      unrollJamFactor == -1 ? std::nullopt
                            : std::optional<unsigned>(unrollJamFactor));
}