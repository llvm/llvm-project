// /*
// Unrolls loop in the SCF Dialect
// */

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
namespace mlir {
#define GEN_PASS_DEF_SCFLOOPUNROLL
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir
using namespace llvm;
using namespace mlir;
using namespace mlir::scf;
using scf::ForOp;

namespace {
struct LoopUnroll : public impl::SCFLoopUnrollBase<LoopUnroll> {
private:
  int unrollFactor;
  bool unrollFull;

public:
  LoopUnroll() = default;
  LoopUnroll(int unrollFactor, bool unrollFull)
      : unrollFactor(unrollFactor), unrollFull(unrollFull) {}
  void runOnOperation() override {
    Operation *parentOp = getOperation();
    OpBuilder builder(parentOp->getContext());
    SmallVector<ForOp, 4> loops;
    gatherInnermostLoops(parentOp, loops);
    if (loops.empty())
      return;
    for (auto forOp : loops) {
      auto result = runOnSCFForOp(builder, forOp);
    }
  }

  Value createUpdatedInductionVar(unsigned i, Value iv, OpBuilder &builder,
                                  int64_t step) {
    /*Function writes ir to update the induction variable*/
    Location loc = iv.getLoc();
    Value constantI = builder.create<arith::ConstantIndexOp>(loc, i);
    Value constantStep = builder.create<arith::ConstantIndexOp>(loc, step);
    Value increment =
        builder.create<arith::MulIOp>(loc, constantI, constantStep);
    Value updatedIv = builder.create<arith::AddIOp>(loc, iv, increment);
    return updatedIv;
  }

  void generateUnroll(ForOp forOp, int64_t step) {
    /*Function unrolls a given loop by the given step*/
    Block *loopBody = forOp.getBody();
    auto returnValues = forOp.getBody()->getTerminator()->getOperands();
    Value forOpInductionVar = forOp.getInductionVar();
    ValueRange iterArgs(forOp.getRegionIterArgs());
    auto builder = OpBuilder::atBlockTerminator(loopBody);
    Block::iterator originalBlockEnd = std::prev(loopBody->end(), 2);
    SmallVector<Value, 4> lastReturnValues(returnValues);
    for (int i = 1; i < unrollFactor; i++) {
      IRMapping mapper;
      mapper.map(iterArgs, lastReturnValues);
      if (!forOpInductionVar.use_empty()) {
        Value updatedInductionVar =
            createUpdatedInductionVar(i, forOpInductionVar, builder, step);
        mapper.map(forOpInductionVar, updatedInductionVar);
      }
      for (auto it = loopBody->begin(); it != std::next(originalBlockEnd);
           it++) {
        Operation *clonedOp = builder.clone(*it, mapper);
      }
      for (int j = 0; j < lastReturnValues.size(); j++) {
        Operation *defOp = returnValues[j].getDefiningOp();
        if (defOp && defOp->getBlock() == loopBody) {
          lastReturnValues[j] = mapper.lookup(returnValues[j]);
        }
      }
    }
    loopBody->getTerminator()->setOperands(lastReturnValues);
  }
  void fullUnroll(ForOp forOp, int64_t tripCount, int64_t step) {
    /*Function to fully unroll a given scf for Op*/
    IRRewriter rewriter(forOp.getContext());
    if (tripCount == 0)
      return;
    if (tripCount == 1) {
      (void)forOp.promoteIfSingleIteration(rewriter);
      return;
    }
    unrollFactor = tripCount;
    generateUnroll(forOp, step);
    return;
  }
  LogicalResult runOnSCFForOp(OpBuilder &builder, ForOp forOp) {
    /*Function process SCF ForOps*/
    IRRewriter rewriter(forOp.getContext());
    Value lowerBound = forOp.getLowerBound();
    Value upperBound = forOp.getUpperBound();
    Value step = forOp.getStep();
    Value newStepValue, upperBoundUnrolled;
    bool createHandlerLoop = false;
    auto lowerBoundConst = lowerBound.getDefiningOp<arith::ConstantIndexOp>();
    auto upperBoundConst = upperBound.getDefiningOp<arith::ConstantIndexOp>();
    auto stepConst = step.getDefiningOp<arith::ConstantIndexOp>();
    if (!lowerBoundConst || !upperBoundConst || !stepConst) {
      forOp.emitError("Expected constant bounds and step for unrolling.");
      return failure();
    }
    int64_t lowerBoundValue = lowerBoundConst.value();
    int64_t upperBoundValue = upperBoundConst.value();
    int64_t stepValue = stepConst.value();
    int64_t tripCount = (upperBoundValue - lowerBoundValue) / stepValue;
    int64_t multipliedStepValue = stepValue * unrollFactor;
    int64_t tripCountUnrolled = tripCount - (tripCount % unrollFactor);
    int64_t unrolledUpperBound =
        lowerBoundValue + (tripCountUnrolled * stepValue);
    createHandlerLoop = unrolledUpperBound < upperBoundValue;
    if (Block *prevBlock = forOp->getBlock()->getPrevNode())
      builder.setInsertionPointToEnd(prevBlock);
    else
      builder.setInsertionPoint(forOp);
    if (createHandlerLoop) {
      upperBoundUnrolled = builder.create<arith::ConstantIndexOp>(
          forOp.getLoc(), unrolledUpperBound);
      newStepValue = builder.create<arith::ConstantIndexOp>(
          forOp.getLoc(), multipliedStepValue);
      builder.setInsertionPoint(forOp);
    } else {
      upperBoundUnrolled = upperBound;
      newStepValue = step;
    }
    if (createHandlerLoop) {
      OpBuilder loopBuilder(forOp->getContext());
      loopBuilder.setInsertionPoint(forOp->getBlock(),
                                    std::next(Block::iterator(forOp)));
      auto handlerForOp = cast<scf::ForOp>(loopBuilder.clone(*forOp));
      handlerForOp.setLowerBound(upperBoundUnrolled);

      auto results = forOp.getResults();
      auto handlerResults = handlerForOp.getResults();
      for (auto element : llvm::zip(results, handlerResults)) {
        std::get<0>(element).replaceAllUsesWith(std::get<1>(element));
      }
      handlerForOp->setOperands(handlerForOp.getNumControlOperands(),
                                handlerForOp.getInitArgs().size(), results);
      (void)handlerForOp.promoteIfSingleIteration(rewriter);
    }
    forOp.setStep(upperBoundUnrolled);
    forOp.setStep(newStepValue);
    if (unrollFull == false)
      generateUnroll(forOp, stepValue);
    else
      fullUnroll(forOp, tripCount, stepValue);
    (void)forOp.promoteIfSingleIteration(rewriter);
    llvm::errs() << "Completed unroll\n";
    return success();
  }
  static bool isInnermostSCFForOp(ForOp op) {
    /*Identifies if a given for loop is the inner most scf ForOp*/
    return !op.getBody()
                ->walk(
                    [&](ForOp nestedForOp) { return WalkResult::interrupt(); })
                .wasInterrupted();
  }

  static void gatherInnermostLoops(Operation *op,
                                   SmallVectorImpl<ForOp> &loops) {
    /*Function gathers all the innermost scf for loops*/
    op->walk([&](ForOp forOp) {
      if (isInnermostSCFForOp(forOp))
        loops.push_back(forOp);
    });
  }

  virtual ~LoopUnroll() = default;
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::createLoopUnroll(int unrollFactor,
                                             bool unrollFull) {
  return std::make_unique<LoopUnroll>(unrollFactor, unrollFull);
}