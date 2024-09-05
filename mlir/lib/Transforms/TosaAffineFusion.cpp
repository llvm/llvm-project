#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_TOSAAFFINEFUSION
#include "mlir/Transforms/Passes.h.inc"
}

#define DEBUG_TYPE "tosa-affine-fusion"

using namespace mlir;
using namespace mlir::affine;

namespace {
class TosaAffineFusion : public mlir::impl::TosaAffineFusionBase<TosaAffineFusion> {

public :
    TosaAffineFusion() = default;
    void runOnOperation() override;
    bool checkFusibility(AffineForOp *dstLoop, AffineForOp *srcLoop);
    void moveIntermediateOps(AffineForOp *dstLoop, AffineForOp *srcLoop);
    void fuseSiblingLoops(AffineForOp *dstLoop, AffineForOp *srcLoop);
    bool useInsideLoop(Operation *user, AffineForOp *srcLoop);
    void fuseLoopsInBlock(Block *block);
};

bool TosaAffineFusion::checkFusibility(AffineForOp *dstLoop, AffineForOp *srcLoop) {
    if (dstLoop->getOperation() == srcLoop->getOperation()) {
        llvm::errs()<<"[CHECKFUSIBILITY LOG] Same Loop\n";
        return false;
    }

    if (dstLoop->getOperation()->getParentOp() != srcLoop->getOperation()->getParentOp()) {
        llvm::errs()<<"[CHECKFUSIBILITY LOG] Parent is not same\n";
        return false;
    }

    if (dstLoop->getConstantLowerBound() != srcLoop->getConstantLowerBound()) {
        llvm::errs()<<"[CHECKFUSIBILITY LOG] Lower Bound is not same\n";
        return false;
    }

    if (dstLoop->getConstantUpperBound() != srcLoop->getConstantUpperBound()) {
        llvm::errs()<<"[CHECKFUSIBILITY LOG] Upper Bound is not same\n";
        return false;
    }

    if (dstLoop->getStepAsInt() != srcLoop->getStepAsInt()) {
        llvm::errs()<<"[CHECKFUSIBILITY LOG] Step is not same\n";
        return false;
    }

    llvm::errs()<<"[CHECKFUSIBILITY LOG] SUCCESS\n";
    return true;
}

bool TosaAffineFusion::useInsideLoop(Operation *user, AffineForOp *srcLoop) {
    while (!isa<func::FuncOp>(user->getParentOp())) {
        auto *parentOp = user->getParentOp();
        if (user->getParentOp() == srcLoop->getOperation())
            return true;
        user = parentOp;
    }
    return false;
}

void TosaAffineFusion::moveIntermediateOps(AffineForOp *dstLoop, AffineForOp *srcLoop) {
    auto *block = dstLoop->getOperation()->getBlock();
    bool dstLoopFound = false;
    for (auto &op : block->getOperations()) {
        if (&op == dstLoop->getOperation()) {
            dstLoopFound = true;
            continue;
        }
        if (!dstLoopFound)
            continue;
        if (&op == srcLoop->getOperation())
            break;
        for (auto *user : op.getUsers())
            if (useInsideLoop(user, srcLoop))
                op.moveBefore(dstLoop->getOperation());
    }
}

void TosaAffineFusion::fuseSiblingLoops(AffineForOp *dstLoop, AffineForOp *srcLoop) {
    IRMapping map;
    map.map(srcLoop->getInductionVar(), dstLoop->getInductionVar());
    OpBuilder builder(*dstLoop);
    builder.setInsertionPoint(dstLoop->getBody()->getTerminator());

    for (auto &op : srcLoop->getBody()->getOperations()) {
        if (&op == srcLoop->getBody()->getTerminator())
            continue;
        builder.clone(op, map);
    }
}

void TosaAffineFusion::fuseLoopsInBlock(Block *block) {
    auto affineFors = block->getOps<AffineForOp>();
    SmallVector<AffineForOp, 4> siblingAffineFors{affineFors.begin(), affineFors.end()};

    for (auto dstLoop : siblingAffineFors) {
        if (!dstLoop.getOperation()) {
            llvm::errs()<<"[FUSELOOPSINBLOCK LOG] 1 - DstLoop refernce dropped\n";
            continue;
        }
        llvm::errs()<<"[FUSELOOPSINBLOCK LOG] DstLoop -> \n";
        dstLoop.dump();
        if (dstLoop->getParentOp() == nullptr) {
            llvm::errs()<<"[FUSELOOPSINBLOCK LOG] 2 - DstLoop refernce dropped\n";
            continue;
        }
        for (auto srcLoop : siblingAffineFors) {
            if (!srcLoop.getOperation()) {
                llvm::errs()<<"[FUSELOOPSINBLOCK LOG] 1 - SrcLoop refernce dropped\n";
                continue;
            }
            llvm::errs()<<"[FUSELOOPSINBLOCK LOG] SrcLoop -> \n";
            srcLoop.dump();
            if (srcLoop->getParentOp() == nullptr) {
                llvm::errs()<<"[FUSELOOPSINBLOCK LOG] 2 - SrcLoop refernce dropped\n";
                continue;
            }
            if (!checkFusibility(&dstLoop, &srcLoop))
                continue;

            llvm::errs()<<"[FUSELOOPSINBLOCK LOG] DSTLoop SRCLoop FUSABLE\n";

            moveIntermediateOps(&dstLoop, &srcLoop);

            fuseSiblingLoops(&dstLoop, &srcLoop);

            srcLoop->dropAllReferences();
            srcLoop->remove();

            llvm::errs()<<"[CHECKFUSIBILITY LOG] New FUSED DSTLoop\n";
            dstLoop.dump();
        }

        for (Region &region : dstLoop->getRegions()) {
            for (Block &block : region.getBlocks()) {
                auto affineFors = block.getOps<AffineForOp>();
                if (!affineFors.empty() && !llvm::hasSingleElement(affineFors)) {
                                llvm::errs()<<"[CHECKFUSIBILITY LOG] Step is not same\n";

                    fuseLoopsInBlock(&block);
                }
            }
        }
    }
    llvm::errs()<<"[CHECKFUSIBILITY LOG] Step is not same\n";
}

void TosaAffineFusion::runOnOperation() {
    getOperation()->walk([&](Operation *op) {
        for (Region &region : op->getRegions()) {
            for (Block &block : region.getBlocks()) {
                auto affineFors = block.getOps<AffineForOp>();
                if (!affineFors.empty() && !llvm::hasSingleElement(affineFors)) {
                    fuseLoopsInBlock(&block);
                }
            }
        }
    });
}

} // end of namespace

std::unique_ptr<Pass> mlir::createTosaAffineFusionPass() {
    return std::make_unique<TosaAffineFusion>();
}