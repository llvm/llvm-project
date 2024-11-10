//===- ParallelUnroll.cpp - Code to perform parallel loop unrolling
//--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements parallel loop unrolling.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINEPARALLELUNROLL
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-parallel-unroll"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Unroll an `affine.parallel` operation by the `unrollFactor` specified in the
/// attribute. Evenly splitting `memref`s that are present in the `parallel`
/// region into smaller banks.
struct ParallelUnroll
    : public affine::impl::AffineParallelUnrollBase<ParallelUnroll> {
  const std::function<unsigned(AffineParallelOp)> getUnrollFactor;
  ParallelUnroll() : getUnrollFactor(nullptr) {}
  ParallelUnroll(const ParallelUnroll &other) = default;
  explicit ParallelUnroll(std::optional<unsigned> unrollFactor = std::nullopt,
                          const std::function<unsigned(AffineParallelOp)>
                              &getUnrollFactor = nullptr)
      : getUnrollFactor(getUnrollFactor) {
    if (unrollFactor)
      this->unrollFactor = *unrollFactor;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override;
  LogicalResult parallelUnrollByFactor(AffineParallelOp parOp,
                                       uint64_t unrollFactor);

private:
  // map from original memory definition to newly allocated banks
  DenseMap<Value, SmallVector<Value>> memoryToBanks;
};
} // namespace

// Collect all memref in the `parOp`'s region'
DenseSet<Value> collectMemRefs(AffineParallelOp parOp) {
  DenseSet<Value> memrefVals;
  parOp.walk([&](Operation *op) {
    for (auto operand : op->getOperands()) {
      if (isa<MemRefType>(operand.getType()))
        memrefVals.insert(operand);
    }
    return WalkResult::advance();
  });
  return memrefVals;
}

MemRefType computeBankedMemRefType(MemRefType originalType,
                                   uint64_t bankingFactor) {
  ArrayRef<int64_t> originalShape = originalType.getShape();
  assert(!originalShape.empty() && "memref shape should not be empty");
  assert(originalType.getRank() == 1 &&
         "currently only support one dimension memories");
  SmallVector<int64_t, 4> newShape(originalShape.begin(), originalShape.end());
  assert(newShape.front() % bankingFactor == 0 &&
         "memref shape must be divided by the banking factor");
  // Now assuming banking the last dimension
  newShape.front() /= bankingFactor;
  MemRefType newMemRefType =
      MemRefType::get(newShape, originalType.getElementType(),
                      originalType.getLayout(), originalType.getMemorySpace());

  return newMemRefType;
}

SmallVector<Value> createBanks(Value originalMem, uint64_t unrollFactor) {
  MemRefType originalMemRefType = cast<MemRefType>(originalMem.getType());
  MemRefType newMemRefType =
      computeBankedMemRefType(originalMemRefType, unrollFactor);
  SmallVector<Value, 4> banks;
  if (auto blockArgMem = dyn_cast<BlockArgument>(originalMem)) {
    Block *block = blockArgMem.getOwner();
    unsigned blockArgNum = blockArgMem.getArgNumber();

    SmallVector<Type> banksType;
    for (unsigned i = 0; i < unrollFactor; ++i) {
      block->insertArgument(blockArgNum + 1 + i, newMemRefType,
                            blockArgMem.getLoc());
    }

    auto blockArgs = block->getArguments().slice(blockArgNum + 1, unrollFactor);
    banks.append(blockArgs.begin(), blockArgs.end());
  } else {
    Operation *originalDef = originalMem.getDefiningOp();
    Location loc = originalDef->getLoc();
    OpBuilder builder(originalDef);
    builder.setInsertionPointAfter(originalDef);
    TypeSwitch<Operation *>(originalDef)
        .Case<memref::AllocOp>([&](memref::AllocOp allocOp) {
          for (uint bankCnt = 0; bankCnt < unrollFactor; bankCnt++) {
            auto bankAllocOp =
                builder.create<memref::AllocOp>(loc, newMemRefType);
            banks.push_back(bankAllocOp);
          }
        })
        .Case<memref::AllocaOp>([&](memref::AllocaOp allocaOp) {
          for (uint bankCnt = 0; bankCnt < unrollFactor; bankCnt++) {
            auto bankAllocaOp =
                builder.create<memref::AllocaOp>(loc, newMemRefType);
            banks.push_back(bankAllocaOp);
          }
        })
        .Default([](Operation *) {
          llvm_unreachable("Unhandled memory operation type");
        });
  }
  return banks;
}

Value computeIntraBankingOffset(OpBuilder &builder, Location loc, Value address,
                                uint availableBanks) {
  Value availBanksVal =
      builder
          .create<arith::ConstantOp>(loc, builder.getIndexAttr(availableBanks))
          .getResult();
  Value offset =
      builder.create<arith::DivUIOp>(loc, address, availBanksVal).getResult();
  return offset;
}

struct BankAffineLoadPattern : public OpRewritePattern<AffineLoadOp> {
  BankAffineLoadPattern(MLIRContext *context, uint64_t unrollFactor,
                        DenseMap<Value, SmallVector<Value>> &memoryToBanks)
      : OpRewritePattern<AffineLoadOp>(context), unrollFactor(unrollFactor),
        memoryToBanks(memoryToBanks) {}

  LogicalResult matchAndRewrite(AffineLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "load pattern matchAndRewrite\n";
    Location loc = loadOp.getLoc();
    auto banks = memoryToBanks[loadOp.getMemref()];
    Value loadIndex = loadOp.getIndices().front();
    rewriter.setInsertionPointToStart(loadOp->getBlock());
    Value bankingFactorValue =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, unrollFactor);
    Value bankIndex = rewriter.create<mlir::arith::RemUIOp>(loc, loadIndex,
                                                            bankingFactorValue);
    Value offset =
        computeIntraBankingOffset(rewriter, loc, loadIndex, unrollFactor);

    SmallVector<Type> resultTypes = {loadOp.getResult().getType()};

    SmallVector<int64_t, 4> caseValues;
    for (unsigned i = 0; i < unrollFactor; ++i)
      caseValues.push_back(i);

    rewriter.setInsertionPoint(loadOp);
    scf::IndexSwitchOp switchOp = rewriter.create<scf::IndexSwitchOp>(
        loc, resultTypes, bankIndex, caseValues,
        /*numRegions=*/unrollFactor);

    for (unsigned i = 0; i < unrollFactor; ++i) {
      Region &caseRegion = switchOp.getCaseRegions()[i];
      rewriter.setInsertionPointToStart(&caseRegion.emplaceBlock());
      Value bankedLoad = rewriter.create<AffineLoadOp>(loc, banks[i], offset);
      rewriter.create<scf::YieldOp>(loc, bankedLoad);
    }

    Region &defaultRegion = switchOp.getDefaultRegion();
    assert(defaultRegion.empty() && "Default region should be empty");
    rewriter.setInsertionPointToStart(&defaultRegion.emplaceBlock());

    TypedAttr zeroAttr =
        cast<TypedAttr>(rewriter.getZeroAttr(loadOp.getType()));
    auto defaultValue = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    rewriter.create<scf::YieldOp>(loc, defaultValue.getResult());

    loadOp.getResult().replaceAllUsesWith(switchOp.getResult(0));

    rewriter.eraseOp(loadOp);
    return success();
  }

private:
  uint64_t unrollFactor;
  DenseMap<Value, SmallVector<Value>> &memoryToBanks;
};

struct BankAffineStorePattern : public OpRewritePattern<AffineStoreOp> {
  BankAffineStorePattern(MLIRContext *context, uint64_t unrollFactor,
                         DenseMap<Value, SmallVector<Value>> &memoryToBanks)
      : OpRewritePattern<AffineStoreOp>(context), unrollFactor(unrollFactor),
        memoryToBanks(memoryToBanks) {}

  LogicalResult matchAndRewrite(AffineStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "store pattern matchAndRewrite\n";
    Location loc = storeOp.getLoc();
    auto banks = memoryToBanks[storeOp.getMemref()];
    Value loadIndex = storeOp.getIndices().front();
    rewriter.setInsertionPointToStart(storeOp->getBlock());
    Value bankingFactorValue =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, unrollFactor);
    Value bankIndex = rewriter.create<mlir::arith::RemUIOp>(loc, loadIndex,
                                                            bankingFactorValue);
    Value offset =
        computeIntraBankingOffset(rewriter, loc, loadIndex, unrollFactor);

    SmallVector<Type> resultTypes = {};

    SmallVector<int64_t, 4> caseValues;
    for (unsigned i = 0; i < unrollFactor; ++i)
      caseValues.push_back(i);

    rewriter.setInsertionPoint(storeOp);
    scf::IndexSwitchOp switchOp = rewriter.create<scf::IndexSwitchOp>(
        loc, resultTypes, bankIndex, caseValues,
        /*numRegions=*/unrollFactor);

    for (unsigned i = 0; i < unrollFactor; ++i) {
      Region &caseRegion = switchOp.getCaseRegions()[i];
      rewriter.setInsertionPointToStart(&caseRegion.emplaceBlock());
      rewriter.create<AffineStoreOp>(loc, storeOp.getValueToStore(), banks[i],
                                     offset);
      rewriter.create<scf::YieldOp>(loc);
    }

    Region &defaultRegion = switchOp.getDefaultRegion();
    assert(defaultRegion.empty() && "Default region should be empty");
    rewriter.setInsertionPointToStart(&defaultRegion.emplaceBlock());

    rewriter.create<scf::YieldOp>(loc);

    rewriter.eraseOp(storeOp);
    return success();
  }

private:
  uint64_t unrollFactor;
  DenseMap<Value, SmallVector<Value>> &memoryToBanks;
};

struct BankReturnPattern : public OpRewritePattern<func::ReturnOp> {
  BankReturnPattern(MLIRContext *context,
                    DenseMap<Value, SmallVector<Value>> &memoryToBanks)
      : OpRewritePattern<func::ReturnOp>(context),
        memoryToBanks(memoryToBanks) {}

  LogicalResult matchAndRewrite(func::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {
    Location loc = returnOp.getLoc();
    SmallVector<Value, 4> newReturnOperands;
    bool allOrigMemsUsedByReturn = true;
    for (auto operand : returnOp.getOperands()) {
      if (!memoryToBanks.contains(operand)) {
        newReturnOperands.push_back(operand);
        continue;
      }
      if (operand.hasOneUse())
        allOrigMemsUsedByReturn = false;
      auto banks = memoryToBanks[operand];
      newReturnOperands.append(banks.begin(), banks.end());
    }
    func::FuncOp funcOp = returnOp.getParentOp();
    rewriter.setInsertionPointToEnd(&funcOp.getBlocks().front());
    auto newReturnOp =
        rewriter.create<func::ReturnOp>(loc, ValueRange(newReturnOperands));
    TypeRange newReturnType = TypeRange(newReturnOperands);
    FunctionType newFuncType =
        FunctionType::get(funcOp.getContext(),
                          funcOp.getFunctionType().getInputs(), newReturnType);
    funcOp.setType(newFuncType);

    if (allOrigMemsUsedByReturn) {
      rewriter.replaceOp(returnOp, newReturnOp);
    }
    return success();
  }

private:
  DenseMap<Value, SmallVector<Value>> &memoryToBanks;
};

void ParallelUnroll::runOnOperation() {
  if (getOperation().isExternal()) {
    return;
  }

  getOperation().walk([&](AffineParallelOp parOp) {
    DenseSet<Value> memrefsInPar = collectMemRefs(parOp);

    for (auto memrefVal : memrefsInPar) {
      SmallVector<Value> banks = createBanks(memrefVal, unrollFactor);
      memoryToBanks[memrefVal] = banks;
    }
  });

  auto *ctx = &getContext();

  RewritePatternSet patterns(ctx);

  patterns.add<BankAffineLoadPattern>(ctx, unrollFactor, memoryToBanks);
  patterns.add<BankAffineStorePattern>(ctx, unrollFactor, memoryToBanks);
  patterns.add<BankReturnPattern>(ctx, memoryToBanks);

  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                          config))) {
    signalPassFailure();
  }

  DenseSet<Block *> blocksToModify;
  for (auto &[memrefVal, banks] : memoryToBanks) {
    if (memrefVal.use_empty()) {
      if (auto blockArg = dyn_cast<BlockArgument>(memrefVal)) {
        blockArg.getOwner()->eraseArgument(blockArg.getArgNumber());
        blocksToModify.insert(blockArg.getOwner());
      } else {
        memrefVal.getDefiningOp()->erase();
      }
    }
  }

  for (auto *block : blocksToModify) {
    if (!isa<func::FuncOp>(block->getParentOp()))
      continue;
    func::FuncOp funcOp = cast<func::FuncOp>(block->getParentOp());
    SmallVector<Type, 4> newArgTypes;
    for (BlockArgument arg : funcOp.getArguments()) {
      newArgTypes.push_back(arg.getType());
    }
    FunctionType newFuncType =
        FunctionType::get(funcOp.getContext(), newArgTypes,
                          funcOp.getFunctionType().getResults());
    funcOp.setType(newFuncType);
  }

  getOperation().dump();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createParallelUnrollPass(
    int unrollFactor,
    const std::function<unsigned(AffineParallelOp)> &getUnrollFactor) {
  return std::make_unique<ParallelUnroll>(
      unrollFactor == -1 ? std::nullopt : std::optional<unsigned>(unrollFactor),
      getUnrollFactor);
}
