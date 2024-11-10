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
  SmallVector<Operation *, 8> opsToErase;
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

/// Unrolls a 'affine.parallel' op. Returns success if the loop was unrolled,
/// failure otherwise. The default unroll factor is 4.
LogicalResult ParallelUnroll::parallelUnrollByFactor(AffineParallelOp parOp,
                                                     uint64_t unrollFactor) {
  // 1. identify memrefs in the parallel region,
  // 2. create memory banks for each of those memories
  //   2.1 maybe result of alloc/getglobal, etc
  //   2.2 maybe block arguments
  //

  DenseSet<Value> memrefsInPar = collectMemRefs(parOp);
  Location loc = parOp.getLoc();
  OpBuilder builder(parOp);

  DenseSet<Block *> blocksToModify;
  for (auto memrefVal : memrefsInPar) {
    SmallVector<Value> banks = createBanks(memrefVal, unrollFactor);
    memoryToBanks[memrefVal] = banks;

    for (auto *user : memrefVal.getUsers()) {
      // if user is within parallel region
      TypeSwitch<Operation *>(user)
          .Case<affine::AffineLoadOp>([&](affine::AffineLoadOp loadOp) {
            Value loadIndex = loadOp.getIndices().front();
            builder.setInsertionPointToStart(parOp.getBody());
            Value bankingFactorValue =
                builder.create<mlir::arith::ConstantIndexOp>(loc, unrollFactor);
            Value bankIndex = builder.create<mlir::arith::RemUIOp>(
                loc, loadIndex, bankingFactorValue);
            Value offset = computeIntraBankingOffset(builder, loc, loadIndex,
                                                     unrollFactor);

            SmallVector<Type> resultTypes = {loadOp.getResult().getType()};

            SmallVector<int64_t, 4> caseValues;
            for (unsigned i = 0; i < unrollFactor; ++i)
              caseValues.push_back(i);

            builder.setInsertionPoint(user);
            scf::IndexSwitchOp switchOp = builder.create<scf::IndexSwitchOp>(
                loc, resultTypes, bankIndex, caseValues,
                /*numRegions=*/unrollFactor);

            for (unsigned i = 0; i < unrollFactor; ++i) {
              Region &caseRegion = switchOp.getCaseRegions()[i];
              builder.setInsertionPointToStart(&caseRegion.emplaceBlock());
              Value bankedLoad =
                  builder.create<AffineLoadOp>(loc, banks[i], offset);
              builder.create<scf::YieldOp>(loc, bankedLoad);
            }

            Region &defaultRegion = switchOp.getDefaultRegion();
            assert(defaultRegion.empty() && "Default region should be empty");
            builder.setInsertionPointToStart(&defaultRegion.emplaceBlock());

            TypedAttr zeroAttr =
                cast<TypedAttr>(builder.getZeroAttr(loadOp.getType()));
            auto defaultValue =
                builder.create<arith::ConstantOp>(loc, zeroAttr);
            builder.create<scf::YieldOp>(loc, defaultValue.getResult());

            loadOp.getResult().replaceAllUsesWith(switchOp.getResult(0));

            user->erase();
          })
          .Case<affine::AffineStoreOp>([&](affine::AffineStoreOp storeOp) {
            Value loadIndex = storeOp.getIndices().front();
            builder.setInsertionPointToStart(parOp.getBody());
            Value bankingFactorValue =
                builder.create<mlir::arith::ConstantIndexOp>(loc, unrollFactor);
            Value bankIndex = builder.create<mlir::arith::RemUIOp>(
                loc, loadIndex, bankingFactorValue);
            Value offset = computeIntraBankingOffset(builder, loc, loadIndex,
                                                     unrollFactor);

            SmallVector<Type> resultTypes = {};

            SmallVector<int64_t, 4> caseValues;
            for (unsigned i = 0; i < unrollFactor; ++i)
              caseValues.push_back(i);

            builder.setInsertionPoint(user);
            scf::IndexSwitchOp switchOp = builder.create<scf::IndexSwitchOp>(
                loc, resultTypes, bankIndex, caseValues,
                /*numRegions=*/unrollFactor);

            for (unsigned i = 0; i < unrollFactor; ++i) {
              Region &caseRegion = switchOp.getCaseRegions()[i];
              builder.setInsertionPointToStart(&caseRegion.emplaceBlock());
              builder.create<AffineStoreOp>(loc, storeOp.getValueToStore(),
                                            banks[i], offset);
              builder.create<scf::YieldOp>(loc);
            }

            Region &defaultRegion = switchOp.getDefaultRegion();
            assert(defaultRegion.empty() && "Default region should be empty");
            builder.setInsertionPointToStart(&defaultRegion.emplaceBlock());

            builder.create<scf::YieldOp>(loc);

            user->erase();
          })
          .Default([](Operation *op) {
            op->emitWarning("Unhandled operation type");
            op->dump();
          });
    }

    for (auto *user : memrefVal.getUsers()) {
      if (auto returnOp = dyn_cast<func::ReturnOp>(user)) {
        OpBuilder builder(returnOp);
        func::FuncOp funcOp = returnOp.getParentOp();
        builder.setInsertionPointToEnd(&funcOp.getBlocks().front());
        auto newReturnOp =
            builder.create<func::ReturnOp>(loc, ValueRange(banks));
        TypeRange newReturnType = TypeRange(banks);
        FunctionType newFuncType = FunctionType::get(
            funcOp.getContext(), funcOp.getFunctionType().getInputs(),
            newReturnType);
        funcOp.setType(newFuncType);
        returnOp->replaceAllUsesWith(newReturnOp);
        opsToErase.push_back(returnOp);
      }
    }

    // TODO: if use is empty, we should delete the original block args; and
    // reset function type
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

  /// - `isDefinedOutsideRegion` returns true if the given value is invariant
  /// with
  ///   respect to the given region. A common implementation might be:
  ///   `value.getParentRegion()->isProperAncestor(region)`.

  if (unrollFactor == 1) {
    // TODO: how to address "expected pattern to replace the root operation" if
    // just simply return success
    return success();
  }

  return success();
}

void ParallelUnroll::runOnOperation() {
  if (getOperation().isExternal()) {
    return;
  }

  getOperation().walk([&](AffineParallelOp parOp) {
    (void)parallelUnrollByFactor(parOp, unrollFactor);
    return WalkResult::advance();
  });
  for (auto *op : opsToErase) {
    op->erase();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createParallelUnrollPass(
    int unrollFactor,
    const std::function<unsigned(AffineParallelOp)> &getUnrollFactor) {
  return std::make_unique<ParallelUnroll>(
      unrollFactor == -1 ? std::nullopt : std::optional<unsigned>(unrollFactor),
      getUnrollFactor);
}
