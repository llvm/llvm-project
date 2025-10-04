//===- Normalize.cpp - IR to simplified IR conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Normalize/Normalize.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
#define GEN_PASS_DEF_NORMALIZE
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "normalize"

namespace {
struct NormalizePass: public impl::NormalizeBase<NormalizePass> {
    NormalizePass() = default;

    void runOnOperation() override;
private:
    const uint64_t MagicHashConstant = 0x6acaa36bef8325c5ULL;
    void collectOutputOperations(Block &block, SmallVector<Operation *, 16>& Output);
    bool isOutput(Operation& op);
    void reorderOperations(SmallVector<Operation*, 16>& Outputs);
    void reorderOperation(
        mlir::Operation *used,
        mlir::Operation *user,
        llvm::SmallPtrSet<const mlir::Operation *, 32> &visited);
    void RenameOperations(Block &block);

};
} // namespace

void NormalizePass::runOnOperation() {
    ModuleOp module = getOperation();

    for (Operation &op : module.getOps()) {
        SmallVector<Operation *, 16> Outputs;

        for (Region &region : op.getRegions()) {
            for (Block &block : region) {
                collectOutputOperations(block, Outputs);
            }
        }

        reorderOperations(Outputs);
    }

    for (Operation &op : module.getOps()) {
        SmallVector<Operation *, 16> Outputs;

        for (Region &region : op.getRegions()) {
            for (Block &block : region) {
              RenameOperations(block);
            }
        }
    }
}

void NormalizePass::RenameOperations(Block &block) {
  static size_t VarCounter = 0;

  for (Operation &innerOp : block) {
    mlir::OpBuilder  b(innerOp.getContext());
    mlir::StringAttr sat = b.getStringAttr(llvm::formatv("v{0}", VarCounter++).str());
    mlir::Location newLoc = mlir::NameLoc::get(sat, innerOp.getLoc());
    innerOp.setLoc(newLoc);
  }
}

void NormalizePass::reorderOperations(SmallVector<Operation*, 16>& Outputs) {
  llvm::SmallPtrSet<const mlir::Operation *, 32> visited;
  for (auto *op : Outputs) {
    for (mlir::Value operand : op->getOperands()) {
      if (mlir::Operation *defOp = operand.getDefiningOp()) {
        reorderOperation(defOp, op, visited);
      }
    }
  }
}

void NormalizePass::reorderOperation(
    mlir::Operation *used, mlir::Operation *user,
    llvm::SmallPtrSet<const mlir::Operation *, 32> &visited) {

  if (!visited.count(used)) {
    visited.insert(used);

    mlir::Block *usedBlock = used->getBlock();
    mlir::Block *userBlock = user->getBlock();

    if (usedBlock == userBlock) {
      used->moveBefore(user);
    } else {
      used->moveBefore(&usedBlock->back());
    }

    for (mlir::Value operand : used->getOperands()) {
      if (mlir::Operation *defOp = operand.getDefiningOp()) {
        reorderOperation(defOp, used, visited);
      }
    }
  }
}

void NormalizePass::collectOutputOperations(Block &block, SmallVector<Operation*, 16>& Outputs) {
  for(Operation& innerOp : block) {
    if(isOutput(innerOp)) {
        Outputs.emplace_back(&innerOp);
    }
  }
}

bool NormalizePass::isOutput(Operation& op) {
    if (op.hasTrait<OpTrait::IsTerminator>())
        return true;

    if (auto memOp = dyn_cast<MemoryEffectOpInterface>(&op)) {
        SmallVector<MemoryEffects::EffectInstance, 4> effects;
        memOp.getEffects(effects);
        for (auto &effect : effects) {
            if (isa<MemoryEffects::Write>( effect.getEffect() )) return true;
        }
    }

    return false;
}
