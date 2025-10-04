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
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "llvm/ADT/Hashing.h"

#include <sstream>

namespace mlir {
#define GEN_PASS_DEF_NORMALIZE
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "normalize"

namespace {
struct NormalizePass : public impl::NormalizeBase<NormalizePass> {
  NormalizePass() = default;

  void runOnOperation() override;

private:
  const uint64_t MagicHashConstant = 0x6acaa36bef8325c5ULL;
  void collectOutputOperations(Block &block,
                               SmallVector<Operation *, 16> &Output);
  bool isOutput(mlir::Operation &op);
  void reorderOperations(SmallVector<mlir::Operation *, 16> &Outputs);
  void
  reorderOperation(mlir::Operation *used, mlir::Operation *user,
                   llvm::SmallPtrSet<const mlir::Operation *, 32> &visited);
  void RenameOperations(SmallVector<Operation *, 16> &Outputs);
  void RenameOperation(mlir::Operation* op, SmallPtrSet<const mlir::Operation *, 32> &visited);
  bool isInitialOperation(mlir::Operation* op);
  void nameAsInitialOperation(mlir::Operation* op);
  void nameAsRegularOperation(mlir::Operation* op, llvm::SmallPtrSet<const mlir::Operation *, 32> &visited);
  bool hasOnlyImmediateOperands(mlir::Operation* op);
  void SetDeterministicNames(Block &block);
};
} // namespace

void NormalizePass::runOnOperation() {
  ModuleOp module = getOperation();

  for (Operation &op : module.getOps()) {

    // for (Region &region : op.getRegions())
    //   for (Block &block : region)
    //     SetDeterministicNames(block);

    SmallVector<Operation *, 16> Outputs;

    for (Region &region : op.getRegions())
      for (Block &block : region)
        collectOutputOperations(block, Outputs);

    reorderOperations(Outputs);

    // RenameOperations(Outputs);
  }
}

void NormalizePass::SetDeterministicNames(Block &block) {
  static size_t VarCounter = 0;

  for (Operation &innerOp : block) {
    mlir::OpBuilder b(innerOp.getContext());
    mlir::StringAttr sat =
        b.getStringAttr(llvm::formatv("v{0}", VarCounter++).str());
    mlir::Location newLoc = mlir::NameLoc::get(sat, innerOp.getLoc());
    innerOp.setLoc(newLoc);
  }
}

void NormalizePass::RenameOperations(SmallVector<Operation *, 16> &Outputs) {
  llvm::SmallPtrSet<const mlir::Operation *, 32> visited;

  for(auto *op : Outputs)
    RenameOperation(op, visited);
}

void NormalizePass::RenameOperation(Operation *op, SmallPtrSet<const mlir::Operation *, 32> &visited) {
  if (!visited.count(op)) {
    visited.insert(op);

    llvm::outs() << op->getName() << " --> ";
  
    if (isInitialOperation(op)) {
      llvm::outs() <<" INITIAL\n";
      nameAsInitialOperation(op);
    } else {
      llvm::outs() << " REGULAR\n";
      nameAsRegularOperation(op, visited);
    }
  }
}

bool NormalizePass::isInitialOperation(mlir::Operation* op) {
  return !op->use_empty() and hasOnlyImmediateOperands(op);
}

bool NormalizePass::hasOnlyImmediateOperands(mlir::Operation* op) {
  for (mlir::Value operand : op->getOperands())
    if (mlir::Operation *defOp = operand.getDefiningOp())
      if (!(defOp->hasTrait<OpTrait::ConstantLike>()))
        return false;
  return true;
}

uint64_t kernel_hash(std::string data)
{
    const uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    const uint64_t FNV_PRIME = 0x100000001b3ULL;
    uint64_t hash = FNV_OFFSET;
    for (unsigned char c : data)
    {
        hash ^= static_cast<uint64_t>(c);
        hash *= FNV_PRIME;
    }
    return hash;
}

void NormalizePass::nameAsInitialOperation(mlir::Operation* op) {
  SmallVector<SmallString<64>, 4> Operands;

  if(op->getNumOperands() == 0) {
    std::string TextRepresentation;
    mlir::AsmState state(op);
    llvm::raw_string_ostream Stream(TextRepresentation);
    op->print(Stream, state);
    Operands.push_back(StringRef(Stream.str()));
  } else {
    for (mlir::Value operand : op->getOperands()) {
      if (mlir::Operation *defOp = operand.getDefiningOp()) {
        std::string TextRepresentation;
        mlir::AsmState state(defOp);
        llvm::raw_string_ostream Stream(TextRepresentation);
        defOp->print(Stream, state);
        Operands.push_back(StringRef(Stream.str()));
      } else {
        std::string TextRepresentation;
        mlir::AsmState state(op);
        llvm::raw_string_ostream Stream(TextRepresentation);
        operand.print(Stream, state);
        Operands.push_back(StringRef(Stream.str()));
      }
    }
  }

  if (op->hasTrait<OpTrait::IsCommutative>()) llvm::sort(Operands);

  uint64_t Hash = MagicHashConstant;

  uint64_t opcodeHash = kernel_hash(op->getName().getStringRef().str());
  Hash = llvm::hashing::detail::hash_16_bytes(Hash, opcodeHash);

  SmallPtrSet<const Instruction *, 32> Visited;
  // Get output footprint for I.
  SetVector<int> OutputFootprint = getOutputFootprint(I, Visited);

  // Consider output footprint in the hash.
  for (const int &Output : OutputFootprint)
    Hash = llvm::hashing::detail::hash_16_bytes(Hash, Output);

  // Base instruction name.
  SmallString<256> Name;
  Name.append("vl" + std::to_string(Hash).substr(0, 5));

  // In case of CallInst, consider callee in the instruction name.
  if (const auto *CI = dyn_cast<CallInst>(I)) {
    Function *F = CI->getCalledFunction();

    if (F != nullptr) {
      Name.append(F->getName());
    }
  }

  Name.append("(");
  for (unsigned long i = 0; i < Operands.size(); ++i) {
    Name.append(Operands[i]);

    if (i < Operands.size() - 1)
      Name.append(", ");
  }
  Name.append(")");

  I->setName(Name);
}

void NormalizePass::nameAsRegularOperation(mlir::Operation* op, llvm::SmallPtrSet<const mlir::Operation *, 32> &visited) {

}

void NormalizePass::reorderOperations(SmallVector<Operation *, 16> &Outputs) {
  llvm::SmallPtrSet<const mlir::Operation *, 32> visited;
  for (auto *op : Outputs)
    for (mlir::Value operand : op->getOperands())
      if (mlir::Operation *defOp = operand.getDefiningOp())
        reorderOperation(defOp, op, visited);
}

void NormalizePass::reorderOperation(
    mlir::Operation *used, mlir::Operation *user,
    llvm::SmallPtrSet<const mlir::Operation *, 32> &visited) {

  if (!visited.count(used)) {
    visited.insert(used);

    mlir::Block *usedBlock = used->getBlock();
    mlir::Block *userBlock = user->getBlock();

    if (usedBlock == userBlock)
      used->moveBefore(user);
    else
      used->moveBefore(&usedBlock->back());

    for (mlir::Value operand : used->getOperands())
      if (mlir::Operation *defOp = operand.getDefiningOp())
        reorderOperation(defOp, used, visited);
  }
}

void NormalizePass::collectOutputOperations(
    Block &block, SmallVector<Operation *, 16> &Outputs) {
  llvm::SmallPtrSet<const mlir::Operation *, 32> visited;
  for (Operation &innerOp : block) {
    RenameOperation(&innerOp, visited);
    if (isOutput(innerOp))
      Outputs.emplace_back(&innerOp);
  }
}

bool NormalizePass::isOutput(Operation &op) {
  if (op.hasTrait<OpTrait::IsTerminator>())
    return true;

  if (auto memOp = dyn_cast<MemoryEffectOpInterface>(&op)) {
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    memOp.getEffects(effects);
    for (auto &effect : effects)
      if (isa<MemoryEffects::Write>(effect.getEffect()))
        return true;
  }

  return false;
}
