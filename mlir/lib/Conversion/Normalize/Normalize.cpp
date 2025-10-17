//===- Normalize.cpp - Conversion from MLIR to its canonical form ---------===//
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
#include "mlir/IR/AsmState.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <iomanip>
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
  void
  collectOutputOperations(Block &block,
                          SmallVector<Operation *, 16> &Output) const noexcept;
  bool isOutput(Operation &op) const noexcept;
  void reorderOperations(const SmallVector<Operation *, 16> &Outputs);
  void reorderOperation(Operation *used, Operation *user,
                        llvm::SmallPtrSet<const Operation *, 32> &visited);
  void renameOperations(const SmallVector<Operation *, 16> &Outputs);
  void RenameOperation(Operation *op,
                       SmallPtrSet<const Operation *, 32> &visited);
  bool isInitialOperation(Operation *const op) const noexcept;
  void
  nameAsInitialOperation(Operation *op,
                         llvm::SmallPtrSet<const Operation *, 32> &visited);
  void
  nameAsRegularOperation(Operation *op,
                         llvm::SmallPtrSet<const Operation *, 32> &visited);
  bool hasOnlyImmediateOperands(Operation *const op) const noexcept;
  llvm::SetVector<int>
  getOutputFootprint(Operation *op,
                     llvm::SmallPtrSet<const Operation *, 32> &visited) const;
  void foldOperation(Operation *op);
  void reorderOperationOperandsByName(Operation *op);
  OpPrintingFlags flags{};
};
} // namespace

void NormalizePass::runOnOperation() {
  flags.printNameLocAsPrefix(true);

  ModuleOp module = getOperation();

  for (auto &op : module.getOps()) {
    SmallVector<Operation *, 16> Outputs;

    for (auto &region : op.getRegions())
      for (auto &block : region)
        collectOutputOperations(block, Outputs);

    reorderOperations(Outputs);
    renameOperations(Outputs);
  }
}

void NormalizePass::renameOperations(
    const SmallVector<Operation *, 16> &Outputs) {
  llvm::SmallPtrSet<const Operation *, 32> visited;

  for (auto *op : Outputs)
    RenameOperation(op, visited);
}

void NormalizePass::RenameOperation(
    Operation *op, SmallPtrSet<const Operation *, 32> &visited) {
  if (!visited.count(op)) {
    visited.insert(op);

    if (isInitialOperation(op)) {
      nameAsInitialOperation(op, visited);
    } else {
      nameAsRegularOperation(op, visited);
    }
    foldOperation(op);
    reorderOperationOperandsByName(op);
  }
}

bool NormalizePass::isInitialOperation(Operation *const op) const noexcept {
  return !op->use_empty() and hasOnlyImmediateOperands(op);
}

bool NormalizePass::hasOnlyImmediateOperands(
    Operation *const op) const noexcept {
  for (Value operand : op->getOperands())
    if (Operation *defOp = operand.getDefiningOp())
      if (!(defOp->hasTrait<OpTrait::ConstantLike>()))
        return false;
  return true;
}

std::string inline to_string(uint64_t const hash) noexcept {
  std::ostringstream oss;
  oss << std::hex << std::setw(5) << std::setfill('0') << hash;
  std::string tmp = oss.str();
  return tmp.size() > 5 ? tmp.substr(tmp.size() - 5, 5) : tmp;
}

uint64_t inline strHash(std::string_view data) noexcept {
  const static uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
  const static uint64_t FNV_PRIME = 0x100000001b3ULL;
  uint64_t hash = FNV_OFFSET;
  for (const auto &c : data) {
    hash ^= static_cast<uint64_t>(c);
    hash *= FNV_PRIME;
  }
  return hash;
}

std::string inline split(std::string_view str, const char &delimiter,
                         int indx = 0) noexcept {
  std::stringstream ss{std::string{str}};
  std::string item;
  int cnt = 0;
  while (std::getline(ss, item, delimiter)) {
    if (cnt == indx) {
      std::replace(item.begin(), item.end(), ':', '_');
      return item;
    } else {
      cnt++;
    }
  }
  return nullptr;
}

void NormalizePass::nameAsInitialOperation(
    Operation *op, llvm::SmallPtrSet<const Operation *, 32> &visited) {

  for (Value operand : op->getOperands())
    if (Operation *defOp = operand.getDefiningOp())
      RenameOperation(defOp, visited);

  uint64_t Hash = MagicHashConstant;

  uint64_t opcodeHash = strHash(op->getName().getStringRef().str());
  Hash = llvm::hashing::detail::hash_16_bytes(Hash, opcodeHash);

  SmallPtrSet<const Operation *, 32> Visited;
  SetVector<int> OutputFootprint = getOutputFootprint(op, Visited);

  for (const auto &Output : OutputFootprint)
    Hash = llvm::hashing::detail::hash_16_bytes(Hash, Output);

  std::string Name{""};
  Name.append("vl" + std::to_string(Hash).substr(0, 5));

  if (auto call = dyn_cast<func::CallOp>(op)) {
    llvm::StringRef callee = call.getCallee();
    Name.append(callee.str());
  }

  if (op->getNumOperands() == 0) {
    Name.append("$");
    if (auto call = dyn_cast<func::CallOp>(op)) {
      Name.append("void");
    } else {
      std::string TextRepresentation;
      AsmState state(op, flags);
      llvm::raw_string_ostream Stream(TextRepresentation);
      op->print(Stream, state);
      std::string hash = to_string(strHash(split(Stream.str(), '=', 1)));
      Name.append(hash);
    }
    Name.append("$");
  }

  OpBuilder b(op->getContext());
  StringAttr sat = b.getStringAttr(Name);
  Location newLoc = NameLoc::get(sat, op->getLoc());
  op->setLoc(newLoc);
}

void NormalizePass::nameAsRegularOperation(
    Operation *op, llvm::SmallPtrSet<const Operation *, 32> &visited) {

  for (Value operand : op->getOperands())
    if (Operation *defOp = operand.getDefiningOp())
      RenameOperation(defOp, visited);

  uint64_t Hash = MagicHashConstant;

  uint64_t opcodeHash = strHash(op->getName().getStringRef().str());
  Hash = llvm::hashing::detail::hash_16_bytes(Hash, opcodeHash);

  SmallVector<uint64_t, 4> OperandsOpcodes;

  for (Value operand : op->getOperands())
    if (Operation *defOp = operand.getDefiningOp())
      OperandsOpcodes.push_back(strHash(defOp->getName().getStringRef().str()));

  if (op->hasTrait<OpTrait::IsCommutative>())
    llvm::sort(OperandsOpcodes.begin(), OperandsOpcodes.end());

  for (const uint64_t Code : OperandsOpcodes)
    Hash = llvm::hashing::detail::hash_16_bytes(Hash, Code);

  SmallString<512> Name;
  Name.append("op" + std::to_string(Hash).substr(0, 5));

  if (auto call = dyn_cast<func::CallOp>(op)) {
    llvm::StringRef callee = call.getCallee();
    Name.append(callee.str());
  }

  OpBuilder b(op->getContext());
  StringAttr sat = b.getStringAttr(Name);
  Location newLoc = NameLoc::get(sat, op->getLoc());
  op->setLoc(newLoc);
}

bool inline starts_with(std::string_view base,
                        std::string_view check) noexcept {
  return base.size() >= check.size() &&
         std::equal(check.begin(), check.end(), base.begin());
}

void NormalizePass::foldOperation(Operation *op) {
  if (isOutput(*op) || op->getNumOperands() == 0)
    return;

  std::string TextRepresentation;
  AsmState state(op, flags);
  llvm::raw_string_ostream Stream(TextRepresentation);
  op->print(Stream, state);

  auto opName = split(Stream.str(), '=', 0);
  if (!starts_with(opName, "%op") && !starts_with(opName, "%vl"))
    return;

  SmallVector<std::string, 4> Operands;

  for (Value operand : op->getOperands()) {
    if (Operation *defOp = operand.getDefiningOp()) {
      std::string TextRepresentation;
      AsmState state(defOp, flags);
      llvm::raw_string_ostream Stream(TextRepresentation);
      defOp->print(Stream, state);
      auto name = split(Stream.str(), '=', 0);

      bool hasNormalName =
          (starts_with(name, "%op") || starts_with(name, "%vl"));

      if (hasNormalName) {
        Operands.push_back(name.substr(1, 7));
      } else {
        Operands.push_back(name);
      }
    } else if (auto ba = dyn_cast<BlockArgument>(operand)) {
      Block *ownerBlock = ba.getOwner();
      unsigned argIndex = ba.getArgNumber();
      if (auto func = dyn_cast<func::FuncOp>(ownerBlock->getParentOp())) {
        if (&func.front() == ownerBlock) {
          Operands.push_back(std::string("funcArg" + std::to_string(argIndex)));
        } else {
          Operands.push_back(
              std::string("blockArg" + std::to_string(argIndex)));
        }
      } else {
        Operands.push_back(std::string("blockArg" + std::to_string(argIndex)));
      }
    }
  }

  if (op->hasTrait<OpTrait::IsCommutative>())
    llvm::sort(Operands.begin(), Operands.end());

  SmallString<512> Name;
  Name.append(opName.substr(1, 7));

  Name.append("$");
  for (size_t i = 0, size_ = Operands.size(); i < size_; ++i) {
    Name.append(Operands[i]);

    if (i < size_ - 1)
      Name.append("-");
  }
  Name.append("$");

  OpBuilder b(op->getContext());
  StringAttr sat = b.getStringAttr(Name);
  Location newLoc = NameLoc::get(sat, op->getLoc());
  op->setLoc(newLoc);
}

void NormalizePass::reorderOperationOperandsByName(Operation *op) {
  if (op->getNumOperands() == 0)
    return;

  SmallVector<std::pair<std::string, Value>, 4> Operands;

  for (Value operand : op->getOperands()) {
    std::string TextRepresentation;
    llvm::raw_string_ostream Stream(TextRepresentation);
    operand.printAsOperand(Stream, flags);
    Operands.push_back({Stream.str(), operand});
  }

  if (op->hasTrait<OpTrait::IsCommutative>()) {
    llvm::sort(
        Operands.begin(), Operands.end(), [](const auto &a, const auto &b) {
          return llvm::StringRef(a.first).compare_insensitive(b.first) < 0;
        });
  }

  for (size_t i = 0, size_ = Operands.size(); i < size_; i++) {
    op->setOperand(i, Operands[i].second);
  }
}

void NormalizePass::reorderOperations(
    const SmallVector<Operation *, 16> &Outputs) {
  llvm::SmallPtrSet<const Operation *, 32> visited;
  for (auto *const op : Outputs)
    for (Value operand : op->getOperands())
      if (Operation *defOp = operand.getDefiningOp())
        reorderOperation(defOp, op, visited);
}

void NormalizePass::reorderOperation(
    Operation *used, Operation *user,
    llvm::SmallPtrSet<const Operation *, 32> &visited) {
  if (!visited.count(used)) {
    visited.insert(used);

    Block *usedBlock = used->getBlock();
    Block *userBlock = user->getBlock();

    if (usedBlock == userBlock)
      used->moveBefore(user);
    else
      used->moveBefore(&usedBlock->back());

    for (Value operand : used->getOperands())
      if (Operation *defOp = operand.getDefiningOp())
        reorderOperation(defOp, used, visited);
  }
}

void NormalizePass::collectOutputOperations(
    Block &block, SmallVector<Operation *, 16> &Outputs) const noexcept {
  for (auto &innerOp : block)
    if (isOutput(innerOp))
      Outputs.emplace_back(&innerOp);
}

bool NormalizePass::isOutput(Operation &op) const noexcept {
  if (op.hasTrait<OpTrait::IsTerminator>())
    return true;

  if (auto memOp = dyn_cast<MemoryEffectOpInterface>(&op)) {
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    memOp.getEffects(effects);
    for (auto &effect : effects)
      if (isa<MemoryEffects::Write>(effect.getEffect()))
        return true;
  }

  if (auto call = dyn_cast<func::CallOp>(op))
    return true;

  return false;
}

llvm::SetVector<int> NormalizePass::getOutputFootprint(
    Operation *op, llvm::SmallPtrSet<const Operation *, 32> &visited) const {
  llvm::SetVector<int> Outputs;
  if (!visited.count(op)) {
    visited.insert(op);

    if (isOutput(*op)) {
      func::FuncOp func = op->getParentOfType<func::FuncOp>();

      unsigned Count = 0;
      for (Block &block : func.getRegion())
        for (Operation &innerOp : block) {
          if (&innerOp == op)
            Outputs.insert(Count);
          Count++;
        }

      return Outputs;
    }

    for (OpOperand &use : op->getUses()) {
      Operation *useOp = use.getOwner();
      if (useOp) {
        llvm::SetVector<int> OutputsUsingUop =
            getOutputFootprint(useOp, visited);

        Outputs.insert(OutputsUsingUop.begin(), OutputsUsingUop.end());
      }
    }
  }

  return Outputs;
}
