//===- Normalize.cpp - Conversion from MLIR to its canonical form ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Normalize/Normalize.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Hashing.h"

#include <iomanip>
#include <sstream>

namespace mlir {
#define GEN_PASS_DEF_NORMALIZE
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "normalize"

namespace {
/// NormalizePass aims to transform MLIR into it's normal form
struct NormalizePass : public impl::NormalizeBase<NormalizePass> {
  NormalizePass() = default;

  void runOnOperation() override;

private:
  // Random constant for hashing, so the state isn't zero.
  const uint64_t magicHashConstant = 0x6acaa36bef8325c5ULL;
  void
  collectOutputOperations(Block &block,
                          SmallVector<Operation *, 16> &outputs) const noexcept;
  bool isOutput(Operation &op) const noexcept;
  void reorderOperations(const SmallVector<Operation *, 16> &outputs);
  void reorderOperation(Operation *used, Operation *user,
                        SmallPtrSet<const Operation *, 32> &visited);
  void renameOperations(const SmallVector<Operation *, 16> &outputs);
  void renameOperation(Operation *op,
                       SmallPtrSet<const Operation *, 32> &visited);
  bool isInitialOperation(Operation *const op) const noexcept;
  void nameAsInitialOperation(Operation *op,
                              SmallPtrSet<const Operation *, 32> &visited);
  void nameAsRegularOperation(Operation *op,
                              SmallPtrSet<const Operation *, 32> &visited);
  bool hasOnlyImmediateOperands(Operation *const op) const noexcept;
  SetVector<int>
  getOutputFootprint(Operation *op,
                     SmallPtrSet<const Operation *, 32> &visited) const;
  void appendRenamedOperands(Operation *op, SmallString<512> &name);
  void reorderOperationOperandsByName(Operation *op);
  OpPrintingFlags flags{};
};
} // namespace

/// Entry method to the NormalizePass
void NormalizePass::runOnOperation() {
  flags.printNameLocAsPrefix(true);

  ModuleOp module = getOperation();

  for (auto &op : module.getOps()) {
    SmallVector<Operation *, 16> outputs;

    for (auto &region : op.getRegions())
      for (auto &block : region)
        collectOutputOperations(block, outputs);

    reorderOperations(outputs);
    renameOperations(outputs);
  }
}

void NormalizePass::renameOperations(
    const SmallVector<Operation *, 16> &outputs) {
  SmallPtrSet<const Operation *, 32> visited;

  for (auto *op : outputs)
    renameOperation(op, visited);
}

/// Renames operations graphically (recursive) in accordance with the
/// def-use tree, starting from the initial operations (defs), finishing at
/// the output (top-most user) operations.
void NormalizePass::renameOperation(
    Operation *op, SmallPtrSet<const Operation *, 32> &visited) {
  if (!visited.count(op)) {
    visited.insert(op);

    if (isInitialOperation(op)) {
      nameAsInitialOperation(op, visited);
    } else {
      nameAsRegularOperation(op, visited);
    }
    if (op->hasTrait<OpTrait::IsCommutative>())
      reorderOperationOperandsByName(op);
  }
}

/// Helper method checking whether a given operation has users and only
/// immediate operands.
bool NormalizePass::isInitialOperation(Operation *const op) const noexcept {
  return !op->use_empty() and hasOnlyImmediateOperands(op);
}

/// Helper method checking whether all operands of a given operation has a
/// ConstantLike OpTrait
bool NormalizePass::hasOnlyImmediateOperands(
    Operation *const op) const noexcept {
  for (Value operand : op->getOperands())
    if (Operation *defOp = operand.getDefiningOp())
      if (!(defOp->hasTrait<OpTrait::ConstantLike>()))
        return false;
  return true;
}

std::string inline toString(uint64_t const hash) noexcept {
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

/// Names operation following the scheme:
/// vl00000Callee$Operands$
///
/// Where 00000 is a hash calculated considering operation's opcode and output
/// footprint. Callee's name is only included when operations's type is
/// CallOp. If the operation has operands, the renaming is further handled
/// in appendRenamedOperands, otherwise if it's a call operation with no
/// arguments, void is appended, else a hash of the definition of the operation
/// is appended.
void NormalizePass::nameAsInitialOperation(
    Operation *op, SmallPtrSet<const Operation *, 32> &visited) {

  for (Value operand : op->getOperands())
    if (Operation *defOp = operand.getDefiningOp())
      renameOperation(defOp, visited);

  uint64_t hash = magicHashConstant;

  uint64_t opcodeHash = strHash(op->getName().getStringRef().str());
  hash = llvm::hashing::detail::hash_16_bytes(hash, opcodeHash);

  SmallPtrSet<const Operation *, 32> visitedLocal;
  SetVector<int> outputFootprint = getOutputFootprint(op, visitedLocal);

  for (const auto &output : outputFootprint)
    hash = llvm::hashing::detail::hash_16_bytes(hash, output);

  SmallString<512> name;
  name.append("vl" + std::to_string(hash).substr(0, 5));

  if (auto call = dyn_cast<func::CallOp>(op)) {
    StringRef callee = call.getCallee();
    name.append(callee.str());
  }

  if (op->getNumOperands() == 0) {
    name.append("$");
    if (auto call = dyn_cast<func::CallOp>(op)) {
      name.append("void");
    } else {
      std::string textRepresentation;
      AsmState state(op, flags);
      llvm::raw_string_ostream stream(textRepresentation);
      op->print(stream, state);
      std::string hashStr = toString(strHash(split(stream.str(), '=', 1)));
      name.append(hashStr);
    }
    name.append("$");

    OpBuilder b(op->getContext());
    StringAttr sat = b.getStringAttr(name);
    Location newLoc = NameLoc::get(sat, op->getLoc());
    op->setLoc(newLoc);

    return;
  }

  appendRenamedOperands(op, name);
}

/// Names operation following the scheme:
/// op00000Callee$Operands$
///
/// Where 00000 is a hash calculated considering operation's opcode and its
/// operands opcode. Callee's name is only included when operations's type is
/// CallOp. A regular operation must have operands, thus the renaming is further
/// handled in appendRenamedOperands.
void NormalizePass::nameAsRegularOperation(
    Operation *op, SmallPtrSet<const Operation *, 32> &visited) {

  for (Value operand : op->getOperands())
    if (Operation *defOp = operand.getDefiningOp())
      renameOperation(defOp, visited);

  uint64_t hash = magicHashConstant;

  uint64_t opcodeHash = strHash(op->getName().getStringRef().str());
  hash = llvm::hashing::detail::hash_16_bytes(hash, opcodeHash);

  SmallVector<uint64_t, 4> operandOpcodes;

  for (Value operand : op->getOperands())
    if (Operation *defOp = operand.getDefiningOp())
      operandOpcodes.push_back(strHash(defOp->getName().getStringRef().str()));

  if (op->hasTrait<OpTrait::IsCommutative>())
    llvm::sort(operandOpcodes.begin(), operandOpcodes.end());

  for (const uint64_t code : operandOpcodes)
    hash = llvm::hashing::detail::hash_16_bytes(hash, code);

  SmallString<512> name;
  name.append("op" + std::to_string(hash).substr(0, 5));

  if (auto call = dyn_cast<func::CallOp>(op)) {
    StringRef callee = call.getCallee();
    name.append(callee.str());
  }

  appendRenamedOperands(op, name);
}

bool inline startsWith(std::string_view base, std::string_view check) noexcept {
  return base.size() >= check.size() &&
         std::equal(check.begin(), check.end(), base.begin());
}

/// This function serves a dual purpose of appending the operands name in the
/// operation while at the same time shortening it. Because of the recursive
/// def-use chain traversal, the operands should already have been renamed and
/// if they were an initial / regular operation, we truncate them by taking the
/// first 7 characters of the renamed operand. The operand could also have been
/// a block/function argument which is handled separately.
void NormalizePass::appendRenamedOperands(Operation *op,
                                          SmallString<512> &name) {
  if (op->getNumOperands() == 0)
    return;

  SmallVector<std::string, 4> operands;

  for (Value operand : op->getOperands()) {
    if (Operation *defOp = operand.getDefiningOp()) {
      std::string textRepresentation;
      AsmState state(defOp, flags);
      llvm::raw_string_ostream stream(textRepresentation);
      defOp->print(stream, state);
      auto operandName = split(stream.str(), '=', 0);

      bool hasNormalName =
          (startsWith(operandName, "%op") || startsWith(operandName, "%vl"));

      if (hasNormalName) {
        operands.push_back(operandName.substr(1, 7));
      } else {
        operands.push_back(operandName);
      }
    } else if (auto ba = dyn_cast<BlockArgument>(operand)) {
      Block *ownerBlock = ba.getOwner();
      unsigned argIndex = ba.getArgNumber();
      if (auto func = dyn_cast<func::FuncOp>(ownerBlock->getParentOp())) {
        if (&func.front() == ownerBlock) {
          operands.push_back(std::string("funcArg" + std::to_string(argIndex)));
        } else {
          operands.push_back(
              std::string("blockArg" + std::to_string(argIndex)));
        }
      } else {
        operands.push_back(std::string("blockArg" + std::to_string(argIndex)));
      }
    }
  }

  if (op->hasTrait<OpTrait::IsCommutative>())
    sort(operands.begin(), operands.end());

  name.append("$");
  for (size_t i = 0, size_ = operands.size(); i < size_; ++i) {
    name.append(operands[i]);

    if (i < size_ - 1)
      name.append("-");
  }
  name.append("$");

  OpBuilder b(op->getContext());
  Location newLoc = NameLoc::get(b.getStringAttr(name), op->getLoc());
  op->setLoc(newLoc);
}

/// Reorders operation's operands alphabetically. This method assumes
/// that passed operation is commutative.
void NormalizePass::reorderOperationOperandsByName(Operation *op) {
  if (op->getNumOperands() == 0)
    return;

  SmallVector<std::pair<std::string, Value>, 4> operands;

  for (Value operand : op->getOperands()) {
    std::string textRepresentation;
    llvm::raw_string_ostream stream(textRepresentation);
    operand.printAsOperand(stream, flags);
    operands.push_back({stream.str(), operand});
  }

  if (op->hasTrait<OpTrait::IsCommutative>()) {
    sort(operands.begin(), operands.end(), [](const auto &a, const auto &b) {
      return StringRef(a.first).compare_insensitive(b.first) < 0;
    });
  }

  for (size_t i = 0, size_ = operands.size(); i < size_; i++) {
    op->setOperand(i, operands[i].second);
  }
}

void NormalizePass::reorderOperations(
    const SmallVector<Operation *, 16> &outputs) {
  SmallPtrSet<const Operation *, 32> visited;
  for (auto *const op : outputs)
    for (Value operand : op->getOperands())
      if (Operation *defOp = operand.getDefiningOp())
        reorderOperation(defOp, op, visited);
}

/// Reorders operations by walking up the tree from each operand of an output
/// operation and reducing the def-use distance.
void NormalizePass::reorderOperation(
    Operation *used, Operation *user,
    SmallPtrSet<const Operation *, 32> &visited) {
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
    Block &block, SmallVector<Operation *, 16> &outputs) const noexcept {
  for (auto &innerOp : block)
    if (isOutput(innerOp))
      outputs.emplace_back(&innerOp);
}

/// The following Operations are termed as output:
///  - Terminator operations are outputs
///  - Any operation that implements MemoryEffectOpInterface and reports at
///    least one MemoryEffects::Write effect is an output
///  - Any operation that implements CallOpInterface is treated as an output
///    (calls are conservatively assumed to possibly produce side effects).
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

  if (dyn_cast<CallOpInterface>(&op))
    return true;

  return false;
}

/// Helper method returning indices (distance from the beginning of the basic
/// block) of output operations using the given operation. It Walks down the
/// def-use tree recursively. 
SetVector<int> NormalizePass::getOutputFootprint(
    Operation *op, SmallPtrSet<const Operation *, 32> &visited) const {
  SetVector<int> outputsVec;
  if (!visited.count(op)) {
    visited.insert(op);

    if (isOutput(*op)) {
      func::FuncOp func = op->getParentOfType<func::FuncOp>();

      unsigned count = 0;
      for (Block &block : func.getRegion())
        for (Operation &innerOp : block) {
          if (&innerOp == op)
            outputsVec.insert(count);
          count++;
        }

      return outputsVec;
    }

    for (OpOperand &use : op->getUses()) {
      Operation *useOp = use.getOwner();
      if (useOp) {
        SetVector<int> outputsUsingUop = getOutputFootprint(useOp, visited);

        outputsVec.insert(outputsUsingUop.begin(), outputsUsingUop.end());
      }
    }
  }

  return outputsVec;
}
