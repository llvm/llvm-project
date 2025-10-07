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
#include <iomanip>

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
  llvm::SetVector<int> getOutputFootprint(mlir::Operation* op, llvm::SmallPtrSet<const mlir::Operation *, 32> &visited);
  void foldOperation(mlir::Operation* op);
  void reorderOperationOperandsByName(mlir::Operation* op);
  mlir::OpPrintingFlags flags{};
};
} // namespace

void NormalizePass::runOnOperation() {
  flags.printNameLocAsPrefix(true);

  ModuleOp module = getOperation();

  for (Operation &op : module.getOps()) {
    SmallVector<Operation *, 16> Outputs;

    for (Region &region : op.getRegions())
      for (Block &block : region)
        collectOutputOperations(block, Outputs);

    reorderOperations(Outputs);

    RenameOperations(Outputs);

    for (Region& region : op.getRegions())
      for (Block &block : region)
        for (Operation &innerOp : block)
          foldOperation(&innerOp);

    for (Region& region : op.getRegions())
      for (Block &block : region)
        for (Operation &innerOp : block)
          reorderOperationOperandsByName(&innerOp);
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

    if (isInitialOperation(op)) {
      nameAsInitialOperation(op);
    } else {
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

std::string to_string(uint64_t const hash)
{
  std::ostringstream oss;
  oss << std::hex << std::setw(16) << std::setfill('0') << hash;
  std::string tmp = oss.str();
  return tmp.substr(11, 5);
}

uint64_t kernel_hash(std::string_view data)
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

void replace(std::string& str, char from, char to) {
  for(auto& it : str) {
    if(it == from) it = to;
  }
}

std::vector<std::string> split(std::string_view str, char delimiter) {
  std::vector<std::string> outs{};
  std::stringstream ss{ std::string {str} };
  std::string item;
  while(std::getline(ss, item, delimiter)) {
    replace(item, ':', '_');
    outs.emplace_back(item);
  }
  return outs;
}

void NormalizePass::nameAsInitialOperation(mlir::Operation* op) {
  SmallVector<SmallString<64>, 4> Operands;

  if(op->getNumOperands() == 0) {
    /**
     * INFO: Constant operations like arith.constant
     */
    if(auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
      Operands.push_back(StringRef(std::string{"void"}));
    } else {
      std::string TextRepresentation;
      mlir::AsmState state(op, flags);
      llvm::raw_string_ostream Stream(TextRepresentation);
      op->print(Stream, state);
      std::string hash = to_string(kernel_hash(split(Stream.str(), '=')[1]));
      Operands.push_back(StringRef(hash));
    }
  } else {
    for (mlir::Value operand : op->getOperands()) {
      if (mlir::Operation *defOp = operand.getDefiningOp()) {
        /**
         * INFO: Constant arguments like arith.constant
         */
        std::string TextRepresentation;
        mlir::AsmState state(defOp, flags);
        llvm::raw_string_ostream Stream(TextRepresentation);
        defOp->print(Stream, state);
        std::string hash = to_string(kernel_hash(split(Stream.str(), '=')[1]));
        Operands.push_back(StringRef(hash));
      } else {
        /**
         * INFO: Function Arguments
         */
        std::string TextRepresentation;
        mlir::AsmState state(op, flags);
        llvm::raw_string_ostream Stream(TextRepresentation);
        operand.print(Stream, state);
        std::string argNum = split(Stream.str(), ':')[1];
        argNum = argNum.substr(1, argNum.size() - 1);
        Operands.push_back(StringRef(std::string("arg" + argNum)));
      }
    }
  }

  if (op->hasTrait<OpTrait::IsCommutative>()) llvm::sort(Operands);

  uint64_t Hash = MagicHashConstant;

  uint64_t opcodeHash = kernel_hash(op->getName().getStringRef().str());
  Hash = llvm::hashing::detail::hash_16_bytes(Hash, opcodeHash);

  SmallPtrSet<const mlir::Operation *, 32> Visited;
  SetVector<int> OutputFootprint = getOutputFootprint(op, Visited);

  for (const int &Output : OutputFootprint)
    Hash = llvm::hashing::detail::hash_16_bytes(Hash, Output);

  std::string Name{""};
  Name.append("vl" + std::to_string(Hash).substr(0, 5));

  if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
    llvm::StringRef callee = call.getCallee();
    Name.append(callee.str());
  }

  Name.append("$");
  for (unsigned long i = 0; i < Operands.size(); ++i) {
    Name.append(std::string(Operands[i]));

    if (i < Operands.size() - 1)
      Name.append("--");
  }
  Name.append("$");

  mlir::OpBuilder b(op->getContext());
  mlir::StringAttr sat = b.getStringAttr(Name);
  mlir::Location newLoc = mlir::NameLoc::get(sat, op->getLoc());
  op->setLoc(newLoc);
}

void NormalizePass::nameAsRegularOperation(mlir::Operation* op, llvm::SmallPtrSet<const mlir::Operation *, 32> &visited) {
  SmallVector<SmallString<64>, 4> Operands;
  for (mlir::Value operand : op->getOperands()) {
    if (mlir::Operation *defOp = operand.getDefiningOp()) {
      RenameOperation(defOp, visited);

      std::string TextRepresentation;
      mlir::AsmState state(defOp, flags);
      llvm::raw_string_ostream Stream(TextRepresentation);
      defOp->print(Stream, state);
      Operands.push_back(StringRef(split(Stream.str(), '=')[0]));
    } else {
      /**
       * INFO: Function Arguments
       */
      std::string TextRepresentation;
      mlir::AsmState state(op, flags);
      llvm::raw_string_ostream Stream(TextRepresentation);
      operand.print(Stream, state);
      std::string argNum = split(Stream.str(), ':')[1];
      argNum = argNum.substr(1, argNum.size() - 1);
      Operands.push_back(StringRef(std::string("arg" + argNum)));
    }
  }

  if (op->hasTrait<OpTrait::IsCommutative>()) llvm::sort(Operands);

  uint64_t Hash = MagicHashConstant;

  uint64_t opcodeHash = kernel_hash(op->getName().getStringRef().str());
  Hash = llvm::hashing::detail::hash_16_bytes(Hash, opcodeHash);

  SmallVector<uint64_t, 4> OperandsOpcodes;

  for (mlir::Value operand : op->getOperands())
    if (mlir::Operation *defOp = operand.getDefiningOp())
      OperandsOpcodes.push_back(kernel_hash(defOp->getName().getStringRef().str()));

  if (op->hasTrait<OpTrait::IsCommutative>()) llvm::sort(OperandsOpcodes.begin(), OperandsOpcodes.end());

  for (const uint64_t Code : OperandsOpcodes)
    Hash = llvm::hashing::detail::hash_16_bytes(Hash, Code);

  SmallString<512> Name;
  Name.append("op" + std::to_string(Hash).substr(0, 5));

  if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
    llvm::StringRef callee = call.getCallee();
    Name.append(callee.str());
  }

  Name.append("$");
  for (unsigned long i = 0; i < Operands.size(); ++i) {
    Name.append(Operands[i]);

    if (i < Operands.size() - 1)
      Name.append("--");
  }
  Name.append("$");

  mlir::OpBuilder b(op->getContext());
  mlir::StringAttr sat = b.getStringAttr(Name);
  mlir::Location newLoc = mlir::NameLoc::get(sat, op->getLoc());
  op->setLoc(newLoc);
}

bool starts_with(std::string_view base, std::string_view check) {
  if(base.size() < check.size()) return false;
  for(int i = 0; i < check.size(); i++) if(base[i] != check[i]) return false;
  return true;
}

void NormalizePass::foldOperation(mlir::Operation* op) {
  if(isOutput(*op)) return;

  std::string TextRepresentation;
  mlir::AsmState state(op, flags);
  llvm::raw_string_ostream Stream(TextRepresentation);
  op->print(Stream, state);
  
  auto opName = split(Stream.str(), '=')[0];
  if(!starts_with(opName, "%op")) return;

  SmallVector<std::string, 4> Operands;

  for (mlir::Value operand : op->getOperands()) {
    if (mlir::Operation *defOp = operand.getDefiningOp()) {
      std::string TextRepresentation;
      mlir::AsmState state(defOp, flags);
      llvm::raw_string_ostream Stream(TextRepresentation);
      defOp->print(Stream, state);
      auto name = split(Stream.str(), '=')[0];

      bool hasNormalName = (starts_with(name, "%op") || starts_with(name, "%vl"));

      if(hasNormalName) {
        Operands.push_back(name.substr(1, 7));
      } else {
        Operands.push_back(name);
      }
    }
  }

  if (op->hasTrait<OpTrait::IsCommutative>()) llvm::sort(Operands.begin(), Operands.end());

  SmallString<512> Name;
  Name.append(opName.substr(1, 7));

  Name.append("$");
  for (unsigned long i = 0; i < Operands.size(); ++i) {
    Name.append(Operands[i]);

    if (i < Operands.size() - 1)
      Name.append("-");
  }
  Name.append("$");

  mlir::OpBuilder b(op->getContext());
  mlir::StringAttr sat = b.getStringAttr(Name);
  mlir::Location newLoc = mlir::NameLoc::get(sat, op->getLoc());
  op->setLoc(newLoc);
}

void NormalizePass::reorderOperationOperandsByName(mlir::Operation* op) {
  if(op->getNumOperands() == 0) return;

  SmallVector<std::pair<std::string, mlir::Value>, 4> Operands;

  for (mlir::Value operand : op->getOperands()) {
    std::string TextRepresentation;
    llvm::raw_string_ostream Stream(TextRepresentation);
    operand.printAsOperand(Stream, flags);
    Operands.push_back({Stream.str(), operand});
  }

  if (op->hasTrait<OpTrait::IsCommutative>()) {
    llvm::sort(Operands.begin(), Operands.end(),
                [](const auto &a, const auto &b) {
                  return llvm::StringRef(a.first).compare_insensitive(b.first) < 0;
                });
  }

  for(size_t i = 0; i < Operands.size(); i++) {
    op->setOperand(i, Operands[i].second);
  }
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

void NormalizePass::collectOutputOperations(Block &block, SmallVector<Operation *, 16> &Outputs) {
  for (Operation &innerOp : block)
    if (isOutput(innerOp))
      Outputs.emplace_back(&innerOp);
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

  if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) return true;

  return false;
}

llvm::SetVector<int> NormalizePass::getOutputFootprint(mlir::Operation* op, llvm::SmallPtrSet<const mlir::Operation *, 32> &visited) {
  llvm::SetVector<int> Outputs;
  if (!visited.count(op)) {
    visited.insert(op);

    if (isOutput(*op)) {
      mlir::func::FuncOp func = op->getParentOfType<mlir::func::FuncOp>();

      unsigned Count = 0;
        for (Block &block : func.getRegion())
          for(mlir::Operation &innerOp : block){
            if(&innerOp==op)
              Outputs.insert(Count);
            Count++;
        }

      return Outputs;
    }

    for (mlir::OpOperand &use : op->getUses()) {
      mlir::Operation *useOp = use.getOwner();
      if (useOp) {
        llvm::SetVector<int> OutputsUsingUop = getOutputFootprint(useOp, visited);

        Outputs.insert(OutputsUsingUop.begin(), OutputsUsingUop.end());
      }
    }
  }

  return Outputs;
}
