//===- SSAContext.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file declares a specialization of the GenericSSAContext<X>
/// class template for LLVM IR.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_SSACONTEXT_H
#define LLVM_IR_SSACONTEXT_H

#include "llvm/ADT/GenericSSAContext.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/Support/Printable.h"

#include <memory>

namespace llvm {
class BasicBlock;
class Function;
class Instruction;
class Value;
template <typename> class SmallVectorImpl;
template <typename, bool> class DominatorTreeBase;

inline auto instrs(const BasicBlock &BB) {
  return llvm::make_range(BB.begin(), BB.end());
}

template <> class GenericSSAContext<Function> {
  Function *F;

public:
  using BlockT = BasicBlock;
  using FunctionT = Function;
  using InstructionT = Instruction;
  using ValueRefT = Value *;
  using ConstValueRefT = const Value *;
  static Value *ValueRefNull;
  using DominatorTreeT = DominatorTreeBase<BlockT, false>;

  void setFunction(Function &Fn);
  Function *getFunction() const { return F; }

  static BasicBlock *getEntryBlock(Function &F);
  static const BasicBlock *getEntryBlock(const Function &F);

  static void appendBlockDefs(SmallVectorImpl<Value *> &defs,
                              BasicBlock &block);
  static void appendBlockDefs(SmallVectorImpl<const Value *> &defs,
                              const BasicBlock &block);

  static void appendBlockTerms(SmallVectorImpl<Instruction *> &terms,
                               BasicBlock &block);
  static void appendBlockTerms(SmallVectorImpl<const Instruction *> &terms,
                               const BasicBlock &block);

  static bool comesBefore(const Instruction *lhs, const Instruction *rhs);
  static bool isConstantValuePhi(const Instruction &Instr);
  const BasicBlock *getDefBlock(const Value *value) const;

  Printable print(const BasicBlock *Block) const;
  Printable print(const Instruction *Inst) const;
  Printable print(const Value *Value) const;
};

using SSAContext = GenericSSAContext<Function>;

} // namespace llvm

#endif // LLVM_IR_SSACONTEXT_H
