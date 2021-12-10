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

template <> class GenericSSAContext<Function> {
  Function *F;

public:
  using BlockT = BasicBlock;
  using FunctionT = Function;
  using InstructionT = Instruction;
  using ValueRefT = Value *;
  using DominatorTreeT = DominatorTreeBase<BlockT, false>;

  static BasicBlock *getEntryBlock(Function &F);

  void setFunction(Function &Fn);
  Function *getFunction() const { return F; }

  Printable print(BasicBlock *Block) const;
  Printable print(Instruction *Inst) const;
  Printable print(Value *Value) const;
};

using SSAContext = GenericSSAContext<Function>;

} // namespace llvm

#endif // LLVM_IR_SSACONTEXT_H
