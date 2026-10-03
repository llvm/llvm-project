//===- llvm/IR/InstructionListener.h - Per-function instruction listener --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares InstructionListener, a per-function interface that
// notifies analyses when an Instruction is removed from a Function or
// RAUW'd (replaced with another value). Think of it as "a value handle to
// many values" — a single per-function registration replaces one handle per
// tracked value.
//
// Registration and deregistration happen automatically via RAII: the
// constructor registers with a Function, and the destructor deregisters.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_INSTRUCTIONLISTENER_H
#define LLVM_IR_INSTRUCTIONLISTENER_H

#include "llvm/IR/Function.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Instruction;
class Value;

/// A per-function listener notified when an Instruction is removed from its
/// parent BasicBlock or when an Instruction is RAUW'd.
///
/// Subclasses implement the callbacks by passing static functions to the
/// constructor, which static_casts the listener back to the derived type.
/// This avoids virtual dispatch overhead while preserving type safety.
///
/// Lifetime is managed via RAII: the constructor registers with the
/// Function, and the destructor deregisters.
class InstructionListener {
public:
  using RemoveCallbackT = void (*)(InstructionListener *, Instruction *);
  using RAUWCallbackT = void (*)(InstructionListener *, Instruction *Old,
                                 Value *New);

private:
  Function *F;
  RemoveCallbackT RemoveCallback;
  RAUWCallbackT RAUWCallback;

public:
  InstructionListener(Function &F, RemoveCallbackT RemoveCB,
                      RAUWCallbackT RAUWCB = nullptr)
      : F(&F), RemoveCallback(RemoveCB), RAUWCallback(RAUWCB) {
    F.addInstructionListener(this);
  }
  ~InstructionListener() {
    if (F)
      F->removeInstructionListener(this);
  }

  InstructionListener(const InstructionListener &) = delete;
  InstructionListener &operator=(const InstructionListener &) = delete;

  /// Called by ~Function() to detach the listener before the Function is
  /// destroyed. After this call, the destructor becomes a no-op.
  void detach() { F = nullptr; }

  void instructionRemoved(Instruction *I) {
    if (RemoveCallback)
      RemoveCallback(this, I);
  }
  void instructionRAUW(Instruction *Old, Value *New) {
    if (RAUWCallback)
      RAUWCallback(this, Old, New);
  }
  Function *getFunction() const { return F; }
};

} // namespace llvm

#endif // LLVM_IR_INSTRUCTIONLISTENER_H
