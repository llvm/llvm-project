//===- Utils.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Collector for SandboxIR related convenience functions that don't belong in
// other classes.

#ifndef LLVM_SANDBOXIR_UTILS_H
#define LLVM_SANDBOXIR_UTILS_H

#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/SandboxIR/SandboxIR.h"

namespace llvm::sandboxir {

class Utils {
public:
  /// \Returns the expected type of \p Value V. For most Values this is
  /// equivalent to getType, but for stores returns the stored type, rather
  /// than void, and for ReturnInsts returns the returned type.
  static Type *getExpectedType(const Value *V) {
    if (auto *I = dyn_cast<Instruction>(V)) {
      // A Return's value operand can be null if it returns void.
      if (auto *RI = dyn_cast<ReturnInst>(I)) {
        if (RI->getReturnValue() == nullptr)
          return RI->getType();
      }
      return getExpectedValue(I)->getType();
    }
    return V->getType();
  }

  /// \Returns the expected Value for this instruction. For most instructions,
  /// this is the instruction itself, but for stores returns the stored
  /// operand, and for ReturnInstructions returns the returned value.
  static Value *getExpectedValue(const Instruction *I) {
    if (auto *SI = dyn_cast<StoreInst>(I))
      return SI->getValueOperand();
    if (auto *RI = dyn_cast<ReturnInst>(I))
      return RI->getReturnValue();
    return const_cast<Instruction *>(I);
  }

  /// \Returns the number of bits required to represent the operands or return
  /// value of \p V in \p DL.
  static unsigned getNumBits(Value *V, const DataLayout &DL) {
    Type *Ty = getExpectedType(V);
    return DL.getTypeSizeInBits(Ty->LLVMTy);
  }

  /// Equivalent to MemoryLocation::getOrNone(I).
  static std::optional<llvm::MemoryLocation>
  memoryLocationGetOrNone(const Instruction *I) {
    return llvm::MemoryLocation::getOrNone(cast<llvm::Instruction>(I->Val));
  }
};
} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_UTILS_H
