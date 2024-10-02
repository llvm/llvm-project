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

#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/SandboxIR/Instruction.h"
#include <optional>

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

  /// \Returns the number of bits required to represent the operands or
  /// return value of \p I.
  static unsigned getNumBits(Instruction *I) {
    return I->getDataLayout().getTypeSizeInBits(getExpectedType(I)->LLVMTy);
  }

  /// Equivalent to MemoryLocation::getOrNone(I).
  static std::optional<llvm::MemoryLocation>
  memoryLocationGetOrNone(const Instruction *I) {
    return llvm::MemoryLocation::getOrNone(cast<llvm::Instruction>(I->Val));
  }

  /// \Returns the gap between the memory locations accessed by \p I0 and
  /// \p I1 in bytes.
  template <typename LoadOrStoreT>
  static std::optional<int>
  getPointerDiffInBytes(LoadOrStoreT *I0, LoadOrStoreT *I1, ScalarEvolution &SE,
                        const DataLayout &DL) {
    static_assert(std::is_same_v<LoadOrStoreT, LoadInst> ||
                      std::is_same_v<LoadOrStoreT, StoreInst>,
                  "Expected sandboxir::Load or sandboxir::Store!");
    llvm::Value *Opnd0 = I0->getPointerOperand()->Val;
    llvm::Value *Opnd1 = I1->getPointerOperand()->Val;
    llvm::Value *Ptr0 = getUnderlyingObject(Opnd0);
    llvm::Value *Ptr1 = getUnderlyingObject(Opnd1);
    if (Ptr0 != Ptr1)
      return false;
    llvm::Type *ElemTy = llvm::Type::getInt8Ty(SE.getContext());
    return getPointersDiff(ElemTy, Opnd0, ElemTy, Opnd1, DL, SE,
                           /*StrictCheck=*/false, /*CheckType=*/false);
  }

  /// \Returns true if \p I0 accesses a memory location lower than \p I1.
  /// Returns false if the difference cannot be determined, if the memory
  /// locations are equal, or if I1 accesses a memory location greater than I0.
  template <typename LoadOrStoreT>
  static bool atLowerAddress(LoadOrStoreT *I0, LoadOrStoreT *I1,
                             ScalarEvolution &SE, const DataLayout &DL) {
    auto Diff = getPointerDiffInBytes(I0, I1, SE, DL);
    if (!Diff)
      return false;
    return *Diff > 0;
  }

  static bool isStackSaveOrRestoreIntrinsic(Instruction *I) {
    auto *LLVMI = cast<llvm::Instruction>(I->Val);
    return match(LLVMI,
                 PatternMatch::m_Intrinsic<llvm::Intrinsic::stackrestore>()) ||
           match(LLVMI,
                 PatternMatch::m_Intrinsic<llvm::Intrinsic::stacksave>());
  }

  /// We consider \p I as a Memory Dependency Candidate instruction if it
  /// reads/write memory or if it has side-effects. This is used by the
  /// dependency graph.
  static bool isMemDepCandidate(Instruction *I) {
    auto *LLVMI = cast<llvm::Instruction>(I->Val);
    return LLVMI->mayReadOrWriteMemory() &&
           (!isa<llvm::IntrinsicInst>(LLVMI) ||
            (cast<llvm::IntrinsicInst>(LLVMI)->getIntrinsicID() !=
                 Intrinsic::sideeffect &&
             cast<llvm::IntrinsicInst>(LLVMI)->getIntrinsicID() !=
                 Intrinsic::pseudoprobe));
  }
};
} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_UTILS_H
