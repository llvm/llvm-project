//===-- IndirectCallVisitor.h - indirect call visitor ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements defines a visitor class and a helper function that find
// all indirect call-sites in a function.

#ifndef LLVM_ANALYSIS_INDIRECTCALLVISITOR_H
#define LLVM_ANALYSIS_INDIRECTCALLVISITOR_H

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/InstVisitor.h"
#include <vector>

namespace llvm {
// Visitor class that finds indirect calls or instructions that gives vtable
// value, depending on Type.
struct PGOIndirectCallVisitor : public InstVisitor<PGOIndirectCallVisitor> {
  enum class InstructionType {
    kIndirectCall = 0,
    kVTableVal = 1,
  };
  std::vector<CallBase *> IndirectCalls;
  SetVector<Instruction *, std::vector<Instruction *>> VTableAddrs;
  PGOIndirectCallVisitor(InstructionType Type) : Type(Type) {}

  void visitCallBase(CallBase &Call) {
    const CallInst *CI = dyn_cast<CallInst>(&Call);
    if (Type == InstructionType::kVTableVal && CI && CI->getCalledFunction()) {
      switch (CI->getCalledFunction()->getIntrinsicID()) {
      case Intrinsic::type_test:
      case Intrinsic::public_type_test:
      case Intrinsic::type_checked_load_relative:
      case Intrinsic::type_checked_load: {
        Value *VTablePtr = CI->getArgOperand(0)->stripPointerCasts();

        if (PtrTestedByTypeIntrinsics.count(VTablePtr) == 0) {
          Instruction *I = dyn_cast_or_null<Instruction>(VTablePtr);
          // This is the first type intrinsic where VTablePtr is used.
          // Assert that the VTablePtr is not found as a type profiling
          // candidate yet. Note nullptr won't be inserted into VTableAddrs in
          // the first place, so this assertion works even if 'VTablePtr' is not
          // an instruction.
          assert(VTableAddrs.count(I) == 0 &&
                 "Expect type intrinsic to record VTablePtr before virtual "
                 "functions are loaded to find vtables that should be "
                 "instrumented");

          PtrTestedByTypeIntrinsics.insert(VTablePtr);
        }
      } break;
      }
    }
    if (Call.isIndirectCall()) {
      IndirectCalls.push_back(&Call);
      if (Type == InstructionType::kVTableVal) {
        // Note without -fstrict-vtable-pointers, vtable pointers of the same
        // objects are loaded multiple times, and current implementation
        // instruments each load once.
        // FIXME: For more efficient instrumentation, analyze load invariant
        // vtable values (e.g., from the same pointer in C++) and instrument
        // them once.
        LoadInst *LI = dyn_cast<LoadInst>(Call.getCalledOperand());
        if (LI != nullptr) {
          Value *MaybeVTablePtr =
              LI->getPointerOperand()->stripInBoundsConstantOffsets();
          Instruction *VTableInstr = dyn_cast<Instruction>(MaybeVTablePtr);
          // If not used by any type intrinsic, this is not a vtable.
          // Inst visitor should see the very first type intrinsic using a
          // vtable before the very first virtual function load from this
          // vtable. This condition is asserted above.
          if (VTableInstr && PtrTestedByTypeIntrinsics.count(MaybeVTablePtr)) {
            VTableAddrs.insert(VTableInstr);
          }
        }
      }
    }
  }

  // write a helper function to vtable ptr for an indirect call.
  static Instruction *getAnnotatedVTableInstruction(CallBase *CB) {
    assert(CB != nullptr && "Caller guaranteed");
    LoadInst *LI = dyn_cast<LoadInst>(CB->getCalledOperand());

    if (LI != nullptr) {
      Value *FuncPtr = LI->getPointerOperand(); // GEP (or bitcast)
      Value *VTablePtr = FuncPtr->stripInBoundsConstantOffsets();
      if (VTablePtr != nullptr && isa<Instruction>(VTablePtr)) {
        return cast<Instruction>(VTablePtr);
      }
    }
    return nullptr;
  }

private:
  // Keeps track of the pointers that are tested by llvm type intrinsics for
  // look up.
  SmallPtrSet<Value *, 4> PtrTestedByTypeIntrinsics;
  InstructionType Type;
};

inline std::vector<CallBase *> findIndirectCalls(Function &F) {
  PGOIndirectCallVisitor ICV(
      PGOIndirectCallVisitor::InstructionType::kIndirectCall);
  ICV.visit(F);
  return ICV.IndirectCalls;
}

inline std::vector<Instruction *> findVTableAddrs(Function &F) {
  PGOIndirectCallVisitor ICV(
      PGOIndirectCallVisitor::InstructionType::kVTableVal);
  ICV.visit(F);
  return ICV.VTableAddrs.takeVector();
}

} // namespace llvm

#endif
