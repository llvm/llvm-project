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
  std::vector<Instruction *> ProfiledAddresses;
  PGOIndirectCallVisitor(InstructionType Type) : Type(Type) {}

  void visitCallBase(CallBase &Call) {
    if (!Call.isIndirectCall())
      return;

    if (Type == InstructionType::kIndirectCall) {
      IndirectCalls.push_back(&Call);
      return;
    }

    assert(Type == InstructionType::kVTableVal && "Control flow guaranteed");

    LoadInst *LI = dyn_cast<LoadInst>(Call.getCalledOperand());
    // The code pattern to look for
    //
    // %vtable = load ptr, ptr %b
    // %vfn = getelementptr inbounds ptr, ptr %vtable, i64 1
    // %2 = load ptr, ptr %vfn
    // %call = tail call i32 %2(ptr %b)
    //
    // %vtable is the vtable address value to profile, and
    // %2 is the indirect call target address to profile.
    if (LI != nullptr) {
      Value *Ptr = LI->getPointerOperand();
      Value *VTablePtr = Ptr->stripInBoundsConstantOffsets();
      // This is a heuristic to find address feeding instructions.
      // FIXME: Add support in the frontend so LLVM type intrinsics are
      // emitted without LTO. This way, added intrinsics could filter
      // non-vtable instructions and reduce instrumentation overhead.
      // Since a non-vtable profiled address is not within the address
      // range of vtable objects, it's stored as zero in indexed profiles.
      // A pass that looks up symbol with an zero hash will (almost) always
      // find nullptr and skip the actual transformation (e.g., comparison
      // of symbols). So the performance overhead from non-vtable profiled
      // address is negligible if exists at all. Comparing loaded address
      // with symbol address guarantees correctness.
      if (VTablePtr != nullptr && isa<Instruction>(VTablePtr))
        ProfiledAddresses.push_back(cast<Instruction>(VTablePtr));
    }
  }

private:
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
  return ICV.ProfiledAddresses;
}

} // namespace llvm

#endif
