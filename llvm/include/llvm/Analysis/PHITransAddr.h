//===- PHITransAddr.h - PHI Translation for Addresses -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the PHITransAddr class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_PHITRANSADDR_H
#define LLVM_ANALYSIS_PHITRANSADDR_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

namespace llvm {
class AssumptionCache;
class DominatorTree;
class DataLayout;
class TargetLibraryInfo;

// SelectAddr - storage of normal Value address or Condition value and pair of
// addresses for true and false variant of select dependency. If Addrs are not
// present (both values are nullptr), V is a normal address; otherwise, V is a
// select condition and SelectAddrs are "true" and "false" addresses.
class SelectAddr {
public:
  using SelectAddrs = std::pair<Value *, Value *>;

  SelectAddr(Value *Addr) : V(Addr), Addrs(nullptr, nullptr) {}
  SelectAddr(Value *Cond, SelectAddrs Addrs) : V(Cond), Addrs(Addrs) {
    assert(Cond && "Condition must be present");
    assert(hasSelectAddrs() && "Addrs must be present");
  };
  Value *getAddr() const {
    assert(!hasSelectAddrs());
    return V;
  }
  std::pair<Value *, SelectAddrs> getSelectCondAndAddrs() const {
    // If Addrs is present, return it.
    if (hasSelectAddrs())
      return {V, Addrs};
    // Otherwise V must be SelectInst; return condition and both addresses from
    // its operands.
    auto *SI = cast<SelectInst>(V);
    return {SI->getCondition(), {SI->getTrueValue(), SI->getFalseValue()}};
  }

private:
  Value *V;
  SelectAddrs Addrs;

  bool hasSelectAddrs() const { return Addrs.first && Addrs.second; }
};

/// PHITransAddr - An address value which tracks and handles phi translation.
/// As we walk "up" the CFG through predecessors, we need to ensure that the
/// address we're tracking is kept up to date.  For example, if we're analyzing
/// an address of "&A[i]" and walk through the definition of 'i' which is a PHI
/// node, we *must* phi translate i to get "&A[j]" or else we will analyze an
/// incorrect pointer in the predecessor block.
///
/// This is designed to be a relatively small object that lives on the stack and
/// is copyable.
///
class PHITransAddr {
  /// Addr - The actual address we're analyzing.
  Value *Addr;

  /// The DataLayout we are playing with.
  const DataLayout &DL;

  /// TLI - The target library info if known, otherwise null.
  const TargetLibraryInfo *TLI = nullptr;

  /// A cache of \@llvm.assume calls used by SimplifyInstruction.
  AssumptionCache *AC;

  /// InstInputs - The inputs for our symbolic address.
  SmallVector<Instruction*, 4> InstInputs;

public:
  PHITransAddr(Value *Addr, const DataLayout &DL, AssumptionCache *AC)
      : Addr(Addr), DL(DL), AC(AC) {
    // If the address is an instruction, the whole thing is considered an input.
    addAsInput(Addr);
  }

  Value *getAddr() const { return Addr; }

  /// getSelectCondition - if address has select input, return its condition
  /// (otherwise nullptr).
  Value *getSelectCondition() const {
    for (auto *I : InstInputs)
      if (auto *SI = dyn_cast<SelectInst>(I))
        return SI->getCondition();
    return nullptr;
  }

  /// needsPHITranslationFromBlock - Return true if moving from the specified
  /// BasicBlock to its predecessors requires PHI translation.
  bool needsPHITranslationFromBlock(BasicBlock *BB) const {
    // We do need translation if one of our input instructions is defined in
    // this block.
    return any_of(InstInputs, [BB](const auto &InstInput) {
      return InstInput->getParent() == BB;
    });
  }

  /// isPotentiallyPHITranslatable - If this needs PHI translation, return true
  /// if we have some hope of doing it.  This should be used as a filter to
  /// avoid calling PHITranslateValue in hopeless situations.
  bool isPotentiallyPHITranslatable() const;

  /// translateValue - PHI translate the current address up the CFG from
  /// CurBB to Pred, updating our state to reflect any needed changes.  If
  /// 'MustDominate' is true, the translated value must dominate PredBB.
  Value *translateValue(BasicBlock *CurBB, BasicBlock *PredBB,
                        const DominatorTree *DT, bool MustDominate);

  /// translateValue - PHI translate the current address from \p CurBB to \p
  /// PredBB, and if the resulted address depends on select instructions with \p
  /// Cond predicate, translate both cases of this selects.
  SelectAddr::SelectAddrs translateValue(BasicBlock *CurBB, BasicBlock *PredBB,
                                         const DominatorTree *DT, Value *Cond);

  /// translateWithInsertion - PHI translate this value into the specified
  /// predecessor block, inserting a computation of the value if it is
  /// unavailable.
  ///
  /// All newly created instructions are added to the NewInsts list.  This
  /// returns null on failure.
  ///
  Value *translateWithInsertion(BasicBlock *CurBB, BasicBlock *PredBB,
                                const DominatorTree &DT,
                                SmallVectorImpl<Instruction *> &NewInsts);

  void dump() const;

  /// verify - Check internal consistency of this data structure.  If the
  /// structure is valid, it returns true.  If invalid, it prints errors and
  /// returns false.
  bool verify() const;

private:
  /// translateSubExpr - recursively translate value \p V from \p CurBB to \p
  /// PredBB, and if value depends from selects with \p Cond condition, also
  /// translate it through these selects with \p CondVal predicate. Return
  /// nullptr on failure.
  Value *translateSubExpr(Value *V, BasicBlock *CurBB, BasicBlock *PredBB,
                          const DominatorTree *DT, Value *Cond = nullptr,
                          bool CondVal = false);

  /// insertTranslatedSubExpr - Insert a computation of the PHI translated
  /// version of 'V' for the edge PredBB->CurBB into the end of the PredBB
  /// block.  All newly created instructions are added to the NewInsts list.
  /// This returns null on failure.
  ///
  Value *insertTranslatedSubExpr(Value *InVal, BasicBlock *CurBB,
                                 BasicBlock *PredBB, const DominatorTree &DT,
                                 SmallVectorImpl<Instruction *> &NewInsts);

  /// addAsInput - If the specified value is an instruction, add it as an input.
  Value *addAsInput(Value *V) {
    // If V is an instruction, it is now an input.
    if (Instruction *VI = dyn_cast<Instruction>(V))
      InstInputs.push_back(VI);
    return V;
  }
};

} // end namespace llvm

#endif
