//===- GISelWorkList.h - Worklist for GISel passes ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_GISEL_WORKLIST_H
#define LLVM_GISEL_WORKLIST_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/Debug.h"

namespace llvm {

class MachineFunction;

// Worklist which mostly works similar to InstCombineWorkList, but on
// MachineInstrs. The main difference with something like a SetVector is that
// erasing an element doesn't move all elements over one place - instead just
// nulls out the element of the vector.
//
// This worklist operates on instructions within a particular function. This is
// important for acquiring the rights to modify/replace instructions a
// GISelChangeObserver reports as the observer doesn't have the right to make
// changes to the instructions it sees so we use our access to the
// MachineFunction to establish that it's ok to add a given instruction to the
// worklist.
//
// FIXME: Does it make sense to factor out common code with the
// instcombinerWorkList?
template<unsigned N>
class GISelWorkList {
  MachineFunction *MF;
  SmallVector<MachineInstr *, N> Worklist;
  DenseMap<MachineInstr *, unsigned> WorklistMap;

public:
  GISelWorkList(MachineFunction *MF) : MF(MF) {}

  bool empty() const { return WorklistMap.empty(); }

  unsigned size() const { return WorklistMap.size(); }

  /// Add the specified instruction to the worklist if it isn't already in it.
  void insert(MachineInstr *I) {
    // It would be safe to add this instruction to the worklist regardless but
    // for consistency with the const version, check that the instruction we're
    // adding would have been accepted if we were given a const pointer instead.
    insert(const_cast<const MachineInstr *>(I));
  }

  void insert(const MachineInstr *I) {
    // Confirm we'd be able to find the non-const pointer we want to schedule if
    // we wanted to. We have the right to schedule work that may modify any
    // instruction in MF.
    assert(I->getParent() && "Expected parent BB");
    assert(I->getParent()->getParent() && "Expected parent function");
    assert((!MF || I->getParent()->getParent() == MF) &&
           "Expected parent function to be current function or not given");

    // But don't actually do the search since we can derive it from the const
    // pointer.
    MachineInstr *NonConstI = const_cast<MachineInstr *>(I);
    if (WorklistMap.try_emplace(NonConstI, Worklist.size()).second) {
      Worklist.push_back(NonConstI);
    }
  }

  /// Remove I from the worklist if it exists.
  void remove(const MachineInstr *I) {
    auto It = WorklistMap.find(I);
    if (It == WorklistMap.end()) return; // Not in worklist.

    // Don't bother moving everything down, just null out the slot.
    Worklist[It->second] = nullptr;

    WorklistMap.erase(It);
  }

  MachineInstr *pop_back_val() {
    MachineInstr *I;
    do {
      I = Worklist.pop_back_val();
    } while(!I);
    assert(I && "Pop back on empty worklist");
    WorklistMap.erase(I);
    return I;
  }
};

} // end namespace llvm.

#endif
