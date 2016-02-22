//===- llvm/CodeGen/MachineInstrBundleIterator.h ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines an iterator class that bundles MachineInstr.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEINSTRBUNDLEITERATOR_H
#define LLVM_CODEGEN_MACHINEINSTRBUNDLEITERATOR_H

#include "llvm/ADT/ilist.h"
#include <iterator>

namespace llvm {

/// MachineBasicBlock iterator that automatically skips over MIs that are
/// inside bundles (i.e. walk top level MIs only).
template <typename Ty>
class MachineInstrBundleIterator
    : public std::iterator<std::bidirectional_iterator_tag, Ty, ptrdiff_t> {
  typedef ilist_iterator<Ty> instr_iterator;
  instr_iterator MII;

public:
  MachineInstrBundleIterator(instr_iterator MI) : MII(MI) {}

  MachineInstrBundleIterator(Ty &MI) : MII(MI) {
    assert(!MI.isBundledWithPred() && "It's not legal to initialize "
                                      "MachineInstrBundleIterator with a "
                                      "bundled MI");
  }
  MachineInstrBundleIterator(Ty *MI) : MII(MI) {
    // FIXME: This conversion should be explicit.
    assert((!MI || !MI->isBundledWithPred()) && "It's not legal to initialize "
                                                "MachineInstrBundleIterator "
                                                "with a bundled MI");
  }
  // Template allows conversion from const to nonconst.
  template <class OtherTy>
  MachineInstrBundleIterator(const MachineInstrBundleIterator<OtherTy> &I)
      : MII(I.getInstrIterator()) {}
  MachineInstrBundleIterator() : MII(nullptr) {}

  Ty &operator*() const { return *MII; }
  Ty *operator->() const { return &operator*(); }

  // FIXME: This conversion should be explicit.
  operator Ty *() const { return MII.getNodePtrUnchecked(); }

  bool operator==(const MachineInstrBundleIterator &X) const {
    return MII == X.MII;
  }
  bool operator!=(const MachineInstrBundleIterator &X) const {
    return !operator==(X);
  }

  // Increment and decrement operators...
  MachineInstrBundleIterator &operator--() {
    do
      --MII;
    while (MII->isBundledWithPred());
    return *this;
  }
  MachineInstrBundleIterator &operator++() {
    while (MII->isBundledWithSucc())
      ++MII;
    ++MII;
    return *this;
  }
  MachineInstrBundleIterator operator--(int) {
    MachineInstrBundleIterator Temp = *this;
    --*this;
    return Temp;
  }
  MachineInstrBundleIterator operator++(int) {
    MachineInstrBundleIterator Temp = *this;
    ++*this;
    return Temp;
  }

  instr_iterator getInstrIterator() const { return MII; }
};

} // end namespace llvm

#endif
