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
template <typename Ty> class MachineInstrBundleIterator {
  typedef ilist_iterator<Ty> instr_iterator;
  instr_iterator MII;

public:
  typedef typename instr_iterator::value_type value_type;
  typedef typename instr_iterator::difference_type difference_type;
  typedef typename instr_iterator::pointer pointer;
  typedef typename instr_iterator::reference reference;
  typedef std::bidirectional_iterator_tag iterator_category;

  typedef typename instr_iterator::const_pointer const_pointer;
  typedef typename instr_iterator::const_reference const_reference;

private:
  typedef typename std::remove_const<value_type>::type nonconst_value_type;
  typedef ilist_node<nonconst_value_type> node_type;
  typedef ilist_iterator<nonconst_value_type> nonconst_instr_iterator;
  typedef MachineInstrBundleIterator<nonconst_value_type> nonconst_iterator;

public:
  MachineInstrBundleIterator(instr_iterator MI) : MII(MI) {}

  MachineInstrBundleIterator(reference MI) : MII(MI) {
    assert(!MI.isBundledWithPred() && "It's not legal to initialize "
                                      "MachineInstrBundleIterator with a "
                                      "bundled MI");
  }
  MachineInstrBundleIterator(pointer MI) : MII(MI) {
    // FIXME: This conversion should be explicit.
    assert((!MI || !MI->isBundledWithPred()) && "It's not legal to initialize "
                                                "MachineInstrBundleIterator "
                                                "with a bundled MI");
  }
  // Template allows conversion from const to nonconst.
  template <class OtherTy>
  MachineInstrBundleIterator(
      const MachineInstrBundleIterator<OtherTy> &I,
      typename std::enable_if<std::is_convertible<OtherTy *, Ty *>::value,
                              void *>::type = nullptr)
      : MII(I.getInstrIterator()) {}
  MachineInstrBundleIterator() : MII(nullptr) {}

  reference operator*() const { return *MII; }
  pointer operator->() const { return &operator*(); }

  /// Check for null.
  bool isValid() const { return MII.getNodePtr(); }

  friend bool operator==(const MachineInstrBundleIterator &L,
                         const MachineInstrBundleIterator &R) {
    return L.MII == R.MII;
  }
  friend bool operator==(const MachineInstrBundleIterator &L, const_pointer R) {
    // Avoid assertion about validity of R.
    return L.MII == instr_iterator(const_cast<pointer>(R));
  }
  friend bool operator==(const_pointer L, const MachineInstrBundleIterator &R) {
    // Avoid assertion about validity of L.
    return instr_iterator(const_cast<pointer>(L)) == R.MII;
  }
  friend bool operator==(const MachineInstrBundleIterator &L,
                         const_reference R) {
    return L == &R; // Avoid assertion about validity of R.
  }
  friend bool operator==(const_reference L,
                         const MachineInstrBundleIterator &R) {
    return &L == R; // Avoid assertion about validity of L.
  }

  friend bool operator!=(const MachineInstrBundleIterator &L,
                         const MachineInstrBundleIterator &R) {
    return !(L == R);
  }
  friend bool operator!=(const MachineInstrBundleIterator &L, const_pointer R) {
    return !(L == R);
  }
  friend bool operator!=(const_pointer L, const MachineInstrBundleIterator &R) {
    return !(L == R);
  }
  friend bool operator!=(const MachineInstrBundleIterator &L,
                         const_reference R) {
    return !(L == R);
  }
  friend bool operator!=(const_reference L,
                         const MachineInstrBundleIterator &R) {
    return !(L == R);
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

  nonconst_iterator getNonConstIterator() const {
    if (auto *N = const_cast<node_type *>(MII.getNodePtr()))
      return nonconst_iterator(nonconst_instr_iterator(*N));
    return nonconst_iterator();
  }
};

} // end namespace llvm

#endif
