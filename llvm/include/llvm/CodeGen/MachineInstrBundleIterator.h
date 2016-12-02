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

template <class T, bool IsReverse> struct MachineInstrBundleIteratorTraits;
template <class T> struct MachineInstrBundleIteratorTraits<T, false> {
  typedef simple_ilist<T, ilist_sentinel_tracking<true>> list_type;
  typedef typename list_type::iterator instr_iterator;
  typedef typename list_type::iterator nonconst_instr_iterator;
  typedef typename list_type::const_iterator const_instr_iterator;
};
template <class T> struct MachineInstrBundleIteratorTraits<T, true> {
  typedef simple_ilist<T, ilist_sentinel_tracking<true>> list_type;
  typedef typename list_type::reverse_iterator instr_iterator;
  typedef typename list_type::reverse_iterator nonconst_instr_iterator;
  typedef typename list_type::const_reverse_iterator const_instr_iterator;
};
template <class T> struct MachineInstrBundleIteratorTraits<const T, false> {
  typedef simple_ilist<T, ilist_sentinel_tracking<true>> list_type;
  typedef typename list_type::const_iterator instr_iterator;
  typedef typename list_type::iterator nonconst_instr_iterator;
  typedef typename list_type::const_iterator const_instr_iterator;
};
template <class T> struct MachineInstrBundleIteratorTraits<const T, true> {
  typedef simple_ilist<T, ilist_sentinel_tracking<true>> list_type;
  typedef typename list_type::const_reverse_iterator instr_iterator;
  typedef typename list_type::reverse_iterator nonconst_instr_iterator;
  typedef typename list_type::const_reverse_iterator const_instr_iterator;
};

template <bool IsReverse> struct MachineInstrBundleIteratorHelper;
template <> struct MachineInstrBundleIteratorHelper<false> {
  /// Get the beginning of the current bundle.
  template <class Iterator> static Iterator getBundleBegin(Iterator I) {
    if (!I.isEnd())
      while (I->isBundledWithPred())
        --I;
    return I;
  }

  /// Get the final node of the current bundle.
  template <class Iterator> static Iterator getBundleFinal(Iterator I) {
    if (!I.isEnd())
      while (I->isBundledWithSucc())
        ++I;
    return I;
  }

  /// Increment forward ilist iterator.
  template <class Iterator> static void increment(Iterator &I) {
    I = std::next(getBundleFinal(I));
  }

  /// Decrement forward ilist iterator.
  template <class Iterator> static void decrement(Iterator &I) {
    I = getBundleBegin(std::prev(I));
  }
};

template <> struct MachineInstrBundleIteratorHelper<true> {
  /// Get the beginning of the current bundle.
  template <class Iterator> static Iterator getBundleBegin(Iterator I) {
    return MachineInstrBundleIteratorHelper<false>::getBundleBegin(
               I.getReverse())
        .getReverse();
  }

  /// Get the final node of the current bundle.
  template <class Iterator> static Iterator getBundleFinal(Iterator I) {
    return MachineInstrBundleIteratorHelper<false>::getBundleFinal(
               I.getReverse())
        .getReverse();
  }

  /// Increment reverse ilist iterator.
  template <class Iterator> static void increment(Iterator &I) {
    I = getBundleBegin(std::next(I));
  }

  /// Decrement reverse ilist iterator.
  template <class Iterator> static void decrement(Iterator &I) {
    I = std::prev(getBundleFinal(I));
  }
};

/// MachineBasicBlock iterator that automatically skips over MIs that are
/// inside bundles (i.e. walk top level MIs only).
template <typename Ty, bool IsReverse = false>
class MachineInstrBundleIterator : MachineInstrBundleIteratorHelper<IsReverse> {
  typedef MachineInstrBundleIteratorTraits<Ty, IsReverse> Traits;
  typedef typename Traits::instr_iterator instr_iterator;
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
  typedef typename Traits::nonconst_instr_iterator nonconst_instr_iterator;
  typedef typename Traits::const_instr_iterator const_instr_iterator;
  typedef MachineInstrBundleIterator<
      typename nonconst_instr_iterator::value_type, IsReverse>
      nonconst_iterator;
  typedef MachineInstrBundleIterator<Ty, !IsReverse> reverse_iterator;

public:
  MachineInstrBundleIterator(instr_iterator MI) : MII(MI) {
    assert((!MI.getNodePtr() || MI.isEnd() || !MI->isBundledWithPred()) &&
           "It's not legal to initialize MachineInstrBundleIterator with a "
           "bundled MI");
  }

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
      const MachineInstrBundleIterator<OtherTy, IsReverse> &I,
      typename std::enable_if<std::is_convertible<OtherTy *, Ty *>::value,
                              void *>::type = nullptr)
      : MII(I.getInstrIterator()) {}
  MachineInstrBundleIterator() : MII(nullptr) {}

  /// Get the bundle iterator for the given instruction's bundle.
  static MachineInstrBundleIterator getAtBundleBegin(instr_iterator MI) {
    return MachineInstrBundleIteratorHelper<IsReverse>::getBundleBegin(MI);
  }

  reference operator*() const { return *MII; }
  pointer operator->() const { return &operator*(); }

  /// Check for null.
  bool isValid() const { return MII.getNodePtr(); }

  friend bool operator==(const MachineInstrBundleIterator &L,
                         const MachineInstrBundleIterator &R) {
    return L.MII == R.MII;
  }
  friend bool operator==(const MachineInstrBundleIterator &L,
                         const const_instr_iterator &R) {
    return L.MII == R; // Avoid assertion about validity of R.
  }
  friend bool operator==(const const_instr_iterator &L,
                         const MachineInstrBundleIterator &R) {
    return L == R.MII; // Avoid assertion about validity of L.
  }
  friend bool operator==(const MachineInstrBundleIterator &L,
                         const nonconst_instr_iterator &R) {
    return L.MII == R; // Avoid assertion about validity of R.
  }
  friend bool operator==(const nonconst_instr_iterator &L,
                         const MachineInstrBundleIterator &R) {
    return L == R.MII; // Avoid assertion about validity of L.
  }
  friend bool operator==(const MachineInstrBundleIterator &L, const_pointer R) {
    return L == const_instr_iterator(R); // Avoid assertion about validity of R.
  }
  friend bool operator==(const_pointer L, const MachineInstrBundleIterator &R) {
    return const_instr_iterator(L) == R; // Avoid assertion about validity of L.
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
  friend bool operator!=(const MachineInstrBundleIterator &L,
                         const const_instr_iterator &R) {
    return !(L == R);
  }
  friend bool operator!=(const const_instr_iterator &L,
                         const MachineInstrBundleIterator &R) {
    return !(L == R);
  }
  friend bool operator!=(const MachineInstrBundleIterator &L,
                         const nonconst_instr_iterator &R) {
    return !(L == R);
  }
  friend bool operator!=(const nonconst_instr_iterator &L,
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
    this->decrement(MII);
    return *this;
  }
  MachineInstrBundleIterator &operator++() {
    this->increment(MII);
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

  nonconst_iterator getNonConstIterator() const { return MII.getNonConst(); }

  reverse_iterator getReverse() const { return MII.getReverse(); }
};

} // end namespace llvm

#endif
