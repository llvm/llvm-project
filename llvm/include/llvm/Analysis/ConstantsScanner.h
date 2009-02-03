//==- llvm/Analysis/ConstantsScanner.h - Iterate over constants -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements an iterator to walk through the constants referenced by
// a method.  This is used by the Bitcode & Assembly writers to build constant
// pools.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CONSTANTSSCANNER_H
#define LLVM_ANALYSIS_CONSTANTSSCANNER_H

#include "llvm/Support/InstIterator.h"
#include "llvm/ADT/iterator.h"

namespace llvm {

class Constant;

class constant_iterator : public forward_iterator<const Constant, ptrdiff_t> {
  const_inst_iterator InstI;                // Method instruction iterator
  unsigned OpIdx;                           // Operand index

  typedef constant_iterator _Self;

  inline bool isAtConstant() const {
    assert(!InstI.atEnd() && OpIdx < InstI->getNumOperands() &&
           "isAtConstant called with invalid arguments!");
    return isa<Constant>(InstI->getOperand(OpIdx));
  }

public:
  inline constant_iterator(const Function *F) : InstI(inst_begin(F)), OpIdx(0) {
    // Advance to first constant... if we are not already at constant or end
    if (InstI != inst_end(F) &&                            // InstI is valid?
        (InstI->getNumOperands() == 0 || !isAtConstant())) // Not at constant?
      operator++();
  }

  inline constant_iterator(const Function *F, bool)   // end ctor
    : InstI(inst_end(F)), OpIdx(0) {
  }

  inline bool operator==(const _Self& x) const { return OpIdx == x.OpIdx &&
                                                        InstI == x.InstI; }
  inline bool operator!=(const _Self& x) const { return !operator==(x); }

  inline pointer operator*() const {
    assert(isAtConstant() && "Dereferenced an iterator at the end!");
    return cast<Constant>(InstI->getOperand(OpIdx));
  }
  inline pointer operator->() const { return operator*(); }

  inline _Self& operator++() {   // Preincrement implementation
    ++OpIdx;
    do {
      unsigned NumOperands = InstI->getNumOperands();
      while (OpIdx < NumOperands && !isAtConstant()) {
        ++OpIdx;
      }

      if (OpIdx < NumOperands) return *this;  // Found a constant!
      ++InstI;
      OpIdx = 0;
    } while (!InstI.atEnd());

    return *this;  // At the end of the method
  }

  inline _Self operator++(int) { // Postincrement
    _Self tmp = *this; ++*this; return tmp;
  }

  inline bool atEnd() const { return InstI.atEnd(); }
};

inline constant_iterator constant_begin(const Function *F) {
  return constant_iterator(F);
}

inline constant_iterator constant_end(const Function *F) {
  return constant_iterator(F, true);
}

} // End llvm namespace

#endif
