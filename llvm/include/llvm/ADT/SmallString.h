//===- llvm/ADT/SmallString.h - 'Normally small' strings --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SmallString class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SMALLSTRING_H
#define LLVM_ADT_SMALLSTRING_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DataTypes.h"
#include <cstring>

namespace llvm {

/// SmallString - A SmallString is just a SmallVector with methods and accessors
/// that make it work better as a string (e.g. operator+ etc).
template<unsigned InternalLen>
class SmallString : public SmallVector<char, InternalLen> {
public:
  // Default ctor - Initialize to empty.
  SmallString() {}

  // Initialize with a range.
  template<typename ItTy>
  SmallString(ItTy S, ItTy E) : SmallVector<char, InternalLen>(S, E) {}

  // Copy ctor.
  SmallString(const SmallString &RHS) : SmallVector<char, InternalLen>(RHS) {}


  // Extra methods.
  const char *c_str() const {
    SmallString *This = const_cast<SmallString*>(this);
    // Ensure that there is a \0 at the end of the string.
    This->reserve(this->size()+1);
    This->End[0] = 0;
    return this->begin();
  }

  // Extra operators.
  const SmallString &operator=(const char *RHS) {
    this->clear();
    return *this += RHS;
  }

  SmallString &operator+=(const char *RHS) {
    this->append(RHS, RHS+strlen(RHS));
    return *this;
  }
  SmallString &operator+=(char C) {
    this->push_back(C);
    return *this;
  }

  SmallString &append_uint_32(uint32_t N) {
    char Buffer[20];
    char *BufPtr = Buffer+20;

    if (N == 0) *--BufPtr = '0';  // Handle special case.

    while (N) {
      *--BufPtr = '0' + char(N % 10);
      N /= 10;
    }
    this->append(BufPtr, Buffer+20);
    return *this;
  }

  SmallString &append_uint(uint64_t N) {
    if (N == uint32_t(N))
      return append_uint_32(uint32_t(N));

    char Buffer[40];
    char *BufPtr = Buffer+40;

    if (N == 0) *--BufPtr = '0';  // Handle special case...

    while (N) {
      *--BufPtr = '0' + char(N % 10);
      N /= 10;
    }

    this->append(BufPtr, Buffer+40);
    return *this;
  }

  SmallString &append_sint(int64_t N) {
    // TODO, wrong for minint64.
    if (N < 0) {
      this->push_back('-');
      N = -N;
    }
    return append_uint(N);
  }

};


}

#endif
