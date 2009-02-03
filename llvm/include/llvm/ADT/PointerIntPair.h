//===- llvm/ADT/PointerIntPair.h - Pair for pointer and int -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PointerIntPair class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_POINTERINTPAIR_H
#define LLVM_ADT_POINTERINTPAIR_H

#include "llvm/Support/DataTypes.h"
#include <cassert>

namespace llvm {

template<typename T>
struct DenseMapInfo;

/// PointerIntPair - This class implements a pair of a pointer and small
/// integer.  It is designed to represent this in the space required by one
/// pointer by bitmangling the integer into the low part of the pointer.  This
/// can only be done for small integers: typically up to 3 bits, but it depends
/// on the alignment returned by the allocator in use.
///
template <typename PointerTy, unsigned IntBits, typename IntType=unsigned>
class PointerIntPair {
  intptr_t Value;
public:
  PointerIntPair() : Value(0) {}
  PointerIntPair(PointerTy Ptr, IntType Int) : Value(0) {
    setPointer(Ptr);
    setInt(Int);
  }

  PointerTy getPointer() const {
    return reinterpret_cast<PointerTy>(Value & ~((1 << IntBits)-1));
  }

  IntType getInt() const {
    return (IntType)(Value & ((1 << IntBits)-1));
  }

  void setPointer(PointerTy Ptr) {
    intptr_t PtrVal = reinterpret_cast<intptr_t>(Ptr);
    assert((PtrVal & ((1 << IntBits)-1)) == 0 &&
           "Pointer is not sufficiently aligned");
    Value = PtrVal | (intptr_t)getInt();
  }

  void setInt(IntType Int) {
    intptr_t IntVal = Int;
    assert(IntVal < (1 << IntBits) && "Integer too large for field");
    Value = reinterpret_cast<intptr_t>(getPointer()) | IntVal;
  }

  void *getOpaqueValue() const { return reinterpret_cast<void*>(Value); }
  void setFromOpaqueValue(void *Val) { Value = reinterpret_cast<intptr_t>(Val);}

  bool operator==(const PointerIntPair &RHS) const {return Value == RHS.Value;}
  bool operator!=(const PointerIntPair &RHS) const {return Value != RHS.Value;}
  bool operator<(const PointerIntPair &RHS) const {return Value < RHS.Value;}
  bool operator>(const PointerIntPair &RHS) const {return Value > RHS.Value;}
  bool operator<=(const PointerIntPair &RHS) const {return Value <= RHS.Value;}
  bool operator>=(const PointerIntPair &RHS) const {return Value >= RHS.Value;}
};

// Provide specialization of DenseMapInfo for PointerIntPair.
template<typename PointerTy, unsigned IntBits, typename IntType>
struct DenseMapInfo<PointerIntPair<PointerTy, IntBits, IntType> > {
  typedef PointerIntPair<PointerTy, IntBits, IntType> Ty;
  static Ty getEmptyKey() {
    return Ty(reinterpret_cast<PointerTy>(-1 << IntBits),
              IntType((1 << IntBits)-1));
  }
  static Ty getTombstoneKey() {
    return Ty(reinterpret_cast<PointerTy>(-2 << IntBits), IntType(0));
  }
  static unsigned getHashValue(Ty V) {
    uintptr_t IV = reinterpret_cast<uintptr_t>(V.getOpaqueValue());
    return unsigned(IV) ^ unsigned(IV >> 9);
  }
  static bool isEqual(const Ty &LHS, const Ty &RHS) { return LHS == RHS; }
  static bool isPod() { return true; }
};

} // end namespace llvm
#endif
