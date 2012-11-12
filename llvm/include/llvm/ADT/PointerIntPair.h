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

#include "llvm/Support/PointerLikeTypeTraits.h"
#include <cassert>

namespace llvm {

template<typename T>
struct DenseMapInfo;

/// PointerIntPair - This class implements a pair of a pointer and small
/// integer.  It is designed to represent this in the space required by one
/// pointer by bitmangling the integer into the low part of the pointer.  This
/// can only be done for small integers: typically up to 3 bits, but it depends
/// on the number of bits available according to PointerLikeTypeTraits for the
/// type.
///
/// Note that PointerIntPair always puts the Int part in the highest bits
/// possible.  For example, PointerIntPair<void*, 1, bool> will put the bit for
/// the bool into bit #2, not bit #0, which allows the low two bits to be used
/// for something else.  For example, this allows:
///   PointerIntPair<PointerIntPair<void*, 1, bool>, 1, bool>
/// ... and the two bools will land in different bits.
///
template <typename PointerTy, unsigned IntBits, typename IntType=unsigned,
          typename PtrTraits = PointerLikeTypeTraits<PointerTy> >
class PointerIntPair {
  intptr_t Value;
  enum {
    /// PointerBitMask - The bits that come from the pointer.
    PointerBitMask =
      ~(uintptr_t)(((intptr_t)1 << PtrTraits::NumLowBitsAvailable)-1),

    /// IntShift - The number of low bits that we reserve for other uses, and
    /// keep zero.
    IntShift = (uintptr_t)PtrTraits::NumLowBitsAvailable-IntBits,
    
    /// IntMask - This is the unshifted mask for valid bits of the int type.
    IntMask = (uintptr_t)(((intptr_t)1 << IntBits)-1),
    
    // ShiftedIntMask - This is the bits for the integer shifted in place.
    ShiftedIntMask = (uintptr_t)(IntMask << IntShift)
  };
public:
  PointerIntPair() : Value(0) {}
  PointerIntPair(PointerTy Ptr, IntType Int) : Value(0) {
    assert(IntBits <= PtrTraits::NumLowBitsAvailable &&
           "PointerIntPair formed with integer size too large for pointer");
    setPointer(Ptr);
    setInt(Int);
  }

  PointerTy getPointer() const {
    return PtrTraits::getFromVoidPointer(
                         reinterpret_cast<void*>(Value & PointerBitMask));
  }

  IntType getInt() const {
    return (IntType)((Value >> IntShift) & IntMask);
  }

  void setPointer(PointerTy Ptr) {
    intptr_t PtrVal
      = reinterpret_cast<intptr_t>(PtrTraits::getAsVoidPointer(Ptr));
    assert((PtrVal & ((1 << PtrTraits::NumLowBitsAvailable)-1)) == 0 &&
           "Pointer is not sufficiently aligned");
    // Preserve all low bits, just update the pointer.
    Value = PtrVal | (Value & ~PointerBitMask);
  }

  void setInt(IntType Int) {
    intptr_t IntVal = Int;
    assert(IntVal < (1 << IntBits) && "Integer too large for field");
    
    // Preserve all bits other than the ones we are updating.
    Value &= ~ShiftedIntMask;     // Remove integer field.
    Value |= IntVal << IntShift;  // Set new integer.
  }

  PointerTy const *getAddrOfPointer() const {
    return const_cast<PointerIntPair *>(this)->getAddrOfPointer();
  }

  PointerTy *getAddrOfPointer() {
    assert(Value == reinterpret_cast<intptr_t>(getPointer()) &&
           "Can only return the address if IntBits is cleared and "
           "PtrTraits doesn't change the pointer");
    return reinterpret_cast<PointerTy *>(&Value);
  }

  void *getOpaqueValue() const { return reinterpret_cast<void*>(Value); }
  void setFromOpaqueValue(void *Val) { Value = reinterpret_cast<intptr_t>(Val);}

  static PointerIntPair getFromOpaqueValue(void *V) {
    PointerIntPair P; P.setFromOpaqueValue(V); return P; 
  }

  // Allow PointerIntPairs to be created from const void * if and only if the
  // pointer type could be created from a const void *.
  static PointerIntPair getFromOpaqueValue(const void *V) {
    (void)PtrTraits::getFromVoidPointer(V);
    return getFromOpaqueValue(const_cast<void *>(V));
  }

  bool operator==(const PointerIntPair &RHS) const {return Value == RHS.Value;}
  bool operator!=(const PointerIntPair &RHS) const {return Value != RHS.Value;}
  bool operator<(const PointerIntPair &RHS) const {return Value < RHS.Value;}
  bool operator>(const PointerIntPair &RHS) const {return Value > RHS.Value;}
  bool operator<=(const PointerIntPair &RHS) const {return Value <= RHS.Value;}
  bool operator>=(const PointerIntPair &RHS) const {return Value >= RHS.Value;}
};

template <typename T> struct isPodLike;
template<typename PointerTy, unsigned IntBits, typename IntType>
struct isPodLike<PointerIntPair<PointerTy, IntBits, IntType> > {
   static const bool value = true;
};
  
// Provide specialization of DenseMapInfo for PointerIntPair.
template<typename PointerTy, unsigned IntBits, typename IntType>
struct DenseMapInfo<PointerIntPair<PointerTy, IntBits, IntType> > {
  typedef PointerIntPair<PointerTy, IntBits, IntType> Ty;
  static Ty getEmptyKey() {
    uintptr_t Val = static_cast<uintptr_t>(-1);
    Val <<= PointerLikeTypeTraits<PointerTy>::NumLowBitsAvailable;
    return Ty(reinterpret_cast<PointerTy>(Val), IntType((1 << IntBits)-1));
  }
  static Ty getTombstoneKey() {
    uintptr_t Val = static_cast<uintptr_t>(-2);
    Val <<= PointerLikeTypeTraits<PointerTy>::NumLowBitsAvailable;
    return Ty(reinterpret_cast<PointerTy>(Val), IntType(0));
  }
  static unsigned getHashValue(Ty V) {
    uintptr_t IV = reinterpret_cast<uintptr_t>(V.getOpaqueValue());
    return unsigned(IV) ^ unsigned(IV >> 9);
  }
  static bool isEqual(const Ty &LHS, const Ty &RHS) { return LHS == RHS; }
};

// Teach SmallPtrSet that PointerIntPair is "basically a pointer".
template<typename PointerTy, unsigned IntBits, typename IntType,
         typename PtrTraits>
class PointerLikeTypeTraits<PointerIntPair<PointerTy, IntBits, IntType,
                                           PtrTraits> > {
public:
  static inline void *
  getAsVoidPointer(const PointerIntPair<PointerTy, IntBits, IntType> &P) {
    return P.getOpaqueValue();
  }
  static inline PointerIntPair<PointerTy, IntBits, IntType>
  getFromVoidPointer(void *P) {
    return PointerIntPair<PointerTy, IntBits, IntType>::getFromOpaqueValue(P);
  }
  static inline PointerIntPair<PointerTy, IntBits, IntType>
  getFromVoidPointer(const void *P) {
    return PointerIntPair<PointerTy, IntBits, IntType>::getFromOpaqueValue(P);
  }
  enum {
    NumLowBitsAvailable = PtrTraits::NumLowBitsAvailable - IntBits
  };
};

} // end namespace llvm
#endif
