//===-- llvm/OperandTraits.h - OperandTraits class definition ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the traits classes that are handy for enforcing the correct
// layout of various User subclasses. It also provides the means for accessing
// the operands in the most efficient manner.
//

#ifndef LLVM_OPERAND_TRAITS_H
#define LLVM_OPERAND_TRAITS_H

#include "llvm/User.h"

namespace llvm {

//===----------------------------------------------------------------------===//
//                          FixedNumOperands Trait Class
//===----------------------------------------------------------------------===//

/// FixedNumOperandTraits - determine the allocation regime of the Use array
/// when it is a prefix to the User object, and the number of Use objects is
/// known at compile time.

template <unsigned ARITY>
struct FixedNumOperandTraits {
  static Use *op_begin(User* U) {
    return reinterpret_cast<Use*>(U) - ARITY;
  }
  static Use *op_end(User* U) {
    return reinterpret_cast<Use*>(U);
  }
  static unsigned operands(const User*) {
    return ARITY;
  }
  struct prefix {
    Use Ops[ARITY];
    prefix(); // DO NOT IMPLEMENT
  };
  template <class U>
  struct Layout {
    struct overlay : prefix, U {
      overlay(); // DO NOT IMPLEMENT
    };
  };
  static inline void *allocate(unsigned); // FIXME
};

//===----------------------------------------------------------------------===//
//                          OptionalOperands Trait Class
//===----------------------------------------------------------------------===//

template <unsigned ARITY = 1>
struct OptionalOperandTraits : FixedNumOperandTraits<ARITY> {
  static unsigned operands(const User *U) {
    return U->getNumOperands();
  }
};

//===----------------------------------------------------------------------===//
//                          VariadicOperand Trait Class
//===----------------------------------------------------------------------===//

/// VariadicOperandTraits - determine the allocation regime of the Use array
/// when it is a prefix to the User object, and the number of Use objects is
/// only known at allocation time.

template <unsigned MINARITY = 0>
struct VariadicOperandTraits {
  static Use *op_begin(User* U) {
    return reinterpret_cast<Use*>(U) - U->getNumOperands();
  }
  static Use *op_end(User* U) {
    return reinterpret_cast<Use*>(U);
  }
  static unsigned operands(const User *U) {
    return U->getNumOperands();
  }
  static inline void *allocate(unsigned); // FIXME
};

//===----------------------------------------------------------------------===//
//                          HungoffOperand Trait Class
//===----------------------------------------------------------------------===//

/// HungoffOperandTraits - determine the allocation regime of the Use array
/// when it is not a prefix to the User object, but allocated at an unrelated
/// heap address.
/// Assumes that the User subclass that is determined by this traits class
/// has an OperandList member of type User::op_iterator. [Note: this is now
/// trivially satisfied, because User has that member for historic reasons.]
///
/// This is the traits class that is needed when the Use array must be
/// resizable.

template <unsigned MINARITY = 1>
struct HungoffOperandTraits {
  static Use *op_begin(User* U) {
    return U->OperandList;
  }
  static Use *op_end(User* U) {
    return U->OperandList + U->getNumOperands();
  }
  static unsigned operands(const User *U) {
    return U->getNumOperands();
  }
  static inline void *allocate(unsigned); // FIXME
};

/// Macro for generating in-class operand accessor declarations.
/// It should only be called in the public section of the interface.
///
#define DECLARE_TRANSPARENT_OPERAND_ACCESSORS(VALUECLASS) \
  public: \
  inline VALUECLASS *getOperand(unsigned) const; \
  inline void setOperand(unsigned, VALUECLASS*); \
  protected: \
  template <unsigned> inline Use &Op(); \
  template <unsigned> inline const Use &Op() const; \
  public: \
  inline unsigned getNumOperands() const

/// Macro for generating out-of-class operand accessor definitions
#define DEFINE_TRANSPARENT_OPERAND_ACCESSORS(CLASS, VALUECLASS) \
VALUECLASS *CLASS::getOperand(unsigned i_nocapture) const { \
  assert(i_nocapture < OperandTraits<CLASS>::operands(this) \
         && "getOperand() out of range!"); \
  return static_cast<VALUECLASS*>( \
    OperandTraits<CLASS>::op_begin(const_cast<CLASS*>(this))[i_nocapture]); \
} \
void CLASS::setOperand(unsigned i_nocapture, VALUECLASS *Val_nocapture) { \
  assert(i_nocapture < OperandTraits<CLASS>::operands(this) \
         && "setOperand() out of range!"); \
  OperandTraits<CLASS>::op_begin(this)[i_nocapture] = Val_nocapture; \
} \
unsigned CLASS::getNumOperands() const { \
  return OperandTraits<CLASS>::operands(this);  \
} \
template <unsigned Idx_nocapture> Use &CLASS::Op() { \
  return OperandTraits<CLASS>::op_begin(this)[Idx_nocapture]; \
} \
template <unsigned Idx_nocapture> const Use &CLASS::Op() const { \
  return OperandTraits<CLASS>::op_begin( \
    const_cast<CLASS*>(this))[Idx_nocapture]; \
}


/// Macro for generating out-of-class operand accessor
/// definitions with casted result
#define DEFINE_TRANSPARENT_CASTED_OPERAND_ACCESSORS(CLASS, VALUECLASS) \
VALUECLASS *CLASS::getOperand(unsigned i_nocapture) const { \
  assert(i_nocapture < OperandTraits<CLASS>::operands(this) \
         && "getOperand() out of range!"); \
  return cast<VALUECLASS>( \
    OperandTraits<CLASS>::op_begin(const_cast<CLASS*>(this))[i_nocapture]); \
} \
void CLASS::setOperand(unsigned i_nocapture, VALUECLASS *Val_nocapture) { \
  assert(i_nocapture < OperandTraits<CLASS>::operands(this) \
         && "setOperand() out of range!"); \
  OperandTraits<CLASS>::op_begin(this)[i_nocapture] = Val_nocapture; \
} \
unsigned CLASS::getNumOperands() const { \
  return OperandTraits<CLASS>::operands(this); \
} \
template <unsigned Idx_nocapture> Use &CLASS::Op() { \
  return OperandTraits<CLASS>::op_begin(this)[Idx_nocapture]; \
} \
template <unsigned Idx_nocapture> const Use &CLASS::Op() const { \
  return OperandTraits<CLASS>::op_begin( \
    const_cast<CLASS*>(this))[Idx_nocapture]; \
}


} // End llvm namespace

#endif
