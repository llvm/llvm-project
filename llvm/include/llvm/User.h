//===-- llvm/User.h - User class definition ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class defines the interface that one who 'use's a Value must implement.
// Each instance of the Value class keeps track of what User's have handles
// to it.
//
//  * Instructions are the largest class of User's.
//  * Constants may be users of other constants (think arrays and stuff)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_USER_H
#define LLVM_USER_H

#include "llvm/Value.h"

namespace llvm {

class User : public Value {
  User(const User &);             // Do not implement
  void *operator new(size_t);     // Do not implement
protected:
  /// OperandList - This is a pointer to the array of Users for this operand.
  /// For nodes of fixed arity (e.g. a binary operator) this array will live
  /// embedded into the derived class.  For nodes of variable arity
  /// (e.g. ConstantArrays, CallInst, PHINodes, ReturnInst etc), this memory 
  /// will be dynamically allocated and should be destroyed by the classes 
  /// virtual dtor.
  Use *OperandList;

  /// NumOperands - The number of values used by this User.
  ///
  unsigned NumOperands;

  void *operator new(size_t s, size_t) {
    return ::operator new(s);
  }
  User(const Type *Ty, unsigned vty, Use *OpList, unsigned NumOps)
    : Value(Ty, vty), OperandList(OpList), NumOperands(NumOps) {}

public:
  Value *getOperand(unsigned i) const {
    assert(i < NumOperands && "getOperand() out of range!");
    return OperandList[i];
  }
  void setOperand(unsigned i, Value *Val) {
    assert(i < NumOperands && "setOperand() out of range!");
    OperandList[i] = Val;
  }
  unsigned getNumOperands() const { return NumOperands; }

  // ---------------------------------------------------------------------------
  // Operand Iterator interface...
  //
  typedef Use*       op_iterator;
  typedef const Use* const_op_iterator;

  inline op_iterator       op_begin()       { return OperandList; }
  inline const_op_iterator op_begin() const { return OperandList; }
  inline op_iterator       op_end()         { return OperandList+NumOperands; }
  inline const_op_iterator op_end()   const { return OperandList+NumOperands; }

  // dropAllReferences() - This function is in charge of "letting go" of all
  // objects that this User refers to.  This allows one to
  // 'delete' a whole class at a time, even though there may be circular
  // references... first all references are dropped, and all use counts go to
  // zero.  Then everything is delete'd for real.  Note that no operations are
  // valid on an object that has "dropped all references", except operator
  // delete.
  //
  void dropAllReferences() {
    Use *OL = OperandList;
    for (unsigned i = 0, e = NumOperands; i != e; ++i)
      OL[i].set(0);
  }

  /// replaceUsesOfWith - Replaces all references to the "From" definition with
  /// references to the "To" definition.
  ///
  void replaceUsesOfWith(Value *From, Value *To);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const User *) { return true; }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) || isa<Constant>(V);
  }
};

template<> struct simplify_type<User::op_iterator> {
  typedef Value* SimpleType;

  static SimpleType getSimplifiedValue(const User::op_iterator &Val) {
    return static_cast<SimpleType>(Val->get());
  }
};

template<> struct simplify_type<const User::op_iterator>
  : public simplify_type<User::op_iterator> {};

template<> struct simplify_type<User::const_op_iterator> {
  typedef Value* SimpleType;

  static SimpleType getSimplifiedValue(const User::const_op_iterator &Val) {
    return static_cast<SimpleType>(Val->get());
  }
};

template<> struct simplify_type<const User::const_op_iterator>
  : public simplify_type<User::const_op_iterator> {};


// value_use_iterator::getOperandNo - Requires the definition of the User class.
template<typename UserTy>
unsigned value_use_iterator<UserTy>::getOperandNo() const {
  return U - U->getUser()->op_begin();
}

} // End llvm namespace

#endif
