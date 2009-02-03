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

/// OperandTraits - Compile-time customization of
/// operand-related allocators and accessors
/// for use of the User class
template <class>
struct OperandTraits;

class User;

/// OperandTraits<User> - specialization to User
template <>
struct OperandTraits<User> {
  static inline Use *op_begin(User*);
  static inline Use *op_end(User*);
  static inline unsigned operands(const User*);
  template <class U>
  struct Layout {
    typedef U overlay;
  };
  static inline void *allocate(unsigned);
};

class User : public Value {
  User(const User &);             // Do not implement
  void *operator new(size_t);     // Do not implement
  template <unsigned>
  friend struct HungoffOperandTraits;
protected:
  /// OperandList - This is a pointer to the array of Users for this operand.
  /// For nodes of fixed arity (e.g. a binary operator) this array will live
  /// prefixed to the derived class.  For nodes of resizable variable arity
  /// (e.g. PHINodes, SwitchInst etc.), this memory will be dynamically
  /// allocated and should be destroyed by the classes' 
  /// virtual dtor.
  Use *OperandList;

  /// NumOperands - The number of values used by this User.
  ///
  unsigned NumOperands;

  void *operator new(size_t s, unsigned Us);
  User(const Type *ty, unsigned vty, Use *OpList, unsigned NumOps)
    : Value(ty, vty), OperandList(OpList), NumOperands(NumOps) {}
  Use *allocHungoffUses(unsigned) const;
  void dropHungoffUses(Use *U) {
    if (OperandList == U) {
      OperandList = 0;
      NumOperands = 0;
    }
    Use::zap(U, U->getImpliedUser(), true);
  }
public:
  ~User() {
    Use::zap(OperandList, OperandList + NumOperands);
  }
  /// operator delete - free memory allocated for User and Use objects
  void operator delete(void *Usr);
  /// placement delete - required by std, but never called.
  void operator delete(void*, unsigned) {
    assert(0 && "Constructor throws?");
  }
  template <unsigned Idx> Use &Op() {
    return OperandTraits<User>::op_begin(this)[Idx];
  }
  template <unsigned Idx> const Use &Op() const {
    return OperandTraits<User>::op_begin(const_cast<User*>(this))[Idx];
  }
  Value *getOperand(unsigned i) const {
    assert(i < NumOperands && "getOperand() out of range!");
    return OperandList[i];
  }
  void setOperand(unsigned i, Value *Val) {
    assert(i < NumOperands && "setOperand() out of range!");
    assert((!isa<Constant>((const Value*)this) ||
            isa<GlobalValue>((const Value*)this)) &&
           "Cannot mutate a constant with setOperand!");
    OperandList[i] = Val;
  }
  const Use &getOperandUse(unsigned i) const {
    assert(i < NumOperands && "getOperand() out of range!");
    return OperandList[i];
  }
  Use &getOperandUse(unsigned i) {
    assert(i < NumOperands && "getOperand() out of range!");
    return OperandList[i];
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
  // references...  First all references are dropped, and all use counts go to
  // zero.  Then everything is deleted for real.  Note that no operations are
  // valid on an object that has "dropped all references", except operator
  // delete.
  //
  void dropAllReferences() {
    for (op_iterator i = op_begin(), e = op_end(); i != e; ++i)
      i->set(0);
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

inline Use *OperandTraits<User>::op_begin(User *U) {
  return U->op_begin();
}

inline Use *OperandTraits<User>::op_end(User *U) {
  return U->op_end();
}

inline unsigned OperandTraits<User>::operands(const User *U) {
  return U->getNumOperands();
}

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
