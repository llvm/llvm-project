//===-- llvm/User.h - User class definition ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class defines the interface that one who uses a Value must implement.
// Each instance of the Value class keeps track of what User's have handles
// to it.
//
//  * Instructions are the largest class of Users.
//  * Constants may be users of other constants (think arrays and stuff)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_USER_H
#define LLVM_USER_H

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Value.h"

namespace llvm {

/// OperandTraits - Compile-time customization of
/// operand-related allocators and accessors
/// for use of the User class
template <class>
struct OperandTraits;

class User : public Value {
  User(const User &) LLVM_DELETED_FUNCTION;
  void *operator new(size_t) LLVM_DELETED_FUNCTION;
  template <unsigned>
  friend struct HungoffOperandTraits;
  virtual void anchor();
protected:
  /// OperandList - This is a pointer to the array of Uses for this User.
  /// For nodes of fixed arity (e.g. a binary operator) this array will live
  /// prefixed to some derived class instance.  For nodes of resizable variable
  /// arity (e.g. PHINodes, SwitchInst etc.), this memory will be dynamically
  /// allocated and should be destroyed by the classes' virtual dtor.
  Use *OperandList;

  /// NumOperands - The number of values used by this User.
  ///
  unsigned NumOperands;

  void *operator new(size_t s, unsigned Us);
  User(Type *ty, unsigned vty, Use *OpList, unsigned NumOps)
    : Value(ty, vty), OperandList(OpList), NumOperands(NumOps) {}
  Use *allocHungoffUses(unsigned) const;
  void dropHungoffUses() {
    Use::zap(OperandList, OperandList + NumOperands, true);
    OperandList = 0;
    // Reset NumOperands so User::operator delete() does the right thing.
    NumOperands = 0;
  }
public:
  ~User() {
    Use::zap(OperandList, OperandList + NumOperands);
  }
  /// operator delete - free memory allocated for User and Use objects
  void operator delete(void *Usr);
  /// placement delete - required by std, but never called.
  void operator delete(void*, unsigned) {
    llvm_unreachable("Constructor throws?");
  }
  /// placement delete - required by std, but never called.
  void operator delete(void*, unsigned, bool) {
    llvm_unreachable("Constructor throws?");
  }
protected:
  template <int Idx, typename U> static Use &OpFrom(const U *that) {
    return Idx < 0
      ? OperandTraits<U>::op_end(const_cast<U*>(that))[Idx]
      : OperandTraits<U>::op_begin(const_cast<U*>(that))[Idx];
  }
  template <int Idx> Use &Op() {
    return OpFrom<Idx>(this);
  }
  template <int Idx> const Use &Op() const {
    return OpFrom<Idx>(this);
  }
public:
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
    assert(i < NumOperands && "getOperandUse() out of range!");
    return OperandList[i];
  }
  Use &getOperandUse(unsigned i) {
    assert(i < NumOperands && "getOperandUse() out of range!");
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

  /// Convenience iterator for directly iterating over the Values in the
  /// OperandList
  class value_op_iterator : public std::iterator<std::forward_iterator_tag,
                                                 Value*> {
    op_iterator OI;
  public:
    explicit value_op_iterator(Use *U) : OI(U) {}

    bool operator==(const value_op_iterator &x) const {
      return OI == x.OI;
    }
    bool operator!=(const value_op_iterator &x) const {
      return !operator==(x);
    }

    /// Iterator traversal: forward iteration only
    value_op_iterator &operator++() {          // Preincrement
      ++OI;
      return *this;
    }
    value_op_iterator operator++(int) {        // Postincrement
      value_op_iterator tmp = *this; ++*this; return tmp;
    }

    /// Retrieve a pointer to the current Value.
    Value *operator*() const {
      return *OI;
    }

    Value *operator->() const { return operator*(); }
  };

  inline value_op_iterator value_op_begin() {
    return value_op_iterator(op_begin());
  }
  inline value_op_iterator value_op_end() {
    return value_op_iterator(op_end());
  }

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
