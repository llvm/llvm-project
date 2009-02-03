//===-- llvm/Constant.h - Constant class definition -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Constant class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CONSTANT_H
#define LLVM_CONSTANT_H

#include "llvm/User.h"

namespace llvm {
  template<typename T> class SmallVectorImpl;

/// This is an important base class in LLVM. It provides the common facilities
/// of all constant values in an LLVM program. A constant is a value that is
/// immutable at runtime. Functions are constants because their address is
/// immutable. Same with global variables. 
/// 
/// All constants share the capabilities provided in this class. All constants
/// can have a null value. They can have an operand list. Constants can be
/// simple (integer and floating point values), complex (arrays and structures),
/// or expression based (computations yielding a constant value composed of 
/// only certain operators and other constant values).
/// 
/// Note that Constants are immutable (once created they never change) 
/// and are fully shared by structural equivalence.  This means that two 
/// structurally equivalent constants will always have the same address.  
/// Constants are created on demand as needed and never deleted: thus clients 
/// don't have to worry about the lifetime of the objects.
/// @brief LLVM Constant Representation
class Constant : public User {
  void operator=(const Constant &);     // Do not implement
  Constant(const Constant &);           // Do not implement
protected:
  Constant(const Type *ty, ValueTy vty, Use *Ops, unsigned NumOps)
    : User(ty, vty, Ops, NumOps) {}

  void destroyConstantImpl();
public:
  /// Static constructor to get a '0' constant of arbitrary type...
  ///
  static Constant *getNullValue(const Type *Ty);

  /// Static constructor to get a '-1' constant.  This supports integers and
  /// vectors.
  ///
  static Constant *getAllOnesValue(const Type *Ty);
  
  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const = 0;

  /// canTrap - Return true if evaluation of this constant could trap.  This is
  /// true for things like constant expressions that could divide by zero.
  bool canTrap() const;

  /// ContaintsRelocations - Return true if the constant value contains
  /// relocations which cannot be resolved at compile time.
  bool ContainsRelocations() const;

  // Specialize get/setOperand for Constants as their operands are always
  // constants as well.
  Constant *getOperand(unsigned i) {
    return static_cast<Constant*>(User::getOperand(i));
  }
  const Constant *getOperand(unsigned i) const {
    return static_cast<const Constant*>(User::getOperand(i));
  }
  void setOperand(unsigned i, Constant *C) {
    User::setOperand(i, C);
  }
  
  /// getVectorElements - This method, which is only valid on constant of vector
  /// type, returns the elements of the vector in the specified smallvector.
  /// This handles breaking down a vector undef into undef elements, etc.  For
  /// constant exprs and other cases we can't handle, we return an empty vector.
  void getVectorElements(SmallVectorImpl<Constant*> &Elts) const;

  /// destroyConstant - Called if some element of this constant is no longer
  /// valid.  At this point only other constants may be on the use_list for this
  /// constant.  Any constants on our Use list must also be destroy'd.  The
  /// implementation must be sure to remove the constant from the list of
  /// available cached constants.  Implementations should call
  /// destroyConstantImpl as the last thing they do, to destroy all users and
  /// delete this.
  virtual void destroyConstant() { assert(0 && "Not reached!"); }

  //// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Constant *) { return true; }
  static inline bool classof(const GlobalValue *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueID() >= ConstantFirstVal &&
           V->getValueID() <= ConstantLastVal;
  }

  /// replaceUsesOfWithOnConstant - This method is a special form of
  /// User::replaceUsesOfWith (which does not work on constants) that does work
  /// on constants.  Basically this method goes through the trouble of building
  /// a new constant that is equivalent to the current one, with all uses of
  /// From replaced with uses of To.  After this construction is completed, all
  /// of the users of 'this' are replaced to use the new constant, and then
  /// 'this' is deleted.  In general, you should not call this method, instead,
  /// use Value::replaceAllUsesWith, which automatically dispatches to this
  /// method as needed.
  ///
  virtual void replaceUsesOfWithOnConstant(Value *, Value *, Use *) {
    // Provide a default implementation for constants (like integers) that
    // cannot use any other values.  This cannot be called at runtime, but needs
    // to be here to avoid link errors.
    assert(getNumOperands() == 0 && "replaceUsesOfWithOnConstant must be "
           "implemented for all constants that have operands!");
    assert(0 && "Constants that do not have operands cannot be using 'From'!");
  }
};

} // End llvm namespace

#endif
