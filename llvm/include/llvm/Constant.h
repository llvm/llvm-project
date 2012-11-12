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
  class APInt;

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
  void operator=(const Constant &) LLVM_DELETED_FUNCTION;
  Constant(const Constant &) LLVM_DELETED_FUNCTION;
  virtual void anchor();
  
protected:
  Constant(Type *ty, ValueTy vty, Use *Ops, unsigned NumOps)
    : User(ty, vty, Ops, NumOps) {}

  void destroyConstantImpl();
public:
  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  bool isNullValue() const;

  /// isAllOnesValue - Return true if this is the value that would be returned by
  /// getAllOnesValue.
  bool isAllOnesValue() const;

  /// isNegativeZeroValue - Return true if the value is what would be returned 
  /// by getZeroValueForNegation.
  bool isNegativeZeroValue() const;

  /// canTrap - Return true if evaluation of this constant could trap.  This is
  /// true for things like constant expressions that could divide by zero.
  bool canTrap() const;

  /// isConstantUsed - Return true if the constant has users other than constant
  /// exprs and other dangling things.
  bool isConstantUsed() const;
  
  enum PossibleRelocationsTy {
    NoRelocation = 0,
    LocalRelocation = 1,
    GlobalRelocations = 2
  };
  
  /// getRelocationInfo - This method classifies the entry according to
  /// whether or not it may generate a relocation entry.  This must be
  /// conservative, so if it might codegen to a relocatable entry, it should say
  /// so.  The return values are:
  /// 
  ///  NoRelocation: This constant pool entry is guaranteed to never have a
  ///     relocation applied to it (because it holds a simple constant like
  ///     '4').
  ///  LocalRelocation: This entry has relocations, but the entries are
  ///     guaranteed to be resolvable by the static linker, so the dynamic
  ///     linker will never see them.
  ///  GlobalRelocations: This entry may have arbitrary relocations.
  ///
  /// FIXME: This really should not be in VMCore.
  PossibleRelocationsTy getRelocationInfo() const;
  
  /// getAggregateElement - For aggregates (struct/array/vector) return the
  /// constant that corresponds to the specified element if possible, or null if
  /// not.  This can return null if the element index is a ConstantExpr, or if
  /// 'this' is a constant expr.
  Constant *getAggregateElement(unsigned Elt) const;
  Constant *getAggregateElement(Constant *Elt) const;
  
  /// destroyConstant - Called if some element of this constant is no longer
  /// valid.  At this point only other constants may be on the use_list for this
  /// constant.  Any constants on our Use list must also be destroy'd.  The
  /// implementation must be sure to remove the constant from the list of
  /// available cached constants.  Implementations should call
  /// destroyConstantImpl as the last thing they do, to destroy all users and
  /// delete this.
  virtual void destroyConstant() { llvm_unreachable("Not reached!"); }

  //// Methods for support type inquiry through isa, cast, and dyn_cast:
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
    llvm_unreachable("Constants that do not have operands cannot be using "
                     "'From'!");
  }

  static Constant *getNullValue(Type* Ty);

  /// @returns the value for an integer or vector of integer constant of the
  /// given type that has all its bits set to true.
  /// @brief Get the all ones value
  static Constant *getAllOnesValue(Type* Ty);

  /// getIntegerValue - Return the value for an integer or pointer constant,
  /// or a vector thereof, with the given scalar value.
  static Constant *getIntegerValue(Type* Ty, const APInt &V);
  
  /// removeDeadConstantUsers - If there are any dead constant users dangling
  /// off of this constant, remove them.  This method is useful for clients
  /// that want to check to see if a global is unused, but don't want to deal
  /// with potentially dead constants hanging off of the globals.
  void removeDeadConstantUsers() const;
};

} // End llvm namespace

#endif
