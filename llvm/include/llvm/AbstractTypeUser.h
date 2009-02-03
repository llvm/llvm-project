//===-- llvm/AbstractTypeUser.h - AbstractTypeUser Interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the AbstractTypeUser class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ABSTRACT_TYPE_USER_H
#define LLVM_ABSTRACT_TYPE_USER_H

#if !defined(LLVM_TYPE_H) && !defined(LLVM_VALUE_H)
#error Do not include this file directly.  Include Type.h instead.
#error Some versions of GCC (e.g. 3.4 and 4.1) can not handle the inlined method
#error PATypeHolder::dropRef() correctly otherwise.
#endif

// This is the "master" include for <cassert> Whether this file needs it or not,
// it must always include <cassert> for the files which include
// llvm/AbstractTypeUser.h
//
// In this way, most every LLVM source file will have access to the assert()
// macro without having to #include <cassert> directly.
//
#include <cassert>

namespace llvm {

class Type;
class DerivedType;
template<typename T> struct simplify_type;

/// The AbstractTypeUser class is an interface to be implemented by classes who
/// could possibly use an abstract type.  Abstract types are denoted by the
/// isAbstract flag set to true in the Type class.  These are classes that
/// contain an Opaque type in their structure somewhere.
///
/// Classes must implement this interface so that they may be notified when an
/// abstract type is resolved.  Abstract types may be resolved into more 
/// concrete types through: linking, parsing, and bitcode reading.  When this 
/// happens, all of the users of the type must be updated to reference the new,
/// more concrete type.  They are notified through the AbstractTypeUser 
/// interface.
///
/// In addition to this, AbstractTypeUsers must keep the use list of the
/// potentially abstract type that they reference up-to-date.  To do this in a
/// nice, transparent way, the PATypeHandle class is used to hold "Potentially
/// Abstract Types", and keep the use list of the abstract types up-to-date.
/// @brief LLVM Abstract Type User Representation
class AbstractTypeUser {
protected:
  virtual ~AbstractTypeUser();                        // Derive from me
public:

  /// refineAbstractType - The callback method invoked when an abstract type is
  /// resolved to another type.  An object must override this method to update
  /// its internal state to reference NewType instead of OldType.
  ///
  virtual void refineAbstractType(const DerivedType *OldTy,
                                  const Type *NewTy) = 0;

  /// The other case which AbstractTypeUsers must be aware of is when a type
  /// makes the transition from being abstract (where it has clients on it's
  /// AbstractTypeUsers list) to concrete (where it does not).  This method
  /// notifies ATU's when this occurs for a type.
  ///
  virtual void typeBecameConcrete(const DerivedType *AbsTy) = 0;

  // for debugging...
  virtual void dump() const = 0;
};


/// PATypeHandle - Handle to a Type subclass.  This class is used to keep the
/// use list of abstract types up-to-date.
///
class PATypeHandle {
  const Type *Ty;
  AbstractTypeUser * const User;

  // These functions are defined at the bottom of Type.h.  See the comment there
  // for justification.
  void addUser();
  void removeUser();
public:
  // ctor - Add use to type if abstract.  Note that Ty must not be null
  inline PATypeHandle(const Type *ty, AbstractTypeUser *user)
    : Ty(ty), User(user) {
    addUser();
  }

  // ctor - Add use to type if abstract.
  inline PATypeHandle(const PATypeHandle &T) : Ty(T.Ty), User(T.User) {
    addUser();
  }

  // dtor - Remove reference to type...
  inline ~PATypeHandle() { removeUser(); }

  // Automatic casting operator so that the handle may be used naturally
  inline operator Type *() const { return const_cast<Type*>(Ty); }
  inline Type *get() const { return const_cast<Type*>(Ty); }

  // operator= - Allow assignment to handle
  inline Type *operator=(const Type *ty) {
    if (Ty != ty) {   // Ensure we don't accidentally drop last ref to Ty
      removeUser();
      Ty = ty;
      addUser();
    }
    return get();
  }

  // operator= - Allow assignment to handle
  inline const Type *operator=(const PATypeHandle &T) {
    return operator=(T.Ty);
  }

  inline bool operator==(const Type *ty) {
    return Ty == ty;
  }

  // operator-> - Allow user to dereference handle naturally...
  inline const Type *operator->() const { return Ty; }
};


/// PATypeHolder - Holder class for a potentially abstract type.  This uses
/// efficient union-find techniques to handle dynamic type resolution.  Unless
/// you need to do custom processing when types are resolved, you should always
/// use PATypeHolders in preference to PATypeHandles.
///
class PATypeHolder {
  mutable const Type *Ty;
public:
  PATypeHolder(const Type *ty) : Ty(ty) {
    addRef();
  }
  PATypeHolder(const PATypeHolder &T) : Ty(T.Ty) {
    addRef();
  }

  ~PATypeHolder() { dropRef(); }

  operator Type *() const { return get(); }
  Type *get() const;

  // operator-> - Allow user to dereference handle naturally...
  Type *operator->() const { return get(); }

  // operator= - Allow assignment to handle
  Type *operator=(const Type *ty) {
    if (Ty != ty) {   // Don't accidentally drop last ref to Ty.
      dropRef();
      Ty = ty;
      addRef();
    }
    return get();
  }
  Type *operator=(const PATypeHolder &H) {
    return operator=(H.Ty);
  }

  /// getRawType - This should only be used to implement the vmcore library.
  ///
  const Type *getRawType() const { return Ty; }

private:
  void addRef();
  void dropRef();
};

// simplify_type - Allow clients to treat uses just like values when using
// casting operators.
template<> struct simplify_type<PATypeHolder> {
  typedef const Type* SimpleType;
  static SimpleType getSimplifiedValue(const PATypeHolder &Val) {
    return static_cast<SimpleType>(Val.get());
  }
};
template<> struct simplify_type<const PATypeHolder> {
  typedef const Type* SimpleType;
  static SimpleType getSimplifiedValue(const PATypeHolder &Val) {
    return static_cast<SimpleType>(Val.get());
  }
};
  
} // End llvm namespace

#endif
