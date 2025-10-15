//===-- llvm/Argument.h - Definition of the Argument class ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Argument class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_ARGUMENT_H
#define LLVM_IR_ARGUMENT_H

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Compiler.h"
#include <optional>

namespace llvm {

class ConstantRange;

/// This class represents an incoming formal argument to a Function. A formal
/// argument, since it is ``formal'', does not contain an actual value but
/// instead represents the type, argument number, and attributes of an argument
/// for a specific function. When used in the body of said function, the
/// argument of course represents the value of the actual argument that the
/// function was called with.
class Argument final : public Value {
  Function *Parent;
  unsigned ArgNo;

  friend class Function;
  void setParent(Function *parent);

public:
  /// Argument constructor.
  LLVM_ABI explicit Argument(Type *Ty, const Twine &Name = "",
                             Function *F = nullptr, unsigned ArgNo = 0);

  inline const Function *getParent() const { return Parent; }
  inline       Function *getParent()       { return Parent; }

  /// Return the index of this formal argument in its containing function.
  ///
  /// For example in "void foo(int a, float b)" a is 0 and b is 1.
  unsigned getArgNo() const {
    assert(Parent && "can't get number of unparented arg");
    return ArgNo;
  }

  /// Return true if this argument has the nonnull attribute. Also returns true
  /// if at least one byte is known to be dereferenceable and the pointer is in
  /// addrspace(0).
  /// If AllowUndefOrPoison is true, respect the semantics of nonnull attribute
  /// and return true even if the argument can be undef or poison.
  LLVM_ABI bool hasNonNullAttr(bool AllowUndefOrPoison = true) const;

  /// If this argument has the dereferenceable attribute, return the number of
  /// bytes known to be dereferenceable. Otherwise, zero is returned.
  LLVM_ABI uint64_t getDereferenceableBytes() const;

  /// If this argument has the dereferenceable_or_null attribute, return the
  /// number of bytes known to be dereferenceable. Otherwise, zero is returned.
  LLVM_ABI uint64_t getDereferenceableOrNullBytes() const;

  /// If this argument has nofpclass attribute, return the mask representing
  /// disallowed floating-point values. Otherwise, fcNone is returned.
  LLVM_ABI FPClassTest getNoFPClass() const;

  /// If this argument has a range attribute, return the value range of the
  /// argument. Otherwise, std::nullopt is returned.
  LLVM_ABI std::optional<ConstantRange> getRange() const;

  /// Return true if this argument has the byval attribute.
  LLVM_ABI bool hasByValAttr() const;

  /// Return true if this argument has the dead_on_return attribute.
  LLVM_ABI bool hasDeadOnReturnAttr() const;

  /// Return true if this argument has the byref attribute.
  LLVM_ABI bool hasByRefAttr() const;

  /// Return true if this argument has the swiftself attribute.
  LLVM_ABI bool hasSwiftSelfAttr() const;

  /// Return true if this argument has the swifterror attribute.
  LLVM_ABI bool hasSwiftErrorAttr() const;

  /// Return true if this argument has the byval, inalloca, or preallocated
  /// attribute. These attributes represent arguments being passed by value,
  /// with an associated copy between the caller and callee
  LLVM_ABI bool hasPassPointeeByValueCopyAttr() const;

  /// If this argument satisfies has hasPassPointeeByValueAttr, return the
  /// in-memory ABI size copied to the stack for the call. Otherwise, return 0.
  LLVM_ABI uint64_t getPassPointeeByValueCopySize(const DataLayout &DL) const;

  /// Return true if this argument has the byval, sret, inalloca, preallocated,
  /// or byref attribute. These attributes represent arguments being passed by
  /// value (which may or may not involve a stack copy)
  LLVM_ABI bool hasPointeeInMemoryValueAttr() const;

  /// If hasPointeeInMemoryValueAttr returns true, the in-memory ABI type is
  /// returned. Otherwise, nullptr.
  LLVM_ABI Type *getPointeeInMemoryValueType() const;

  /// If this is a byval or inalloca argument, return its alignment.
  /// FIXME: Remove this function once transition to Align is over.
  /// Use getParamAlign() instead.
  LLVM_ABI LLVM_DEPRECATED("Use getParamAlign() instead",
                           "getParamAlign") uint64_t getParamAlignment() const;

  /// If this is a byval or inalloca argument, return its alignment.
  LLVM_ABI MaybeAlign getParamAlign() const;

  LLVM_ABI MaybeAlign getParamStackAlign() const;

  /// If this is a byval argument, return its type.
  LLVM_ABI Type *getParamByValType() const;

  /// If this is an sret argument, return its type.
  LLVM_ABI Type *getParamStructRetType() const;

  /// If this is a byref argument, return its type.
  LLVM_ABI Type *getParamByRefType() const;

  /// If this is an inalloca argument, return its type.
  LLVM_ABI Type *getParamInAllocaType() const;

  /// Return true if this argument has the nest attribute.
  LLVM_ABI bool hasNestAttr() const;

  /// Return true if this argument has the noalias attribute.
  LLVM_ABI bool hasNoAliasAttr() const;

  /// Return true if this argument has the nocapture attribute.
  LLVM_ABI bool hasNoCaptureAttr() const;

  /// Return true if this argument has the nofree attribute.
  LLVM_ABI bool hasNoFreeAttr() const;

  /// Return true if this argument has the sret attribute.
  LLVM_ABI bool hasStructRetAttr() const;

  /// Return true if this argument has the inreg attribute.
  LLVM_ABI bool hasInRegAttr() const;

  /// Return true if this argument has the returned attribute.
  LLVM_ABI bool hasReturnedAttr() const;

  /// Return true if this argument has the readonly or readnone attribute.
  LLVM_ABI bool onlyReadsMemory() const;

  /// Return true if this argument has the inalloca attribute.
  LLVM_ABI bool hasInAllocaAttr() const;

  /// Return true if this argument has the preallocated attribute.
  LLVM_ABI bool hasPreallocatedAttr() const;

  /// Return true if this argument has the zext attribute.
  LLVM_ABI bool hasZExtAttr() const;

  /// Return true if this argument has the sext attribute.
  LLVM_ABI bool hasSExtAttr() const;

  /// Add attributes to an argument.
  LLVM_ABI void addAttrs(AttrBuilder &B);

  LLVM_ABI void addAttr(Attribute::AttrKind Kind);

  LLVM_ABI void addAttr(Attribute Attr);

  /// Remove attributes from an argument.
  LLVM_ABI void removeAttr(Attribute::AttrKind Kind);

  LLVM_ABI void removeAttrs(const AttributeMask &AM);

  /// Check if an argument has a given attribute.
  LLVM_ABI bool hasAttribute(Attribute::AttrKind Kind) const;

  LLVM_ABI bool hasAttribute(StringRef Kind) const;

  LLVM_ABI Attribute getAttribute(Attribute::AttrKind Kind) const;

  LLVM_ABI AttributeSet getAttributes() const;

  /// Method for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Value *V) {
    return V->getValueID() == ArgumentVal;
  }
};

} // End llvm namespace

#endif
