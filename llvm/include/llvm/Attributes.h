//===-- llvm/Attributes.h - Container for Attributes ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the simple types necessary to represent the
// attributes associated with functions and their calls.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ATTRIBUTES_H
#define LLVM_ATTRIBUTES_H

#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include <cassert>
#include <string>

namespace llvm {

class AttrBuilder;
class AttributesImpl;
class LLVMContext;
class Type;

/// Attributes - A bitset of attributes.
class Attributes {
public:
  /// Function parameters and results can have attributes to indicate how they
  /// should be treated by optimizations and code generation. This enumeration
  /// lists the attributes that can be associated with parameters, function
  /// results or the function itself.
  ///
  /// Note that uwtable is about the ABI or the user mandating an entry in the
  /// unwind table. The nounwind attribute is about an exception passing by the
  /// function.
  ///
  /// In a theoretical system that uses tables for profiling and sjlj for
  /// exceptions, they would be fully independent. In a normal system that uses
  /// tables for both, the semantics are:
  ///
  /// nil                = Needs an entry because an exception might pass by.
  /// nounwind           = No need for an entry
  /// uwtable            = Needs an entry because the ABI says so and because
  ///                      an exception might pass by.
  /// uwtable + nounwind = Needs an entry because the ABI says so.

  enum AttrVal {
    // IR-Level Attributes
    None,                  ///< No attributes have been set
    AddressSafety,         ///< Address safety checking is on.
    Alignment,             ///< Alignment of parameter (5 bits)
                           ///< stored as log2 of alignment with +1 bias
                           ///< 0 means unaligned different from align 1
    AlwaysInline,          ///< inline=always
    ByVal,                 ///< Pass structure by value
    InlineHint,            ///< Source said inlining was desirable
    InReg,                 ///< Force argument to be passed in register
    MinSize,               ///< Function must be optimized for size first
    Naked,                 ///< Naked function
    Nest,                  ///< Nested function static chain
    NoAlias,               ///< Considered to not alias after call
    NoCapture,             ///< Function creates no aliases of pointer
    NoImplicitFloat,       ///< Disable implicit floating point insts
    NoInline,              ///< inline=never
    NonLazyBind,           ///< Function is called early and/or
                           ///< often, so lazy binding isn't worthwhile
    NoRedZone,             ///< Disable redzone
    NoReturn,              ///< Mark the function as not returning
    NoUnwind,              ///< Function doesn't unwind stack
    OptimizeForSize,       ///< opt_size
    ReadNone,              ///< Function does not access memory
    ReadOnly,              ///< Function only reads from memory
    ReturnsTwice,          ///< Function can return twice
    SExt,                  ///< Sign extended before/after call
    StackAlignment,        ///< Alignment of stack for function (3 bits)
                           ///< stored as log2 of alignment with +1 bias 0
                           ///< means unaligned (different from
                           ///< alignstack={1))
    StackProtect,          ///< Stack protection.
    StackProtectReq,       ///< Stack protection required.
    StructRet,             ///< Hidden pointer to structure to return
    UWTable,               ///< Function must be in a unwind table
    ZExt                   ///< Zero extended before/after call
  };
private:
  AttributesImpl *Attrs;
  Attributes(AttributesImpl *A) : Attrs(A) {}
public:
  Attributes() : Attrs(0) {}
  Attributes(const Attributes &A) : Attrs(A.Attrs) {}
  Attributes &operator=(const Attributes &A) {
    Attrs = A.Attrs;
    return *this;
  }

  /// get - Return a uniquified Attributes object. This takes the uniquified
  /// value from the Builder and wraps it in the Attributes class.
  static Attributes get(LLVMContext &Context, ArrayRef<AttrVal> Vals);
  static Attributes get(LLVMContext &Context, AttrBuilder &B);

  /// @brief Return true if the attribute is present.
  bool hasAttribute(AttrVal Val) const;

  /// @brief Return true if attributes exist
  bool hasAttributes() const;

  /// @brief Return true if the attributes are a non-null intersection.
  bool hasAttributes(const Attributes &A) const;

  /// @brief Returns the alignment field of an attribute as a byte alignment
  /// value.
  unsigned getAlignment() const;

  /// @brief Returns the stack alignment field of an attribute as a byte
  /// alignment value.
  unsigned getStackAlignment() const;

  /// @brief Parameter attributes that do not apply to vararg call arguments.
  bool hasIncompatibleWithVarArgsAttrs() const {
    return hasAttribute(Attributes::StructRet);
  }

  /// @brief Attributes that only apply to function parameters.
  bool hasParameterOnlyAttrs() const {
    return hasAttribute(Attributes::ByVal) ||
      hasAttribute(Attributes::Nest) ||
      hasAttribute(Attributes::StructRet) ||
      hasAttribute(Attributes::NoCapture);
  }

  /// @brief Attributes that may be applied to the function itself.  These cannot
  /// be used on return values or function parameters.
  bool hasFunctionOnlyAttrs() const {
    return hasAttribute(Attributes::NoReturn) ||
      hasAttribute(Attributes::NoUnwind) ||
      hasAttribute(Attributes::ReadNone) ||
      hasAttribute(Attributes::ReadOnly) ||
      hasAttribute(Attributes::NoInline) ||
      hasAttribute(Attributes::AlwaysInline) ||
      hasAttribute(Attributes::OptimizeForSize) ||
      hasAttribute(Attributes::StackProtect) ||
      hasAttribute(Attributes::StackProtectReq) ||
      hasAttribute(Attributes::NoRedZone) ||
      hasAttribute(Attributes::NoImplicitFloat) ||
      hasAttribute(Attributes::Naked) ||
      hasAttribute(Attributes::InlineHint) ||
      hasAttribute(Attributes::StackAlignment) ||
      hasAttribute(Attributes::UWTable) ||
      hasAttribute(Attributes::NonLazyBind) ||
      hasAttribute(Attributes::ReturnsTwice) ||
      hasAttribute(Attributes::AddressSafety) ||
      hasAttribute(Attributes::MinSize);
  }

  bool operator==(const Attributes &A) const {
    return Attrs == A.Attrs;
  }
  bool operator!=(const Attributes &A) const {
    return Attrs != A.Attrs;
  }

  uint64_t Raw() const;

  /// @brief Which attributes cannot be applied to a type.
  static Attributes typeIncompatible(Type *Ty);

  /// encodeLLVMAttributesForBitcode - This returns an integer containing an
  /// encoding of all the LLVM attributes found in the given attribute bitset.
  /// Any change to this encoding is a breaking change to bitcode compatibility.
  static uint64_t encodeLLVMAttributesForBitcode(Attributes Attrs);

  /// decodeLLVMAttributesForBitcode - This returns an attribute bitset
  /// containing the LLVM attributes that have been decoded from the given
  /// integer.  This function must stay in sync with
  /// 'encodeLLVMAttributesForBitcode'.
  static Attributes decodeLLVMAttributesForBitcode(LLVMContext &C,
                                                   uint64_t EncodedAttrs);

  /// getAsString - The set of Attributes set in Attributes is converted to a
  /// string of equivalent mnemonics. This is, presumably, for writing out the
  /// mnemonics for the assembly writer.
  /// @brief Convert attribute bits to text
  std::string getAsString() const;
};

//===----------------------------------------------------------------------===//
/// AttrBuilder - This class is used in conjunction with the Attributes::get
/// method to create an Attributes object. The object itself is uniquified. The
/// Builder's value, however, is not. So this can be used as a quick way to test
/// for equality, presence of attributes, etc.
class AttrBuilder {
  uint64_t Bits;
public:
  AttrBuilder() : Bits(0) {}
  explicit AttrBuilder(uint64_t B) : Bits(B) {}
  AttrBuilder(const Attributes &A) : Bits(A.Raw()) {}
  AttrBuilder(const AttrBuilder &B) : Bits(B.Bits) {}

  void clear() { Bits = 0; }

  /// addAttribute - Add an attribute to the builder.
  AttrBuilder &addAttribute(Attributes::AttrVal Val);

  /// removeAttribute - Remove an attribute from the builder.
  AttrBuilder &removeAttribute(Attributes::AttrVal Val);

  /// addAttribute - Add the attributes from A to the builder.
  AttrBuilder &addAttributes(const Attributes &A);

  /// removeAttribute - Remove the attributes from A from the builder.
  AttrBuilder &removeAttributes(const Attributes &A);

  /// hasAttribute - Return true if the builder has the specified attribute.
  bool hasAttribute(Attributes::AttrVal A) const;

  /// hasAttributes - Return true if the builder has IR-level attributes.
  bool hasAttributes() const;

  /// hasAttributes - Return true if the builder has any attribute that's in the
  /// specified attribute.
  bool hasAttributes(const Attributes &A) const;

  /// hasAlignmentAttr - Return true if the builder has an alignment attribute.
  bool hasAlignmentAttr() const;

  /// getAlignment - Retrieve the alignment attribute, if it exists.
  uint64_t getAlignment() const;

  /// getStackAlignment - Retrieve the stack alignment attribute, if it exists.
  uint64_t getStackAlignment() const;

  /// addAlignmentAttr - This turns an int alignment (which must be a power of
  /// 2) into the form used internally in Attributes.
  AttrBuilder &addAlignmentAttr(unsigned Align);

  /// addStackAlignmentAttr - This turns an int stack alignment (which must be a
  /// power of 2) into the form used internally in Attributes.
  AttrBuilder &addStackAlignmentAttr(unsigned Align);

  /// addRawValue - Add the raw value to the internal representation.
  /// N.B. This should be used ONLY for decoding LLVM bitcode!
  AttrBuilder &addRawValue(uint64_t Val);

  /// @brief Remove attributes that are used on functions only.
  void removeFunctionOnlyAttrs() {
    removeAttribute(Attributes::NoReturn)
      .removeAttribute(Attributes::NoUnwind)
      .removeAttribute(Attributes::ReadNone)
      .removeAttribute(Attributes::ReadOnly)
      .removeAttribute(Attributes::NoInline)
      .removeAttribute(Attributes::AlwaysInline)
      .removeAttribute(Attributes::OptimizeForSize)
      .removeAttribute(Attributes::StackProtect)
      .removeAttribute(Attributes::StackProtectReq)
      .removeAttribute(Attributes::NoRedZone)
      .removeAttribute(Attributes::NoImplicitFloat)
      .removeAttribute(Attributes::Naked)
      .removeAttribute(Attributes::InlineHint)
      .removeAttribute(Attributes::StackAlignment)
      .removeAttribute(Attributes::UWTable)
      .removeAttribute(Attributes::NonLazyBind)
      .removeAttribute(Attributes::ReturnsTwice)
      .removeAttribute(Attributes::AddressSafety)
      .removeAttribute(Attributes::MinSize);
  }

  uint64_t Raw() const { return Bits; }

  bool operator==(const AttrBuilder &B) {
    return Bits == B.Bits;
  }
  bool operator!=(const AttrBuilder &B) {
    return Bits != B.Bits;
  }
};

//===----------------------------------------------------------------------===//
// AttributeWithIndex
//===----------------------------------------------------------------------===//

/// AttributeWithIndex - This is just a pair of values to associate a set of
/// attributes with an index.
struct AttributeWithIndex {
  Attributes Attrs;  ///< The attributes that are set, or'd together.
  unsigned Index;    ///< Index of the parameter for which the attributes apply.
                     ///< Index 0 is used for return value attributes.
                     ///< Index ~0U is used for function attributes.

  static AttributeWithIndex get(LLVMContext &C, unsigned Idx,
                                ArrayRef<Attributes::AttrVal> Attrs) {
    return get(Idx, Attributes::get(C, Attrs));
  }
  static AttributeWithIndex get(unsigned Idx, Attributes Attrs) {
    AttributeWithIndex P;
    P.Index = Idx;
    P.Attrs = Attrs;
    return P;
  }
};

//===----------------------------------------------------------------------===//
// AttrListPtr Smart Pointer
//===----------------------------------------------------------------------===//

class AttributeListImpl;

/// AttrListPtr - This class manages the ref count for the opaque
/// AttributeListImpl object and provides accessors for it.
class AttrListPtr {
public:
  enum AttrIndex {
    ReturnIndex = 0U,
    FunctionIndex = ~0U
  };
private:
  /// AttrList - The attributes that we are managing.  This can be null to
  /// represent the empty attributes list.
  AttributeListImpl *AttrList;
public:
  AttrListPtr() : AttrList(0) {}
  AttrListPtr(const AttrListPtr &P);
  const AttrListPtr &operator=(const AttrListPtr &RHS);
  ~AttrListPtr();

  //===--------------------------------------------------------------------===//
  // Attribute List Construction and Mutation
  //===--------------------------------------------------------------------===//

  /// get - Return a Attributes list with the specified parameters in it.
  static AttrListPtr get(ArrayRef<AttributeWithIndex> Attrs);

  /// addAttr - Add the specified attribute at the specified index to this
  /// attribute list.  Since attribute lists are immutable, this
  /// returns the new list.
  AttrListPtr addAttr(LLVMContext &C, unsigned Idx, Attributes Attrs) const;

  /// removeAttr - Remove the specified attribute at the specified index from
  /// this attribute list.  Since attribute lists are immutable, this
  /// returns the new list.
  AttrListPtr removeAttr(LLVMContext &C, unsigned Idx, Attributes Attrs) const;

  //===--------------------------------------------------------------------===//
  // Attribute List Accessors
  //===--------------------------------------------------------------------===//
  /// getParamAttributes - The attributes for the specified index are
  /// returned.
  Attributes getParamAttributes(unsigned Idx) const {
    return getAttributes(Idx);
  }

  /// getRetAttributes - The attributes for the ret value are
  /// returned.
  Attributes getRetAttributes() const {
    return getAttributes(ReturnIndex);
  }

  /// getFnAttributes - The function attributes are returned.
  Attributes getFnAttributes() const {
    return getAttributes(FunctionIndex);
  }

  /// paramHasAttr - Return true if the specified parameter index has the
  /// specified attribute set.
  bool paramHasAttr(unsigned Idx, Attributes Attr) const {
    return getAttributes(Idx).hasAttributes(Attr);
  }

  /// getParamAlignment - Return the alignment for the specified function
  /// parameter.
  unsigned getParamAlignment(unsigned Idx) const {
    return getAttributes(Idx).getAlignment();
  }

  /// hasAttrSomewhere - Return true if the specified attribute is set for at
  /// least one parameter or for the return value.
  bool hasAttrSomewhere(Attributes::AttrVal Attr) const;

  unsigned getNumAttrs() const;
  Attributes &getAttributesAtIndex(unsigned i) const;

  /// operator==/!= - Provide equality predicates.
  bool operator==(const AttrListPtr &RHS) const
  { return AttrList == RHS.AttrList; }
  bool operator!=(const AttrListPtr &RHS) const
  { return AttrList != RHS.AttrList; }

  //===--------------------------------------------------------------------===//
  // Attribute List Introspection
  //===--------------------------------------------------------------------===//

  /// getRawPointer - Return a raw pointer that uniquely identifies this
  /// attribute list.
  void *getRawPointer() const {
    return AttrList;
  }

  // Attributes are stored as a dense set of slots, where there is one
  // slot for each argument that has an attribute.  This allows walking over the
  // dense set instead of walking the sparse list of attributes.

  /// isEmpty - Return true if there are no attributes.
  ///
  bool isEmpty() const {
    return AttrList == 0;
  }

  /// getNumSlots - Return the number of slots used in this attribute list.
  /// This is the number of arguments that have an attribute set on them
  /// (including the function itself).
  unsigned getNumSlots() const;

  /// getSlot - Return the AttributeWithIndex at the specified slot.  This
  /// holds a index number plus a set of attributes.
  const AttributeWithIndex &getSlot(unsigned Slot) const;

  void dump() const;

private:
  explicit AttrListPtr(AttributeListImpl *L);

  /// getAttributes - The attributes for the specified index are
  /// returned.  Attributes for the result are denoted with Idx = 0.
  Attributes getAttributes(unsigned Idx) const;
};

} // End llvm namespace

#endif
