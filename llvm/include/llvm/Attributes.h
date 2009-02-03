//===-- llvm/Attributes.h - Container for Attributes ---*---------- C++ -*-===//
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
#include <cassert>
#include <string>

namespace llvm {
class Type;

/// Attributes - A bitset of attributes.
typedef unsigned Attributes;

namespace Attribute {

/// Function parameters and results can have attributes to indicate how they 
/// should be treated by optimizations and code generation. This enumeration 
/// lists the attributes that can be associated with parameters, function 
/// results or the function itself.
/// @brief Function attributes.

const Attributes None      = 0;     ///< No attributes have been set
const Attributes ZExt      = 1<<0;  ///< Zero extended before/after call
const Attributes SExt      = 1<<1;  ///< Sign extended before/after call
const Attributes NoReturn  = 1<<2;  ///< Mark the function as not returning
const Attributes InReg     = 1<<3;  ///< Force argument to be passed in register
const Attributes StructRet = 1<<4;  ///< Hidden pointer to structure to return
const Attributes NoUnwind  = 1<<5;  ///< Function doesn't unwind stack
const Attributes NoAlias   = 1<<6;  ///< Considered to not alias after call
const Attributes ByVal     = 1<<7;  ///< Pass structure by value
const Attributes Nest      = 1<<8;  ///< Nested function static chain
const Attributes ReadNone  = 1<<9;  ///< Function does not access memory
const Attributes ReadOnly  = 1<<10; ///< Function only reads from memory
const Attributes NoInline        = 1<<11; ///< inline=never 
const Attributes AlwaysInline    = 1<<12; ///< inline=always
const Attributes OptimizeForSize = 1<<13; ///< opt_size
const Attributes StackProtect    = 1<<14; ///< Stack protection.
const Attributes StackProtectReq = 1<<15; ///< Stack protection required.
const Attributes Alignment = 31<<16; ///< Alignment of parameter (5 bits)
                                     // stored as log2 of alignment with +1 bias
                                     // 0 means unaligned different from align 1
const Attributes NoCapture = 1<<21; ///< Function creates no aliases of pointer

/// @brief Attributes that only apply to function parameters.
const Attributes ParameterOnly = ByVal | Nest | StructRet | NoCapture;

/// @brief Attributes that only apply to function.
const Attributes FunctionOnly = NoReturn | NoUnwind | ReadNone | ReadOnly | 
  NoInline | AlwaysInline | OptimizeForSize | StackProtect | StackProtectReq;

/// @brief Parameter attributes that do not apply to vararg call arguments.
const Attributes VarArgsIncompatible = StructRet;

/// @brief Attributes that are mutually incompatible.
const Attributes MutuallyIncompatible[4] = {
  ByVal | InReg | Nest | StructRet,
  ZExt  | SExt,
  ReadNone | ReadOnly,
  NoInline | AlwaysInline
};

/// @brief Which attributes cannot be applied to a type.
Attributes typeIncompatible(const Type *Ty);

/// This turns an int alignment (a power of 2, normally) into the
/// form used internally in Attributes.
inline Attributes constructAlignmentFromInt(unsigned i) {
  // Default alignment, allow the target to define how to align it.
  if (i == 0)
    return 0;

  assert(isPowerOf2_32(i) && "Alignment must be a power of two.");
  assert(i <= 0x40000000 && "Alignment too large.");
  return (Log2_32(i)+1) << 16;
}

/// This returns the alignment field of an attribute as a byte alignment value.
inline unsigned getAlignmentFromAttrs(Attributes A) {
  Attributes Align = A & Attribute::Alignment;
  if (Align == 0)
    return 0;
  
  return 1U << ((Align >> 16) - 1);
}
  
  
/// The set of Attributes set in Attributes is converted to a
/// string of equivalent mnemonics. This is, presumably, for writing out
/// the mnemonics for the assembly writer. 
/// @brief Convert attribute bits to text
std::string getAsString(Attributes Attrs);
} // end namespace Attribute

/// This is just a pair of values to associate a set of attributes
/// with an index. 
struct AttributeWithIndex {
  Attributes Attrs; ///< The attributes that are set, or'd together.
  unsigned Index; ///< Index of the parameter for which the attributes apply.
                  ///< Index 0 is used for return value attributes.
                  ///< Index ~0U is used for function attributes.
  
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
  /// AttrList - The attributes that we are managing.  This can be null
  /// to represent the empty attributes list.
  AttributeListImpl *AttrList;
public:
  AttrListPtr() : AttrList(0) {}
  AttrListPtr(const AttrListPtr &P);
  const AttrListPtr &operator=(const AttrListPtr &RHS);
  ~AttrListPtr();
  
  //===--------------------------------------------------------------------===//
  // Attribute List Construction and Mutation
  //===--------------------------------------------------------------------===//
  
  /// get - Return a Attributes list with the specified parameter in it.
  static AttrListPtr get(const AttributeWithIndex *Attr, unsigned NumAttrs);
  
  /// get - Return a Attribute list with the parameters specified by the
  /// consecutive random access iterator range.
  template <typename Iter>
  static AttrListPtr get(const Iter &I, const Iter &E) {
    if (I == E) return AttrListPtr();  // Empty list.
    return get(&*I, static_cast<unsigned>(E-I));
  }

  /// addAttr - Add the specified attribute at the specified index to this
  /// attribute list.  Since attribute lists are immutable, this
  /// returns the new list.
  AttrListPtr addAttr(unsigned Idx, Attributes Attrs) const;
  
  /// removeAttr - Remove the specified attribute at the specified index from
  /// this attribute list.  Since attribute lists are immutable, this
  /// returns the new list.
  AttrListPtr removeAttr(unsigned Idx, Attributes Attrs) const;
  
  //===--------------------------------------------------------------------===//
  // Attribute List Accessors
  //===--------------------------------------------------------------------===//
  /// getParamAttributes - The attributes for the specified index are
  /// returned. 
  Attributes getParamAttributes(unsigned Idx) const {
    assert (Idx && Idx != ~0U && "Invalid parameter index!");
    return getAttributes(Idx);
  }

  /// getRetAttributes - The attributes for the ret value are
  /// returned. 
  Attributes getRetAttributes() const {
    return getAttributes(0);
  }

  /// getFnAttributes - The function attributes are returned.
  Attributes getFnAttributes() const {
    return getAttributes(~0);
  }
  
  /// paramHasAttr - Return true if the specified parameter index has the
  /// specified attribute set.
  bool paramHasAttr(unsigned Idx, Attributes Attr) const {
    return getAttributes(Idx) & Attr;
  }
  
  /// getParamAlignment - Return the alignment for the specified function
  /// parameter.
  unsigned getParamAlignment(unsigned Idx) const {
    return Attribute::getAlignmentFromAttrs(getAttributes(Idx));
  }
  
  /// hasAttrSomewhere - Return true if the specified attribute is set for at
  /// least one parameter or for the return value.
  bool hasAttrSomewhere(Attributes Attr) const;

  /// operator==/!= - Provide equality predicates.
  bool operator==(const AttrListPtr &RHS) const { return AttrList == RHS.AttrList; }
  bool operator!=(const AttrListPtr &RHS) const { return AttrList != RHS.AttrList; }
  
  void dump() const;

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
  
private:
  explicit AttrListPtr(AttributeListImpl *L);

  /// getAttributes - The attributes for the specified index are
  /// returned.  Attributes for the result are denoted with Idx = 0.
  Attributes getAttributes(unsigned Idx) const;

};

} // End llvm namespace

#endif
