//===-- AttributeSetNode.h - AttributeList Internal Node ---------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines the node class used internally by AttributeList.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_ATTRIBUTESETNODE_H
#define LLVM_IR_ATTRIBUTESETNODE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Support/TrailingObjects.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>
#include <utility>

namespace llvm {

//===----------------------------------------------------------------------===//
/// \class
/// \brief This class represents a group of attributes that apply to one
/// element: function, return type, or parameter.
class AttributeSetNode final
    : public FoldingSetNode,
      private TrailingObjects<AttributeSetNode, Attribute> {
  friend TrailingObjects;

  unsigned NumAttrs; ///< Number of attributes in this node.
  /// Bitset with a bit for each available attribute Attribute::AttrKind.
  uint64_t AvailableAttrs;

  AttributeSetNode(ArrayRef<Attribute> Attrs)
    : NumAttrs(Attrs.size()), AvailableAttrs(0) {
    static_assert(Attribute::EndAttrKinds <= sizeof(AvailableAttrs) * CHAR_BIT,
                  "Too many attributes for AvailableAttrs");
    // There's memory after the node where we can store the entries in.
    std::copy(Attrs.begin(), Attrs.end(), getTrailingObjects<Attribute>());

    for (Attribute I : *this) {
      if (!I.isStringAttribute()) {
        AvailableAttrs |= ((uint64_t)1) << I.getKindAsEnum();
      }
    }
  }

public:
  // AttributesSetNode is uniqued, these should not be available.
  AttributeSetNode(const AttributeSetNode &) = delete;
  AttributeSetNode &operator=(const AttributeSetNode &) = delete;

  void operator delete(void *p) { ::operator delete(p); }

  static AttributeSetNode *get(LLVMContext &C, ArrayRef<Attribute> Attrs);

  static AttributeSetNode *get(AttributeList AS, unsigned Index) {
    return AS.getAttributes(Index);
  }

  /// \brief Return the number of attributes this AttributeList contains.
  unsigned getNumAttributes() const { return NumAttrs; }

  bool hasAttribute(Attribute::AttrKind Kind) const {
    return AvailableAttrs & ((uint64_t)1) << Kind;
  }
  bool hasAttribute(StringRef Kind) const;
  bool hasAttributes() const { return NumAttrs != 0; }

  Attribute getAttribute(Attribute::AttrKind Kind) const;
  Attribute getAttribute(StringRef Kind) const;

  unsigned getAlignment() const;
  unsigned getStackAlignment() const;
  uint64_t getDereferenceableBytes() const;
  uint64_t getDereferenceableOrNullBytes() const;
  std::pair<unsigned, Optional<unsigned>> getAllocSizeArgs() const;
  std::string getAsString(bool InAttrGrp) const;

  typedef const Attribute *iterator;
  iterator begin() const { return getTrailingObjects<Attribute>(); }
  iterator end() const { return begin() + NumAttrs; }

  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, makeArrayRef(begin(), end()));
  }
  static void Profile(FoldingSetNodeID &ID, ArrayRef<Attribute> AttrList) {
    for (const auto &Attr : AttrList)
      Attr.Profile(ID);
  }
};

} // end namespace llvm

#endif // LLVM_IR_ATTRIBUTESETNODE_H
