//===- TBAAMetadata.h - Type-Based Alias Analysis metadata ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Readers for the !tbaa metadata format described in LangRef.
///
/// TBAA metadata exists in two formats, and the operand layout differs between
/// them:
///
///   old scalar:     { name, parent, offset }
///   old struct:     { name, field, offset, field, offset, ... }
///   old access tag: { base type, access type, offset }
///
///   new scalar:     { parent, size, name }
///   new struct:     { parent, size, name, field, offset, size, ... }
///   new access tag: { base type, access type, offset, size }
///
/// These wrappers hide that difference so that producers and consumers of TBAA
/// metadata do not each have to open-code the operand indices. The nodes
/// themselves are created by the createTBAA*() methods of MDBuilder.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_TBAAMETADATA_H
#define LLVM_IR_TBAAMETADATA_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"
#include <cstdint>

namespace llvm {

/// isNewFormatTypeNode - Return true iff the given type node is in the new
/// size-aware format.
inline bool isNewFormatTypeNode(const MDNode *N) {
  if (!N || N->getNumOperands() < 3)
    return false;
  // In the new format type nodes have a reference to the parent type as their
  // first operand; in the old format the first operand is the name string.
  return isa_and_nonnull<MDNode>(N->getOperand(0));
}

/// This is a simple wrapper around an MDNode which provides a higher-level
/// interface by hiding the details of how alias analysis information is encoded
/// in its operands.
template <typename MDNodeTy> class TBAANodeImpl {
  MDNodeTy *Node = nullptr;

public:
  TBAANodeImpl() = default;
  explicit TBAANodeImpl(MDNodeTy *N) : Node(N) {}

  /// getNode - Get the MDNode for this TBAANode.
  MDNodeTy *getNode() const { return Node; }

  /// isNewFormat - Return true iff the wrapped type node is in the new
  /// size-aware format.
  bool isNewFormat() const { return isNewFormatTypeNode(Node); }

  /// getParent - Get this TBAANode's Alias tree parent.
  TBAANodeImpl<MDNodeTy> getParent() const {
    if (isNewFormat())
      return TBAANodeImpl(cast<MDNodeTy>(Node->getOperand(0)));

    if (Node->getNumOperands() < 2)
      return TBAANodeImpl<MDNodeTy>();
    MDNodeTy *P = dyn_cast_or_null<MDNodeTy>(Node->getOperand(1));
    if (!P)
      return TBAANodeImpl<MDNodeTy>();
    // Ok, this node has a valid parent. Return it.
    return TBAANodeImpl<MDNodeTy>(P);
  }

  /// Test if this TBAANode represents a type for objects which are
  /// not modified (by any means) in the context where this
  /// AliasAnalysis is relevant.
  bool isTypeImmutable() const {
    if (Node->getNumOperands() < 3)
      return false;
    ConstantInt *CI = mdconst::dyn_extract<ConstantInt>(Node->getOperand(2));
    if (!CI)
      return false;
    return CI->getValue()[0];
  }
};

/// \name Specializations of \c TBAANodeImpl for const and non const qualified
/// \c MDNode.
/// @{
using TBAANode = TBAANodeImpl<const MDNode>;
using MutableTBAANode = TBAANodeImpl<MDNode>;
/// @}

/// This is a simple wrapper around an MDNode which provides a
/// higher-level interface by hiding the details of how alias analysis
/// information is encoded in its operands.
template <typename MDNodeTy> class TBAAStructTagNodeImpl {
  /// This node should be created with createTBAAAccessTag().
  MDNodeTy *Node;

public:
  explicit TBAAStructTagNodeImpl(MDNodeTy *N) : Node(N) {}

  /// Get the MDNode for this TBAAStructTagNode.
  MDNodeTy *getNode() const { return Node; }

  /// isNewFormat - Return true iff the wrapped access tag is in the new
  /// size-aware format.
  bool isNewFormat() const {
    if (Node->getNumOperands() < 4)
      return false;
    if (MDNodeTy *AccessType = getAccessType())
      if (!TBAANodeImpl<MDNodeTy>(AccessType).isNewFormat())
        return false;
    return true;
  }

  MDNodeTy *getBaseType() const {
    return dyn_cast_or_null<MDNode>(Node->getOperand(0));
  }

  MDNodeTy *getAccessType() const {
    return dyn_cast_or_null<MDNode>(Node->getOperand(1));
  }

  uint64_t getOffset() const {
    return mdconst::extract<ConstantInt>(Node->getOperand(2))->getZExtValue();
  }

  uint64_t getSize() const {
    if (!isNewFormat())
      return UINT64_MAX;
    return mdconst::extract<ConstantInt>(Node->getOperand(3))->getZExtValue();
  }

  /// Test if this TBAAStructTagNode represents a type for objects
  /// which are not modified (by any means) in the context where this
  /// AliasAnalysis is relevant.
  bool isTypeImmutable() const {
    unsigned OpNo = isNewFormat() ? 4 : 3;
    if (Node->getNumOperands() < OpNo + 1)
      return false;
    ConstantInt *CI = mdconst::dyn_extract<ConstantInt>(Node->getOperand(OpNo));
    if (!CI)
      return false;
    return CI->getValue()[0];
  }
};

/// \name Specializations of \c TBAAStructTagNodeImpl for const and non const
/// qualified \c MDNods.
/// @{
using TBAAStructTagNode = TBAAStructTagNodeImpl<const MDNode>;
using MutableTBAAStructTagNode = TBAAStructTagNodeImpl<MDNode>;
/// @}

/// This is a simple wrapper around an MDNode which provides a
/// higher-level interface by hiding the details of how alias analysis
/// information is encoded in its operands.
class TBAAStructTypeNode {
  /// This node should be created with createTBAATypeNode().
  const MDNode *Node = nullptr;

public:
  TBAAStructTypeNode() = default;
  explicit TBAAStructTypeNode(const MDNode *N) : Node(N) {}

  /// Get the MDNode for this TBAAStructTypeNode.
  const MDNode *getNode() const { return Node; }

  /// isNewFormat - Return true iff the wrapped type node is in the new
  /// size-aware format.
  bool isNewFormat() const { return isNewFormatTypeNode(Node); }

  bool operator==(const TBAAStructTypeNode &Other) const {
    return getNode() == Other.getNode();
  }

  /// getId - Return type identifier.
  Metadata *getId() const { return Node->getOperand(isNewFormat() ? 2 : 0); }

  unsigned getNumFields() const {
    unsigned FirstFieldOpNo = isNewFormat() ? 3 : 1;
    unsigned NumOpsPerField = isNewFormat() ? 3 : 2;
    return (getNode()->getNumOperands() - FirstFieldOpNo) / NumOpsPerField;
  }

  TBAAStructTypeNode getFieldType(unsigned FieldIndex) const {
    unsigned FirstFieldOpNo = isNewFormat() ? 3 : 1;
    unsigned NumOpsPerField = isNewFormat() ? 3 : 2;
    unsigned OpIndex = FirstFieldOpNo + FieldIndex * NumOpsPerField;
    auto *TypeNode = cast<MDNode>(getNode()->getOperand(OpIndex));
    return TBAAStructTypeNode(TypeNode);
  }

  /// Get the offset of the field at \p FieldIndex within this type.
  uint64_t getFieldOffset(unsigned FieldIndex) const {
    unsigned FirstFieldOpNo = isNewFormat() ? 3 : 1;
    unsigned NumOpsPerField = isNewFormat() ? 3 : 2;
    unsigned OpIndex = FirstFieldOpNo + FieldIndex * NumOpsPerField;
    return mdconst::extract<ConstantInt>(getNode()->getOperand(OpIndex + 1))
        ->getZExtValue();
  }

  /// Get this TBAAStructTypeNode's field in the type DAG with
  /// given offset. Update the offset to be relative to the field type.
  TBAAStructTypeNode getField(uint64_t &Offset) const {
    bool NewFormat = isNewFormat();
    const ArrayRef<MDOperand> Operands = Node->operands();
    const unsigned NumOperands = Operands.size();

    if (NewFormat) {
      // New-format root and scalar type nodes have no fields.
      if (NumOperands < 6)
        return TBAAStructTypeNode();
    } else {
      // Parent can be omitted for the root node.
      if (NumOperands < 2)
        return TBAAStructTypeNode();

      // Fast path for a scalar type node and a struct type node with a single
      // field.
      if (NumOperands <= 3) {
        uint64_t Cur =
            NumOperands == 2
                ? 0
                : mdconst::extract<ConstantInt>(Operands[2])->getZExtValue();
        Offset -= Cur;
        MDNode *P = dyn_cast_or_null<MDNode>(Operands[1]);
        if (!P)
          return TBAAStructTypeNode();
        return TBAAStructTypeNode(P);
      }
    }

    // Assume the offsets are in order. We return the previous field if
    // the current offset is bigger than the given offset.
    unsigned FirstFieldOpNo = NewFormat ? 3 : 1;
    unsigned NumOpsPerField = NewFormat ? 3 : 2;
    unsigned TheIdx = 0;

    for (unsigned Idx = FirstFieldOpNo; Idx < NumOperands;
         Idx += NumOpsPerField) {
      uint64_t Cur =
          mdconst::extract<ConstantInt>(Operands[Idx + 1])->getZExtValue();
      if (Cur > Offset) {
        assert(Idx >= FirstFieldOpNo + NumOpsPerField &&
               "TBAAStructTypeNode::getField should have an offset match!");
        TheIdx = Idx - NumOpsPerField;
        break;
      }
    }
    // Move along the last field.
    if (TheIdx == 0)
      TheIdx = NumOperands - NumOpsPerField;
    uint64_t Cur =
        mdconst::extract<ConstantInt>(Operands[TheIdx + 1])->getZExtValue();
    Offset -= Cur;
    MDNode *P = dyn_cast_or_null<MDNode>(Operands[TheIdx]);
    if (!P)
      return TBAAStructTypeNode();
    return TBAAStructTypeNode(P);
  }
};

} // end namespace llvm

#endif // LLVM_IR_TBAAMETADATA_H
