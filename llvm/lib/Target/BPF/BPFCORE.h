//===- BPFCORE.h - Common info for Compile-Once Run-EveryWhere  -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_BPFCORE_H
#define LLVM_LIB_TARGET_BPF_BPFCORE_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

class BasicBlock;
class Instruction;
class Module;

/// Whether a record element becomes a BTF member: data members, variant parts
/// and non-virtual C++ bases, the latter as anonymous members. Virtual bases
/// have no fixed offset.
inline bool isBTFRecordElement(const DINode *Element) {
  switch (Element->getTag()) {
  case dwarf::DW_TAG_member:
    return !cast<DIDerivedType>(Element)->isStaticMember();
  case dwarf::DW_TAG_inheritance:
    return !cast<DIDerivedType>(Element)->isVirtual();
  case dwarf::DW_TAG_variant_part:
    return true;
  default:
    return false;
  }
}

/// Whether a record element is a field that a CO-RE access index counts.
/// Clang does not count bases or its vtable pointer member ("_vptr$<class>").
inline bool isDIRecordField(const DINode *Element) {
  if (Element->getTag() == dwarf::DW_TAG_inheritance ||
      !isBTFRecordElement(Element))
    return false;
  const auto *DTy = dyn_cast<DIDerivedType>(Element);
  return !DTy || !(DTy->isArtificial() && DTy->getName().starts_with("_vptr$"));
}

/// Return the field a CO-RE access index refers to, or null if the debug info
/// does not describe the record's fields.
inline DINode *getDIRecordField(const DICompositeType *CTy,
                                uint64_t AccessIndex) {
  for (DINode *Element : CTy->getElements())
    if (isDIRecordField(Element) && AccessIndex-- == 0)
      return Element;
  return nullptr;
}

/// Return the bit offset used to order an element of a BTF structure record.
inline uint64_t getBTFRecordElementOffset(const DINode *Element) {
  switch (Element->getTag()) {
  case dwarf::DW_TAG_member:
  case dwarf::DW_TAG_inheritance:
    return cast<DIDerivedType>(Element)->getOffsetInBits();
  case dwarf::DW_TAG_variant_part:
    return cast<DICompositeType>(Element)->getOffsetInBits();
  default:
    llvm_unreachable("Unexpected DI tag of a struct element");
  }
}

/// Return the BTF members of a record in BTF order: structure members by
/// offset (BTF requires nondecreasing offsets), stable for equal offsets.
inline SmallVector<const DINode *, 8>
getBTFRecordElements(const DICompositeType *CTy) {
  SmallVector<const DINode *, 8> Elements;
  for (const DINode *Element : CTy->getElements())
    if (isBTFRecordElement(Element))
      Elements.push_back(Element);
  if (CTy->getTag() == dwarf::DW_TAG_structure_type)
    llvm::stable_sort(Elements, [](const DINode *LHS, const DINode *RHS) {
      return getBTFRecordElementOffset(LHS) < getBTFRecordElementOffset(RHS);
    });
  return Elements;
}

class BPFCoreSharedInfo {
public:
  enum BTFTypeIdFlag : uint32_t {
    BTF_TYPE_ID_LOCAL_RELOC = 0,
    BTF_TYPE_ID_REMOTE_RELOC,

    MAX_BTF_TYPE_ID_FLAG,
  };

  enum PreserveTypeInfo : uint32_t {
    PRESERVE_TYPE_INFO_EXISTENCE = 0,
    PRESERVE_TYPE_INFO_SIZE,
    PRESERVE_TYPE_INFO_MATCH,

    MAX_PRESERVE_TYPE_INFO_FLAG,
  };

  enum PreserveEnumValue : uint32_t {
    PRESERVE_ENUM_VALUE_EXISTENCE = 0,
    PRESERVE_ENUM_VALUE,

    MAX_PRESERVE_ENUM_VALUE_FLAG,
  };

  /// The attribute attached to globals representing a field access
  static constexpr StringRef AmaAttr = "btf_ama";
  /// The attribute attached to globals representing a type id
  static constexpr StringRef TypeIdAttr = "btf_type_id";

  /// llvm.bpf.passthrough builtin seq number
  static uint32_t SeqNum;

  /// Insert a bpf passthrough builtin function.
  static Instruction *insertPassThrough(Module *M, BasicBlock *BB,
                                        Instruction *Input,
                                        Instruction *Before);
  static void removeArrayAccessCall(CallInst *Call);
  static void removeStructAccessCall(CallInst *Call);
  static void removeUnionAccessCall(CallInst *Call);
};

} // namespace llvm

#endif
