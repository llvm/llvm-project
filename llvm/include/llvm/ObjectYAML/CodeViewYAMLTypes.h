//===- CodeViewYAMLTypes.h - CodeView YAMLIO Type Record implementation ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines classes for handling the YAML representation of CodeView
// Debug Info.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_CODEVIEWYAMLTYPES_H
#define LLVM_OBJECTYAML_CODEVIEWYAMLTYPES_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/Allocator.h"

namespace llvm {
namespace codeview {
class TypeTableBuilder;
}
namespace CodeViewYAML {
namespace detail {
struct LeafRecordBase;
struct MemberRecordBase;
}

struct MemberRecord {
  std::shared_ptr<detail::MemberRecordBase> Member;
};

struct LeafRecord {
  std::shared_ptr<detail::LeafRecordBase> Leaf;

  codeview::CVType toCodeViewRecord(BumpPtrAllocator &Allocator) const;
  codeview::CVType toCodeViewRecord(codeview::TypeTableBuilder &TS) const;
  static Expected<LeafRecord> fromCodeViewRecord(codeview::CVType Type);
};

std::vector<LeafRecord> fromDebugT(ArrayRef<uint8_t> DebugT);
ArrayRef<uint8_t> toDebugT(ArrayRef<LeafRecord>, BumpPtrAllocator &Alloc);
} // namespace CodeViewYAML
} // namespace llvm

LLVM_YAML_DECLARE_MAPPING_TRAITS(CodeViewYAML::LeafRecord)
LLVM_YAML_DECLARE_MAPPING_TRAITS(CodeViewYAML::MemberRecord)

LLVM_YAML_IS_SEQUENCE_VECTOR(CodeViewYAML::LeafRecord)
LLVM_YAML_IS_SEQUENCE_VECTOR(CodeViewYAML::MemberRecord)

#endif
