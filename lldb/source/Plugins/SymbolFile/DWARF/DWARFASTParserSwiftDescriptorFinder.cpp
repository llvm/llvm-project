//===-- DWARFASTParserSwiftDescriptorFinder.cpp ---------------------------===//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//
//
// Implements DWARFASTParserSwift's Descriptor finder interface
//
//===----------------------------------------------------------------------===//

#include <sstream>

#include "DWARFDebugInfo.h"
#include "DWARFASTParserSwift.h"

#include "DWARFDIE.h"

#include "Plugins/TypeSystem/Swift/TypeSystemSwiftTypeRef.h"
#include "swift/RemoteInspection/TypeLowering.h"

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::dwarf;
using namespace lldb_private::plugin::dwarf;

/// Given a type system and a typeref, return the compiler type and die of the
/// type that matches that mangled name, looking up the in the type system's
/// module's debug information.
static std::optional<std::pair<CompilerType, DWARFDIE>>
getTypeAndDie(TypeSystemSwiftTypeRef &ts,
              const swift::reflection::TypeRef *TR) {
  swift::Demangle::Demangler dem;
  swift::Demangle::NodePointer node = TR->getDemangling(dem);
  auto type = ts.RemangleAsType(dem, node);
  if (!type) {
    if (auto log = GetLog(LLDBLog::Types)) {
      std::stringstream ss;
      TR->dump(ss);
      LLDB_LOG(log, "Could not find type for typeref: {0}", ss.str());
    }
    return {};
  }

  auto *dwarf = llvm::cast_or_null<SymbolFileDWARF>(ts.GetSymbolFile());
  if (!dwarf)
    return {};
  auto lldb_type = ts.FindTypeInModule(type.GetOpaqueQualType());
  if (!lldb_type) {
    // TODO: for embedded Swift this is fine but consult other modules here for
    // general case?
    LLDB_LOGV(GetLog(LLDBLog::Types), "Could not find type {0} in module",
              type.GetMangledTypeName());
    return {};
  }
  auto die = dwarf->GetDIE(lldb_type->GetID());
  return {{type, die}};
}

static std::optional<swift::reflection::FieldDescriptorKind>
getFieldDescriptorKindForDie(CompilerType type) {
  auto type_class = type.GetTypeClass();
  switch (type_class) {
  case lldb::eTypeClassClass:
    return swift::reflection::FieldDescriptorKind::Class;
  case lldb::eTypeClassStruct:
    return swift::reflection::FieldDescriptorKind::Struct;
  case lldb::eTypeClassUnion:
    return swift::reflection::FieldDescriptorKind::Enum;
  default:
    LLDB_LOG(GetLog(LLDBLog::Types),
             "Could not determine file descriptor kind for type: {0}",
             type.GetMangledTypeName());
    return {};
  }
}

namespace {
// Class that implements the same layout as a builtin type descriptor, only it's
// built from DWARF instead.
class DWARFBuiltinTypeDescriptorImpl
    : public swift::reflection::BuiltinTypeDescriptorBase {
  ConstString m_type_name;

public:
  DWARFBuiltinTypeDescriptorImpl(uint32_t size, uint32_t alignment,
                                 uint32_t stride,
                                 uint32_t num_extra_inhabitants,
                                 bool is_bitwise_takable, ConstString type_name)
      : swift::reflection::BuiltinTypeDescriptorBase(
            size, alignment, stride, num_extra_inhabitants, is_bitwise_takable),
        m_type_name(type_name) {}
  ~DWARFBuiltinTypeDescriptorImpl() override = default;

  llvm::StringRef getMangledTypeName() override { return m_type_name; }
};

class DWARFFieldRecordImpl : public swift::reflection::FieldRecordBase {
  ConstString m_field_name;
  ConstString m_type_name;
  swift::Demangle::Demangler m_dem;

public:
  DWARFFieldRecordImpl(bool is_indirect_case, bool is_var,
                       ConstString field_name, ConstString type_name)
      : swift::reflection::FieldRecordBase(is_indirect_case, is_var,
                                           !type_name.IsEmpty()),
        m_field_name(field_name), m_type_name(type_name) {}

  ~DWARFFieldRecordImpl() override = default;
  llvm::StringRef getFieldName() override { return m_field_name; }

  NodePointer getDemangledTypeName() override {
    return m_dem.demangleSymbol(m_type_name);
  }
};

class DWARFFieldDescriptorImpl : public swift::reflection::FieldDescriptorBase {
  TypeSystemSwiftTypeRef &m_type_system;
  ConstString m_mangled_name;
  DIERef m_die_ref;
  NodePointer m_superclass_node;

public:
  DWARFFieldDescriptorImpl(swift::reflection::FieldDescriptorKind kind,
                           NodePointer superclass_node,
                           TypeSystemSwiftTypeRef &type_system,
                           ConstString mangled_name, DIERef die_ref)
      : swift::reflection::FieldDescriptorBase(kind,
                                               superclass_node != nullptr),
        m_type_system(type_system), m_mangled_name(mangled_name),
        m_die_ref(die_ref), m_superclass_node(superclass_node) {}

  ~DWARFFieldDescriptorImpl() override = default;

  swift::Demangle::NodePointer demangleSuperclass() override {
    return m_superclass_node;
  }

  std::vector<std::unique_ptr<swift::reflection::FieldRecordBase>>
  getFieldRecords() override {
    if (!Target::GetGlobalProperties().GetSwiftEnableFullDwarfDebugging())
      return {};
    auto *dwarf =
        llvm::dyn_cast<SymbolFileDWARF>(m_type_system.GetSymbolFile());
    auto *dwarf_parser = m_type_system.GetDWARFParser();
    if (!dwarf || !dwarf_parser)
      return {};

    auto die = dwarf->GetDIE(m_die_ref);
    if (!die)
      return {};

    switch (Kind) {
    case swift::reflection::FieldDescriptorKind::Struct:
    case swift::reflection::FieldDescriptorKind::Class:
      return getFieldRecordsFromStructOrClass(die, dwarf_parser);
    case swift::reflection::FieldDescriptorKind::Enum:
      return getFieldRecordsFromEnum(die, dwarf_parser);
    default:
      // TODO: handle more cases.
      LLDB_LOG(GetLog(LLDBLog::Types),
               "Trying to get field records of unexpected kind: {0}",
               (uint8_t)Kind);
      assert(false && "Trying to get field records of unexpected kind");
      return {};
    }
  }

  std::vector<std::unique_ptr<swift::reflection::FieldRecordBase>>
  getFieldRecordsFromStructOrClass(const DWARFDIE &die,
                          plugin::dwarf::DWARFASTParser *dwarf_parser) {
    std::vector<std::unique_ptr<swift::reflection::FieldRecordBase>> fields;
    for (DWARFDIE child_die : die.children()) {
      auto tag = child_die.Tag();
      if (tag != DW_TAG_member)
        continue;
      const auto *member_field_name =
          child_die.GetAttributeValueAsString(llvm::dwarf::DW_AT_name, "");
      auto *member_type = dwarf_parser->GetTypeForDIE(child_die);
      auto member_mangled_typename =
          member_type->GetForwardCompilerType().GetMangledTypeName();

      // Only matters for enums, so set to false for structs.
      bool is_indirect_case = false;
      // Unused by type info construction.
      bool is_var = false;
      fields.emplace_back(std::make_unique<DWARFFieldRecordImpl>(
          is_indirect_case, is_var, ConstString(member_field_name),
          member_mangled_typename));
    }
    return fields;
  }

  std::vector<std::unique_ptr<swift::reflection::FieldRecordBase>>
  getFieldRecordsFromEnum(const DWARFDIE &die,
                          plugin::dwarf::DWARFASTParser *dwarf_parser) {
    // Type lowering expects the payload fields to come before the non-payload
    // ones.
    std::vector<std::unique_ptr<swift::reflection::FieldRecordBase>>
        payload_fields;
    std::vector<std::unique_ptr<swift::reflection::FieldRecordBase>>
        non_payload_fields;
    auto variant_part = die.GetFirstChild();
    for (DWARFDIE child_die : variant_part.children()) {
      auto tag = child_die.Tag();
      if (tag != llvm::dwarf::DW_TAG_variant)
        continue;
      auto member = child_die.GetFirstChild();
      tag = member.Tag();
      if (tag != llvm::dwarf::DW_TAG_member)
        continue;
      const auto *member_field_name =
          member.GetAttributeValueAsString(llvm::dwarf::DW_AT_name, "");
      auto *member_type = dwarf_parser->GetTypeForDIE(member);

      // Empty enum cases don' have a type.
      auto member_mangled_typename =
          member_type
              ? member_type->GetForwardCompilerType().GetMangledTypeName()
              : ConstString();

      // Only matters for enums, so set to false for structs.
      bool is_indirect_case = false;
      // Unused by type info construction.
      bool is_var = false;
     
      // If there is a type, this case has a payload.
      if (member_type)
        payload_fields.emplace_back(std::make_unique<DWARFFieldRecordImpl>(
            is_indirect_case, is_var, ConstString(member_field_name),
            member_mangled_typename));
      else
        non_payload_fields.emplace_back(std::make_unique<DWARFFieldRecordImpl>(
            is_indirect_case, is_var, ConstString(member_field_name),
            member_mangled_typename));
    }
    // Add the non-payload cases to the end.
    payload_fields.insert(payload_fields.end(),
                          std::make_move_iterator(non_payload_fields.begin()),
                          std::make_move_iterator(non_payload_fields.end()));
    return payload_fields;
  }
};

class DWARFMultiPayloadEnumDescriptorImpl
    : public swift::reflection::MultiPayloadEnumDescriptorBase {
  ConstString m_mangled_name;
  DIERef m_die_ref;
  std::vector<uint8_t> m_spare_bits_mask;
  uint64_t m_byte_offset;

public:
  ~DWARFMultiPayloadEnumDescriptorImpl() override = default;

  DWARFMultiPayloadEnumDescriptorImpl(ConstString mangled_name, DIERef die_ref,
                                      std::vector<uint8_t> &&spare_bits_mask,
                                      uint64_t byte_offset)
      : swift::reflection::MultiPayloadEnumDescriptorBase(),
        m_mangled_name(mangled_name), m_die_ref(die_ref),
        m_spare_bits_mask(std::move(spare_bits_mask)),
        m_byte_offset(byte_offset) {}

  llvm::StringRef getMangledTypeName() override {
    return m_mangled_name.GetStringRef();
  }

  uint32_t getContentsSizeInWords() const override {
    return m_spare_bits_mask.size() / 4;
  }

  size_t getSizeInBytes() const override { return m_spare_bits_mask.size(); }

  uint32_t getFlags() const override { return usesPayloadSpareBits(); }

  bool usesPayloadSpareBits() const override {
    return !m_spare_bits_mask.empty();
  }

  uint32_t getPayloadSpareBitMaskByteOffset() const override {
    return m_byte_offset;
  }

  uint32_t getPayloadSpareBitMaskByteCount() const override {
    return getSizeInBytes();
  }

  const uint8_t *getPayloadSpareBits() const override {
    return m_spare_bits_mask.data();
  }
};
} // namespace

/// Constructs a builtin type descriptor from DWARF information.
std::unique_ptr<swift::reflection::BuiltinTypeDescriptorBase>
DWARFASTParserSwift::getBuiltinTypeDescriptor(
    const swift::reflection::TypeRef *TR) {
  if (!Target::GetGlobalProperties().GetSwiftEnableFullDwarfDebugging())
    return nullptr;

  auto pair = getTypeAndDie(m_swift_typesystem, TR);
  if (!pair)
    return nullptr;
  auto &[type, die] = *pair;

  if (!TypeSystemSwiftTypeRef::IsBuiltinType(type)) {
    if (die.Tag() == llvm::dwarf::DW_TAG_structure_type) {
      auto child = die.GetFirstChild();
      if (child.Tag() != llvm::dwarf::DW_TAG_variant_part)
        return nullptr;
    } else if (die.Tag() != llvm::dwarf::DW_TAG_base_type)
      return nullptr;
  }

  auto byte_size =
      die.GetAttributeValueAsUnsigned(DW_AT_byte_size, LLDB_INVALID_ADDRESS);
  if (byte_size == LLDB_INVALID_ADDRESS)
    return {};

  auto alignment = die.GetAttributeValueAsUnsigned(DW_AT_alignment, 8);

  // TODO: this seems simple to calculate but maybe we should encode the stride
  // in DWARF? That's what reflection metadata does.
  unsigned stride = ((byte_size + alignment - 1) & ~(alignment - 1));

  auto num_extra_inhabitants =
      die.GetAttributeValueAsUnsigned(DW_AT_APPLE_num_extra_inhabitants, 0);

  auto is_bitwise_takable = true; // TODO: encode it in DWARF

  return std::make_unique<DWARFBuiltinTypeDescriptorImpl>(
      byte_size, alignment, stride, num_extra_inhabitants, is_bitwise_takable,
      type.GetMangledTypeName());
}

std::unique_ptr<swift::reflection::MultiPayloadEnumDescriptorBase>
DWARFASTParserSwift::getMultiPayloadEnumDescriptor(
    const swift::reflection::TypeRef *TR) {
  if (!Target::GetGlobalProperties().GetSwiftEnableFullDwarfDebugging())
    return nullptr;

  auto pair = getTypeAndDie(m_swift_typesystem, TR);
  if (!pair)
    return nullptr;

  auto [type, die] = *pair;
  if (!die)
    return nullptr;

  auto kind = getFieldDescriptorKindForDie(type);
  if (!kind)
    return nullptr;

  auto child_die = die.GetFirstChild();
  auto bit_offset =
      child_die.GetAttributeValueAsUnsigned(llvm::dwarf::DW_AT_bit_offset, 0);

  auto byte_offset = (bit_offset + 7) / 8;

  const auto &attributes = child_die.GetAttributes();
  auto spare_bits_mask_idx =
      attributes.FindAttributeIndex(llvm::dwarf::DW_AT_APPLE_spare_bits_mask);
  if (spare_bits_mask_idx == UINT32_MAX)
    return nullptr;

  DWARFFormValue form_value;
  attributes.ExtractFormValueAtIndex(spare_bits_mask_idx, form_value);

  if (!form_value.IsValid()) {
    if (auto *log = GetLog(LLDBLog::Types)) {
      std::stringstream ss;
      TR->dump(ss);
      LLDB_LOG(log,
               "Could not produce MultiPayloadEnumTypeInfo for typeref: {0}",
               ss.str());
    }
    return nullptr;
  }
  // If there's a block data, this is a number bigger than 64 bits already
  // encoded as an array.
  if (form_value.BlockData()) {
    uint64_t block_length = form_value.Unsigned();
    std::vector<uint8_t> bytes(form_value.BlockData(),
                             form_value.BlockData() + block_length);
    return std::make_unique<DWARFMultiPayloadEnumDescriptorImpl>(
        type.GetMangledTypeName(), *die.GetDIERef(),
        std::move(bytes), byte_offset);
  }

  // If there is no block data, the spare bits mask is encoded as a single 64
  // bit number. Convert this to a byte array with only the amount of bytes
  // necessary to cover the whole number (see
  // MultiPayloadEnumDescriptorBuilder::layout on GenReflection.cpp for a
  // similar calculation when emitting this into metadata).
  llvm::APInt bits(64, form_value.Unsigned());
  auto bitsInMask = bits.getActiveBits(); 
  uint32_t bytesInMask = (bitsInMask + 7) / 8;
  auto wordsInMask = (bytesInMask + 3) / 4;
  bits = bits.zextOrTrunc(wordsInMask * 32);

  std::vector<uint8_t> bytes;
  for (size_t i = 0; i < bytesInMask; ++i) {
    uint8_t byte = bits.extractBitsAsZExtValue(8, 0);
    bytes.push_back(byte);
    bits.lshrInPlace(8);
  }

  return std::make_unique<DWARFMultiPayloadEnumDescriptorImpl>(
      type.GetMangledTypeName(), *die.GetDIERef(), std::move(bytes),
      byte_offset);
}

namespace {
DWARFDIE FindSuperClassDIE(DWARFDIE &die) {
  const auto inheritance_die_it =
      llvm::find_if(die.children(), [&](const DWARFDIE &child_die) {
        return child_die.Tag() == llvm::dwarf::DW_TAG_inheritance;
      });

  if (inheritance_die_it == die.children().end())
    return {};

  auto inheritance_die = *inheritance_die_it;
  const auto superclass_type_die =
      inheritance_die.GetAttributeValueAsReferenceDIE(llvm::dwarf::DW_AT_type);
  return superclass_type_die;
}
} // namespace

NodePointer DWARFASTParserSwift::GetCanonicalDemangleTree(DWARFDIE &die) {
  const auto name = StringRef(
      die.GetAttributeValueAsString(llvm::dwarf::DW_AT_linkage_name, ""));

  if (name.empty())
    return nullptr;

  auto *node =
      m_swift_typesystem.GetCanonicalDemangleTree(m_dem, name);
  return node;
}

std::unique_ptr<swift::reflection::FieldDescriptorBase>
DWARFASTParserSwift::getFieldDescriptor(const swift::reflection::TypeRef *TR) {
  if (!Target::GetGlobalProperties().GetSwiftEnableFullDwarfDebugging())
    return nullptr;

  auto pair = getTypeAndDie(m_swift_typesystem, TR);
  if (!pair)
    return nullptr;
  auto [type, die] = *pair;
  if (!die)
    return nullptr;
  auto kind = getFieldDescriptorKindForDie(type);
  if (!kind)
    return nullptr;

  DWARFDIE superclass_die = FindSuperClassDIE(die);
  NodePointer superclass_pointer = GetCanonicalDemangleTree(superclass_die);

  return std::make_unique<DWARFFieldDescriptorImpl>(
      *kind, superclass_pointer, m_swift_typesystem, type.GetMangledTypeName(),
      *die.GetDIERef());
}
