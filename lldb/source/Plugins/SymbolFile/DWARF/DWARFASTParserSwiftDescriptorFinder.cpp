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

#include "DWARFASTParserSwift.h"

#include "DWARFDIE.h"
#include "DWARFDebugInfo.h"

#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "Plugins/TypeSystem/Swift/TypeSystemSwiftTypeRef.h"

#include "swift/RemoteInspection/TypeLowering.h"


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
  if (!type)
    return {};

  auto *dwarf = llvm::cast_or_null<SymbolFileDWARF>(ts.GetSymbolFile());
  if (!dwarf)
    return {};
  auto lldb_type = ts.FindTypeInModule(type.GetOpaqueQualType());
  if (!lldb_type)
    // TODO: for embedded Swift this is fine but consult other modules here for
    // general case?
    return {};
  auto die = dwarf->GetDIE(lldb_type->GetID());
  return {{type, die}};
}

static std::optional<swift::reflection::FieldDescriptorKind>
getFieldDescriptorKindForDie(DWARFDIE &die) {
  if (die.Tag() == DW_TAG_structure_type) {
    if (die.HasChildren() && die.GetFirstChild().Tag() == llvm::dwarf::DW_TAG_variant_part)
      return swift::reflection::FieldDescriptorKind::Enum;
    return swift::reflection::FieldDescriptorKind::Struct;
  }
  // TODO: handle more cases, for now we only support structs and enums.
  return {};
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

public:
  DWARFFieldDescriptorImpl(swift::reflection::FieldDescriptorKind kind,
                           bool has_superclass,
                           TypeSystemSwiftTypeRef &type_system,
                           ConstString mangled_name, DIERef die_ref)
      : swift::reflection::FieldDescriptorBase(kind, has_superclass),
        m_type_system(type_system), m_mangled_name(mangled_name),
        m_die_ref(die_ref) {}

  ~DWARFFieldDescriptorImpl() override = default;

  // TODO: implement this.
  swift::Demangle::NodePointer demangleSuperclass() override { return nullptr; }

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
      return getFieldRecordsFromStruct(die, dwarf_parser);
    case swift::reflection::FieldDescriptorKind::Enum:
      return getFieldRecordsFromEnum(die, dwarf_parser);
    default:
      // TODO: handle more cases.
      return {};
    }
  }

  std::vector<std::unique_ptr<swift::reflection::FieldRecordBase>>
  getFieldRecordsFromStruct(const DWARFDIE &die,
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
    std::vector<std::unique_ptr<swift::reflection::FieldRecordBase>> fields;
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
      fields.emplace_back(std::make_unique<DWARFFieldRecordImpl>(
          is_indirect_case, is_var, ConstString(member_field_name),
          member_mangled_typename));
    }
    return fields;
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

  if (die.Tag() == llvm::dwarf::DW_TAG_structure_type) {
    auto child = die.GetFirstChild();
    if (child.Tag() != llvm::dwarf::DW_TAG_variant_part)
      return nullptr;
  } else if (die.Tag() != llvm::dwarf::DW_TAG_base_type)
    return nullptr;

  auto byte_size =
      die.GetAttributeValueAsUnsigned(DW_AT_byte_size, LLDB_INVALID_ADDRESS);
  if (byte_size == LLDB_INVALID_ADDRESS)
    return {};
  auto alignment = die.GetAttributeValueAsUnsigned(
      DW_AT_alignment, byte_size == 0 ? 1 : byte_size);

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
  auto kind = getFieldDescriptorKindForDie(die);
  if (!kind)
    return nullptr;
// TODO: encode this in DWARF, maybe as a DW_AT_containing_type?
  bool has_superclass = false; 
  return std::make_unique<DWARFFieldDescriptorImpl>(
      *kind, has_superclass, m_swift_typesystem, type.GetMangledTypeName(),
      *die.GetDIERef());
}
