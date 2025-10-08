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
#include "swift/Demangling/ManglingFlavor.h"
#include "swift/RemoteInspection/TypeLowering.h"

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;

/// Given a die to a substituted generic Swift type, return the analogous
/// unsubstituted version.
///
/// Given a generic type (for example, a generic Pair), the compiler will emit
/// full debug information for the unsubstituted type (Pair<T, U>), and opaque
/// debug information for each every specialization (for example, Pair<Int,
/// Double>), with a link back to the unsubstituted type. When looking up one of
/// the specialized generics, return the unsubstituted version instead.
static std::optional<std::pair<CompilerType, DWARFDIE>>
findUnsubstitutedGenericTypeAndDIE(TypeSystemSwiftTypeRef &ts,
                                   const DWARFDIE &die) {
  auto unsubstituted_die =
      die.GetAttributeValueAsReferenceDIE(llvm::dwarf::DW_AT_specification);
  if (!unsubstituted_die)
    return {};

  const auto *mangled_name = unsubstituted_die.GetAttributeValueAsString(
      llvm::dwarf::DW_AT_linkage_name, nullptr);
  assert(mangled_name);
  auto unsubstituted_type =
      ts.GetTypeFromMangledTypename(ConstString(mangled_name));
  return {{unsubstituted_type, unsubstituted_die}};
}

lldb_private::CompilerType static MapTypeIntoContext(
    TypeSystemSwiftTypeRef &ts, lldb_private::CompilerType context,
    lldb_private::CompilerType type) {
  return ts.ApplySubstitutions(
      type.GetOpaqueQualType(),
      ts.GetSubstitutions(context.GetOpaqueQualType()));
}

std::pair<lldb::TypeSP, lldb_private::CompilerType>
DWARFASTParserSwift::ResolveTypeAlias(lldb_private::CompilerType alias) {
  if (!alias)
    return {};
  auto ts_sp = alias.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>();
  if (!ts_sp)
    return {};
  auto &ts = *ts_sp;
  auto *dwarf = llvm::dyn_cast_or_null<SymbolFileDWARF>(ts.GetSymbolFile());
  if (!dwarf)
    return {};

  // Type aliases are (for LLVM implementation reasons) using the
  // DW_AT_name as linkage name, so they can't be looked up by base
  // name. This should be fixed.
  // Meanwhile, instead find them inside their parent type.
  CompilerType parent_ctx = ts.GetParentType(alias.GetOpaqueQualType());
  if (!parent_ctx)
    return {};

  DWARFDIE parent_die;
  if (TypeSP parent_type =
          ts.FindTypeInModule(parent_ctx.GetOpaqueQualType())) {
    parent_die = dwarf->GetDIE(parent_type->GetID());
    auto unsubstituted_pair =
        findUnsubstitutedGenericTypeAndDIE(ts, parent_die);
    if (unsubstituted_pair)
      parent_die = unsubstituted_pair->second;
  }
  if (!parent_die)
    return {};
  std::string alias_name = ts.GetBaseName(alias.GetOpaqueQualType());
  for (DWARFDIE child_die : parent_die.children()) {
    auto tag = child_die.Tag();
    if (tag == llvm::dwarf::DW_TAG_member)
      continue;
    std::string base_name;
    const auto *name =
        child_die.GetAttributeValueAsString(llvm::dwarf::DW_AT_name, "");
    if (name && *name == '$') {
      CompilerType candidate = ts.GetTypeFromMangledTypename(ConstString(name));
      base_name = ts.GetBaseName(candidate.GetOpaqueQualType());
    } else {
      base_name = name;
    }
    if (base_name != alias_name)
      continue;

    // Follow the typedef.
    auto *dwarf_parser = ts.GetDWARFParser();
    if (!dwarf_parser)
      return {};
    Type *t = dwarf_parser->GetTypeForDIE(child_die);
    if (!t)
      return {};
    CompilerType cty = t->GetForwardCompilerType();
    if (ts.IsMeaninglessWithoutDynamicResolution(cty.GetOpaqueQualType())) {
      // Substitute the parameters in the LHS of the BGTAT.
      if (ts.IsBoundGenericAliasType(alias.GetOpaqueQualType())) {
        auto subs = ts.GetSubstitutions(alias.GetOpaqueQualType());
        while (subs.size() > 1)
          subs.erase(subs.begin());
        cty = ts.ApplySubstitutions(cty.GetOpaqueQualType(), subs);
      }
      // Substitute the parameters of the RHS of the (BGT)AT.
      return {t->shared_from_this(), MapTypeIntoContext(ts, parent_ctx, cty)};
    }
    return {t->shared_from_this(), cty};
  }
  return {};
}

/// Given a type system and a typeref, return the compiler type and die of the
/// type that matches that mangled name, looking up the in the type system's
/// module's debug information.
static std::optional<std::pair<CompilerType, DWARFDIE>>
getTypeAndDie(TypeSystemSwiftTypeRef &ts,
              const swift::reflection::TypeRef *TR) {
  swift::Demangle::Demangler dem;
  swift::Demangle::NodePointer node = TR->getDemangling(dem);
  // TODO: mangling flavor should come from the TypeRef.
  auto type =
      ts.RemangleAsType(dem, node, swift::Mangle::ManglingFlavor::Embedded);
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
  TypeSP lldb_type = ts.FindTypeInModule(type.GetOpaqueQualType());
  if (!lldb_type) {
    if (ts.ContainsBoundGenericType(type.GetOpaqueQualType())) {
      CompilerType generic_type = ts.MapOutOfContext(type.GetOpaqueQualType());
      lldb_type = ts.FindTypeInModule(generic_type.GetOpaqueQualType());
    }
  }
  if (!lldb_type) {
    std::tie(lldb_type, type) = DWARFASTParserSwift::ResolveTypeAlias(type);
    if (lldb_type) {
      auto die = dwarf->GetDIE(lldb_type->GetID());
      return {{type, die}};
    }
  }
  if (!lldb_type) {
    // TODO: for embedded Swift this is fine but consult other modules
    // here for general case?
    LLDB_LOGV(GetLog(LLDBLog::Types), "Could not find type {0} in module",
              type.GetMangledTypeName());
    return {};
  }
  auto die = dwarf->GetDIE(lldb_type->GetID());

  if (auto unsubstituted_pair = findUnsubstitutedGenericTypeAndDIE(ts, die))
    return unsubstituted_pair;

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
    assert(ModuleList::GetGlobalModuleListProperties()
                   .GetSwiftEnableFullDwarfDebugging() !=
               lldb_private::AutoBool::False &&
           "Full DWARF debugging for Swift is disabled!");

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
      if (tag != llvm::dwarf::DW_TAG_member)
        continue;
      const auto *member_field_name =
          child_die.GetAttributeValueAsString(llvm::dwarf::DW_AT_name, "");
      auto *member_type = dwarf_parser->GetTypeForDIE(child_die);
      if (!member_type)
        continue;
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
} // namespace

/// Constructs a builtin type descriptor from DWARF information.
std::unique_ptr<swift::reflection::BuiltinTypeDescriptorBase>
DWARFASTParserSwift::getBuiltinTypeDescriptor(
    const swift::reflection::TypeRef *TR) {
  assert(ModuleList::GetGlobalModuleListProperties()
                 .GetSwiftEnableFullDwarfDebugging() !=
             lldb_private::AutoBool::False &&
         "Full DWARF debugging for Swift is disabled!");

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

  auto byte_size = die.GetAttributeValueAsUnsigned(llvm::dwarf::DW_AT_byte_size,
                                                   LLDB_INVALID_ADDRESS);
  if (byte_size == LLDB_INVALID_ADDRESS)
    return {};

  auto alignment = die.GetAttributeValueAsUnsigned(llvm::dwarf::DW_AT_alignment,
                                                   byte_size ? byte_size : 8);

  // TODO: this seems simple to calculate but maybe we should encode the stride
  // in DWARF? That's what reflection metadata does.
  unsigned stride = ((byte_size + alignment - 1) & ~(alignment - 1));

  auto num_extra_inhabitants = die.GetAttributeValueAsUnsigned(
      llvm::dwarf::DW_AT_LLVM_num_extra_inhabitants, 0);

  auto is_bitwise_takable = true; // TODO: encode it in DWARF

  return std::make_unique<DWARFBuiltinTypeDescriptorImpl>(
      byte_size, alignment, stride, num_extra_inhabitants, is_bitwise_takable,
      type.GetMangledTypeName());
}

std::unique_ptr<swift::reflection::MultiPayloadEnumDescriptorBase>
DWARFASTParserSwift::getMultiPayloadEnumDescriptor(
    const swift::reflection::TypeRef *TR) {
  // Remote mirrors is able to calculate type information without needing a MultiPayloadEnumDescriptor.
  return nullptr;
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
  assert(ModuleList::GetGlobalModuleListProperties()
                 .GetSwiftEnableFullDwarfDebugging() !=
             lldb_private::AutoBool::False &&
         "Full DWARF debugging for Swift is disabled!");

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
