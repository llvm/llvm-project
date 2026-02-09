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

/// Given a mangled name in the format $snameD, return "name". The type info
/// builder machinery expects builtins to have their mangled names stored in
/// this format.
/// This is the inverse operation of LLDBTypeInfoProvider::getTypeInfo, which
/// consumes the name from TypeRefBuilder.
static llvm::StringRef ExtractTypeName(llvm::StringRef name) {
  if ((name.starts_with("$s") || name.starts_with("$e")) &&
      name.ends_with("D") && name.size() > 3) {
    return name.drop_front(2).drop_back(1);
  }
  return name;
}

// Class that implements the same layout as a builtin type descriptor, only it's
// built from DWARF instead.
class DWARFBuiltinTypeDescriptorImpl
    : public swift::reflection::BuiltinTypeDescriptorBase {
  // We store the typename as a StringRef, this is safe to do because the
  // backing variable which is passed in the constructor is a ConstString. If we
  // ever change the constructor to receive something else, we'd need to update
  // the type of this variable to own the data.
  llvm::StringRef m_type_name;

public:
  DWARFBuiltinTypeDescriptorImpl(uint32_t size, uint32_t alignment,
                                 uint32_t stride,
                                 uint32_t num_extra_inhabitants,
                                 bool is_bitwise_takable, ConstString type_name)
      : swift::reflection::BuiltinTypeDescriptorBase(
            size, alignment, stride, num_extra_inhabitants, is_bitwise_takable),
        m_type_name(ExtractTypeName(type_name.GetStringRef())) {}
  ~DWARFBuiltinTypeDescriptorImpl() override = default;

  llvm::StringRef getMangledTypeName() override { return m_type_name; }
};

/// Builtin type descriptor that owns its type name string, used for hardcoded
/// fallback descriptors.
class HardcodedBuiltinTypeDescriptorImpl
    : public swift::reflection::BuiltinTypeDescriptorBase {
  std::string m_type_name;

public:
  HardcodedBuiltinTypeDescriptorImpl(uint32_t size, uint32_t alignment,
                                     uint32_t stride,
                                     uint32_t num_extra_inhabitants,
                                     bool is_bitwise_takable,
                                     std::string type_name)
      : swift::reflection::BuiltinTypeDescriptorBase(
            size, alignment, stride, num_extra_inhabitants, is_bitwise_takable),
        m_type_name(std::move(type_name)) {}
  ~HardcodedBuiltinTypeDescriptorImpl() override = default;

  llvm::StringRef getMangledTypeName() override { return m_type_name; }
};

/// Returns a hardcoded builtin type descriptor for special stdlib builtin
/// types. This mirrors the types created by
/// IRGenModule::getOrCreateSpecialStlibBuiltinTypes() in the Swift compiler.
static std::unique_ptr<swift::reflection::BuiltinTypeDescriptorBase>
getHardcodedBuiltinTypeDescriptor(const swift::reflection::TypeRef *TR,
                                  uint32_t pointer_size) {
  auto *builtin_TR = llvm::dyn_cast<swift::reflection::BuiltinTypeRef>(TR);
  if (!builtin_TR)
    return nullptr;

  llvm::StringRef mangled_name = builtin_TR->getMangledName();

  auto makePointerSizedDescriptor =
      [pointer_size](std::string name, uint32_t num_extra_inhabitants) {
        return std::make_unique<HardcodedBuiltinTypeDescriptorImpl>(
            /*size=*/pointer_size,
            /*alignment=*/pointer_size,
            /*stride=*/pointer_size,
            /*num_extra_inhabitants=*/num_extra_inhabitants,
            /*is_bitwise_takable=*/true, std::move(name));
      };

  // Builtin.NativeObject (Bo).
  if (mangled_name == "Bo")
    return makePointerSizedDescriptor("Bo", /*num_extra_inhabitants=*/1);

  // Builtin.UnknownObject (BO).
  if (mangled_name == "BO")
    return makePointerSizedDescriptor("BO", /*num_extra_inhabitants=*/1);

  // Builtin.BridgeObject (Bb).
  if (mangled_name == "Bb") {
    uint32_t extra_inhabitants = pointer_size == 8 ? 0x7FFFFFFF : 0x3FFFFFFF;
    return makePointerSizedDescriptor("Bb", extra_inhabitants);
  }

  // Builtin.RawPointer.
  if (mangled_name == "Bp")
    return makePointerSizedDescriptor("Bp", /*num_extra_inhabitants=*/1);

  // Builtin.UnsafeValueBuffer.
  if (mangled_name == "BB") {
    uint32_t size = pointer_size * 3;
    return std::make_unique<HardcodedBuiltinTypeDescriptorImpl>(
        /*size=*/size,
        /*alignment=*/pointer_size,
        /*stride=*/size,
        /*num_extra_inhabitants=*/0,
        /*is_bitwise_takable=*/true, "BB");
  }

  // Thin function type () -> ().
  if (mangled_name == "yyXf")
    return makePointerSizedDescriptor("yyXf", /*num_extra_inhabitants=*/1);

  // Existential metatype Any.Type.
  if (mangled_name == "ypXp") {
    uint32_t extra_inhabitants = pointer_size == 8 ? 0x7FFFFFFF : 0x0FFFFFFF;
    return makePointerSizedDescriptor("ypXp", extra_inhabitants);
  }

  return nullptr;
}

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
      const char *member_field_name =
          child_die.GetAttributeValueAsString(llvm::dwarf::DW_AT_name, "");
      auto *member_type = dwarf_parser->GetTypeForDIE(child_die);
      if (!member_type)
        continue;
      auto member_compiler_type = member_type->GetForwardCompilerType();

      // TypeRefBuilder expects types to be canonical.
      CompilerType canonical = m_type_system.Canonicalize(member_compiler_type);
      if (!canonical) {
        LLDB_LOG(GetLog(LLDBLog::Types), "Could not build canonical type: {0}",
                 member_compiler_type.GetMangledTypeName());
        canonical = member_compiler_type;
      }
      // Only matters for enums, so set to false for structs.
      bool is_indirect_case = false;
      // Unused by type info construction.
      bool is_var = false;
      fields.emplace_back(std::make_unique<DWARFFieldRecordImpl>(
          is_indirect_case, is_var, ConstString(member_field_name),
          canonical.GetMangledTypeName()));
    }
    return fields;
  }

  /// Get field records from a raw value enum (DW_TAG_enumeration_type).
  /// Raw value enums have DW_TAG_enumerator children directly under the
  /// enumeration type DIE, with no payload types.
  std::vector<std::unique_ptr<swift::reflection::FieldRecordBase>>
  getFieldRecordsFromRawValueEnum(const DWARFDIE &die) {
    std::vector<std::unique_ptr<swift::reflection::FieldRecordBase>> fields;
    for (DWARFDIE child_die : die.children()) {
      if (child_die.Tag() != llvm::dwarf::DW_TAG_enumerator)
        continue;
      const auto *case_name =
          child_die.GetAttributeValueAsString(llvm::dwarf::DW_AT_name, "");
      // Raw value enum cases have no payload
      bool is_indirect_case = false;
      bool is_var = false;
      fields.emplace_back(std::make_unique<DWARFFieldRecordImpl>(
          is_indirect_case, is_var, ConstString(case_name), ConstString()));
    }
    return fields;
  }

  /// Get field records from a payload enum (DW_TAG_structure_type with
  /// DW_TAG_variant_part). Payload enums have a variant_part containing
  /// variant children, each with a member that may have an associated type.
  std::vector<std::unique_ptr<swift::reflection::FieldRecordBase>>
  getFieldRecordsFromPayloadEnum(const DWARFDIE &die,
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

  /// Get field records from an enum DIE, dispatching to the appropriate
  /// implementation based on the DWARF tag.
  std::vector<std::unique_ptr<swift::reflection::FieldRecordBase>>
  getFieldRecordsFromEnum(const DWARFDIE &die,
                          plugin::dwarf::DWARFASTParser *dwarf_parser) {
    // Raw value enums use DW_TAG_enumeration_type with DW_TAG_enumerator
    // children, while payload enums use DW_TAG_structure_type with
    // DW_TAG_variant_part containing DW_TAG_variant children.
    if (die.Tag() == llvm::dwarf::DW_TAG_enumeration_type)
      return getFieldRecordsFromRawValueEnum(die);
    return getFieldRecordsFromPayloadEnum(die, dwarf_parser);
  }
};
} // namespace

/// Constructs a builtin type descriptor from DWARF information.
static std::unique_ptr<swift::reflection::BuiltinTypeDescriptorBase>
getDWARFBuiltinTypeDescriptor(TypeSystemSwiftTypeRef &swift_typesystem,
                              const swift::reflection::TypeRef *TR) {
  auto pair = getTypeAndDie(swift_typesystem, TR);
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
    return nullptr;

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

/// Constructs a builtin type descriptor from DWARF information.
/// Falls back to hardcoded descriptors for special stdlib builtin types
/// that may not be present in DWARF.
std::unique_ptr<swift::reflection::BuiltinTypeDescriptorBase>
DWARFASTParserSwift::getBuiltinTypeDescriptor(
    const swift::reflection::TypeRef *TR) {
  assert(ModuleList::GetGlobalModuleListProperties()
                 .GetSwiftEnableFullDwarfDebugging() !=
             lldb_private::AutoBool::False &&
         "Full DWARF debugging for Swift is disabled!");

  if (auto descriptor = getDWARFBuiltinTypeDescriptor(m_swift_typesystem, TR))
    return descriptor;

  uint32_t pointer_size = m_swift_typesystem.GetPointerByteSize();
  return getHardcodedBuiltinTypeDescriptor(TR, pointer_size);
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
