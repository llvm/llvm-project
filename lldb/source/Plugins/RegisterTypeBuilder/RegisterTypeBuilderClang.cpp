//===-- RegisterTypeBuilderClang.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"

#include "RegisterTypeBuilderClang.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Utility/RegisterType.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/bit.h"

#include <algorithm>

using namespace lldb_private;

LLDB_PLUGIN_DEFINE(RegisterTypeBuilderClang)

void RegisterTypeBuilderClang::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void RegisterTypeBuilderClang::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb::RegisterTypeBuilderSP
RegisterTypeBuilderClang::CreateInstance(Target &target) {
  return std::make_shared<RegisterTypeBuilderClang>(target);
}

RegisterTypeBuilderClang::RegisterTypeBuilderClang(Target &target)
    : m_target(target) {}

CompilerType RegisterTypeBuilderClang::BuildBuiltinType(
    const RegisterTypeBuiltin *builtin_type, uint32_t expected_byte_size,
    lldb::TypeSystemClangSP type_system) {
  if (auto type = GetExistingCompilerType(builtin_type, expected_byte_size))
    return *type;

  CompilerType compiler_type;
  clang::ASTContext &ast = type_system->getASTContext();
  // These GDB types have semantics that encoding and byte size cannot express.
  if (builtin_type->GetID() == "data_ptr" ||
      builtin_type->GetID() == "code_ptr")
    compiler_type = type_system->GetType(ast.VoidPtrTy);
  else if (builtin_type->GetID() == "bool")
    compiler_type = type_system->GetType(ast.BoolTy);
  else if (builtin_type->GetID() == "bfloat16")
    compiler_type = type_system->GetType(ast.BFloat16Ty);
  else if (std::optional<uint64_t> byte_size = builtin_type->GetByteSize())
    compiler_type = type_system->GetBuiltinTypeForEncodingAndBitSize(
        builtin_type->GetEncoding(), *byte_size * 8);

  if (!compiler_type.IsValid() ||
      llvm::expectedToOptional(compiler_type.GetByteSize(nullptr)) !=
          expected_byte_size)
    return {};

  m_type_cache.try_emplace(
      std::make_pair(builtin_type->GetUID(), expected_byte_size),
      compiler_type);
  return compiler_type;
}

CompilerType
RegisterTypeBuilderClang::BuildEnumType(const RegisterTypeEnum *enum_type_info,
                                        uint32_t register_byte_size,
                                        lldb::TypeSystemClangSP type_system) {
  if (auto maybe_compiler_type =
          GetExistingCompilerType(enum_type_info, register_byte_size))
    return *maybe_compiler_type;

  CompilerType register_uint_type =
      type_system->GetBuiltinTypeForEncodingAndBitSize(lldb::eEncodingUint,
                                                       register_byte_size * 8);
  CompilerType enum_type = type_system->CreateEnumerationType(
      "", type_system->GetTranslationUnitDecl(), OptionalClangModuleID(),
      Declaration(), register_uint_type, false);

  type_system->StartTagDeclarationDefinition(enum_type);

  Declaration decl;
  for (const auto &enumerator : enum_type_info->GetEnumerators()) {
    type_system->AddEnumerationValueToEnumerationType(
        enum_type, decl, enumerator.m_name.c_str(), enumerator.m_value,
        register_byte_size * 8);
  }

  type_system->CompleteTagDeclarationDefinition(enum_type);

  m_type_cache.try_emplace(
      std::make_pair(enum_type_info->GetUID(), register_byte_size), enum_type);
  return enum_type;
}

CompilerType RegisterTypeBuilderClang::BuildFlagsType(
    const lldb_private::RegisterTypeFlags *flags_info,
    uint32_t register_byte_size, lldb::TypeSystemClangSP type_system) {
  if (auto maybe_compiler_type =
          GetExistingCompilerType(flags_info, register_byte_size))
    return *maybe_compiler_type;

  // In most ABI, a change of field type means a change in storage unit.
  // We want it all in one unit, so we use a field type the same as the
  // register's size.
  CompilerType field_uint_type =
      type_system->GetBuiltinTypeForEncodingAndBitSize(lldb::eEncodingUint,
                                                       register_byte_size * 8);

  CompilerType flags_type = type_system->CreateRecordType(
      nullptr, OptionalClangModuleID(), "",
      llvm::to_underlying(clang::TagTypeKind::Struct), lldb::eLanguageTypeC);
  type_system->StartTagDeclarationDefinition(flags_type);

  for (auto field : flags_info->GetFields()) {
    CompilerType field_type = field_uint_type;

    if (const RegisterTypeEnum *enum_type_info = field.GetEnum())
      if (!enum_type_info->GetEnumerators().empty())
        field_type =
            BuildEnumType(enum_type_info, register_byte_size, type_system);

    type_system->AddFieldToRecordType(flags_type, field.GetName(), field_type,
                                      field.GetSizeInBits());
  }

  type_system->CompleteTagDeclarationDefinition(flags_type);
  // So that the size of the type matches the size of the register.
  type_system->SetIsPacked(flags_type);

  // This should be true if RegisterTypeFlags padded correctly.
  assert(
      llvm::expectedToOptional(flags_type.GetByteSize(nullptr)).value_or(0) ==
      flags_info->GetSize());

  m_type_cache.try_emplace(
      std::make_pair(flags_info->GetUID(), register_byte_size), flags_type);
  return flags_type;
}

CompilerType
RegisterTypeBuilderClang::BuildVectorType(const RegisterTypeVector *vector_type,
                                          uint32_t expected_byte_size,
                                          lldb::TypeSystemClangSP type_system) {
  if (!expected_byte_size)
    return {};
  if (auto type = GetExistingCompilerType(vector_type, expected_byte_size))
    return *type;

  std::optional<uint32_t> element_size =
      GetTargetByteSize(vector_type->GetElementType(), type_system);
  if (!element_size || *element_size > UINT32_MAX / vector_type->GetCount() ||
      *element_size * vector_type->GetCount() != expected_byte_size)
    return {};

  const RegisterType *element_register_type = vector_type->GetElementType();
  CompilerType element_type =
      BuildType(element_register_type, *element_size, type_system);
  if (!element_type.IsValid())
    return {};

  const auto *builtin_element =
      llvm::dyn_cast<RegisterTypeBuiltin>(element_register_type);
  bool pointer_element =
      builtin_element && (builtin_element->GetID() == "data_ptr" ||
                          builtin_element->GetID() == "code_ptr");
  // Preserve vector semantics when Clang can represent the XML shape without
  // padding. Use an array for pointer, boolean, nested and non-power-of-two
  // vectors so that their layout matches the XML exactly.
  bool use_vector = builtin_element && !pointer_element &&
                    builtin_element->GetID() != "bool" &&
                    llvm::has_single_bit(vector_type->GetCount());
  CompilerType compiler_type = type_system->CreateArrayType(
      element_type, vector_type->GetCount(), use_vector);

  auto compiler_size =
      llvm::expectedToOptional(compiler_type.GetByteSize(nullptr));
  // Target ABI rules can still pad a Clang vector. Fall back to an array when
  // that gives the exact byte size described by the register XML.
  if (use_vector && compiler_size != expected_byte_size) {
    compiler_type = type_system->CreateArrayType(
        element_type, vector_type->GetCount(), /*is_vector=*/false);
    compiler_size =
        llvm::expectedToOptional(compiler_type.GetByteSize(nullptr));
  }
  if (compiler_size != expected_byte_size)
    return {};

  m_type_cache.try_emplace(
      std::make_pair(vector_type->GetUID(), expected_byte_size), compiler_type);
  return compiler_type;
}

CompilerType
RegisterTypeBuilderClang::BuildUnionType(const RegisterTypeUnion *union_type,
                                         uint32_t expected_byte_size,
                                         lldb::TypeSystemClangSP type_system) {
  if (!expected_byte_size)
    return {};
  if (auto type = GetExistingCompilerType(union_type, expected_byte_size))
    return *type;

  std::vector<std::pair<llvm::StringRef, CompilerType>> fields;
  fields.reserve(union_type->GetFields().size());
  for (const RegisterTypeUnion::Field &field : union_type->GetFields()) {
    std::optional<uint32_t> field_size =
        GetTargetByteSize(field.GetType(), type_system);
    if (!field_size || *field_size > expected_byte_size)
      return {};

    CompilerType field_type =
        BuildType(field.GetType(), *field_size, type_system);
    if (!field_type.IsValid())
      return {};
    fields.emplace_back(field.GetName(), field_type);
  }

  std::string type_name = "__lldb_register_union_" +
                          std::to_string(union_type->GetUID()) + "_" +
                          std::to_string(expected_byte_size);
  CompilerType compiler_type = type_system->CreateRecordType(
      nullptr, OptionalClangModuleID(), type_name,
      llvm::to_underlying(clang::TagTypeKind::Union), lldb::eLanguageTypeC);
  type_system->StartTagDeclarationDefinition(compiler_type);
  for (const auto &[field_name, field_type] : fields)
    type_system->AddFieldToRecordType(compiler_type, field_name, field_type,
                                      /*bitfield_bit_size=*/0);
  type_system->CompleteTagDeclarationDefinition(compiler_type);
  type_system->SetIsPacked(compiler_type);

  if (llvm::expectedToOptional(compiler_type.GetByteSize(nullptr)) !=
      expected_byte_size)
    return {};

  TypeSummaryImpl::Flags summary_flags;
  summary_flags.SetShowMembersOneLiner(true);
  auto summary_sp = std::make_shared<StringSummaryFormat>(summary_flags, "");
  lldb::TypeCategoryImplSP category_sp;
  DataVisualization::Categories::GetCategory(ConstString("default"),
                                             category_sp);
  if (category_sp)
    category_sp->AddTypeSummary(type_name, lldb::eFormatterMatchExact,
                                summary_sp);

  m_type_cache.try_emplace(
      std::make_pair(union_type->GetUID(), expected_byte_size), compiler_type);
  return compiler_type;
}

CompilerType
RegisterTypeBuilderClang::BuildType(const RegisterType *register_type,
                                    uint32_t expected_byte_size,
                                    lldb::TypeSystemClangSP type_system) {
  switch (register_type->getKind()) {
  case RegisterType::eRegisterTypeKindBuiltin:
    return BuildBuiltinType(llvm::cast<RegisterTypeBuiltin>(register_type),
                            expected_byte_size, type_system);
  case RegisterType::eRegisterTypeKindVector:
    return BuildVectorType(llvm::cast<RegisterTypeVector>(register_type),
                           expected_byte_size, type_system);
  case RegisterType::eRegisterTypeKindUnion:
    return BuildUnionType(llvm::cast<RegisterTypeUnion>(register_type),
                          expected_byte_size, type_system);
  case RegisterType::eRegisterTypeKindEnum:
  case RegisterType::eRegisterTypeKindFlags:
    return {};
  }
}

std::optional<uint32_t> RegisterTypeBuilderClang::GetTargetByteSize(
    const RegisterType *register_type, lldb::TypeSystemClangSP type_system) {
  if (std::optional<uint64_t> fixed_size = register_type->GetByteSize()) {
    if (*fixed_size <= UINT32_MAX)
      return static_cast<uint32_t>(*fixed_size);
    return std::nullopt;
  }

  switch (register_type->getKind()) {
  case RegisterType::eRegisterTypeKindBuiltin: {
    const auto *builtin_type = llvm::cast<RegisterTypeBuiltin>(register_type);
    if (builtin_type->GetID() != "data_ptr" &&
        builtin_type->GetID() != "code_ptr")
      return std::nullopt;
    std::optional<uint64_t> pointer_size = llvm::expectedToOptional(
        type_system->GetType(type_system->getASTContext().VoidPtrTy)
            .GetByteSize(nullptr));
    if (!pointer_size || *pointer_size > UINT32_MAX)
      return std::nullopt;
    return static_cast<uint32_t>(*pointer_size);
  }
  case RegisterType::eRegisterTypeKindVector: {
    const auto *vector_type = llvm::cast<RegisterTypeVector>(register_type);
    std::optional<uint32_t> element_size =
        GetTargetByteSize(vector_type->GetElementType(), type_system);
    if (!element_size || *element_size > UINT32_MAX / vector_type->GetCount())
      return std::nullopt;
    return *element_size * vector_type->GetCount();
  }
  case RegisterType::eRegisterTypeKindUnion: {
    uint32_t byte_size = 0;
    for (const RegisterTypeUnion::Field &field :
         llvm::cast<RegisterTypeUnion>(register_type)->GetFields()) {
      std::optional<uint32_t> field_size =
          GetTargetByteSize(field.GetType(), type_system);
      if (!field_size)
        return std::nullopt;
      byte_size = std::max(byte_size, *field_size);
    }
    return byte_size;
  }
  case RegisterType::eRegisterTypeKindEnum:
  case RegisterType::eRegisterTypeKindFlags:
    return std::nullopt;
  }
}

CompilerType
RegisterTypeBuilderClang::GetRegisterType(const RegisterInfo &reg_info) {
  lldb::TypeSystemClangSP type_system =
      ScratchTypeSystemClang::GetForTarget(m_target);
  assert(type_system);

  if (m_cached_type_system.lock() != type_system) {
    m_type_cache.clear();
    m_cached_type_system = type_system;
  }

  if (!reg_info.register_type)
    return CompilerType();

  // Note that we do not check the type cache here because types can be nested.
  // There is a cache check in each of the Build<subtype> methods, and those
  // methods may call each other (Flags may use Enums for example).

  switch (reg_info.register_type->getKind()) {
  case RegisterType::eRegisterTypeKindBuiltin:
    return BuildType(reg_info.register_type, reg_info.byte_size, type_system);
  case RegisterType::eRegisterTypeKindFlags:
    return BuildFlagsType(
        llvm::dyn_cast<RegisterTypeFlags>(reg_info.register_type),
        reg_info.byte_size, type_system);
  case RegisterType::eRegisterTypeKindEnum:
    return BuildEnumType(
        llvm::dyn_cast<RegisterTypeEnum>(reg_info.register_type),
        reg_info.byte_size, type_system);
  case RegisterType::eRegisterTypeKindVector:
    return BuildType(reg_info.register_type, reg_info.byte_size, type_system);
  case RegisterType::eRegisterTypeKindUnion: {
    std::optional<uint32_t> byte_size =
        GetTargetByteSize(reg_info.register_type, type_system);
    if (!byte_size || *byte_size > reg_info.byte_size)
      return {};
    return BuildType(reg_info.register_type, *byte_size, type_system);
  }
  }
}

std::optional<CompilerType> RegisterTypeBuilderClang::GetExistingCompilerType(
    const RegisterType *register_type, uint32_t register_byte_size) {
  auto cached =
      m_type_cache.find({register_type->GetUID(), register_byte_size});
  if (cached != m_type_cache.end())
    return cached->second;

  return {};
}
