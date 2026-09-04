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
#include "lldb/Utility/RegisterType.h"
#include "lldb/lldb-enumerations.h"

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
    return BuildBuiltinType(
        llvm::cast<RegisterTypeBuiltin>(reg_info.register_type),
        reg_info.byte_size, type_system);
  case RegisterType::eRegisterTypeKindFlags:
    return BuildFlagsType(
        llvm::dyn_cast<RegisterTypeFlags>(reg_info.register_type),
        reg_info.byte_size, type_system);
  case RegisterType::eRegisterTypeKindEnum:
    return BuildEnumType(
        llvm::dyn_cast<RegisterTypeEnum>(reg_info.register_type),
        reg_info.byte_size, type_system);
  case RegisterType::eRegisterTypeKindVector:
    return {};
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
