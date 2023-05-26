//===-- RegisterTypeBuilderClang.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "RegisterTypeBuilderClang.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/RegisterFlags.h"
#include "lldb/lldb-enumerations.h"

using namespace lldb_private;

LLDB_PLUGIN_DEFINE(RegisterTypeBuilderClang)

void RegisterTypeBuilderClang::Initialize() {
  static llvm::once_flag g_once_flag;
  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(), CreateInstance);
  });
}

void RegisterTypeBuilderClang::Terminate() {}

lldb::RegisterTypeBuilderSP
RegisterTypeBuilderClang::CreateInstance(Target &target) {
  return std::make_shared<RegisterTypeBuilderClang>(target);
}

RegisterTypeBuilderClang::RegisterTypeBuilderClang(Target &target)
    : m_target(target) {}

CompilerType RegisterTypeBuilderClang::GetRegisterType(
    const std::string &name, const lldb_private::RegisterFlags &flags,
    uint32_t byte_size) {
  lldb::TypeSystemClangSP type_system =
      ScratchTypeSystemClang::GetForTarget(m_target);
  assert(type_system);

  std::string register_type_name = "__lldb_register_fields_";
  register_type_name += name;
  // See if we have made this type before and can reuse it.
  CompilerType fields_type =
      type_system->GetTypeForIdentifier<clang::CXXRecordDecl>(
          ConstString(register_type_name.c_str()));

  if (!fields_type) {
    // In most ABI, a change of field type means a change in storage unit.
    // We want it all in one unit, so we use a field type the same as the
    // register's size.
    CompilerType field_uint_type =
        type_system->GetBuiltinTypeForEncodingAndBitSize(lldb::eEncodingUint,
                                                         byte_size * 8);

    fields_type = type_system->CreateRecordType(
        nullptr, OptionalClangModuleID(), lldb::eAccessPublic,
        register_type_name, clang::TTK_Struct, lldb::eLanguageTypeC);
    type_system->StartTagDeclarationDefinition(fields_type);

    // We assume that RegisterFlags has padded and sorted the fields
    // already.
    for (const RegisterFlags::Field &field : flags.GetFields()) {
      type_system->AddFieldToRecordType(fields_type, field.GetName(),
                                        field_uint_type, lldb::eAccessPublic,
                                        field.GetSizeInBits());
    }

    type_system->CompleteTagDeclarationDefinition(fields_type);
    // So that the size of the type matches the size of the register.
    type_system->SetIsPacked(fields_type);

    // This should be true if RegisterFlags padded correctly.
    assert(*fields_type.GetByteSize(nullptr) == flags.GetSize());
  }

  return fields_type;
}
