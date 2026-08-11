//===-- RegisterTypeBuilderClang.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_REGISTERTYPEBUILDER_REGISTERTYPEBUILDERCLANG_H
#define LLDB_SOURCE_PLUGINS_REGISTERTYPEBUILDER_REGISTERTYPEBUILDERCLANG_H

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Target/RegisterTypeBuilder.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/RegisterTypeFlags.h"

namespace lldb_private {
class RegisterTypeBuilderClang : public RegisterTypeBuilder {
public:
  RegisterTypeBuilderClang(Target &target);

  static void Initialize();
  static void Terminate();
  static llvm::StringRef GetPluginNameStatic() {
    return "register-types-clang";
  }
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
  static llvm::StringRef GetPluginDescriptionStatic() {
    return "Create register types using TypeSystemClang";
  }
  static lldb::RegisterTypeBuilderSP CreateInstance(Target &target);

  CompilerType GetRegisterType(const RegisterInfo &reg_info) override;

private:
  CompilerType BuildEnumType(const RegisterTypeEnum &enum_type_info,
                             uint32_t register_byte_size,
                             lldb::TypeSystemClangSP type_system);

  CompilerType BuildFlagsType(const RegisterTypeFlags &flags_info,
                              uint32_t register_byte_size,
                              lldb::TypeSystemClangSP type_system);

  Target &m_target;
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_REGISTERTYPEBUILDER_REGISTERTYPEBUILDERCLANG_H
