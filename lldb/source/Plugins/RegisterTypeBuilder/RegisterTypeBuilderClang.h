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
  CompilerType BuildEnumType(const RegisterTypeEnum *enum_type_info,
                             uint32_t register_byte_size,
                             lldb::TypeSystemClangSP type_system);

  CompilerType BuildFlagsType(const RegisterTypeFlags *flags_info,
                              uint32_t register_byte_size,
                              lldb::TypeSystemClangSP type_system);

  Target &m_target;

  // A cache of previously created types. We do not cache by element ID because
  // IDs are not unique across xml <feature> elements and this class does not
  // know anything about features.
  //
  // The key contains the process-wide UID of the type and the size of the
  // register we made it for. Some types (enums for example) use the register
  // size in their type and must be rebuilt for a different size.
  //
  // 8 is chosen because types are only made when needed, and most lldb commands
  // do not need them.
  llvm::SmallDenseMap<std::pair<uint64_t, uint32_t>, CompilerType, 8>
      m_type_cache;
  std::weak_ptr<TypeSystemClang> m_cached_type_system;

  std::optional<CompilerType>
  GetExistingCompilerType(const RegisterType *register_type,
                          uint32_t register_byte_size);
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_REGISTERTYPEBUILDER_REGISTERTYPEBUILDERCLANG_H
