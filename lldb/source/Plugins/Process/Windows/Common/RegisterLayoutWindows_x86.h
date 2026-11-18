//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROCESS_WINDOWS_COMMON_REGISTERLAYOUTWINDOWS_X86_H
#define LLDB_PLUGINS_PROCESS_WINDOWS_COMMON_REGISTERLAYOUTWINDOWS_X86_H
#include "lldb/Register/Descriptions/RegisterDescription_x86_32.h"
#include "lldb/Register/Descriptions/RegisterDescription_x86_64.h"
#include "lldb/Register/Layouts/Common/RegisterLayout_x86.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/ADT/STLExtras.h"
namespace lldb_private {

class RegisterLayoutWindows_x86 final : public RegisterLayout_x86 {
public:
  enum class Mode { Target, X86, X86_64 };
  enum class ContextKind { Native, Frame };

  explicit RegisterLayoutWindows_x86(
      const ArchSpec &arch, Mode mode,
      ContextKind context = ContextKind::Native) {
    auto description = BuildDescription(arch, mode, context);
    // Windows contexts access CONTEXT fields directly and have no saved banks.
    llvm::cantFail(AddRegisterLocations(
        *description, [](const RegisterDescriptor &) { return std::nullopt; }));
    llvm::cantFail(Finalize(std::move(description), arch.GetByteOrder()));
  }

  RegisterLayoutWindows_x86() = delete;
  RegisterLayoutWindows_x86(const RegisterLayoutWindows_x86 &) = delete;
  RegisterLayoutWindows_x86(RegisterLayoutWindows_x86 &&) = delete;

private:
  static std::shared_ptr<const RegisterDescription>
  BuildDescription(const ArchSpec &arch, Mode mode, ContextKind context) {
    const bool is_64 =
        mode == Mode::X86_64 ||
        (mode == Mode::Target && arch.GetMachine() == llvm::Triple::x86_64);
    const bool frame = context == ContextKind::Frame;
    std::shared_ptr<const RegisterDescription> description;

    if (is_64) {
      using Config = RegisterDescription_x86_64::Config;
      Config config;
      config.abi = Config::ABI::Windows;
      config.features = Flags(frame ? Config::eRegsetMaskAliases
                                    : Config::eRegsetMaskSegments);
      description = std::make_shared<RegisterDescription_x86_64>(config);
    } else {
      using Config = RegisterDescription_x86_32::Config;
      Config config;
      config.abi = Config::ABI::Windows;
      config.features = Flags(frame ? 0 : Config::eRegsetMaskSegments);
      description = std::make_shared<RegisterDescription_x86_32>(config);
    }

    std::vector<RegisterDescriptor> registers(
        description->GetRegisters().begin(), description->GetRegisters().end());
    for (auto &reg : registers) {
      // Native CONTEXT segment selectors are 16-bit values, unlike the
      // full-width slots in the common x86-64 description.
      if (!frame) {
        if (is_64 && llvm::is_contained({"cs", "ds", "es", "fs", "gs", "ss"},
                                        llvm::StringRef(reg.name)))
          reg.byte_size = 2;
        continue;
      }

      // Frame contexts access subregisters directly through their CONTEXT
      // fields and historically expose no dependency lists.
      reg.value_registers.clear();
      reg.invalidate_registers.clear();
      switch (reg.numbers[lldb::eRegisterKindGeneric]) {
      case LLDB_REGNUM_GENERIC_FP:
        reg.alt_name = "fp";
        break;
      case LLDB_REGNUM_GENERIC_SP:
        reg.alt_name = "sp";
        break;
      case LLDB_REGNUM_GENERIC_PC:
        reg.alt_name = "pc";
        break;
      case LLDB_REGNUM_GENERIC_FLAGS:
        reg.name = "eflags";
        reg.alt_name = "flags";
        reg.format = lldb::eFormatBinary;
        if (is_64) {
          reg.numbers[lldb::eRegisterKindDWARF] = LLDB_INVALID_REGNUM;
          reg.numbers[lldb::eRegisterKindEHFrame] = LLDB_INVALID_REGNUM;
        }
        break;
      }
    }

    return std::make_shared<RegisterDescription>(
        llvm::cantFail(RegisterDescription::Create(
            std::move(registers),
            std::vector<RegisterGroup>(description->GetGroups().begin(),
                                       description->GetGroups().end()),
            RegisterDescription::OwnedTypes(
                description->GetOwnedTypes().begin(),
                description->GetOwnedTypes().end()))));
  }
};

} // namespace lldb_private
#endif
