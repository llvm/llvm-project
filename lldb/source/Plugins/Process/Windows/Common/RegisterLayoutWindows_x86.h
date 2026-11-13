//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROCESS_WINDOWS_COMMON_REGISTERLAYOUTWINDOWS_X86_H
#define LLDB_PLUGINS_PROCESS_WINDOWS_COMMON_REGISTERLAYOUTWINDOWS_X86_H
#include "lldb/Register/Common/RegisterInfo_x86.h"
#include "llvm/ADT/STLExtras.h"
namespace lldb_private {
enum class WindowsX86ContextKind { Native, Frame };
inline RegisterLayout CreateWindowsX86RegisterLayout(
    const ArchSpec &arch, X86RegisterMode mode,
    WindowsX86ContextKind context = WindowsX86ContextKind::Native) {
  auto describe = [&]() -> std::shared_ptr<const RegisterDescription> {
    const bool is_64 = mode == X86RegisterMode::X86_64 ||
                       (mode == X86RegisterMode::Target &&
                        arch.GetMachine() == llvm::Triple::x86_64);
    const bool frame = context == WindowsX86ContextKind::Frame;
    X86RegisterProfile profile;
    profile.gpr_aliases = frame && is_64;
    profile.segments = !frame;
    profile.x87 = profile.avx = profile.mpx = profile.debug = false;
    profile.simd = is_64;
    auto description = RegisterInfo_x86::CreateDescription(
        arch, false, X86RegisterABI::Windows, mode, profile);
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
  };
  return llvm::cantFail(
      RegisterLayout::Create(describe(), RegisterLayout(arch.GetByteOrder())));
}
} // namespace lldb_private
#endif
