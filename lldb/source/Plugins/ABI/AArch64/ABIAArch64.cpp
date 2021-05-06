//===-- AArch66.h ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIAArch64.h"
#include "ABIMacOSX_arm64.h"
#include "ABISysV_arm64.h"
#include "Utility/ARM64_DWARF_Registers.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"

LLDB_PLUGIN_DEFINE(ABIAArch64)

void ABIAArch64::Initialize() {
  ABISysV_arm64::Initialize();
  ABIMacOSX_arm64::Initialize();
}

void ABIAArch64::Terminate() {
  ABISysV_arm64::Terminate();
  ABIMacOSX_arm64::Terminate();
}

lldb::addr_t ABIAArch64::FixCodeAddress(lldb::addr_t pc) {
  if (lldb::ProcessSP process_sp = GetProcessSP())
    return FixAddress(pc, process_sp->GetCodeAddressMask());
  return pc;
}

lldb::addr_t ABIAArch64::FixDataAddress(lldb::addr_t pc) {
  if (lldb::ProcessSP process_sp = GetProcessSP())
    return FixAddress(pc, process_sp->GetDataAddressMask());
  return pc;
}

std::pair<uint32_t, uint32_t>
ABIAArch64::GetEHAndDWARFNums(llvm::StringRef name) {
  if (name == "pc")
    return {LLDB_INVALID_REGNUM, arm64_dwarf::pc};
  if (name == "cpsr")
    return {LLDB_INVALID_REGNUM, arm64_dwarf::cpsr};
  return MCBasedABI::GetEHAndDWARFNums(name);
}

std::string ABIAArch64::GetMCName(std::string reg) {
  MapRegisterName(reg, "v", "q");
  MapRegisterName(reg, "x29", "fp");
  MapRegisterName(reg, "x30", "lr");
  return reg;
}
uint32_t ABIAArch64::GetGenericNum(llvm::StringRef name) {
  return llvm::StringSwitch<uint32_t>(name)
      .Case("pc", LLDB_REGNUM_GENERIC_PC)
      .Case("lr", LLDB_REGNUM_GENERIC_RA)
      .Case("sp", LLDB_REGNUM_GENERIC_SP)
      .Case("fp", LLDB_REGNUM_GENERIC_FP)
      .Case("cpsr", LLDB_REGNUM_GENERIC_FLAGS)
      .Case("x0", LLDB_REGNUM_GENERIC_ARG1)
      .Case("x1", LLDB_REGNUM_GENERIC_ARG2)
      .Case("x2", LLDB_REGNUM_GENERIC_ARG3)
      .Case("x3", LLDB_REGNUM_GENERIC_ARG4)
      .Case("x4", LLDB_REGNUM_GENERIC_ARG5)
      .Case("x5", LLDB_REGNUM_GENERIC_ARG6)
      .Case("x6", LLDB_REGNUM_GENERIC_ARG7)
      .Case("x7", LLDB_REGNUM_GENERIC_ARG8)
      .Default(LLDB_INVALID_REGNUM);
}
