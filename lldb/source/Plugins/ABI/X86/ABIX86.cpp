//===-- X86.h -------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIX86.h"
#include "ABIMacOSX_i386.h"
#include "ABISysV_i386.h"
#include "ABISysV_x86_64.h"
#include "ABIWindows_x86_64.h"
#include "lldb/Core/PluginManager.h"

LLDB_PLUGIN_DEFINE(ABIX86)

void ABIX86::Initialize() {
  ABIMacOSX_i386::Initialize();
  ABISysV_i386::Initialize();
  ABISysV_x86_64::Initialize();
  ABIWindows_x86_64::Initialize();
}

void ABIX86::Terminate() {
  ABIMacOSX_i386::Terminate();
  ABISysV_i386::Terminate();
  ABISysV_x86_64::Terminate();
  ABIWindows_x86_64::Terminate();
}

uint32_t ABIX86::GetGenericNum(llvm::StringRef name) {
  return llvm::StringSwitch<uint32_t>(name)
      .Case("eip", LLDB_REGNUM_GENERIC_PC)
      .Case("esp", LLDB_REGNUM_GENERIC_SP)
      .Case("ebp", LLDB_REGNUM_GENERIC_FP)
      .Case("eflags", LLDB_REGNUM_GENERIC_FLAGS)
      .Case("edi", LLDB_REGNUM_GENERIC_ARG1)
      .Case("esi", LLDB_REGNUM_GENERIC_ARG2)
      .Case("edx", LLDB_REGNUM_GENERIC_ARG3)
      .Case("ecx", LLDB_REGNUM_GENERIC_ARG4)
      .Default(LLDB_INVALID_REGNUM);
}
