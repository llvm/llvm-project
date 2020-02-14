//===-- AArch64.h ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIAArch64.h"
#include "ABIMacOSX_arm64.h"
#include "ABISysV_arm64.h"
#include "lldb/Core/PluginManager.h"

LLDB_PLUGIN(ABIAArch64)

void ABIAArch64::Initialize() {
  ABISysV_arm64::Initialize();
  ABIMacOSX_arm64::Initialize();
}

void ABIAArch64::Terminate() {
  ABISysV_arm64::Terminate();
  ABIMacOSX_arm64::Terminate();
}
