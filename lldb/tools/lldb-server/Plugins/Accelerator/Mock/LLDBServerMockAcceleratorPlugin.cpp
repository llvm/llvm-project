//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerMockAcceleratorPlugin.h"

using namespace lldb_private;
using namespace lldb_private::lldb_server;

LLDBServerMockAcceleratorPlugin::LLDBServerMockAcceleratorPlugin(
    GDBServer &gdb_server, MainLoop &main_loop)
    : LLDBServerAcceleratorPlugin(gdb_server, main_loop) {}

llvm::StringRef LLDBServerMockAcceleratorPlugin::GetPluginName() {
  return "mock";
}

std::optional<AcceleratorActions>
LLDBServerMockAcceleratorPlugin::GetInitializeActions() {
  return AcceleratorActions(GetPluginName(), 0);
}
