//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerPluginMockAccelerator.h"

#include "llvm/Support/Process.h"

using namespace lldb_private::lldb_server;

LLDBServerPluginMockAccelerator::LLDBServerPluginMockAccelerator(
    GDBServer &gdb_server, MainLoop &main_loop)
    : LLDBServerPluginAccelerator(gdb_server, main_loop) {}

llvm::StringRef LLDBServerPluginMockAccelerator::GetPluginName() {
  return "mock-accelerator";
}

bool LLDBServerPluginMockAccelerator::TryActivate() {
  // Check if the plugin should be enabled
  std::optional<std::string> val = llvm::sys::Process::GetEnv(kEnableEnvVar);
  if (!val || val->empty())
    return false;

  // Activate the plugin by writing the parent PID to the test file
  std::error_code ec;
  llvm::raw_fd_ostream out("/tmp/accelerator_plugin_test.txt", ec,
                          llvm::sys::fs::OF_Text);
  out << llvm::sys::Process::getProcessId() << '\n';

  return true;
}
