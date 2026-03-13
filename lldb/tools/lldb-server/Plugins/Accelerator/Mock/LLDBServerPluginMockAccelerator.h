//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_LLDBSERVERPLUGINMOCKACCELERATOR_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_LLDBSERVERPLUGINMOCKACCELERATOR_H

#include "../LLDBServerPluginAccelerator.h"

namespace lldb_private {
namespace lldb_server {

/// A mock accelerator plugin used for testing the accelerator plugin
/// infrastructure.
///
/// This plugin is enabled when the environment variable
/// LLDB_SERVER_ENABLE_MOCK_ACCELERATOR_PLUGIN is set to any non-empty value.
class LLDBServerPluginMockAccelerator : public LLDBServerPluginAccelerator {
public:
  /// Environment variable that enables this plugin.
  static constexpr const char *kEnableEnvVar =
      "LLDB_SERVER_ENABLE_MOCK_ACCELERATOR_PLUGIN";

  LLDBServerPluginMockAccelerator(GDBServer &gdb_server, MainLoop &main_loop);

  llvm::StringRef GetPluginName() override;
  bool TryActivate() override;
  void Teardown() override {}
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_LLDBSERVERPLUGINMOCKACCELERATOR_H
