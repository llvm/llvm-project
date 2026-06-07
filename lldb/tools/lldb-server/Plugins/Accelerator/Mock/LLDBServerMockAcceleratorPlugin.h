//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_MOCK_LLDBSERVERMOCKACCELERATORPLUGIN_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_MOCK_LLDBSERVERMOCKACCELERATORPLUGIN_H

#include "Plugins/Process/gdb-remote/LLDBServerAcceleratorPlugin.h"

namespace lldb_private {
namespace lldb_server {

class LLDBServerMockAcceleratorPlugin : public LLDBServerAcceleratorPlugin {
public:
  LLDBServerMockAcceleratorPlugin(GDBServer &gdb_server, MainLoop &main_loop);

  llvm::StringRef GetPluginName() override;
  std::optional<AcceleratorActions> GetInitializeActions() override;
  llvm::Expected<AcceleratorBreakpointHitResponse>
  BreakpointWasHit(AcceleratorBreakpointHitArgs &args) override;

  std::optional<AcceleratorDynamicLoaderResponse>
  GetDynamicLoaderLibraryInfos(
      const AcceleratorDynamicLoaderArgs &args) override;

private:
  static constexpr int64_t kBreakpointIDInitialize = 1;
  static constexpr int64_t kBreakpointIDExit = 2;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_MOCK_LLDBSERVERMOCKACCELERATORPLUGIN_H
