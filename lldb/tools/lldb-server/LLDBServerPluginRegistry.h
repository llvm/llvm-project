//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINREGISTRY_H
#define LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINREGISTRY_H

#include "Plugins/Accelerator/LLDBServerPluginAccelerator.h"

namespace lldb_private {

namespace lldb_server {

class LLDBServerPluginRegistry {
public:
  /// Identifies the category of a plugin.
  enum class PluginKind {
    Accelerator,
  };

  using AcceleratorFactory =
      std::function<std::unique_ptr<LLDBServerPluginAccelerator>(
          LLDBServerPluginAccelerator::GDBServer &, MainLoop &)>;

  /// Return a singleton instance of the registry that is initialized on first
  /// use and that knows of all available plugins.
  static LLDBServerPluginRegistry &Instance();

  /// Instantiate all registered accelerator plugins and return those that
  /// successfully activated (TryActivate() returned true).
  std::vector<std::unique_ptr<LLDBServerPluginAccelerator>>
  TryInstantiateAllAcceleratorPlugins(
      LLDBServerPluginAccelerator::GDBServer &gdb_server, MainLoop &main_loop);

private:
  /// Initialize and register all known plugins.
  void Initialize();

  /// Register an accelerator plugin factory.
  void Register(PluginKind kind, AcceleratorFactory factory);

  LLDBServerPluginRegistry() = default;
  std::vector<AcceleratorFactory> m_accelerator_factories;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_LLDBSERVERPLUGINREGISTRY_H
