//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_LLDBSERVERPLUGINACCELERATOR_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_LLDBSERVERPLUGINACCELERATOR_H

#include "llvm/ADT/StringRef.h"
#include "lldb/Host/MainLoop.h"

namespace lldb_private {

namespace process_gdb_remote {
class GDBRemoteCommunicationServerLLGS;
} // namespace process_gdb_remote

namespace lldb_server {

/// Abstract base class for lldb-server accelerator plugins.
///
/// An accelerator plugin allows lldb-server to support debugging of hardware
/// accelerators (e.g. GPUs, FPGAs) alongside the native host process.
class LLDBServerPluginAccelerator {
public:
  using GDBServer = process_gdb_remote::GDBRemoteCommunicationServerLLGS;

  LLDBServerPluginAccelerator(GDBServer &gdb_server, MainLoop &main_loop);
  virtual ~LLDBServerPluginAccelerator() = default;

  /// Returns a short, unique name identifying this plugin (e.g. "mock-accel").
  virtual llvm::StringRef GetPluginName() = 0;

  /// Attempts to activate the plugin if it should be enabled in the current
  /// environment (e.g. required hardware is present, an enabling environment
  /// variable is set, etc.). Returns true if the plugin was activated.
  virtual bool TryActivate() = 0;

  /// Called before lldb-server exits to allow the plugin to perform cleanup.
  /// This is called for all plugins that successfully activated.
  virtual void Teardown() = 0;

protected:
  GDBServer &m_gdb_server;
  MainLoop &m_main_loop;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_LLDBSERVERPLUGINACCELERATOR_H
