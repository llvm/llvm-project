//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_WASM_PLATFORMWASMREMOTEGDBSERVER_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_WASM_PLATFORMWASMREMOTEGDBSERVER_H

#include "Plugins/Platform/gdb-server/PlatformRemoteGDBServer.h"

namespace lldb_private {

class PlatformWasmRemoteGDBServer
    : public platform_gdb_server::PlatformRemoteGDBServer {
public:
  PlatformWasmRemoteGDBServer() = default;

  ~PlatformWasmRemoteGDBServer() override;

  virtual llvm::StringRef GetDefaultProcessPluginName() const override;

private:
  PlatformWasmRemoteGDBServer(const PlatformWasmRemoteGDBServer &) = delete;
  const PlatformWasmRemoteGDBServer &
  operator=(const PlatformWasmRemoteGDBServer &) = delete;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_WASM_PLATFORMWASMREMOTEGDBSERVER_H
