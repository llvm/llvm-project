//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PlatformWasmRemoteGDBServer.h"

using namespace lldb_private;

PlatformWasmRemoteGDBServer::~PlatformWasmRemoteGDBServer() {}

llvm::StringRef
PlatformWasmRemoteGDBServer::GetDefaultProcessPluginName() const {
  return "wasm";
}
