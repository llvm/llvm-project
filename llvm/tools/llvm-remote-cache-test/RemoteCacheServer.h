//===-- RemoteCacheServer.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_REMOTECACHETEST_REMOTECACHESERVER_H
#define LLVM_TOOLS_REMOTECACHETEST_REMOTECACHESERVER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace remote_cache_test {

class RemoteCacheProvider;
class RemoteCacheServer;

/// Returns a gRPC server for the remote caching protocol, with \p CacheProvider
/// doing the actual work of storing and retrieving the data.
RemoteCacheServer
createServer(StringRef SocketPath,
             std::unique_ptr<RemoteCacheProvider> CacheProvider);

/// A gRPC service for the remote caching protocol.
class RemoteCacheServer {
public:
  ~RemoteCacheServer();

  void Run();
  void Shutdown();

private:
  class Implementation;

  RemoteCacheServer(std::unique_ptr<RemoteCacheServer::Implementation> Impl);
  std::unique_ptr<RemoteCacheServer::Implementation> Impl;

  friend RemoteCacheServer
  createServer(StringRef SocketPath,
               std::unique_ptr<RemoteCacheProvider> CacheProvider);
};

} // namespace remote_cache_test
} // namespace llvm

#endif
