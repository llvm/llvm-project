//===- llvm/RemoteCachingService/RemoteCacheServer.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REMOTECACHINGSERVICE_REMOTECACHESERVER_H
#define LLVM_REMOTECACHINGSERVICE_REMOTECACHESERVER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm::cas {

class ActionCache;
class ObjectStore;

namespace remote {

/// A gRPC service for the remote caching protocol.
class RemoteCacheServer {
public:
  /// Returns a gRPC server for the remote caching protocol.
  RemoteCacheServer(StringRef SocketPath, StringRef TempPath,
                    std::unique_ptr<ObjectStore> CAS,
                    std::unique_ptr<ActionCache> Cache);

  ~RemoteCacheServer();

  void Start();
  void Listen();
  void Shutdown();

private:
  class Implementation;

  RemoteCacheServer(std::unique_ptr<RemoteCacheServer::Implementation> Impl);
  std::unique_ptr<RemoteCacheServer::Implementation> Impl;
};

} // namespace remote
} // namespace llvm::cas

#endif
