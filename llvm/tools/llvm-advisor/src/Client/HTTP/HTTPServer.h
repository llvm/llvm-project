//===------------------- HTTPServer.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HTTP server for REST API and embedded web UI.
// Accepts HTTP connections and routes to handlers.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Client/CoreClient.h"
#include "llvm/Support/ThreadPool.h"

namespace llvm::advisor {

/// HTTP server for the advisor REST API and embedded UI.
class HTTPServer {
public:
  HTTPServer(CoreClient &Client, unsigned Port)
      : Client(Client), Port(Port),
        Pool(llvm::ThreadPoolStrategy{llvm::hardware_concurrency()}),
        ShutdownFlag(false) {}
  /// Start serving requests on the configured port.
  Error run();
  /// Request graceful shutdown.
  void shutdown();

private:
  CoreClient &Client;
  unsigned Port;
  llvm::DefaultThreadPool Pool;
  std::atomic<bool> ShutdownFlag;
  int PipeFD[2] = {-1, -1};
  std::string AuthToken;

  bool checkAuth(const std::string &AuthHeader) const;
};

} // namespace llvm::advisor
