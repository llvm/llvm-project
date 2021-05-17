//===------------- OffloadingServer.cpp - Server Application --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Offloading server for remote host.
//
//===----------------------------------------------------------------------===//

#include <future>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <iostream>
#include <thread>

#include "Server.h"

using grpc::Server;
using grpc::ServerBuilder;

std::promise<void> ShutdownPromise;

int main() {
  RPCConfig Config;
  parseEnvironment(Config);

  RemoteOffloadImpl Service(Config.MaxSize, Config.BlockSize);

  ServerBuilder Builder;
  Builder.AddListeningPort(Config.ServerAddresses[0],
                           grpc::InsecureServerCredentials());
  Builder.RegisterService(&Service);
  Builder.SetMaxMessageSize(INT_MAX);
  std::unique_ptr<Server> Server(Builder.BuildAndStart());
  if (getDebugLevel())
    std::cerr << "Server listening on " << Config.ServerAddresses[0]
              << std::endl;

  auto WaitForServer = [&]() { Server->Wait(); };

  std::thread ServerThread(WaitForServer);

  auto ShutdownFuture = ShutdownPromise.get_future();
  ShutdownFuture.wait();
  Server->Shutdown();
  ServerThread.join();

  return 0;
}
