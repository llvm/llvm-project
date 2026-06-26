//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/DomainSocketWindows.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <chrono>
#include <memory>

using namespace lldb;
using namespace lldb_private;

llvm::Expected<DomainSocket::Pair> DomainSocketWindows::CreatePair() {
  // Windows has no socketpair(). Emulate it the same way TCPSocket::CreatePair
  // does for loopback TCP: bind a listener to a unique temporary path, connect
  // a client to it, and accept. AF_UNIX SOCK_STREAM connect() completes once
  // the connection is queued (backlog >= 1), so a single thread can connect
  // and then accept without deadlocking.
  llvm::SmallString<128> model;
  llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/true, model);
  llvm::sys::path::append(model, "lldb-domain-socketpair-%%%%%%%%.sock");
  llvm::SmallString<128> path;
  llvm::sys::fs::createUniquePath(model, path, /*MakeAbsolute=*/false);
  auto remove_file =
      llvm::make_scope_exit([&] { llvm::sys::fs::remove(path); });

  auto listen_socket =
      std::make_unique<DomainSocketWindows>(/*should_close=*/true);
  if (Status error = listen_socket->Listen(path, /*backlog=*/1); error.Fail())
    return error.takeError();

  auto connect_socket =
      std::make_unique<DomainSocketWindows>(/*should_close=*/true);
  if (Status error = connect_socket->Connect(path); error.Fail())
    return error.takeError();

  // The connection is already queued, so a short timeout is sufficient.
  Socket *accept_socket = nullptr;
  if (Status error =
          listen_socket->Accept(std::chrono::seconds(1), accept_socket);
      error.Fail())
    return error.takeError();

  return Pair(std::move(connect_socket),
              std::unique_ptr<DomainSocket>(
                  static_cast<DomainSocket *>(accept_socket)));
}
