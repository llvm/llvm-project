//===--------------------- SocketTestUtilities.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_HOST_SOCKETTESTUTILITIES_H
#define LLDB_UNITTESTS_HOST_SOCKETTESTUTILITIES_H

#include <cstdio>
#include <functional>
#include <thread>

#include "lldb/Host/Config.h"
#include "lldb/Host/Socket.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Host/common/UDPSocket.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"

#if LLDB_ENABLE_POSIX
#include "lldb/Host/posix/DomainSocket.h"
#endif

namespace lldb_private {
template <typename SocketType>
void CreateConnectedSockets(
    llvm::StringRef listen_remote_address,
    const std::function<std::string(const SocketType &)> &get_connect_addr,
    std::unique_ptr<SocketType> *a_up, std::unique_ptr<SocketType> *b_up);
bool CreateTCPConnectedSockets(std::string listen_remote_ip,
                               std::unique_ptr<TCPSocket> *a_up,
                               std::unique_ptr<TCPSocket> *b_up);
#if LLDB_ENABLE_POSIX
void CreateDomainConnectedSockets(llvm::StringRef path,
                                  std::unique_ptr<DomainSocket> *a_up,
                                  std::unique_ptr<DomainSocket> *b_up);
#endif

bool HostSupportsIPv6();
bool HostSupportsIPv4();
} // namespace lldb_private

#endif
