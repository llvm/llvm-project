//===-- llvm/Support/raw_socket_stream.h - Socket streams --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains raw_ostream implementations for streams to communicate
// via UNIX sockets
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RAW_SOCKET_STREAM_H
#define LLVM_SUPPORT_RAW_SOCKET_STREAM_H

#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <chrono>

namespace llvm {

class raw_socket_stream;

// Make sure that calls to WSAStartup and WSACleanup are balanced.
#ifdef _WIN32
class WSABalancer {
public:
  WSABalancer();
  ~WSABalancer();
};
#endif // _WIN32

class ListeningSocket {
  std::atomic<int> FD;
  std::string SocketPath;
  ListeningSocket(int SocketFD, StringRef SocketPath);
#ifdef _WIN32
  WSABalancer _;
#endif // _WIN32

public:
  ~ListeningSocket();
  ListeningSocket(ListeningSocket &&LS);
  ListeningSocket(const ListeningSocket &LS) = delete;
  ListeningSocket &operator=(const ListeningSocket &) = delete;

  void shutdown();

  Expected<std::unique_ptr<raw_socket_stream>>
  accept(std::optional<std::chrono::microseconds> Timeout = std::nullopt);

  static Expected<ListeningSocket> createUnix(
      StringRef SocketPath,
      int MaxBacklog = llvm::hardware_concurrency().compute_thread_count());
};

class raw_socket_stream : public raw_fd_stream {
  uint64_t current_pos() const override { return 0; }
#ifdef _WIN32
  WSABalancer _;
#endif // _WIN32

public:
  // TODO: Should probably be private
  raw_socket_stream(int SocketFD);
  /// Create a \p raw_socket_stream connected to the Unix domain socket at \p
  /// SocketPath.
  static Expected<std::unique_ptr<raw_socket_stream>>
  createConnectedUnix(StringRef SocketPath);
  ~raw_socket_stream();
};

} // end namespace llvm

#endif
