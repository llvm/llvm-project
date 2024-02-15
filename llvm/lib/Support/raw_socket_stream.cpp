//===-- llvm/Support/raw_socket_stream.cpp - Socket streams --*- C++ -*-===//
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

#include "llvm/Support/raw_socket_stream.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <atomic>

#ifndef _WIN32
#include <sys/socket.h>
#include <sys/un.h>
#else
#include "llvm/Support/Windows/WindowsSupport.h"
// winsock2.h must be included before afunix.h. Briefly turn off clang-format to
// avoid error.
// clang-format off
#include <winsock2.h>
#include <afunix.h>
// clang-format on
#include <io.h>
#endif // _WIN32

#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif

using namespace llvm;

#ifdef _WIN32
WSABalancer::WSABalancer() {
  WSADATA WsaData;
  ::memset(&WsaData, 0, sizeof(WsaData));
  if (WSAStartup(MAKEWORD(2, 2), &WsaData) != 0) {
    llvm::report_fatal_error("WSAStartup failed");
  }
}

WSABalancer::~WSABalancer() { WSACleanup(); }

#endif // _WIN32

static std::error_code getLastSocketErrorCode() {
#ifdef _WIN32
  return std::error_code(::WSAGetLastError(), std::system_category());
#else
  return std::error_code(errno, std::system_category());
#endif
}

static void closeFD(int FD) {
#ifdef _WIN32
  _close(FD);
#else
  ::close(FD);
#endif
}

static void unlinkFile(StringRef Path) {
#ifdef _WIN32
  _unlink(Path.str().c_str());
#else
  ::unlink(Path.str().c_str());
#endif
}

static Expected<int> getSocketFD(StringRef SocketPath) {
#ifdef _WIN32
  SOCKET MaybeWinsocket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (MaybeWinsocket == INVALID_SOCKET) {
#else
  int MaybeWinsocket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (MaybeWinsocket == -1) {
#endif // _WIN32
    return llvm::make_error<StringError>(getLastSocketErrorCode(),
                                         "Create socket failed");
  }

  struct sockaddr_un Addr;
  memset(&Addr, 0, sizeof(Addr));
  Addr.sun_family = AF_UNIX;
  strncpy(Addr.sun_path, SocketPath.str().c_str(), sizeof(Addr.sun_path) - 1);

  int status = connect(MaybeWinsocket, (struct sockaddr *)&Addr, sizeof(Addr));
  if (status == -1) {
    return llvm::make_error<StringError>(getLastSocketErrorCode(),
                                         "Connect socket failed");
  }
#ifdef _WIN32
  return _open_osfhandle(MaybeWinsocket, 0);
#else
  return MaybeWinsocket;
#endif // _WIN32
}

ListeningSocket::ListeningSocket(int SocketFD, StringRef SocketPath)
    : FD(SocketFD), SocketPath(SocketPath) {}

ListeningSocket::ListeningSocket(ListeningSocket &&LS)
    : FD(LS.FD.load()), SocketPath(LS.SocketPath) {

  LS.SocketPath.clear();
  LS.FD = -1;
}

Expected<ListeningSocket> ListeningSocket::createUnix(StringRef SocketPath,
                                                      int MaxBacklog) {

  // Handle instances where the target socket address already exists
  // ::bind will return std::errc:address_in_use if the socket address already
  // exists (e.g., file was not properly unlinked due to a crash) even if
  // another socket has not yet binded to that address
  if (llvm::sys::fs::exists(SocketPath)) {
    Expected<int> MaybeFD = getSocketFD(SocketPath);
    if (!MaybeFD) {

      // Regardless of error returned by getSocketFD notify caller that a file
      // already exists at the desired socket address
      consumeError(MaybeFD.takeError());
      return llvm::make_error<StringError>(
          std::make_error_code(std::errc::file_exists),
          "Socket address unavailable");
    }
    closeFD(std::move(*MaybeFD));

    // Notify caller that the provided socket address already has a bound socket
    return llvm::make_error<StringError>(
        std::make_error_code(std::errc::address_in_use),
        "Socket address unavailable");
  }

#ifdef _WIN32
  WSABalancer _;
  SOCKET MaybeWinsocket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (MaybeWinsocket == INVALID_SOCKET) {
#else
  int MaybeWinsocket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (MaybeWinsocket == -1) {
#endif
    return llvm::make_error<StringError>(getLastSocketErrorCode(),
                                         "socket create failed");
  }

  struct sockaddr_un Addr;
  memset(&Addr, 0, sizeof(Addr));
  Addr.sun_family = AF_UNIX;
  strncpy(Addr.sun_path, SocketPath.str().c_str(), sizeof(Addr.sun_path) - 1);

  if (::bind(MaybeWinsocket, (struct sockaddr *)&Addr, sizeof(Addr)) == -1) {
    std::error_code EC = getLastSocketErrorCode();
    ::close(MaybeWinsocket);
    return llvm::make_error<StringError>(EC, "Bind error");
  }
  if (::listen(MaybeWinsocket, MaxBacklog) == -1) {
    return llvm::make_error<StringError>(getLastSocketErrorCode(),
                                         "Listen error");
  }
  int UnixSocket;
#ifdef _WIN32
  UnixSocket = _open_osfhandle(MaybeWinsocket, 0);
#else
  UnixSocket = MaybeWinsocket;
#endif // _WIN32
  return ListeningSocket{UnixSocket, SocketPath};
}

Expected<std::unique_ptr<raw_socket_stream>>
ListeningSocket::accept(std::optional<std::chrono::microseconds> Timeout) {

  int SelectStatus;
  int AcceptFD;

#ifdef _WIN32
  SOCKET WinServerSock = _get_osfhandle(FD);
#endif

  fd_set Readfds;
  if (Timeout.has_value()) {
    timeval TV = {0, Timeout.value().count()};
    FD_ZERO(&Readfds);
#ifdef _WIN32
    FD_SET(WinServerSock, &Readfds);
#else
    FD_SET(FD, &Readfds);
#endif
    SelectStatus = ::select(FD + 1, &Readfds, NULL, NULL, &TV);
  } else
    SelectStatus = ::select(FD + 1, &Readfds, NULL, NULL, NULL);

  if (SelectStatus == -1)
    return llvm::make_error<StringError>(getLastSocketErrorCode(),
                                         "Select failed");
  else if (SelectStatus) {
#ifdef _WIN32
    SOCKET WinAcceptSock = ::accept(WinServerSock, NULL, NULL);
    AcceptFD = _open_osfhandle(WinAcceptSock, 0);
#else
    AcceptFD = ::accept(FD, NULL, NULL);
#endif
  } else
    return llvm::make_error<StringError>(
        std::make_error_code(std::errc::timed_out), "Accept timed out");

  if (AcceptFD == -1)
    return llvm::make_error<StringError>(getLastSocketErrorCode(),
                                         "Accept failed");

  return std::make_unique<raw_socket_stream>(AcceptFD);
}

void ListeningSocket::shutdown() {
  if (FD == -1)
    return;
  closeFD(FD);
  unlinkFile(SocketPath);
  FD = -1;
}

ListeningSocket::~ListeningSocket() { shutdown(); }

raw_socket_stream::raw_socket_stream(int SocketFD)
    : raw_fd_stream(SocketFD, true) {}

Expected<std::unique_ptr<raw_socket_stream>>
raw_socket_stream::createConnectedUnix(StringRef SocketPath) {
#ifdef _WIN32
  WSABalancer _;
#endif // _WIN32
  Expected<int> FD = getSocketFD(SocketPath);
  if (!FD)
    return FD.takeError();
  return std::make_unique<raw_socket_stream>(*FD);
}

raw_socket_stream::~raw_socket_stream() {}

