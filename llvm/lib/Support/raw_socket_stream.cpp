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
  return errnoAsErrorCode();
#endif
}

ListeningSocket::ListeningSocket(int SocketFD, StringRef SocketPath)
    : FD(SocketFD), SocketPath(SocketPath) {}

ListeningSocket::ListeningSocket(ListeningSocket &&LS)
    : FD(LS.FD), SocketPath(LS.SocketPath) {
  LS.FD = -1;
}

Expected<ListeningSocket> ListeningSocket::createUnix(StringRef SocketPath,
                                                      int MaxBacklog) {

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

  if (bind(MaybeWinsocket, (struct sockaddr *)&Addr, sizeof(Addr)) == -1) {
    std::error_code Err = getLastSocketErrorCode();
    if (Err == std::errc::address_in_use)
      ::close(MaybeWinsocket);
    return llvm::make_error<StringError>(Err, "Bind error");
  }
  if (listen(MaybeWinsocket, MaxBacklog) == -1) {
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

Expected<std::unique_ptr<raw_socket_stream>> ListeningSocket::accept() {
  int AcceptFD;
#ifdef _WIN32
  SOCKET WinServerSock = _get_osfhandle(FD);
  SOCKET WinAcceptSock = ::accept(WinServerSock, NULL, NULL);
  AcceptFD = _open_osfhandle(WinAcceptSock, 0);
#else
  AcceptFD = ::accept(FD, NULL, NULL);
#endif //_WIN32
  if (AcceptFD == -1)
    return llvm::make_error<StringError>(getLastSocketErrorCode(),
                                         "Accept failed");
  return std::make_unique<raw_socket_stream>(AcceptFD);
}

ListeningSocket::~ListeningSocket() {
  if (FD == -1)
    return;
  ::close(FD);
  unlink(SocketPath.c_str());
}

static Expected<int> GetSocketFD(StringRef SocketPath) {
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

raw_socket_stream::raw_socket_stream(int SocketFD)
    : raw_fd_stream(SocketFD, true) {}

Expected<std::unique_ptr<raw_socket_stream>>
raw_socket_stream::createConnectedUnix(StringRef SocketPath) {
#ifdef _WIN32
  WSABalancer _;
#endif // _WIN32
  Expected<int> FD = GetSocketFD(SocketPath);
  if (!FD)
    return FD.takeError();
  return std::make_unique<raw_socket_stream>(*FD);
}

raw_socket_stream::~raw_socket_stream() {}

