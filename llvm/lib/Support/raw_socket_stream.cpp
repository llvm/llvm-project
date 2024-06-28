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
#include <fcntl.h>
#include <thread>

#ifndef _WIN32
#include <poll.h>
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

static sockaddr_un setSocketAddr(StringRef SocketPath) {
  struct sockaddr_un Addr;
  memset(&Addr, 0, sizeof(Addr));
  Addr.sun_family = AF_UNIX;
  strncpy(Addr.sun_path, SocketPath.str().c_str(), sizeof(Addr.sun_path) - 1);
  return Addr;
}

static Expected<int> getSocketFD(StringRef SocketPath) {
#ifdef _WIN32
  SOCKET Socket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (Socket == INVALID_SOCKET) {
#else
  int Socket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (Socket == -1) {
#endif // _WIN32
    return llvm::make_error<StringError>(getLastSocketErrorCode(),
                                         "Create socket failed");
  }

  struct sockaddr_un Addr = setSocketAddr(SocketPath);
  if (::connect(Socket, (struct sockaddr *)&Addr, sizeof(Addr)) == -1)
    return llvm::make_error<StringError>(getLastSocketErrorCode(),
                                         "Connect socket failed");

#ifdef _WIN32
  return _open_osfhandle(Socket, 0);
#else
  return Socket;
#endif // _WIN32
}

ListeningSocket::ListeningSocket(int SocketFD, StringRef SocketPath,
                                 int PipeFD[2])
    : FD(SocketFD), SocketPath(SocketPath), PipeFD{PipeFD[0], PipeFD[1]} {}

ListeningSocket::ListeningSocket(ListeningSocket &&LS)
    : FD(LS.FD.load()), SocketPath(LS.SocketPath),
      PipeFD{LS.PipeFD[0], LS.PipeFD[1]} {

  LS.FD = -1;
  LS.SocketPath.clear();
  LS.PipeFD[0] = -1;
  LS.PipeFD[1] = -1;
}

Expected<ListeningSocket> ListeningSocket::createUnix(StringRef SocketPath,
                                                      int MaxBacklog) {

  // Handle instances where the target socket address already exists and
  // differentiate between a preexisting file with and without a bound socket
  //
  // ::bind will return std::errc:address_in_use if a file at the socket address
  // already exists (e.g., the file was not properly unlinked due to a crash)
  // even if another socket has not yet binded to that address
  if (llvm::sys::fs::exists(SocketPath)) {
    Expected<int> MaybeFD = getSocketFD(SocketPath);
    if (!MaybeFD) {

      // Regardless of the error, notify the caller that a file already exists
      // at the desired socket address and that there is no bound socket at that
      // address. The file must be removed before ::bind can use the address
      consumeError(MaybeFD.takeError());
      return llvm::make_error<StringError>(
          std::make_error_code(std::errc::file_exists),
          "Socket address unavailable");
    }
    ::close(std::move(*MaybeFD));

    // Notify caller that the provided socket address already has a bound socket
    return llvm::make_error<StringError>(
        std::make_error_code(std::errc::address_in_use),
        "Socket address unavailable");
  }

#ifdef _WIN32
  WSABalancer _;
  SOCKET Socket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (Socket == INVALID_SOCKET)
#else
  int Socket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (Socket == -1)
#endif
    return llvm::make_error<StringError>(getLastSocketErrorCode(),
                                         "socket create failed");

  struct sockaddr_un Addr = setSocketAddr(SocketPath);
  if (::bind(Socket, (struct sockaddr *)&Addr, sizeof(Addr)) == -1) {
    // Grab error code from call to ::bind before calling ::close
    std::error_code EC = getLastSocketErrorCode();
    ::close(Socket);
    return llvm::make_error<StringError>(EC, "Bind error");
  }

  // Mark socket as passive so incoming connections can be accepted
  if (::listen(Socket, MaxBacklog) == -1)
    return llvm::make_error<StringError>(getLastSocketErrorCode(),
                                         "Listen error");

  int PipeFD[2];
#ifdef _WIN32
  // Reserve 1 byte for the pipe and use default textmode
  if (::_pipe(PipeFD, 1, 0) == -1)
#else
  if (::pipe(PipeFD) == -1)
#endif // _WIN32
    return llvm::make_error<StringError>(getLastSocketErrorCode(),
                                         "pipe failed");

#ifdef _WIN32
  return ListeningSocket{_open_osfhandle(Socket, 0), SocketPath, PipeFD};
#else
  return ListeningSocket{Socket, SocketPath, PipeFD};
#endif // _WIN32
}

Expected<std::unique_ptr<raw_socket_stream>>
ListeningSocket::accept(std::chrono::milliseconds Timeout) {

  struct pollfd FDs[2];
  FDs[0].events = POLLIN;
#ifdef _WIN32
  SOCKET WinServerSock = _get_osfhandle(FD);
  FDs[0].fd = WinServerSock;
#else
  FDs[0].fd = FD;
#endif
  FDs[1].events = POLLIN;
  FDs[1].fd = PipeFD[0];

  // Keep track of how much time has passed in case poll is interupted by a
  // signal and needs to be recalled
  int RemainingTime = Timeout.count();
  std::chrono::milliseconds ElapsedTime = std::chrono::milliseconds(0);
  int PollStatus = -1;

  while (PollStatus == -1 && (Timeout.count() == -1 || ElapsedTime < Timeout)) {
    if (Timeout.count() != -1)
      RemainingTime -= ElapsedTime.count();

    auto Start = std::chrono::steady_clock::now();
#ifdef _WIN32
    PollStatus = WSAPoll(FDs, 2, RemainingTime);
#else
    PollStatus = ::poll(FDs, 2, RemainingTime);
#endif
    // If FD equals -1 then ListeningSocket::shutdown has been called and it is
    // appropriate to return operation_canceled
    if (FD.load() == -1)
      return llvm::make_error<StringError>(
          std::make_error_code(std::errc::operation_canceled),
          "Accept canceled");

#if _WIN32
    if (PollStatus == SOCKET_ERROR) {
#else
    if (PollStatus == -1) {
#endif
      std::error_code PollErrCode = getLastSocketErrorCode();
      // Ignore EINTR (signal occured before any request event) and retry
      if (PollErrCode != std::errc::interrupted)
        return llvm::make_error<StringError>(PollErrCode, "FD poll failed");
    }
    if (PollStatus == 0)
      return llvm::make_error<StringError>(
          std::make_error_code(std::errc::timed_out),
          "No client requests within timeout window");

    if (FDs[0].revents & POLLNVAL)
      return llvm::make_error<StringError>(
          std::make_error_code(std::errc::bad_file_descriptor));

    auto Stop = std::chrono::steady_clock::now();
    ElapsedTime +=
        std::chrono::duration_cast<std::chrono::milliseconds>(Stop - Start);
  }

  int AcceptFD;
#ifdef _WIN32
  SOCKET WinAcceptSock = ::accept(WinServerSock, NULL, NULL);
  AcceptFD = _open_osfhandle(WinAcceptSock, 0);
#else
  AcceptFD = ::accept(FD, NULL, NULL);
#endif

  if (AcceptFD == -1)
    return llvm::make_error<StringError>(getLastSocketErrorCode(),
                                         "Socket accept failed");
  return std::make_unique<raw_socket_stream>(AcceptFD);
}

void ListeningSocket::shutdown() {
  int ObservedFD = FD.load();

  if (ObservedFD == -1)
    return;

  // If FD equals ObservedFD set FD to -1; If FD doesn't equal ObservedFD then
  // another thread is responsible for shutdown so return
  if (!FD.compare_exchange_strong(ObservedFD, -1))
    return;

  ::close(ObservedFD);
  ::unlink(SocketPath.c_str());

  // Ensure ::poll returns if shutdown is called by a separate thread
  char Byte = 'A';
  ssize_t written = ::write(PipeFD[1], &Byte, 1);

  // Ignore any write() error
  (void)written;
}

ListeningSocket::~ListeningSocket() {
  shutdown();

  // Close the pipe's FDs in the destructor instead of within
  // ListeningSocket::shutdown to avoid unnecessary synchronization issues that
  // would occur as PipeFD's values would have to be changed to -1
  //
  // The move constructor sets PipeFD to -1
  if (PipeFD[0] != -1)
    ::close(PipeFD[0]);
  if (PipeFD[1] != -1)
    ::close(PipeFD[1]);
}

//===----------------------------------------------------------------------===//
//  raw_socket_stream
//===----------------------------------------------------------------------===//

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
