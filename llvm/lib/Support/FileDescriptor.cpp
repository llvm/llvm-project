//===-- FileDescriptor.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a utility functions for working with file descriptors
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Error.h"
#include "llvm/Support/FileDescriptor.h"
#include <atomic>
#include <chrono>
#include <poll.h>

static std::error_code getLastSocketErrorCode() {
#ifdef _WIN32
  return std::error_code(::WSAGetLastError(), std::system_category());
#else
  return llvm::errnoAsErrorCode();
#endif
}

template <typename T>
llvm::Error llvm::manageTimeout(std::chrono::milliseconds Timeout, T &FD, int PipeFD) {
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, std::atomic<int>>,
                "FD must be of type int& or std::atomic<int>&");

  struct pollfd FDs[2];
  FDs[0].events = POLLIN;
#ifdef _WIN32
  SOCKET WinServerSock = _get_osfhandle(FD);
  FDs[0].fd = WinServerSock;
#else
  FDs[0].fd = llvm::getFD(FD);
#endif
  FDs[1].events = POLLIN;
  FDs[1].fd = PipeFD;

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
    if (FD == -1)
      return llvm::make_error<llvm::StringError>(
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
        return llvm::make_error<llvm::StringError>(PollErrCode,
                                                   "FD poll failed");
    }
    if (PollStatus == 0)
      return llvm::make_error<llvm::StringError>(
          std::make_error_code(std::errc::timed_out),
          "No client requests within timeout window");

    if (FDs[0].revents & POLLNVAL)
      return llvm::make_error<llvm::StringError>(
          std::make_error_code(std::errc::bad_file_descriptor));

    auto Stop = std::chrono::steady_clock::now();
    ElapsedTime +=
        std::chrono::duration_cast<std::chrono::milliseconds>(Stop - Start);
  }
  return llvm::Error::success();
}
