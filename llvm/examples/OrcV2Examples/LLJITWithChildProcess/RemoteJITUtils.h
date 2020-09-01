//===-- RemoteJITUtils.h - Utilities for remote-JITing ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for remote-JITing
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXAMPLES_ORCV2EXAMPLES_LLJITWITHCHILDPROCESS_REMOTEJITUTILS_H
#define LLVM_EXAMPLES_ORCV2EXAMPLES_LLJITWITHCHILDPROCESS_REMOTEJITUTILS_H

#include "llvm/ExecutionEngine/Orc/RPC/RawByteChannel.h"
#include <mutex>

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

/// RPC channel that reads from and writes from file descriptors.
class FDRawChannel final : public llvm::orc::rpc::RawByteChannel {
public:
  FDRawChannel(int InFD, int OutFD) : InFD(InFD), OutFD(OutFD) {}

  llvm::Error readBytes(char *Dst, unsigned Size) override {
    assert(Dst && "Attempt to read into null.");
    ssize_t Completed = 0;
    while (Completed < static_cast<ssize_t>(Size)) {
      ssize_t Read = ::read(InFD, Dst + Completed, Size - Completed);
      if (Read <= 0) {
        auto ErrNo = errno;
        if (ErrNo == EAGAIN || ErrNo == EINTR)
          continue;
        else
          return llvm::errorCodeToError(
              std::error_code(errno, std::generic_category()));
      }
      Completed += Read;
    }
    return llvm::Error::success();
  }

  llvm::Error appendBytes(const char *Src, unsigned Size) override {
    assert(Src && "Attempt to append from null.");
    ssize_t Completed = 0;
    while (Completed < static_cast<ssize_t>(Size)) {
      ssize_t Written = ::write(OutFD, Src + Completed, Size - Completed);
      if (Written < 0) {
        auto ErrNo = errno;
        if (ErrNo == EAGAIN || ErrNo == EINTR)
          continue;
        else
          return llvm::errorCodeToError(
              std::error_code(errno, std::generic_category()));
      }
      Completed += Written;
    }
    return llvm::Error::success();
  }

  llvm::Error send() override { return llvm::Error::success(); }

private:
  int InFD, OutFD;
};

// Launch child process and return a channel to it.
std::unique_ptr<FDRawChannel> launchRemote(std::string ExecPath,
                                           pid_t &ChildPID) {
  // Create two pipes.
  int PipeFD[2][2];
  if (pipe(PipeFD[0]) != 0 || pipe(PipeFD[1]) != 0)
    perror("Error creating pipe: ");

  ChildPID = fork();

  if (ChildPID == 0) {
    // In the child...

    // Close the parent ends of the pipes
    close(PipeFD[0][1]);
    close(PipeFD[1][0]);

    // Execute the child process.
    std::unique_ptr<char[]> ChildPath, ChildIn, ChildOut;
    {
      ChildPath.reset(new char[ExecPath.size() + 1]);
      std::copy(ExecPath.begin(), ExecPath.end(), &ChildPath[0]);
      ChildPath[ExecPath.size()] = '\0';
      std::string ChildInStr = llvm::utostr(PipeFD[0][0]);
      ChildIn.reset(new char[ChildInStr.size() + 1]);
      std::copy(ChildInStr.begin(), ChildInStr.end(), &ChildIn[0]);
      ChildIn[ChildInStr.size()] = '\0';
      std::string ChildOutStr = llvm::utostr(PipeFD[1][1]);
      ChildOut.reset(new char[ChildOutStr.size() + 1]);
      std::copy(ChildOutStr.begin(), ChildOutStr.end(), &ChildOut[0]);
      ChildOut[ChildOutStr.size()] = '\0';
    }

    char *const args[] = {&ChildPath[0], &ChildIn[0], &ChildOut[0], nullptr};
    int rc = execv(ExecPath.c_str(), args);
    if (rc != 0)
      perror("Error executing child process: ");
    llvm_unreachable("Error executing child process");
  }
  // else we're the parent...

  // Close the child ends of the pipes
  close(PipeFD[0][0]);
  close(PipeFD[1][1]);

  // Return an RPC channel connected to our end of the pipes.
  return std::make_unique<FDRawChannel>(PipeFD[1][0], PipeFD[0][1]);
}

#endif
