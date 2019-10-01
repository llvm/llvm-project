//===-- RemoteJITUtils.h - Utilities for remote-JITing with LLI -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for remote-JITing with LLI.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLI_REMOTEJITUTILS_H
#define LLVM_TOOLS_LLI_REMOTEJITUTILS_H

#include "llvm/ExecutionEngine/Orc/RawByteChannel.h"
#include "llvm/Support/Error.h"
#include <cassert>
#include <cerrno>
#include <system_error>

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

/// RPC channel that reads from and writes from file descriptors.
class FDRPCChannel final : public llvm::orc::rpc::RawByteChannel {
public:
  FDRPCChannel(int InFD, int OutFD) : InFD(InFD), OutFD(OutFD) {}

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

#endif // LLVM_TOOLS_LLI_REMOTEJITUTILS_H
