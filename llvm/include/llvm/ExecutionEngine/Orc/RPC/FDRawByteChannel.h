//===- FDRawByteChannel.h - File descriptor based byte-channel -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// File descriptor based RawByteChannel.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RPC_FDRAWBYTECHANNEL_H
#define LLVM_EXECUTIONENGINE_ORC_RPC_FDRAWBYTECHANNEL_H

#include "llvm/ExecutionEngine/Orc/RPC/RawByteChannel.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

namespace llvm {
namespace orc {
namespace rpc {

/// RPC channel that reads from and writes from file descriptors.
class FDRawByteChannel final : public RawByteChannel {
public:
  FDRawByteChannel(int InFD, int OutFD) : InFD(InFD), OutFD(OutFD) {}

  llvm::Error readBytes(char *Dst, unsigned Size) override {
    // dbgs() << "Reading " << Size << " bytes: [";
    assert(Dst && "Attempt to read into null.");
    ssize_t Completed = 0;
    while (Completed < static_cast<ssize_t>(Size)) {
      ssize_t Read = ::read(InFD, Dst + Completed, Size - Completed);
      if (Read <= 0) {
        // dbgs() << " <<<\n";
        auto ErrNo = errno;
        if (ErrNo == EAGAIN || ErrNo == EINTR)
          continue;
        else
          return llvm::errorCodeToError(
              std::error_code(errno, std::generic_category()));
      }
      // for (size_t I = 0; I != Read; ++I)
      //  dbgs() << " " << formatv("{0:x2}", Dst[Completed + I]);
      Completed += Read;
    }
    // dbgs() << " ]\n";
    return llvm::Error::success();
  }

  llvm::Error appendBytes(const char *Src, unsigned Size) override {
    // dbgs() << "Appending " << Size << " bytes: [";
    assert(Src && "Attempt to append from null.");
    ssize_t Completed = 0;
    while (Completed < static_cast<ssize_t>(Size)) {
      ssize_t Written = ::write(OutFD, Src + Completed, Size - Completed);
      if (Written < 0) {
        // dbgs() << " <<<\n";
        auto ErrNo = errno;
        if (ErrNo == EAGAIN || ErrNo == EINTR)
          continue;
        else
          return llvm::errorCodeToError(
              std::error_code(errno, std::generic_category()));
      }
      // for (size_t I = 0; I != Written; ++I)
      //  dbgs() << " " << formatv("{0:x2}", Src[Completed + I]);
      Completed += Written;
    }
    // dbgs() << " ]\n";
    return llvm::Error::success();
  }

  llvm::Error send() override { return llvm::Error::success(); }

private:
  int InFD, OutFD;
};

} // namespace rpc
} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_RPC_FDRAWBYTECHANNEL_H
