//===- LibcallLoweringInfo.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/RuntimeLibcalls.h"

namespace llvm {

class LibcallLoweringInfo {
private:
  LLVM_ABI const RTLIB::RuntimeLibcallsInfo &RTLCI;
  /// Stores the implementation choice for each each libcall.
  LLVM_ABI RTLIB::LibcallImpl LibcallImpls[RTLIB::UNKNOWN_LIBCALL + 1] = {
      RTLIB::Unsupported};

public:
  LLVM_ABI LibcallLoweringInfo(const RTLIB::RuntimeLibcallsInfo &RTLCI);

  /// Get the libcall routine name for the specified libcall.
  // FIXME: This should be removed. Only LibcallImpl should have a name.
  LLVM_ABI const char *getLibcallName(RTLIB::Libcall Call) const {
    // FIXME: Return StringRef
    return RTLIB::RuntimeLibcallsInfo::getLibcallImplName(LibcallImpls[Call])
        .data();
  }

  /// Return the lowering's selection of implementation call for \p Call
  LLVM_ABI RTLIB::LibcallImpl getLibcallImpl(RTLIB::Libcall Call) const {
    return LibcallImpls[Call];
  }

  /// Rename the default libcall routine name for the specified libcall.
  LLVM_ABI void setLibcallImpl(RTLIB::Libcall Call, RTLIB::LibcallImpl Impl) {
    LibcallImpls[Call] = Impl;
  }

  // FIXME: Remove this wrapper in favor of directly using
  // getLibcallImplCallingConv
  LLVM_ABI CallingConv::ID getLibcallCallingConv(RTLIB::Libcall Call) const {
    return RTLCI.LibcallImplCallingConvs[LibcallImpls[Call]];
  }

  /// Get the CallingConv that should be used for the specified libcall.
  LLVM_ABI CallingConv::ID
  getLibcallImplCallingConv(RTLIB::LibcallImpl Call) const {
    return RTLCI.LibcallImplCallingConvs[Call];
  }

  /// Return a function name compatible with RTLIB::MEMCPY, or nullptr if fully
  /// unsupported.
  LLVM_ABI StringRef getMemcpyName() const {
    RTLIB::LibcallImpl Memcpy = getLibcallImpl(RTLIB::MEMCPY);
    if (Memcpy != RTLIB::Unsupported)
      return RTLIB::RuntimeLibcallsInfo::getLibcallImplName(Memcpy);

    // Fallback to memmove if memcpy isn't available.
    return getLibcallName(RTLIB::MEMMOVE);
  }
};

} // end namespace llvm
