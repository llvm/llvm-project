//===- OffloadError.cpp - Error extensions for offload --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OffloadError.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace llvm::omp::target::plugin;

namespace {
// FIXME: This class is only here to support the transition to llvm::Error. It
// will be removed once this transition is complete. Clients should prefer to
// deal with the Error value directly, rather than converting to error_code.
class OffloadErrorCategory : public std::error_category {
public:
  const char *name() const noexcept override { return "llvm.offload"; }
  std::string message(int Condition) const override {
    switch (static_cast<ErrorCode>(Condition)) {
#define OFFLOAD_ERRC(Name, Desc, Value)                                        \
  case ErrorCode::Name:                                                        \
    return #Desc;
#include "OffloadErrcodes.inc"
#undef OFFLOAD_ERRC
    }
    llvm_unreachable("Unrecognized offload ErrorCode");
  }
};
} // namespace

const std::error_category &llvm::omp::target::plugin::OffloadErrCategory() {
  static OffloadErrorCategory MSFCategory;
  return MSFCategory;
}

char OffloadError::ID;
