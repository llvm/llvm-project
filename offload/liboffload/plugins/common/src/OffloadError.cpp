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
using namespace error;

namespace {
// OffloadError inherits from llvm::StringError which requires a
// std::error_code. Once/if that requirement is removed, then this
// std::error_code machinery can be removed.
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

const std::error_category &error::OffloadErrCategory() {
  static OffloadErrorCategory MSFCategory;
  return MSFCategory;
}

char OffloadError::ID;
