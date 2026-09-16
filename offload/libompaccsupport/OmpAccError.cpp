//===- OmpAccError.cpp - Error class for the OpenMP offload RTL -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OmpAccError.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace llvm::omp::target;

namespace {
// OmpAccError inherits from llvm::StringError which requires a
// std::error_code. Once/if that requirement is removed, then this
// std::error_code machinery can be removed.
class OmpAccErrorCategory : public std::error_category {
public:
  const char *name() const noexcept override { return "llvm.omptarget"; }
  std::string message(int Condition) const override {
    switch (static_cast<ErrorCode>(Condition)) {
    case ErrorCode::Unknown:
      return "unknown error";
    case ErrorCode::InvalidBinary:
      return "invalid binary";
    case ErrorCode::InvalidValue:
      return "invalid value";
    case ErrorCode::BackendFailure:
      return "backend failure";
    }
    llvm_unreachable("Unrecognized offload RTL ErrorCode");
  }
};
} // namespace

const std::error_category &llvm::omp::target::OmpAccErrCategory() {
  static OmpAccErrorCategory Category;
  return Category;
}

char OmpAccError::ID;
