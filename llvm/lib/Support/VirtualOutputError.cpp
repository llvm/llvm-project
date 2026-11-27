//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the errors for output virtualization.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/VirtualOutputError.h"

using namespace llvm;
using namespace llvm::vfs;

void OutputError::anchor() {}
void OutputConfigError::anchor() {}
void TempFileOutputError::anchor() {}

char OutputError::ID = 0;
char OutputConfigError::ID = 0;
char TempFileOutputError::ID = 0;

void OutputError::log(raw_ostream &OS) const {
  OS << getOutputPath() << ": ";
  ECError::log(OS);
}

void OutputConfigError::log(raw_ostream &OS) const {
  OutputError::log(OS);
  OS << ": " << Config;
}

void TempFileOutputError::log(raw_ostream &OS) const {
  OS << getTempPath() << " => ";
  OutputError::log(OS);
}

namespace {
class OutputErrorCategory : public std::error_category {
public:
  const char *name() const noexcept override;
  std::string message(int EV) const override;
};
} // end namespace

const std::error_category &vfs::output_category() {
  static OutputErrorCategory ErrorCategory;
  return ErrorCategory;
}

const char *OutputErrorCategory::name() const noexcept {
  return "llvm.vfs.output";
}

std::string OutputErrorCategory::message(int EV) const {
  OutputErrorCode E = static_cast<OutputErrorCode>(EV);
  switch (E) {
  case OutputErrorCode::invalid_config:
    return "invalid config";
  case OutputErrorCode::not_closed:
    return "output not closed";
  case OutputErrorCode::already_closed:
    return "output already closed";
  case OutputErrorCode::has_open_proxy:
    return "output has open proxy";
  }
  llvm_unreachable(
      "An enumerator of OutputErrorCode does not have a message defined.");
}
