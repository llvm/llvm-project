//===- Error.cpp - system_error extensions for llvm-readobj -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines a new error_category for the llvm-readobj tool.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
// FIXME: This class is only here to support the transition to llvm::Error. It
// will be removed once this transition is complete. Clients should prefer to
// deal with the Error value directly, rather than converting to error_code.
class _readobj_error_category : public std::error_category {
public:
  const char* name() const noexcept override;
  std::string message(int ev) const override;
};
} // namespace

const char *_readobj_error_category::name() const noexcept {
  return "llvm.readobj";
}

std::string _readobj_error_category::message(int EV) const {
  switch (static_cast<readobj_error>(EV)) {
  case readobj_error::success: return "Success";
  case readobj_error::file_not_found:
    return "No such file.";
  case readobj_error::unsupported_file_format:
    return "The file was not recognized as a valid object file.";
  case readobj_error::unrecognized_file_format:
    return "Unrecognized file type.";
  case readobj_error::unsupported_obj_file_format:
    return "Unsupported object file format.";
  case readobj_error::unknown_symbol:
    return "Unknown symbol.";
  }
  llvm_unreachable("An enumerator of readobj_error does not have a message "
                   "defined.");
}

namespace llvm {
const std::error_category &readobj_category() {
  static _readobj_error_category o;
  return o;
}
} // namespace llvm
