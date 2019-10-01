//===- Error.h - system_error extensions for llvm-readobj -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This declares a new error_category for the llvm-readobj tool.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_READOBJ_ERROR_H
#define LLVM_TOOLS_LLVM_READOBJ_ERROR_H

#include <system_error>

namespace llvm {
const std::error_category &readobj_category();

enum class readobj_error {
  success = 0,
  file_not_found,
  unsupported_file_format,
  unrecognized_file_format,
  unsupported_obj_file_format,
  unknown_symbol
};

inline std::error_code make_error_code(readobj_error e) {
  return std::error_code(static_cast<int>(e), readobj_category());
}

} // namespace llvm

namespace std {
template <> struct is_error_code_enum<llvm::readobj_error> : std::true_type {};
}

#endif
