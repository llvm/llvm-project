//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file declares llvm::sys::fs::file_t type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FILE_H
#define LLVM_SUPPORT_FILE_H

namespace llvm::sys::fs {

/// This class wraps the platform specific file handle/descriptor type to
/// provide an unified representation.
struct file_t {
#if defined(_WIN32)
  // A Win32 HANDLE is a typedef of void*
  using value_type = void *;
  static const value_type Invalid;
#else
  // A file descriptor on UNIX.
  using value_type = int;
  static constexpr value_type Invalid = -1;
#endif
  value_type Value;

  /// Default constructor to invalid file.
  file_t() : Value(Invalid) {}

  /// Implicit constructor from underlying value.
  // TODO: Make this explicit to flush out type mismatches.
  file_t(value_type Value) : Value(Value) {}

  /// Is a valid file.
  bool isValid() const { return Value != Invalid; }

  /// Get the underlying value and return a platform specific value.
  value_type get() const { return Value; }
};

inline bool operator==(file_t LHS, file_t RHS) {
  return LHS.get() == RHS.get();
}

inline bool operator!=(file_t LHS, file_t RHS) { return !(LHS == RHS); }

} // namespace llvm::sys::fs

#endif
