//===- OmpAccError.h - Error class for the OpenMP offload RTL -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_OMPACCERROR_H
#define OMPTARGET_OMPACCERROR_H

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm::omp::target {

enum class ErrorCode {
  Unknown,
  InvalidBinary,
  InvalidValue,
  BackendFailure,
};

} // namespace llvm::omp::target

namespace std {
template <>
struct is_error_code_enum<llvm::omp::target::ErrorCode> : std::true_type {};
} // namespace std

namespace llvm::omp::target {

const std::error_category &OmpAccErrCategory();

inline std::error_code make_error_code(ErrorCode E) {
  return std::error_code(static_cast<int>(E), OmpAccErrCategory());
}

/// Error class used by the OpenMP offload runtime (libomptarget).
class OmpAccError : public llvm::ErrorInfo<OmpAccError, llvm::StringError> {
public:
  using ErrorInfo<OmpAccError, StringError>::ErrorInfo;

  OmpAccError(const llvm::Twine &S) : ErrorInfo(S, ErrorCode::Unknown) {}

  static char ID;
};

/// Create an offload runtime error.
template <typename... ArgsTy>
[[maybe_unused]] static llvm::Error
createError(ErrorCode Code, const char *ErrFmt, ArgsTy... Args) {
  std::string Buffer;
  llvm::raw_string_ostream(Buffer) << llvm::format(ErrFmt, Args...);
  return llvm::make_error<OmpAccError>(Code, Buffer);
}

inline llvm::Error createError(ErrorCode Code, const char *S) {
  return llvm::make_error<OmpAccError>(Code, S);
}

} // namespace llvm::omp::target

#endif
