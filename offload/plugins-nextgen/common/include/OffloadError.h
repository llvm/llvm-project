//===- OffloadError.h - Definition of error class -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_OFFLOAD_ERROR_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_OFFLOAD_ERROR_H

#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

namespace error {

enum class ErrorCode {
#define OFFLOAD_ERRC(Name, _, Value) Name = Value,
#include "OffloadErrcodes.inc"
#undef OFFLOAD_ERRC
};

} // namespace error

namespace std {
template <> struct is_error_code_enum<error::ErrorCode> : std::true_type {};
} // namespace std

namespace error {

const std::error_category &OffloadErrCategory();

inline std::error_code make_error_code(ErrorCode E) {
  return std::error_code(static_cast<int>(E), OffloadErrCategory());
}

/// Base class for errors originating in DIA SDK, e.g. COM calls
class OffloadError : public llvm::ErrorInfo<OffloadError, llvm::StringError> {
public:
  using ErrorInfo<OffloadError, StringError>::ErrorInfo;

  OffloadError(const llvm::Twine &S) : ErrorInfo(S, ErrorCode::UNKNOWN) {}

  // The definition for this resides in the plugin static library
  static char ID;
};

/// Create an Offload error.
template <typename... ArgsTy>
static llvm::Error createOffloadError(error::ErrorCode Code, const char *ErrFmt,
                                      ArgsTy... Args) {
  std::string Buffer;
  llvm::raw_string_ostream(Buffer) << llvm::format(ErrFmt, Args...);
  return llvm::make_error<error::OffloadError>(Code, Buffer);
}

inline llvm::Error createOffloadError(error::ErrorCode Code, const char *S) {
  return llvm::make_error<error::OffloadError>(Code, S);
}

// The OffloadError will have a message of either:
// * "{Context}: {Message}" if the other error is a StringError
// * "{Context}" otherwise
inline llvm::Error createOffloadError(error::ErrorCode Code,
                                      llvm::Error &&OtherError,
                                      const char *Context) {
  std::string Buffer{Context};
  llvm::raw_string_ostream buffer(Buffer);

  handleAllErrors(
      std::move(OtherError),
      [&](llvm::StringError &Err) {
        buffer << ": ";
        buffer << Err.getMessage();
      },
      [&](llvm::ErrorInfoBase &Err) {
        // Non-string error message don't add anything to the offload error's
        // error message
      });

  return llvm::make_error<error::OffloadError>(Code, Buffer);
}
} // namespace error

#endif
