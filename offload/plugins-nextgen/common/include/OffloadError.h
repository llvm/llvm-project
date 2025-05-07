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

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

enum class ErrorCode {
#define OFFLOAD_ERRC(Name, _, Value) Name = Value,
#include "OffloadErrcodes.inc"
#undef OFFLOAD_ERRC
};

class OffloadErrorCategory : public std::error_category {
  const char *name() const noexcept override { return "Offload Error"; }
  std::string message(int ev) const override {
    switch (static_cast<ErrorCode>(ev)) {
#define OFFLOAD_ERRC(Name, Desc, Value)                                        \
  case ErrorCode::Name:                                                        \
    return #Desc;
#include "OffloadErrcodes.inc"
#undef OFFLOAD_ERRC
    }
  }
};
} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

namespace std {
template <>
struct is_error_code_enum<llvm::omp::target::plugin::ErrorCode>
    : std::true_type {};
} // namespace std

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

const std::error_category &OffloadErrCategory();

inline std::error_code make_error_code(ErrorCode E) {
  return std::error_code(static_cast<int>(E), OffloadErrCategory());
}

/// Base class for errors originating in DIA SDK, e.g. COM calls
class OffloadError : public ErrorInfo<OffloadError, StringError> {
public:
  using ErrorInfo<OffloadError, StringError>::ErrorInfo;

  OffloadError(const Twine &S) : ErrorInfo(S, ErrorCode::UNKNOWN) {}

  static char ID;
};
} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif
