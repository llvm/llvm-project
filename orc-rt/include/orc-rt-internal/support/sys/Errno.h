//===- Errno.h - errno support ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for working with errno values.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INTERNAL_SUPPORT_SYS_ERRNO_H
#define ORC_RT_INTERNAL_SUPPORT_SYS_ERRNO_H

#include <string>

namespace orc_rt::sys {

/// Returns a human-readable description of the given errno value.
///
/// Prefer this to strerror, which is not guaranteed to be thread-safe. If no
/// description is available the value itself is reported, so the result is
/// always non-empty.
inline std::string strError(int ErrNum);

} // namespace orc_rt::sys

// Definition of the above, selected by system. This is header-only so that
// Support carries no per-system objects: Support is embedded into every library
// that ships, including the JIT-linked SPIRE, so anything it compiles is code
// those libraries have to carry.
#if !defined(_WIN32)

#include <string.h>
#include <type_traits>

namespace orc_rt::sys {

namespace detail {

/// Interpret strerror_r's result.
///
/// strerror_r comes in two incompatible forms:
///
/// POSIX: int   strerror_r(int errnum, char *buf, size_t buflen);
///   GNU: char *strerror_r(int errnum, char *buf, size_t buflen);
///
/// POSIX returns zero on success, whereas GNU returns a char* that may point
/// somewhere other than buf. Use return type to distinguish them below.
template <typename RetT>
std::string fromStrerrorR(RetT Ret, int ErrNum, const char *Buf) {
  if constexpr (std::is_pointer_v<RetT>)
    return Ret;
  else {
    // Some systems (Darwin among them) fill Buf even while reporting EINVAL for
    // an unrecognised value, so prefer whatever was written and synthesise a
    // description only if nothing was.
    if (Ret == 0 || Buf[0] != '\0')
      return Buf;
    return "unknown error " + std::to_string(ErrNum);
  }
}

} // namespace detail

inline std::string strError(int ErrNum) {
  char Buf[256] = {};
  // Leave the final byte NUL: on ERANGE the buffer contents are unspecified
  // and are not required to be terminated.
  return detail::fromStrerrorR(strerror_r(ErrNum, Buf, sizeof(Buf) - 1), ErrNum,
                               Buf);
}

} // namespace orc_rt::sys

#else

#error "No strError implementation for this target"

#endif

#endif // ORC_RT_INTERNAL_SUPPORT_SYS_ERRNO_H
