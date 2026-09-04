//===- Errno.cpp - POSIX errno support -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of orc-rt-internal/support/sys/Errno.h on POSIX systems, in
// terms of strerror_r.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/support/sys/Errno.h"

#include <string.h>
#include <type_traits>

namespace orc_rt::sys {

namespace {

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

} // namespace

std::string strError(int ErrNum) {
  char Buf[256] = {};
  // Leave the final byte NUL: on ERANGE the buffer contents are unspecified
  // and are not required to be terminated.
  return fromStrerrorR(strerror_r(ErrNum, Buf, sizeof(Buf) - 1), ErrNum, Buf);
}

} // namespace orc_rt::sys
