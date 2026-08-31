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
std::string strError(int ErrNum);

} // namespace orc_rt::sys

#endif // ORC_RT_INTERNAL_SUPPORT_SYS_ERRNO_H
