//===- PageSize.cpp - POSIX page-size detection -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Page-size detection on POSIX systems, in terms of sysconf.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/bedrock/sys/PageSize.h"
#include "orc-rt-internal/support/StringExtras.h"
#include "orc-rt-internal/support/sys/Errno.h"
#include "orc-rt/support/bit.h"

#include <cerrno>
#include <unistd.h>

namespace orc_rt::sys {

Expected<size_t> detectPageSize() noexcept {
  long PageSize = sysconf(_SC_PAGESIZE);
  if (PageSize == -1)
    return make_error<StringError>((StringOutputStream()
                                    << "sysconf did not return a page size: "
                                    << strError(errno))
                                       .str());
  if (PageSize <= 0 || !has_single_bit(static_cast<size_t>(PageSize)))
    return make_error<StringError>((StringOutputStream()
                                    << "reported page size " << PageSize
                                    << " is not a power of two")
                                       .str());
  return static_cast<size_t>(PageSize);
}

} // namespace orc_rt::sys
