//===- PageSize.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/bedrock/sys/Memory.h"

#include <windows.h>

namespace orc_rt::sys {

Expected<uint64_t> detectPageSize() {
  SYSTEM_INFO SI;
  GetSystemInfo(&SI);
  return static_cast<uint64_t>(SI.dwPageSize);
}

} // namespace orc_rt::sys
