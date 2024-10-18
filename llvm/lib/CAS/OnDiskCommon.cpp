//===- OnDiskCommon.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OnDiskCommon.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

using namespace llvm;

Expected<std::optional<uint64_t>> cas::ondisk::getOverriddenMaxMappingSize() {
  constexpr const char *EnvVar = "LLVM_CAS_MAX_MAPPING_SIZE";
  const char *Value = getenv(EnvVar);
  if (!Value)
    return std::nullopt;

  uint64_t Size;
  if (StringRef(Value).getAsInteger(/*auto*/ 0, Size))
    return createStringError(inconvertibleErrorCode(),
                             "invalid value for %s: expected integer", EnvVar);
  return Size;
}
