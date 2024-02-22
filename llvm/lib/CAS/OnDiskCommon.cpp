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
#include "llvm/Support/Process.h"
#include <mutex>
#include <optional>

using namespace llvm;

static uint64_t OnDiskCASMaxMappingSize = 0;

Expected<std::optional<uint64_t>> cas::ondisk::getOverriddenMaxMappingSize() {
  static std::once_flag Flag;
  Error Err = Error::success();
  std::call_once(Flag, [&Err] {
    ErrorAsOutParameter EAO(&Err);
    constexpr const char *EnvVar = "LLVM_CAS_MAX_MAPPING_SIZE";
    auto Value = sys::Process::GetEnv(EnvVar);
    if (!Value)
      return;

    uint64_t Size;
    if (StringRef(*Value).getAsInteger(/*auto*/ 0, Size))
      Err = createStringError(inconvertibleErrorCode(),
                              "invalid value for %s: expected integer", EnvVar);
    OnDiskCASMaxMappingSize = Size;
  });

  if (Err)
    return std::move(Err);

  if (OnDiskCASMaxMappingSize == 0)
    return std::nullopt;

  return OnDiskCASMaxMappingSize;
}

void cas::ondisk::setMaxMappingSize(uint64_t Size) {
  OnDiskCASMaxMappingSize = Size;
}
