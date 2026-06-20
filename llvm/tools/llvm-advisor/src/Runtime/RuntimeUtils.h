//===------------------- RuntimeUtils.h - LLVM Advisor ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared helpers for runtime ingestors.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"

namespace llvm::advisor {

/// Read an integer from a JSON object, returning Default if the key is absent
/// or not a number.
inline int64_t getInteger(const json::Object &Obj, StringRef Key,
                          int64_t Default = 0) {
  if (std::optional<int64_t> V = Obj.getInteger(Key))
    return *V;
  return Default;
}

} // namespace llvm::advisor
