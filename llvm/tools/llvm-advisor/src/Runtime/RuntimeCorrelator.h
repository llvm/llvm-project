//===------------------- RuntimeCorrelator.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of RuntimeCorrelator in Runtime
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Storage/StorageManager.h"

namespace llvm::advisor {
class RuntimeCorrelator {
public:
  Expected<json::Value> correlate(StorageManager &Storage,
                                  StringRef SnapshotID) const;
};
} // namespace llvm::advisor
