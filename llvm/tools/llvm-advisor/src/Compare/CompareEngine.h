//===------------------- CompareEngine.h - LLVM Advisor --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of CompareEngine in Compare
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Storage/StorageManager.h"

namespace llvm::advisor {

class CompareEngine {
public:
  explicit CompareEngine(StorageManager &Storage) : Storage(Storage) {}
  json::Value compare(StringRef Before, StringRef After) const;
  json::Value compareCapability(StringRef Before, StringRef After,
                                StringRef CapID) const;

private:
  StorageManager &Storage;
};

} // namespace llvm::advisor
