//===------------------- RetentionManager.h - LLVM Advisor -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of RetentionManager in Storage
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "llvm/CAS/ObjectStore.h"

namespace llvm::advisor {

class RetentionManager {
public:
  explicit RetentionManager(cas::ObjectStore &CAS, StringRef Root)
      : CAS(CAS), Root(Root.str()) {}

  Error compact();
  Error tierSnapshot(StringRef SnapshotID, StringRef Tier);
  Expected<uint64_t> estimateStorageUsage() const;

private:
  std::string getPolicyPath() const;

  cas::ObjectStore &CAS;
  std::string Root;
};

} // namespace llvm::advisor
