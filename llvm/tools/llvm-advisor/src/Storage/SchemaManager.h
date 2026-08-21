//===------------------- SchemaManager.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of SchemaManager in Storage
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Storage/BlobStore.h"

namespace llvm::advisor {

class SchemaManager {
public:
  static constexpr unsigned CurrentVersion = 1;

  SchemaManager(BlobStore &Blobs, std::string AnchorPath)
      : Blobs(Blobs), AnchorPath(std::move(AnchorPath)) {}

  Error load();
  Error flush();

  unsigned getCurrentVersion() const { return CurrentVersion; }
  unsigned getStoredVersion() const { return StoredVersion; }
  Error migrate(unsigned From, unsigned To);
  Error validateOnStartup();

private:
  BlobStore &Blobs;
  std::string AnchorPath;
  unsigned StoredVersion = 0;
};

} // namespace llvm::advisor
