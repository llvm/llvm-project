//===------------------- IndexManager.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of IndexManager in Storage
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Storage/BlobStore.h"

namespace llvm::advisor {

class IndexManager {
public:
  IndexManager(BlobStore &Blobs, std::string AnchorPath)
      : Blobs(Blobs), AnchorPath(std::move(AnchorPath)) {}

  Error load();
  Error flush();

  Error add(StringRef Index, StringRef Key, StringRef Value);
  SmallVector<std::string, 16> lookup(StringRef Index, StringRef Key) const;
  void clear(StringRef Index);

private:
  BlobStore &Blobs;
  std::string AnchorPath;
  StringMap<StringMap<SmallVector<std::string, 4>>> Indexes;
};

} // namespace llvm::advisor
