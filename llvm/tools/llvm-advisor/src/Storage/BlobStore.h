//===------------------- BlobStore.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of BlobStore in Storage
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "llvm/CAS/ObjectStore.h"

namespace llvm::advisor {

class BlobStore {
public:
  explicit BlobStore(cas::ObjectStore &CAS) : CAS(CAS) {}

  Expected<std::string> put(StringRef Data);
  Expected<std::string> putFile(StringRef Path);
  Expected<std::string> get(StringRef ID);
  Expected<bool> exists(StringRef ID);

private:
  cas::ObjectStore &CAS;
};

} // namespace llvm::advisor
