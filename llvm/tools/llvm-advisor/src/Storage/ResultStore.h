//===------------------- ResultStore.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of ResultStore in Storage
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Storage/BlobStore.h"

namespace llvm::advisor {

class ResultStore {
public:
  ResultStore(BlobStore &Blobs, std::string AnchorPath)
      : Blobs(Blobs), AnchorPath(std::move(AnchorPath)) {}

  Error load();
  Error flush();

  Error registerSchema(StringRef CapabilityID, StringRef Version);
  Expected<std::string> put(StringRef RunKey, const json::Value &Result);
  Expected<std::string> get(StringRef RunKey) const;
  bool contains(StringRef RunKey) const { return Results.contains(RunKey); }

private:
  BlobStore &Blobs;
  std::string AnchorPath;
  StringMap<std::string> Schemas;
  StringMap<std::string> Results;
};

} // namespace llvm::advisor
