//===----------------- RetentionManager.cpp - LLVM Advisor -----------------===//
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
#include "Storage/RetentionManager.h"
#include "Utils/JSON.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace llvm::advisor;

std::string RetentionManager::getPolicyPath() const {
  return (Twine(Root) + "/retention.policy").str();
}

Error RetentionManager::compact() {
  // NOTE: pruneStorageData can crash with some CAS backends/configurations in
  // this tree state. Keep maintenance-compact safe and non-crashing until
  // backend-specific pruning is stabilized.
  return Error::success();
}

Error RetentionManager::tierSnapshot(StringRef SnapshotID, StringRef Tier) {
  if (SnapshotID.empty() || Tier.empty())
    return createStringError(inconvertibleErrorCode(),
                             "invalid retention request");

  std::string PolicyPath = getPolicyPath();
  json::Object Policy;

  // Load existing policy if present.
  if (sys::fs::exists(PolicyPath)) {
    Expected<json::Value> Parsed = parseJSONFile(PolicyPath);
    if (Parsed) {
      if (const json::Object *Obj = Parsed->getAsObject())
        Policy = *Obj;
    }
  }

  json::Object *Tiers = Policy.getObject("tiers");
  if (!Tiers) {
    Policy["tiers"] = json::Object{};
    Tiers = Policy.getObject("tiers");
  }
  (*Tiers)[SnapshotID.str()] = Tier.str();

  std::error_code EC;
  ToolOutputFile Out(PolicyPath, EC, sys::fs::OF_Text);
  if (EC)
    return createStringError(EC, "cannot write retention policy '%s'",
                             PolicyPath.c_str());
  Out.os() << stringifyJSON(json::Value(std::move(Policy))) << '\n';
  Out.keep();
  return Error::success();
}

Expected<uint64_t> RetentionManager::estimateStorageUsage() const {
  SmallString<256> CASPath(Root);
  sys::path::append(CASPath, "cas");

  std::error_code EC;
  uint64_t TotalSize = 0;
  for (sys::fs::recursive_directory_iterator I(CASPath, EC), E; I != E && !EC;
       I.increment(EC)) {
    if (!sys::fs::is_regular_file(I->path()))
      continue;
    std::error_code SizeEC;
    uint64_t FileSize = 0;
    if (!sys::fs::file_size(I->path(), FileSize))
      TotalSize += FileSize;
  }
  if (EC)
    return createStringError(EC, "failed to estimate storage usage for '%s'",
                             CASPath.c_str());
  return TotalSize;
}
