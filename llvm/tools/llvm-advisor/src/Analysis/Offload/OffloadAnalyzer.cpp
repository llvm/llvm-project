//===--- OffloadAnalyzer.cpp - LLVM Advisor ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Offload/OffloadAnalyzer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::advisor;

static std::string findOffloadProfileJSON(const CapabilityContext &Context) {
  SmallVector<std::string, 4> Candidates;
  if (!Context.WorkingDirectory.empty()) {
    Candidates.push_back(Context.WorkingDirectory + "/runtime.json");
    Candidates.push_back(Context.WorkingDirectory + "/offload.json");
  }
  for (const std::string &C : Candidates)
    if (sys::fs::exists(C))
      return C;
  return {};
}

Expected<std::unique_ptr<CapabilityResult>>
OffloadAnalyzer::run(const CapabilityContext &Context) {
  std::string Path = findOffloadProfileJSON(Context);
  if (Path.empty()) {
    return makeUnavailableResult(getCapabilityID(), Context.Unit.ID,
                                 "missing offload runtime profile JSON");
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB = MemoryBuffer::getFile(Path);
  if (!MB)
    return createStringError(MB.getError(), "cannot read offload profile: %s",
                             Path.c_str());

  Expected<json::Value> Parsed = json::parse(MB.get()->getBuffer());
  if (!Parsed)
    return Parsed.takeError();

  int64_t KernelCount = 0;
  int64_t TransferCount = 0;
  if (const json::Object *Obj = Parsed->getAsObject()) {
    if (const json::Array *K = Obj->getArray("kernels"))
      KernelCount = static_cast<int64_t>(K->size());
    if (const json::Array *T = Obj->getArray("transfers"))
      TransferCount = static_cast<int64_t>(T->size());
  }

  return makeJSONResult(getCapabilityID(), Context.Unit.ID, json::Object{
      {"profile_path", Path},
      {"kernel_count", KernelCount},
      {"transfer_count", TransferCount},
      {"profile", std::move(*Parsed)},
  });
}
