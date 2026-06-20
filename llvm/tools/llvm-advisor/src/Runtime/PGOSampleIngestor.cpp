//===------------------- PGOSampleIngestor.cpp - LLVM Advisor --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of PGOSampleIngestor in Runtime
//
//===----------------------------------------------------------------------===//
#include "Runtime/PGOSampleIngestor.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/ProfileData/SampleProfReader.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<json::Value> PGOSampleIngestor::load(StringRef Path) {
  if (Path.empty())
    return createStringError(inconvertibleErrorCode(),
                             "empty sample profile path");
  LLVMContext Context;
  IntrusiveRefCntPtr<vfs::FileSystem> FS = vfs::getRealFileSystem();
  ErrorOr<std::unique_ptr<sampleprof::SampleProfileReader>> Reader =
      sampleprof::SampleProfileReader::create(Path, Context, *FS);
  if (!Reader)
    return createStringError(Reader.getError(),
                             "cannot read sample profile '%s'",
                             Path.data());
  if (std::error_code EC = (*Reader)->read())
    return createStringError(EC, "invalid sample profile '%s'",
                             Path.data());

  uint64_t FunctionCount = 0;
  uint64_t BodySampleSites = 0;
  uint64_t TotalSamples = 0;
  uint64_t HeadSamples = 0;
  json::Array Functions;

  for (const auto &Entry : (*Reader)->getProfiles()) {
    const FunctionSamples &Samples = Entry.second;
    uint64_t FunctionTotal = Samples.getTotalSamples();
    TotalSamples += FunctionTotal;
    HeadSamples += Samples.getHeadSamples();
    BodySampleSites += Samples.getBodySamples().size();
    ++FunctionCount;

    if (Functions.size() < 256)
      Functions.push_back(json::Object{
          {"function", Samples.getFuncName()},
          {"total_samples", static_cast<int64_t>(FunctionTotal)},
          {"head_samples", static_cast<int64_t>(Samples.getHeadSamples())},
          {"body_sample_sites",
           static_cast<int64_t>(Samples.getBodySamples().size())}});
  }

  return json::Object{
      {"kind", "pgo-profile"},
      {"format", "sampleprof"},
      {"path", Path},
      {"function_count", static_cast<int64_t>(FunctionCount)},
      {"body_sample_sites", static_cast<int64_t>(BodySampleSites)},
      {"total_samples", static_cast<int64_t>(TotalSamples)},
      {"head_samples", static_cast<int64_t>(HeadSamples)},
      {"functions", std::move(Functions)}};
}
