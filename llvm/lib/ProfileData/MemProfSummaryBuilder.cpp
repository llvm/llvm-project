//=-- MemProfSummaryBuilder.cpp - MemProf summary building ---------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains MemProf summary builder.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/MemProfSummaryBuilder.h"
#include "llvm/ProfileData/MemProfCommon.h"

using namespace llvm;
using namespace llvm::memprof;

std::unique_ptr<MemProfSummary> MemProfSummaryBuilder::getSummary() {
  return std::make_unique<MemProfSummary>(NumContexts, NumColdContexts,
                                          NumHotContexts, MaxColdTotalSize,
                                          MaxWarmTotalSize, MaxHotTotalSize);
}

void MemProfSummaryBuilder::addRecord(uint64_t CSId,
                                      const PortableMemInfoBlock &Info) {
  auto I = Contexts.insert(CSId);
  if (!I.second)
    return;
  NumContexts++;
  auto AllocType = getAllocType(Info.getTotalLifetimeAccessDensity(),
                                Info.getAllocCount(), Info.getTotalLifetime());
  auto TotalSize = Info.getTotalSize();
  switch (AllocType) {
  case AllocationType::Cold:
    NumColdContexts++;
    if (TotalSize > MaxColdTotalSize)
      MaxColdTotalSize = TotalSize;
    break;
  case AllocationType::NotCold:
    if (TotalSize > MaxWarmTotalSize)
      MaxWarmTotalSize = TotalSize;
    break;
  case AllocationType::Hot:
    NumHotContexts++;
    if (TotalSize > MaxHotTotalSize)
      MaxHotTotalSize = TotalSize;
    break;
  default:
    assert(false);
  }
}

void MemProfSummaryBuilder::addRecord(const IndexedMemProfRecord &Record) {
  for (auto &Alloc : Record.AllocSites)
    addRecord(Alloc.CSId, Alloc.Info);
}

void MemProfSummaryBuilder::addRecord(const MemProfRecord &Record) {
  for (auto &Alloc : Record.AllocSites)
    addRecord(computeFullStackId(Alloc.CallStack), Alloc.Info);
}
