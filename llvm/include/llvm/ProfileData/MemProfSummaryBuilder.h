//===- MemProfSummaryBuilder.h - MemProf summary building -------*- C++ -*-===//
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

#ifndef LLVM_PROFILEDATA_MEMPROFSUMMARYBUILDER_H
#define LLVM_PROFILEDATA_MEMPROFSUMMARYBUILDER_H

#include "llvm/ProfileData/MemProf.h"
#include "llvm/ProfileData/MemProfSummary.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace memprof {

class MemProfSummaryBuilder {
private:
  // The set of full context IDs that we've recorded so far. This is needed to
  // dedup the MIBs, which are duplicated between functions containing inline
  // instances of the same allocations.
  DenseSet<uint64_t> Contexts;

  // Helper called by the public raw and indexed profile addRecord interfaces.
  void addRecord(uint64_t, const PortableMemInfoBlock &);

  uint64_t MaxColdTotalSize = 0;
  uint64_t MaxWarmTotalSize = 0;
  uint64_t MaxHotTotalSize = 0;
  uint64_t NumContexts = 0;
  uint64_t NumColdContexts = 0;
  uint64_t NumHotContexts = 0;

public:
  MemProfSummaryBuilder() = default;
  ~MemProfSummaryBuilder() = default;

  LLVM_ABI void addRecord(const IndexedMemProfRecord &);
  LLVM_ABI void addRecord(const MemProfRecord &);
  LLVM_ABI std::unique_ptr<MemProfSummary> getSummary();
};

} // namespace memprof
} // namespace llvm

#endif // LLVM_PROFILEDATA_MEMPROFSUMMARYBUILDER_H
