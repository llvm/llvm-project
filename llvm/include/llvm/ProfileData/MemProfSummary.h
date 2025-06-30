//===- MemProfSummary.h - MemProf summary support ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains MemProf summary support.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_MEMPROFSUMMARY_H
#define LLVM_PROFILEDATA_MEMPROFSUMMARY_H

#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace memprof {

class MemProfSummary {
private:
  /// The number of summary fields below, which is used to enable some forwards
  /// and backwards compatibility for the summary when serialized in the indexed
  /// MemProf format. As long as no existing summary fields are removed or
  /// reordered, and new summary fields are added after existing summary fields,
  /// the MemProf indexed profile version does not need to be bumped to
  /// accommodate new summary fields.
  static constexpr unsigned NumSummaryFields = 6;

  const uint64_t NumContexts, NumColdContexts, NumHotContexts;
  const uint64_t MaxColdTotalSize, MaxWarmTotalSize, MaxHotTotalSize;

public:
  MemProfSummary(uint64_t NumContexts, uint64_t NumColdContexts,
                 uint64_t NumHotContexts, uint64_t MaxColdTotalSize,
                 uint64_t MaxWarmTotalSize, uint64_t MaxHotTotalSize)
      : NumContexts(NumContexts), NumColdContexts(NumColdContexts),
        NumHotContexts(NumHotContexts), MaxColdTotalSize(MaxColdTotalSize),
        MaxWarmTotalSize(MaxWarmTotalSize), MaxHotTotalSize(MaxHotTotalSize) {}

  static constexpr unsigned getNumSummaryFields() { return NumSummaryFields; }
  uint64_t getNumContexts() const { return NumContexts; }
  uint64_t getNumColdContexts() const { return NumColdContexts; }
  uint64_t getNumHotContexts() const { return NumHotContexts; }
  uint64_t getMaxColdTotalSize() const { return MaxColdTotalSize; }
  uint64_t getMaxWarmTotalSize() const { return MaxWarmTotalSize; }
  uint64_t getMaxHotTotalSize() const { return MaxHotTotalSize; }
  LLVM_ABI void printSummaryYaml(raw_ostream &OS) const;
  /// Write to indexed MemProf profile.
  LLVM_ABI void write(ProfOStream &OS) const;
  /// Read from indexed MemProf profile.
  LLVM_ABI static std::unique_ptr<MemProfSummary>
  deserialize(const unsigned char *&);
};

} // namespace memprof
} // namespace llvm

#endif // LLVM_PROFILEDATA_MEMPROFSUMMARY_H
