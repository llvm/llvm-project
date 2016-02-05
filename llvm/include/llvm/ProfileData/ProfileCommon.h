//===-- ProfileCommon.h - Common profiling APIs. ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains data structures and functions common to both instrumented
// and sample profiling.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <functional>
#include <map>
#include <vector>

#ifndef LLVM_PROFILEDATA_PROFILE_COMMON_H
#define LLVM_PROFILEDATA_PROFILE_COMMON_H

namespace llvm {
namespace IndexedInstrProf {
struct Summary;
}
struct InstrProfRecord;
///// Profile summary computation ////
// The 'show' command displays richer summary of the profile data. The profile
// summary is one or more (Cutoff, MinBlockCount, NumBlocks) triplets. Given a
// target execution count percentile, we compute the minimum number of blocks
// needed to reach this target and the minimum execution count of these blocks.
struct ProfileSummaryEntry {
  uint32_t Cutoff;        ///< The required percentile of total execution count.
  uint64_t MinBlockCount; ///< The minimum execution count for this percentile.
  uint64_t NumBlocks;     ///< Number of blocks >= the minumum execution count.
  ProfileSummaryEntry(uint32_t TheCutoff, uint64_t TheMinBlockCount,
                      uint64_t TheNumBlocks)
      : Cutoff(TheCutoff), MinBlockCount(TheMinBlockCount),
        NumBlocks(TheNumBlocks) {}
};

class ProfileSummary {
  // We keep track of the number of times a count appears in the profile and
  // keep the map sorted in the descending order of counts.
  std::map<uint64_t, uint32_t, std::greater<uint64_t>> CountFrequencies;
  std::vector<ProfileSummaryEntry> DetailedSummary;
  std::vector<uint32_t> DetailedSummaryCutoffs;
  // Sum of all counts.
  uint64_t TotalCount;
  uint64_t MaxBlockCount, MaxInternalBlockCount, MaxFunctionCount;
  uint32_t NumBlocks, NumFunctions;
  inline void addCount(uint64_t Count, bool IsEntry);

public:
  static const int Scale = 1000000;
  ProfileSummary(std::vector<uint32_t> Cutoffs)
      : DetailedSummaryCutoffs(Cutoffs), TotalCount(0), MaxBlockCount(0),
        MaxInternalBlockCount(0), MaxFunctionCount(0), NumBlocks(0),
        NumFunctions(0) {}
  ProfileSummary(const IndexedInstrProf::Summary &S);
  void addRecord(const InstrProfRecord &);
  inline std::vector<ProfileSummaryEntry> &getDetailedSummary();
  void computeDetailedSummary();
  uint32_t getNumBlocks() { return NumBlocks; }
  uint64_t getTotalCount() { return TotalCount; }
  uint32_t getNumFunctions() { return NumFunctions; }
  uint64_t getMaxFunctionCount() { return MaxFunctionCount; }
  uint64_t getMaxBlockCount() { return MaxBlockCount; }
  uint64_t getMaxInternalBlockCount() { return MaxInternalBlockCount; }
};

// This is called when a count is seen in the profile.
void ProfileSummary::addCount(uint64_t Count, bool IsEntry) {
  TotalCount += Count;
  if (Count > MaxBlockCount)
    MaxBlockCount = Count;
  if (!IsEntry && Count > MaxInternalBlockCount)
    MaxInternalBlockCount = Count;
  NumBlocks++;
  CountFrequencies[Count]++;
}

std::vector<ProfileSummaryEntry> &ProfileSummary::getDetailedSummary() {
  if (!DetailedSummaryCutoffs.empty() && DetailedSummary.empty())
    computeDetailedSummary();
  return DetailedSummary;
}

} // end namespace llvm
#endif
