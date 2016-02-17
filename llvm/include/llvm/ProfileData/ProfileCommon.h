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
// The profile summary is one or more (Cutoff, MinCount, NumCounts) triplets.
// The semantics of counts depend on the type of profile. For instrumentation
// profile, counts are block counts and for sample profile, counts are
// per-line samples. Given a target counts percentile, we compute the minimum
// number of counts needed to reach this target and the minimum among these
// counts.
struct ProfileSummaryEntry {
  uint32_t Cutoff;    ///< The required percentile of counts.
  uint64_t MinCount;  ///< The minimum count for this percentile.
  uint64_t NumCounts; ///< Number of counts >= the minimum count.
  ProfileSummaryEntry(uint32_t TheCutoff, uint64_t TheMinCount,
                      uint64_t TheNumCounts)
      : Cutoff(TheCutoff), MinCount(TheMinCount), NumCounts(TheNumCounts) {}
};

class ProfileSummary {
  // We keep track of the number of times a count (block count or samples)
  // appears in the profile. The map is kept sorted in the descending order of
  // counts.
  std::map<uint64_t, uint32_t, std::greater<uint64_t>> CountFrequencies;

protected:
  std::vector<ProfileSummaryEntry> DetailedSummary;
  std::vector<uint32_t> DetailedSummaryCutoffs;
  uint64_t TotalCount, MaxCount;
  uint32_t NumCounts;
  ProfileSummary(std::vector<uint32_t> Cutoffs)
      : DetailedSummaryCutoffs(Cutoffs), TotalCount(0), MaxCount(0),
        NumCounts(0) {}
  ProfileSummary() : TotalCount(0), MaxCount(0), NumCounts(0) {}
  inline void addCount(uint64_t Count);

public:
  static const int Scale = 1000000;
  inline std::vector<ProfileSummaryEntry> &getDetailedSummary();
  void computeDetailedSummary();
};

class InstrProfSummary : public ProfileSummary {
  uint64_t MaxInternalBlockCount, MaxFunctionCount;
  uint32_t NumFunctions;
  inline void addEntryCount(uint64_t Count);
  inline void addInternalCount(uint64_t Count);

public:
  InstrProfSummary(std::vector<uint32_t> Cutoffs)
      : ProfileSummary(Cutoffs), MaxInternalBlockCount(0), MaxFunctionCount(0),
        NumFunctions(0) {}
  InstrProfSummary(const IndexedInstrProf::Summary &S);
  void addRecord(const InstrProfRecord &);
  uint32_t getNumBlocks() { return NumCounts; }
  uint64_t getTotalCount() { return TotalCount; }
  uint32_t getNumFunctions() { return NumFunctions; }
  uint64_t getMaxFunctionCount() { return MaxFunctionCount; }
  uint64_t getMaxBlockCount() { return MaxCount; }
  uint64_t getMaxInternalBlockCount() { return MaxInternalBlockCount; }
};

// This is called when a count is seen in the profile.
void ProfileSummary::addCount(uint64_t Count) {
  TotalCount += Count;
  if (Count > MaxCount)
    MaxCount = Count;
  NumCounts++;
  CountFrequencies[Count]++;
}

std::vector<ProfileSummaryEntry> &ProfileSummary::getDetailedSummary() {
  if (!DetailedSummaryCutoffs.empty() && DetailedSummary.empty())
    computeDetailedSummary();
  return DetailedSummary;
}

} // end namespace llvm
#endif
