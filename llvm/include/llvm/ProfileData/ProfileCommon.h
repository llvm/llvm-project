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

#ifndef LLVM_PROFILEDATA_PROFILE_COMMON_H
#define LLVM_PROFILEDATA_PROFILE_COMMON_H

#include <cstdint>
#include <functional>
#include <map>
#include <utility>
#include <vector>

#include "llvm/IR/ProfileSummary.h"
#include "llvm/Support/Error.h"

namespace llvm {
class Function;
namespace IndexedInstrProf {
struct Summary;
}
namespace sampleprof {
class FunctionSamples;
}
struct InstrProfRecord;
class LLVMContext;
class Metadata;
class MDTuple;
class MDNode;

inline const char *getHotSectionPrefix() { return ".hot"; }
inline const char *getUnlikelySectionPrefix() { return ".unlikely"; }

class ProfileSummaryBuilder {

private:
  // We keep track of the number of times a count (block count or samples)
  // appears in the profile. The map is kept sorted in the descending order of
  // counts.
  std::map<uint64_t, uint32_t, std::greater<uint64_t>> CountFrequencies;
  std::vector<uint32_t> DetailedSummaryCutoffs;

protected:
  SummaryEntryVector DetailedSummary;
  ProfileSummaryBuilder(std::vector<uint32_t> Cutoffs)
      : DetailedSummaryCutoffs(std::move(Cutoffs)), TotalCount(0), MaxCount(0),
        MaxFunctionCount(0), NumCounts(0), NumFunctions(0) {}
  inline void addCount(uint64_t Count);
  ~ProfileSummaryBuilder() = default;
  void computeDetailedSummary();
  uint64_t TotalCount, MaxCount, MaxFunctionCount;
  uint32_t NumCounts, NumFunctions;

public:
  /// \brief A vector of useful cutoff values for detailed summary.
  static const ArrayRef<uint32_t> DefaultCutoffs;
};

class InstrProfSummaryBuilder final : public ProfileSummaryBuilder {
  uint64_t MaxInternalBlockCount;
  inline void addEntryCount(uint64_t Count);
  inline void addInternalCount(uint64_t Count);

public:
  InstrProfSummaryBuilder(std::vector<uint32_t> Cutoffs)
      : ProfileSummaryBuilder(std::move(Cutoffs)), MaxInternalBlockCount(0) {}
  void addRecord(const InstrProfRecord &);
  std::unique_ptr<ProfileSummary> getSummary();
};

class SampleProfileSummaryBuilder final : public ProfileSummaryBuilder {

public:
  void addRecord(const sampleprof::FunctionSamples &FS);
  SampleProfileSummaryBuilder(std::vector<uint32_t> Cutoffs)
      : ProfileSummaryBuilder(std::move(Cutoffs)) {}
  std::unique_ptr<ProfileSummary> getSummary();
};

// This is called when a count is seen in the profile.
void ProfileSummaryBuilder::addCount(uint64_t Count) {
  TotalCount += Count;
  if (Count > MaxCount)
    MaxCount = Count;
  NumCounts++;
  CountFrequencies[Count]++;
}


} // end namespace llvm
#endif
