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

#include "llvm/ADT/APInt.h"
#include <cstdint>
#include <functional>
#include <map>
#include <vector>

#include "llvm/Support/Casting.h"

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

typedef std::vector<ProfileSummaryEntry> SummaryEntryVector;

class ProfileSummary {
public:
  enum Kind { PSK_Instr, PSK_Sample };

private:
  const Kind PSK;
  static const char *KindStr[2];
  // We keep track of the number of times a count (block count or samples)
  // appears in the profile. The map is kept sorted in the descending order of
  // counts.
  std::map<uint64_t, uint32_t, std::greater<uint64_t>> CountFrequencies;
protected:
  SummaryEntryVector DetailedSummary;
  std::vector<uint32_t> DetailedSummaryCutoffs;
  uint64_t TotalCount, MaxCount;
  uint32_t NumCounts;
  ProfileSummary(Kind K, std::vector<uint32_t> Cutoffs)
      : PSK(K), DetailedSummaryCutoffs(Cutoffs), TotalCount(0), MaxCount(0),
        NumCounts(0) {}
  ProfileSummary(Kind K) : PSK(K), TotalCount(0), MaxCount(0), NumCounts(0) {}
  ProfileSummary(Kind K, SummaryEntryVector DetailedSummary,
                 uint64_t TotalCount, uint64_t MaxCount, uint32_t NumCounts)
      : PSK(K), DetailedSummary(DetailedSummary), TotalCount(TotalCount),
        MaxCount(MaxCount), NumCounts(NumCounts) {}
  ~ProfileSummary() = default;
  inline void addCount(uint64_t Count);
  /// \brief Return metadata specific to the profile format.
  /// Derived classes implement this method to return a vector of Metadata.
  virtual std::vector<Metadata *> getFormatSpecificMD(LLVMContext &Context) = 0;
  /// \brief Return detailed summary as metadata.
  Metadata *getDetailedSummaryMD(LLVMContext &Context);

public:
  static const int Scale = 1000000;
  Kind getKind() const { return PSK; }
  const char *getKindStr() const { return KindStr[PSK]; }
  // \brief Returns true if F is a hot function.
  static bool isFunctionHot(const Function *F);
  // \brief Returns true if F is unlikley executed.
  static bool isFunctionUnlikely(const Function *F);
  inline SummaryEntryVector &getDetailedSummary();
  void computeDetailedSummary();
  /// \brief A vector of useful cutoff values for detailed summary.
  static const std::vector<uint32_t> DefaultCutoffs;
  /// \brief Return summary information as metadata.
  Metadata *getMD(LLVMContext &Context);
  /// \brief Construct profile summary from metdata.
  static ProfileSummary *getFromMD(Metadata *MD);
};

class InstrProfSummary final : public ProfileSummary {
  uint64_t MaxInternalBlockCount, MaxFunctionCount;
  uint32_t NumFunctions;
  inline void addEntryCount(uint64_t Count);
  inline void addInternalCount(uint64_t Count);

protected:
  std::vector<Metadata *> getFormatSpecificMD(LLVMContext &Context) override;

public:
  InstrProfSummary(std::vector<uint32_t> Cutoffs)
      : ProfileSummary(PSK_Instr, Cutoffs), MaxInternalBlockCount(0),
        MaxFunctionCount(0), NumFunctions(0) {}
  InstrProfSummary(const IndexedInstrProf::Summary &S);
  InstrProfSummary(uint64_t TotalCount, uint64_t MaxBlockCount,
                   uint64_t MaxInternalBlockCount, uint64_t MaxFunctionCount,
                   uint32_t NumBlocks, uint32_t NumFunctions,
                   SummaryEntryVector Summary)
      : ProfileSummary(PSK_Instr, Summary, TotalCount, MaxBlockCount,
                       NumBlocks),
        MaxInternalBlockCount(MaxInternalBlockCount),
        MaxFunctionCount(MaxFunctionCount), NumFunctions(NumFunctions) {}
  static bool classof(const ProfileSummary *PS) {
    return PS->getKind() == PSK_Instr;
  }
  void addRecord(const InstrProfRecord &);
  uint32_t getNumBlocks() { return NumCounts; }
  uint64_t getTotalCount() { return TotalCount; }
  uint32_t getNumFunctions() { return NumFunctions; }
  uint64_t getMaxFunctionCount() { return MaxFunctionCount; }
  uint64_t getMaxBlockCount() { return MaxCount; }
  uint64_t getMaxInternalBlockCount() { return MaxInternalBlockCount; }
};

class SampleProfileSummary final : public ProfileSummary {
  uint64_t MaxHeadSamples;
  uint32_t NumFunctions;

protected:
  std::vector<Metadata *> getFormatSpecificMD(LLVMContext &Context) override;

public:
  uint32_t getNumLinesWithSamples() { return NumCounts; }
  uint64_t getTotalSamples() { return TotalCount; }
  uint32_t getNumFunctions() { return NumFunctions; }
  uint64_t getMaxHeadSamples() { return MaxHeadSamples; }
  uint64_t getMaxSamplesPerLine() { return MaxCount; }
  void addRecord(const sampleprof::FunctionSamples &FS);
  SampleProfileSummary(std::vector<uint32_t> Cutoffs)
      : ProfileSummary(PSK_Sample, Cutoffs), MaxHeadSamples(0),
        NumFunctions(0) {}
  SampleProfileSummary(uint64_t TotalSamples, uint64_t MaxSamplesPerLine,
                       uint64_t MaxHeadSamples, int32_t NumLinesWithSamples,
                       uint32_t NumFunctions,
                       SummaryEntryVector DetailedSummary)
      : ProfileSummary(PSK_Sample, DetailedSummary, TotalSamples,
                       MaxSamplesPerLine, NumLinesWithSamples),
        MaxHeadSamples(MaxHeadSamples), NumFunctions(NumFunctions) {}
  static bool classof(const ProfileSummary *PS) {
    return PS->getKind() == PSK_Sample;
  }
};

// This is called when a count is seen in the profile.
void ProfileSummary::addCount(uint64_t Count) {
  TotalCount += Count;
  if (Count > MaxCount)
    MaxCount = Count;
  NumCounts++;
  CountFrequencies[Count]++;
}

SummaryEntryVector &ProfileSummary::getDetailedSummary() {
  if (!DetailedSummaryCutoffs.empty() && DetailedSummary.empty())
    computeDetailedSummary();
  return DetailedSummary;
}

} // end namespace llvm
#endif
