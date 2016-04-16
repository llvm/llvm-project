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
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"

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
class Module;

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
  // Compute profile summary for a module.
  static ProfileSummary *computeProfileSummary(Module *M);
  // Cache of last seen module and its profile summary.
  static ManagedStatic<std::pair<Module *, std::unique_ptr<ProfileSummary>>>
      CachedSummary;
  // Mutex to access summary cache
  static ManagedStatic<sys::SmartMutex<true>> CacheMutex;

protected:
  SummaryEntryVector DetailedSummary;
  std::vector<uint32_t> DetailedSummaryCutoffs;
  uint64_t TotalCount, MaxCount, MaxFunctionCount;
  uint32_t NumCounts, NumFunctions;
  ProfileSummary(Kind K, std::vector<uint32_t> Cutoffs)
      : PSK(K), DetailedSummaryCutoffs(Cutoffs), TotalCount(0), MaxCount(0),
        MaxFunctionCount(0), NumCounts(0), NumFunctions(0) {}
  ProfileSummary(Kind K)
      : PSK(K), TotalCount(0), MaxCount(0), MaxFunctionCount(0), NumCounts(0),
        NumFunctions(0) {}
  ProfileSummary(Kind K, SummaryEntryVector DetailedSummary,
                 uint64_t TotalCount, uint64_t MaxCount,
                 uint64_t MaxFunctionCount, uint32_t NumCounts,
                 uint32_t NumFunctions)
      : PSK(K), DetailedSummary(DetailedSummary), TotalCount(TotalCount),
        MaxCount(MaxCount), MaxFunctionCount(MaxFunctionCount),
        NumCounts(NumCounts), NumFunctions(NumFunctions) {}
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
  uint32_t getNumFunctions() { return NumFunctions; }
  uint64_t getMaxFunctionCount() { return MaxFunctionCount; }
  /// \brief Get profile summary associated with module \p M
  static inline ProfileSummary *getProfileSummary(Module *M);
  virtual ~ProfileSummary() = default;
  virtual bool operator==(ProfileSummary &Other);
};

class InstrProfSummary final : public ProfileSummary {
  uint64_t MaxInternalBlockCount;
  inline void addEntryCount(uint64_t Count);
  inline void addInternalCount(uint64_t Count);

protected:
  std::vector<Metadata *> getFormatSpecificMD(LLVMContext &Context) override;

public:
  InstrProfSummary(std::vector<uint32_t> Cutoffs)
      : ProfileSummary(PSK_Instr, Cutoffs), MaxInternalBlockCount(0) {}
  InstrProfSummary(const IndexedInstrProf::Summary &S);
  InstrProfSummary(uint64_t TotalCount, uint64_t MaxBlockCount,
                   uint64_t MaxInternalBlockCount, uint64_t MaxFunctionCount,
                   uint32_t NumBlocks, uint32_t NumFunctions,
                   SummaryEntryVector Summary)
      : ProfileSummary(PSK_Instr, Summary, TotalCount, MaxBlockCount,
                       MaxFunctionCount, NumBlocks, NumFunctions),
        MaxInternalBlockCount(MaxInternalBlockCount) {}
  static bool classof(const ProfileSummary *PS) {
    return PS->getKind() == PSK_Instr;
  }
  void addRecord(const InstrProfRecord &);
  uint32_t getNumBlocks() { return NumCounts; }
  uint64_t getTotalCount() { return TotalCount; }
  uint64_t getMaxBlockCount() { return MaxCount; }
  uint64_t getMaxInternalBlockCount() { return MaxInternalBlockCount; }
  bool operator==(ProfileSummary &Other) override;
};

class SampleProfileSummary final : public ProfileSummary {
protected:
  std::vector<Metadata *> getFormatSpecificMD(LLVMContext &Context) override;

public:
  uint32_t getNumLinesWithSamples() { return NumCounts; }
  uint64_t getTotalSamples() { return TotalCount; }
  uint64_t getMaxSamplesPerLine() { return MaxCount; }
  void addRecord(const sampleprof::FunctionSamples &FS);
  SampleProfileSummary(std::vector<uint32_t> Cutoffs)
      : ProfileSummary(PSK_Sample, Cutoffs) {}
  SampleProfileSummary(uint64_t TotalSamples, uint64_t MaxSamplesPerLine,
                       uint64_t MaxFunctionCount, int32_t NumLinesWithSamples,
                       uint32_t NumFunctions,
                       SummaryEntryVector DetailedSummary)
      : ProfileSummary(PSK_Sample, DetailedSummary, TotalSamples,
                       MaxSamplesPerLine, MaxFunctionCount, NumLinesWithSamples,
                       NumFunctions) {}
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

ProfileSummary *ProfileSummary::getProfileSummary(Module *M) {
  if (!M)
    return nullptr;
  sys::SmartScopedLock<true> Lock(*CacheMutex);
  // Computing profile summary for a module involves parsing a fairly large
  // metadata and could be expensive. We use a simple cache of the last seen
  // module and its profile summary.
  if (CachedSummary->first != M) {
    auto *Summary = computeProfileSummary(M);
    // Do not cache if the summary is empty. This is because a later pass
    // (sample profile loader, for example) could attach the summary metadata on
    // the module.
    if (!Summary)
      return nullptr;
    CachedSummary->first = M;
    CachedSummary->second.reset(Summary);
  }
  return CachedSummary->second.get();
}
} // end namespace llvm
#endif
