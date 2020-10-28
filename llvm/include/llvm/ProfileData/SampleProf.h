//===- SampleProf.h - Sampling profiling format support ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains common definitions used in the reading and writing of
// sample profile data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_SAMPLEPROF_H
#define LLVM_PROFILEDATA_SAMPLEPROF_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <system_error>
#include <utility>

namespace llvm {

class raw_ostream;

const std::error_category &sampleprof_category();

enum class sampleprof_error {
  success = 0,
  bad_magic,
  unsupported_version,
  too_large,
  truncated,
  malformed,
  unrecognized_format,
  unsupported_writing_format,
  truncated_name_table,
  not_implemented,
  counter_overflow,
  ostream_seek_unsupported,
  compress_failed,
  uncompress_failed,
  zlib_unavailable
};

inline std::error_code make_error_code(sampleprof_error E) {
  return std::error_code(static_cast<int>(E), sampleprof_category());
}

inline sampleprof_error MergeResult(sampleprof_error &Accumulator,
                                    sampleprof_error Result) {
  // Prefer first error encountered as later errors may be secondary effects of
  // the initial problem.
  if (Accumulator == sampleprof_error::success &&
      Result != sampleprof_error::success)
    Accumulator = Result;
  return Accumulator;
}

} // end namespace llvm

namespace std {

template <>
struct is_error_code_enum<llvm::sampleprof_error> : std::true_type {};

} // end namespace std

namespace llvm {
namespace sampleprof {

enum SampleProfileFormat {
  SPF_None = 0,
  SPF_Text = 0x1,
  SPF_Compact_Binary = 0x2,
  SPF_GCC = 0x3,
  SPF_Ext_Binary = 0x4,
  SPF_Binary = 0xff
};

static inline uint64_t SPMagic(SampleProfileFormat Format = SPF_Binary) {
  return uint64_t('S') << (64 - 8) | uint64_t('P') << (64 - 16) |
         uint64_t('R') << (64 - 24) | uint64_t('O') << (64 - 32) |
         uint64_t('F') << (64 - 40) | uint64_t('4') << (64 - 48) |
         uint64_t('2') << (64 - 56) | uint64_t(Format);
}

/// Get the proper representation of a string according to whether the
/// current Format uses MD5 to represent the string.
static inline StringRef getRepInFormat(StringRef Name, bool UseMD5,
                                       std::string &GUIDBuf) {
  if (Name.empty())
    return Name;
  GUIDBuf = std::to_string(Function::getGUID(Name));
  return UseMD5 ? StringRef(GUIDBuf) : Name;
}

static inline uint64_t SPVersion() { return 103; }

// Section Type used by SampleProfileExtBinaryBaseReader and
// SampleProfileExtBinaryBaseWriter. Never change the existing
// value of enum. Only append new ones.
enum SecType {
  SecInValid = 0,
  SecProfSummary = 1,
  SecNameTable = 2,
  SecProfileSymbolList = 3,
  SecFuncOffsetTable = 4,
  // marker for the first type of profile.
  SecFuncProfileFirst = 32,
  SecLBRProfile = SecFuncProfileFirst
};

static inline std::string getSecName(SecType Type) {
  switch (Type) {
  case SecInValid:
    return "InvalidSection";
  case SecProfSummary:
    return "ProfileSummarySection";
  case SecNameTable:
    return "NameTableSection";
  case SecProfileSymbolList:
    return "ProfileSymbolListSection";
  case SecFuncOffsetTable:
    return "FuncOffsetTableSection";
  case SecLBRProfile:
    return "LBRProfileSection";
  }
  llvm_unreachable("A SecType has no name for output");
}

// Entry type of section header table used by SampleProfileExtBinaryBaseReader
// and SampleProfileExtBinaryBaseWriter.
struct SecHdrTableEntry {
  SecType Type;
  uint64_t Flags;
  uint64_t Offset;
  uint64_t Size;
};

// Flags common for all sections are defined here. In SecHdrTableEntry::Flags,
// common flags will be saved in the lower 32bits and section specific flags
// will be saved in the higher 32 bits.
enum class SecCommonFlags : uint32_t {
  SecFlagInValid = 0,
  SecFlagCompress = (1 << 0)
};

// Section specific flags are defined here.
// !!!Note: Everytime a new enum class is created here, please add
// a new check in verifySecFlag.
enum class SecNameTableFlags : uint32_t {
  SecFlagInValid = 0,
  SecFlagMD5Name = (1 << 0)
};
enum class SecProfSummaryFlags : uint32_t {
  SecFlagInValid = 0,
  /// SecFlagPartial means the profile is for common/shared code.
  /// The common profile is usually merged from profiles collected
  /// from running other targets.
  SecFlagPartial = (1 << 0)
};

// Verify section specific flag is used for the correct section.
template <class SecFlagType>
static inline void verifySecFlag(SecType Type, SecFlagType Flag) {
  // No verification is needed for common flags.
  if (std::is_same<SecCommonFlags, SecFlagType>())
    return;

  // Verification starts here for section specific flag.
  bool IsFlagLegal = false;
  switch (Type) {
  case SecNameTable:
    IsFlagLegal = std::is_same<SecNameTableFlags, SecFlagType>();
    break;
  case SecProfSummary:
    IsFlagLegal = std::is_same<SecProfSummaryFlags, SecFlagType>();
    break;
  default:
    break;
  }
  if (!IsFlagLegal)
    llvm_unreachable("Misuse of a flag in an incompatible section");
}

template <class SecFlagType>
static inline void addSecFlag(SecHdrTableEntry &Entry, SecFlagType Flag) {
  verifySecFlag(Entry.Type, Flag);
  auto FVal = static_cast<uint64_t>(Flag);
  bool IsCommon = std::is_same<SecCommonFlags, SecFlagType>();
  Entry.Flags |= IsCommon ? FVal : (FVal << 32);
}

template <class SecFlagType>
static inline void removeSecFlag(SecHdrTableEntry &Entry, SecFlagType Flag) {
  verifySecFlag(Entry.Type, Flag);
  auto FVal = static_cast<uint64_t>(Flag);
  bool IsCommon = std::is_same<SecCommonFlags, SecFlagType>();
  Entry.Flags &= ~(IsCommon ? FVal : (FVal << 32));
}

template <class SecFlagType>
static inline bool hasSecFlag(const SecHdrTableEntry &Entry, SecFlagType Flag) {
  verifySecFlag(Entry.Type, Flag);
  auto FVal = static_cast<uint64_t>(Flag);
  bool IsCommon = std::is_same<SecCommonFlags, SecFlagType>();
  return Entry.Flags & (IsCommon ? FVal : (FVal << 32));
}

/// Represents the relative location of an instruction.
///
/// Instruction locations are specified by the line offset from the
/// beginning of the function (marked by the line where the function
/// header is) and the discriminator value within that line.
///
/// The discriminator value is useful to distinguish instructions
/// that are on the same line but belong to different basic blocks
/// (e.g., the two post-increment instructions in "if (p) x++; else y++;").
struct LineLocation {
  LineLocation(uint32_t L, uint32_t D) : LineOffset(L), Discriminator(D) {}

  void print(raw_ostream &OS) const;
  void dump() const;

  bool operator<(const LineLocation &O) const {
    return LineOffset < O.LineOffset ||
           (LineOffset == O.LineOffset && Discriminator < O.Discriminator);
  }

  uint32_t LineOffset;
  uint32_t Discriminator;
};

raw_ostream &operator<<(raw_ostream &OS, const LineLocation &Loc);

/// Representation of a single sample record.
///
/// A sample record is represented by a positive integer value, which
/// indicates how frequently was the associated line location executed.
///
/// Additionally, if the associated location contains a function call,
/// the record will hold a list of all the possible called targets. For
/// direct calls, this will be the exact function being invoked. For
/// indirect calls (function pointers, virtual table dispatch), this
/// will be a list of one or more functions.
class SampleRecord {
public:
  using CallTarget = std::pair<StringRef, uint64_t>;
  struct CallTargetComparator {
    bool operator()(const CallTarget &LHS, const CallTarget &RHS) const {
      if (LHS.second != RHS.second)
        return LHS.second > RHS.second;

      return LHS.first < RHS.first;
    }
  };

  using SortedCallTargetSet = std::set<CallTarget, CallTargetComparator>;
  using CallTargetMap = StringMap<uint64_t>;
  SampleRecord() = default;

  /// Increment the number of samples for this record by \p S.
  /// Optionally scale sample count \p S by \p Weight.
  ///
  /// Sample counts accumulate using saturating arithmetic, to avoid wrapping
  /// around unsigned integers.
  sampleprof_error addSamples(uint64_t S, uint64_t Weight = 1) {
    bool Overflowed;
    NumSamples = SaturatingMultiplyAdd(S, Weight, NumSamples, &Overflowed);
    return Overflowed ? sampleprof_error::counter_overflow
                      : sampleprof_error::success;
  }

  /// Add called function \p F with samples \p S.
  /// Optionally scale sample count \p S by \p Weight.
  ///
  /// Sample counts accumulate using saturating arithmetic, to avoid wrapping
  /// around unsigned integers.
  sampleprof_error addCalledTarget(StringRef F, uint64_t S,
                                   uint64_t Weight = 1) {
    uint64_t &TargetSamples = CallTargets[F];
    bool Overflowed;
    TargetSamples =
        SaturatingMultiplyAdd(S, Weight, TargetSamples, &Overflowed);
    return Overflowed ? sampleprof_error::counter_overflow
                      : sampleprof_error::success;
  }

  /// Return true if this sample record contains function calls.
  bool hasCalls() const { return !CallTargets.empty(); }

  uint64_t getSamples() const { return NumSamples; }
  const CallTargetMap &getCallTargets() const { return CallTargets; }
  const SortedCallTargetSet getSortedCallTargets() const {
    return SortCallTargets(CallTargets);
  }

  /// Sort call targets in descending order of call frequency.
  static const SortedCallTargetSet SortCallTargets(const CallTargetMap &Targets) {
    SortedCallTargetSet SortedTargets;
    for (const auto &I : Targets) {
      SortedTargets.emplace(I.first(), I.second);
    }
    return SortedTargets;
  }

  /// Merge the samples in \p Other into this record.
  /// Optionally scale sample counts by \p Weight.
  sampleprof_error merge(const SampleRecord &Other, uint64_t Weight = 1) {
    sampleprof_error Result = addSamples(Other.getSamples(), Weight);
    for (const auto &I : Other.getCallTargets()) {
      MergeResult(Result, addCalledTarget(I.first(), I.second, Weight));
    }
    return Result;
  }

  void print(raw_ostream &OS, unsigned Indent) const;
  void dump() const;

private:
  uint64_t NumSamples = 0;
  CallTargetMap CallTargets;
};

raw_ostream &operator<<(raw_ostream &OS, const SampleRecord &Sample);

class FunctionSamples;
class SampleProfileReaderItaniumRemapper;

using BodySampleMap = std::map<LineLocation, SampleRecord>;
// NOTE: Using a StringMap here makes parsed profiles consume around 17% more
// memory, which is *very* significant for large profiles.
using FunctionSamplesMap = std::map<std::string, FunctionSamples, std::less<>>;
using CallsiteSampleMap = std::map<LineLocation, FunctionSamplesMap>;

/// Representation of the samples collected for a function.
///
/// This data structure contains all the collected samples for the body
/// of a function. Each sample corresponds to a LineLocation instance
/// within the body of the function.
class FunctionSamples {
public:
  FunctionSamples() = default;

  void print(raw_ostream &OS = dbgs(), unsigned Indent = 0) const;
  void dump() const;

  sampleprof_error addTotalSamples(uint64_t Num, uint64_t Weight = 1) {
    bool Overflowed;
    TotalSamples =
        SaturatingMultiplyAdd(Num, Weight, TotalSamples, &Overflowed);
    return Overflowed ? sampleprof_error::counter_overflow
                      : sampleprof_error::success;
  }

  sampleprof_error addHeadSamples(uint64_t Num, uint64_t Weight = 1) {
    bool Overflowed;
    TotalHeadSamples =
        SaturatingMultiplyAdd(Num, Weight, TotalHeadSamples, &Overflowed);
    return Overflowed ? sampleprof_error::counter_overflow
                      : sampleprof_error::success;
  }

  sampleprof_error addBodySamples(uint32_t LineOffset, uint32_t Discriminator,
                                  uint64_t Num, uint64_t Weight = 1) {
    return BodySamples[LineLocation(LineOffset, Discriminator)].addSamples(
        Num, Weight);
  }

  sampleprof_error addCalledTargetSamples(uint32_t LineOffset,
                                          uint32_t Discriminator,
                                          StringRef FName, uint64_t Num,
                                          uint64_t Weight = 1) {
    return BodySamples[LineLocation(LineOffset, Discriminator)].addCalledTarget(
        FName, Num, Weight);
  }

  /// Return the number of samples collected at the given location.
  /// Each location is specified by \p LineOffset and \p Discriminator.
  /// If the location is not found in profile, return error.
  ErrorOr<uint64_t> findSamplesAt(uint32_t LineOffset,
                                  uint32_t Discriminator) const {
    const auto &ret = BodySamples.find(LineLocation(LineOffset, Discriminator));
    if (ret == BodySamples.end())
      return std::error_code();
    else
      return ret->second.getSamples();
  }

  /// Returns the call target map collected at a given location.
  /// Each location is specified by \p LineOffset and \p Discriminator.
  /// If the location is not found in profile, return error.
  ErrorOr<SampleRecord::CallTargetMap>
  findCallTargetMapAt(uint32_t LineOffset, uint32_t Discriminator) const {
    const auto &ret = BodySamples.find(LineLocation(LineOffset, Discriminator));
    if (ret == BodySamples.end())
      return std::error_code();
    return ret->second.getCallTargets();
  }

  /// Return the function samples at the given callsite location.
  FunctionSamplesMap &functionSamplesAt(const LineLocation &Loc) {
    return CallsiteSamples[Loc];
  }

  /// Returns the FunctionSamplesMap at the given \p Loc.
  const FunctionSamplesMap *
  findFunctionSamplesMapAt(const LineLocation &Loc) const {
    auto iter = CallsiteSamples.find(Loc);
    if (iter == CallsiteSamples.end())
      return nullptr;
    return &iter->second;
  }

  /// Returns a pointer to FunctionSamples at the given callsite location
  /// \p Loc with callee \p CalleeName. If no callsite can be found, relax
  /// the restriction to return the FunctionSamples at callsite location
  /// \p Loc with the maximum total sample count. If \p Remapper is not
  /// nullptr, use \p Remapper to find FunctionSamples with equivalent name
  /// as \p CalleeName.
  const FunctionSamples *
  findFunctionSamplesAt(const LineLocation &Loc, StringRef CalleeName,
                        SampleProfileReaderItaniumRemapper *Remapper) const;

  bool empty() const { return TotalSamples == 0; }

  /// Return the total number of samples collected inside the function.
  uint64_t getTotalSamples() const { return TotalSamples; }

  /// Return the total number of branch samples that have the function as the
  /// branch target. This should be equivalent to the sample of the first
  /// instruction of the symbol. But as we directly get this info for raw
  /// profile without referring to potentially inaccurate debug info, this
  /// gives more accurate profile data and is preferred for standalone symbols.
  uint64_t getHeadSamples() const { return TotalHeadSamples; }

  /// Return the sample count of the first instruction of the function.
  /// The function can be either a standalone symbol or an inlined function.
  uint64_t getEntrySamples() const {
    uint64_t Count = 0;
    // Use either BodySamples or CallsiteSamples which ever has the smaller
    // lineno.
    if (!BodySamples.empty() &&
        (CallsiteSamples.empty() ||
         BodySamples.begin()->first < CallsiteSamples.begin()->first))
      Count = BodySamples.begin()->second.getSamples();
    else if (!CallsiteSamples.empty()) {
      // An indirect callsite may be promoted to several inlined direct calls.
      // We need to get the sum of them.
      for (const auto &N_FS : CallsiteSamples.begin()->second)
        Count += N_FS.second.getEntrySamples();
    }
    // Return at least 1 if total sample is not 0.
    return Count ? Count : TotalSamples > 0;
  }

  /// Return all the samples collected in the body of the function.
  const BodySampleMap &getBodySamples() const { return BodySamples; }

  /// Return all the callsite samples collected in the body of the function.
  const CallsiteSampleMap &getCallsiteSamples() const {
    return CallsiteSamples;
  }

  /// Return the maximum of sample counts in a function body including functions
  /// inlined in it.
  uint64_t getMaxCountInside() const {
    uint64_t MaxCount = 0;
    for (const auto &L : getBodySamples())
      MaxCount = std::max(MaxCount, L.second.getSamples());
    for (const auto &C : getCallsiteSamples())
      for (const FunctionSamplesMap::value_type &F : C.second)
        MaxCount = std::max(MaxCount, F.second.getMaxCountInside());
    return MaxCount;
  }

  /// Merge the samples in \p Other into this one.
  /// Optionally scale samples by \p Weight.
  sampleprof_error merge(const FunctionSamples &Other, uint64_t Weight = 1) {
    sampleprof_error Result = sampleprof_error::success;
    Name = Other.getName();
    if (!GUIDToFuncNameMap)
      GUIDToFuncNameMap = Other.GUIDToFuncNameMap;
    MergeResult(Result, addTotalSamples(Other.getTotalSamples(), Weight));
    MergeResult(Result, addHeadSamples(Other.getHeadSamples(), Weight));
    for (const auto &I : Other.getBodySamples()) {
      const LineLocation &Loc = I.first;
      const SampleRecord &Rec = I.second;
      MergeResult(Result, BodySamples[Loc].merge(Rec, Weight));
    }
    for (const auto &I : Other.getCallsiteSamples()) {
      const LineLocation &Loc = I.first;
      FunctionSamplesMap &FSMap = functionSamplesAt(Loc);
      for (const auto &Rec : I.second)
        MergeResult(Result, FSMap[Rec.first].merge(Rec.second, Weight));
    }
    return Result;
  }

  /// Recursively traverses all children, if the total sample count of the
  /// corresponding function is no less than \p Threshold, add its corresponding
  /// GUID to \p S. Also traverse the BodySamples to add hot CallTarget's GUID
  /// to \p S.
  void findInlinedFunctions(DenseSet<GlobalValue::GUID> &S, const Module *M,
                            uint64_t Threshold) const {
    if (TotalSamples <= Threshold)
      return;
    auto isDeclaration = [](const Function *F) {
      return !F || F->isDeclaration();
    };
    if (isDeclaration(M->getFunction(getFuncName()))) {
      // Add to the import list only when it's defined out of module.
      S.insert(getGUID(Name));
    }
    // Import hot CallTargets, which may not be available in IR because full
    // profile annotation cannot be done until backend compilation in ThinLTO.
    for (const auto &BS : BodySamples)
      for (const auto &TS : BS.second.getCallTargets())
        if (TS.getValue() > Threshold) {
          const Function *Callee = M->getFunction(getFuncName(TS.getKey()));
          if (isDeclaration(Callee))
            S.insert(getGUID(TS.getKey()));
        }
    for (const auto &CS : CallsiteSamples)
      for (const auto &NameFS : CS.second)
        NameFS.second.findInlinedFunctions(S, M, Threshold);
  }

  /// Set the name of the function.
  void setName(StringRef FunctionName) { Name = FunctionName; }

  /// Return the function name.
  StringRef getName() const { return Name; }

  /// Return the original function name.
  StringRef getFuncName() const { return getFuncName(Name); }

  /// Return the canonical name for a function, taking into account
  /// suffix elision policy attributes.
  static StringRef getCanonicalFnName(const Function &F) {
    static const char *knownSuffixes[] = { ".llvm.", ".part." };
    auto AttrName = "sample-profile-suffix-elision-policy";
    auto Attr = F.getFnAttribute(AttrName).getValueAsString();
    if (Attr == "" || Attr == "all") {
      return F.getName().split('.').first;
    } else if (Attr == "selected") {
      StringRef Cand(F.getName());
      for (const auto &Suf : knownSuffixes) {
        StringRef Suffix(Suf);
        auto It = Cand.rfind(Suffix);
        if (It == StringRef::npos)
          return Cand;
        auto Dit = Cand.rfind('.');
        if (Dit == It + Suffix.size() - 1)
          Cand = Cand.substr(0, It);
      }
      return Cand;
    } else if (Attr == "none") {
      return F.getName();
    } else {
      assert(false && "internal error: unknown suffix elision policy");
    }
    return F.getName();
  }

  /// Translate \p Name into its original name.
  /// When profile doesn't use MD5, \p Name needs no translation.
  /// When profile uses MD5, \p Name in current FunctionSamples
  /// is actually GUID of the original function name. getFuncName will
  /// translate \p Name in current FunctionSamples into its original name
  /// by looking up in the function map GUIDToFuncNameMap.
  /// If the original name doesn't exist in the map, return empty StringRef.
  StringRef getFuncName(StringRef Name) const {
    if (!UseMD5)
      return Name;

    assert(GUIDToFuncNameMap && "GUIDToFuncNameMap needs to be popluated first");
    auto iter = GUIDToFuncNameMap->find(std::stoull(Name.data()));
    if (iter == GUIDToFuncNameMap->end())
      return StringRef();
    return iter->second;
  }

  /// Returns the line offset to the start line of the subprogram.
  /// We assume that a single function will not exceed 65535 LOC.
  static unsigned getOffset(const DILocation *DIL);

  /// Get the FunctionSamples of the inline instance where DIL originates
  /// from.
  ///
  /// The FunctionSamples of the instruction (Machine or IR) associated to
  /// \p DIL is the inlined instance in which that instruction is coming from.
  /// We traverse the inline stack of that instruction, and match it with the
  /// tree nodes in the profile.
  ///
  /// \returns the FunctionSamples pointer to the inlined instance.
  /// If \p Remapper is not nullptr, it will be used to find matching
  /// FunctionSamples with not exactly the same but equivalent name.
  const FunctionSamples *findFunctionSamples(
      const DILocation *DIL,
      SampleProfileReaderItaniumRemapper *Remapper = nullptr) const;

  static SampleProfileFormat Format;

  /// Whether the profile uses MD5 to represent string.
  static bool UseMD5;

  /// GUIDToFuncNameMap saves the mapping from GUID to the symbol name, for
  /// all the function symbols defined or declared in current module.
  DenseMap<uint64_t, StringRef> *GUIDToFuncNameMap = nullptr;

  // Assume the input \p Name is a name coming from FunctionSamples itself.
  // If UseMD5 is true, the name is already a GUID and we
  // don't want to return the GUID of GUID.
  static uint64_t getGUID(StringRef Name) {
    return UseMD5 ? std::stoull(Name.data()) : Function::getGUID(Name);
  }

  // Find all the names in the current FunctionSamples including names in
  // all the inline instances and names of call targets.
  void findAllNames(DenseSet<StringRef> &NameSet) const;

private:
  /// Mangled name of the function.
  StringRef Name;

  /// Total number of samples collected inside this function.
  ///
  /// Samples are cumulative, they include all the samples collected
  /// inside this function and all its inlined callees.
  uint64_t TotalSamples = 0;

  /// Total number of samples collected at the head of the function.
  /// This is an approximation of the number of calls made to this function
  /// at runtime.
  uint64_t TotalHeadSamples = 0;

  /// Map instruction locations to collected samples.
  ///
  /// Each entry in this map contains the number of samples
  /// collected at the corresponding line offset. All line locations
  /// are an offset from the start of the function.
  BodySampleMap BodySamples;

  /// Map call sites to collected samples for the called function.
  ///
  /// Each entry in this map corresponds to all the samples
  /// collected for the inlined function call at the given
  /// location. For example, given:
  ///
  ///     void foo() {
  ///  1    bar();
  ///  ...
  ///  8    baz();
  ///     }
  ///
  /// If the bar() and baz() calls were inlined inside foo(), this
  /// map will contain two entries.  One for all the samples collected
  /// in the call to bar() at line offset 1, the other for all the samples
  /// collected in the call to baz() at line offset 8.
  CallsiteSampleMap CallsiteSamples;
};

raw_ostream &operator<<(raw_ostream &OS, const FunctionSamples &FS);

/// Sort a LocationT->SampleT map by LocationT.
///
/// It produces a sorted list of <LocationT, SampleT> records by ascending
/// order of LocationT.
template <class LocationT, class SampleT> class SampleSorter {
public:
  using SamplesWithLoc = std::pair<const LocationT, SampleT>;
  using SamplesWithLocList = SmallVector<const SamplesWithLoc *, 20>;

  SampleSorter(const std::map<LocationT, SampleT> &Samples) {
    for (const auto &I : Samples)
      V.push_back(&I);
    llvm::stable_sort(V, [](const SamplesWithLoc *A, const SamplesWithLoc *B) {
      return A->first < B->first;
    });
  }

  const SamplesWithLocList &get() const { return V; }

private:
  SamplesWithLocList V;
};

/// ProfileSymbolList records the list of function symbols shown up
/// in the binary used to generate the profile. It is useful to
/// to discriminate a function being so cold as not to shown up
/// in the profile and a function newly added.
class ProfileSymbolList {
public:
  /// copy indicates whether we need to copy the underlying memory
  /// for the input Name.
  void add(StringRef Name, bool copy = false) {
    if (!copy) {
      Syms.insert(Name);
      return;
    }
    Syms.insert(Name.copy(Allocator));
  }

  bool contains(StringRef Name) { return Syms.count(Name); }

  void merge(const ProfileSymbolList &List) {
    for (auto Sym : List.Syms)
      add(Sym, true);
  }

  unsigned size() { return Syms.size(); }

  void setToCompress(bool TC) { ToCompress = TC; }
  bool toCompress() { return ToCompress; }

  std::error_code read(const uint8_t *Data, uint64_t ListSize);
  std::error_code write(raw_ostream &OS);
  void dump(raw_ostream &OS = dbgs()) const;

private:
  // Determine whether or not to compress the symbol list when
  // writing it into profile. The variable is unused when the symbol
  // list is read from an existing profile.
  bool ToCompress = false;
  DenseSet<StringRef> Syms;
  BumpPtrAllocator Allocator;
};

} // end namespace sampleprof
} // end namespace llvm

#endif // LLVM_PROFILEDATA_SAMPLEPROF_H
