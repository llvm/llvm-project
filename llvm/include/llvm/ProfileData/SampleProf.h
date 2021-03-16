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
  zlib_unavailable,
  hash_mismatch
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
  SecFuncMetadata = 5,
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
  case SecFuncMetadata:
    return "FunctionMetadata";
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
  // The index indicating the location of the current entry in
  // SectionHdrLayout table.
  uint32_t LayoutIndex;
};

// Flags common for all sections are defined here. In SecHdrTableEntry::Flags,
// common flags will be saved in the lower 32bits and section specific flags
// will be saved in the higher 32 bits.
enum class SecCommonFlags : uint32_t {
  SecFlagInValid = 0,
  SecFlagCompress = (1 << 0),
  // Indicate the section contains only profile without context.
  SecFlagFlat = (1 << 1)
};

// Section specific flags are defined here.
// !!!Note: Everytime a new enum class is created here, please add
// a new check in verifySecFlag.
enum class SecNameTableFlags : uint32_t {
  SecFlagInValid = 0,
  SecFlagMD5Name = (1 << 0),
  // Store MD5 in fixed length instead of ULEB128 so NameTable can be
  // accessed like an array.
  SecFlagFixedLengthMD5 = (1 << 1),
  // Profile contains ".__uniq." suffix name. Compiler shouldn't strip
  // the suffix when doing profile matching when seeing the flag.
  SecFlagUniqSuffix = (1 << 2)
};
enum class SecProfSummaryFlags : uint32_t {
  SecFlagInValid = 0,
  /// SecFlagPartial means the profile is for common/shared code.
  /// The common profile is usually merged from profiles collected
  /// from running other targets.
  SecFlagPartial = (1 << 0),
  /// SecFlagContext means this is context-sensitive profile for
  /// CSSPGO
  SecFlagFullContext = (1 << 1)
};

enum class SecFuncMetadataFlags : uint32_t {
  SecFlagInvalid = 0,
  SecFlagIsProbeBased = (1 << 0),
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
  case SecFuncMetadata:
    IsFlagLegal = std::is_same<SecFuncMetadataFlags, SecFlagType>();
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

  bool operator==(const LineLocation &O) const {
    return LineOffset == O.LineOffset && Discriminator == O.Discriminator;
  }

  bool operator!=(const LineLocation &O) const {
    return LineOffset != O.LineOffset || Discriminator != O.Discriminator;
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

  /// Prorate call targets by a distribution factor.
  static const CallTargetMap adjustCallTargets(const CallTargetMap &Targets,
                                               float DistributionFactor) {
    CallTargetMap AdjustedTargets;
    for (const auto &I : Targets) {
      AdjustedTargets[I.first()] = I.second * DistributionFactor;
    }
    return AdjustedTargets;
  }

  /// Merge the samples in \p Other into this record.
  /// Optionally scale sample counts by \p Weight.
  sampleprof_error merge(const SampleRecord &Other, uint64_t Weight = 1);
  void print(raw_ostream &OS, unsigned Indent) const;
  void dump() const;

private:
  uint64_t NumSamples = 0;
  CallTargetMap CallTargets;
};

raw_ostream &operator<<(raw_ostream &OS, const SampleRecord &Sample);

// State of context associated with FunctionSamples
enum ContextStateMask {
  UnknownContext = 0x0,   // Profile without context
  RawContext = 0x1,       // Full context profile from input profile
  SyntheticContext = 0x2, // Synthetic context created for context promotion
  InlinedContext = 0x4,   // Profile for context that is inlined into caller
  MergedContext = 0x8     // Profile for context merged into base profile
};

// Sample context for FunctionSamples. It consists of the calling context,
// the function name and context state. Internally sample context is represented
// using StringRef, which is also the input for constructing a `SampleContext`.
// It can accept and represent both full context string as well as context-less
// function name.
// Example of full context string (note the wrapping `[]`):
//    `[main:3 @ _Z5funcAi:1 @ _Z8funcLeafi]`
// Example of context-less function name (same as AutoFDO):
//    `_Z8funcLeafi`
class SampleContext {
public:
  SampleContext() : State(UnknownContext) {}
  SampleContext(StringRef ContextStr,
                ContextStateMask CState = UnknownContext) {
    setContext(ContextStr, CState);
  }

  // Promote context by removing top frames (represented by `ContextStrToRemove`).
  // Note that with string representation of context, the promotion is effectively
  // a substr operation with `ContextStrToRemove` removed from left.
  void promoteOnPath(StringRef ContextStrToRemove) {
    assert(FullContext.startswith(ContextStrToRemove));

    // Remove leading context and frame separator " @ ".
    FullContext = FullContext.substr(ContextStrToRemove.size() + 3);
    CallingContext = CallingContext.substr(ContextStrToRemove.size() + 3);
  }

  // Split the top context frame (left-most substr) from context.
  static std::pair<StringRef, StringRef>
  splitContextString(StringRef ContextStr) {
    return ContextStr.split(" @ ");
  }

  // Decode context string for a frame to get function name and location.
  // `ContextStr` is in the form of `FuncName:StartLine.Discriminator`.
  static void decodeContextString(StringRef ContextStr, StringRef &FName,
                                  LineLocation &LineLoc) {
    // Get function name
    auto EntrySplit = ContextStr.split(':');
    FName = EntrySplit.first;

    LineLoc = {0, 0};
    if (!EntrySplit.second.empty()) {
      // Get line offset, use signed int for getAsInteger so string will
      // be parsed as signed.
      int LineOffset = 0;
      auto LocSplit = EntrySplit.second.split('.');
      LocSplit.first.getAsInteger(10, LineOffset);
      LineLoc.LineOffset = LineOffset;

      // Get discriminator
      if (!LocSplit.second.empty())
        LocSplit.second.getAsInteger(10, LineLoc.Discriminator);
    }
  }

  operator StringRef() const { return FullContext; }
  bool hasState(ContextStateMask S) { return State & (uint32_t)S; }
  void setState(ContextStateMask S) { State |= (uint32_t)S; }
  void clearState(ContextStateMask S) { State &= (uint32_t)~S; }
  bool hasContext() const { return State != UnknownContext; }
  bool isBaseContext() const { return CallingContext.empty(); }
  StringRef getNameWithoutContext() const { return Name; }
  StringRef getCallingContext() const { return CallingContext; }
  StringRef getNameWithContext(bool WithBracket = false) const {
    return WithBracket ? InputContext : FullContext;
  }

private:
  // Give a context string, decode and populate internal states like
  // Function name, Calling context and context state. Example of input
  // `ContextStr`: `[main:3 @ _Z5funcAi:1 @ _Z8funcLeafi]`
  void setContext(StringRef ContextStr, ContextStateMask CState) {
    assert(!ContextStr.empty());
    InputContext = ContextStr;
    // Note that `[]` wrapped input indicates a full context string, otherwise
    // it's treated as context-less function name only.
    bool HasContext = ContextStr.startswith("[");
    if (!HasContext && CState == UnknownContext) {
      State = UnknownContext;
      Name = FullContext = ContextStr;
    } else {
      // Assume raw context profile if unspecified
      if (CState == UnknownContext)
        State = RawContext;
      else
        State = CState;

      // Remove encapsulating '[' and ']' if any
      if (HasContext)
        FullContext = ContextStr.substr(1, ContextStr.size() - 2);
      else
        FullContext = ContextStr;

      // Caller is to the left of callee in context string
      auto NameContext = FullContext.rsplit(" @ ");
      if (NameContext.second.empty()) {
        Name = NameContext.first;
        CallingContext = NameContext.second;
      } else {
        Name = NameContext.second;
        CallingContext = NameContext.first;
      }
    }
  }

  // Input context string including bracketed calling context and leaf function
  // name
  StringRef InputContext;
  // Full context string including calling context and leaf function name
  StringRef FullContext;
  // Function name for the associated sample profile
  StringRef Name;
  // Calling context (leaf function excluded) for the associated sample profile
  StringRef CallingContext;
  // State of the associated sample profile
  uint32_t State;
};

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

  void setTotalSamples(uint64_t Num) { TotalSamples = Num; }

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

  sampleprof_error addBodySamplesForProbe(uint32_t Index, uint64_t Num,
                                          uint64_t Weight = 1) {
    SampleRecord S;
    S.addSamples(Num, Weight);
    return BodySamples[LineLocation(Index, 0)].merge(S, Weight);
  }

  /// Return the number of samples collected at the given location.
  /// Each location is specified by \p LineOffset and \p Discriminator.
  /// If the location is not found in profile, return error.
  ErrorOr<uint64_t> findSamplesAt(uint32_t LineOffset,
                                  uint32_t Discriminator) const {
    const auto &ret = BodySamples.find(LineLocation(LineOffset, Discriminator));
    if (ret == BodySamples.end()) {
      // For CSSPGO, in order to conserve profile size, we no longer write out
      // locations profile for those not hit during training, so we need to
      // treat them as zero instead of error here.
      if (FunctionSamples::ProfileIsCS || FunctionSamples::ProfileIsProbeBased)
        return 0;
      return std::error_code();
    } else {
      // Return error for an invalid sample count which is usually assigned to
      // dangling probe.
      if (FunctionSamples::ProfileIsProbeBased &&
          ret->second.getSamples() == FunctionSamples::InvalidProbeCount)
        return std::error_code();
      return ret->second.getSamples();
    }
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

  /// Returns the call target map collected at a given location specified by \p
  /// CallSite. If the location is not found in profile, return error.
  ErrorOr<SampleRecord::CallTargetMap>
  findCallTargetMapAt(const LineLocation &CallSite) const {
    const auto &Ret = BodySamples.find(CallSite);
    if (Ret == BodySamples.end())
      return std::error_code();
    return Ret->second.getCallTargets();
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
    if (FunctionSamples::ProfileIsCS && getHeadSamples()) {
      // For CS profile, if we already have more accurate head samples
      // counted by branch sample from caller, use them as entry samples.
      return getHeadSamples();
    }
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
    if (Context.getNameWithContext(true).empty())
      Context = Other.getContext();
    if (FunctionHash == 0) {
      // Set the function hash code for the target profile.
      FunctionHash = Other.getFunctionHash();
    } else if (FunctionHash != Other.getFunctionHash()) {
      // The two profiles coming with different valid hash codes indicates
      // either:
      // 1. They are same-named static functions from different compilation
      // units (without using -unique-internal-linkage-names), or
      // 2. They are really the same function but from different compilations.
      // Let's bail out in either case for now, which means one profile is
      // dropped.
      return sampleprof_error::hash_mismatch;
    }

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
  void findInlinedFunctions(DenseSet<GlobalValue::GUID> &S,
                            const StringMap<Function *> &SymbolMap,
                            uint64_t Threshold) const {
    if (TotalSamples <= Threshold)
      return;
    auto isDeclaration = [](const Function *F) {
      return !F || F->isDeclaration();
    };
    if (isDeclaration(SymbolMap.lookup(getFuncName()))) {
      // Add to the import list only when it's defined out of module.
      S.insert(getGUID(Name));
    }
    // Import hot CallTargets, which may not be available in IR because full
    // profile annotation cannot be done until backend compilation in ThinLTO.
    for (const auto &BS : BodySamples)
      for (const auto &TS : BS.second.getCallTargets())
        if (TS.getValue() > Threshold) {
          const Function *Callee = SymbolMap.lookup(getFuncName(TS.getKey()));
          if (isDeclaration(Callee))
            S.insert(getGUID(TS.getKey()));
        }
    for (const auto &CS : CallsiteSamples)
      for (const auto &NameFS : CS.second)
        NameFS.second.findInlinedFunctions(S, SymbolMap, Threshold);
  }

  /// Set the name of the function.
  void setName(StringRef FunctionName) { Name = FunctionName; }

  /// Return the function name.
  StringRef getName() const { return Name; }

  /// Return function name with context.
  StringRef getNameWithContext(bool WithBracket = false) const {
    return FunctionSamples::ProfileIsCS
               ? Context.getNameWithContext(WithBracket)
               : Name;
  }

  /// Return the original function name.
  StringRef getFuncName() const { return getFuncName(Name); }

  void setFunctionHash(uint64_t Hash) { FunctionHash = Hash; }

  uint64_t getFunctionHash() const { return FunctionHash; }

  /// Return the canonical name for a function, taking into account
  /// suffix elision policy attributes.
  static StringRef getCanonicalFnName(const Function &F) {
    auto AttrName = "sample-profile-suffix-elision-policy";
    auto Attr = F.getFnAttribute(AttrName).getValueAsString();
    return getCanonicalFnName(F.getName(), Attr);
  }

  /// Name suffixes which canonicalization should handle to avoid
  /// profile mismatch.
  static constexpr const char *LLVMSuffix = ".llvm.";
  static constexpr const char *PartSuffix = ".part.";
  static constexpr const char *UniqSuffix = ".__uniq.";

  static StringRef getCanonicalFnName(StringRef FnName,
                                      StringRef Attr = "selected") {
    // Note the sequence of the suffixes in the knownSuffixes array matters.
    // If suffix "A" is appended after the suffix "B", "A" should be in front
    // of "B" in knownSuffixes.
    const char *knownSuffixes[] = {LLVMSuffix, PartSuffix, UniqSuffix};
    if (Attr == "" || Attr == "all") {
      return FnName.split('.').first;
    } else if (Attr == "selected") {
      StringRef Cand(FnName);
      for (const auto &Suf : knownSuffixes) {
        StringRef Suffix(Suf);
        // If the profile contains ".__uniq." suffix, don't strip the
        // suffix for names in the IR.
        if (Suffix == UniqSuffix && FunctionSamples::HasUniqSuffix)
          continue;
        auto It = Cand.rfind(Suffix);
        if (It == StringRef::npos)
          continue;
        auto Dit = Cand.rfind('.');
        if (Dit == It + Suffix.size() - 1)
          Cand = Cand.substr(0, It);
      }
      return Cand;
    } else if (Attr == "none") {
      return FnName;
    } else {
      assert(false && "internal error: unknown suffix elision policy");
    }
    return FnName;
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
    return GUIDToFuncNameMap->lookup(std::stoull(Name.data()));
  }

  /// Returns the line offset to the start line of the subprogram.
  /// We assume that a single function will not exceed 65535 LOC.
  static unsigned getOffset(const DILocation *DIL);

  /// Returns a unique call site identifier for a given debug location of a call
  /// instruction. This is wrapper of two scenarios, the probe-based profile and
  /// regular profile, to hide implementation details from the sample loader and
  /// the context tracker.
  static LineLocation getCallSiteIdentifier(const DILocation *DIL);

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

  // The invalid sample count is used to represent samples collected for a
  // dangling probe.
  static constexpr uint64_t InvalidProbeCount = UINT64_MAX;

  static bool ProfileIsProbeBased;

  static bool ProfileIsCS;

  SampleContext &getContext() const { return Context; }

  void setContext(const SampleContext &FContext) { Context = FContext; }

  static SampleProfileFormat Format;

  /// Whether the profile uses MD5 to represent string.
  static bool UseMD5;

  /// Whether the profile contains any ".__uniq." suffix in a name.
  static bool HasUniqSuffix;

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

  /// CFG hash value for the function.
  uint64_t FunctionHash = 0;

  /// Calling context for function profile
  mutable SampleContext Context;

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
