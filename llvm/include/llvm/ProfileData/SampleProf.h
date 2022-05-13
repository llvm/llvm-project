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
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace llvm {

class DILocation;
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
  if (Name.empty() || !UseMD5)
    return Name;
  GUIDBuf = std::to_string(Function::getGUID(Name));
  return GUIDBuf;
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
  SecCSNameTable = 6,
  // marker for the first type of profile.
  SecFuncProfileFirst = 32,
  SecLBRProfile = SecFuncProfileFirst
};

static inline std::string getSecName(SecType Type) {
  switch ((int)Type) { // Avoid -Wcovered-switch-default
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
  case SecCSNameTable:
    return "CSNameTableSection";
  case SecLBRProfile:
    return "LBRProfileSection";
  default:
    return "UnknownSection";
  }
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
  /// SecFlagContext means this is context-sensitive flat profile for
  /// CSSPGO
  SecFlagFullContext = (1 << 1),
  /// SecFlagFSDiscriminator means this profile uses flow-sensitive
  /// discriminators.
  SecFlagFSDiscriminator = (1 << 2),
  /// SecFlagIsPreInlined means this profile contains ShouldBeInlined
  /// contexts thus this is CS preinliner computed.
  SecFlagIsPreInlined = (1 << 4),
};

enum class SecFuncMetadataFlags : uint32_t {
  SecFlagInvalid = 0,
  SecFlagIsProbeBased = (1 << 0),
  SecFlagHasAttribute = (1 << 1),
};

enum class SecFuncOffsetFlags : uint32_t {
  SecFlagInvalid = 0,
  // Store function offsets in an order of contexts. The order ensures that
  // callee contexts of a given context laid out next to it.
  SecFlagOrdered = (1 << 0),
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
  case SecFuncOffsetTable:
    IsFlagLegal = std::is_same<SecFuncOffsetFlags, SecFlagType>();
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

  /// Decrease the number of samples for this record by \p S. Return the amout
  /// of samples actually decreased.
  uint64_t removeSamples(uint64_t S) {
    if (S > NumSamples)
      S = NumSamples;
    NumSamples -= S;
    return S;
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

  /// Remove called function from the call target map. Return the target sample
  /// count of the called function.
  uint64_t removeCalledTarget(StringRef F) {
    uint64_t Count = 0;
    auto I = CallTargets.find(F);
    if (I != CallTargets.end()) {
      Count = I->second;
      CallTargets.erase(I);
    }
    return Count;
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

// Attribute of context associated with FunctionSamples
enum ContextAttributeMask {
  ContextNone = 0x0,
  ContextWasInlined = 0x1,      // Leaf of context was inlined in previous build
  ContextShouldBeInlined = 0x2, // Leaf of context should be inlined
  ContextDuplicatedIntoBase =
      0x4, // Leaf of context is duplicated into the base profile
};

// Represents a context frame with function name and line location
struct SampleContextFrame {
  StringRef FuncName;
  LineLocation Location;

  SampleContextFrame() : Location(0, 0) {}

  SampleContextFrame(StringRef FuncName, LineLocation Location)
      : FuncName(FuncName), Location(Location) {}

  bool operator==(const SampleContextFrame &That) const {
    return Location == That.Location && FuncName == That.FuncName;
  }

  bool operator!=(const SampleContextFrame &That) const {
    return !(*this == That);
  }

  std::string toString(bool OutputLineLocation) const {
    std::ostringstream OContextStr;
    OContextStr << FuncName.str();
    if (OutputLineLocation) {
      OContextStr << ":" << Location.LineOffset;
      if (Location.Discriminator)
        OContextStr << "." << Location.Discriminator;
    }
    return OContextStr.str();
  }
};

static inline hash_code hash_value(const SampleContextFrame &arg) {
  return hash_combine(arg.FuncName, arg.Location.LineOffset,
                      arg.Location.Discriminator);
}

using SampleContextFrameVector = SmallVector<SampleContextFrame, 1>;
using SampleContextFrames = ArrayRef<SampleContextFrame>;

struct SampleContextFrameHash {
  uint64_t operator()(const SampleContextFrameVector &S) const {
    return hash_combine_range(S.begin(), S.end());
  }
};

// Sample context for FunctionSamples. It consists of the calling context,
// the function name and context state. Internally sample context is represented
// using ArrayRef, which is also the input for constructing a `SampleContext`.
// It can accept and represent both full context string as well as context-less
// function name.
// For a CS profile, a full context vector can look like:
//    `main:3 _Z5funcAi:1 _Z8funcLeafi`
// For a base CS profile without calling context, the context vector should only
// contain the leaf frame name.
// For a non-CS profile, the context vector should be empty.
class SampleContext {
public:
  SampleContext() : State(UnknownContext), Attributes(ContextNone) {}

  SampleContext(StringRef Name)
      : Name(Name), State(UnknownContext), Attributes(ContextNone) {}

  SampleContext(SampleContextFrames Context,
                ContextStateMask CState = RawContext)
      : Attributes(ContextNone) {
    assert(!Context.empty() && "Context is empty");
    setContext(Context, CState);
  }

  // Give a context string, decode and populate internal states like
  // Function name, Calling context and context state. Example of input
  // `ContextStr`: `[main:3 @ _Z5funcAi:1 @ _Z8funcLeafi]`
  SampleContext(StringRef ContextStr,
                std::list<SampleContextFrameVector> &CSNameTable,
                ContextStateMask CState = RawContext)
      : Attributes(ContextNone) {
    assert(!ContextStr.empty());
    // Note that `[]` wrapped input indicates a full context string, otherwise
    // it's treated as context-less function name only.
    bool HasContext = ContextStr.startswith("[");
    if (!HasContext) {
      State = UnknownContext;
      Name = ContextStr;
    } else {
      CSNameTable.emplace_back();
      SampleContextFrameVector &Context = CSNameTable.back();
      createCtxVectorFromStr(ContextStr, Context);
      setContext(Context, CState);
    }
  }

  /// Create a context vector from a given context string and save it in
  /// `Context`.
  static void createCtxVectorFromStr(StringRef ContextStr,
                                     SampleContextFrameVector &Context) {
    // Remove encapsulating '[' and ']' if any
    ContextStr = ContextStr.substr(1, ContextStr.size() - 2);
    StringRef ContextRemain = ContextStr;
    StringRef ChildContext;
    StringRef CalleeName;
    while (!ContextRemain.empty()) {
      auto ContextSplit = ContextRemain.split(" @ ");
      ChildContext = ContextSplit.first;
      ContextRemain = ContextSplit.second;
      LineLocation CallSiteLoc(0, 0);
      decodeContextString(ChildContext, CalleeName, CallSiteLoc);
      Context.emplace_back(CalleeName, CallSiteLoc);
    }
  }

  // Promote context by removing top frames with the length of
  // `ContextFramesToRemove`. Note that with array representation of context,
  // the promotion is effectively a slice operation with first
  // `ContextFramesToRemove` elements removed from left.
  void promoteOnPath(uint32_t ContextFramesToRemove) {
    assert(ContextFramesToRemove <= FullContext.size() &&
           "Cannot remove more than the whole context");
    FullContext = FullContext.drop_front(ContextFramesToRemove);
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

  operator SampleContextFrames() const { return FullContext; }
  bool hasAttribute(ContextAttributeMask A) { return Attributes & (uint32_t)A; }
  void setAttribute(ContextAttributeMask A) { Attributes |= (uint32_t)A; }
  uint32_t getAllAttributes() { return Attributes; }
  void setAllAttributes(uint32_t A) { Attributes = A; }
  bool hasState(ContextStateMask S) { return State & (uint32_t)S; }
  void setState(ContextStateMask S) { State |= (uint32_t)S; }
  void clearState(ContextStateMask S) { State &= (uint32_t)~S; }
  bool hasContext() const { return State != UnknownContext; }
  bool isBaseContext() const { return FullContext.size() == 1; }
  StringRef getName() const { return Name; }
  SampleContextFrames getContextFrames() const { return FullContext; }

  static std::string getContextString(SampleContextFrames Context,
                                      bool IncludeLeafLineLocation = false) {
    std::ostringstream OContextStr;
    for (uint32_t I = 0; I < Context.size(); I++) {
      if (OContextStr.str().size()) {
        OContextStr << " @ ";
      }
      OContextStr << Context[I].toString(I != Context.size() - 1 ||
                                         IncludeLeafLineLocation);
    }
    return OContextStr.str();
  }

  std::string toString() const {
    if (!hasContext())
      return Name.str();
    return getContextString(FullContext, false);
  }

  uint64_t getHashCode() const {
    return hasContext() ? hash_value(getContextFrames())
                        : hash_value(getName());
  }

  /// Set the name of the function and clear the current context.
  void setName(StringRef FunctionName) {
    Name = FunctionName;
    FullContext = SampleContextFrames();
    State = UnknownContext;
  }

  void setContext(SampleContextFrames Context,
                  ContextStateMask CState = RawContext) {
    assert(CState != UnknownContext);
    FullContext = Context;
    Name = Context.back().FuncName;
    State = CState;
  }

  bool operator==(const SampleContext &That) const {
    return State == That.State && Name == That.Name &&
           FullContext == That.FullContext;
  }

  bool operator!=(const SampleContext &That) const { return !(*this == That); }

  bool operator<(const SampleContext &That) const {
    if (State != That.State)
      return State < That.State;

    if (!hasContext()) {
      return (Name.compare(That.Name)) == -1;
    }

    uint64_t I = 0;
    while (I < std::min(FullContext.size(), That.FullContext.size())) {
      auto &Context1 = FullContext[I];
      auto &Context2 = That.FullContext[I];
      auto V = Context1.FuncName.compare(Context2.FuncName);
      if (V)
        return V == -1;
      if (Context1.Location != Context2.Location)
        return Context1.Location < Context2.Location;
      I++;
    }

    return FullContext.size() < That.FullContext.size();
  }

  struct Hash {
    uint64_t operator()(const SampleContext &Context) const {
      return Context.getHashCode();
    }
  };

  bool IsPrefixOf(const SampleContext &That) const {
    auto ThisContext = FullContext;
    auto ThatContext = That.FullContext;
    if (ThatContext.size() < ThisContext.size())
      return false;
    ThatContext = ThatContext.take_front(ThisContext.size());
    // Compare Leaf frame first
    if (ThisContext.back().FuncName != ThatContext.back().FuncName)
      return false;
    // Compare leading context
    return ThisContext.drop_back() == ThatContext.drop_back();
  }

private:
  /// Mangled name of the function.
  StringRef Name;
  // Full context including calling context and leaf function name
  SampleContextFrames FullContext;
  // State of the associated sample profile
  uint32_t State;
  // Attribute of the associated sample profile
  uint32_t Attributes;
};

static inline hash_code hash_value(const SampleContext &arg) {
  return arg.hasContext() ? hash_value(arg.getContextFrames())
                          : hash_value(arg.getName());
}

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

  void removeTotalSamples(uint64_t Num) {
    if (TotalSamples < Num)
      TotalSamples = 0;
    else
      TotalSamples -= Num;
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

  // Remove a call target and decrease the body sample correspondingly. Return
  // the number of body samples actually decreased.
  uint64_t removeCalledTargetAndBodySample(uint32_t LineOffset,
                                           uint32_t Discriminator,
                                           StringRef FName) {
    uint64_t Count = 0;
    auto I = BodySamples.find(LineLocation(LineOffset, Discriminator));
    if (I != BodySamples.end()) {
      Count = I->second.removeCalledTarget(FName);
      Count = I->second.removeSamples(Count);
      if (!I->second.getSamples())
        BodySamples.erase(I);
    }
    return Count;
  }

  sampleprof_error addBodySamplesForProbe(uint32_t Index, uint64_t Num,
                                          uint64_t Weight = 1) {
    SampleRecord S;
    S.addSamples(Num, Weight);
    return BodySamples[LineLocation(Index, 0)].merge(S, Weight);
  }

  // Accumulate all body samples to set total samples.
  void updateTotalSamples() {
    setTotalSamples(0);
    for (const auto &I : BodySamples)
      addTotalSamples(I.second.getSamples());

    for (auto &I : CallsiteSamples) {
      for (auto &CS : I.second) {
        CS.second.updateTotalSamples();
        addTotalSamples(CS.second.getTotalSamples());
      }
    }
  }

  // Set current context and all callee contexts to be synthetic.
  void SetContextSynthetic() {
    Context.setState(SyntheticContext);
    for (auto &I : CallsiteSamples) {
      for (auto &CS : I.second) {
        CS.second.SetContextSynthetic();
      }
    }
  }

  /// Return the number of samples collected at the given location.
  /// Each location is specified by \p LineOffset and \p Discriminator.
  /// If the location is not found in profile, return error.
  ErrorOr<uint64_t> findSamplesAt(uint32_t LineOffset,
                                  uint32_t Discriminator) const {
    const auto &ret = BodySamples.find(LineLocation(LineOffset, Discriminator));
    if (ret == BodySamples.end())
      return std::error_code();
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
    if (!GUIDToFuncNameMap)
      GUIDToFuncNameMap = Other.GUIDToFuncNameMap;
    if (Context.getName().empty())
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
      S.insert(getGUID(getName()));
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
  void setName(StringRef FunctionName) { Context.setName(FunctionName); }

  /// Return the function name.
  StringRef getName() const { return Context.getName(); }

  /// Return the original function name.
  StringRef getFuncName() const { return getFuncName(getName()); }

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

    assert(GUIDToFuncNameMap && "GUIDToFuncNameMap needs to be populated first");
    return GUIDToFuncNameMap->lookup(std::stoull(Name.data()));
  }

  /// Returns the line offset to the start line of the subprogram.
  /// We assume that a single function will not exceed 65535 LOC.
  static unsigned getOffset(const DILocation *DIL);

  /// Returns a unique call site identifier for a given debug location of a call
  /// instruction. This is wrapper of two scenarios, the probe-based profile and
  /// regular profile, to hide implementation details from the sample loader and
  /// the context tracker.
  static LineLocation getCallSiteIdentifier(const DILocation *DIL,
                                            bool ProfileIsFS = false);

  /// Returns a unique hash code for a combination of a callsite location and
  /// the callee function name.
  static uint64_t getCallSiteHash(StringRef CalleeName,
                                  const LineLocation &Callsite);

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

  static bool ProfileIsProbeBased;

  static bool ProfileIsCS;

  static bool ProfileIsPreInlined;

  SampleContext &getContext() const { return Context; }

  void setContext(const SampleContext &FContext) { Context = FContext; }

  /// Whether the profile uses MD5 to represent string.
  static bool UseMD5;

  /// Whether the profile contains any ".__uniq." suffix in a name.
  static bool HasUniqSuffix;

  /// If this profile uses flow sensitive discriminators.
  static bool ProfileIsFS;

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

using SampleProfileMap =
    std::unordered_map<SampleContext, FunctionSamples, SampleContext::Hash>;

using NameFunctionSamples = std::pair<SampleContext, const FunctionSamples *>;

void sortFuncProfiles(const SampleProfileMap &ProfileMap,
                      std::vector<NameFunctionSamples> &SortedProfiles);

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

/// SampleContextTrimmer impelements helper functions to trim, merge cold
/// context profiles. It also supports context profile canonicalization to make
/// sure ProfileMap's key is consistent with FunctionSample's name/context.
class SampleContextTrimmer {
public:
  SampleContextTrimmer(SampleProfileMap &Profiles) : ProfileMap(Profiles){};
  // Trim and merge cold context profile when requested. TrimBaseProfileOnly
  // should only be effective when TrimColdContext is true. On top of
  // TrimColdContext, TrimBaseProfileOnly can be used to specify to trim all
  // cold profiles or only cold base profiles. Trimming base profiles only is
  // mainly to honor the preinliner decsion. Note that when MergeColdContext is
  // true, preinliner decsion is not honored anyway so TrimBaseProfileOnly will
  // be ignored.
  void trimAndMergeColdContextProfiles(uint64_t ColdCountThreshold,
                                       bool TrimColdContext,
                                       bool MergeColdContext,
                                       uint32_t ColdContextFrameLength,
                                       bool TrimBaseProfileOnly);
  // Canonicalize context profile name and attributes.
  void canonicalizeContextProfiles();

private:
  SampleProfileMap &ProfileMap;
};

// CSProfileConverter converts a full context-sensitive flat sample profile into
// a nested context-sensitive sample profile.
class CSProfileConverter {
public:
  CSProfileConverter(SampleProfileMap &Profiles);
  void convertProfiles();
  struct FrameNode {
    FrameNode(StringRef FName = StringRef(),
              FunctionSamples *FSamples = nullptr,
              LineLocation CallLoc = {0, 0})
        : FuncName(FName), FuncSamples(FSamples), CallSiteLoc(CallLoc){};

    // Map line+discriminator location to child frame
    std::map<uint64_t, FrameNode> AllChildFrames;
    // Function name for current frame
    StringRef FuncName;
    // Function Samples for current frame
    FunctionSamples *FuncSamples;
    // Callsite location in parent context
    LineLocation CallSiteLoc;

    FrameNode *getOrCreateChildFrame(const LineLocation &CallSite,
                                     StringRef CalleeName);
  };

private:
  // Nest all children profiles into the profile of Node.
  void convertProfiles(FrameNode &Node);
  FrameNode *getOrCreateContextPath(const SampleContext &Context);

  SampleProfileMap &ProfileMap;
  FrameNode RootFrame;
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

using namespace sampleprof;
// Provide DenseMapInfo for SampleContext.
template <> struct DenseMapInfo<SampleContext> {
  static inline SampleContext getEmptyKey() { return SampleContext(); }

  static inline SampleContext getTombstoneKey() { return SampleContext("@"); }

  static unsigned getHashValue(const SampleContext &Val) {
    return Val.getHashCode();
  }

  static bool isEqual(const SampleContext &LHS, const SampleContext &RHS) {
    return LHS == RHS;
  }
};
} // end namespace llvm

#endif // LLVM_PROFILEDATA_SAMPLEPROF_H
