//=-- SampleProf.h - Sampling profiling format support --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains common definitions used in the reading and writing of
// sample profile data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_SAMPLEPROF_H_
#define LLVM_PROFILEDATA_SAMPLEPROF_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <system_error>

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
  counter_overflow
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
}

namespace llvm {

namespace sampleprof {

static inline uint64_t SPMagic() {
  return uint64_t('S') << (64 - 8) | uint64_t('P') << (64 - 16) |
         uint64_t('R') << (64 - 24) | uint64_t('O') << (64 - 32) |
         uint64_t('F') << (64 - 40) | uint64_t('4') << (64 - 48) |
         uint64_t('2') << (64 - 56) | uint64_t(0xff);
}

static inline uint64_t SPVersion() { return 102; }

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

/// Represents the relative location of a callsite.
///
/// Callsite locations are specified by the line offset from the
/// beginning of the function (marked by the line where the function
/// head is), the discriminator value within that line, and the callee
/// function name.
struct CallsiteLocation : public LineLocation {
  CallsiteLocation(uint32_t L, uint32_t D, StringRef N)
      : LineLocation(L, D), CalleeName(N) {}
  void print(raw_ostream &OS) const;
  void dump() const;

  StringRef CalleeName;
};

raw_ostream &operator<<(raw_ostream &OS, const CallsiteLocation &Loc);

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
  typedef StringMap<uint64_t> CallTargetMap;

  SampleRecord() : NumSamples(0), CallTargets() {}

  /// Increment the number of samples for this record by \p S.
  /// Optionally scale sample count \p S by \p Weight.
  ///
  /// Sample counts accumulate using saturating arithmetic, to avoid wrapping
  /// around unsigned integers.
  sampleprof_error addSamples(uint64_t S, uint64_t Weight = 1) {
    bool Overflowed;
    if (Weight > 1) {
      S = SaturatingMultiply(S, Weight, &Overflowed);
      if (Overflowed)
        return sampleprof_error::counter_overflow;
    }
    NumSamples = SaturatingAdd(NumSamples, S, &Overflowed);
    if (Overflowed)
      return sampleprof_error::counter_overflow;

    return sampleprof_error::success;
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
    if (Weight > 1) {
      S = SaturatingMultiply(S, Weight, &Overflowed);
      if (Overflowed)
        return sampleprof_error::counter_overflow;
    }
    TargetSamples = SaturatingAdd(TargetSamples, S, &Overflowed);
    if (Overflowed)
      return sampleprof_error::counter_overflow;

    return sampleprof_error::success;
  }

  /// Return true if this sample record contains function calls.
  bool hasCalls() const { return CallTargets.size() > 0; }

  uint64_t getSamples() const { return NumSamples; }
  const CallTargetMap &getCallTargets() const { return CallTargets; }

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
  uint64_t NumSamples;
  CallTargetMap CallTargets;
};

raw_ostream &operator<<(raw_ostream &OS, const SampleRecord &Sample);

typedef std::map<LineLocation, SampleRecord> BodySampleMap;
class FunctionSamples;
typedef std::map<CallsiteLocation, FunctionSamples> CallsiteSampleMap;

/// Representation of the samples collected for a function.
///
/// This data structure contains all the collected samples for the body
/// of a function. Each sample corresponds to a LineLocation instance
/// within the body of the function.
class FunctionSamples {
public:
  FunctionSamples() : TotalSamples(0), TotalHeadSamples(0) {}
  void print(raw_ostream &OS = dbgs(), unsigned Indent = 0) const;
  void dump() const;
  sampleprof_error addTotalSamples(uint64_t Num, uint64_t Weight = 1) {
    bool Overflowed;
    if (Weight > 1) {
      Num = SaturatingMultiply(Num, Weight, &Overflowed);
      if (Overflowed)
        return sampleprof_error::counter_overflow;
    }
    TotalSamples = SaturatingAdd(TotalSamples, Num, &Overflowed);
    if (Overflowed)
      return sampleprof_error::counter_overflow;

    return sampleprof_error::success;
  }
  sampleprof_error addHeadSamples(uint64_t Num, uint64_t Weight = 1) {
    bool Overflowed;
    if (Weight > 1) {
      Num = SaturatingMultiply(Num, Weight, &Overflowed);
      if (Overflowed)
        return sampleprof_error::counter_overflow;
    }
    TotalHeadSamples = SaturatingAdd(TotalHeadSamples, Num, &Overflowed);
    if (Overflowed)
      return sampleprof_error::counter_overflow;

    return sampleprof_error::success;
  }
  sampleprof_error addBodySamples(uint32_t LineOffset, uint32_t Discriminator,
                                  uint64_t Num, uint64_t Weight = 1) {
    return BodySamples[LineLocation(LineOffset, Discriminator)].addSamples(
        Num, Weight);
  }
  sampleprof_error addCalledTargetSamples(uint32_t LineOffset,
                                          uint32_t Discriminator,
                                          std::string FName, uint64_t Num,
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

  /// Return the function samples at the given callsite location.
  FunctionSamples &functionSamplesAt(const CallsiteLocation &Loc) {
    return CallsiteSamples[Loc];
  }

  /// Return a pointer to function samples at the given callsite location.
  const FunctionSamples *
  findFunctionSamplesAt(const CallsiteLocation &Loc) const {
    auto iter = CallsiteSamples.find(Loc);
    if (iter == CallsiteSamples.end()) {
      return nullptr;
    } else {
      return &iter->second;
    }
  }

  bool empty() const { return TotalSamples == 0; }

  /// Return the total number of samples collected inside the function.
  uint64_t getTotalSamples() const { return TotalSamples; }

  /// Return the total number of samples collected at the head of the
  /// function.
  uint64_t getHeadSamples() const { return TotalHeadSamples; }

  /// Return all the samples collected in the body of the function.
  const BodySampleMap &getBodySamples() const { return BodySamples; }

  /// Return all the callsite samples collected in the body of the function.
  const CallsiteSampleMap &getCallsiteSamples() const {
    return CallsiteSamples;
  }

  /// Merge the samples in \p Other into this one.
  /// Optionally scale samples by \p Weight.
  sampleprof_error merge(const FunctionSamples &Other, uint64_t Weight = 1) {
    sampleprof_error Result = sampleprof_error::success;
    MergeResult(Result, addTotalSamples(Other.getTotalSamples(), Weight));
    MergeResult(Result, addHeadSamples(Other.getHeadSamples(), Weight));
    for (const auto &I : Other.getBodySamples()) {
      const LineLocation &Loc = I.first;
      const SampleRecord &Rec = I.second;
      MergeResult(Result, BodySamples[Loc].merge(Rec, Weight));
    }
    for (const auto &I : Other.getCallsiteSamples()) {
      const CallsiteLocation &Loc = I.first;
      const FunctionSamples &Rec = I.second;
      MergeResult(Result, functionSamplesAt(Loc).merge(Rec, Weight));
    }
    return Result;
  }

private:
  /// Total number of samples collected inside this function.
  ///
  /// Samples are cumulative, they include all the samples collected
  /// inside this function and all its inlined callees.
  uint64_t TotalSamples;

  /// Total number of samples collected at the head of the function.
  /// This is an approximation of the number of calls made to this function
  /// at runtime.
  uint64_t TotalHeadSamples;

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
  typedef std::pair<const LocationT, SampleT> SamplesWithLoc;
  typedef SmallVector<const SamplesWithLoc *, 20> SamplesWithLocList;

  SampleSorter(const std::map<LocationT, SampleT> &Samples) {
    for (const auto &I : Samples)
      V.push_back(&I);
    std::stable_sort(V.begin(), V.end(),
                     [](const SamplesWithLoc *A, const SamplesWithLoc *B) {
                       return A->first < B->first;
                     });
  }
  const SamplesWithLocList &get() const { return V; }

private:
  SamplesWithLocList V;
};

} // end namespace sampleprof

} // end namespace llvm

#endif // LLVM_PROFILEDATA_SAMPLEPROF_H_
