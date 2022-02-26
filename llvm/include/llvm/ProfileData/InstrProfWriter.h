//===- InstrProfWriter.h - Instrumented profiling writer --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing profiling data for instrumentation
// based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_INSTRPROFWRITER_H
#define LLVM_PROFILEDATA_INSTRPROFWRITER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>

namespace llvm {

/// Writer for instrumentation based profile data.
class InstrProfRecordWriterTrait;
class ProfOStream;
class MemoryBuffer;
class raw_fd_ostream;

class InstrProfWriter {
public:
  using ProfilingData = SmallDenseMap<uint64_t, InstrProfRecord>;

private:
  bool Sparse;
  StringMap<ProfilingData> FunctionData;

  // A map to hold memprof data per function. The lower 64 bits obtained from
  // the md5 hash of the function name is used to index into the map.
  memprof::FunctionMemProfMap MemProfData;

  // An enum describing the attributes of the profile.
  InstrProfKind ProfileKind = InstrProfKind::Unknown;
  // Use raw pointer here for the incomplete type object.
  InstrProfRecordWriterTrait *InfoObj;

public:
  InstrProfWriter(bool Sparse = false);
  ~InstrProfWriter();

  StringMap<ProfilingData> &getProfileData() { return FunctionData; }

  /// Add function counts for the given function. If there are already counts
  /// for this function and the hash and number of counts match, each counter is
  /// summed. Optionally scale counts by \p Weight.
  void addRecord(NamedInstrProfRecord &&I, uint64_t Weight,
                 function_ref<void(Error)> Warn);
  void addRecord(NamedInstrProfRecord &&I, function_ref<void(Error)> Warn) {
    addRecord(std::move(I), 1, Warn);
  }

  void addRecord(const ::llvm::memprof::MemProfRecord &MR,
                 function_ref<void(Error)> Warn);

  /// Merge existing function counts from the given writer.
  void mergeRecordsFromWriter(InstrProfWriter &&IPW,
                              function_ref<void(Error)> Warn);

  /// Write the profile to \c OS
  Error write(raw_fd_ostream &OS);

  /// Write the profile in text format to \c OS
  Error writeText(raw_fd_ostream &OS);

  Error validateRecord(const InstrProfRecord &Func);

  /// Write \c Record in text format to \c OS
  static void writeRecordInText(StringRef Name, uint64_t Hash,
                                const InstrProfRecord &Counters,
                                InstrProfSymtab &Symtab, raw_fd_ostream &OS);

  /// Write the profile, returning the raw data. For testing.
  std::unique_ptr<MemoryBuffer> writeBuffer();

  /// Update the attributes of the current profile from the attributes
  /// specified. An error is returned if IR and FE profiles are mixed.
  Error mergeProfileKind(const InstrProfKind Other) {
    // If the kind is unset, this is the first profile we are merging so just
    // set it to the given type.
    if (ProfileKind == InstrProfKind::Unknown) {
      ProfileKind = Other;
      return Error::success();
    }

    // Returns true if merging is should fail assuming A and B are incompatible.
    auto testIncompatible = [&](InstrProfKind A, InstrProfKind B) {
      return (static_cast<bool>(ProfileKind & A) &&
              static_cast<bool>(Other & B)) ||
             (static_cast<bool>(ProfileKind & B) &&
              static_cast<bool>(Other & A));
    };

    // Check if the profiles are in-compatible. Clang frontend profiles can't be
    // merged with other profile types.
    if (static_cast<bool>(
            (ProfileKind & InstrProfKind::FrontendInstrumentation) ^
            (Other & InstrProfKind::FrontendInstrumentation))) {
      return make_error<InstrProfError>(instrprof_error::unsupported_version);
    }
    if (testIncompatible(InstrProfKind::FunctionEntryOnly,
                         InstrProfKind::FunctionEntryInstrumentation)) {
      return make_error<InstrProfError>(
          instrprof_error::unsupported_version,
          "cannot merge FunctionEntryOnly profiles and BB profiles together");
    }

    // Now we update the profile type with the bits that are set.
    ProfileKind |= Other;
    return Error::success();
  }

  InstrProfKind getProfileKind() const { return ProfileKind; }

  // Internal interface for testing purpose only.
  void setValueProfDataEndianness(support::endianness Endianness);
  void setOutputSparse(bool Sparse);
  // Compute the overlap b/w this object and Other. Program level result is
  // stored in Overlap and function level result is stored in FuncLevelOverlap.
  void overlapRecord(NamedInstrProfRecord &&Other, OverlapStats &Overlap,
                     OverlapStats &FuncLevelOverlap,
                     const OverlapFuncFilters &FuncFilter);

private:
  void addRecord(StringRef Name, uint64_t Hash, InstrProfRecord &&I,
                 uint64_t Weight, function_ref<void(Error)> Warn);
  bool shouldEncodeData(const ProfilingData &PD);

  Error writeImpl(ProfOStream &OS);
};

} // end namespace llvm

#endif // LLVM_PROFILEDATA_INSTRPROFWRITER_H
