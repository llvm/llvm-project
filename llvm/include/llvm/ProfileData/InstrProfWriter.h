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
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstdint>
#include <memory>

namespace llvm {

/// Writer for instrumentation based profile data.
class InstrProfRecordWriterTrait;
class ProfOStream;
class raw_fd_ostream;

class InstrProfWriter {
public:
  using ProfilingData = SmallDenseMap<uint64_t, InstrProfRecord>;
  // PF_IRLevelWithCS is the profile from context sensitive IR instrumentation.
  enum ProfKind { PF_Unknown = 0, PF_FE, PF_IRLevel, PF_IRLevelWithCS };

private:
  bool Sparse;
  StringMap<ProfilingData> FunctionData;
  ProfKind ProfileKind = PF_Unknown;
  bool InstrEntryBBEnabled;
  // Use raw pointer here for the incomplete type object.
  InstrProfRecordWriterTrait *InfoObj;

public:
  InstrProfWriter(bool Sparse = false, bool InstrEntryBBEnabled = false);
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

  /// Merge existing function counts from the given writer.
  void mergeRecordsFromWriter(InstrProfWriter &&IPW,
                              function_ref<void(Error)> Warn);

  /// Write the profile to \c OS
  void write(raw_fd_ostream &OS);

  /// Write the profile in text format to \c OS
  Error writeText(raw_fd_ostream &OS);

  /// Write \c Record in text format to \c OS
  static void writeRecordInText(StringRef Name, uint64_t Hash,
                                const InstrProfRecord &Counters,
                                InstrProfSymtab &Symtab, raw_fd_ostream &OS);

  /// Write the profile, returning the raw data. For testing.
  std::unique_ptr<MemoryBuffer> writeBuffer();

  /// Set the ProfileKind. Report error if mixing FE and IR level profiles.
  /// \c WithCS indicates if this is for contenxt sensitive instrumentation.
  Error setIsIRLevelProfile(bool IsIRLevel, bool WithCS) {
    if (ProfileKind == PF_Unknown) {
      if (IsIRLevel)
        ProfileKind = WithCS ? PF_IRLevelWithCS : PF_IRLevel;
      else
        ProfileKind = PF_FE;
      return Error::success();
    }

    if (((ProfileKind != PF_FE) && !IsIRLevel) ||
        ((ProfileKind == PF_FE) && IsIRLevel))
      return make_error<InstrProfError>(instrprof_error::unsupported_version);

    // When merging a context-sensitive profile (WithCS == true) with an IRLevel
    // profile, set the kind to PF_IRLevelWithCS.
    if (ProfileKind == PF_IRLevel && WithCS)
      ProfileKind = PF_IRLevelWithCS;

    return Error::success();
  }

  void setInstrEntryBBEnabled(bool Enabled) { InstrEntryBBEnabled = Enabled; }
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
  void writeImpl(ProfOStream &OS);
};

} // end namespace llvm

#endif // LLVM_PROFILEDATA_INSTRPROFWRITER_H
