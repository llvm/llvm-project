//=-- InstrProfWriter.h - Instrumented profiling writer -----------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// Writer for instrumentation based profile data.
class InstrProfWriter {
public:
  typedef SmallDenseMap<uint64_t, InstrProfRecord, 1> ProfilingData;

private:
  InstrProfStringTable StringTable;
  StringMap<ProfilingData> FunctionData;
  uint64_t MaxFunctionCount;
public:
  InstrProfWriter() : MaxFunctionCount(0) {}

  /// Update string entries in profile data with references to StringTable.
  void updateStringTableReferences(InstrProfRecord &I);
  /// Add function counts for the given function. If there are already counts
  /// for this function and the hash and number of counts match, each counter is
  /// summed.
  std::error_code addRecord(InstrProfRecord &&I);
  /// Write the profile to \c OS
  void write(raw_fd_ostream &OS);
  /// Write the profile in text format to \c OS
  void writeText(raw_fd_ostream &OS);
  /// Write \c Record in text format to \c OS
  static void writeRecordInText(const InstrProfRecord &Record,
                                raw_fd_ostream &OS);
  /// Write the profile, returning the raw data. For testing.
  std::unique_ptr<MemoryBuffer> writeBuffer();

  // Internal interface for testing purpose only.
  void setValueProfDataEndianness(support::endianness Endianness);

private:
  std::pair<uint64_t, uint64_t> writeImpl(raw_ostream &OS);
};

} // end namespace llvm

#endif
