//===- XRayRecord.h - XRay Trace Record -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file replicates the record definition for XRay log entries. This should
// follow the evolution of the log record versions supported in the compiler-rt
// xray project.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_XRAY_XRAY_RECORD_H
#define LLVM_XRAY_XRAY_RECORD_H

#include <cstdint>

namespace llvm {
namespace xray {

/// XRay traces all have a header providing some top-matter information useful
/// to help tools determine how to interpret the information available in the
/// trace.
struct XRayFileHeader {
  /// Version of the XRay implementation that produced this file.
  uint16_t Version = 0;

  /// A numeric identifier for the type of file this is. Best used in
  /// combination with Version.
  uint16_t Type = 0;

  /// Whether the CPU that produced the timestamp counters (TSC) move at a
  /// constant rate.
  bool ConstantTSC;

  /// Whether the CPU that produced the timestamp counters (TSC) do not stop.
  bool NonstopTSC;

  /// The number of cycles per second for the CPU that produced the timestamp
  /// counter (TSC) values. Useful for estimating the amount of time that
  /// elapsed between two TSCs on some platforms.
  uint64_t CycleFrequency = 0;

  // This is different depending on the type of xray record. The naive format
  // stores a Wallclock timespec. FDR logging stores the size of a thread
  // buffer.
  char FreeFormData[16];
};

/// Determines the supported types of records that could be seen in XRay traces.
/// This may or may not correspond to actual record types in the raw trace (as
/// the loader implementation may synthesize this information in the process of
/// of loading).
enum class RecordTypes { ENTER, EXIT };

struct XRayRecord {
  /// The type of record.
  uint16_t RecordType;

  /// The CPU where the thread is running. We assume number of CPUs <= 65536.
  uint16_t CPU;

  /// Identifies the type of record.
  RecordTypes Type;

  /// The function ID for the record.
  int32_t FuncId;

  /// Get the full 8 bytes of the TSC when we get the log record.
  uint64_t TSC;

  /// The thread ID for the currently running thread.
  uint32_t TId;
};

} // namespace xray
} // namespace llvm

#endif // LLVM_XRAY_XRAY_RECORD_H
