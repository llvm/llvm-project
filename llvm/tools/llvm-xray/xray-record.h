//===- xray-record.h - XRay Trace Record ----------------------------------===//
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
#ifndef LLVM_TOOLS_LLVM_XRAY_XRAY_RECORD_H
#define LLVM_TOOLS_LLVM_XRAY_XRAY_RECORD_H

#include <cstdint>

namespace llvm {
namespace xray {

struct XRayFileHeader {
  uint16_t Version = 0;
  uint16_t Type = 0;
  bool ConstantTSC;
  bool NonstopTSC;
  uint64_t CycleFrequency = 0;
};

enum class RecordTypes { ENTER, EXIT };

struct XRayRecord {
  uint16_t RecordType;

  // The CPU where the thread is running. We assume number of CPUs <= 256.
  uint8_t CPU;

  // Identifies the type of record.
  RecordTypes Type;

  // The function ID for the record.
  int32_t FuncId;

  // Get the full 8 bytes of the TSC when we get the log record.
  uint64_t TSC;

  // The thread ID for the currently running thread.
  uint32_t TId;
};

} // namespace xray
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_XRAY_XRAY_RECORD_H
