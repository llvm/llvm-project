//===-- AMDGPUCacheCapacityHazardRecognizer.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a hazard recognizer for modeling L1 data cache capacity
// pressure during instruction scheduling on AMDGPU processors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUCACHECAPACITYHAZARDRECOGNIZER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUCACHECAPACITYHAZARDRECOGNIZER_H

#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/TargetParser/TargetParser.h"

#include <utility>
#include <vector>

namespace llvm {
namespace AMDGPU {

/// Per-thread effective L1 drain rate as a fraction of bytes per cycle.
/// Named fields prevent dividend/divisor argument swaps at call sites.
struct BytesPerCycle {
  unsigned Dividend;
  unsigned Divisor;
};

class CacheCapacityHazardRecognizer : public ScheduleHazardRecognizer {
  AMDGPU::IsaVersion IsaVer;
  /// All byte quantities below are in "scaled" units (real bytes *
  /// BytesScaleFactor) so that the per-cycle drain rate is an integer.
  unsigned ScaledCapacityBytes;
  unsigned ScaledUsedBytes = 0;
  unsigned ScaledDrainPerCycle;
  unsigned BytesScaleFactor;
  unsigned Latency;
  unsigned CurrentCycle = 0;
  struct IssueEntry {
    unsigned ScaledBytes;
    unsigned Cycle;
  };

  std::vector<IssueEntry> IssueHistory;
  /// Valid entries: [IssueTail, IssueHead), with wrap-around.
  unsigned IssueHead = 0;
  unsigned IssueTail = 0;

  /// Number of in-flight entries between Tail and Head in a circular buffer
  /// of capacity BufferSize.
  static unsigned circularDistance(unsigned Head, unsigned Tail,
                                   unsigned BufferSize) {
    return Head >= Tail ? Head - Tail : Head + BufferSize - Tail;
  }

  unsigned historySize() const {
    return circularDistance(IssueHead, IssueTail, IssueHistory.size());
  }

  /// Drain up to ScaledDrainPerCycle units from the head of the FIFO at
  /// CurrCycle, advancing Tail and updating UsedBytes / Remain in place. On
  /// exit, Remain holds the bytes left in the tail entry the loop stopped at
  /// (or 0 if the FIFO drained empty). The caller must write Remain back to
  /// History[Tail].ScaledBytes when it owns the storage and wants the partial
  /// drain to persist.
  static void drainOneCycle(unsigned ScaledDrainPerCycle, unsigned CurrCycle,
                            unsigned Latency, unsigned &Tail, unsigned Head,
                            ArrayRef<IssueEntry> History, unsigned &UsedBytes,
                            unsigned &Remain);

  void simulateAdvanceCycles(int Stalls, unsigned &RemBytesScaled,
                             unsigned &RemInstrs) const;

  bool canIssueBytes(unsigned Bytes, int Stalls) const;

  bool canWaitLessThanInstr(unsigned InstrCount, int Stalls) const;

  void issueBytes(unsigned Bytes);

  void waitLessThanInstr(unsigned InstrCount);

  HazardType getHazardType(MachineInstr *MI, int Stalls);

public:
  CacheCapacityHazardRecognizer(unsigned CapacityBytes, BytesPerCycle DrainRate,
                                unsigned Latency, AMDGPU::IsaVersion IsaVer);

  void Reset() override;

  void EmitInstruction(SUnit *) override;

  void EmitInstruction(MachineInstr *) override;

  HazardType getHazardType(SUnit *SU, int Stalls) override;

  void AdvanceCycle() override;

  void RecedeCycle() override;
};

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUCACHECAPACITYHAZARDRECOGNIZER_H
