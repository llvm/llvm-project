//===-- ARMHazardRecognizer.h - ARM Hazard Recognizers ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines hazard recognizers for scheduling ARM functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMHAZARDRECOGNIZER_H
#define LLVM_LIB_TARGET_ARM_ARMHAZARDRECOGNIZER_H

#include "ARMBaseInstrInfo.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/Support/DataTypes.h"
#include <initializer_list>

namespace llvm {

class DataLayout;
class MachineFunction;
class MachineInstr;
class ScheduleDAG;

// Hazards related to FP MLx instructions
class ARMHazardRecognizerFPMLx : public ScheduleHazardRecognizer {
  MachineInstr *LastMI = nullptr;
  unsigned FpMLxStalls = 0;

public:
  ARMHazardRecognizerFPMLx() { MaxLookAhead = 1; }

  HazardType getHazardType(SUnit *SU, int Stalls) override;
  void Reset() override;
  void EmitInstruction(SUnit *SU) override;
  void AdvanceCycle() override;
  void RecedeCycle() override;
};

// Hazards related to bank conflicts
class ARMBankConflictHazardRecognizer : public ScheduleHazardRecognizer {
  SmallVector<MachineInstr *, 8> Accesses;
  const MachineFunction &MF;
  const DataLayout &DL;
  int64_t DataMask;
  bool AssumeITCMBankConflict;

public:
  ARMBankConflictHazardRecognizer(const ScheduleDAG *DAG, int64_t DDM,
                                  bool ABC);
  HazardType getHazardType(SUnit *SU, int Stalls) override;
  void Reset() override;
  void EmitInstruction(SUnit *SU) override;
  void AdvanceCycle() override;
  void RecedeCycle() override;

private:
  inline HazardType CheckOffsets(unsigned O0, unsigned O1);
};

/**
  Hazards related to alignment of single-cycle fp instructions on cortex-m4f.

  https://developer.arm.com/documentation/ka006138/latest

  "A long sequence of T32 single-cycle floating-point instructions aligned on
  odd halfword boundaries will experience a performance drop. Specifically, one
  stall cycle is inserted for every three instructions executed."
*/
class ARMCortexM4AlignmentHazardRecognizer : public ScheduleHazardRecognizer {
  const MCSubtargetInfo &STI;
  const MachineBasicBlock *MBB;
  MachineDominatorTree MDT;
  MachineLoopInfo MLI;
  MachineFunction *MF;
  size_t Offset;
  bool Advanced;
  bool EmittingNoop;

public:
  ARMCortexM4AlignmentHazardRecognizer(const MCSubtargetInfo &STI);

  void Reset() override;

  void AdvanceCycle() override;
  void RecedeCycle() override;
  void EmitNoop() override;

  void EmitInstruction(SUnit *SU) override;
  void EmitInstruction(MachineInstr *MI) override;

  HazardType getHazardType(SUnit *SU, int Stalls = 0) override;
  HazardType getHazardType(MachineInstr *MI);
  HazardType getHazardTypeAssumingOffset(MachineInstr *MI, size_t Offset);

  unsigned PreEmitNoops(SUnit *SU) override;
  unsigned PreEmitNoops(MachineInstr *MI) override;

private:
  const MachineLoop *getLoopFor(MachineInstr *MI);
};

} // end namespace llvm

#endif
