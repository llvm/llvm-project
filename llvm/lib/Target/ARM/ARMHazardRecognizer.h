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

#include "llvm/CodeGen/ScheduleHazardRecognizer.h"

namespace llvm {

// Hazards related to FP MLx instructions
class ARMHazardRecognizerFPMLx : public ScheduleHazardRecognizer {
  MachineInstr *LastMI = nullptr;
  unsigned FpMLxStalls = 0;

public:
  ARMHazardRecognizerFPMLx() : ScheduleHazardRecognizer() { MaxLookAhead = 1; }

  HazardType getHazardType(SUnit *SU, int Stalls) override;
  void Reset() override;
  void EmitInstruction(SUnit *SU) override;
  void AdvanceCycle() override;
  void RecedeCycle() override;
};

} // end namespace llvm

#endif
