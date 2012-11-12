//===-- SPUHazardRecognizers.h - Cell SPU Hazard Recognizer -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines hazard recognizers for scheduling on the Cell SPU
// processor.
//
//===----------------------------------------------------------------------===//

#ifndef SPUHAZRECS_H
#define SPUHAZRECS_H

#include "llvm/CodeGen/ScheduleHazardRecognizer.h"

namespace llvm {

class TargetInstrInfo;

/// SPUHazardRecognizer
class SPUHazardRecognizer : public ScheduleHazardRecognizer
{
public:
  SPUHazardRecognizer(const TargetInstrInfo &/*TII*/) {}
  virtual HazardType getHazardType(SUnit *SU, int Stalls);
  virtual void EmitInstruction(SUnit *SU);
  virtual void AdvanceCycle();
  virtual void EmitNoop();
};

} // end namespace llvm

#endif
