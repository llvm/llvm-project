//===--- HexagonHazardRecognizer.h - Hexagon Post RA Hazard Recognizer ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file defines the hazard recognizer for scheduling on Hexagon.
//===----------------------------------------------------------------------===//

#ifndef HEXAGONPROFITRECOGNIZER_H
#define HEXAGONPROFITRECOGNIZER_H

#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"

namespace llvm {

class HexagonHazardRecognizer : public ScheduleHazardRecognizer {
  DFAPacketizer *Resources;
  const HexagonInstrInfo *TII;
  unsigned PacketNum;
  // If the packet contains a potential dot cur instruction. This is
  // used for the scheduling priority function.
  SUnit *UsesDotCur;
  // The packet number when a dor cur is emitted. If its use is not generated
  // in the same packet, then try to wait another cycle before emitting.
  int DotCurPNum;
  // The set of registers defined by instructions in the current packet.
  SmallSet<unsigned, 8> RegDefs;

public:
  HexagonHazardRecognizer(const InstrItineraryData *II,
                          const HexagonInstrInfo *HII,
                          const HexagonSubtarget &ST)
    : Resources(ST.createDFAPacketizer(II)), TII(HII), PacketNum(0),
    UsesDotCur(nullptr), DotCurPNum(-1) { }

  ~HexagonHazardRecognizer() {
    if (Resources)
      delete Resources;
  }

  /// This callback is invoked when a new block of instructions is about to be
  /// scheduled. The hazard state is set to an initialized state.
  virtual void Reset() override;

  /// Return the hazard type of emitting this node.  There are three
  /// possible results.  Either:
  ///  * NoHazard: it is legal to issue this instruction on this cycle.
  ///  * Hazard: issuing this instruction would stall the machine.  If some
  ///     other instruction is available, issue it first.
  virtual HazardType getHazardType(SUnit *SU, int stalls) override;

  /// This callback is invoked when an instruction is emitted to be scheduled,
  /// to advance the hazard state.
  virtual void EmitInstruction(SUnit *) override;

  /// This callback may be invoked if getHazardType returns NoHazard. If, even
  /// though there is no hazard, it would be better to schedule another
  /// available instruction, this callback should return true.
  virtual bool ShouldPreferAnother(SUnit *) override;

  /// This callback is invoked whenever the next top-down instruction to be
  /// scheduled cannot issue in the current cycle, either because of latency
  /// or resource conflicts.  This should increment the internal state of the
  /// hazard recognizer so that previously "Hazard" instructions will now not
  /// be hazards.
  virtual void AdvanceCycle() override;
};

} // end namespace llvm

#endif // HEXAGONPROFITRECOGNIZER_H
