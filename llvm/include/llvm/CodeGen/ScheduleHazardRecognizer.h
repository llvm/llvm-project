//=- llvm/CodeGen/ScheduleHazardRecognizer.h - Scheduling Support -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ScheduleHazardRecognizer class, which implements
// hazard-avoidance heuristics for scheduling.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDULEHAZARDRECOGNIZER_H
#define LLVM_CODEGEN_SCHEDULEHAZARDRECOGNIZER_H

namespace llvm {

class SUnit;

/// HazardRecognizer - This determines whether or not an instruction can be
/// issued this cycle, and whether or not a noop needs to be inserted to handle
/// the hazard.
class ScheduleHazardRecognizer {
public:
  virtual ~ScheduleHazardRecognizer();

  enum HazardType {
    NoHazard,      // This instruction can be emitted at this cycle.
    Hazard,        // This instruction can't be emitted at this cycle.
    NoopHazard     // This instruction can't be emitted, and needs noops.
  };

  /// getHazardType - Return the hazard type of emitting this node.  There are
  /// three possible results.  Either:
  ///  * NoHazard: it is legal to issue this instruction on this cycle.
  ///  * Hazard: issuing this instruction would stall the machine.  If some
  ///     other instruction is available, issue it first.
  ///  * NoopHazard: issuing this instruction would break the program.  If
  ///     some other instruction can be issued, do so, otherwise issue a noop.
  virtual HazardType getHazardType(SUnit *) {
    return NoHazard;
  }

  /// EmitInstruction - This callback is invoked when an instruction is
  /// emitted, to advance the hazard state.
  virtual void EmitInstruction(SUnit *) {}

  /// AdvanceCycle - This callback is invoked when no instructions can be
  /// issued on this cycle without a hazard.  This should increment the
  /// internal state of the hazard recognizer so that previously "Hazard"
  /// instructions will now not be hazards.
  virtual void AdvanceCycle() {}

  /// EmitNoop - This callback is invoked when a noop was added to the
  /// instruction stream.
  virtual void EmitNoop() {
    // Default implementation: count it as a cycle.
    AdvanceCycle();
  }
};

}

#endif
