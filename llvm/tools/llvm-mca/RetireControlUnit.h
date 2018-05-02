//===---------------------- RetireControlUnit.h -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the logic for retiring instructions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_RETIRE_CONTROL_UNIT_H
#define LLVM_TOOLS_LLVM_MCA_RETIRE_CONTROL_UNIT_H

#include "llvm/MC/MCSchedule.h"

namespace mca {

class DispatchUnit;

/// This class tracks which instructions are in-flight (i.e., dispatched but not
/// retired) in the OoO backend.
//
/// This class checks on every cycle if/which instructions can be retired.
/// Instructions are retired in program order.
/// In the event of instruction retired, the DispatchUnit object that owns
/// this RetireControlUnit (RCU) gets notified.
/// On instruction retired, register updates are all architecturally
/// committed, and any temporary registers originally allocated for the
/// retired instruction are freed.
struct RetireControlUnit {
  // A RUToken is created by the RCU for every instruction dispatched to the
  // schedulers.  These "tokens" are managed by the RCU in its token Queue.
  //
  // On evey cycle ('cycleEvent'), the RCU iterates through the token queue
  // looking for any token with its 'Executed' flag set.  If a token has that
  // flag set, then the instruction has reached the write-back stage and will
  // be retired by the RCU.
  //
  // 'NumSlots' represents the number of entries consumed by the instruction in
  // the reorder buffer. Those entries will become available again once the
  // instruction is retired.
  //
  // Note that the size of the reorder buffer is defined by the scheduling
  // model via field 'NumMicroOpBufferSize'.
  struct RUToken {
    unsigned Index;    // Instruction index.
    unsigned NumSlots; // Slots reserved to this instruction.
    bool Executed;     // True if the instruction is past the WB stage.
  };

private:
  unsigned NextAvailableSlotIdx;
  unsigned CurrentInstructionSlotIdx;
  unsigned AvailableSlots;
  unsigned MaxRetirePerCycle; // 0 means no limit.
  std::vector<RUToken> Queue;
  DispatchUnit *Owner;

public:
  RetireControlUnit(const llvm::MCSchedModel &SM, DispatchUnit *DU);

  bool isFull() const { return !AvailableSlots; }
  bool isEmpty() const { return AvailableSlots == Queue.size(); }
  bool isAvailable(unsigned Quantity = 1) const {
    // Some instructions may declare a number of uOps which exceedes the size
    // of the reorder buffer. To avoid problems, cap the amount of slots to
    // the size of the reorder buffer.
    Quantity = std::min(Quantity, static_cast<unsigned>(Queue.size()));
    return AvailableSlots >= Quantity;
  }

  // Reserves a number of slots, and returns a new token.
  unsigned reserveSlot(unsigned Index, unsigned NumMicroOps);

  /// Retires instructions in program order.
  void cycleEvent();

  void onInstructionExecuted(unsigned TokenID);

#ifndef NDEBUG
  void dump() const;
#endif
};

} // namespace mca

#endif // LLVM_TOOLS_LLVM_MCA_RETIRE_CONTROL_UNIT_H
