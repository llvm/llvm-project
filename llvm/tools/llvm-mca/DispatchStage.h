//===----------------------- DispatchStage.h --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file models the dispatch component of an instruction pipeline.
///
/// The DispatchStage is responsible for updating instruction dependencies
/// and communicating to the simulated instruction scheduler that an instruction
/// is ready to be scheduled for execution.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_DISPATCH_STAGE_H
#define LLVM_TOOLS_LLVM_MCA_DISPATCH_STAGE_H

#include "Instruction.h"
#include "RegisterFile.h"
#include "RetireControlUnit.h"
#include "Stage.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace mca {

class WriteState;
class Scheduler;
class Pipeline;

// Implements the hardware dispatch logic.
//
// This class is responsible for the dispatch stage, in which instructions are
// dispatched in groups to the Scheduler.  An instruction can be dispatched if
// the following conditions are met:
//  1) There are enough entries in the reorder buffer (see class
//     RetireControlUnit) to write the opcodes associated with the instruction.
//  2) There are enough temporaries to rename output register operands.
//  3) There are enough entries available in the used buffered resource(s).
//
// The number of micro opcodes that can be dispatched in one cycle is limited by
// the value of field 'DispatchWidth'. A "dynamic dispatch stall" occurs when
// processor resources are not available. Dispatch stall events are counted
// during the entire execution of the code, and displayed by the performance
// report when flag '-dispatch-stats' is specified.
//
// If the number of micro opcodes exceedes DispatchWidth, then the instruction
// is dispatched in multiple cycles.
class DispatchStage : public Stage {
  unsigned DispatchWidth;
  unsigned AvailableEntries;
  unsigned CarryOver;
  Pipeline *Owner;
  const llvm::MCSubtargetInfo &STI;
  RetireControlUnit &RCU;
  RegisterFile &PRF;
  Scheduler &SC;

  bool checkRCU(const InstRef &IR);
  bool checkPRF(const InstRef &IR);
  bool checkScheduler(const InstRef &IR);
  void dispatch(InstRef IR);
  void updateRAWDependencies(ReadState &RS, const llvm::MCSubtargetInfo &STI);

  void notifyInstructionDispatched(const InstRef &IR,
                                   llvm::ArrayRef<unsigned> UsedPhysRegs);

  bool isAvailable(unsigned NumEntries) const {
    return NumEntries <= AvailableEntries || AvailableEntries == DispatchWidth;
  }

  bool canDispatch(const InstRef &IR) {
    assert(isAvailable(IR.getInstruction()->getDesc().NumMicroOps));
    return checkRCU(IR) && checkPRF(IR) && checkScheduler(IR);
  }

  void collectWrites(llvm::SmallVectorImpl<WriteState *> &Vec,
                     unsigned RegID) const {
    return PRF.collectWrites(Vec, RegID);
  }

public:
  DispatchStage(Pipeline *P, const llvm::MCSubtargetInfo &Subtarget,
                const llvm::MCRegisterInfo &MRI, unsigned RegisterFileSize,
                unsigned MaxDispatchWidth, RetireControlUnit &R,
                RegisterFile &F, Scheduler &Sched)
      : DispatchWidth(MaxDispatchWidth), AvailableEntries(MaxDispatchWidth),
        CarryOver(0U), Owner(P), STI(Subtarget), RCU(R), PRF(F), SC(Sched) {}

  // We can always try to dispatch, so returning false is okay in this case.
  // The retire stage, which controls the RCU, might have items to complete but
  // RetireStage::hasWorkToComplete will check for that case.
  virtual bool hasWorkToComplete() const override final { return false; }
  virtual void preExecute(const InstRef &IR) override final;
  virtual bool execute(InstRef &IR) override final;
  void notifyDispatchStall(const InstRef &IR, unsigned EventType);

#ifndef NDEBUG
  void dump() const;
#endif
};
} // namespace mca

#endif // LLVM_TOOLS_LLVM_MCA_DISPATCH_STAGE_H
