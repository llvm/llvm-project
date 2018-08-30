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

#include "HWEventListener.h"
#include "HardwareUnits/RegisterFile.h"
#include "HardwareUnits/RetireControlUnit.h"
#include "Instruction.h"
#include "Stages/Stage.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace mca {

// Implements the hardware dispatch logic.
//
// This class is responsible for the dispatch stage, in which instructions are
// dispatched in groups to the Scheduler.  An instruction can be dispatched if
// the following conditions are met:
//  1) There are enough entries in the reorder buffer (see class
//     RetireControlUnit) to write the opcodes associated with the instruction.
//  2) There are enough physical registers to rename output register operands.
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
class DispatchStage final : public Stage {
  unsigned DispatchWidth;
  unsigned AvailableEntries;
  unsigned CarryOver;
  InstRef CarriedOver;
  const llvm::MCSubtargetInfo &STI;
  RetireControlUnit &RCU;
  RegisterFile &PRF;

  bool checkRCU(const InstRef &IR) const;
  bool checkPRF(const InstRef &IR) const;
  bool canDispatch(const InstRef &IR) const;
  llvm::Error dispatch(InstRef IR);

  void updateRAWDependencies(ReadState &RS, const llvm::MCSubtargetInfo &STI);

  void notifyInstructionDispatched(const InstRef &IR,
                                   llvm::ArrayRef<unsigned> UsedPhysRegs,
                                   unsigned uOps);

  void collectWrites(llvm::SmallVectorImpl<WriteRef> &Vec,
                     unsigned RegID) const {
    return PRF.collectWrites(Vec, RegID);
  }

public:
  DispatchStage(const llvm::MCSubtargetInfo &Subtarget,
                const llvm::MCRegisterInfo &MRI, unsigned MaxDispatchWidth,
                RetireControlUnit &R, RegisterFile &F)
      : DispatchWidth(MaxDispatchWidth), AvailableEntries(MaxDispatchWidth),
        CarryOver(0U), CarriedOver(), STI(Subtarget), RCU(R), PRF(F) {}

  bool isAvailable(const InstRef &IR) const override;

  // The dispatch logic internally doesn't buffer instructions. So there is
  // never work to do at the beginning of every cycle.
  bool hasWorkToComplete() const override { return false; }
  llvm::Error cycleStart() override;
  llvm::Error execute(InstRef &IR) override;

#ifndef NDEBUG
  void dump() const;
#endif
};
} // namespace mca

#endif // LLVM_TOOLS_LLVM_MCA_DISPATCH_STAGE_H
