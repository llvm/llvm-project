//===-- ParasolFrameLowering.h - Define frame lowering for Parasol --------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file contains the ParasolTargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PARASOL_PARASOLFRAMELOWERING_H
#define LLVM_LIB_TARGET_PARASOL_PARASOLFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {
class ParasolSubtarget;

class ParasolFrameLowering : public TargetFrameLowering {
protected:
  const ParasolSubtarget &STI;

public:
  explicit ParasolFrameLowering(const ParasolSubtarget &STI)
      : TargetFrameLowering(TargetFrameLowering::StackGrowsDown,
                            /*StackAlignment*/ Align(4),
                            /*LocalAreaOffset*/ 0,
                            /*TransAl*/ Align(4)),
        STI(STI) {}

  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  bool hasReservedCallFrame(const MachineFunction &MF) const override;
  MachineBasicBlock::iterator
  eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I) const override;

  void determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                            RegScavenger *RS) const override;

  bool hasFP(const MachineFunction &MF) const override;
};
} // namespace llvm

#endif // end LLVM_LIB_TARGET_PARASOL_PARASOLFRAMELOWERING_H
