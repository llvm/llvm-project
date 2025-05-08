//===-- ParasolInstrInfo.h - Parasol Instruction Information --------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file contains the Parasol implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PARASOL_PARASOLINSTRINFO_H
#define LLVM_LIB_TARGET_PARASOL_PARASOLINSTRINFO_H

#include "Parasol.h"
#include "ParasolRegisterInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "ParasolGenInstrInfo.inc"

namespace llvm {

class ParasolInstrInfo : public ParasolGenInstrInfo {
public:
  explicit ParasolInstrInfo(const ParasolSubtarget &STI);

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                   const DebugLoc &DL, MCRegister DestReg, MCRegister SrcReg,
                   bool KillSrc) const override;

protected:
  const ParasolSubtarget &Subtarget;
};
} // namespace llvm

#endif // end LLVM_LIB_TARGET_PARASOL_PARASOLINSTRINFO_H
