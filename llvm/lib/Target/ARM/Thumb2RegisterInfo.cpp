//===-- Thumb2RegisterInfo.cpp - Thumb-2 Register Information -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-2 implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#include "Thumb2RegisterInfo.h"
#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMSubtarget.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

Thumb2RegisterInfo::Thumb2RegisterInfo(const ARMBaseInstrInfo &tii,
                                       const ARMSubtarget &sti)
  : ARMBaseRegisterInfo(tii, sti) {
}

/// emitLoadConstPool - Emits a load from constpool to materialize the
/// specified immediate.
void
Thumb2RegisterInfo::emitLoadConstPool(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator &MBBI,
                                      DebugLoc dl,
                                      unsigned DestReg, unsigned SubIdx,
                                      int Val,
                                      ARMCC::CondCodes Pred, unsigned PredReg,
                                      unsigned MIFlags) const {
  MachineFunction &MF = *MBB.getParent();
  MachineConstantPool *ConstantPool = MF.getConstantPool();
  const Constant *C = ConstantInt::get(
           Type::getInt32Ty(MBB.getParent()->getFunction()->getContext()), Val);
  unsigned Idx = ConstantPool->getConstantPoolIndex(C, 4);

  BuildMI(MBB, MBBI, dl, TII.get(ARM::t2LDRpci))
    .addReg(DestReg, getDefRegState(true), SubIdx)
    .addConstantPoolIndex(Idx).addImm((int64_t)ARMCC::AL).addReg(0)
    .setMIFlags(MIFlags);
}
