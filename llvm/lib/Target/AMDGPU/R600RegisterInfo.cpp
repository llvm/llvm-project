//===-- R600RegisterInfo.cpp - R600 Register Information ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief R600 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "R600RegisterInfo.h"
#include "AMDGPUTargetMachine.h"
#include "R600Defines.h"
#include "R600InstrInfo.h"
#include "R600MachineFunctionInfo.h"

using namespace llvm;

R600RegisterInfo::R600RegisterInfo() : AMDGPURegisterInfo() {
  RCW.RegWeight = 0;
  RCW.WeightLimit = 0;
}

BitVector R600RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());

  const R600Subtarget &ST = MF.getSubtarget<R600Subtarget>();
  const R600InstrInfo *TII = ST.getInstrInfo();

  Reserved.set(AMDGPU::ZERO);
  Reserved.set(AMDGPU::HALF);
  Reserved.set(AMDGPU::ONE);
  Reserved.set(AMDGPU::ONE_INT);
  Reserved.set(AMDGPU::NEG_HALF);
  Reserved.set(AMDGPU::NEG_ONE);
  Reserved.set(AMDGPU::PV_X);
  Reserved.set(AMDGPU::ALU_LITERAL_X);
  Reserved.set(AMDGPU::ALU_CONST);
  Reserved.set(AMDGPU::PREDICATE_BIT);
  Reserved.set(AMDGPU::PRED_SEL_OFF);
  Reserved.set(AMDGPU::PRED_SEL_ZERO);
  Reserved.set(AMDGPU::PRED_SEL_ONE);
  Reserved.set(AMDGPU::INDIRECT_BASE_ADDR);

  for (TargetRegisterClass::iterator I = AMDGPU::R600_AddrRegClass.begin(),
                        E = AMDGPU::R600_AddrRegClass.end(); I != E; ++I) {
    Reserved.set(*I);
  }

  TII->reserveIndirectRegisters(Reserved, MF);

  return Reserved;
}

unsigned R600RegisterInfo::getHWRegChan(unsigned reg) const {
  return this->getEncodingValue(reg) >> HW_CHAN_SHIFT;
}

unsigned R600RegisterInfo::getHWRegIndex(unsigned Reg) const {
  return GET_REG_INDEX(getEncodingValue(Reg));
}

const TargetRegisterClass * R600RegisterInfo::getCFGStructurizerRegClass(
                                                                   MVT VT) const {
  switch(VT.SimpleTy) {
  default:
  case MVT::i32: return &AMDGPU::R600_TReg32RegClass;
  }
}

const RegClassWeight &R600RegisterInfo::getRegClassWeight(
  const TargetRegisterClass *RC) const {
  return RCW;
}

bool R600RegisterInfo::isPhysRegLiveAcrossClauses(unsigned Reg) const {
  assert(!TargetRegisterInfo::isVirtualRegister(Reg));

  switch (Reg) {
  case AMDGPU::OQAP:
  case AMDGPU::OQBP:
  case AMDGPU::AR_X:
    return false;
  default:
    return true;
  }
}

void R600RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                           int SPAdj,
                                           unsigned FIOperandNum,
                                           RegScavenger *RS) const {
  llvm_unreachable("Subroutines not supported yet");
}
