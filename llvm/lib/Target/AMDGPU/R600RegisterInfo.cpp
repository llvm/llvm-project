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

  reserveRegisterTuples(Reserved, AMDGPU::ZERO);
  reserveRegisterTuples(Reserved, AMDGPU::HALF);
  reserveRegisterTuples(Reserved, AMDGPU::ONE);
  reserveRegisterTuples(Reserved, AMDGPU::ONE_INT);
  reserveRegisterTuples(Reserved, AMDGPU::NEG_HALF);
  reserveRegisterTuples(Reserved, AMDGPU::NEG_ONE);
  reserveRegisterTuples(Reserved, AMDGPU::PV_X);
  reserveRegisterTuples(Reserved, AMDGPU::ALU_LITERAL_X);
  reserveRegisterTuples(Reserved, AMDGPU::ALU_CONST);
  reserveRegisterTuples(Reserved, AMDGPU::PREDICATE_BIT);
  reserveRegisterTuples(Reserved, AMDGPU::PRED_SEL_OFF);
  reserveRegisterTuples(Reserved, AMDGPU::PRED_SEL_ZERO);
  reserveRegisterTuples(Reserved, AMDGPU::PRED_SEL_ONE);
  reserveRegisterTuples(Reserved, AMDGPU::INDIRECT_BASE_ADDR);

  for (TargetRegisterClass::iterator I = AMDGPU::R600_AddrRegClass.begin(),
                        E = AMDGPU::R600_AddrRegClass.end(); I != E; ++I) {
    reserveRegisterTuples(Reserved, *I);
  }

  TII->reserveIndirectRegisters(Reserved, MF, *this);

  return Reserved;
}

// Dummy to not crash RegisterClassInfo.
static const MCPhysReg CalleeSavedReg = AMDGPU::NoRegister;

const MCPhysReg *R600RegisterInfo::getCalleeSavedRegs(
  const MachineFunction *) const {
  return &CalleeSavedReg;
}

unsigned R600RegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return AMDGPU::NoRegister;
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
