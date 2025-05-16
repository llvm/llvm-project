/* --- PERegisterInfo.cpp --- */

/* ------------------------------------------
author: 高宇翔
date: 4/1/2025
------------------------------------------ */

#include "PERegisterInfo.h"
#include "MCTargetDesc/PEMCTargetDesc.h"
#include "llvm/ADT/BitVector.h"
#include "PESubtarget.h"
using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#include "PEGenRegisterInfo.inc"

PERegisterInfo::PERegisterInfo() : PEGenRegisterInfo(PE::RA){
}

PERegisterInfo::~PERegisterInfo() {
    // Destructor
}

const MCPhysReg *PERegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const{
    static const MCPhysReg CalleeSavedRegs[] = {PE::SP,0};
    return CalleeSavedRegs;

}

//BitVector代表一个位向量0&1
BitVector PERegisterInfo::getReservedRegs(const MachineFunction &MF) const {
    BitVector Reserved(getNumRegs());
    Reserved.set(PE::ZERO);
    Reserved.set(PE::RA);
    Reserved.set(PE::SP);
    return Reserved;

}

bool PERegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
    unsigned FIOperandNum,
    RegScavenger *RS) const {
        return false;//false表示成功

}
    
Register PERegisterInfo::getFrameRegister(const MachineFunction &MF) const {
    return PE::SP;

}