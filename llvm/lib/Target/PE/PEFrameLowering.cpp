/* --- PEFrameLowering.cpp --- */

/* ------------------------------------------
author: undefined
date: 4/2/2025
------------------------------------------ */

#include "PEFrameLowering.h"

using namespace llvm;

//函数的汇编代码开头序言
void PEFrameLowering::emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const{

}

//函数的汇编代码结尾序言
void PEFrameLowering::emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const{

}

bool PEFrameLowering::hasFPImpl(const MachineFunction &MF) const{
    return false;
}


PEFrameLowering::~PEFrameLowering() {
    // Destructor
}
