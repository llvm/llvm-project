/* --- PERegisterInfo.h --- */

/* ------------------------------------------
Author: 高宇翔
Date: 4/1/2025
------------------------------------------ */

#ifndef PEREGISTERINFO_H
#define PEREGISTERINFO_H

#define GET_REGINFO_HEADER
#include "PEGenRegisterInfo.inc"

namespace llvm{
class PERegisterInfo : public PEGenRegisterInfo{
public:
    PERegisterInfo();
    ~PERegisterInfo();

    //实现codegen生成的虚函数
    const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;

    BitVector getReservedRegs(const MachineFunction &MF) const override;

    bool eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
        unsigned FIOperandNum,
        RegScavenger *RS = nullptr) const override;

    Register getFrameRegister(const MachineFunction &MF) const override;
private:

};
}
#endif // PEREGISTERINFO_H
