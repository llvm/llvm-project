/* --- PEFrameLowering.h --- */

/* ------------------------------------------
Author: 高宇翔
Date: 4/2/2025
------------------------------------------ */

#ifndef PEFRAMELOWERING_H
#define PEFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm{

class PESubtarget;

class PEFrameLowering : public TargetFrameLowering{
    const PESubtarget &STI;
public:
    explicit PEFrameLowering(const PESubtarget &STI)
        : TargetFrameLowering(StackGrowsDown, Align(4), 0,
            Align(4)),STI(STI){
            }
    ~PEFrameLowering();

    //函数的汇编代码开头序言
    void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

    //函数的汇编代码结尾序言
    void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

protected:
    bool hasFPImpl(const MachineFunction &MF) const override;

};
}

#endif // PEFRAMELOWERING_H
