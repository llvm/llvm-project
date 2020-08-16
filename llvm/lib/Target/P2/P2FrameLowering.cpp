//===-- P2FrameLowering.cpp - P2 Frame Information --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the P2 implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "P2FrameLowering.h"

#include "P2InstrInfo.h"
#include "P2TargetMachine.h"
#include "P2MachineFunctionInfo.h"
#include "MCTargetDesc/P2BaseInfo.h"

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetOptions.h"

/*
How the call stack will work:

The stack will grow up. regsiter sp always points to the TOP of the stack, which is the start of free stack space.
Selection will generate FRMIDX pseudo instructions that will be lowered in register info by be subtracting from the
current stack pointer (sp) by the frame index offset. The callee will not save any regsiters it uses. The data in the stack frame
will be organized as follows:
SP ----------> ----------------- (4)
                local variables
                ...
               ----------------- (3)
                callee saved regs
                ...
               ----------------- (2)
                return address (pushed automatically)
               ----------------- (1)
                arguments into function (descending).
                formal arguments come first (highest in the stack)
                followed by variable argument (last var arg is lowest in the stack)
                ...
SP (previous)  -----------------

Here's the ordering of sp adjustment: when calling, SP (previous) is adjusted for arguments (1). The function is called and return address
(and status word) is pushed onto the stack with the call (2). The function then allocates space it needs to save registers (3), and local
variables (4). SP now becomes SP (previous) when getting
ready to call another function.

[This doesn't work yet] Callee saved register spilling/restoring will be done via setq and wrlong/rdlong to do a block transfer of
registers to memory determin callee saves gives us a list of regsiters to save and their frame indices. We count up the number of
continuous registers that need to be saved in a single setq/wrlong pair. Restoring does the same thing in reverse.

*/

#define DEBUG_TYPE "p2-frame-lower"

using namespace llvm;

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas,
// if it needs dynamic stack realignment, if frame pointer elimination is
// disabled, or if the frame address is taken.
//
// TODO figure out if this function is actually needed or if can always return a static value
bool P2FrameLowering::hasFP(const MachineFunction &MF) const {
    const MachineFrameInfo *MFI = &MF.getFrameInfo();
    const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

    // LLVM_DEBUG(errs() << "hasFP = disable FP elim: " << MF.getTarget().Options.DisableFramePointerElim(MF) <<
    //             "; var sized objects: " << MFI->hasVarSizedObjects() <<
    //             "; frame address is taken: " << MFI->isFrameAddressTaken() <<
    //             "; needs stack realignment: " << TRI->needsStackRealignment(MF) << "\n");

    return MF.getTarget().Options.DisableFramePointerElim(MF) ||
            MFI->hasVarSizedObjects() || MFI->isFrameAddressTaken() ||
            TRI->needsStackRealignment(MF);
}

void P2FrameLowering::emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const {
    LLVM_DEBUG(dbgs() << "Emit Prologue: " << MF.getName() << "\n");

    const P2InstrInfo *TII = MF.getSubtarget<P2Subtarget>().getInstrInfo();
    MachineBasicBlock::iterator MBBI = MBB.begin();
    P2FunctionInfo *P2FI = MF.getInfo<P2FunctionInfo>();
    MachineFrameInfo &MFI = MF.getFrameInfo();

    LLVM_DEBUG(errs() << "prologue mbb\n");
    LLVM_DEBUG(MBB.dump());

    // the stack gets preallocated for incoming arguments + 4 bytes for the PC/SW + regs already saved to the stack,
    // so don't allocate that in the prologue
    uint64_t StackSize = MFI.getStackSize() - 4 - P2FI->getIncomingArgSize() - P2FI->getCalleeSavedFrameSize();
    LLVM_DEBUG(errs() << "Allocating " << StackSize << " bytes for stack (original value: " << MFI.getStackSize() << ")\n");

    // No need to allocate space on the stack.
    if (StackSize == 0) {
        LLVM_DEBUG(errs() << "No need to allocate stack space\n");
        return;
    }

    // we want to iterate MBBI until we hit the first function instruction, we marked the callee saving instructions
    // as FrameSetup instructions
    if (P2FI->getCalleeSavedFrameSize())
        while ((*MBBI).getFlag(MachineInstr::FrameSetup)) MBBI++;

    TII->adjustStackPtr(P2::PTRA, StackSize, MBB, MBBI);
}

void P2FrameLowering::emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const {
    LLVM_DEBUG(dbgs() << "Emit Epilogue: " << MF.getName() << "\n");
    MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
    const MachineFrameInfo &MFI = MF.getFrameInfo();
    P2FunctionInfo *P2FI = MF.getInfo<P2FunctionInfo>();

    const P2InstrInfo *TII = MF.getSubtarget<P2Subtarget>().getInstrInfo();
    uint64_t StackSize = MFI.getStackSize() - 4 - P2FI->getIncomingArgSize() - P2FI->getCalleeSavedFrameSize();

    LLVM_DEBUG(errs() << "epilogue mbb\n");
    LLVM_DEBUG(MBB.dump());

    // allocate 0s for now for testing
    if (StackSize == 0) {
        LLVM_DEBUG(errs() << "No need to de-allocate stack space\n");
        return;
    }

    // back up before the callee restore instructions/return instruction, then insert the stack pointer adjustment
    if (P2FI->getCalleeSavedFrameSize())
        while (MBBI != MBB.begin() && MBBI->getPrevNode()->getFlag(MachineInstr::FrameDestroy)) MBBI--;

    // Adjust stack.
    TII->adjustStackPtr(P2::PTRA, -StackSize, MBB, MBBI);
}

void P2FrameLowering::determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs, RegScavenger *RS) const {
    LLVM_DEBUG(errs() << "=== Function: " << MF.getName() << " ===\n");
    LLVM_DEBUG(errs() << "Determining callee saves\n");
    TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
    // eventually might need to add to this to re-order the frame index based to match what will happen in spilling/restoring
}

bool P2FrameLowering::spillCalleeSavedRegisters(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                                                ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {


    unsigned CalleeFrameSize = 0;
    DebugLoc DL = MBB.findDebugLoc(MI);
    MachineFunction &MF = *MBB.getParent();
    const P2Subtarget &STI = MF.getSubtarget<P2Subtarget>();
    const TargetInstrInfo &TII = *STI.getInstrInfo();
    P2FunctionInfo *P2FI = MF.getInfo<P2FunctionInfo>();
    MachineFrameInfo *MFI = &MF.getFrameInfo();

    LLVM_DEBUG(errs() << "=== Function: " << MF.getName() << " ===\n");
    LLVM_DEBUG(errs() << "Spilling callee saves\n");

    if (CSI.empty())
        return false;

    CalleeFrameSize = CSI.size()*4;

    // for (unsigned i = 0; i < CSI.size(); i++) {
    //     unsigned Reg = CSI[i].getReg();
    //     bool IsNotLiveIn = !MBB.isLiveIn(Reg);
    //     // Add the callee-saved register as live-in only if it is not already a
    //     // live-in register, this usually happens with arguments that are passed
    //     // through callee-saved registers.
    //     if (IsNotLiveIn) {
    //         MBB.addLiveIn(Reg);
    //     }

    //     const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    //     TII.storeRegToStackSlot(MBB, MI, Reg, false, CSI[i].getFrameIdx(), RC, TRI);

    //     CalleeFrameSize += 4;

    //     LLVM_DEBUG(errs() << "--- spilling " << Reg << " to index " << CSI[i].getFrameIdx() << "\n");
    // }

    // return true;

    // experimental code that doesn't work yet...
    // block size is 1 less than number of regs to write in a block transfer (which is also the number to give to setq)
    uint16_t block_size = 0;
    int block_first_reg = CSI[0].getReg();

    LLVM_DEBUG(errs() << "reg: " << block_first_reg << "\n");

    for (int i = 1; i < CSI.size(); i++) {

        unsigned reg = CSI[i].getReg();
        unsigned prev_reg = CSI[i-1].getReg();

        uint16_t reg_encoding = TRI->getEncodingValue(reg);
        uint16_t prev_reg_encoding = TRI->getEncodingValue(prev_reg);

        bool IsNotLiveIn = !MBB.isLiveIn(reg);
        // Add the callee-saved register as live-in only if it is not already a
        // live-in register, this usually happens with arguments that are passed
        // through callee-saved registers.
        if (IsNotLiveIn) {
            MBB.addLiveIn(reg);
        }

        if (reg_encoding - prev_reg_encoding != 1) {
            // this is a new register block, so let's write the previous block first.

            if (block_size) {
                // if we have more than 1 reg to write, add setq.
                BuildMI(MBB, MI, DL, TII.get(P2::SETQi))
                    .addImm(block_size)
                    .setMIFlag(MachineInstr::FrameSetup);
            }

            // write the first block register to ptra, incrementing ptra. if we added setq above, it will write
            // a block of registers.
            BuildMI(MBB, MI, DL, TII.get(P2::WRLONGri), block_first_reg)
                .addImm(P2::PTRA_POSTINC)
                .setMIFlag(MachineInstr::FrameSetup);

            LLVM_DEBUG(errs() << "New block transfer to reg " << block_first_reg << "\n");

            block_size = 0;
            block_first_reg = reg;
        } else {
            block_size++;
        }

        LLVM_DEBUG(errs() << "reg: " << reg << "\n");
    }

    if (block_size) {
        BuildMI(MBB, MI, DL, TII.get(P2::SETQi))
            .addImm(block_size)
            .setMIFlag(MachineInstr::FrameSetup);
    }

    BuildMI(MBB, MI, DL, TII.get(P2::WRLONGri), block_first_reg)
        .addImm(P2::PTRA_POSTINC)
        .setMIFlag(MachineInstr::FrameSetup);

    LLVM_DEBUG(errs() << "New block transfer to reg " << block_first_reg << "\n");

    P2FI->setCalleeSavedFrameSize(CalleeFrameSize);

    return true;
}

bool P2FrameLowering::restoreCalleeSavedRegisters(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                                                MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
    MachineFunction &MF = *MBB.getParent();
    const P2Subtarget &STI = MF.getSubtarget<P2Subtarget>();
    const TargetInstrInfo &TII = *STI.getInstrInfo();
    MachineFrameInfo *MFI = &MF.getFrameInfo();
    DebugLoc DL = MBB.findDebugLoc(MI);

    LLVM_DEBUG(errs() << "=== Function: " << MF.getName() << " ===\n");

    LLVM_DEBUG(errs() << "Restore CSRs\n");
    if (CSI.empty()) {
        LLVM_DEBUG(errs() << "--- nothing to restore\n");
        return false;
    }

    // for (unsigned i = CSI.size(); i != 0; --i) {
    //     unsigned Reg = CSI[i-1].getReg();
    //     const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    //     TII.loadRegFromStackSlot(MBB, MI, Reg, CSI[i-1].getFrameIdx(), RC, TRI);

    //     LLVM_DEBUG(errs() << "--- restoring " << Reg << " from index " << CSI[i-1].getFrameIdx() << "\n");
    // }

    // return true;

    // experimental code that doesn't work yet...
    // block size is 1 less than number of regs to write in a block transfer (which is also the number to give to setq)
    // go in reverse order since we are auto-decrementing ptra
    uint16_t block_size = 0;
    int block_first_reg = CSI[CSI.size()-1].getReg();

    LLVM_DEBUG(errs() << "reg: " << block_first_reg << "\n");

    for (int i = CSI.size()-2; i >= 0; i--) {
        unsigned reg = CSI[i].getReg();
        unsigned prev_reg = CSI[i+1].getReg();

        LLVM_DEBUG(errs() << "reg: " << reg << "\n");

        uint16_t reg_encoding = TRI->getEncodingValue(reg);
        uint16_t prev_reg_encoding = TRI->getEncodingValue(prev_reg);

        bool IsNotLiveIn = !MBB.isLiveIn(reg);
        // Add the callee-saved register as live-in only if it is not already a
        // live-in register, this usually happens with arguments that are passed
        // through callee-saved registers.
        if (IsNotLiveIn) {
            MBB.addLiveIn(reg);
        }

        if (prev_reg_encoding - reg_encoding != 1) {
            // this is a new register block, so let's write the previous block first.

            if (block_size) {
                BuildMI(MBB, MI, DL, TII.get(P2::SETQi))
                    .addImm(block_size)
                    .setMIFlag(MachineInstr::FrameDestroy);
            }

            BuildMI(MBB, MI, DL, TII.get(P2::RDLONGri), block_first_reg)
                .addImm(P2::PTRA_PREDEC)
                .setMIFlag(MachineInstr::FrameDestroy);

            block_size = 0;
        } else {
            block_size++;
            block_first_reg = reg;
        }
    }

    // write the final block out

    if (block_size) {
        BuildMI(MBB, MI, DL, TII.get(P2::SETQi))
            .addImm(block_size)
            .setMIFlag(MachineInstr::FrameDestroy);
    }

    BuildMI(MBB, MI, DL, TII.get(P2::RDLONGri), block_first_reg)
        .addImm(P2::PTRA_PREDEC)
        .setMIFlag(MachineInstr::FrameDestroy);

    return true;
}

MachineBasicBlock::iterator P2FrameLowering::eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I) const {

    LLVM_DEBUG(errs() << "=== eliminate call frame pseudo\n");

    if (!hasReservedCallFrame(MF)) {
        int64_t Amount = I->getOperand(0).getImm();

        if (I->getOpcode() == P2::ADJCALLSTACKDOWN)
            Amount = -Amount;

        if (Amount)
            tm.getInstrInfo()->adjustStackPtr(P2::PTRA, Amount, MBB, I);
    }


    return MBB.erase(I);
}