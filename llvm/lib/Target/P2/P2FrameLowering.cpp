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

Callee saved register spilling/restoring is done via setq and wrlong/rdlong to do a block transfer of
registers to memory. determine callee saves gives us a list of regsiters to save and their frame indices. We count up the number of
continuous registers that need to be saved in a single setq/wrlong pair. Restoring does the same thing in reverse.
Eventually, we might need to adjust determine callee saves to line up the register blocks with their corresponing frame indicies,
though they might be aligned now?

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

    if (MF.getFunction().hasFnAttribute(Attribute::Cogmain)) {
        LLVM_DEBUG(errs() << "cog entry function, saving ptra to r0\n");
        DebugLoc DL = MBB.findDebugLoc(MBBI);
        BuildMI(MBB, MBBI, DL, TII->get(P2::MOVrr))
            .addReg(P2::R0)
            .addReg(P2::PTRA)
            .addImm(P2::ALWAYS)
            .addImm(P2::NOEFF);
    }

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

    if (MF.getFunction().hasFnAttribute(Attribute::Cogmain)) {
        return;
    }

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

    // use auto-incrementing feature of ptra to write blocks of regsiters to ptra (instead of one register at a time)
    // when doing this, we want to make sure the manual allocation of ptra happens after this code, so we mark each instruction
    // as a FrameSetup instruction, then in emitPrologue, skip over any FrameSetup instructions we have.
    //
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
                    .addImm(P2::ALWAYS)
                    .setMIFlag(MachineInstr::FrameSetup);
            }

            // write the first block register to ptra, incrementing ptra. if we added setq above, it will write
            // a block of registers.
            BuildMI(MBB, MI, DL, TII.get(P2::WRLONGri), block_first_reg)
                .addImm(P2::PTRA_POSTINC)
                .addImm(P2::ALWAYS)
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
            .addImm(P2::ALWAYS)
            .setMIFlag(MachineInstr::FrameSetup);
    }

    BuildMI(MBB, MI, DL, TII.get(P2::WRLONGri), block_first_reg)
        .addImm(P2::PTRA_POSTINC)
        .addImm(P2::ALWAYS)
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

    // see spillCalleeSavedRegisters for explanation, this is just doing the same thin in reverse
    //
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
                    .addImm(P2::ALWAYS)
                    .setMIFlag(MachineInstr::FrameDestroy);
            }

            BuildMI(MBB, MI, DL, TII.get(P2::RDLONGri), block_first_reg)
                .addImm(P2::PTRA_PREDEC)
                .addImm(P2::ALWAYS)
                .addImm(P2::NOEFF)
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
            .addImm(P2::ALWAYS)
            .setMIFlag(MachineInstr::FrameDestroy);
    }

    BuildMI(MBB, MI, DL, TII.get(P2::RDLONGri), block_first_reg)
        .addImm(P2::PTRA_PREDEC)
        .addImm(P2::ALWAYS)
        .addImm(P2::NOEFF)
        .setMIFlag(MachineInstr::FrameDestroy);

    return true;
}

MachineBasicBlock::iterator P2FrameLowering::eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I) const {

    LLVM_DEBUG(errs() << "=== eliminate call frame pseudo\n");

    if (!hasReservedCallFrame(MF)) {

        int64_t adjust = I->getOperand(0).getImm();


        if (I->getOpcode() == P2::ADJCALLSTACKDOWN) {
            adjust = -adjust;
            I = MBB.erase(I);
        } else {
            I = MBB.erase(I); // first erase our psuedo instruction.

            auto op = I->getOpcode();
            while (op != P2::CALLa && op != P2::CALLAa && op != P2::CALLAr && op != P2::CALLr) {
                LLVM_DEBUG(errs() << "instruction to skip: \n");
                LLVM_DEBUG(I->dump());
                LLVM_DEBUG(errs() << "op code is " << op << "\n");
                I++; // skip ahead to the call instruction.

                op = I->getOpcode();
            }
        }

        // adjust the stack pointer, if necessary
        if (adjust)
            tm.getInstrInfo()->adjustStackPtr(P2::PTRA, adjust, MBB, I);
    }

    return I;
}