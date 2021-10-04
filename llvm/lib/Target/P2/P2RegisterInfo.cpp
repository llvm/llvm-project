//===-- P2RegisterInfo.cpp - P2 Register Information --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the P2 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "P2.h"
#include "P2MachineFunctionInfo.h"
#include "P2RegisterInfo.h"
#include "P2TargetMachine.h"
#include "MCTargetDesc/P2BaseInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define DEBUG_TYPE "p2-reg-info"

#define GET_REGINFO_TARGET_DESC
#include "P2GenRegisterInfo.inc"

P2RegisterInfo::P2RegisterInfo() : P2GenRegisterInfo(0) {}

const MCPhysReg* P2RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
    return CSR_SaveList;
}

const uint32_t *P2RegisterInfo::getCallPreservedMask(const MachineFunction &MF,CallingConv::ID CC) const {
    return CSR_RegMask;
}

BitVector P2RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
    BitVector Reserved(getNumRegs());

    Reserved.set(P2::IJMP3);
    Reserved.set(P2::IRET3);
    Reserved.set(P2::IJMP2);
    Reserved.set(P2::IRET2);
    Reserved.set(P2::IJMP1);
    Reserved.set(P2::IRET1);
    Reserved.set(P2::PA);
    Reserved.set(P2::PB);
    Reserved.set(P2::PTRA);
    Reserved.set(P2::PTRB);
    Reserved.set(P2::OUTA);
    Reserved.set(P2::OUTB);
    Reserved.set(P2::DIRA);
    Reserved.set(P2::DIRB);
    Reserved.set(P2::DIRA);
    Reserved.set(P2::DIRB);
    Reserved.set(P2::INA);
    Reserved.set(P2::INB);

    Reserved.set(P2::QX);
    Reserved.set(P2::QY);

    Reserved.set(P2::R30); // reserve R30 since it's used for returns
    Reserved.set(P2::R31); // reserve R31 since it's used for returns
    return Reserved;
}

void P2RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj, unsigned FIOperandNum, RegScavenger *RS) const {

    // get a bunch of target info classes
    MachineInstr &MI = *II;
    DebugLoc dl = MI.getDebugLoc();
    MachineFunction &MF = *MI.getParent()->getParent();
    MachineFrameInfo &MFI = MF.getFrameInfo();
    //P2FunctionInfo *P2FI = MF.getInfo<P2FunctionInfo>();
    const P2TargetMachine &TM = (const P2TargetMachine &)MF.getTarget();
    const TargetInstrInfo &inst_info = *TM.getInstrInfo();
    //const TargetFrameLowering *TFI = TM.getFrameLowering();

    LLVM_DEBUG(
        errs() << "\nFunction : " << MF.getFunction().getName() << "\n";
        errs() << "<--------->\n" << MI);

    int frame_idx = MI.getOperand(FIOperandNum).getIndex();
    uint64_t stack_size = MFI.getStackSize();
    int64_t fi_offset = MFI.getObjectOffset(frame_idx); // offset from the start of the frame (low address)
    // int local_frame_size = MF.getFrameInfo().getLocalFrameSize();
    int64_t offset = stack_size-fi_offset;

    LLVM_DEBUG(errs() << "frame_idx : " << frame_idx << "\n"
                        << "stack size  : " << stack_size << "\n"
                        << "fi offset : " << fi_offset << "\n");
    // LLVM_DEBUG(dbgs() << "LocalFrameSize : " << local_frame_size << "\n");

    LLVM_DEBUG(errs() << "Offset     : " << offset << "\n" << "<--------->\n");

    assert(offset >= 0 && "Invalid offset"); // offset should be positive or 0

    int op = MI.getOpcode();

    // if the op code using the frame index is rdlong or wrlong, we can use a special immediate to read/write PTRA
    // bool can_use_ptr_off = (op == P2::WRLONGri) || (op == P2::RDLONGri);

    if (MI.getOpcode() == P2::FRMIDX) {
        MI.setDesc(inst_info.get(P2::MOVrr)); // change our psesudo instruction to a mov
        MI.getOperand(FIOperandNum).ChangeToRegister(P2::PTRA, false); // change the abstract frame index register to our real stack pointer register
        MI.addOperand(MachineOperand::CreateImm(P2::ALWAYS));
        MI.addOperand(MachineOperand::CreateImm(P2::NOEFF));

        Register dst_reg = MI.getOperand(0).getReg();
        II++; // skip forward by 1 instruction

        BuildMI(*MI.getParent(), II, dl, inst_info.get(P2::SUBri), dst_reg)
                                .addReg(dst_reg, RegState::Kill)
                                .addImm(offset)
                                .addImm(P2::ALWAYS)
                                .addImm(P2::NOEFF);
    } else if ((op == P2::WRLONGri) || (op == P2::RDLONGri)) {
        int imm = 0x4;

        // offset can be up to 128, since the index for the special immediate is scaled by 4 when using rdlong/wrlong
        // later we can make this more generic for bytes and words too--adjusting the scale appropriately
        if (offset > 128) {
            imm <<= 20;
            imm += ((1<<20) - offset/4) & 0xfffff;

            BuildMI(*MI.getParent(), II, dl, inst_info.get(P2::AUGS)).addImm(imm >> 9);
            MI.getOperand(1).ChangeToImmediate(imm & 0x1ff);
        } else {
            imm <<= 6;
            imm += ((1<<6) - offset/4) & 0x3f;

            MI.getOperand(1).ChangeToImmediate(imm & 0x1ff);
        }

    } else {
        // if we decide we need to scavange registers, we need to create an emergency stack slock in frame lowering,
        // then make sure to kill the register after it is used here. For now, we can just use PTRB as a second stack pointer
        // register for writing to this frame index

        Register reg = P2::PA;

        auto inst = II->getPrevNode();
        if (inst && (inst->getOpcode() == P2::SETQr || inst->getOpcode() == P2::SETQi)) {
            // if we have a setq before the instruction with the frame index, back up so keep the setq next to the
            // desired instruction
            LLVM_DEBUG(errs() << "have a setq, backing up by one instruction\n");
            II--;
        }

        BuildMI(*MI.getParent(), II, dl, inst_info.get(P2::MOVrr), reg)
                                .addReg(P2::PTRA)
                                .addImm(P2::ALWAYS)
                                .addImm(P2::NOEFF); // save the SP to an unused register

        BuildMI(*MI.getParent(), II, dl, inst_info.get(P2::SUBri), reg)
                                .addReg(reg, RegState::Kill)
                                .addImm(offset)
                                .addImm(P2::ALWAYS)
                                .addImm(P2::NOEFF); // adjust saved SP by frame index offset

        MI.getOperand(FIOperandNum).ChangeToRegister(reg, false);

        LLVM_DEBUG(MI.dump());
    }
}

Register P2RegisterInfo::getFrameRegister(const MachineFunction &MF) const {
    return P2::PTRA;
}
