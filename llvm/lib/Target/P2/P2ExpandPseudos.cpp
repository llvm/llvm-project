//===- P2ExpandPseudosPass - P2 expand pseudo loads -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands stores with large offsets into an appropriate sequence.
//===----------------------------------------------------------------------===//

#include "P2.h"
#include "P2InstrInfo.h"
#include "P2RegisterInfo.h"
#include "P2Subtarget.h"
#include "P2TargetMachine.h"
#include "MCTargetDesc/P2BaseInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "p2-expand-pseudos"

namespace {

    class P2ExpandPseudos : public MachineFunctionPass {
    public:
        static char ID;
        P2ExpandPseudos(P2TargetMachine &tm) : MachineFunctionPass(ID), TM(tm) {}

        bool runOnMachineFunction(MachineFunction &Fn) override;

        StringRef getPassName() const override { return "P2 Expand Pseudos"; }

    private:
        const P2InstrInfo *TII;
        const P2TargetMachine &TM;

        void expand_QUDIV(MachineFunction &MF, MachineBasicBlock::iterator SII);
        void expand_QUREM(MachineFunction &MF, MachineBasicBlock::iterator SII);
        void expand_MOVri32(MachineFunction &MF, MachineBasicBlock::iterator SII);
        void expand_SELECTCC(MachineFunction &MF, MachineBasicBlock::iterator SII);
    };

    char P2ExpandPseudos::ID = 0;

} // end anonymous namespace

void P2ExpandPseudos::expand_QUDIV(MachineFunction &MF, MachineBasicBlock::iterator SII) {
    MachineInstr &SI = *SII;

    LLVM_DEBUG(errs()<<"== lower pseudo unsigned division\n");
    LLVM_DEBUG(SI.dump());

    BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::QDIVrr))
            .addReg(SI.getOperand(1).getReg())
            .addReg(SI.getOperand(2).getReg())
            .addImm(P2::ALWAYS);
    BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::GETQX), SI.getOperand(0).getReg())
            .addReg(P2::QX)
            .addImm(P2::ALWAYS)
            .addImm(P2::NOEFF);

    SI.eraseFromParent();
}

void P2ExpandPseudos::expand_QUREM(MachineFunction &MF, MachineBasicBlock::iterator SII) {
    MachineInstr &SI = *SII;

    LLVM_DEBUG(errs()<<"== lower pseudo unsigned remainder\n");
    LLVM_DEBUG(SI.dump());

    BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::QDIVrr))
            .addReg(SI.getOperand(1).getReg())
            .addReg(SI.getOperand(2).getReg())
            .addImm(P2::ALWAYS);
    BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::GETQY), SI.getOperand(0).getReg())
            .addReg(P2::QY)
            .addImm(P2::ALWAYS)
            .addImm(P2::NOEFF);

    SI.eraseFromParent();
}

/*
 * eventually we should have an operand in InstrInfo that will automatically convert any immediate to
 * aug the top 23 bits, then mov the lower 9. TBD how to do that. we will still need something like this
 * for global symbols where we don't know the value until linking, so we should always have an AUG instruction
 */
void P2ExpandPseudos::expand_MOVri32(MachineFunction &MF, MachineBasicBlock::iterator SII) {
    MachineInstr &SI = *SII;

    LLVM_DEBUG(errs()<<"== lower pseudo mov i32imm\n");
    LLVM_DEBUG(errs() << "operand type = " << (unsigned)SI.getOperand(1).getType() << "\n");

    if (SI.getOperand(1).isGlobal()) {
        BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::AUGS))
            .addImm(0)  // we will encode the correct value into this later. if just printing assembly,
                        // the final optimization pass should remove this instruction (TODO)
                        // as a result, the exact printing of this instruction won't be correct
            .addImm(P2::ALWAYS);
        BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::MOVri), SI.getOperand(0).getReg())
            .addGlobalAddress(SI.getOperand(1).getGlobal())
            .addImm(P2::ALWAYS).addImm(P2::NOEFF);

    } else if (SI.getOperand(1).isJTI()) {
        BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::AUGS))
            .addImm(0)  // we will encode the correct value into this later. if just printing assembly,
                        // the final optimization pass should remove this instruction (TODO)
                        // as a result, the exact printing of this instruction won't be correct
            .addImm(P2::ALWAYS);
        BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::MOVri), SI.getOperand(0).getReg())
            .addJumpTableIndex(SI.getOperand(1).getIndex())
            .addImm(P2::ALWAYS).addImm(P2::NOEFF);

    } else {
        uint32_t imm = SI.getOperand(1).getImm();

        // expand into an AUGS for the top 23 bits of the immediate and MOVri for the lower 9 bits
        BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::AUGS))
            .addImm(imm>>9)
            .addImm(P2::ALWAYS);

        BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::MOVri), SI.getOperand(0).getReg())
            .addImm(imm&0x1ff)
            .addImm(P2::ALWAYS).addImm(P2::NOEFF);
    }

    SI.eraseFromParent();
}

/*
 * expand to
 * mov d (operand 0), f (operand 4)
 * cmp lhs (operand 1), rhs (operand 2)
 * <cond move> d (operand 0), t (operand 3) based on operand 5
 */
void P2ExpandPseudos::expand_SELECTCC(MachineFunction &MF, MachineBasicBlock::iterator SII) {
    MachineInstr &SI = *SII;

    LLVM_DEBUG(errs()<<"== lower selectcc\n");
    LLVM_DEBUG(SI.dump());

    const MachineOperand &d = SI.getOperand(0);     // always a register
    const MachineOperand &lhs = SI.getOperand(1);   // always a register
    const MachineOperand &rhs = SI.getOperand(2);   // register or immediate
    const MachineOperand &t = SI.getOperand(3);     // register or immediate
    const MachineOperand &f = SI.getOperand(4);     // register or immediate
    int cc = SI.getOperand(5).getImm();             // condition code

    unsigned movt_op = P2::MOVrr;
    unsigned movf_op = P2::MOVrr;
    unsigned cmp_op;

    if (f.isImm()) {
        movf_op = P2::MOVri;
    }

    if (t.isImm()) {
        movt_op = P2::MOVri;
    }

    switch(cc) {
        case P2::SETUEQ:
        case P2::SETUNE:
        case P2::SETULE:
        case P2::SETULT:
        case P2::SETUGT:
        case P2::SETUGE:
            if (rhs.isImm()) {
                cmp_op = P2::CMPri;
            } else {
                cmp_op = P2::CMPrr;
            }
            break;

        case P2::SETEQ:
        case P2::SETNE:
        case P2::SETLE:
        case P2::SETLT:
        case P2::SETGT:
        case P2::SETGE:
            if (rhs.isImm()) {
                cmp_op = P2::CMPSri;
            } else {
                cmp_op = P2::CMPSrr;
            }
            break;
        default:
            llvm_unreachable("unknown condition code in expand_SELECTCC for cmp");
    }

    int true_cond_imm;
    int false_cond_imm;

    switch(cc) {
        case P2::SETUEQ:
        case P2::SETEQ:
            true_cond_imm = P2::IF_Z;
            false_cond_imm = P2::IF_NZ;
            break;
        case P2::SETUNE:
        case P2::SETNE:
            true_cond_imm = P2::IF_NZ;
            false_cond_imm = P2::IF_Z;
            break;
        case P2::SETULE:
        case P2::SETLE:
            true_cond_imm = P2::IF_C;
            false_cond_imm = P2::IF_NC;
            break;
        case P2::SETULT:
        case P2::SETLT:
            true_cond_imm = P2::IF_C_OR_Z;
            false_cond_imm = P2::IF_NC_AND_NZ;
            break;
        case P2::SETUGT:
        case P2::SETGT:
            true_cond_imm = P2::IF_NC_AND_NZ;
            false_cond_imm = P2::IF_C_OR_Z;
            break;
        case P2::SETUGE:
        case P2::SETGE:
            true_cond_imm = P2::IF_NC;
            false_cond_imm = P2::IF_C;
            break;
        default:
            llvm_unreachable("unknown condition code in expand_SELECTCC for move");
    }

    // mov false into the destination
    auto builder = BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(cmp_op), P2::SW)
            .addReg(lhs.getReg());
    builder.getInstr()->addOperand(rhs);
    builder.addImm(P2::ALWAYS).addImm(P2::WCZ);

    builder = BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(movt_op), d.getReg());
    builder.getInstr()->addOperand(t);
    builder.addImm(true_cond_imm).addImm(P2::NOEFF);

    builder = BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(movf_op))
            .addReg(d.getReg());
    builder.getInstr()->addOperand(f);
    builder.addImm(false_cond_imm).addImm(P2::NOEFF);

    SI.eraseFromParent();
}

bool P2ExpandPseudos::runOnMachineFunction(MachineFunction &MF) {
    TII = TM.getInstrInfo();

    for (auto &MBB : MF) {
        MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
        while (MBBI != E) {
            MachineBasicBlock::iterator NMBBI = std::next(MBBI);
            switch (MBBI->getOpcode()) {
                case P2::QUDIV:
                    expand_QUDIV(MF, MBBI);
                    break;
                case P2::QUREM:
                    expand_QUREM(MF, MBBI);
                    break;
                case P2::MOVri32:
                    expand_MOVri32(MF, MBBI);
                    break;
                case P2::SELECTCC:
                    expand_SELECTCC(MF, MBBI);
                    break;
            }

            MBBI = NMBBI;
        }
    }

    LLVM_DEBUG(errs()<<"done with pseudo expansion\n");

    return true;
}

FunctionPass *llvm::createP2ExpandPseudosPass(P2TargetMachine &tm) {
    return new P2ExpandPseudos(tm);
}
