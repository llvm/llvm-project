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
        //void expand_QSREM(MachineFunction &MF, MachineBasicBlock::iterator SII);
        void expand_QUREM(MachineFunction &MF, MachineBasicBlock::iterator SII);
        void expand_MOVri32(MachineFunction &MF, MachineBasicBlock::iterator SII);
        void expand_SELECTCC(MachineFunction &MF, MachineBasicBlock::iterator SII, ISD::CondCode cc);
    };

    char P2ExpandPseudos::ID = 0;

} // end anonymous namespace

void P2ExpandPseudos::expand_QUDIV(MachineFunction &MF, MachineBasicBlock::iterator SII) {
    MachineInstr &SI = *SII;

    LLVM_DEBUG(errs()<<"== lower pseudo unsigned division\n");
    LLVM_DEBUG(SI.dump());

    BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::QDIVrr))
            .addReg(SI.getOperand(1).getReg())
            .addReg(SI.getOperand(2).getReg());
    BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::GETQX), SI.getOperand(0).getReg())
            .addReg(P2::QX);

    SI.eraseFromParent();
}

// void P2ExpandPseudos::expand_QSREM(MachineFunction &MF, MachineBasicBlock::iterator SII) {
//     MachineInstr &SI = *SII;

//     errs() << "WARNING: Using srem, which hasn't been updated to handle negative numbers\n";

//     LLVM_DEBUG(errs()<<"== lower pseudo signed remainder\n");
//     LLVM_DEBUG(SI.dump());

//     BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::QDIVrr))
//             .addReg(SI.getOperand(1).getReg())
//             .addReg(SI.getOperand(2).getReg());
//     BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::GETQY), SI.getOperand(0).getReg())
//             .addReg(P2::QY);

//     SI.eraseFromParent();
// }

void P2ExpandPseudos::expand_QUREM(MachineFunction &MF, MachineBasicBlock::iterator SII) {
    MachineInstr &SI = *SII;

    LLVM_DEBUG(errs()<<"== lower pseudo unsigned remainder\n");
    LLVM_DEBUG(SI.dump());

    BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::QDIVrr))
            .addReg(SI.getOperand(1).getReg())
            .addReg(SI.getOperand(2).getReg());
    BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::GETQY), SI.getOperand(0).getReg())
            .addReg(P2::QY);

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
            .addImm(0); // we will encode the correct value into this later. if just printing assembly,
                        // the final optimization pass should remove this instruction (TODO)
                        // as a result, the exact printing of this instruction won't be correct
        BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::MOVri), SI.getOperand(0).getReg())
            .addGlobalAddress(SI.getOperand(1).getGlobal());

    } else if (SI.getOperand(1).isJTI()) {
        BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::AUGS))
            .addImm(0); // we will encode the correct value into this later. if just printing assembly,
                        // the final optimization pass should remove this instruction (TODO)
                        // as a result, the exact printing of this instruction won't be correct
        BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::MOVri), SI.getOperand(0).getReg())
            .addJumpTableIndex(SI.getOperand(1).getIndex());

    } else {
        uint32_t imm = SI.getOperand(1).getImm();

        // expand into an AUGS for the top 23 bits of the immediate and MOVri for the lower 9 bits
        BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::AUGS))
            .addImm(imm>>9);

        BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(P2::MOVri), SI.getOperand(0).getReg())
            .addImm(imm&0x1ff);
    }

    SI.eraseFromParent();
}

/*
 * expand to
 * mov d (operand 0), f (operand 4)
 * cmp lhs (operand 1), rhs (operand 2)
 * <cond move> d (operand 0), t (operand 3) based on operand 5
 */
void P2ExpandPseudos::expand_SELECTCC(MachineFunction &MF, MachineBasicBlock::iterator SII, ISD::CondCode cc) {
    MachineInstr &SI = *SII;

    LLVM_DEBUG(errs()<<"== lower selectcc\n");
    LLVM_DEBUG(SI.dump());

    const MachineOperand &d = SI.getOperand(0);     // always a register
    const MachineOperand &lhs = SI.getOperand(1);   // always a register
    const MachineOperand &rhs = SI.getOperand(2);   // register or immediate
    const MachineOperand &t = SI.getOperand(3);     // register or immediate
    const MachineOperand &f = SI.getOperand(4);     // register or immediate

    unsigned movf_op = P2::MOVrr;
    unsigned mov_op;
    unsigned cmp_op;

    if (f.isImm()) {
        movf_op = P2::MOVri;
    }

    switch(cc) {
        case ISD::SETUEQ:
        case ISD::SETUNE:
        case ISD::SETULE:
        case ISD::SETULT:
        case ISD::SETUGT:
        case ISD::SETUGE:
            if (rhs.isImm()) {
                cmp_op = P2::CMPri;
            } else {
                cmp_op = P2::CMPrr;
            }
            break;

        case ISD::SETEQ:
        case ISD::SETNE:
        case ISD::SETLE:
        case ISD::SETLT:
        case ISD::SETGT:
        case ISD::SETGE:
            if (rhs.isImm()) {
                cmp_op = P2::CMPSri;
            } else {
                cmp_op = P2::CMPSrr;
            }
            break;
        default:
            llvm_unreachable("unknown condition code in expand_SELECTCC for compare");
    }

    switch(cc) {
        case ISD::SETUEQ:
        case ISD::SETEQ:
            if (t.isImm()) {
                mov_op = P2::MOVeqri;
            } else {
                mov_op = P2::MOVeqrr;
            }
            break;
        case ISD::SETUNE:
        case ISD::SETNE:
            if (t.isImm()) {
                mov_op = P2::MOVneri;
            } else {
                mov_op = P2::MOVnerr;
            }
            break;
        case ISD::SETULE:
        case ISD::SETLE:
            if (t.isImm()) {
                mov_op = P2::MOVlteri;
            } else {
                mov_op = P2::MOVlterr;
            }
            break;
        case ISD::SETULT:
        case ISD::SETLT:
            if (t.isImm()) {
                mov_op = P2::MOVltri;
            } else {
                mov_op = P2::MOVltrr;
            }
            break;
        case ISD::SETUGT:
        case ISD::SETGT:
            if (t.isImm()) {
                mov_op = P2::MOVgtri;
            } else {
                mov_op = P2::MOVgtrr;
            }
            break;
        case ISD::SETUGE:
        case ISD::SETGE:
            if (t.isImm()) {
                mov_op = P2::MOVgteri;
            } else {
                mov_op = P2::MOVgterr;
            }
            break;
        default:
            llvm_unreachable("unknown condition code in expand_SELECTCC for move");
    }

    // mov false into the destination
    BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(movf_op), d.getReg())
            .getInstr()->addOperand(f);
    BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(cmp_op), P2::SW)
            .addReg(lhs.getReg())
            .getInstr()->addOperand(rhs);
    BuildMI(*SI.getParent(), SI, SI.getDebugLoc(), TII->get(mov_op))
            .addReg(d.getReg())
            .getInstr()->addOperand(t);

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

                // FIXME: There might be a smarter way to do this than list out 8x12 cases...
                // unsigned equal
                case P2::SELECTueqrrr:
                case P2::SELECTueqrri:
                case P2::SELECTueqrir:
                case P2::SELECTueqrii:
                case P2::SELECTueqirr:
                case P2::SELECTueqiri:
                case P2::SELECTueqiir:
                case P2::SELECTueqiii:
                    expand_SELECTCC(MF, MBBI, ISD::SETUEQ);
                    break;
                // unsigned not equal
                case P2::SELECTunerrr:
                case P2::SELECTunerri:
                case P2::SELECTunerir:
                case P2::SELECTunerii:
                case P2::SELECTuneirr:
                case P2::SELECTuneiri:
                case P2::SELECTuneiir:
                case P2::SELECTuneiii:
                    expand_SELECTCC(MF, MBBI, ISD::SETUNE);
                    break;
                // unsigned less than or equal
                case P2::SELECTulerrr:
                case P2::SELECTulerri:
                case P2::SELECTulerir:
                case P2::SELECTulerii:
                case P2::SELECTuleirr:
                case P2::SELECTuleiri:
                case P2::SELECTuleiir:
                case P2::SELECTuleiii:
                    expand_SELECTCC(MF, MBBI, ISD::SETULE);
                    break;

                // unsigned less than
                case P2::SELECTultrrr:
                case P2::SELECTultrri:
                case P2::SELECTultrir:
                case P2::SELECTultrii:
                case P2::SELECTultirr:
                case P2::SELECTultiri:
                case P2::SELECTultiir:
                case P2::SELECTultiii:
                    expand_SELECTCC(MF, MBBI, ISD::SETULT);
                    break;

                // unsigned greater than
                case P2::SELECTugtrrr:
                case P2::SELECTugtrri:
                case P2::SELECTugtrir:
                case P2::SELECTugtrii:
                case P2::SELECTugtirr:
                case P2::SELECTugtiri:
                case P2::SELECTugtiir:
                case P2::SELECTugtiii:
                    expand_SELECTCC(MF, MBBI, ISD::SETUGT);
                    break;

                // unsigned greater than or equal
                case P2::SELECTugerrr:
                case P2::SELECTugerri:
                case P2::SELECTugerir:
                case P2::SELECTugerii:
                case P2::SELECTugeirr:
                case P2::SELECTugeiri:
                case P2::SELECTugeiir:
                case P2::SELECTugeiii:
                    expand_SELECTCC(MF, MBBI, ISD::SETUGE);
                    break;

                // signed equal
                case P2::SELECTeqrrr:
                case P2::SELECTeqrri:
                case P2::SELECTeqrir:
                case P2::SELECTeqrii:
                case P2::SELECTeqirr:
                case P2::SELECTeqiri:
                case P2::SELECTeqiir:
                case P2::SELECTeqiii:
                    expand_SELECTCC(MF, MBBI, ISD::SETEQ);
                    break;

                // signed not equal
                case P2::SELECTnerrr:
                case P2::SELECTnerri:
                case P2::SELECTnerir:
                case P2::SELECTnerii:
                case P2::SELECTneirr:
                case P2::SELECTneiri:
                case P2::SELECTneiir:
                case P2::SELECTneiii:
                    expand_SELECTCC(MF, MBBI, ISD::SETNE);
                    break;

                // unsigned less than or equal
                case P2::SELECTlerrr:
                case P2::SELECTlerri:
                case P2::SELECTlerir:
                case P2::SELECTlerii:
                case P2::SELECTleirr:
                case P2::SELECTleiri:
                case P2::SELECTleiir:
                case P2::SELECTleiii:
                    expand_SELECTCC(MF, MBBI, ISD::SETLE);
                    break;

                // unsigned less than
                case P2::SELECTltrrr:
                case P2::SELECTltrri:
                case P2::SELECTltrir:
                case P2::SELECTltrii:
                case P2::SELECTltirr:
                case P2::SELECTltiri:
                case P2::SELECTltiir:
                case P2::SELECTltiii:
                    expand_SELECTCC(MF, MBBI, ISD::SETLT);
                    break;

                // unsigned greater than
                case P2::SELECTgtrrr:
                case P2::SELECTgtrri:
                case P2::SELECTgtrir:
                case P2::SELECTgtrii:
                case P2::SELECTgtirr:
                case P2::SELECTgtiri:
                case P2::SELECTgtiir:
                case P2::SELECTgtiii:
                    expand_SELECTCC(MF, MBBI, ISD::SETGT);
                    break;

                // unsigned greater than or equal
                case P2::SELECTgerrr:
                case P2::SELECTgerri:
                case P2::SELECTgerir:
                case P2::SELECTgerii:
                case P2::SELECTgeirr:
                case P2::SELECTgeiri:
                case P2::SELECTgeiir:
                case P2::SELECTgeiii:
                    expand_SELECTCC(MF, MBBI, ISD::SETGE);
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
