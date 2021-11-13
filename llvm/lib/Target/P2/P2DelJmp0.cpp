//===-- P2DelJmp0.cpp - P2 DelJmp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Simple pass to remove jumps that will result in jmp #0
//
//===----------------------------------------------------------------------===//

#include "P2.h"

#include "P2TargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"

using namespace llvm;

#define DEBUG_TYPE "del-jmp0"

STATISTIC(NumDelJmp, "Number of useless jmp deleted");

static cl::opt<bool> EnableDelJmp("enable-p2-del-jmp0",
    cl::init(true),
    cl::desc("Delete useless jmp instructions: jmp #0."),
    cl::Hidden);

namespace {
    struct DelJmp0 : public MachineFunctionPass {
        static char ID;
        DelJmp0(TargetMachine &tm) : MachineFunctionPass(ID) {}

        StringRef getPassName() const override {
            return "P2 Delete JMP #0";
        }

        bool runOnMachineBasicBlock(MachineBasicBlock &MBB, MachineBasicBlock &MBBN);
        bool runOnMachineFunction(MachineFunction &F) override {
            bool Changed = false;
            if (EnableDelJmp) {
                MachineFunction::iterator FJ = F.begin();
                if (FJ != F.end())
                    FJ++;
                if (FJ == F.end())
                    return Changed;
                for (MachineFunction::iterator FI = F.begin(), FE = F.end(); FJ != FE; ++FI, ++FJ)
                    Changed |= runOnMachineBasicBlock(*FI, *FJ);
            }
            return Changed;
        }
    };
    char DelJmp0::ID = 0;
} // end of anonymous namespace

bool DelJmp0::runOnMachineBasicBlock(MachineBasicBlock &MBB, MachineBasicBlock &MBBN) {
    bool Changed = false;

    MachineBasicBlock::iterator I = MBB.end();
    if (I != MBB.begin())
        I--;	// set I to the last instruction
    else
        return Changed;
    
    if (I->getOpcode() == P2::JMP && I->getOperand(0).getMBB() == &MBBN) {
    // I is the instruction of "jmp #offset=0", as follows,
    //     jmp	$BB0_3
    // $BB0_3:
    //     ld	$4, 28($sp)
        ++NumDelJmp;
        MBB.erase(I);	// delete the "JMP 0" instruction
        Changed = true;	// Notify LLVM kernel Changed
    }
    return Changed;
}

FunctionPass *llvm::createP2DelJmp0Pass(P2TargetMachine &tm) {
    return new DelJmp0(tm);
}