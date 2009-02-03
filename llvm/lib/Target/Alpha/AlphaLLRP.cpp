//===-- AlphaLLRP.cpp - Alpha Load Load Replay Trap elimination pass. -- --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Here we check for potential replay traps introduced by the spiller
// We also align some branch targets if we can do so for free.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "alpha-nops"
#include "Alpha.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

STATISTIC(nopintro, "Number of nops inserted");
STATISTIC(nopalign, "Number of nops inserted for alignment");

namespace {
  cl::opt<bool>
  AlignAll("alpha-align-all", cl::Hidden,
                   cl::desc("Align all blocks"));

  struct AlphaLLRPPass : public MachineFunctionPass {
    /// Target machine description which we query for reg. names, data
    /// layout, etc.
    ///
    AlphaTargetMachine &TM;

    static char ID;
    AlphaLLRPPass(AlphaTargetMachine &tm) 
      : MachineFunctionPass(&ID), TM(tm) { }

    virtual const char *getPassName() const {
      return "Alpha NOP inserter";
    }

    bool runOnMachineFunction(MachineFunction &F) {
      const TargetInstrInfo *TII = F.getTarget().getInstrInfo();
      bool Changed = false;
      MachineInstr* prev[3] = {0,0,0};
      unsigned count = 0;
      for (MachineFunction::iterator FI = F.begin(), FE = F.end();
           FI != FE; ++FI) {
        MachineBasicBlock& MBB = *FI;
        bool ub = false;
        for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ) {
          if (count%4 == 0)
            prev[0] = prev[1] = prev[2] = 0; //Slots cleared at fetch boundary
          ++count;
          MachineInstr *MI = I++;
          switch (MI->getOpcode()) {
          case Alpha::LDQ:  case Alpha::LDL:
          case Alpha::LDWU: case Alpha::LDBU:
          case Alpha::LDT: case Alpha::LDS:
          case Alpha::STQ:  case Alpha::STL:
          case Alpha::STW:  case Alpha::STB:
          case Alpha::STT: case Alpha::STS:
           if (MI->getOperand(2).getReg() == Alpha::R30) {
             if (prev[0] && 
                 prev[0]->getOperand(2).getReg() == MI->getOperand(2).getReg()&&
                 prev[0]->getOperand(1).getImm() == MI->getOperand(1).getImm()){
               prev[0] = prev[1];
               prev[1] = prev[2];
               prev[2] = 0;
               BuildMI(MBB, MI, TII->get(Alpha::BISr), Alpha::R31)
                 .addReg(Alpha::R31)
                 .addReg(Alpha::R31); 
               Changed = true; nopintro += 1;
               count += 1;
             } else if (prev[1] 
                        && prev[1]->getOperand(2).getReg() == 
                        MI->getOperand(2).getReg()
                        && prev[1]->getOperand(1).getImm() == 
                        MI->getOperand(1).getImm()) {
               prev[0] = prev[2];
               prev[1] = prev[2] = 0;
               BuildMI(MBB, MI, TII->get(Alpha::BISr), Alpha::R31)
                 .addReg(Alpha::R31)
                 .addReg(Alpha::R31); 
               BuildMI(MBB, MI, TII->get(Alpha::BISr), Alpha::R31)
                 .addReg(Alpha::R31)
                 .addReg(Alpha::R31);
               Changed = true; nopintro += 2;
               count += 2;
             } else if (prev[2] 
                        && prev[2]->getOperand(2).getReg() == 
                        MI->getOperand(2).getReg()
                        && prev[2]->getOperand(1).getImm() == 
                        MI->getOperand(1).getImm()) {
               prev[0] = prev[1] = prev[2] = 0;
               BuildMI(MBB, MI, TII->get(Alpha::BISr), Alpha::R31).addReg(Alpha::R31)
                 .addReg(Alpha::R31);
               BuildMI(MBB, MI, TII->get(Alpha::BISr), Alpha::R31).addReg(Alpha::R31)
                 .addReg(Alpha::R31);
               BuildMI(MBB, MI, TII->get(Alpha::BISr), Alpha::R31).addReg(Alpha::R31)
                 .addReg(Alpha::R31);
               Changed = true; nopintro += 3;
               count += 3;
             }
             prev[0] = prev[1];
             prev[1] = prev[2];
             prev[2] = MI;
             break;
           }
           prev[0] = prev[1];
           prev[1] = prev[2];
           prev[2] = 0;
           break;
          case Alpha::ALTENT:
          case Alpha::MEMLABEL:
          case Alpha::PCLABEL:
            --count;
            break;
          case Alpha::BR:
          case Alpha::JMP:
            ub = true;
            //fall through
          default:
            prev[0] = prev[1];
            prev[1] = prev[2];
            prev[2] = 0;
            break;
          }
        }
        if (ub || AlignAll) {
          //we can align stuff for free at this point
          while (count % 4) {
            BuildMI(MBB, MBB.end(), TII->get(Alpha::BISr), Alpha::R31)
              .addReg(Alpha::R31).addReg(Alpha::R31);
            ++count;
            ++nopalign;
            prev[0] = prev[1];
            prev[1] = prev[2];
            prev[2] = 0;
          }
        }
      }
      return Changed;
    }
  };
  char AlphaLLRPPass::ID = 0;
} // end of anonymous namespace

FunctionPass *llvm::createAlphaLLRPPass(AlphaTargetMachine &tm) {
  return new AlphaLLRPPass(tm);
}
