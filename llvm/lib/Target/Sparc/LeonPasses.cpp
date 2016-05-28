//===------ LeonPasses.cpp - Define passes specific to LEON ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "LeonPasses.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
using namespace llvm;

LEONMachineFunctionPass::LEONMachineFunctionPass(TargetMachine &tm, char& ID) :
  MachineFunctionPass(ID)
{
}

LEONMachineFunctionPass::LEONMachineFunctionPass(char& ID) :
  MachineFunctionPass(ID)
{
}

//*****************************************************************************
//**** InsertNOPLoad pass
//*****************************************************************************
//This pass inserts a NOP after any LD or LDF instruction.
//
char InsertNOPLoad::ID = 0;

InsertNOPLoad::InsertNOPLoad(TargetMachine &tm) :
                    LEONMachineFunctionPass(tm, ID)
{
}

bool InsertNOPLoad::runOnMachineFunction(MachineFunction& MF)
{
  Subtarget = &MF.getSubtarget<SparcSubtarget>();
  const TargetInstrInfo& TII = *Subtarget->getInstrInfo();
  DebugLoc DL = DebugLoc();

  bool Modified = false;
  for (auto MFI = MF.begin(), E = MF.end(); MFI != E; ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E; ++ MBBI) {
      MachineInstr &MI = *MBBI;
      unsigned Opcode = MI.getOpcode();
      if (Opcode >=  SP::LDDArr && Opcode <= SP::LDrr) {
        //errs() << "Inserting NOP after LD instruction\n";
        MachineBasicBlock::iterator NMBBI = std::next(MBBI);
        BuildMI(MBB, NMBBI, DL, TII.get(SP::NOP));
        Modified = true;
      }
      else if (MI.isInlineAsm()) {
        // Look for an inline ld or ldf instruction.
        StringRef AsmString =
            MI.getOperand(InlineAsm::MIOp_AsmString).getSymbolName();
        if (AsmString.startswith_lower("ld")) {
          //errs() << "Inserting NOP after LD instruction\n";
          MachineBasicBlock::iterator NMBBI = std::next(MBBI);
          BuildMI(MBB, NMBBI, DL, TII.get(SP::NOP));
          Modified = true;
        }
      }
    }
  }

  return Modified;
}
