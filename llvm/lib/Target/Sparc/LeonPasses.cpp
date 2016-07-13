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
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

LEONMachineFunctionPass::LEONMachineFunctionPass(TargetMachine &tm, char &ID)
    : MachineFunctionPass(ID) {}

LEONMachineFunctionPass::LEONMachineFunctionPass(char &ID)
    : MachineFunctionPass(ID) {}

int LEONMachineFunctionPass::GetRegIndexForOperand(MachineInstr &MI,
                                                   int OperandIndex) {
  if (MI.getNumOperands() > 0) {
    if (OperandIndex == LAST_OPERAND) {
      OperandIndex = MI.getNumOperands() - 1;
    }

    if (MI.getNumOperands() > (unsigned)OperandIndex &&
        MI.getOperand(OperandIndex).isReg()) {
      return (int)MI.getOperand(OperandIndex).getReg();
    }
  }

  static int NotFoundIndex = -10;
  // Return a different number each time to avoid any comparisons between the
  // values returned.
  NotFoundIndex -= 10;
  return NotFoundIndex;
}

// finds a new free FP register
// checks also the AllocatedRegisters vector
int LEONMachineFunctionPass::getUnusedFPRegister(MachineRegisterInfo &MRI) {
  for (int RegisterIndex = SP::F0; RegisterIndex <= SP::F31; ++RegisterIndex) {
    if (!MRI.isPhysRegUsed(RegisterIndex) &&
        !(std::find(UsedRegisters.begin(), UsedRegisters.end(),
                    RegisterIndex) != UsedRegisters.end())) {
      return RegisterIndex;
    }
  }

  return -1;
}

//*****************************************************************************
//**** InsertNOPLoad pass
//*****************************************************************************
// This pass fixes the incorrectly working Load instructions that exists for
// some earlier versions of the LEON processor line. NOP instructions must
// be inserted after the load instruction to ensure that the Load instruction
// behaves as expected for these processors.
//
// This pass inserts a NOP after any LD or LDF instruction.
//
char InsertNOPLoad::ID = 0;

InsertNOPLoad::InsertNOPLoad(TargetMachine &tm)
    : LEONMachineFunctionPass(tm, ID) {}

bool InsertNOPLoad::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<SparcSubtarget>();
  const TargetInstrInfo &TII = *Subtarget->getInstrInfo();
  DebugLoc DL = DebugLoc();

  bool Modified = false;
  for (auto MFI = MF.begin(), E = MF.end(); MFI != E; ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E; ++MBBI) {
      MachineInstr &MI = *MBBI;
      unsigned Opcode = MI.getOpcode();
      if (Opcode >= SP::LDDArr && Opcode <= SP::LDrr) {
        MachineBasicBlock::iterator NMBBI = std::next(MBBI);
        BuildMI(MBB, NMBBI, DL, TII.get(SP::NOP));
        Modified = true;
      } else if (MI.isInlineAsm()) {
        // Look for an inline ld or ldf instruction.
        StringRef AsmString =
            MI.getOperand(InlineAsm::MIOp_AsmString).getSymbolName();
        if (AsmString.startswith_lower("ld")) {
          MachineBasicBlock::iterator NMBBI = std::next(MBBI);
          BuildMI(MBB, NMBBI, DL, TII.get(SP::NOP));
          Modified = true;
        }
      }
    }
  }

  return Modified;
}

//*****************************************************************************
//**** FixFSMULD pass
//*****************************************************************************
// This pass fixes the incorrectly working FSMULD instruction that exists for
// some earlier versions of the LEON processor line.
//
// The pass should convert the FSMULD operands to double precision in scratch
// registers, then calculate the result with the FMULD instruction. Therefore,
// the pass should replace operations of the form:
// fsmuld %f20,%f21,%f8
// with the sequence:
// fstod %f20,%f0
// fstod %f21,%f2
// fmuld %f0,%f2,%f8
//
char FixFSMULD::ID = 0;

FixFSMULD::FixFSMULD(TargetMachine &tm) : LEONMachineFunctionPass(tm, ID) {}

bool FixFSMULD::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<SparcSubtarget>();
  const TargetInstrInfo &TII = *Subtarget->getInstrInfo();
  DebugLoc DL = DebugLoc();

  bool Modified = false;
  for (auto MFI = MF.begin(), E = MF.end(); MFI != E; ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E; ++MBBI) {

      MachineInstr &MI = *MBBI;
      unsigned Opcode = MI.getOpcode();

      const int UNASSIGNED_INDEX = -1;
      int Reg1Index = UNASSIGNED_INDEX;
      int Reg2Index = UNASSIGNED_INDEX;
      int Reg3Index = UNASSIGNED_INDEX;

      if (Opcode == SP::FSMULD && MI.getNumOperands() == 3) {
        // take the registers from fsmuld %f20,%f21,%f8
        Reg1Index = MI.getOperand(0).getReg();
        Reg2Index = MI.getOperand(1).getReg();
        Reg3Index = MI.getOperand(2).getReg();
      } else if (MI.isInlineAsm()) {
        StringRef AsmString =
            MI.getOperand(InlineAsm::MIOp_AsmString).getSymbolName();
        if (AsmString.startswith_lower("fsmuld")) {
          // this is an inline FSMULD instruction

          unsigned StartOp = InlineAsm::MIOp_FirstOperand;

          // extracts the registers from the inline assembly instruction
          for (unsigned i = StartOp, e = MI.getNumOperands(); i != e; ++i) {
            const MachineOperand &MO = MI.getOperand(i);
            if (MO.isReg()) {
              if (Reg1Index == UNASSIGNED_INDEX)
                Reg1Index = MO.getReg();
              else if (Reg2Index == UNASSIGNED_INDEX)
                Reg2Index = MO.getReg();
              else if (Reg3Index == UNASSIGNED_INDEX)
                Reg3Index = MO.getReg();
            }
            if (Reg3Index != UNASSIGNED_INDEX)
              break;
          }
        }
      }

      if (Reg1Index != UNASSIGNED_INDEX && Reg2Index != UNASSIGNED_INDEX &&
          Reg3Index != UNASSIGNED_INDEX) {
        clearUsedRegisterList();
        MachineBasicBlock::iterator NMBBI = std::next(MBBI);
        // Whatever Reg3Index is hasn't been used yet, so we need to reserve it.
        markRegisterUsed(Reg3Index);
        const int ScratchReg1Index = getUnusedFPRegister(MF.getRegInfo());
        markRegisterUsed(ScratchReg1Index);
        const int ScratchReg2Index = getUnusedFPRegister(MF.getRegInfo());
        markRegisterUsed(ScratchReg2Index);

        if (ScratchReg1Index == UNASSIGNED_INDEX ||
            ScratchReg2Index == UNASSIGNED_INDEX) {
          errs() << "Cannot allocate free scratch registers for the FixFSMULD "
                    "pass."
                 << "\n";
        } else {
          // create fstod %f20,%f0
          BuildMI(MBB, MBBI, DL, TII.get(SP::FSTOD))
              .addReg(ScratchReg1Index)
              .addReg(Reg1Index);

          // create fstod %f21,%f2
          BuildMI(MBB, MBBI, DL, TII.get(SP::FSTOD))
              .addReg(ScratchReg2Index)
              .addReg(Reg2Index);

          // create fmuld %f0,%f2,%f8
          BuildMI(MBB, MBBI, DL, TII.get(SP::FMULD))
              .addReg(Reg3Index)
              .addReg(ScratchReg1Index)
              .addReg(ScratchReg2Index);

          MI.eraseFromParent();
          MBBI = NMBBI;

          Modified = true;
        }
      }
    }
  }

  return Modified;
}

//*****************************************************************************
//**** ReplaceFMULS pass
//*****************************************************************************
// This pass fixes the incorrectly working FMULS instruction that exists for
// some earlier versions of the LEON processor line.
//
// This pass converts the FMULS operands to double precision in scratch
// registers, then calculates the result with the FMULD instruction.
// The pass should replace operations of the form:
// fmuls %f20,%f21,%f8
// with the sequence:
// fstod %f20,%f0
// fstod %f21,%f2
// fmuld %f0,%f2,%f8
//
char ReplaceFMULS::ID = 0;

ReplaceFMULS::ReplaceFMULS(TargetMachine &tm)
    : LEONMachineFunctionPass(tm, ID) {}

bool ReplaceFMULS::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<SparcSubtarget>();
  const TargetInstrInfo &TII = *Subtarget->getInstrInfo();
  DebugLoc DL = DebugLoc();

  bool Modified = false;
  for (auto MFI = MF.begin(), E = MF.end(); MFI != E; ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E; ++MBBI) {
      MachineInstr &MI = *MBBI;
      unsigned Opcode = MI.getOpcode();

      const int UNASSIGNED_INDEX = -1;
      int Reg1Index = UNASSIGNED_INDEX;
      int Reg2Index = UNASSIGNED_INDEX;
      int Reg3Index = UNASSIGNED_INDEX;

      if (Opcode == SP::FMULS && MI.getNumOperands() == 3) {
        // take the registers from fmuls %f20,%f21,%f8
        Reg1Index = MI.getOperand(0).getReg();
        Reg2Index = MI.getOperand(1).getReg();
        Reg3Index = MI.getOperand(2).getReg();
      } else if (MI.isInlineAsm()) {
        StringRef AsmString =
            MI.getOperand(InlineAsm::MIOp_AsmString).getSymbolName();
        if (AsmString.startswith_lower("fmuls")) {
          // this is an inline FMULS instruction
          unsigned StartOp = InlineAsm::MIOp_FirstOperand;

          // extracts the registers from the inline assembly instruction
          for (unsigned i = StartOp, e = MI.getNumOperands(); i != e; ++i) {
            const MachineOperand &MO = MI.getOperand(i);
            if (MO.isReg()) {
              if (Reg1Index == UNASSIGNED_INDEX)
                Reg1Index = MO.getReg();
              else if (Reg2Index == UNASSIGNED_INDEX)
                Reg2Index = MO.getReg();
              else if (Reg3Index == UNASSIGNED_INDEX)
                Reg3Index = MO.getReg();
            }
            if (Reg3Index != UNASSIGNED_INDEX)
              break;
          }
        }
      }

      if (Reg1Index != UNASSIGNED_INDEX && Reg2Index != UNASSIGNED_INDEX &&
          Reg3Index != UNASSIGNED_INDEX) {
        clearUsedRegisterList();
        MachineBasicBlock::iterator NMBBI = std::next(MBBI);
        // Whatever Reg3Index is hasn't been used yet, so we need to reserve it.
        markRegisterUsed(Reg3Index);
        const int ScratchReg1Index = getUnusedFPRegister(MF.getRegInfo());
        markRegisterUsed(ScratchReg1Index);
        const int ScratchReg2Index = getUnusedFPRegister(MF.getRegInfo());
        markRegisterUsed(ScratchReg2Index);

        if (ScratchReg1Index == UNASSIGNED_INDEX ||
            ScratchReg2Index == UNASSIGNED_INDEX) {
          errs() << "Cannot allocate free scratch registers for the "
                    "ReplaceFMULS pass."
                 << "\n";
        } else {
          // create fstod %f20,%f0
          BuildMI(MBB, MBBI, DL, TII.get(SP::FSTOD))
              .addReg(ScratchReg1Index)
              .addReg(Reg1Index);

          // create fstod %f21,%f2
          BuildMI(MBB, MBBI, DL, TII.get(SP::FSTOD))
              .addReg(ScratchReg2Index)
              .addReg(Reg2Index);

          // create fmuld %f0,%f2,%f8
          BuildMI(MBB, MBBI, DL, TII.get(SP::FMULD))
              .addReg(Reg3Index)
              .addReg(ScratchReg1Index)
              .addReg(ScratchReg2Index);

          MI.eraseFromParent();
          MBBI = NMBBI;

          Modified = true;
        }
      }
    }
  }

  return Modified;
}

//*****************************************************************************
//**** FixAllFDIVSQRT pass
//*****************************************************************************
// This pass fixes the incorrectly working FDIVx and FSQRTx instructions that
// exist for some earlier versions of the LEON processor line. Five NOP
// instructions need to be inserted after these instructions to ensure the
// correct result is placed in the destination registers before they are used.
//
// This pass implements two fixes:
//  1) fixing the FSQRTS and FSQRTD instructions.
//  2) fixing the FDIVS and FDIVD instructions.
//
// FSQRTS and FDIVS are converted to FDIVD and FSQRTD respectively earlier in
// the pipeline when this option is enabled, so this pass needs only to deal
// with the changes that still need implementing for the "double" versions
// of these instructions.
//
char FixAllFDIVSQRT::ID = 0;

FixAllFDIVSQRT::FixAllFDIVSQRT(TargetMachine &tm)
    : LEONMachineFunctionPass(tm, ID) {}

bool FixAllFDIVSQRT::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<SparcSubtarget>();
  const TargetInstrInfo &TII = *Subtarget->getInstrInfo();
  DebugLoc DL = DebugLoc();

  bool Modified = false;
  for (auto MFI = MF.begin(), E = MF.end(); MFI != E; ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E; ++MBBI) {
      MachineInstr &MI = *MBBI;
      unsigned Opcode = MI.getOpcode();

      if (MI.isInlineAsm()) {
        StringRef AsmString =
            MI.getOperand(InlineAsm::MIOp_AsmString).getSymbolName();
        if (AsmString.startswith_lower("fsqrtd")) {
          // this is an inline fsqrts instruction
          Opcode = SP::FSQRTD;
        } else if (AsmString.startswith_lower("fdivd")) {
          // this is an inline fsqrts instruction
          Opcode = SP::FDIVD;
        }
      }

      // Note: FDIVS and FSQRTS cannot be generated when this erratum fix is
      // switched on so we don't need to check for them here. They will
      // already have been converted to FSQRTD or FDIVD earlier in the
      // pipeline.
      if (Opcode == SP::FSQRTD || Opcode == SP::FDIVD) {
        // Insert 5 NOPs before FSQRTD,FDIVD.
        for (int InsertedCount = 0; InsertedCount < 5; InsertedCount++)
          BuildMI(MBB, MBBI, DL, TII.get(SP::NOP));

        MachineBasicBlock::iterator NMBBI = std::next(MBBI);
        // ... and inserting 28 NOPs after FSQRTD,FDIVD.
        for (int InsertedCount = 0; InsertedCount < 28; InsertedCount++)
          BuildMI(MBB, NMBBI, DL, TII.get(SP::NOP));

        Modified = true;
      }
    }
  }

  return Modified;
}

//*****************************************************************************
//**** ReplaceSDIV pass
//*****************************************************************************
// This pass fixes the incorrectly working SDIV instruction that
// exist for some earlier versions of the LEON processor line. The instruction
// is replaced with an SDIVcc instruction instead, which is working.
//
char ReplaceSDIV::ID = 0;

ReplaceSDIV::ReplaceSDIV() : LEONMachineFunctionPass(ID) {}

ReplaceSDIV::ReplaceSDIV(TargetMachine &tm) : LEONMachineFunctionPass(tm, ID) {}

bool ReplaceSDIV::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<SparcSubtarget>();
  const TargetInstrInfo &TII = *Subtarget->getInstrInfo();

  bool Modified = false;
  for (auto MFI = MF.begin(), E = MF.end(); MFI != E; ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E; ++MBBI) {
      MachineInstr &MI = *MBBI;
      unsigned Opcode = MI.getOpcode();
      if (Opcode == SP::SDIVrr) {
        MI.setDesc(TII.get(SP::SDIVCCrr));
        Modified = true;
      } else if (Opcode == SP::SDIVri) {
        MI.setDesc(TII.get(SP::SDIVCCri));
        Modified = true;
      }
    }
  }

  return Modified;
}

static RegisterPass<ReplaceSDIV> X("replace-sdiv", "Replase SDIV Pass", false,
                                   false);

//*****************************************************************************
//**** FixCALL pass
//*****************************************************************************
// This pass restricts the size of the immediate operand of the CALL
// instruction, which can cause problems on some earlier versions of the LEON
// processor, which can interpret some of the call address bits incorrectly.
//
char FixCALL::ID = 0;

FixCALL::FixCALL(TargetMachine &tm) : LEONMachineFunctionPass(tm, ID) {}

bool FixCALL::runOnMachineFunction(MachineFunction &MF) {
  bool Modified = false;

  for (auto MFI = MF.begin(), E = MF.end(); MFI != E; ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E; ++MBBI) {
      MachineInstr &MI = *MBBI;
      MI.print(errs());
      errs() << "\n";

      unsigned Opcode = MI.getOpcode();
      if (Opcode == SP::CALL || Opcode == SP::CALLrr) {
        unsigned NumOperands = MI.getNumOperands();
        for (unsigned OperandIndex = 0; OperandIndex < NumOperands;
             OperandIndex++) {
          MachineOperand &MO = MI.getOperand(OperandIndex);
          if (MO.isImm()) {
            int64_t Value = MO.getImm();
            MO.setImm(Value & 0x000fffffL);
            Modified = true;
            break;
          }
        }
      } else if (MI.isInlineAsm()) // inline assembly immediate call
      {
        StringRef AsmString =
            MI.getOperand(InlineAsm::MIOp_AsmString).getSymbolName();
        if (AsmString.startswith_lower("call")) {
          // this is an inline call instruction
          unsigned StartOp = InlineAsm::MIOp_FirstOperand;

          // extracts the registers from the inline assembly instruction
          for (unsigned i = StartOp, e = MI.getNumOperands(); i != e; ++i) {
            MachineOperand &MO = MI.getOperand(i);
            if (MO.isImm()) {
              int64_t Value = MO.getImm();
              MO.setImm(Value & 0x000fffffL);
              Modified = true;
            }
          }
        }
      }
    }
  }

  return Modified;
}

//*****************************************************************************
//**** IgnoreZeroFlag pass
//*****************************************************************************
// This erratum fix fixes the overflow behavior of SDIVCC and UDIVCC
// instructions that exists on some earlier LEON processors. Where these
// instructions are detected, they are replaced by a sequence that will
// explicitly write the overflow bit flag if this is required.
//
char IgnoreZeroFlag::ID = 0;

IgnoreZeroFlag::IgnoreZeroFlag(TargetMachine &tm)
    : LEONMachineFunctionPass(tm, ID) {}

bool IgnoreZeroFlag::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<SparcSubtarget>();
  const TargetInstrInfo &TII = *Subtarget->getInstrInfo();
  DebugLoc DL = DebugLoc();

  bool Modified = false;
  for (auto MFI = MF.begin(), E = MF.end(); MFI != E; ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E; ++MBBI) {
      MachineInstr &MI = *MBBI;
      unsigned Opcode = MI.getOpcode();
      if (Opcode == SP::SDIVCCrr || Opcode == SP::SDIVCCri ||
          Opcode == SP::UDIVCCrr || Opcode == SP::UDIVCCri) {

        // split the current machine basic block - just after the sdivcc/udivcc
        // instruction
        // create a label that help us skip the zero flag update (of PSR -
        // Processor Status Register)
        // if conditions are not met
        const BasicBlock *LLVM_BB = MBB.getBasicBlock();
        MachineFunction::iterator It =
            std::next(MachineFunction::iterator(MBB));

        MachineBasicBlock *dneBB = MF.CreateMachineBasicBlock(LLVM_BB);
        MF.insert(It, dneBB);

        // Transfer the remainder of MBB and its successor edges to dneBB.
        dneBB->splice(dneBB->begin(), &MBB,
                      std::next(MachineBasicBlock::iterator(MI)), MBB.end());
        dneBB->transferSuccessorsAndUpdatePHIs(&MBB);

        MBB.addSuccessor(dneBB);

        MachineBasicBlock::iterator NextMBBI = std::next(MBBI);

        // bvc - branch if overflow flag not set
        BuildMI(MBB, NextMBBI, DL, TII.get(SP::BCOND))
            .addMBB(dneBB)
            .addImm(SPCC::ICC_VS);

        // bnz - branch if not zero
        BuildMI(MBB, NextMBBI, DL, TII.get(SP::BCOND))
            .addMBB(dneBB)
            .addImm(SPCC::ICC_NE);

        // use the WRPSR (Write Processor State Register) instruction to set the
        // zeo flag to 1
        // create wr %g0, 1, %psr
        BuildMI(MBB, NextMBBI, DL, TII.get(SP::WRPSRri))
            .addReg(SP::G0)
            .addImm(1);

        BuildMI(MBB, NextMBBI, DL, TII.get(SP::NOP));

        Modified = true;
      } else if (MI.isInlineAsm()) {
        StringRef AsmString =
            MI.getOperand(InlineAsm::MIOp_AsmString).getSymbolName();
        if (AsmString.startswith_lower("sdivcc") ||
            AsmString.startswith_lower("udivcc")) {
          // this is an inline SDIVCC or UDIVCC instruction

          // split the current machine basic block - just after the
          // sdivcc/udivcc instruction
          // create a label that help us skip the zero flag update (of PSR -
          // Processor Status Register)
          // if conditions are not met
          const BasicBlock *LLVM_BB = MBB.getBasicBlock();
          MachineFunction::iterator It =
              std::next(MachineFunction::iterator(MBB));

          MachineBasicBlock *dneBB = MF.CreateMachineBasicBlock(LLVM_BB);
          MF.insert(It, dneBB);

          // Transfer the remainder of MBB and its successor edges to dneBB.
          dneBB->splice(dneBB->begin(), &MBB,
                        std::next(MachineBasicBlock::iterator(MI)), MBB.end());
          dneBB->transferSuccessorsAndUpdatePHIs(&MBB);

          MBB.addSuccessor(dneBB);

          MachineBasicBlock::iterator NextMBBI = std::next(MBBI);

          // bvc - branch if overflow flag not set
          BuildMI(MBB, NextMBBI, DL, TII.get(SP::BCOND))
              .addMBB(dneBB)
              .addImm(SPCC::ICC_VS);

          // bnz - branch if not zero
          BuildMI(MBB, NextMBBI, DL, TII.get(SP::BCOND))
              .addMBB(dneBB)
              .addImm(SPCC::ICC_NE);

          // use the WRPSR (Write Processor State Register) instruction to set
          // the zeo flag to 1
          // create wr %g0, 1, %psr
          BuildMI(MBB, NextMBBI, DL, TII.get(SP::WRPSRri))
              .addReg(SP::G0)
              .addImm(1);

          BuildMI(MBB, NextMBBI, DL, TII.get(SP::NOP));

          Modified = true;
        }
      }
    }
  }

  return Modified;
}

//*****************************************************************************
//**** InsertNOPDoublePrecision pass
//*****************************************************************************
// This erratum fix for some earlier LEON processors fixes a problem where a
// double precision load will not yield the correct result if used in FMUL,
// FDIV, FADD, FSUB or FSQRT instructions later. If this sequence is detected,
// inserting a NOP between the two instructions will fix the erratum.
// 1.scans the code after register allocation;
// 2.checks for the problem conditions as described in the AT697E erratum
// “Odd-Numbered FPU Register Dependency not Properly Checked in some
// Double-Precision FPU Operations”;
// 3.inserts NOPs if the problem exists.
//
char InsertNOPDoublePrecision::ID = 0;

InsertNOPDoublePrecision::InsertNOPDoublePrecision(TargetMachine &tm)
    : LEONMachineFunctionPass(tm, ID) {}

bool InsertNOPDoublePrecision::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<SparcSubtarget>();
  const TargetInstrInfo &TII = *Subtarget->getInstrInfo();
  DebugLoc DL = DebugLoc();

  bool Modified = false;
  for (auto MFI = MF.begin(), E = MF.end(); MFI != E; ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E; ++MBBI) {
      MachineInstr &MI = *MBBI;
      unsigned Opcode = MI.getOpcode();
      if (Opcode == SP::LDDFri || Opcode == SP::LDDFrr) {
        MachineBasicBlock::iterator NMBBI = std::next(MBBI);
        MachineInstr &NMI = *NMBBI;

        unsigned NextOpcode = NMI.getOpcode();
        // NMI.print(errs());
        if (NextOpcode == SP::FADDD || NextOpcode == SP::FSUBD ||
            NextOpcode == SP::FMULD || NextOpcode == SP::FDIVD) {
          int RegAIndex = GetRegIndexForOperand(MI, 0);
          int RegBIndex = GetRegIndexForOperand(NMI, 0);
          int RegCIndex =
              GetRegIndexForOperand(NMI, 2); // Second source operand is index 2
          int RegDIndex =
              GetRegIndexForOperand(NMI, 1); // Destination operand is index 1

          if ((RegAIndex == RegBIndex + 1 && RegBIndex == RegDIndex) ||
              (RegAIndex == RegCIndex + 1 && RegCIndex == RegDIndex) ||
              (RegAIndex == RegBIndex + 1 && RegCIndex == RegDIndex) ||
              (RegAIndex == RegCIndex + 1 && RegBIndex == RegDIndex)) {
            // Insert NOP between the two instructions.
            BuildMI(MBB, NMBBI, DL, TII.get(SP::NOP));
            Modified = true;
          }

          // Check the errata patterns that only happen for FADDD and FMULD
          if (Modified == false &&
              (NextOpcode == SP::FADDD || NextOpcode == SP::FMULD)) {
            RegAIndex = GetRegIndexForOperand(MI, 1);
            if (RegAIndex == RegBIndex + 1 && RegBIndex == RegCIndex &&
                RegBIndex == RegDIndex) {
              // Insert NOP between the two instructions.
              BuildMI(MBB, NMBBI, DL, TII.get(SP::NOP));
              Modified = true;
            }
          }
        } else if (NextOpcode == SP::FSQRTD) {
          int RegAIndex = GetRegIndexForOperand(MI, 1);
          int RegBIndex = GetRegIndexForOperand(NMI, 0);
          int RegCIndex = GetRegIndexForOperand(NMI, 1);

          if (RegAIndex == RegBIndex + 1 && RegBIndex == RegCIndex) {
            // Insert NOP between the two instructions.
            BuildMI(MBB, NMBBI, DL, TII.get(SP::NOP));
            Modified = true;
          }
        }
      }
    }
  }

  return Modified;
}

//*****************************************************************************
//**** PreventRoundChange pass
//*****************************************************************************
// To prevent any explicit change of the default rounding mode, this pass
// detects any call of the fesetround function and removes this call from the
// list of generated operations.
//
char PreventRoundChange::ID = 0;

PreventRoundChange::PreventRoundChange(TargetMachine &tm)
    : LEONMachineFunctionPass(tm, ID) {}

bool PreventRoundChange::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<SparcSubtarget>();

  bool Modified = false;
  for (auto MFI = MF.begin(), E = MF.end(); MFI != E; ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E; ++MBBI) {
      MachineInstr &MI = *MBBI;
      unsigned Opcode = MI.getOpcode();
      if (Opcode == SP::CALL && MI.getNumOperands() > 0) {
        MachineOperand &MO = MI.getOperand(0);

        if (MO.isGlobal()) {
          StringRef FuncName = MO.getGlobal()->getName();
          if (FuncName.compare_lower("fesetround") == 0) {
            MachineBasicBlock::iterator NMBBI = std::next(MBBI);
            MI.eraseFromParent();
            MBBI = NMBBI;
            Modified = true;
          }
        }
      }
    }
  }

  return Modified;
}
//*****************************************************************************
//**** FlushCacheLineSWAP pass
//*****************************************************************************
// This pass inserts FLUSHW just before any SWAP atomic instruction.
//
char FlushCacheLineSWAP::ID = 0;

FlushCacheLineSWAP::FlushCacheLineSWAP(TargetMachine &tm)
    : LEONMachineFunctionPass(tm, ID) {}

bool FlushCacheLineSWAP::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<SparcSubtarget>();
  const TargetInstrInfo &TII = *Subtarget->getInstrInfo();
  DebugLoc DL = DebugLoc();

  bool Modified = false;
  for (auto MFI = MF.begin(), E = MF.end(); MFI != E; ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E; ++MBBI) {
      MachineInstr &MI = *MBBI;
      unsigned Opcode = MI.getOpcode();
      if (Opcode == SP::SWAPrr || Opcode == SP::SWAPri ||
          Opcode == SP::LDSTUBrr || Opcode == SP::LDSTUBri) {
        // insert flush and 5 NOPs before the swap/ldstub instruction
        BuildMI(MBB, MBBI, DL, TII.get(SP::FLUSH));
        BuildMI(MBB, MBBI, DL, TII.get(SP::NOP));
        BuildMI(MBB, MBBI, DL, TII.get(SP::NOP));
        BuildMI(MBB, MBBI, DL, TII.get(SP::NOP));
        BuildMI(MBB, MBBI, DL, TII.get(SP::NOP));
        BuildMI(MBB, MBBI, DL, TII.get(SP::NOP));

        Modified = true;
      } else if (MI.isInlineAsm()) {
        StringRef AsmString =
            MI.getOperand(InlineAsm::MIOp_AsmString).getSymbolName();
        if (AsmString.startswith_lower("swap") ||
            AsmString.startswith_lower("ldstub")) {
          // this is an inline swap or ldstub instruction

          // insert flush and 5 NOPs before the swap/ldstub instruction
          BuildMI(MBB, MBBI, DL, TII.get(SP::FLUSH));
          BuildMI(MBB, MBBI, DL, TII.get(SP::NOP));
          BuildMI(MBB, MBBI, DL, TII.get(SP::NOP));
          BuildMI(MBB, MBBI, DL, TII.get(SP::NOP));
          BuildMI(MBB, MBBI, DL, TII.get(SP::NOP));
          BuildMI(MBB, MBBI, DL, TII.get(SP::NOP));

          Modified = true;
        }
      }
    }
  }

  return Modified;
}

//*****************************************************************************
//**** InsertNOPsLoadStore pass
//*****************************************************************************
// This pass shall insert NOPs between floating point loads and stores when the
// following circumstances are present [5]:
// Pattern 1:
// 1. single-precision load or single-precision FPOP to register %fX, where X is
// the same register as the store being checked;
// 2. single-precision load or single-precision FPOP to register %fY , where Y
// is the opposite register in the same double-precision pair;
// 3. 0-3 instructions of any kind, except stores from %fX or %fY or operations
// with %fX as destination;
// 4. the store (from register %fX) being considered.
// Pattern 2:
// 1. double-precision FPOP;
// 2. any number of operations on any kind, except no double-precision FPOP and
// at most one (less than two) single-precision or single-to-double FPOPs;
// 3. the store (from register %fX) being considered.
//
char InsertNOPsLoadStore::ID = 0;

InsertNOPsLoadStore::InsertNOPsLoadStore(TargetMachine &tm)
    : LEONMachineFunctionPass(tm, ID) {}

bool InsertNOPsLoadStore::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<SparcSubtarget>();
  const TargetInstrInfo &TII = *Subtarget->getInstrInfo();
  DebugLoc DL = DebugLoc();

  MachineInstr *Pattern1FirstInstruction = NULL;
  MachineInstr *Pattern2FirstInstruction = NULL;
  unsigned int StoreInstructionsToCheck = 0;
  int FxRegIndex, FyRegIndex;

  bool Modified = false;
  for (auto MFI = MF.begin(), E = MF.end(); MFI != E; ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E; ++MBBI) {
      MachineInstr &MI = *MBBI;

      if (StoreInstructionsToCheck > 0) {
        if (((MI.getOpcode() == SP::STFrr || MI.getOpcode() == SP::STFri) &&
             (GetRegIndexForOperand(MI, LAST_OPERAND) == FxRegIndex ||
              GetRegIndexForOperand(MI, LAST_OPERAND) == FyRegIndex)) ||
            GetRegIndexForOperand(MI, 0) == FxRegIndex) {
          // Insert four NOPs
          for (unsigned InsertedCount = 0; InsertedCount < 4; InsertedCount++) {
            BuildMI(MBB, MBBI, DL, TII.get(SP::NOP));
          }
          Modified = true;
        }
        StoreInstructionsToCheck--;
      }

      switch (MI.getOpcode()) {
      // Watch for Pattern 1 FPop instructions
      case SP::LDrr:
      case SP::LDri:
      case SP::LDFrr:
      case SP::LDFri:
      case SP::FADDS:
      case SP::FSUBS:
      case SP::FMULS:
      case SP::FDIVS:
      case SP::FSQRTS:
      case SP::FCMPS:
      case SP::FMOVS:
      case SP::FNEGS:
      case SP::FABSS:
      case SP::FITOS:
      case SP::FSTOI:
      case SP::FITOD:
      case SP::FDTOI:
      case SP::FDTOS:
        if (Pattern1FirstInstruction != NULL) {
          FxRegIndex = GetRegIndexForOperand(*Pattern1FirstInstruction, 0);
          FyRegIndex = GetRegIndexForOperand(MI, 0);

          // Check to see if these registers are part of the same double
          // precision
          // register pair.
          int DoublePrecRegIndexForX = (FxRegIndex - SP::F0) / 2;
          int DoublePrecRegIndexForY = (FyRegIndex - SP::F0) / 2;

          if (DoublePrecRegIndexForX == DoublePrecRegIndexForY)
            StoreInstructionsToCheck = 4;
        }

        Pattern1FirstInstruction = &MI;
        break;
      // End of Pattern 1

      // Search for Pattern 2
      case SP::FADDD:
      case SP::FSUBD:
      case SP::FMULD:
      case SP::FDIVD:
      case SP::FSQRTD:
      case SP::FCMPD:
        Pattern2FirstInstruction = &MI;
        Pattern1FirstInstruction = NULL;
        break;

      case SP::STFrr:
      case SP::STFri:
      case SP::STDFrr:
      case SP::STDFri:
        if (Pattern2FirstInstruction != NULL) {
          if (GetRegIndexForOperand(MI, LAST_OPERAND) ==
              GetRegIndexForOperand(*Pattern2FirstInstruction, 0)) {
            // Insert four NOPs
            for (unsigned InsertedCount = 0; InsertedCount < 4;
                 InsertedCount++) {
              BuildMI(MBB, MBBI, DL, TII.get(SP::NOP));
            }

            Pattern2FirstInstruction = NULL;
          }
        }
        Pattern1FirstInstruction = NULL;
        break;
      // End of Pattern 2

      default:
        // Ensure we don't count debug-only values while we're testing for the
        // patterns.
        if (!MI.isDebugValue())
          Pattern1FirstInstruction = NULL;
        break;
      }
    }
  }

  return Modified;
}
