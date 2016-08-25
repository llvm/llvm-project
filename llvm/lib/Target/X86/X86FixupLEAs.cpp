//===-- X86FixupLEAs.cpp - use or replace LEA instructions -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass that finds instructions that can be
// re-written as LEA instructions in order to reduce pipeline delays.
// When optimizing for size it replaces suitable LEAs with INC or DEC.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
using namespace llvm;

#define DEBUG_TYPE "x86-fixup-LEAs"

STATISTIC(NumLEAs, "Number of LEA instructions created");

namespace {
class FixupLEAPass : public MachineFunctionPass {
  enum RegUsageState { RU_NotUsed, RU_Write, RU_Read };
  static char ID;
  /// \brief Loop over all of the instructions in the basic block
  /// replacing applicable instructions with LEA instructions,
  /// where appropriate.
  bool processBasicBlock(MachineFunction &MF, MachineFunction::iterator MFI);

  const char *getPassName() const override { return "X86 LEA Fixup"; }

  /// \brief Given a machine register, look for the instruction
  /// which writes it in the current basic block. If found,
  /// try to replace it with an equivalent LEA instruction.
  /// If replacement succeeds, then also process the newly created
  /// instruction.
  void seekLEAFixup(MachineOperand &p, MachineBasicBlock::iterator &I,
                    MachineFunction::iterator MFI);

  /// \brief Given a memory access or LEA instruction
  /// whose address mode uses a base and/or index register, look for
  /// an opportunity to replace the instruction which sets the base or index
  /// register with an equivalent LEA instruction.
  void processInstruction(MachineBasicBlock::iterator &I,
                          MachineFunction::iterator MFI);

  /// \brief Given a LEA instruction which is unprofitable
  /// on Silvermont try to replace it with an equivalent ADD instruction
  void processInstructionForSLM(MachineBasicBlock::iterator &I,
                                MachineFunction::iterator MFI);

  /// \brief Look for LEAs that add 1 to reg or subtract 1 from reg
  /// and convert them to INC or DEC respectively.
  bool fixupIncDec(MachineBasicBlock::iterator &I,
                   MachineFunction::iterator MFI) const;

  /// \brief Determine if an instruction references a machine register
  /// and, if so, whether it reads or writes the register.
  RegUsageState usesRegister(MachineOperand &p, MachineBasicBlock::iterator I);

  /// \brief Step backwards through a basic block, looking
  /// for an instruction which writes a register within
  /// a maximum of INSTR_DISTANCE_THRESHOLD instruction latency cycles.
  MachineBasicBlock::iterator searchBackwards(MachineOperand &p,
                                              MachineBasicBlock::iterator &I,
                                              MachineFunction::iterator MFI);

  /// \brief if an instruction can be converted to an
  /// equivalent LEA, insert the new instruction into the basic block
  /// and return a pointer to it. Otherwise, return zero.
  MachineInstr *postRAConvertToLEA(MachineFunction::iterator &MFI,
                                   MachineBasicBlock::iterator &MBBI) const;

public:
  FixupLEAPass() : MachineFunctionPass(ID) {}

  /// \brief Loop over all of the basic blocks,
  /// replacing instructions by equivalent LEA instructions
  /// if needed and when possible.
  bool runOnMachineFunction(MachineFunction &MF) override;

  // This pass runs after regalloc and doesn't support VReg operands.
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

private:
  MachineFunction *MF;
  const X86InstrInfo *TII; // Machine instruction info.
  bool OptIncDec;
  bool OptLEA;
};
char FixupLEAPass::ID = 0;
}

MachineInstr *
FixupLEAPass::postRAConvertToLEA(MachineFunction::iterator &MFI,
                                 MachineBasicBlock::iterator &MBBI) const {
  MachineInstr &MI = *MBBI;
  switch (MI.getOpcode()) {
  case X86::MOV32rr:
  case X86::MOV64rr: {
    const MachineOperand &Src = MI.getOperand(1);
    const MachineOperand &Dest = MI.getOperand(0);
    MachineInstr *NewMI =
        BuildMI(*MF, MI.getDebugLoc(),
                TII->get(MI.getOpcode() == X86::MOV32rr ? X86::LEA32r
                                                        : X86::LEA64r))
            .addOperand(Dest)
            .addOperand(Src)
            .addImm(1)
            .addReg(0)
            .addImm(0)
            .addReg(0);
    MFI->insert(MBBI, NewMI); // Insert the new inst
    return NewMI;
  }
  case X86::ADD64ri32:
  case X86::ADD64ri8:
  case X86::ADD64ri32_DB:
  case X86::ADD64ri8_DB:
  case X86::ADD32ri:
  case X86::ADD32ri8:
  case X86::ADD32ri_DB:
  case X86::ADD32ri8_DB:
  case X86::ADD16ri:
  case X86::ADD16ri8:
  case X86::ADD16ri_DB:
  case X86::ADD16ri8_DB:
    if (!MI.getOperand(2).isImm()) {
      // convertToThreeAddress will call getImm()
      // which requires isImm() to be true
      return nullptr;
    }
    break;
  case X86::ADD16rr:
  case X86::ADD16rr_DB:
    if (MI.getOperand(1).getReg() != MI.getOperand(2).getReg()) {
      // if src1 != src2, then convertToThreeAddress will
      // need to create a Virtual register, which we cannot do
      // after register allocation.
      return nullptr;
    }
  }
  return TII->convertToThreeAddress(MFI, MI, nullptr);
}

FunctionPass *llvm::createX86FixupLEAs() { return new FixupLEAPass(); }

bool FixupLEAPass::runOnMachineFunction(MachineFunction &Func) {
  if (skipFunction(*Func.getFunction()))
    return false;

  MF = &Func;
  const X86Subtarget &ST = Func.getSubtarget<X86Subtarget>();
  OptIncDec = !ST.slowIncDec() || Func.getFunction()->optForMinSize();
  OptLEA = ST.LEAusesAG() || ST.slowLEA();

  if (!OptLEA && !OptIncDec)
    return false;

  TII = ST.getInstrInfo();

  DEBUG(dbgs() << "Start X86FixupLEAs\n";);
  // Process all basic blocks.
  for (MachineFunction::iterator I = Func.begin(), E = Func.end(); I != E; ++I)
    processBasicBlock(Func, I);
  DEBUG(dbgs() << "End X86FixupLEAs\n";);

  return true;
}

FixupLEAPass::RegUsageState
FixupLEAPass::usesRegister(MachineOperand &p, MachineBasicBlock::iterator I) {
  RegUsageState RegUsage = RU_NotUsed;
  MachineInstr &MI = *I;

  for (unsigned int i = 0; i < MI.getNumOperands(); ++i) {
    MachineOperand &opnd = MI.getOperand(i);
    if (opnd.isReg() && opnd.getReg() == p.getReg()) {
      if (opnd.isDef())
        return RU_Write;
      RegUsage = RU_Read;
    }
  }
  return RegUsage;
}

/// getPreviousInstr - Given a reference to an instruction in a basic
/// block, return a reference to the previous instruction in the block,
/// wrapping around to the last instruction of the block if the block
/// branches to itself.
static inline bool getPreviousInstr(MachineBasicBlock::iterator &I,
                                    MachineFunction::iterator MFI) {
  if (I == MFI->begin()) {
    if (MFI->isPredecessor(&*MFI)) {
      I = --MFI->end();
      return true;
    } else
      return false;
  }
  --I;
  return true;
}

MachineBasicBlock::iterator
FixupLEAPass::searchBackwards(MachineOperand &p, MachineBasicBlock::iterator &I,
                              MachineFunction::iterator MFI) {
  int InstrDistance = 1;
  MachineBasicBlock::iterator CurInst;
  static const int INSTR_DISTANCE_THRESHOLD = 5;

  CurInst = I;
  bool Found;
  Found = getPreviousInstr(CurInst, MFI);
  while (Found && I != CurInst) {
    if (CurInst->isCall() || CurInst->isInlineAsm())
      break;
    if (InstrDistance > INSTR_DISTANCE_THRESHOLD)
      break; // too far back to make a difference
    if (usesRegister(p, CurInst) == RU_Write) {
      return CurInst;
    }
    InstrDistance += TII->getInstrLatency(
        MF->getSubtarget().getInstrItineraryData(), *CurInst);
    Found = getPreviousInstr(CurInst, MFI);
  }
  return MachineBasicBlock::iterator();
}

static inline bool isLEA(const int opcode) {
  return opcode == X86::LEA16r || opcode == X86::LEA32r ||
         opcode == X86::LEA64r || opcode == X86::LEA64_32r;
}

/// isLEASimpleIncOrDec - Does this LEA have one these forms:
/// lea  %reg, 1(%reg)
/// lea  %reg, -1(%reg)
static inline bool isLEASimpleIncOrDec(MachineInstr &LEA) {
  unsigned SrcReg = LEA.getOperand(1 + X86::AddrBaseReg).getReg();
  unsigned DstReg = LEA.getOperand(0).getReg();
  unsigned AddrDispOp = 1 + X86::AddrDisp;
  return SrcReg == DstReg &&
         LEA.getOperand(1 + X86::AddrIndexReg).getReg() == 0 &&
         LEA.getOperand(1 + X86::AddrSegmentReg).getReg() == 0 &&
         LEA.getOperand(AddrDispOp).isImm() &&
         (LEA.getOperand(AddrDispOp).getImm() == 1 ||
          LEA.getOperand(AddrDispOp).getImm() == -1);
}

bool FixupLEAPass::fixupIncDec(MachineBasicBlock::iterator &I,
                               MachineFunction::iterator MFI) const {
  MachineInstr &MI = *I;
  int Opcode = MI.getOpcode();
  if (!isLEA(Opcode))
    return false;

  if (isLEASimpleIncOrDec(MI) && TII->isSafeToClobberEFLAGS(*MFI, I)) {
    int NewOpcode;
    bool isINC = MI.getOperand(4).getImm() == 1;
    switch (Opcode) {
    case X86::LEA16r:
      NewOpcode = isINC ? X86::INC16r : X86::DEC16r;
      break;
    case X86::LEA32r:
    case X86::LEA64_32r:
      NewOpcode = isINC ? X86::INC32r : X86::DEC32r;
      break;
    case X86::LEA64r:
      NewOpcode = isINC ? X86::INC64r : X86::DEC64r;
      break;
    }

    MachineInstr *NewMI =
        BuildMI(*MFI, I, MI.getDebugLoc(), TII->get(NewOpcode))
            .addOperand(MI.getOperand(0))
            .addOperand(MI.getOperand(1));
    MFI->erase(I);
    I = static_cast<MachineBasicBlock::iterator>(NewMI);
    return true;
  }
  return false;
}

void FixupLEAPass::processInstruction(MachineBasicBlock::iterator &I,
                                      MachineFunction::iterator MFI) {
  // Process a load, store, or LEA instruction.
  MachineInstr &MI = *I;
  const MCInstrDesc &Desc = MI.getDesc();
  int AddrOffset = X86II::getMemoryOperandNo(Desc.TSFlags);
  if (AddrOffset >= 0) {
    AddrOffset += X86II::getOperandBias(Desc);
    MachineOperand &p = MI.getOperand(AddrOffset + X86::AddrBaseReg);
    if (p.isReg() && p.getReg() != X86::ESP) {
      seekLEAFixup(p, I, MFI);
    }
    MachineOperand &q = MI.getOperand(AddrOffset + X86::AddrIndexReg);
    if (q.isReg() && q.getReg() != X86::ESP) {
      seekLEAFixup(q, I, MFI);
    }
  }
}

void FixupLEAPass::seekLEAFixup(MachineOperand &p,
                                MachineBasicBlock::iterator &I,
                                MachineFunction::iterator MFI) {
  MachineBasicBlock::iterator MBI = searchBackwards(p, I, MFI);
  if (MBI != MachineBasicBlock::iterator()) {
    MachineInstr *NewMI = postRAConvertToLEA(MFI, MBI);
    if (NewMI) {
      ++NumLEAs;
      DEBUG(dbgs() << "FixLEA: Candidate to replace:"; MBI->dump(););
      // now to replace with an equivalent LEA...
      DEBUG(dbgs() << "FixLEA: Replaced by: "; NewMI->dump(););
      MFI->erase(MBI);
      MachineBasicBlock::iterator J =
          static_cast<MachineBasicBlock::iterator>(NewMI);
      processInstruction(J, MFI);
    }
  }
}

void FixupLEAPass::processInstructionForSLM(MachineBasicBlock::iterator &I,
                                            MachineFunction::iterator MFI) {
  MachineInstr &MI = *I;
  const int opcode = MI.getOpcode();
  if (!isLEA(opcode))
    return;
  if (MI.getOperand(5).getReg() != 0 || !MI.getOperand(4).isImm() ||
      !TII->isSafeToClobberEFLAGS(*MFI, I))
    return;
  const unsigned DstR = MI.getOperand(0).getReg();
  const unsigned SrcR1 = MI.getOperand(1).getReg();
  const unsigned SrcR2 = MI.getOperand(3).getReg();
  if ((SrcR1 == 0 || SrcR1 != DstR) && (SrcR2 == 0 || SrcR2 != DstR))
    return;
  if (MI.getOperand(2).getImm() > 1)
    return;
  int addrr_opcode, addri_opcode;
  switch (opcode) {
  default:
    llvm_unreachable("Unexpected LEA instruction");
  case X86::LEA16r:
    addrr_opcode = X86::ADD16rr;
    addri_opcode = X86::ADD16ri;
    break;
  case X86::LEA32r:
    addrr_opcode = X86::ADD32rr;
    addri_opcode = X86::ADD32ri;
    break;
  case X86::LEA64_32r:
  case X86::LEA64r:
    addrr_opcode = X86::ADD64rr;
    addri_opcode = X86::ADD64ri32;
    break;
  }
  DEBUG(dbgs() << "FixLEA: Candidate to replace:"; I->dump(););
  DEBUG(dbgs() << "FixLEA: Replaced by: ";);
  MachineInstr *NewMI = nullptr;
  const MachineOperand &Dst = MI.getOperand(0);
  // Make ADD instruction for two registers writing to LEA's destination
  if (SrcR1 != 0 && SrcR2 != 0) {
    const MachineOperand &Src1 = MI.getOperand(SrcR1 == DstR ? 1 : 3);
    const MachineOperand &Src2 = MI.getOperand(SrcR1 == DstR ? 3 : 1);
    NewMI = BuildMI(*MF, MI.getDebugLoc(), TII->get(addrr_opcode))
                .addOperand(Dst)
                .addOperand(Src1)
                .addOperand(Src2);
    MFI->insert(I, NewMI);
    DEBUG(NewMI->dump(););
  }
  // Make ADD instruction for immediate
  if (MI.getOperand(4).getImm() != 0) {
    const MachineOperand &SrcR = MI.getOperand(SrcR1 == DstR ? 1 : 3);
    NewMI = BuildMI(*MF, MI.getDebugLoc(), TII->get(addri_opcode))
                .addOperand(Dst)
                .addOperand(SrcR)
                .addImm(MI.getOperand(4).getImm());
    MFI->insert(I, NewMI);
    DEBUG(NewMI->dump(););
  }
  if (NewMI) {
    MFI->erase(I);
    I = static_cast<MachineBasicBlock::iterator>(NewMI);
  }
}

bool FixupLEAPass::processBasicBlock(MachineFunction &MF,
                                     MachineFunction::iterator MFI) {

  for (MachineBasicBlock::iterator I = MFI->begin(); I != MFI->end(); ++I) {
    if (OptIncDec)
      if (fixupIncDec(I, MFI))
        continue;

    if (OptLEA) {
      if (MF.getSubtarget<X86Subtarget>().isSLM())
        processInstructionForSLM(I, MFI);
      else
        processInstruction(I, MFI);
    }
  }
  return false;
}
