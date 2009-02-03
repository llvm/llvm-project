//===- MipsInstrInfo.cpp - Mips Instruction Information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "MipsInstrInfo.h"
#include "MipsTargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "MipsGenInstrInfo.inc"

using namespace llvm;

MipsInstrInfo::MipsInstrInfo(MipsTargetMachine &tm)
  : TargetInstrInfoImpl(MipsInsts, array_lengthof(MipsInsts)),
    TM(tm), RI(*TM.getSubtargetImpl(), *this) {}

static bool isZeroImm(const MachineOperand &op) {
  return op.isImm() && op.getImm() == 0;
}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
bool MipsInstrInfo::
isMoveInstr(const MachineInstr &MI, unsigned &SrcReg, unsigned &DstReg,
            unsigned &SrcSubIdx, unsigned &DstSubIdx) const 
{
  SrcSubIdx = DstSubIdx = 0; // No sub-registers.

  //  addu  $dst, $src, $zero || addu  $dst, $zero, $src
  //  or    $dst, $src, $zero || or    $dst, $zero, $src
  if ((MI.getOpcode() == Mips::ADDu) || (MI.getOpcode() == Mips::OR)) {
    if (MI.getOperand(1).getReg() == Mips::ZERO) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(2).getReg();
      return true;
    } else if (MI.getOperand(2).getReg() == Mips::ZERO) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(1).getReg();
      return true;
    }
  }

  // mov $fpDst, $fpSrc
  // mfc $gpDst, $fpSrc
  // mtc $fpDst, $gpSrc
  if (MI.getOpcode() == Mips::FMOV_SO32 || MI.getOpcode() == Mips::FMOV_AS32 ||
      MI.getOpcode() == Mips::FMOV_D32 || MI.getOpcode() == Mips::MFC1A ||
      MI.getOpcode() == Mips::MFC1 || MI.getOpcode() == Mips::MTC1A ||
      MI.getOpcode() == Mips::MTC1 ) {
    DstReg = MI.getOperand(0).getReg();
    SrcReg = MI.getOperand(1).getReg();
    return true;
  }

  //  addiu $dst, $src, 0
  if (MI.getOpcode() == Mips::ADDiu) {
    if ((MI.getOperand(1).isReg()) && (isZeroImm(MI.getOperand(2)))) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(1).getReg();
      return true;
    }
  }
  return false;
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned MipsInstrInfo::
isLoadFromStackSlot(const MachineInstr *MI, int &FrameIndex) const 
{
  if ((MI->getOpcode() == Mips::LW) || (MI->getOpcode() == Mips::LWC1) ||
      (MI->getOpcode() == Mips::LWC1A) || (MI->getOpcode() == Mips::LDC1)) {
    if ((MI->getOperand(2).isFI()) && // is a stack slot
        (MI->getOperand(1).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(1)))) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
  }

  return 0;
}

/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned MipsInstrInfo::
isStoreToStackSlot(const MachineInstr *MI, int &FrameIndex) const 
{
  if ((MI->getOpcode() == Mips::SW) || (MI->getOpcode() == Mips::SWC1) ||
      (MI->getOpcode() == Mips::SWC1A) || (MI->getOpcode() == Mips::SDC1)) {
    if ((MI->getOperand(2).isFI()) && // is a stack slot
        (MI->getOperand(1).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(1)))) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
  }
  return 0;
}

/// insertNoop - If data hazard condition is found insert the target nop
/// instruction.
void MipsInstrInfo::
insertNoop(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI) const 
{
  BuildMI(MBB, MI, get(Mips::NOP));
}

bool MipsInstrInfo::
copyRegToReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
             unsigned DestReg, unsigned SrcReg,
             const TargetRegisterClass *DestRC,
             const TargetRegisterClass *SrcRC) const {
  if (DestRC != SrcRC) {
    if ((DestRC == Mips::CPURegsRegisterClass) && 
        (SrcRC == Mips::FGR32RegisterClass))
      BuildMI(MBB, I, get(Mips::MFC1), DestReg).addReg(SrcReg);
    else if ((DestRC == Mips::CPURegsRegisterClass) && 
             (SrcRC == Mips::AFGR32RegisterClass))
      BuildMI(MBB, I, get(Mips::MFC1A), DestReg).addReg(SrcReg);
    else if ((DestRC == Mips::FGR32RegisterClass) &&
             (SrcRC == Mips::CPURegsRegisterClass))
      BuildMI(MBB, I, get(Mips::MTC1), DestReg).addReg(SrcReg);
    else if ((DestRC == Mips::AFGR32RegisterClass) &&
             (SrcRC == Mips::CPURegsRegisterClass))
      BuildMI(MBB, I, get(Mips::MTC1A), DestReg).addReg(SrcReg);
    else if ((DestRC == Mips::AFGR32RegisterClass) &&
             (SrcRC == Mips::CPURegsRegisterClass))
      BuildMI(MBB, I, get(Mips::MTC1A), DestReg).addReg(SrcReg);
    else if ((SrcRC == Mips::CCRRegisterClass) && 
             (SrcReg == Mips::FCR31))
      return true; // This register is used implicitly, no copy needed.
    else if ((DestRC == Mips::CCRRegisterClass) && 
             (DestReg == Mips::FCR31))
      return true; // This register is used implicitly, no copy needed.
    else if ((DestRC == Mips::HILORegisterClass) &&
             (SrcRC == Mips::CPURegsRegisterClass)) {
      unsigned Opc = (DestReg == Mips::HI) ? Mips::MTHI : Mips::MTLO;
      BuildMI(MBB, I, get(Opc), DestReg);
    } else if ((SrcRC == Mips::HILORegisterClass) &&
               (DestRC == Mips::CPURegsRegisterClass)) {
      unsigned Opc = (SrcReg == Mips::HI) ? Mips::MFHI : Mips::MFLO;
      BuildMI(MBB, I, get(Opc), DestReg);
    } else
      // DestRC != SrcRC, Can't copy this register
      return false;

    return true;
  }

  if (DestRC == Mips::CPURegsRegisterClass)
    BuildMI(MBB, I, get(Mips::ADDu), DestReg).addReg(Mips::ZERO)
      .addReg(SrcReg);
  else if (DestRC == Mips::FGR32RegisterClass) 
    BuildMI(MBB, I, get(Mips::FMOV_SO32), DestReg).addReg(SrcReg);
  else if (DestRC == Mips::AFGR32RegisterClass)
    BuildMI(MBB, I, get(Mips::FMOV_AS32), DestReg).addReg(SrcReg);
  else if (DestRC == Mips::AFGR64RegisterClass)
    BuildMI(MBB, I, get(Mips::FMOV_D32), DestReg).addReg(SrcReg);
  else
    // Can't copy this register
    return false;
  
  return true;
}

void MipsInstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
          unsigned SrcReg, bool isKill, int FI, 
          const TargetRegisterClass *RC) const 
{
  unsigned Opc;
  if (RC == Mips::CPURegsRegisterClass) 
    Opc = Mips::SW;
  else if (RC == Mips::FGR32RegisterClass)
    Opc = Mips::SWC1;
  else if (RC == Mips::AFGR32RegisterClass)
    Opc = Mips::SWC1A;
  else if (RC == Mips::AFGR64RegisterClass)
    Opc = Mips::SDC1;
  else 
    assert(0 && "Can't store this register to stack slot");

  BuildMI(MBB, I, get(Opc)).addReg(SrcReg, false, false, isKill)
          .addImm(0).addFrameIndex(FI);
}

void MipsInstrInfo::storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
  bool isKill, SmallVectorImpl<MachineOperand> &Addr, 
  const TargetRegisterClass *RC, SmallVectorImpl<MachineInstr*> &NewMIs) const 
{
  unsigned Opc;
  if (RC == Mips::CPURegsRegisterClass) 
    Opc = Mips::SW;
  else if (RC == Mips::FGR32RegisterClass)
    Opc = Mips::SWC1;
  else if (RC == Mips::AFGR32RegisterClass)
    Opc = Mips::SWC1A;
  else if (RC == Mips::AFGR64RegisterClass)
    Opc = Mips::SDC1;
  else 
    assert(0 && "Can't store this register");

  MachineInstrBuilder MIB = BuildMI(MF, get(Opc))
    .addReg(SrcReg, false, false, isKill);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i) {
    MachineOperand &MO = Addr[i];
    if (MO.isReg())
      MIB.addReg(MO.getReg());
    else if (MO.isImm())
      MIB.addImm(MO.getImm());
    else
      MIB.addFrameIndex(MO.getIndex());
  }
  NewMIs.push_back(MIB);
  return;
}

void MipsInstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const 
{
  unsigned Opc;
  if (RC == Mips::CPURegsRegisterClass) 
    Opc = Mips::LW;
  else if (RC == Mips::FGR32RegisterClass)
    Opc = Mips::LWC1;
  else if (RC == Mips::AFGR32RegisterClass)
    Opc = Mips::LWC1A;
  else if (RC == Mips::AFGR64RegisterClass)
    Opc = Mips::LDC1;
  else 
    assert(0 && "Can't load this register from stack slot");
    
  BuildMI(MBB, I, get(Opc), DestReg).addImm(0).addFrameIndex(FI);
}

void MipsInstrInfo::loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                                       SmallVectorImpl<MachineOperand> &Addr,
                                       const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  unsigned Opc;
  if (RC == Mips::CPURegsRegisterClass) 
    Opc = Mips::LW;
  else if (RC == Mips::FGR32RegisterClass)
    Opc = Mips::LWC1;
  else if (RC == Mips::AFGR32RegisterClass)
    Opc = Mips::LWC1A;
  else if (RC == Mips::AFGR64RegisterClass)
    Opc = Mips::LDC1;
  else 
    assert(0 && "Can't load this register");

  MachineInstrBuilder MIB = BuildMI(MF, get(Opc), DestReg);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i) {
    MachineOperand &MO = Addr[i];
    if (MO.isReg())
      MIB.addReg(MO.getReg());
    else if (MO.isImm())
      MIB.addImm(MO.getImm());
    else
      MIB.addFrameIndex(MO.getIndex());
  }
  NewMIs.push_back(MIB);
  return;
}

MachineInstr *MipsInstrInfo::
foldMemoryOperandImpl(MachineFunction &MF,
                      MachineInstr* MI,
                      const SmallVectorImpl<unsigned> &Ops, int FI) const 
{
  if (Ops.size() != 1) return NULL;

  MachineInstr *NewMI = NULL;

  switch (MI->getOpcode()) {
  case Mips::ADDu:
    if ((MI->getOperand(0).isReg()) &&
        (MI->getOperand(1).isReg()) &&
        (MI->getOperand(1).getReg() == Mips::ZERO) &&
        (MI->getOperand(2).isReg())) {
      if (Ops[0] == 0) {    // COPY -> STORE
        unsigned SrcReg = MI->getOperand(2).getReg();
        bool isKill = MI->getOperand(2).isKill();
        NewMI = BuildMI(MF, get(Mips::SW)).addReg(SrcReg, false, false, isKill)
          .addImm(0).addFrameIndex(FI);
      } else {              // COPY -> LOAD
        unsigned DstReg = MI->getOperand(0).getReg();
        bool isDead = MI->getOperand(0).isDead();
        NewMI = BuildMI(MF, get(Mips::LW))
          .addReg(DstReg, true, false, false, isDead)
          .addImm(0).addFrameIndex(FI);
      }
    }
    break;
  case Mips::FMOV_SO32:
  case Mips::FMOV_AS32:
  case Mips::FMOV_D32:
    if ((MI->getOperand(0).isReg()) &&
        (MI->getOperand(1).isReg())) {
      const TargetRegisterClass 
        *RC = RI.getRegClass(MI->getOperand(0).getReg());
      unsigned StoreOpc, LoadOpc;

      if (RC == Mips::FGR32RegisterClass) {
        LoadOpc = Mips::LWC1; StoreOpc = Mips::SWC1;
      } else if (RC == Mips::AFGR32RegisterClass) {
        LoadOpc = Mips::LWC1A; StoreOpc = Mips::SWC1A;
      } else if (RC == Mips::AFGR64RegisterClass) {
        LoadOpc = Mips::LDC1; StoreOpc = Mips::SDC1;
      } else
        assert(0 && "foldMemoryOperandImpl register unknown");

      if (Ops[0] == 0) {    // COPY -> STORE
        unsigned SrcReg = MI->getOperand(1).getReg();
        bool isKill = MI->getOperand(1).isKill();
        NewMI = BuildMI(MF, get(StoreOpc)).addReg(SrcReg, false, false, isKill)
          .addImm(0).addFrameIndex(FI) ;
      } else {              // COPY -> LOAD
        unsigned DstReg = MI->getOperand(0).getReg();
        bool isDead = MI->getOperand(0).isDead();
        NewMI = BuildMI(MF, get(LoadOpc))
          .addReg(DstReg, true, false, false, isDead)
          .addImm(0).addFrameIndex(FI);
      }
    }
    break;
  }

  return NewMI;
}

//===----------------------------------------------------------------------===//
// Branch Analysis
//===----------------------------------------------------------------------===//

/// GetCondFromBranchOpc - Return the Mips CC that matches 
/// the correspondent Branch instruction opcode.
static Mips::CondCode GetCondFromBranchOpc(unsigned BrOpc) 
{
  switch (BrOpc) {
  default: return Mips::COND_INVALID;
  case Mips::BEQ  : return Mips::COND_E;
  case Mips::BNE  : return Mips::COND_NE;
  case Mips::BGTZ : return Mips::COND_GZ;
  case Mips::BGEZ : return Mips::COND_GEZ;
  case Mips::BLTZ : return Mips::COND_LZ;
  case Mips::BLEZ : return Mips::COND_LEZ;

  // We dont do fp branch analysis yet!  
  case Mips::BC1T : 
  case Mips::BC1F : return Mips::COND_INVALID;
  }
}

/// GetCondBranchFromCond - Return the Branch instruction
/// opcode that matches the cc.
unsigned Mips::GetCondBranchFromCond(Mips::CondCode CC) 
{
  switch (CC) {
  default: assert(0 && "Illegal condition code!");
  case Mips::COND_E   : return Mips::BEQ;
  case Mips::COND_NE  : return Mips::BNE;
  case Mips::COND_GZ  : return Mips::BGTZ;
  case Mips::COND_GEZ : return Mips::BGEZ;
  case Mips::COND_LZ  : return Mips::BLTZ;
  case Mips::COND_LEZ : return Mips::BLEZ;

  case Mips::FCOND_F:
  case Mips::FCOND_UN:
  case Mips::FCOND_EQ:
  case Mips::FCOND_UEQ:
  case Mips::FCOND_OLT:
  case Mips::FCOND_ULT:
  case Mips::FCOND_OLE:
  case Mips::FCOND_ULE:
  case Mips::FCOND_SF:
  case Mips::FCOND_NGLE:
  case Mips::FCOND_SEQ:
  case Mips::FCOND_NGL:
  case Mips::FCOND_LT:
  case Mips::FCOND_NGE:
  case Mips::FCOND_LE:
  case Mips::FCOND_NGT: return Mips::BC1T;

  case Mips::FCOND_T:
  case Mips::FCOND_OR:
  case Mips::FCOND_NEQ:
  case Mips::FCOND_OGL:
  case Mips::FCOND_UGE:
  case Mips::FCOND_OGE:
  case Mips::FCOND_UGT:
  case Mips::FCOND_OGT:
  case Mips::FCOND_ST:
  case Mips::FCOND_GLE:
  case Mips::FCOND_SNE:
  case Mips::FCOND_GL:
  case Mips::FCOND_NLT:
  case Mips::FCOND_GE:
  case Mips::FCOND_NLE:
  case Mips::FCOND_GT: return Mips::BC1F;
  }
}

/// GetOppositeBranchCondition - Return the inverse of the specified 
/// condition, e.g. turning COND_E to COND_NE.
Mips::CondCode Mips::GetOppositeBranchCondition(Mips::CondCode CC) 
{
  switch (CC) {
  default: assert(0 && "Illegal condition code!");
  case Mips::COND_E   : return Mips::COND_NE;
  case Mips::COND_NE  : return Mips::COND_E;
  case Mips::COND_GZ  : return Mips::COND_LEZ;
  case Mips::COND_GEZ : return Mips::COND_LZ;
  case Mips::COND_LZ  : return Mips::COND_GEZ;
  case Mips::COND_LEZ : return Mips::COND_GZ;
  case Mips::FCOND_F  : return Mips::FCOND_T;
  case Mips::FCOND_UN : return Mips::FCOND_OR;
  case Mips::FCOND_EQ : return Mips::FCOND_NEQ;
  case Mips::FCOND_UEQ: return Mips::FCOND_OGL;
  case Mips::FCOND_OLT: return Mips::FCOND_UGE;
  case Mips::FCOND_ULT: return Mips::FCOND_OGE;
  case Mips::FCOND_OLE: return Mips::FCOND_UGT;
  case Mips::FCOND_ULE: return Mips::FCOND_OGT;
  case Mips::FCOND_SF:  return Mips::FCOND_ST;
  case Mips::FCOND_NGLE:return Mips::FCOND_GLE;
  case Mips::FCOND_SEQ: return Mips::FCOND_SNE;
  case Mips::FCOND_NGL: return Mips::FCOND_GL;
  case Mips::FCOND_LT:  return Mips::FCOND_NLT;
  case Mips::FCOND_NGE: return Mips::FCOND_GE;
  case Mips::FCOND_LE:  return Mips::FCOND_NLE;
  case Mips::FCOND_NGT: return Mips::FCOND_GT;
  }
}

bool MipsInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB, 
                                  MachineBasicBlock *&TBB,
                                  MachineBasicBlock *&FBB,
                                  SmallVectorImpl<MachineOperand> &Cond) const 
{
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I))
    return false;
  
  // Get the last instruction in the block.
  MachineInstr *LastInst = I;
  
  // If there is only one terminator instruction, process it.
  unsigned LastOpc = LastInst->getOpcode();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (!LastInst->getDesc().isBranch())
      return true;

    // Unconditional branch
    if (LastOpc == Mips::J) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }

    Mips::CondCode BranchCode = GetCondFromBranchOpc(LastInst->getOpcode());
    if (BranchCode == Mips::COND_INVALID)
      return true;  // Can't handle indirect branch.

    // Conditional branch
    // Block ends with fall-through condbranch.
    if (LastOpc != Mips::COND_INVALID) {
      int LastNumOp = LastInst->getNumOperands();

      TBB = LastInst->getOperand(LastNumOp-1).getMBB();
      Cond.push_back(MachineOperand::CreateImm(BranchCode));

      for (int i=0; i<LastNumOp-1; i++) {
        Cond.push_back(LastInst->getOperand(i));
      }

      return false;
    }
  }
  
  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = I;
  
  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() && isUnpredicatedTerminator(--I))
    return true;

  // If the block ends with Mips::J and a Mips::BNE/Mips::BEQ, handle it.
  unsigned SecondLastOpc    = SecondLastInst->getOpcode();
  Mips::CondCode BranchCode = GetCondFromBranchOpc(SecondLastOpc);

  if (BranchCode != Mips::COND_INVALID && LastOpc == Mips::J) {
    int SecondNumOp = SecondLastInst->getNumOperands();

    TBB = SecondLastInst->getOperand(SecondNumOp-1).getMBB();
    Cond.push_back(MachineOperand::CreateImm(BranchCode));

    for (int i=0; i<SecondNumOp-1; i++) {
      Cond.push_back(SecondLastInst->getOperand(i));
    }

    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }
  
  // If the block ends with two unconditional branches, handle it. The last 
  // one is not executed, so remove it.
  if ((SecondLastOpc == Mips::J) && (LastOpc == Mips::J)) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned MipsInstrInfo::
InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB, 
             MachineBasicBlock *FBB,
             const SmallVectorImpl<MachineOperand> &Cond) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 3 || Cond.size() == 2 || Cond.size() == 0) &&
         "Mips branch conditions can have two|three components!");

  if (FBB == 0) { // One way branch.
    if (Cond.empty()) {
      // Unconditional branch?
      BuildMI(&MBB, get(Mips::J)).addMBB(TBB);
    } else {
      // Conditional branch.
      unsigned Opc = GetCondBranchFromCond((Mips::CondCode)Cond[0].getImm());
      const TargetInstrDesc &TID = get(Opc);

      if (TID.getNumOperands() == 3)
        BuildMI(&MBB, TID).addReg(Cond[1].getReg())
                          .addReg(Cond[2].getReg())
                          .addMBB(TBB);
      else
        BuildMI(&MBB, TID).addReg(Cond[1].getReg())
                          .addMBB(TBB);

    }                             
    return 1;
  }
  
  // Two-way Conditional branch.
  unsigned Opc = GetCondBranchFromCond((Mips::CondCode)Cond[0].getImm());
  const TargetInstrDesc &TID = get(Opc);

  if (TID.getNumOperands() == 3)
    BuildMI(&MBB, TID).addReg(Cond[1].getReg()).addReg(Cond[2].getReg())
                      .addMBB(TBB);
  else
    BuildMI(&MBB, TID).addReg(Cond[1].getReg()).addMBB(TBB);

  BuildMI(&MBB, get(Mips::J)).addMBB(FBB);
  return 2;
}

unsigned MipsInstrInfo::
RemoveBranch(MachineBasicBlock &MBB) const 
{
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  if (I->getOpcode() != Mips::J && 
      GetCondFromBranchOpc(I->getOpcode()) == Mips::COND_INVALID)
    return 0;
  
  // Remove the branch.
  I->eraseFromParent();
  
  I = MBB.end();
  
  if (I == MBB.begin()) return 1;
  --I;
  if (GetCondFromBranchOpc(I->getOpcode()) == Mips::COND_INVALID)
    return 1;
  
  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

/// BlockHasNoFallThrough - Analyze if MachineBasicBlock does not
/// fall-through into its successor block.
bool MipsInstrInfo::
BlockHasNoFallThrough(const MachineBasicBlock &MBB) const 
{
  if (MBB.empty()) return false;
  
  switch (MBB.back().getOpcode()) {
  case Mips::RET:     // Return.
  case Mips::JR:      // Indirect branch.
  case Mips::J:       // Uncond branch.
    return true;
  default: return false;
  }
}

/// ReverseBranchCondition - Return the inverse opcode of the 
/// specified Branch instruction.
bool MipsInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const 
{
  assert( (Cond.size() == 3 || Cond.size() == 2) && 
          "Invalid Mips branch condition!");
  Cond[0].setImm(GetOppositeBranchCondition((Mips::CondCode)Cond[0].getImm()));
  return false;
}
