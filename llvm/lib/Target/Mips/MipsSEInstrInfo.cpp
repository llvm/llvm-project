//===-- MipsSEInstrInfo.cpp - Mips32/64 Instruction Information -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips32/64 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "MipsSEInstrInfo.h"
#include "MipsTargetMachine.h"
#include "MipsMachineFunction.h"
#include "InstPrinter/MipsInstPrinter.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm;

MipsSEInstrInfo::MipsSEInstrInfo(MipsTargetMachine &tm)
  : MipsInstrInfo(tm,
                  tm.getRelocationModel() == Reloc::PIC_ ? Mips::B : Mips::J),
    RI(*tm.getSubtargetImpl(), *this),
    IsN64(tm.getSubtarget<MipsSubtarget>().isABI_N64()) {}

const MipsRegisterInfo &MipsSEInstrInfo::getRegisterInfo() const {
  return RI;
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned MipsSEInstrInfo::
isLoadFromStackSlot(const MachineInstr *MI, int &FrameIndex) const
{
  unsigned Opc = MI->getOpcode();

  if ((Opc == Mips::LW)    || (Opc == Mips::LW_P8)  || (Opc == Mips::LD) ||
      (Opc == Mips::LD_P8) || (Opc == Mips::LWC1)   || (Opc == Mips::LWC1_P8) ||
      (Opc == Mips::LDC1)  || (Opc == Mips::LDC164) ||
      (Opc == Mips::LDC164_P8)) {
    if ((MI->getOperand(1).isFI()) && // is a stack slot
        (MI->getOperand(2).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(2)))) {
      FrameIndex = MI->getOperand(1).getIndex();
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
unsigned MipsSEInstrInfo::
isStoreToStackSlot(const MachineInstr *MI, int &FrameIndex) const
{
  unsigned Opc = MI->getOpcode();

  if ((Opc == Mips::SW)    || (Opc == Mips::SW_P8)  || (Opc == Mips::SD) ||
      (Opc == Mips::SD_P8) || (Opc == Mips::SWC1)   || (Opc == Mips::SWC1_P8) ||
      (Opc == Mips::SDC1)  || (Opc == Mips::SDC164) ||
      (Opc == Mips::SDC164_P8)) {
    if ((MI->getOperand(1).isFI()) && // is a stack slot
        (MI->getOperand(2).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(2)))) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
  }
  return 0;
}

void MipsSEInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I, DebugLoc DL,
                                  unsigned DestReg, unsigned SrcReg,
                                  bool KillSrc) const {
  unsigned Opc = 0, ZeroReg = 0;

  if (Mips::CPURegsRegClass.contains(DestReg)) { // Copy to CPU Reg.
    if (Mips::CPURegsRegClass.contains(SrcReg))
      Opc = Mips::ADDu, ZeroReg = Mips::ZERO;
    else if (Mips::CCRRegClass.contains(SrcReg))
      Opc = Mips::CFC1;
    else if (Mips::FGR32RegClass.contains(SrcReg))
      Opc = Mips::MFC1;
    else if (SrcReg == Mips::HI)
      Opc = Mips::MFHI, SrcReg = 0;
    else if (SrcReg == Mips::LO)
      Opc = Mips::MFLO, SrcReg = 0;
  }
  else if (Mips::CPURegsRegClass.contains(SrcReg)) { // Copy from CPU Reg.
    if (Mips::CCRRegClass.contains(DestReg))
      Opc = Mips::CTC1;
    else if (Mips::FGR32RegClass.contains(DestReg))
      Opc = Mips::MTC1;
    else if (DestReg == Mips::HI)
      Opc = Mips::MTHI, DestReg = 0;
    else if (DestReg == Mips::LO)
      Opc = Mips::MTLO, DestReg = 0;
  }
  else if (Mips::FGR32RegClass.contains(DestReg, SrcReg))
    Opc = Mips::FMOV_S;
  else if (Mips::AFGR64RegClass.contains(DestReg, SrcReg))
    Opc = Mips::FMOV_D32;
  else if (Mips::FGR64RegClass.contains(DestReg, SrcReg))
    Opc = Mips::FMOV_D64;
  else if (Mips::CCRRegClass.contains(DestReg, SrcReg))
    Opc = Mips::MOVCCRToCCR;
  else if (Mips::CPU64RegsRegClass.contains(DestReg)) { // Copy to CPU64 Reg.
    if (Mips::CPU64RegsRegClass.contains(SrcReg))
      Opc = Mips::DADDu, ZeroReg = Mips::ZERO_64;
    else if (SrcReg == Mips::HI64)
      Opc = Mips::MFHI64, SrcReg = 0;
    else if (SrcReg == Mips::LO64)
      Opc = Mips::MFLO64, SrcReg = 0;
    else if (Mips::FGR64RegClass.contains(SrcReg))
      Opc = Mips::DMFC1;
  }
  else if (Mips::CPU64RegsRegClass.contains(SrcReg)) { // Copy from CPU64 Reg.
    if (DestReg == Mips::HI64)
      Opc = Mips::MTHI64, DestReg = 0;
    else if (DestReg == Mips::LO64)
      Opc = Mips::MTLO64, DestReg = 0;
    else if (Mips::FGR64RegClass.contains(DestReg))
      Opc = Mips::DMTC1;
  }

  assert(Opc && "Cannot copy registers");

  MachineInstrBuilder MIB = BuildMI(MBB, I, DL, get(Opc));

  if (DestReg)
    MIB.addReg(DestReg, RegState::Define);

  if (ZeroReg)
    MIB.addReg(ZeroReg);

  if (SrcReg)
    MIB.addReg(SrcReg, getKillRegState(KillSrc));
}

void MipsSEInstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC,
                    const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end()) DL = I->getDebugLoc();
  MachineMemOperand *MMO = GetMemOperand(MBB, FI, MachineMemOperand::MOStore);

  unsigned Opc = 0;

  if (Mips::CPURegsRegClass.hasSubClassEq(RC))
    Opc = IsN64 ? Mips::SW_P8 : Mips::SW;
  else if (Mips::CPU64RegsRegClass.hasSubClassEq(RC))
    Opc = IsN64 ? Mips::SD_P8 : Mips::SD;
  else if (Mips::FGR32RegClass.hasSubClassEq(RC))
    Opc = IsN64 ? Mips::SWC1_P8 : Mips::SWC1;
  else if (Mips::AFGR64RegClass.hasSubClassEq(RC))
    Opc = Mips::SDC1;
  else if (Mips::FGR64RegClass.hasSubClassEq(RC))
    Opc = IsN64 ? Mips::SDC164_P8 : Mips::SDC164;

  assert(Opc && "Register class not handled!");
  BuildMI(MBB, I, DL, get(Opc)).addReg(SrcReg, getKillRegState(isKill))
    .addFrameIndex(FI).addImm(0).addMemOperand(MMO);
}

void MipsSEInstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC,
                     const TargetRegisterInfo *TRI) const
{
  DebugLoc DL;
  if (I != MBB.end()) DL = I->getDebugLoc();
  MachineMemOperand *MMO = GetMemOperand(MBB, FI, MachineMemOperand::MOLoad);
  unsigned Opc = 0;

  if (Mips::CPURegsRegClass.hasSubClassEq(RC))
    Opc = IsN64 ? Mips::LW_P8 : Mips::LW;
  else if (Mips::CPU64RegsRegClass.hasSubClassEq(RC))
    Opc = IsN64 ? Mips::LD_P8 : Mips::LD;
  else if (Mips::FGR32RegClass.hasSubClassEq(RC))
    Opc = IsN64 ? Mips::LWC1_P8 : Mips::LWC1;
  else if (Mips::AFGR64RegClass.hasSubClassEq(RC))
    Opc = Mips::LDC1;
  else if (Mips::FGR64RegClass.hasSubClassEq(RC))
    Opc = IsN64 ? Mips::LDC164_P8 : Mips::LDC164;

  assert(Opc && "Register class not handled!");
  BuildMI(MBB, I, DL, get(Opc), DestReg).addFrameIndex(FI).addImm(0)
    .addMemOperand(MMO);
}

bool MipsSEInstrInfo::expandPostRAPseudo(MachineBasicBlock::iterator MI) const {
  MachineBasicBlock &MBB = *MI->getParent();

  switch(MI->getDesc().getOpcode()) {
  default:
    return false;
  case Mips::RetRA:
    ExpandRetRA(MBB, MI, Mips::RET);
    break;
  case Mips::BuildPairF64:
    ExpandBuildPairF64(MBB, MI);
    break;
  case Mips::ExtractElementF64:
    ExpandExtractElementF64(MBB, MI);
    break;
  }

  MBB.erase(MI);
  return true;
}

/// GetOppositeBranchOpc - Return the inverse of the specified
/// opcode, e.g. turning BEQ to BNE.
unsigned MipsSEInstrInfo::GetOppositeBranchOpc(unsigned Opc) const {
  switch (Opc) {
  default:           llvm_unreachable("Illegal opcode!");
  case Mips::BEQ:    return Mips::BNE;
  case Mips::BNE:    return Mips::BEQ;
  case Mips::BGTZ:   return Mips::BLEZ;
  case Mips::BGEZ:   return Mips::BLTZ;
  case Mips::BLTZ:   return Mips::BGEZ;
  case Mips::BLEZ:   return Mips::BGTZ;
  case Mips::BEQ64:  return Mips::BNE64;
  case Mips::BNE64:  return Mips::BEQ64;
  case Mips::BGTZ64: return Mips::BLEZ64;
  case Mips::BGEZ64: return Mips::BLTZ64;
  case Mips::BLTZ64: return Mips::BGEZ64;
  case Mips::BLEZ64: return Mips::BGTZ64;
  case Mips::BC1T:   return Mips::BC1F;
  case Mips::BC1F:   return Mips::BC1T;
  }
}

/// Adjust SP by Amount bytes.
void MipsSEInstrInfo::adjustStackPtr(unsigned SP, int64_t Amount,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const {
  const MipsSubtarget &STI = TM.getSubtarget<MipsSubtarget>();
  DebugLoc DL = I != MBB.end() ? I->getDebugLoc() : DebugLoc();
  unsigned ADDu = STI.isABI_N64() ? Mips::DADDu : Mips::ADDu;
  unsigned ADDiu = STI.isABI_N64() ? Mips::DADDiu : Mips::ADDiu;

  if (isInt<16>(Amount))// addi sp, sp, amount
    BuildMI(MBB, I, DL, get(ADDiu), SP).addReg(SP).addImm(Amount);
  else { // Expand immediate that doesn't fit in 16-bit.
    unsigned Reg = loadImmediate(Amount, MBB, I, DL, 0);
    BuildMI(MBB, I, DL, get(ADDu), SP).addReg(SP).addReg(Reg, RegState::Kill);
  }
}

/// This function generates the sequence of instructions needed to get the
/// result of adding register REG and immediate IMM.
unsigned
MipsSEInstrInfo::loadImmediate(int64_t Imm, MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator II, DebugLoc DL,
                               unsigned *NewImm) const {
  MipsAnalyzeImmediate AnalyzeImm;
  const MipsSubtarget &STI = TM.getSubtarget<MipsSubtarget>();
  MachineRegisterInfo &RegInfo = MBB.getParent()->getRegInfo();
  unsigned Size = STI.isABI_N64() ? 64 : 32;
  unsigned LUi = STI.isABI_N64() ? Mips::LUi64 : Mips::LUi;
  unsigned ZEROReg = STI.isABI_N64() ? Mips::ZERO_64 : Mips::ZERO;
  const TargetRegisterClass *RC = STI.isABI_N64() ?
    &Mips::CPU64RegsRegClass : &Mips::CPURegsRegClass;
  bool LastInstrIsADDiu = NewImm;

  const MipsAnalyzeImmediate::InstSeq &Seq =
    AnalyzeImm.Analyze(Imm, Size, LastInstrIsADDiu);
  MipsAnalyzeImmediate::InstSeq::const_iterator Inst = Seq.begin();

  assert(Seq.size() && (!LastInstrIsADDiu || (Seq.size() > 1)));

  // The first instruction can be a LUi, which is different from other
  // instructions (ADDiu, ORI and SLL) in that it does not have a register
  // operand.
  unsigned Reg = RegInfo.createVirtualRegister(RC);

  if (Inst->Opc == LUi)
    BuildMI(MBB, II, DL, get(LUi), Reg).addImm(SignExtend64<16>(Inst->ImmOpnd));
  else
    BuildMI(MBB, II, DL, get(Inst->Opc), Reg).addReg(ZEROReg)
      .addImm(SignExtend64<16>(Inst->ImmOpnd));

  // Build the remaining instructions in Seq.
  for (++Inst; Inst != Seq.end() - LastInstrIsADDiu; ++Inst)
    BuildMI(MBB, II, DL, get(Inst->Opc), Reg).addReg(Reg, RegState::Kill)
      .addImm(SignExtend64<16>(Inst->ImmOpnd));

  if (LastInstrIsADDiu)
    *NewImm = Inst->ImmOpnd;

  return Reg;
}

unsigned MipsSEInstrInfo::GetAnalyzableBrOpc(unsigned Opc) const {
  return (Opc == Mips::BEQ    || Opc == Mips::BNE    || Opc == Mips::BGTZ   ||
          Opc == Mips::BGEZ   || Opc == Mips::BLTZ   || Opc == Mips::BLEZ   ||
          Opc == Mips::BEQ64  || Opc == Mips::BNE64  || Opc == Mips::BGTZ64 ||
          Opc == Mips::BGEZ64 || Opc == Mips::BLTZ64 || Opc == Mips::BLEZ64 ||
          Opc == Mips::BC1T   || Opc == Mips::BC1F   || Opc == Mips::B      ||
          Opc == Mips::J) ?
         Opc : 0;
}

void MipsSEInstrInfo::ExpandRetRA(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I,
                                unsigned Opc) const {
  BuildMI(MBB, I, I->getDebugLoc(), get(Opc)).addReg(Mips::RA);
}

void MipsSEInstrInfo::ExpandExtractElementF64(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator I) const {
  unsigned DstReg = I->getOperand(0).getReg();
  unsigned SrcReg = I->getOperand(1).getReg();
  unsigned N = I->getOperand(2).getImm();
  const MCInstrDesc& Mfc1Tdd = get(Mips::MFC1);
  DebugLoc dl = I->getDebugLoc();

  assert(N < 2 && "Invalid immediate");
  unsigned SubIdx = N ? Mips::sub_fpodd : Mips::sub_fpeven;
  unsigned SubReg = getRegisterInfo().getSubReg(SrcReg, SubIdx);

  BuildMI(MBB, I, dl, Mfc1Tdd, DstReg).addReg(SubReg);
}

void MipsSEInstrInfo::ExpandBuildPairF64(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I) const {
  unsigned DstReg = I->getOperand(0).getReg();
  unsigned LoReg = I->getOperand(1).getReg(), HiReg = I->getOperand(2).getReg();
  const MCInstrDesc& Mtc1Tdd = get(Mips::MTC1);
  DebugLoc dl = I->getDebugLoc();
  const TargetRegisterInfo &TRI = getRegisterInfo();

  // mtc1 Lo, $fp
  // mtc1 Hi, $fp + 1
  BuildMI(MBB, I, dl, Mtc1Tdd, TRI.getSubReg(DstReg, Mips::sub_fpeven))
    .addReg(LoReg);
  BuildMI(MBB, I, dl, Mtc1Tdd, TRI.getSubReg(DstReg, Mips::sub_fpodd))
    .addReg(HiReg);
}

const MipsInstrInfo *llvm::createMipsSEInstrInfo(MipsTargetMachine &TM) {
  return new MipsSEInstrInfo(TM);
}
