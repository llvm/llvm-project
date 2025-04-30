//===-- Next32InstrInfo.cpp - Next32 Instruction Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Next32 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "Next32InstrInfo.h"
#include "Next32.h"
#include "Next32Subtarget.h"
#include "TargetInfo/Next32BaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <iterator>

#define GET_INSTRINFO_CTOR_DTOR
#include "Next32GenInstrInfo.inc"

using namespace llvm;
#define DEBUG_TYPE "next32-instrinfo"

Next32InstrInfo::Next32InstrInfo(const Next32Subtarget &STI)
    : Next32GenInstrInfo(
          /* Next32::ADJCALLSTACKDOWN, Next32::ADJCALLSTACKUP */),
      Subtarget(STI), RI() {}

bool Next32InstrInfo::IsRegisterCopyable(Register Reg) const {
  for (unsigned i = 0; i < RI.getNumRegClasses(); i++)
    if (RI.getRegClass(i)->contains(Reg))
      return true;

  return false;
}

void Next32InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I,
                                  const DebugLoc &DL, MCRegister DestReg,
                                  MCRegister SrcReg, bool KillSrc) const {
  if (IsRegisterCopyable(SrcReg) && IsRegisterCopyable(DestReg)) {
    BuildMI(MBB, I, DL, get(Next32::DUP), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addReg(SrcReg);
  } else {
    llvm_unreachable("Impossible reg-to-reg copy");
  }
}

void Next32InstrInfo::storeRegToStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI, Register SrcReg,
    bool isKill, int FrameIndex, const TargetRegisterClass *RC,
    const TargetRegisterInfo *TRI, Register VReg) const {
  assert(RC == &Next32::GPR32RegClass && "Unexpected register class");

  DebugLoc DL;
  if (MI != MBB.end())
    DL = MI->getDebugLoc();

  BuildMI(MBB, MI, DL, get(Next32::STORE_REG))
      .addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FrameIndex);
}

void Next32InstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                           Register DestReg, int FrameIndex,
                                           const TargetRegisterClass *RC,
                                           const TargetRegisterInfo *TRI,
                                           Register VReg) const {
  assert(RC == &Next32::GPR32RegClass && "Unexpected register class");

  DebugLoc DL;
  if (MI != MBB.end())
    DL = MI->getDebugLoc();

  BuildMI(MBB, MI, DL, get(Next32::LOAD_REG), DestReg)
      .addFrameIndex(FrameIndex);
}

void Next32InstrInfo::expandLoadStoreReg(MachineBasicBlock &MBB,
                                         MachineInstr &I) const {
  DebugLoc DL = I.getDebugLoc();
  const unsigned int Reg = I.getOperand(0).getReg();
  const bool KillReg = I.getOperand(0).isKill();
  unsigned int Opc;

  switch (I.getOpcode()) {
  case Next32::LOAD_REG:
    Opc = Next32::MEMREAD;
    break;
  case Next32::STORE_REG:
    Opc = Next32::MEMWRITE;
    break;
  default:
    llvm_unreachable("Unexpected opcode");
  }

  const int Offset = I.getOperand(1).getImm();
  unsigned int AddrLowReg = Next32::SP_LOW;
  unsigned int AddrHighReg = Next32::SP_HIGH;
  bool KillAddrRegs = false;
  auto TFI = Subtarget.getFrameLowering();
  auto Align = TFI->getStackAlign();
  if (Offset) {
    Align = commonAlignment(Align, Offset);
    AddrLowReg = Next32::SPREL_LOW;
    AddrHighReg = Next32::SPREL_HIGH;
    KillAddrRegs = true;

    // Higher part of the offset is zero because stack grows up.
    assert(Subtarget.getFrameLowering()->getStackGrowthDirection() ==
           TargetFrameLowering::StackGrowsUp);

    if (Subtarget.hasLEA()) {
      // Emit following instructions:
      //   movl scratch1, 0
      //   movl scratch2, offset_low
      //   dup SPREL_HIGH, SP_HIGH
      //   dup SPREL_LOW, SP_LOW
      BuildMI(MBB, I, DL, get(Next32::MOVL), Next32::SCRATCH1).addImm(0);
      BuildMI(MBB, I, DL, get(Next32::MOVL), Next32::SCRATCH2).addImm(Offset);
      copyPhysReg(MBB, I, DL, AddrHighReg, Next32::SP_HIGH, false);
      copyPhysReg(MBB, I, DL, AddrLowReg, Next32::SP_LOW, false);

      // Emit leadisp hi, lo
      BuildMI(MBB, I, I.getDebugLoc(), get(Next32::LEADISP))
          .addReg(Next32::SCRATCH1)                  // Disp High
          .addReg(Next32::SCRATCH2, RegState::Kill); // Disp Low

      // Emit leascale hi, lo
      BuildMI(MBB, I, I.getDebugLoc(), get(Next32::LEASCALE))
          .addReg(Next32::SCRATCH1)  // Scale High
          .addReg(Next32::SCRATCH1); // Scale Low

      // Emit leaindex hi, lo
      BuildMI(MBB, I, I.getDebugLoc(), get(Next32::LEAINDEX))
          .addReg(Next32::SCRATCH1, RegState::Kill)  // Index High
          .addReg(Next32::SCRATCH1, RegState::Kill); // Index Low

      // Emit leabase hi, lo
      BuildMI(MBB, I, I.getDebugLoc(), get(Next32::LEABASE))
          .addReg(AddrHighReg, RegState::Define) // Dest High
          .addReg(AddrLowReg, RegState::Define)  // Dest Low
          .addReg(AddrHighReg)                   // Base High
          .addReg(AddrLowReg);                   // Base Low
    } else {
      BuildMI(MBB, I, DL, get(Next32::MOVL), AddrLowReg).addImm(Offset);
      BuildMI(MBB, I, DL, get(Next32::ADD), AddrLowReg)
          .addReg(AddrLowReg)
          .addReg(Next32::SP_LOW);
      copyPhysReg(MBB, I, DL, Next32::SCRATCH1, AddrLowReg, false);
      BuildMI(MBB, I, DL, get(Next32::FLAGS), Next32::SCRATCH1)
          .addReg(Next32::SCRATCH1);

      BuildMI(MBB, I, DL, get(Next32::MOVL), AddrHighReg).addImm(0);
      BuildMI(MBB, I, DL, get(Next32::ADC), AddrHighReg)
          .addReg(AddrHighReg)
          .addReg(Next32::SP_HIGH)
          .addReg(Next32::SCRATCH1, RegState::Kill);
    }
  }

  BuildMI(MBB, I, DL, get(Opc))
      .addImm(Next32Helpers::BytesToLog2AlignValue(Align.value()))
      .addImm(llvm::Next32Constants::InstructionSize::InstructionSize32)
      .addReg(AddrHighReg, getKillRegState(KillAddrRegs))
      .addReg(AddrLowReg, getKillRegState(KillAddrRegs))
      .addReg(Next32::TID)
      .addImm(Next32Constants::InstCodeAddressSpace::GENERIC);
  BuildMI(MBB, I, DL, get(Next32::MEMDATA), Reg)
      .addReg(Reg, getKillRegState(KillReg));
  BuildMI(MBB, I, DL, get(Next32::BARRIER), Next32::TID)
      .addReg(Next32::TID)
      .addReg(Reg);
}

void Next32InstrInfo::expandPseudoLEA(MachineBasicBlock &MBB,
                                      MachineInstr &I) const {
  // Emit leadisp hi, lo
  BuildMI(MBB, I, I.getDebugLoc(), get(Next32::LEADISP))
      .add(I.getOperand(8))  // Disp High
      .add(I.getOperand(9)); // Disp Low

  // Emit leascale hi, lo
  BuildMI(MBB, I, I.getDebugLoc(), get(Next32::LEASCALE))
      .add(I.getOperand(6))  // Scale High
      .add(I.getOperand(7)); // Scale Low

  // Emit leaindex hi, lo, scale
  BuildMI(MBB, I, I.getDebugLoc(), get(Next32::LEAINDEX))
      .add(I.getOperand(4))  // Index High
      .add(I.getOperand(5)); // Index Low

  // Emit leabase hi, lo
  BuildMI(MBB, I, I.getDebugLoc(), get(Next32::LEABASE))
      .add(I.getOperand(0))  // Dest High
      .add(I.getOperand(1))  // Dest Low
      .add(I.getOperand(2))  // Base High
      .add(I.getOperand(3)); // Base Low
}

void Next32InstrInfo::expandGVMemWrite(MachineBasicBlock &MBB,
                                       MachineInstr &I) const {
  // Operand 0 is the return indication, constrained to be the same register as
  // operand 6 (first data).
  const unsigned VMemParams = I.getOperand(1).getImm();
  BuildMI(MBB, I, I.getDebugLoc(), get(Next32::VMEMWRITE))
      .addImm((VMemParams >> 16) & 0xFF)  // Align
      .addImm((VMemParams >> 8) & 0xFF)   // Type
      .add(I.getOperand(2))               // Addr Hi
      .add(I.getOperand(3))               // Addr Low
      .add(I.getOperand(4))               // TID
      .add(I.getOperand(5))               // Pred
      .addImm(VMemParams & 0xFF)          // Count
      .addImm((VMemParams >> 24) & 0xFF); // Addr Space

  for (unsigned i = 6, e = I.getNumOperands(); i != e; ++i) {
    Register R = I.getOperand(i).getReg();
    BuildMI(MBB, I, I.getDebugLoc(), get(Next32::MEMDATA), R).addReg(R);
  }
}

void Next32InstrInfo::expandGVMemRead(MachineBasicBlock &MBB,
                                      MachineInstr &I) const {
  const unsigned MemDataCnt = I.getDesc().getNumDefs();
  const unsigned VMemParams = I.getOperand(MemDataCnt + 0).getImm();
  BuildMI(MBB, I, I.getDebugLoc(), get(Next32::VMEMREAD))
      .addImm((VMemParams >> 16) & 0xFF)  // Align
      .addImm((VMemParams >> 8) & 0xFF)   // Type
      .add(I.getOperand(MemDataCnt + 1))  // Addr Hi
      .add(I.getOperand(MemDataCnt + 2))  // Addr Low
      .add(I.getOperand(MemDataCnt + 3))  // TID
      .add(I.getOperand(MemDataCnt + 4))  // Pred
      .addImm(VMemParams & 0xFF)          // Count
      .addImm((VMemParams >> 24) & 0xFF); // Addr Space

  for (unsigned i = 0; i < MemDataCnt; ++i) {
    Register R = I.getOperand(i).getReg();
    BuildMI(MBB, I, I.getDebugLoc(), get(Next32::MEMDATA), R).addReg(R);
  }
}

void Next32InstrInfo::expandGMemWrite(MachineBasicBlock &MBB,
                                      MachineInstr &I) const {
  // Operand 0 is the return indication, constrained to be the same register as
  // operand 5 (first data).
  const uint64_t MemParams = I.getOperand(1).getImm();
  const uint32_t MemWriteOpCode = MemParams >> 32;

  BuildMI(MBB, I, I.getDebugLoc(), get(MemWriteOpCode))
      .addImm((MemParams >> 8) & 0xFF)   // Align
      .addImm(MemParams & 0xFF)          // Size
      .addReg(I.getOperand(2).getReg())  // Addr Hi
      .addReg(I.getOperand(3).getReg())  // Addr Lo
      .addReg(I.getOperand(4).getReg())  // TID
      .addImm((MemParams >> 24) & 0xFF)  // Count
      .addImm((MemParams >> 16) & 0xFF); // Addr Space

  for (unsigned i = 5, e = I.getNumOperands(); i != e; ++i) {
    Register R = I.getOperand(i).getReg();
    BuildMI(MBB, I, I.getDebugLoc(), get(Next32::MEMDATA), R).addReg(R);
  }
}

void Next32InstrInfo::expandGMemRead(MachineBasicBlock &MBB,
                                     MachineInstr &I) const {
  unsigned MemDataCnt;
  switch (I.getOpcode()) {
  case Next32::GMEMREAD_1:
    MemDataCnt = 1;
    break;
  case Next32::GMEMREAD_2:
    MemDataCnt = 2;
    break;
  case Next32::GMEMREAD_4:
    MemDataCnt = 4;
    break;
  case Next32::GMEMREAD_8:
    MemDataCnt = 8;
    break;
  case Next32::GMEMREAD_16:
    MemDataCnt = 16;
    break;
  default:
    report_fatal_error("Unexpected memory size");
  }

  const uint64_t MemParams = I.getOperand(MemDataCnt + 0).getImm();
  const uint32_t MemReadOpCode = MemParams >> 32;

  BuildMI(MBB, I, I.getDebugLoc(), get(MemReadOpCode))
      .addImm((MemParams >> 8) & 0xFF)               // Align
      .addImm(MemParams & 0xFF)                      // Size
      .addReg(I.getOperand(MemDataCnt + 1).getReg()) // Addr Hi
      .addReg(I.getOperand(MemDataCnt + 2).getReg()) // Addr Lo
      .addReg(I.getOperand(MemDataCnt + 3).getReg()) // TID
      .addImm((MemParams >> 24) & 0xFF)              // Count
      .addImm((MemParams >> 16) & 0xFF);             // Addr Space

  for (unsigned i = 0; i < MemDataCnt; ++i) {
    Register R = I.getOperand(i).getReg();
    BuildMI(MBB, I, I.getDebugLoc(), get(Next32::MEMDATA), R).addReg(R);
  }
}

void Next32InstrInfo::expandFetchAndOp(MachineBasicBlock &MBB,
                                       MachineInstr &I) const {
  const unsigned int MemDataResCnt = I.getDesc().getNumDefs();
  const uint64_t MemFaOpParams = I.getOperand(MemDataResCnt + 0).getImm();
  const uint32_t FaOpCode = MemFaOpParams >> 32;

  BuildMI(MBB, I, I.getDebugLoc(), get(FaOpCode))
      .addImm((MemFaOpParams >> 8) & 0xFF)              // Align
      .addImm(MemFaOpParams & 0xFF)                     // Size
      .addReg(I.getOperand(MemDataResCnt + 1).getReg()) // Addr Hi
      .addReg(I.getOperand(MemDataResCnt + 2).getReg()) // Addr Lo
      .addReg(I.getOperand(MemDataResCnt + 3).getReg()) // TID
      .addImm((MemFaOpParams >> 24) & 0xFF)             // Count
      .addImm((MemFaOpParams >> 16) & 0xFF);            // Addr Space

  for (unsigned i = MemDataResCnt + 4, e = I.getNumOperands(); i != e; i++) {
    Register R = I.getOperand(i).getReg();
    BuildMI(MBB, I, I.getDebugLoc(), get(Next32::MEMDATA), R).addReg(R);
  }
}

void Next32InstrInfo::expandCompareAndSwap(MachineBasicBlock &MBB,
                                           MachineInstr &I) const {
  const unsigned int MemDataResCnt = I.getDesc().getNumDefs();
  const uint64_t MemCasOpParams = I.getOperand(MemDataResCnt + 0).getImm();
  const uint32_t CasOpCode = MemCasOpParams >> 32;

  BuildMI(MBB, I, I.getDebugLoc(), get(CasOpCode))
      .addImm((MemCasOpParams >> 8) & 0xFF)             // Align
      .addImm(MemCasOpParams & 0xFF)                    // Size
      .addReg(I.getOperand(MemDataResCnt + 1).getReg()) // Addr Hi
      .addReg(I.getOperand(MemDataResCnt + 2).getReg()) // Addr Lo
      .addReg(I.getOperand(MemDataResCnt + 3).getReg()) // TID
      .addImm((MemCasOpParams >> 24) & 0xFF)            // Count
      .addImm((MemCasOpParams >> 16) & 0xFF);           // Addr Space

  for (unsigned i = MemDataResCnt + 4, e = I.getNumOperands(); i != e; i++) {
    Register R = I.getOperand(i).getReg();
    BuildMI(MBB, I, I.getDebugLoc(), get(Next32::MEMDATA), R).addReg(R);
  }
}

// These are pseudo-instructions rather than regular instructions because while
// the FLAGS instruction is modeled as an instruction with a single in/out
// register operand, it actually acts as a modifier that must be attached to the
// compute operation whose flags are desired (possibly through a DUP), and
// cannot be transferred across basic-blocks. A better solution would require
// changes to the instruction set itself.
void Next32InstrInfo::expandArithWithFlags(MachineBasicBlock &MBB,
                                           MachineInstr &I) const {
  unsigned Opcode;
  switch (I.getOpcode()) {
  case Next32::ADDFLAGS:
    Opcode = Next32::ADD;
    break;
  case Next32::SUBFLAGS:
    Opcode = Next32::SUB;
    break;
  case Next32::ADCFLAGS:
    Opcode = Next32::ADC;
    break;
  case Next32::SBBFLAGS:
    Opcode = Next32::SBB;
    break;
  default:
    llvm_unreachable("Unexpected opcode");
  }

  const auto DL = I.getDebugLoc();
  const auto &ValueOp = I.getOperand(0);
  const unsigned ValueReg = ValueOp.getReg();
  const auto &FlagsOp = I.getOperand(1);
  const unsigned FlagsReg = FlagsOp.getReg();

  assert(ValueOp.isDef() && FlagsOp.isDef() && "Unexpected operand format");

  auto ArithMI = BuildMI(MBB, I, DL, get(Opcode), ValueReg);
  for (unsigned i = 2; i < I.getNumOperands(); ++i)
    ArithMI.add(I.getOperand(i));

  if (!FlagsOp.isDead()) {
    copyPhysReg(MBB, I, DL, FlagsReg, ValueReg,
                /* KillSrc = */ ValueOp.isDead());
    BuildMI(MBB, I, DL, get(Next32::FLAGS), FlagsReg).addReg(FlagsReg);
  }
}

void Next32InstrInfo::expandSetTID(MachineBasicBlock &MBB,
                                   MachineInstr &I) const {
  const auto DL = I.getDebugLoc();
  BuildMI(MBB, I, DL, get(Next32::DUP), Next32::TID)
      .addReg(Next32::TID)
      .addReg(I.getOperand(0).getReg());
}

void Next32InstrInfo::expandFrameOffsetWrapper(MachineBasicBlock &MBB,
                                               MachineInstr &I) const {
  // FrameOffsetWrapper reg:offset_high, reg:offset_low, frame_index,
  // internal_offset
  const auto DL = I.getDebugLoc();

  assert(I.getOperand(0).isReg() && I.getOperand(1).isReg() &&
         "FrameOffsetWrapper first operands are not registers");
  assert(I.getOperand(2).isImm() && I.getOperand(3).isImm() &&
         "FrameoffsetWrapper last opearnds are not immediates");

  int64_t Offset = I.getOperand(2).getImm() + I.getOperand(3).getImm();

  // High Offset Register
  BuildMI(MBB, I, DL, get(Next32::MOVL), I.getOperand(0).getReg())
      .addImm(0xFFFFFFFF & (Offset >> 32));
  // Low Offset Register
  BuildMI(MBB, I, DL, get(Next32::MOVL), I.getOperand(1).getReg())
      .addImm(0xFFFFFFFF & Offset);
}

bool Next32InstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  MachineBasicBlock &MBB = *MI.getParent();
  switch (MI.getOpcode()) {
  default:
    return false;
  case Next32::LOAD_REG:
  case Next32::STORE_REG:
    expandLoadStoreReg(MBB, MI);
    break;
  case Next32::PSEUDOLEA:
    expandPseudoLEA(MBB, MI);
    break;
  case Next32::GVMEMWRITE:
    expandGVMemWrite(MBB, MI);
    break;
  case Next32::GVMEMREAD_1:
  case Next32::GVMEMREAD_2:
  case Next32::GVMEMREAD_4:
  case Next32::GVMEMREAD_8:
  case Next32::GVMEMREAD_16:
    expandGVMemRead(MBB, MI);
    break;
  case Next32::GMEMWRITE:
    expandGMemWrite(MBB, MI);
    break;
  case Next32::GMEMREAD_1:
  case Next32::GMEMREAD_2:
  case Next32::GMEMREAD_4:
  case Next32::GMEMREAD_8:
  case Next32::GMEMREAD_16:
    expandGMemRead(MBB, MI);
    break;
  case Next32::GMEMFAOP_S:
  case Next32::GMEMFAOP_D:
    expandFetchAndOp(MBB, MI);
    break;
  case Next32::GMEMCAS_S:
  case Next32::GMEMCAS_D:
    expandCompareAndSwap(MBB, MI);
    break;
  case Next32::ADDFLAGS:
  case Next32::SUBFLAGS:
  case Next32::ADCFLAGS:
  case Next32::SBBFLAGS:
    expandArithWithFlags(MBB, MI);
    break;
  case Next32::SET_TID:
    expandSetTID(MBB, MI);
    break;
  case Next32::FRAME_OFFSET_WRAPPER:
    expandFrameOffsetWrapper(MBB, MI);
  }

  MBB.erase(MI);
  return true;
}
