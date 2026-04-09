//===-- LX32InstrInfo.cpp - LX32 Instruction Info Implementation ---------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements LX32InstrInfo, the runtime instruction-management class
// used by the LX32 backend's code generation pipeline.
//
// It is organized into the following sections:
//
//   Section 0 — TableGen-generated instruction descriptor tables
//   Section 1 — Constructor
//   Section 2 — Physical register copy
//   Section 3 — Stack-slot spill and reload
//   Section 4 — Post-RA pseudo expansion
//   Section 5 — Stack-adjustment utility
//
//===----------------------------------------------------------------------===//

#include "LX32InstrInfo.h"
#include "LX32Subtarget.h"

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "lx32-instrinfo"

//===----------------------------------------------------------------------===//
// Section 0 — TableGen-generated instruction descriptor tables
//
// GET_INSTRINFO_MC_DESC expands to the implementation of:
//   LX32GenInstrInfo::LX32GenInstrInfo()  — fills the MCInstrDesc tables
//   InitLX32MCInstrInfo()                 — used by the MC layer
// These must appear at file scope before any namespace.
//===----------------------------------------------------------------------===//

#define GET_INSTRINFO_CTOR_DTOR
#include "../TableGen/LX32GenInstrInfo.inc"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Section 1 — Constructor
//===----------------------------------------------------------------------===//

LX32InstrInfo::LX32InstrInfo(const LX32Subtarget &STI)
    // LX32GenInstrInfo(CallFrameSetupOpcode, CallFrameDestroyOpcode)
    //   ADJCALLSTACKDOWN — decrements sp before a call to reserve argument space.
    //   ADJCALLSTACKUP   — restores sp after a call.
    //   Both are defined as codegen-only pseudos in Section 12 of
    //   LX32InstrInfo.td and expanded by LX32FrameLowering::
    //   eliminateCallFramePseudoInstr (Day 8).
    : LX32GenInstrInfo(STI, *STI.getRegisterInfo(),
                       LX32::ADJCALLSTACKDOWN, LX32::ADJCALLSTACKUP),
      STI(STI) {}

//===----------------------------------------------------------------------===//
// Section 2 — Physical register copy
//
// LX32 has no dedicated MOV instruction.  A register copy is expressed as:
//
//   ADD rd, rs, x0
//
// where x0 is the zero register (always 0), so the result is simply the
// value of rs.  This is the standard idiom for register copies on LX32 base-
// derived ISAs.
//
// The RegisterAllocator emits copyPhysReg whenever it needs to move a value
// between two physical registers — for example, when placing an argument into
// a calling-convention register or when a value is live in a non-preferred
// register and must be relocated.
//===----------------------------------------------------------------------===//

void LX32InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MBBI,
                                   const DebugLoc &DL, Register DstReg,
                                   Register SrcReg, bool KillSrc,
                                   bool RenamableDest,
                                   bool RenamableSrc) const {
  // LX32 v1 only has integer GPRs.  If either register is outside the GPR
  // class, the backend has an internal inconsistency.
  if (!LX32::GPRRegClass.contains(DstReg, SrcReg))
    llvm_unreachable("LX32InstrInfo::copyPhysReg: unsupported register class "
                     "(only GPR → GPR copies are legal in LX32 v1)");

  // ADD DstReg, SrcReg, x0
  //   SrcReg is marked Kill if KillSrc is true, which tells downstream passes
  //   that SrcReg is no longer live after this instruction.
  BuildMI(MBB, MBBI, DL, get(LX32::ADD), DstReg)
      .addReg(SrcReg, getKillRegState(KillSrc))
      .addReg(LX32::X0);
}

//===----------------------------------------------------------------------===//
// Section 3 — Stack-slot spill and reload
//
// When the register allocator runs out of physical registers, it must spill
// the contents of a live virtual register to the stack and reload it later.
//
// storeRegToStackSlot:
//   Emits  SW SrcReg, FrameIndex + 0
//   The FrameIndex is an abstract placeholder; LX32RegisterInfo::
//   eliminateFrameIndex rewrites it to (sp + concrete_offset) after the
//   frame layout is computed.
//
// loadRegFromStackSlot:
//   Emits  LW DstReg, FrameIndex + 0
//   Symmetric to the store.  After eliminateFrameIndex, this becomes
//   LW DstReg, concrete_offset(sp).
//
// Both functions assert that the register class is GPR (i32).  Other register
// classes (FPR, vector) do not exist in LX32 v1 and would require different
// store/load widths.
//===----------------------------------------------------------------------===//

void LX32InstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MBBI,
                                         Register SrcReg, bool isKill,
                                         int FrameIndex,
                                         const TargetRegisterClass *RC,
                                         Register VReg,
                                         MachineInstr::MIFlag Flags) const {
  // Only 32-bit GPR spills are supported in LX32 v1.
  assert(RC == &LX32::GPRRegClass &&
         "storeRegToStackSlot: only GPR class is supported");

  DebugLoc DL =
      MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  // SW SrcReg, FrameIndex + 0
  //   The offset of 0 is a placeholder that eliminateFrameIndex will replace
  //   with the real sp-relative byte offset once the frame layout is known.
  BuildMI(MBB, MBBI, DL, get(LX32::SW))
      .addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FrameIndex)
      .addImm(0)
      .setMIFlag(Flags);
}

void LX32InstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI,
                                          Register DstReg, int FrameIndex,
                                          const TargetRegisterClass *RC,
                                           Register VReg, unsigned SubReg,
                                           MachineInstr::MIFlag Flags) const {
  assert(RC == &LX32::GPRRegClass &&
         "loadRegFromStackSlot: only GPR class is supported");

  DebugLoc DL =
      MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  // LW DstReg, FrameIndex + 0
  BuildMI(MBB, MBBI, DL, get(LX32::LW), DstReg)
      .addFrameIndex(FrameIndex)
      .addImm(0)
      .setMIFlag(Flags);
}

//===----------------------------------------------------------------------===//
// Section 4 — Post-RA pseudo expansion
//
// expandPostRAPseudo is called by the PseudoExpansionPass after register
// allocation.  It converts pseudo-instructions (which have no hardware
// encoding) to real instructions that the MCCodeEmitter can emit.
//
// The pseudos expanded here are defined as isCodeGenOnly in LX32InstrInfo.td.
// After expansion the pseudo is erased from the MBB, so the function returns
// true to signal that the iterator has been invalidated.
//
// PseudoRET:
//   The architecturally-correct return sequence on LX32 is:
//     JALR x0, ra, 0
//   x0 as destination discards the return-address link value (we are jumping
//   to the return address, not saving a new one).  The instruction reads X1
//   (ra) as declared in the Uses list of PseudoRET in LX32InstrInfo.td.
//
// PseudoNOP:
//   The canonical NOP encoding is ADDI x0, x0, 0 — add zero to the zero
//   register and discard the result.  Because x0 is hardwired to zero and
//   the destination is x0, this instruction has no observable effect.
//===----------------------------------------------------------------------===//

bool LX32InstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  auto isCondBranchOpc = [](unsigned Opc) {
    switch (Opc) {
    default:
      return false;
    case LX32::PseudoBEQ:
    case LX32::PseudoBNE:
    case LX32::PseudoBLT:
    case LX32::PseudoBGE:
    case LX32::PseudoBLTU:
    case LX32::PseudoBGEU:
    case LX32::BEQ:
    case LX32::BNE:
    case LX32::BLT:
    case LX32::BGE:
    case LX32::BLTU:
    case LX32::BGEU:
      return true;
    }
  };

  auto getBranchTargetMBB = [](const MachineInstr &BrMI) -> MachineBasicBlock * {
    for (const MachineOperand &MO : BrMI.operands())
      if (MO.isMBB())
        return MO.getMBB();
    return nullptr;
  };

  auto expandCondBr = [&](unsigned RealOpc) {
    SmallVector<MachineOperand, 4> RegOps;
    const MachineOperand *TargetMBBOp = nullptr;
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isMBB() && !TargetMBBOp) {
        TargetMBBOp = &MO;
        continue;
      }
      if (!MO.isReg() || MO.getReg() == 0 || MO.isImplicit())
        continue;
      RegOps.push_back(MO);
    }
    if (RegOps.size() < 2)
      report_fatal_error("lx32: malformed conditional-branch pseudo operands");
    if (!TargetMBBOp)
      report_fatal_error("lx32: conditional branch pseudo missing target MBB");

    auto MIB = BuildMI(MBB, MI, DL, get(RealOpc));
    MIB->addOperand(RegOps[RegOps.size() - 2]);
    MIB->addOperand(RegOps[RegOps.size() - 1]);
    MIB->addOperand(*TargetMBBOp);
    MBB.erase(MI);
    return true;
  };

  switch (MI.getOpcode()) {
  default:
    return false; // Unknown pseudo — leave it for another pass.

  case LX32::PseudoRET:
    // Expand to: JALR x0, ra, 0
    //   x0 (define, dead) — result register; marked Dead because nobody reads
    //   it (x0 is always zero anyway, but marking it Dead lets the register
    //   allocator know the result is intentionally discarded).
    //   X1 (ra, kill)     — the return address; Kill means ra is consumed.
    //   0                 — no offset added to the return address.
    BuildMI(MBB, MI, DL, get(LX32::JALR))
        .addReg(LX32::X0, RegState::Define | RegState::Dead)
        .addReg(LX32::X1, RegState::Kill)
        .addImm(0);
    MBB.erase(MI);
    return true;

  case LX32::PseudoBR: {
    // Expand to: JAL x0, target
    //   x0 (define, dead) — result register; we discard the PC+4 return address
    //   target            — the branch target (simm21 immediate)
    MachineBasicBlock *TargetMBB = getBranchTargetMBB(MI);

    MachineBasicBlock *CondTarget = nullptr;
    for (MachineInstr *Prev = MI.getPrevNode(); Prev; Prev = Prev->getPrevNode()) {
      if (Prev->isDebugInstr())
        continue;
      if (isCondBranchOpc(Prev->getOpcode()))
        CondTarget = getBranchTargetMBB(*Prev);
      break;
    }

    if (CondTarget) {
      MachineBasicBlock *OtherSucc = nullptr;
      for (MachineBasicBlock *Succ : MBB.successors()) {
        if (Succ == CondTarget)
          continue;
        if (!OtherSucc) {
          OtherSucc = Succ;
          continue;
        }
        if (OtherSucc != Succ)
          report_fatal_error("lx32: ambiguous non-conditional successor for PseudoBR");
      }
      if (OtherSucc)
        TargetMBB = OtherSucc;
    }

    if (!TargetMBB) {
      if (MBB.succ_empty())
        report_fatal_error("lx32: branch pseudo has no branch target");
      if (MBB.succ_size() > 1)
        report_fatal_error("lx32: branch pseudo target is ambiguous without explicit MBB operand");
      TargetMBB = *MBB.succ_begin();
    }

    BuildMI(MBB, MI, DL, get(LX32::JAL))
        .addReg(LX32::X0, RegState::Define | RegState::Dead)
        .addMBB(TargetMBB);
    MBB.erase(MI);
    return true;
  }

  case LX32::PseudoBEQ:
    return expandCondBr(LX32::BEQ);
  case LX32::PseudoBNE:
    return expandCondBr(LX32::BNE);
  case LX32::PseudoBLT:
    return expandCondBr(LX32::BLT);
  case LX32::PseudoBGE:
    return expandCondBr(LX32::BGE);
  case LX32::PseudoBLTU:
    return expandCondBr(LX32::BLTU);
  case LX32::PseudoBGEU:
    return expandCondBr(LX32::BGEU);

  case LX32::PseudoNOP:
    // Expand to: ADDI x0, x0, 0
    //   Both source and destination are x0, immediate is 0.  The instruction
    //   has no effect but occupies one instruction slot for alignment or
    //   pipeline padding purposes.
    BuildMI(MBB, MI, DL, get(LX32::ADDI))
        .addReg(LX32::X0, RegState::Define | RegState::Dead)
        .addReg(LX32::X0, RegState::Kill)
        .addImm(0);
    MBB.erase(MI);
    return true;
  }
}

//===----------------------------------------------------------------------===//
// Section 5 — Stack-adjustment utility
//
// adjustReg is a helper for FrameLowering.  It emits the instruction(s)
// needed to compute DstReg = SrcReg + Val.
//
// When Val fits in a 12-bit signed immediate (simm12, range [-2048, 2047]):
//   ADDI DstReg, SrcReg, Val        — one instruction
//
// When Val is larger (rare; happens with frames > 2 KB):
//   LUI  scratch, hi20(Val)
//   ADD  DstReg, SrcReg, scratch
//   ADDI DstReg, DstReg, lo12(Val)  — omitted when lo12 == 0
//
// The slow path requires that there is a free scratch register at the
// insertion point.  FrameLowering is responsible for ensuring this (it
// calls adjustReg before any callee-saved registers have been spilled, so
// temporaries are available).
//
// The hi20/lo12 decomposition uses the same +0x800 bias as the constant-
// materialisation patterns in LX32InstrInfo.td Section 10, compensating for
// the sign extension that ADDI applies to its 12-bit immediate.
//===----------------------------------------------------------------------===//

void LX32InstrInfo::adjustReg(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI,
                                const DebugLoc &DL, Register DstReg,
                                Register SrcReg, int64_t Val,
                                MachineInstr::MIFlag Flag) const {
  if (Val == 0) {
    // Zero adjustment: if DstReg == SrcReg, nothing to do.
    // If DstReg != SrcReg, emit a register copy (ADD DstReg, SrcReg, x0).
    if (DstReg != SrcReg)
      copyPhysReg(MBB, MBBI, DL, DstReg, SrcReg, /*KillSrc=*/false);
    return;
  }

  // Fast path: adjustment fits in simm12.
  if (isInt<12>(Val)) {
    BuildMI(MBB, MBBI, DL, get(LX32::ADDI), DstReg)
        .addReg(SrcReg)
        .addImm(Val)
        .setMIFlag(Flag);
    return;
  }

  // Slow path: adjustment does not fit in simm12.
  // This is only expected for functions with stack frames larger than 2 KB.
  assert(isInt<32>(Val) && "Frame adjustment exceeds 32-bit range");

  // Decompose Val into hi20 (upper bits) and lo12 (lower 12, sign-extended).
  int64_t Hi20 = ((Val + 0x800) >> 12) & 0xFFFFF;
  int64_t Lo12 = Val - (Hi20 << 12);

  // Use a caller-saved temporary as scratch.  t0 (X5) is used here because
  // adjustReg is only called from FrameLowering where t0 is not live.
  // A more robust implementation would ask the RegScavenger, but for the
  // current skeleton the direct choice is sufficient.
  Register Scratch = LX32::X5; // t0 — caller-saved, safe in prologue/epilogue

  // LUI scratch, hi20
  BuildMI(MBB, MBBI, DL, get(LX32::LUI), Scratch)
      .addImm(Hi20)
      .setMIFlag(Flag);

  // ADD DstReg, SrcReg, scratch
  BuildMI(MBB, MBBI, DL, get(LX32::ADD), DstReg)
      .addReg(SrcReg)
      .addReg(Scratch)
      .setMIFlag(Flag);

  // ADDI DstReg, DstReg, lo12  (skip if lo12 == 0 to avoid a redundant NOP)
  if (Lo12 != 0) {
    BuildMI(MBB, MBBI, DL, get(LX32::ADDI), DstReg)
        .addReg(DstReg)
        .addImm(Lo12)
        .setMIFlag(Flag);
  }
}