//===-- LX32InstrInfo.h - LX32 Instruction Info Interface ----------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file declares LX32InstrInfo, the runtime instruction-management class
// used by LLVM's code generation pipeline.
//
// It is organized into the following sections:
//
//   Section 0 — Role in the backend pipeline
//   Section 1 — Includes and generated base class
//   Section 2 — Class declaration
//   Section 3 — Physical register copy
//   Section 4 — Stack-slot spill and reload
//   Section 5 — Post-RA pseudo expansion
//   Section 6 — Stack-adjustment utility
//
//===----------------------------------------------------------------------===//
//
// Section 0 — Role in the backend pipeline
//
// LX32InstrInfo is the primary interface between LLVM's machine-code passes
// and the LX32 instruction set.  Its responsibilities are:
//
//   copyPhysReg (Section 3)
//     Called by the register allocator and copy propagation passes whenever
//     a value must be moved between two physical registers.  On LX32 there is
//     no dedicated MOV instruction; a register copy is encoded as
//     ADD rd, rs, x0 (adding zero to the source register).
//
//   storeRegToStackSlot / loadRegFromStackSlot (Section 4)
//     Called by the RA when it needs to spill a live register to the stack
//     (store) or reload it after a spill (load).  The implementation emits
//     the appropriate SW / LW instruction with a FrameIndex operand;
//     eliminateFrameIndex() in LX32RegisterInfo resolves the FI to a concrete
//     sp+offset address after the frame layout is finalised.
//
//   expandPostRAPseudo (Section 5)
//     Called by the PseudoExpansionPass after register allocation.  Expands
//     codegen-only pseudo-instructions (PseudoRET, PseudoNOP) to their real
//     machine-instruction equivalents.
//
//   adjustReg (Section 6)
//     A helper used internally by FrameLowering to emit the ADDI sp, sp, ±N
//     instruction in function prologues and epilogues.  Handles the edge case
//     where the adjustment exceeds the simm12 range by falling back to a
//     LUI+ADD+ADDI sequence.
//
// All static instruction metadata (encoding, operand types, scheduling
// attributes, DAG patterns) lives in the TableGen-generated base class
// LX32GenInstrInfo, derived from the definitions in LX32InstrInfo.td.
//
//===----------------------------------------------------------------------===//

#ifndef LX32_LX32INSTRINFO_H
#define LX32_LX32INSTRINFO_H

#include "LX32RegisterInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

// Pull in the TableGen-generated LX32 opcode enum first, then the
// LX32GenInstrInfo class declaration.
#define GET_INSTRINFO_ENUM
#include "../TableGen/LX32GenInstrInfo.inc"

// GET_INSTRINFO_HEADER emits the LX32GenInstrInfo class declaration.
#define GET_INSTRINFO_HEADER
#include "../TableGen/LX32GenInstrInfo.inc"

namespace llvm {

class LX32Subtarget;

//===----------------------------------------------------------------------===//
// Section 2 — Class declaration
//===----------------------------------------------------------------------===//

class LX32InstrInfo : public LX32GenInstrInfo {
  const LX32Subtarget &STI;

public:
  // Construct with a reference to the active subtarget.
  // The subtarget is stored so that register-class and feature queries can be
  // made without passing the subtarget through every call site.
  explicit LX32InstrInfo(const LX32Subtarget &STI);

  //===--------------------------------------------------------------------===//
  // Section 3 — Physical register copy
  //===--------------------------------------------------------------------===//

  // copyPhysReg — emit an instruction to copy SrcReg into DstReg.
  //
  // LX32 has no dedicated register-move instruction.  A copy is encoded as:
  //   ADD DstReg, SrcReg, x0
  // which adds the zero register (always 0) to SrcReg and stores the result
  // in DstReg.  This is the canonical register copy idiom for RV32I-derived
  // architectures.
  //
  // KillSrc: if true, the copy consumes SrcReg (marks it as killed).  The RA
  // sets this when SrcReg is no longer live after the copy, allowing downstream
  // passes to reclaim the register.
  //
  // Asserts if DstReg or SrcReg are not in the GPR register class, because LX32
  // v1 has no floating-point or vector registers.
  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                   const DebugLoc &DL, Register DstReg, Register SrcReg,
                   bool KillSrc, bool RenamableDest = false,
                   bool RenamableSrc = false) const override;

  //===--------------------------------------------------------------------===//
  // Section 4 — Stack-slot spill and reload
  //===--------------------------------------------------------------------===//

  // storeRegToStackSlot — spill SrcReg to a stack frame slot.
  //
  // Emits:  SW SrcReg, FrameIndex + 0
  // The FrameIndex is an abstract slot identifier; eliminateFrameIndex in
  // LX32RegisterInfo.cpp later rewrites it to the concrete sp+offset form.
  //
  // RC specifies the register class of SrcReg.  For LX32, the only valid
  // class is GPR (i32); wider classes do not exist yet.
  //
  // VReg is the virtual register being spilled.  It is informational only
  // (used for debug output) and does not affect code generation.
  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI, Register SrcReg,
                           bool isKill, int FrameIndex,
                           const TargetRegisterClass *RC,
                           Register VReg,
                           MachineInstr::MIFlag Flags =
                               MachineInstr::NoFlags) const override;

  // loadRegFromStackSlot — reload DstReg from a stack frame slot.
  //
  // Emits:  LW DstReg, FrameIndex + 0
  // Symmetric to storeRegToStackSlot; the FrameIndex is resolved by
  // eliminateFrameIndex after the frame layout is finalised.
  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI, Register DstReg,
                            int FrameIndex, const TargetRegisterClass *RC,
                             Register VReg, unsigned SubReg = 0,
                             MachineInstr::MIFlag Flags =
                                 MachineInstr::NoFlags) const override;

  //===--------------------------------------------------------------------===//
  // Section 5 — Post-RA pseudo expansion
  //===--------------------------------------------------------------------===//

  // expandPostRAPseudo — expand codegen-only pseudos to real instructions.
  //
  // Called by the PseudoExpansionPass after register allocation.  Returns true
  // if the pseudo was expanded (and therefore removed from the MBB), false if
  // the opcode was not recognised.
  //
  // Pseudos handled:
  //
  //   PseudoRET — function return.
  //     Expands to: JALR x0, ra, 0
  //     x0 as destination discards the link address (we are returning, not
  //     calling).  Reads X1 (ra) as the return address.
  //
  //   PseudoNOP — explicit no-operation.
  //     Expands to: ADDI x0, x0, 0
  //     The canonical NOP encoding on RV32I-derived architectures.
  bool expandPostRAPseudo(MachineInstr &MI) const override;

  //===--------------------------------------------------------------------===//
  // Section 6 — Stack-adjustment utility
  //===--------------------------------------------------------------------===//

  // adjustReg — emit an ADDI (or LUI+ADD+ADDI) to adjust a register by Val.
  //
  // Used by LX32FrameLowering to generate the prologue/epilogue stack
  // pointer adjustments:
  //   Prologue: adjustReg(MBB, MBBI, DL, sp, sp, -FrameSize, FrameSetup)
  //   Epilogue: adjustReg(MBB, MBBI, DL, sp, sp, +FrameSize, FrameDestroy)
  //
  // Fast path (|Val| <= 2047): emits a single ADDI DstReg, SrcReg, Val.
  //
  // Slow path (|Val| > 2047): emits a three-instruction sequence:
  //   LUI  scratch, hi20(Val)
  //   ADD  DstReg,  SrcReg, scratch
  //   ADDI DstReg,  DstReg, lo12(Val)   (omitted when lo12 == 0)
  // The slow path requires a scratch register that is not live at the
  // insertion point; the caller must ensure one is available.
  //
  // Flag should be MachineInstr::FrameSetup for prologue instructions and
  // MachineInstr::FrameDestroy for epilogue instructions.  These flags cause
  // LLVM to emit correct .cfi_adjust_cfa_offset directives.
  void adjustReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                 const DebugLoc &DL, Register DstReg, Register SrcReg,
                 int64_t Val, MachineInstr::MIFlag Flag) const;
};

} // namespace llvm

#endif // LX32_LX32INSTRINFO_H
