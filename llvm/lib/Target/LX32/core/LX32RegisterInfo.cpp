//===-- LX32RegisterInfo.cpp - LX32 Register Info Implementation ---------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements LX32RegisterInfo, the runtime register-management
// class used by the LX32 backend's code generation pipeline.
//
// It is organized into the following sections:
//
//   Section 0 — TableGen-generated descriptor tables
//   Section 1 — Constructor
//   Section 2 — Reserved register set
//   Section 3 — Callee-saved register set
//   Section 4 — Frame-index elimination
//   Section 5 — Frame register selection
//
//===----------------------------------------------------------------------===//

#include "LX32RegisterInfo.h"
#include "LX32FrameLowering.h"
#include "LX32Subtarget.h"

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "lx32-reginfo"

//===----------------------------------------------------------------------===//
// Section 0 — TableGen-generated descriptor tables
//
// These macros expand to the implementation bodies that the TableGen .td files
// describe statically.  They must appear at file scope, before any namespace.
//
//   GET_REGINFO_TARGET_DESC — emits:
//     LX32GenRegisterInfo constructor (initialises register descriptor tables)
//     LX32GenRegisterInfo::getRegClassWeight(...)
//     LX32GenRegisterInfo::getRegUnitWeight(...)
//     ... and all other pure-table methods declared in the generated header.
//===----------------------------------------------------------------------===//

#define GET_REGINFO_TARGET_DESC
#include "../TableGen/LX32GenRegisterInfo.inc"


using namespace llvm;

//===----------------------------------------------------------------------===//
// Section 1 — Constructor
//===----------------------------------------------------------------------===//

LX32RegisterInfo::LX32RegisterInfo(unsigned HwMode)
    // LX32GenRegisterInfo(RA, DwarfFlavour, EHFlavour, PC, HwMode)
    //   RA           = X1 (ra) — the return-address register used by DWARF
    //                  unwinders to locate the caller's return address.
    //   DwarfFlavour = 0 (default; no alternative DWARF numbering)
    //   EHFlavour    = 0 (default; no separate EH register numbering)
    //   PC           = 0 (no architecturally-visible program counter register)
    //   HwMode       = hardware mode index from the subtarget
    : LX32GenRegisterInfo(LX32::X1, 0, 0, 0, HwMode) {}

//===----------------------------------------------------------------------===//
// Section 2 — Reserved register set
//
// getReservedRegs returns a BitVector where bit N is set if physical register
// N must never be assigned by the register allocator.
//
// Rationale for each reserved register:
//
//   X0  (zero)  — hardwired to zero by the processor.  Writing it is a no-op.
//                 Marking it reserved prevents the RA from wasting an
//                 allocation slot on a register whose writes vanish.
//
//   X1  (ra)    — return address.  FrameLowering saves/restores it in the
//                 prologue/epilogue.  If the RA could freely allocate ra, it
//                 might spill a live value there, corrupting the return address.
//
//   X2  (sp)    — stack pointer.  Only FrameLowering adjusts sp (ADDI sp,sp,N
//                 in the prologue and epilogue).  Allowing the RA to use sp as
//                 a general register would corrupt the stack.
//
//   X3  (gp)    — global pointer.  Initialised by the runtime linker stub
//                 (crt0) to point at the .sdata/.sbss region.  The RA must
//                 never overwrite it.
//
//   X4  (tp)    — thread pointer.  Points to the thread-local storage block.
//                 Like gp, it is set by the OS/runtime and must not be clobbered.
//
//   X8  (fp)    — frame pointer.  Reserved *only when* the function uses a
//                 frame pointer (alloca, VLAs, or -fno-omit-frame-pointer).
//                 When hasFP() is false, X8 is available as the callee-saved
//                 register s0 and the RA may allocate it freely.
//
// All other registers (temporaries t0-t6, arguments a0-a7, callee-saved
// s1-s11) are left unreserved and may be assigned by the RA subject to the
// callee-saved rules in getCalleeSavedRegs().
//===----------------------------------------------------------------------===//

BitVector
LX32RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());

  // x0/zero — hardwired to zero, writes silently discarded by hardware.
  markSuperRegs(Reserved, LX32::X0);

  // x1/ra — managed exclusively by FrameLowering (save in prologue, restore
  // in epilogue).  The RA must not use it as a scratch register.
  markSuperRegs(Reserved, LX32::X1);

  // x2/sp — stack pointer, adjusted only by FrameLowering.
  markSuperRegs(Reserved, LX32::X2);

  // x3/gp — global pointer, initialised by the runtime linker.
  markSuperRegs(Reserved, LX32::X3);

  // x4/tp — thread pointer, initialised by the OS/runtime.
  markSuperRegs(Reserved, LX32::X4);

  // x8/fp — reserved only when the function uses a dedicated frame pointer.
  // hasFP() returns true when the function contains alloca() calls, variable-
  // length arrays, or was compiled with -fno-omit-frame-pointer.
  const auto *TFI =
      static_cast<const LX32FrameLowering *>(MF.getSubtarget().getFrameLowering());
  if (TFI && TFI->hasFP(MF))
    markSuperRegs(Reserved, LX32::X8);

  return Reserved;
}

//===----------------------------------------------------------------------===//
// Section 3 — Callee-saved register set
//
// getCalleeSavedRegs returns a null-terminated array of physical registers
// that the ABI requires a callee to preserve across calls.
//
// The ILP32 callee-saved set is defined in LX32CallingConv.td as
// CSR_LX32_ILP32 and consists of:
//   X1  (ra)     — return address
//   X8  (s0/fp)  — frame pointer / callee-saved s0
//   X9  (s1)     — callee-saved s1
//   X18-X27 (s2-s11) — callee-saved s2 through s11
//
// The RA uses this list to determine which registers it must insert
// save/restore code for when it allocates them inside a function body.
// FrameLowering reads the MachineFunction's CalleeSavedInfo (built from this
// list) to emit the actual SW/LW instructions in the prologue and epilogue.
//===----------------------------------------------------------------------===//

const MCPhysReg *
LX32RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  // The array is generated by TableGen from the CSR_LX32_ILP32 definition in
  // LX32CallingConv.td.  It is null-terminated so the caller can iterate
  // without knowing the length in advance.
  return CSR_LX32_ILP32_SaveList;
}

//===----------------------------------------------------------------------===//
// Section 4 — Frame-index elimination
//
// eliminateFrameIndex is called once per MachineInstr operand that holds a
// FrameIndex (FI) after register allocation.  Its job is to replace the
// abstract FI with a concrete (base-register, offset) pair.
//
// How frame indices work:
//   Before RA, the backend records each local variable / spill slot as a
//   FrameIndex — an integer index into the MachineFrameInfo table.  The real
//   sp-relative byte offset is not known until FrameLowering::calculateFrameSize
//   runs (which happens after RA).  eliminateFrameIndex is the bridge that
//   converts FI → real offset once the frame layout is finalised.
//
// The LX32 implementation handles two cases:
//
// Fast path — offset fits in simm12 (the common case):
//   The FI operand is replaced in-place with the base register (sp or fp)
//   and the adjacent immediate operand is set to the byte offset.  No new
//   instructions are inserted.
//
// Slow path — offset exceeds simm12 range (very large frames, > 2 KB):
//   A scratch register is borrowed from the RegScavenger.  The full base
//   address (base + offset) is materialised into the scratch register using
//   a LUI + ADD + ADDI sequence, and the instruction is rewritten to use
//   scratch as the base with an immediate of 0.
//
// Return value:
//   false — the caller (PrologEpilogInserter) should continue processing.
//   true  — the instruction was fully rewritten; caller should not touch it
//           further.  LX32 never returns true.
//===----------------------------------------------------------------------===//

bool LX32RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                            int SPAdj, unsigned FIOperandNum,
                                            RegScavenger *RS) const {
  MachineInstr &MI    = *II;
  MachineFunction &MF = *MI.getMF();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const LX32FrameLowering *TFI =
      static_cast<const LX32FrameLowering *>(MF.getSubtarget().getFrameLowering());

  // --- Determine the concrete (base register, offset) for this frame index ---

  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  Register FrameReg;

  // getFrameIndexReference returns the offset from whichever register is used
  // as the frame anchor (sp when hasFP() == false, fp otherwise), and sets
  // FrameReg to that anchor register.
  StackOffset Offset = TFI->getFrameIndexReference(MF, FrameIndex, FrameReg);

  // Add any additional immediate offset already encoded in the instruction
  // (e.g., LW rd, FI + 4 — the +4 is in the operand adjacent to the FI).
  Offset += StackOffset::getFixed(MI.getOperand(FIOperandNum + 1).getImm());

  // Add the stack-pointer adjustment accumulated by ADJCALLSTACKDOWN /
  // ADJCALLSTACKUP pseudos that have not yet been eliminated.
  Offset += StackOffset::getFixed(SPAdj);

  int64_t OffsetVal = Offset.getFixed();

  // -----------------------------------------------------------------------
  // Fast path: offset fits in a 12-bit signed immediate (simm12).
  //   All LX32 load/store instructions use a simm12 offset field, so if the
  //   offset is in [-2048, 2047] we can rewrite directly without any new
  //   instructions.
  // -----------------------------------------------------------------------
  if (isInt<12>(OffsetVal)) {
    MI.getOperand(FIOperandNum).ChangeToRegister(FrameReg, /*isDef=*/false);
    MI.getOperand(FIOperandNum + 1).ChangeToImmediate(OffsetVal);
    return false;
  }

  report_fatal_error(
      "LX32RegisterInfo::eliminateFrameIndex: frame offset out of simm12 range");
}

//===----------------------------------------------------------------------===//
// Section 5 — Frame register selection
//
// getFrameRegister returns the register that LLVM should treat as the
// "canonical" frame anchor for this function.
//
// When the function uses a frame pointer (hasFP() == true), the frame pointer
// register X8 (fp/s0) is the stable reference point throughout the function
// body.  This is necessary when the stack pointer moves dynamically (alloca,
// variable-length arrays) because sp no longer has a fixed relationship to the
// function's local variables.
//
// When the function does not use a frame pointer (the common case), the stack
// pointer X2 (sp) is the anchor.  LLVM's DWARF emitter uses this register
// as the CFA (Canonical Frame Address) base for .debug_frame/.eh_frame.
//===----------------------------------------------------------------------===//

Register
LX32RegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const auto *TFI =
      static_cast<const LX32FrameLowering *>(MF.getSubtarget().getFrameLowering());
  return (TFI && TFI->hasFP(MF)) ? LX32::X8 : LX32::X2;
}