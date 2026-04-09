//===-- LX32FrameLowering.h - LX32 Frame Lowering Interface --------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file declares LX32FrameLowering, the class responsible for generating
// function prologue and epilogue code for the LX32 backend.
//
// It is organized into the following sections:
//
//   Section 0 — Role in the backend pipeline
//   Section 1 — Class declaration and constructor
//   Section 2 — Prologue and epilogue emission
//   Section 3 — Frame-pointer policy
//   Section 4 — Call-frame pseudo elimination
//   Section 5 — Frame-index reference resolution
//
//===----------------------------------------------------------------------===//
//
// Section 0 — Role in the backend pipeline
//
// FrameLowering is responsible for everything related to the activation record
// (stack frame) of a function.  Its main responsibilities are:
//
//   emitPrologue (Section 2)
//     Emits the instructions at function entry that set up the stack frame:
//       1. Decrement sp by the total frame size (ADDI sp, sp, -N).
//       2. Save all callee-saved registers that the function uses (SW reg, k(sp)).
//       3. Optionally establish fp (ADDI fp, sp, N) when a frame pointer is needed.
//       4. Emit CFI directives so debuggers and unwinders can reconstruct the
//          caller's frame.
//
//   emitEpilogue (Section 2)
//     Emits the instructions before a return that tear down the stack frame:
//       1. Restore all callee-saved registers (LW reg, k(sp)).
//       2. Restore sp (ADDI sp, sp, +N or ADDI sp, fp, 0 when dynamic).
//       Note: the actual JALR x0,ra,0 return instruction comes from the
//       PseudoRET expansion in LX32InstrInfo::expandPostRAPseudo.
//
//   hasFP (Section 3)
//     Decides whether the function needs a dedicated frame pointer register
//     (X8/fp).  Required when the frame size is not statically known — e.g.,
//     alloca(), variable-length arrays, or -fno-omit-frame-pointer.
//
//   eliminateCallFramePseudoInstr (Section 4)
//     Removes or converts ADJCALLSTACKDOWN/ADJCALLSTACKUP pseudos that bracket
//     call sequences.  When the frame has a reserved call-frame area (static
//     allocation), the pseudos are simply deleted.  Otherwise they are
//     converted to real ADDI sp, sp, ±N instructions.
//
//   getFrameIndexReference (Section 5)
//     Computes the byte offset from the frame anchor register (sp or fp) for
//     a given FrameIndex.  Called by LX32RegisterInfo::eliminateFrameIndex to
//     resolve abstract FI references to concrete addresses.
//
//===----------------------------------------------------------------------===//

#ifndef LX32_LX32FRAMELOWERING_H
#define LX32_LX32FRAMELOWERING_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

class LX32Subtarget;
class LX32InstrInfo;

//===----------------------------------------------------------------------===//
// Section 1 — Class declaration and constructor
//===----------------------------------------------------------------------===//

class LX32FrameLowering : public TargetFrameLowering {
  const LX32Subtarget &STI;

public:
  // Construct with the active subtarget.
  //
  // TargetFrameLowering constructor parameters:
  //   StackGrowsDown — LX32 uses a downward-growing stack (sp decrements on
  //                    function entry), which is the standard convention.
  //   StackAlignment  — sp must be aligned to 16 bytes before each call to
  //                    maintain ABI compatibility with functions that require
  //                    aligned arguments on the stack.
  //   LocalAreaOffset — 0, meaning local variables start immediately at the
  //                    top of the frame (no reserved area above locals).
  explicit LX32FrameLowering(const LX32Subtarget &STI);

  //===--------------------------------------------------------------------===//
  // Section 2 — Prologue and epilogue emission
  //===--------------------------------------------------------------------===//

  // emitPrologue — emit function-entry frame setup code.
  //
  // The prologue performs these steps in order:
  //
  //   1. Compute the total frame size:
  //        local variables  (assigned by the RA and MachineFrameInfo)
  //      + callee-saved slots  (one 4-byte slot per callee-saved register used)
  //      + alignment padding  (to keep sp 16-byte aligned before calls)
  //
  //   2. ADDI sp, sp, -FrameSize
  //        Decrements the stack pointer.  Uses adjustReg from LX32InstrInfo
  //        which handles the rare case where FrameSize > 2047.
  //
  //   3. .cfi_def_cfa_offset FrameSize
  //        CFI directive so the unwinder knows the new CFA offset.
  //
  //   4. For each callee-saved register in CalleeSavedInfo:
  //        SW reg, offset(sp)            — save the register
  //        .cfi_offset reg, offset-FrameSize  — CFI for the saved register
  //
  //   5. If hasFP(MF):
  //        ADDI fp, sp, FrameSize        — fp points to the pre-prologue sp
  //        .cfi_def_cfa_register fp      — update CFA base to fp
  //
  // Leaf functions (no calls, no stack usage) skip steps 1-3 and return early.
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  // emitEpilogue — emit function-exit frame teardown code.
  //
  // The epilogue performs these steps in reverse order from the prologue:
  //
  //   1. If hasFP(MF) and the frame has dynamic-size objects (alloca):
  //        ADDI sp, fp, 0    — restore sp from fp before reloading saves
  //
  //   2. For each callee-saved register (in reverse save order):
  //        LW reg, offset(sp)    — restore the register
  //
  //   3. ADDI sp, sp, +FrameSize    — restore sp to its pre-prologue value
  //
  // The return instruction (JALR x0, ra, 0) is emitted separately by
  // LX32InstrInfo::expandPostRAPseudo when it expands PseudoRET.
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  //===--------------------------------------------------------------------===//
  // Section 3 — Frame-pointer policy
  //===--------------------------------------------------------------------===//

  // hasFP — return true if this function requires a dedicated frame pointer.
  //
  // A frame pointer is required when:
  //   - The frame has variable-size objects (alloca, VLAs).  Without fp, sp
  //     moves during the function body and locals can no longer be addressed
  //     as sp + fixed_offset.
  //   - The function takes the address of its own frame (llvm.frameaddress).
  //   - Frame pointer elimination is disabled (-fno-omit-frame-pointer or the
  //     MachineTargetOptions::DisableFramePointerElim flag).
  bool hasFPImpl(const MachineFunction &MF) const override;

  // hasReservedCallFrame — return true if the function pre-allocates space
  // for outgoing call arguments in the frame.
  //
  // When true, ADJCALLSTACKDOWN/ADJCALLSTACKUP pseudos are eliminated without
  // emitting any real instructions.  When false, they are converted to sp
  // adjustments around each call.
  //
  // LX32 reserves call-frame space only when the function does not use alloca
  // (i.e., the frame size is known at compile time).  Dynamic frames cannot
  // pre-allocate because the required space depends on the maximum call in the
  // function, which may vary at runtime.
  bool hasReservedCallFrame(const MachineFunction &MF) const override;

  //===--------------------------------------------------------------------===//
  // Section 4 — Call-frame pseudo elimination
  //===--------------------------------------------------------------------===//

  // eliminateCallFramePseudoInstr — convert or remove ADJCALLSTACKDOWN/UP.
  //
  // ADJCALLSTACKDOWN and ADJCALLSTACKUP bracket every call sequence:
  //   ADJCALLSTACKDOWN N   — sp must decrease by N before the call
  //   < argument setup >
  //   CALL target
  //   ADJCALLSTACKUP   N   — sp must be restored by N after the call
  //
  // This function is called for each such pseudo.  It either:
  //   - Deletes the pseudo (when hasReservedCallFrame is true, meaning the
  //     frame already includes the call-argument space and no runtime sp
  //     adjustment is needed), or
  //   - Converts it to ADDI sp, sp, ±N (when the frame is dynamic).
  MachineBasicBlock::iterator
  eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI) const override;

  //===--------------------------------------------------------------------===//
  // Section 5 — Frame-index reference resolution
  //===--------------------------------------------------------------------===//

  // getFrameIndexReference — compute the offset from the frame anchor to a FI.
  //
  // Called by LX32RegisterInfo::eliminateFrameIndex to resolve abstract
  // FrameIndex values to concrete (register, offset) pairs.
  //
  // Sets FrameReg to:
  //   X8 (fp) — when hasFP(MF) is true (fp is the stable reference point)
  //   X2 (sp) — otherwise
  //
  // Returns the signed byte offset from FrameReg to the start of the slot
  // identified by FI.  Positive offsets are above the current sp; negative
  // offsets are below (into the frame).
  StackOffset getFrameIndexReference(const MachineFunction &MF, int FI,
                                     Register &FrameReg) const override;

  // LX32 v1 does not use a dedicated base pointer.
};

} // namespace llvm

#endif // LX32_LX32FRAMELOWERING_H
