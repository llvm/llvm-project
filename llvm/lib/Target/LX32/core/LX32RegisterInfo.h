//===-- LX32RegisterInfo.h - LX32 Register Info Interface ----------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file declares LX32RegisterInfo, the runtime register-management class
// used by LLVM's code generation pipeline.
//
// It is organized into the following sections:
//
//   Section 0 — Role in the backend pipeline
//   Section 1 — Class declaration and ABI hooks
//   Section 2 — Frame-index elimination
//   Section 3 — Utility and policy overrides
//
//===----------------------------------------------------------------------===//
//
// Section 0 — Role in the backend pipeline
//
// LLVM's register allocator (RA) needs two things the TableGen .td file alone
// cannot provide:
//
//   1. A runtime BitVector of *reserved* registers — registers the RA must
//      never assign to a virtual register.  Examples: x0 (hardwired zero),
//      x2 (sp), x3 (gp), x4 (tp).  These are set in getReservedRegs().
//
//   2. A way to *resolve* abstract frame-index references after allocation.
//      Before RA, the backend uses symbolic frame indices (FI) instead of
//      concrete sp+offset addresses.  After RA, eliminateFrameIndex() rewrites
//      each FI reference to "base register + offset", where the base is sp or
//      fp depending on whether the function uses a frame pointer.
//
// The TableGen-generated base class (LX32GenRegisterInfo) provides the static
// descriptor tables — register enumeration, register classes, callee-saved
// sets, DWARF register numbers, etc.  LX32RegisterInfo adds the runtime logic
// on top of that static foundation.
//
//===----------------------------------------------------------------------===//

#ifndef LX32_LX32REGISTERINFO_H
#define LX32_LX32REGISTERINFO_H

#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

// Pull in generated register enums first so X0..X31 and *RegClassID are
// visible everywhere this header is included.
#define GET_REGINFO_ENUM
#include "../TableGen/LX32GenRegisterInfo.inc"

// Pull in the TableGen-generated LX32GenRegisterInfo class declaration.
// GET_REGINFO_HEADER emits:
//   struct LX32GenRegisterInfo : public TargetRegisterInfo { ... };
//   namespace LX32 { extern const TargetRegisterClass GPRRegClass; ... }
#define GET_REGINFO_HEADER
#include "../TableGen/LX32GenRegisterInfo.inc"

namespace llvm {

class RegScavenger;

//===----------------------------------------------------------------------===//
// Section 1 — Class declaration and ABI hooks
//===----------------------------------------------------------------------===//

struct LX32RegisterInfo : public LX32GenRegisterInfo {
  // Construct with the hardware mode index produced by TableGen.
  // For LX32 v1 there is only one hardware mode (index 0), but passing the
  // correct value keeps the API compatible with future multi-mode extensions.
  explicit LX32RegisterInfo(unsigned HwMode);

  //===--------------------------------------------------------------------===//
  // ABI / calling-convention hooks
  //===--------------------------------------------------------------------===//

  // getCalleeSavedRegs — return the set of registers a function must preserve.
  //
  // The returned array is parallel with LX32CallingConv.td's CSR_* definitions:
  //   CSR_LX32_ILP32 : { X1(ra), X8(s0/fp), X9(s1), X18..X27(s2..s11) }
  //
  // The RA uses this list to decide which registers it must spill/restore in
  // the function prologue/epilogue if it wants to use them.
  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;

  // getReservedRegs — return the BitVector of always-reserved registers.
  //
  // Reserved registers are never assigned by the RA, regardless of liveness.
  // See Section 2 of LX32RegisterInfo.cpp for the complete rationale for each
  // reserved register.
  BitVector getReservedRegs(const MachineFunction &MF) const override;

  //===--------------------------------------------------------------------===//
  // Section 2 — Frame-index elimination (see .cpp for full implementation)
  //===--------------------------------------------------------------------===//

  // eliminateFrameIndex — rewrite a frame-index reference to base+offset.
  //
  // Called once for every MachineInstr operand that holds a FrameIndex after
  // register allocation.  The implementation in LX32RegisterInfo.cpp handles
  // two sub-cases:
  //
  //   Fast path (common): offset fits in simm12
  //     Replace FI operand with the base register (sp or fp) and set the
  //     adjacent immediate operand to the concrete byte offset.
  //
  //   Slow path (large frame): offset does not fit in simm12
  //     Use the RegScavenger to find a free scratch register, materialise
  //     the full address with LUI+ADD+ADDI, and rewrite the instruction to
  //     use that scratch register as the base with immediate 0.
  //
  // Returns false in both cases (true would mean the instruction was
  // completely rewritten and the caller should not touch it further — LX32
  // never needs that path).
  bool eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override;

  // getFrameRegister — return the register that acts as the frame anchor.
  //
  // Returns X8 (fp) when the function uses a frame pointer (alloca, VLAs,
  // -fno-omit-frame-pointer), or X2 (sp) otherwise.  LLVM's DWARF CFI
  // directives use this to describe where the caller's frame is.
  Register getFrameRegister(const MachineFunction &MF) const override;

  //===--------------------------------------------------------------------===//
  // Section 3 — Utility and policy overrides
  //===--------------------------------------------------------------------===//

  // requiresRegisterScavenging — allow the RA to use a RegScavenger.
  //
  // Must be true so that eliminateFrameIndex's slow path (large-frame offsets)
  // can call RS->scavengeRegisterBackwards() to borrow a scratch register.
  // Without scavenging, large frames would have no way to materialise
  // addresses that exceed the 12-bit simm range.
  bool requiresRegisterScavenging(const MachineFunction &MF) const override {
    return true;
  }

  // requiresFrameIndexScavenging — enable frame-index scavenging.
  //
  // When true, the RA pre-allocates a scavenge slot in the frame so that
  // eliminateFrameIndex always has a valid slot to spill the scratch register
  // into, even in the worst case where no register is free at the point of the
  // large-frame access.
  bool requiresFrameIndexScavenging(const MachineFunction &MF) const override {
    return true;
  }

  // getPointerRegClass — return the register class for pointer-typed values.
  //
  // All pointer operations on LX32 use the 32-bit GPR class.  This override
  // ensures that the RA and legalization layers select GPR for i32* types.
  const TargetRegisterClass *
  getPointerRegClass(unsigned Kind = 0) const override {
    return &LX32::GPRRegClass;
  }
};

} // namespace llvm

#endif // LX32_LX32REGISTERINFO_H
