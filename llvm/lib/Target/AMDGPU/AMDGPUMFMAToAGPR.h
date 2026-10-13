//===-- AMDGPUMFMAToAGPR.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Shared analysis for moving MFMA instructions from VGPR form to AGPR form:
/// walk the accumulator chain, decide whether its registers may be placed in
/// AGPRs, and flip the opcodes and register classes. Used by
/// RewriteMFMAFormStage before allocation and AMDGPURewriteAGPRCopyMFMA after.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUMFMATOAGPR_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUMFMATOAGPR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

namespace llvm {

class MachineFunction;
class MachineInstr;
class MachineOperand;
class MachineRegisterInfo;
class SIInstrInfo;
class SIRegisterInfo;

namespace AMDGPU {

/// \returns true if \p MI is an MFMA that has an AGPR-form encoding.
bool hasAGPRForm(const MachineInstr &MI, const SIInstrInfo &TII);

/// An accumulator chain is the transitive closure of MFMAs linked through their
/// dst and src2 operands. Both users work at chain granularity: converting one
/// member alone needs a bridge copy at every boundary.
class MFMAToAGPR {
public:
  MFMAToAGPR(MachineFunction &MF, const SIInstrInfo &TII,
             const SIRegisterInfo &TRI, MachineRegisterInfo &MRI)
      : MF(MF), TII(TII), TRI(TRI), MRI(MRI) {}

  /// Restrict which MFMAs are being converted. MFMAs in \p S do not constrain a
  /// recolor; a convertible MFMA outside \p S stays in VGPR form and pins its
  /// operands to the VGPR bank. Null, the default, means all of them convert.
  void setActiveSet(const SmallPtrSetImpl<MachineInstr *> *S) { ActiveSet = S; }

  /// Decide whether \p Reg alone may leave the VGPR bank, by narrowing its
  /// AV_* superclass by the constraints of its defs and uses. Landing back on
  /// the original VGPR class means some operand rejected AGPRs. Operands in
  /// \p ExemptOps are ignored; use this for uses the caller will redirect to a
  /// bridge copy. If given, \p Chain and \p ChainRegs collect the MFMAs
  /// touching \p Reg and their other accumulator registers.
  ///
  /// \returns false if \p Reg is physical, has no AGPR equivalent, or some def
  /// or use forbids the class.
  bool canRecolorToAGPR(Register Reg, ArrayRef<MachineOperand *> ExemptOps = {},
                        SmallVectorImpl<MachineInstr *> *Chain = nullptr,
                        SmallSetVector<Register, 4> *ChainRegs = nullptr) const;

  /// Walk the accumulator chain from \p StartReg, collecting the MFMAs whose
  /// opcode has to change into \p Chain and the registers whose class has to
  /// change into \p ChainRegs, which must already contain \p StartReg.
  ///
  /// \returns false if any def or use forbids the class, or if the chain
  /// reaches a physical register.
  bool canMoveChainToAGPR(Register StartReg,
                          SmallVectorImpl<MachineInstr *> &Chain,
                          SmallSetVector<Register, 4> &ChainRegs,
                          ArrayRef<MachineOperand *> ExemptOps = {}) const;

  /// \returns true if \p MO can hold a register of class \p RC once its
  /// instruction is in its final form. An MFMA being converted is checked
  /// against its AGPR form, where dst and src2 take an AGPR but the scale
  /// variant's scale operands still do not.
  bool operandAcceptsClass(const MachineOperand &MO,
                           const TargetRegisterClass *RC) const;

  /// Partition every convertible MFMA in the function into accumulator chains.
  /// Chains and their members come back in program order.
  SmallVector<SmallVector<MachineInstr *, 8>, 16> partitionChains() const;

  /// Switch \p MFMAs to their AGPR-form opcode and \p Regs to their equivalent
  /// AGPR class. Appends the replaced opcodes to \p OrigOpcodes, if given, so
  /// the change can be undone.
  void applyAGPRForm(ArrayRef<MachineInstr *> MFMAs, ArrayRef<Register> Regs,
                     SmallVectorImpl<unsigned> *OrigOpcodes = nullptr) const;

  /// Undo applyAGPRForm().
  void revertToVGPRForm(ArrayRef<MachineInstr *> MFMAs,
                        ArrayRef<unsigned> OrigOpcodes,
                        ArrayRef<Register> Regs) const;

  /// Scoped applyAGPRForm(): reverts on destruction unless release() is called,
  /// so a caller can measure pressure in AGPR form without unwinding by hand.
  class FormScope {
  public:
    FormScope(const MFMAToAGPR &Impl, ArrayRef<MachineInstr *> MFMAs,
              ArrayRef<Register> Regs);
    ~FormScope();

    FormScope(const FormScope &) = delete;
    FormScope &operator=(const FormScope &) = delete;

    /// Keep the AGPR form instead of reverting.
    void release() { Active = false; }

  private:
    const MFMAToAGPR &Impl;
    ArrayRef<MachineInstr *> MFMAs;
    ArrayRef<Register> Regs;
    SmallVector<unsigned, 32> OrigOpcodes;
    bool Active = true;
  };

private:
  /// \returns true if \p MI will be in AGPR form after the conversion.
  bool isActive(const MachineInstr &MI) const;

  MachineFunction &MF;
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  MachineRegisterInfo &MRI;
  const SmallPtrSetImpl<MachineInstr *> *ActiveSet = nullptr;
};

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUMFMATOAGPR_H
