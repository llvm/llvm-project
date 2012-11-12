//===-- RegisterCoalescer.h - Register Coalescing Interface -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the abstract interface for register coalescers,
// allowing them to interact with and query register allocators.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGISTER_COALESCER_H
#define LLVM_CODEGEN_REGISTER_COALESCER_H

namespace llvm {

  class MachineInstr;
  class TargetRegisterInfo;
  class TargetRegisterClass;
  class TargetInstrInfo;

  /// CoalescerPair - A helper class for register coalescers. When deciding if
  /// two registers can be coalesced, CoalescerPair can determine if a copy
  /// instruction would become an identity copy after coalescing.
  class CoalescerPair {
    const TargetRegisterInfo &TRI;

    /// DstReg - The register that will be left after coalescing. It can be a
    /// virtual or physical register.
    unsigned DstReg;

    /// SrcReg - the virtual register that will be coalesced into dstReg.
    unsigned SrcReg;

    /// DstIdx - The sub-register index of the old DstReg in the new coalesced
    /// register.
    unsigned DstIdx;

    /// SrcIdx - The sub-register index of the old SrcReg in the new coalesced
    /// register.
    unsigned SrcIdx;

    /// Partial - True when the original copy was a partial subregister copy.
    bool Partial;

    /// CrossClass - True when both regs are virtual, and newRC is constrained.
    bool CrossClass;

    /// Flipped - True when DstReg and SrcReg are reversed from the original
    /// copy instruction.
    bool Flipped;

    /// NewRC - The register class of the coalesced register, or NULL if DstReg
    /// is a physreg. This register class may be a super-register of both
    /// SrcReg and DstReg.
    const TargetRegisterClass *NewRC;

  public:
    CoalescerPair(const TargetRegisterInfo &tri)
      : TRI(tri), DstReg(0), SrcReg(0), DstIdx(0), SrcIdx(0),
        Partial(false), CrossClass(false), Flipped(false), NewRC(0) {}

    /// Create a CoalescerPair representing a virtreg-to-physreg copy.
    /// No need to call setRegisters().
    CoalescerPair(unsigned VirtReg, unsigned PhysReg,
                  const TargetRegisterInfo &tri)
      : TRI(tri), DstReg(PhysReg), SrcReg(VirtReg), DstIdx(0), SrcIdx(0),
        Partial(false), CrossClass(false), Flipped(false), NewRC(0) {}

    /// setRegisters - set registers to match the copy instruction MI. Return
    /// false if MI is not a coalescable copy instruction.
    bool setRegisters(const MachineInstr*);

    /// flip - Swap SrcReg and DstReg. Return false if swapping is impossible
    /// because DstReg is a physical register, or SubIdx is set.
    bool flip();

    /// isCoalescable - Return true if MI is a copy instruction that will become
    /// an identity copy after coalescing.
    bool isCoalescable(const MachineInstr*) const;

    /// isPhys - Return true if DstReg is a physical register.
    bool isPhys() const { return !NewRC; }

    /// isPartial - Return true if the original copy instruction did not copy
    /// the full register, but was a subreg operation.
    bool isPartial() const { return Partial; }

    /// isCrossClass - Return true if DstReg is virtual and NewRC is a smaller
    /// register class than DstReg's.
    bool isCrossClass() const { return CrossClass; }

    /// isFlipped - Return true when getSrcReg is the register being defined by
    /// the original copy instruction.
    bool isFlipped() const { return Flipped; }

    /// getDstReg - Return the register (virtual or physical) that will remain
    /// after coalescing.
    unsigned getDstReg() const { return DstReg; }

    /// getSrcReg - Return the virtual register that will be coalesced away.
    unsigned getSrcReg() const { return SrcReg; }

    /// getDstIdx - Return the subregister index that DstReg will be coalesced
    /// into, or 0.
    unsigned getDstIdx() const { return DstIdx; }

    /// getSrcIdx - Return the subregister index that SrcReg will be coalesced
    /// into, or 0.
    unsigned getSrcIdx() const { return SrcIdx; }

    /// getNewRC - Return the register class of the coalesced register.
    const TargetRegisterClass *getNewRC() const { return NewRC; }
  };
} // End llvm namespace

#endif
