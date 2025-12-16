//===- CodeGen/MachineInstrBuilder.h - Simplify creation of MIs --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes a function named BuildMI, which is useful for dramatically
// simplifying how MachineInstr's are created.  It allows use of code like this:
//
//   MIMetadata MIMD(MI);  // Propagates DebugLoc and other metadata
//   M = BuildMI(MBB, MI, MIMD, TII.get(X86::ADD8rr), Dst)
//           .addReg(argVal1)
//           .addReg(argVal2);
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEINSTRBUILDER_H
#define LLVM_CODEGEN_MACHINEINSTRBUILDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstdint>

namespace llvm {

class MCInstrDesc;
class MDNode;

/// Flags to represent properties of register accesses.
///
/// These values never use the `0x1` bit so we can tell if someone is passing
/// boolean `true` instead of one of these values.
///
/// Keep this in sync with the table in MIRLangReg.rst
enum class RegState : unsigned {
  /// No Flags, used in initializers
  NoFlags = 0x0,

  /// Register definition.
  Define = 0x2,
  /// Not emitted register (e.g. carry, or temporary result).
  Implicit = 0x4,
  /// The last use of a register.
  Kill = 0x8,
  /// Unused definition.
  Dead = 0x10,
  /// Value of the register doesn't matter.
  Undef = 0x20,
  /// Register definition happens before uses.
  EarlyClobber = 0x40,
  /// Register 'use' is for debugging purpose.
  Debug = 0x80,
  /// Register reads a value that is defined inside the same instruction or
  /// bundle.
  InternalRead = 0x100,
  /// Register that may be renamed.
  Renamable = 0x200,

  LLVM_MARK_AS_BITMASK_ENUM(Renamable),

  DefineNoRead = Define | Undef,
  ImplicitDefine = Implicit | Define,
  ImplicitKill = Implicit | Kill
};

/// Set of metadata that should be preserved when using BuildMI(). This provides
/// a more convenient way of preserving certain data from the original
/// instruction.
class MIMetadata {
public:
  MIMetadata() = default;
  MIMetadata(DebugLoc DL, MDNode *PCSections = nullptr, MDNode *MMRA = nullptr,
             Value *DeactivationSymbol = nullptr)
      : DL(std::move(DL)), PCSections(PCSections), MMRA(MMRA),
        DeactivationSymbol(DeactivationSymbol) {}
  MIMetadata(const DILocation *DI, MDNode *PCSections = nullptr,
             MDNode *MMRA = nullptr)
      : DL(DI), PCSections(PCSections), MMRA(MMRA) {}
  explicit MIMetadata(const Instruction &From)
      : DL(From.getDebugLoc()),
        PCSections(From.getMetadata(LLVMContext::MD_pcsections)),
        DeactivationSymbol(getDeactivationSymbol(&From)) {}
  explicit MIMetadata(const MachineInstr &From)
      : DL(From.getDebugLoc()), PCSections(From.getPCSections()),
        DeactivationSymbol(From.getDeactivationSymbol()) {}

  const DebugLoc &getDL() const { return DL; }
  MDNode *getPCSections() const { return PCSections; }
  MDNode *getMMRAMetadata() const { return MMRA; }
  Value *getDeactivationSymbol() const { return DeactivationSymbol; }

private:
  DebugLoc DL;
  MDNode *PCSections = nullptr;
  MDNode *MMRA = nullptr;
  Value *DeactivationSymbol = nullptr;

  static inline Value *getDeactivationSymbol(const Instruction *I) {
    if (auto *CB = dyn_cast<CallBase>(I))
      if (auto Bundle =
              CB->getOperandBundle(llvm::LLVMContext::OB_deactivation_symbol))
        return Bundle->Inputs[0].get();
    return nullptr;
  }
};

class MachineInstrBuilder {
  MachineFunction *MF = nullptr;
  MachineInstr *MI = nullptr;

public:
  MachineInstrBuilder() = default;

  /// Create a MachineInstrBuilder for manipulating an existing instruction.
  /// F must be the machine function that was used to allocate I.
  MachineInstrBuilder(MachineFunction &F, MachineInstr *I) : MF(&F), MI(I) {}
  MachineInstrBuilder(MachineFunction &F, MachineBasicBlock::iterator I)
      : MF(&F), MI(&*I) {}

  /// Allow automatic conversion to the machine instruction we are working on.
  operator MachineInstr*() const { return MI; }
  MachineInstr *operator->() const { return MI; }
  operator MachineBasicBlock::iterator() const { return MI; }

  /// If conversion operators fail, use this method to get the MachineInstr
  /// explicitly.
  MachineInstr *getInstr() const { return MI; }

  /// Get the register for the operand index.
  /// The operand at the index should be a register (asserted by
  /// MachineOperand).
  Register getReg(unsigned Idx) const { return MI->getOperand(Idx).getReg(); }

  /// Add a new virtual register operand.
  const MachineInstrBuilder &addReg(Register RegNo, RegState Flags = RegState::NoFlags,
                                    unsigned SubReg = 0) const {
    assert((Flags & static_cast<RegState>(0x1)) == RegState::NoFlags &&
           "Passing in 'true' to addReg is forbidden! Use enums instead.");

    auto HasFlag = [Flags](RegState Flag) -> bool {
      return (Flags & Flag) == Flag;
    };

    MI->addOperand(*MF, MachineOperand::CreateReg(RegNo,
                                                  HasFlag(RegState::Define),
                                                  HasFlag(RegState::Implicit),
                                                  HasFlag(RegState::Kill),
                                                  HasFlag(RegState::Dead),
                                                  HasFlag(RegState::Undef),
                                                  HasFlag(RegState::EarlyClobber),
                                                  SubReg,
                                                  HasFlag(RegState::Debug),
                                                  HasFlag(RegState::InternalRead),
                                                  HasFlag(RegState::Renamable)));
    return *this;
  }

  /// Add a new virtual register operand with explicit subregister
  ///
  /// This works like `addReg`, but the subregister index comes before the flags so
  /// the flags can be omitted.
  const MachineInstrBuilder &addSubReg(Register RegNo, unsigned SubReg, RegState Flags = RegState::NoFlags) const {
    return addReg(RegNo, Flags, SubReg);
  }

  /// Add a virtual register definition operand.
  const MachineInstrBuilder &addDef(Register RegNo, RegState Flags = RegState::NoFlags,
                                    unsigned SubReg = 0) const {
    return addReg(RegNo, Flags | RegState::Define, SubReg);
  }

  /// Add a virtual register definition with explicit subregister

  ///
  /// This is like `addDef`, but the subregister index comes before the flags so the
  /// flags can be omitted.
  const MachineInstrBuilder &addSubDef(Register RegNo, unsigned SubReg, RegState Flags = RegState::NoFlags) const {
    return addReg(RegNo, Flags | RegState::Define, SubReg);
  }

  /// Add a virtual register use operand. It is an error for Flags to contain
  /// `RegState::Define` when calling this function.
  const MachineInstrBuilder &addUse(Register RegNo, RegState Flags = RegState::NoFlags,
                                    unsigned SubReg = 0) const {
    assert((Flags & RegState::Define) != RegState::Define &&
           "Misleading addUse defines register, use addReg instead.");
    return addReg(RegNo, Flags, SubReg);
  }

  /// Add a virtual register use operand with explicit subregister
  ///
  /// It is an error for Flags to contain `RegState::Define` when calling this function.
  ///
  /// This is like `addUse`, but the subregister index comes before the flags so the flags can be omitted.
  const MachineInstrBuilder &addSubUse(Register RegNo, unsigned SubReg, RegState Flags = RegState::NoFlags) const {
    assert((Flags & RegState::Define) == RegState::NoFlags &&
           "Misleading addSubUse defines register, use addSubReg instead.");
    return addReg(RegNo, Flags, SubReg);
  }

  /// Add a new immediate operand.
  const MachineInstrBuilder &addImm(int64_t Val) const {
    MI->addOperand(*MF, MachineOperand::CreateImm(Val));
    return *this;
  }

  const MachineInstrBuilder &addCImm(const ConstantInt *Val) const {
    MI->addOperand(*MF, MachineOperand::CreateCImm(Val));
    return *this;
  }

  const MachineInstrBuilder &addFPImm(const ConstantFP *Val) const {
    MI->addOperand(*MF, MachineOperand::CreateFPImm(Val));
    return *this;
  }

  const MachineInstrBuilder &addMBB(MachineBasicBlock *MBB,
                                    unsigned TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateMBB(MBB, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addFrameIndex(int Idx) const {
    MI->addOperand(*MF, MachineOperand::CreateFI(Idx));
    return *this;
  }

  const MachineInstrBuilder &
  addConstantPoolIndex(unsigned Idx, int Offset = 0,
                       unsigned TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateCPI(Idx, Offset, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addTargetIndex(unsigned Idx, int64_t Offset = 0,
                                          unsigned TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateTargetIndex(Idx, Offset,
                                                          TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addJumpTableIndex(unsigned Idx,
                                               unsigned TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateJTI(Idx, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addGlobalAddress(const GlobalValue *GV,
                                              int64_t Offset = 0,
                                              unsigned TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateGA(GV, Offset, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addExternalSymbol(const char *FnName,
                                               unsigned TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateES(FnName, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addBlockAddress(const BlockAddress *BA,
                                             int64_t Offset = 0,
                                             unsigned TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateBA(BA, Offset, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addRegMask(const uint32_t *Mask) const {
    MI->addOperand(*MF, MachineOperand::CreateRegMask(Mask));
    return *this;
  }

  const MachineInstrBuilder &addMemOperand(MachineMemOperand *MMO) const {
    MI->addMemOperand(*MF, MMO);
    return *this;
  }

  const MachineInstrBuilder &
  setMemRefs(ArrayRef<MachineMemOperand *> MMOs) const {
    MI->setMemRefs(*MF, MMOs);
    return *this;
  }

  const MachineInstrBuilder &cloneMemRefs(const MachineInstr &OtherMI) const {
    MI->cloneMemRefs(*MF, OtherMI);
    return *this;
  }

  const MachineInstrBuilder &
  cloneMergedMemRefs(ArrayRef<const MachineInstr *> OtherMIs) const {
    MI->cloneMergedMemRefs(*MF, OtherMIs);
    return *this;
  }

  const MachineInstrBuilder &add(const MachineOperand &MO) const {
    MI->addOperand(*MF, MO);
    return *this;
  }

  const MachineInstrBuilder &add(ArrayRef<MachineOperand> MOs) const {
    for (const MachineOperand &MO : MOs)
      MI->addOperand(*MF, MO);
    return *this;
  }

  const MachineInstrBuilder &addMetadata(const MDNode *MD) const {
    MI->addOperand(*MF, MachineOperand::CreateMetadata(MD));
    assert((MI->isDebugValueLike() ? static_cast<bool>(MI->getDebugVariable())
                                   : true) &&
           "first MDNode argument of a DBG_VALUE not a variable");
    assert((MI->isDebugLabel() ? static_cast<bool>(MI->getDebugLabel())
                               : true) &&
           "first MDNode argument of a DBG_LABEL not a label");
    return *this;
  }

  const MachineInstrBuilder &addCFIIndex(unsigned CFIIndex) const {
    MI->addOperand(*MF, MachineOperand::CreateCFIIndex(CFIIndex));
    return *this;
  }

  const MachineInstrBuilder &addIntrinsicID(Intrinsic::ID ID) const {
    MI->addOperand(*MF, MachineOperand::CreateIntrinsicID(ID));
    return *this;
  }

  const MachineInstrBuilder &addPredicate(CmpInst::Predicate Pred) const {
    MI->addOperand(*MF, MachineOperand::CreatePredicate(Pred));
    return *this;
  }

  const MachineInstrBuilder &addShuffleMask(ArrayRef<int> Val) const {
    MI->addOperand(*MF, MachineOperand::CreateShuffleMask(Val));
    return *this;
  }

  const MachineInstrBuilder &addLaneMask(LaneBitmask LaneMask) const {
    MI->addOperand(*MF, MachineOperand::CreateLaneMask(LaneMask));
    return *this;
  }

  const MachineInstrBuilder &addSym(MCSymbol *Sym,
                                    unsigned char TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateMCSymbol(Sym, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &setMIFlags(unsigned Flags) const {
    MI->setFlags(Flags);
    return *this;
  }

  const MachineInstrBuilder &setMIFlag(MachineInstr::MIFlag Flag) const {
    MI->setFlag(Flag);
    return *this;
  }

  const MachineInstrBuilder &setOperandDead(unsigned OpIdx) const {
    MI->getOperand(OpIdx).setIsDead();
    return *this;
  }

  // Add a displacement from an existing MachineOperand with an added offset.
  const MachineInstrBuilder &addDisp(const MachineOperand &Disp, int64_t off,
                                     unsigned char TargetFlags = 0) const {
    // If caller specifies new TargetFlags then use it, otherwise the
    // default behavior is to copy the target flags from the existing
    // MachineOperand. This means if the caller wants to clear the
    // target flags it needs to do so explicitly.
    if (0 == TargetFlags)
      TargetFlags = Disp.getTargetFlags();

    switch (Disp.getType()) {
      default:
        llvm_unreachable("Unhandled operand type in addDisp()");
      case MachineOperand::MO_Immediate:
        return addImm(Disp.getImm() + off);
      case MachineOperand::MO_ConstantPoolIndex:
        return addConstantPoolIndex(Disp.getIndex(), Disp.getOffset() + off,
                                    TargetFlags);
      case MachineOperand::MO_GlobalAddress:
        return addGlobalAddress(Disp.getGlobal(), Disp.getOffset() + off,
                                TargetFlags);
      case MachineOperand::MO_BlockAddress:
        return addBlockAddress(Disp.getBlockAddress(), Disp.getOffset() + off,
                               TargetFlags);
      case MachineOperand::MO_JumpTableIndex:
        assert(off == 0 && "cannot create offset into jump tables");
        return addJumpTableIndex(Disp.getIndex(), TargetFlags);
    }
  }

  const MachineInstrBuilder &copyMIMetadata(const MIMetadata &MIMD) const {
    if (MIMD.getPCSections())
      MI->setPCSections(*MF, MIMD.getPCSections());
    if (MIMD.getMMRAMetadata())
      MI->setMMRAMetadata(*MF, MIMD.getMMRAMetadata());
    if (MIMD.getDeactivationSymbol())
      MI->setDeactivationSymbol(*MF, MIMD.getDeactivationSymbol());
    return *this;
  }

  /// Copy all the implicit operands from OtherMI onto this one.
  const MachineInstrBuilder &
  copyImplicitOps(const MachineInstr &OtherMI) const {
    MI->copyImplicitOps(*MF, OtherMI);
    return *this;
  }

  bool constrainAllUses(const TargetInstrInfo &TII,
                        const TargetRegisterInfo &TRI,
                        const RegisterBankInfo &RBI) const {
    return constrainSelectedInstRegOperands(*MI, TII, TRI, RBI);
  }
};

/// Builder interface. Specify how to create the initial instruction itself.
inline MachineInstrBuilder BuildMI(MachineFunction &MF, const MIMetadata &MIMD,
                                   const MCInstrDesc &MCID) {
  return MachineInstrBuilder(MF, MF.CreateMachineInstr(MCID, MIMD.getDL()))
      .copyMIMetadata(MIMD);
}

/// This version of the builder sets up the first operand as a
/// destination virtual register.
inline MachineInstrBuilder BuildMI(MachineFunction &MF, const MIMetadata &MIMD,
                                   const MCInstrDesc &MCID, Register DestReg) {
  return MachineInstrBuilder(MF, MF.CreateMachineInstr(MCID, MIMD.getDL()))
      .copyMIMetadata(MIMD)
      .addReg(DestReg, RegState::Define);
}

/// This version of the builder inserts the newly-built instruction before
/// the given position in the given MachineBasicBlock, and sets up the first
/// operand as a destination virtual register.
inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineBasicBlock::iterator I,
                                   const MIMetadata &MIMD,
                                   const MCInstrDesc &MCID, Register DestReg) {
  MachineFunction &MF = *BB.getParent();
  MachineInstr *MI = MF.CreateMachineInstr(MCID, MIMD.getDL());
  BB.insert(I, MI);
  return MachineInstrBuilder(MF, MI).copyMIMetadata(MIMD).addReg(
      DestReg, RegState::Define);
}

/// This version of the builder inserts the newly-built instruction before
/// the given position in the given MachineBasicBlock, and sets up the first
/// operand as a destination virtual register.
///
/// If \c I is inside a bundle, then the newly inserted \a MachineInstr is
/// added to the same bundle.
inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineBasicBlock::instr_iterator I,
                                   const MIMetadata &MIMD,
                                   const MCInstrDesc &MCID, Register DestReg) {
  MachineFunction &MF = *BB.getParent();
  MachineInstr *MI = MF.CreateMachineInstr(MCID, MIMD.getDL());
  BB.insert(I, MI);
  return MachineInstrBuilder(MF, MI).copyMIMetadata(MIMD).addReg(
      DestReg, RegState::Define);
}

inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB, MachineInstr &I,
                                   const MIMetadata &MIMD,
                                   const MCInstrDesc &MCID, Register DestReg) {
  // Calling the overload for instr_iterator is always correct.  However, the
  // definition is not available in headers, so inline the check.
  if (I.isInsideBundle())
    return BuildMI(BB, MachineBasicBlock::instr_iterator(I), MIMD, MCID,
                   DestReg);
  return BuildMI(BB, MachineBasicBlock::iterator(I), MIMD, MCID, DestReg);
}

inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB, MachineInstr *I,
                                   const MIMetadata &MIMD,
                                   const MCInstrDesc &MCID, Register DestReg) {
  return BuildMI(BB, *I, MIMD, MCID, DestReg);
}

/// This version of the builder inserts the newly-built instruction before the
/// given position in the given MachineBasicBlock, and does NOT take a
/// destination register.
inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineBasicBlock::iterator I,
                                   const MIMetadata &MIMD,
                                   const MCInstrDesc &MCID) {
  MachineFunction &MF = *BB.getParent();
  MachineInstr *MI = MF.CreateMachineInstr(MCID, MIMD.getDL());
  BB.insert(I, MI);
  return MachineInstrBuilder(MF, MI).copyMIMetadata(MIMD);
}

inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineBasicBlock::instr_iterator I,
                                   const MIMetadata &MIMD,
                                   const MCInstrDesc &MCID) {
  MachineFunction &MF = *BB.getParent();
  MachineInstr *MI = MF.CreateMachineInstr(MCID, MIMD.getDL());
  BB.insert(I, MI);
  return MachineInstrBuilder(MF, MI).copyMIMetadata(MIMD);
}

inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB, MachineInstr &I,
                                   const MIMetadata &MIMD,
                                   const MCInstrDesc &MCID) {
  // Calling the overload for instr_iterator is always correct.  However, the
  // definition is not available in headers, so inline the check.
  if (I.isInsideBundle())
    return BuildMI(BB, MachineBasicBlock::instr_iterator(I), MIMD, MCID);
  return BuildMI(BB, MachineBasicBlock::iterator(I), MIMD, MCID);
}

inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB, MachineInstr *I,
                                   const MIMetadata &MIMD,
                                   const MCInstrDesc &MCID) {
  return BuildMI(BB, *I, MIMD, MCID);
}

/// This version of the builder inserts the newly-built instruction at the end
/// of the given MachineBasicBlock, and does NOT take a destination register.
inline MachineInstrBuilder BuildMI(MachineBasicBlock *BB,
                                   const MIMetadata &MIMD,
                                   const MCInstrDesc &MCID) {
  return BuildMI(*BB, BB->end(), MIMD, MCID);
}

/// This version of the builder inserts the newly-built instruction at the
/// end of the given MachineBasicBlock, and sets up the first operand as a
/// destination virtual register.
inline MachineInstrBuilder BuildMI(MachineBasicBlock *BB,
                                   const MIMetadata &MIMD,
                                   const MCInstrDesc &MCID, Register DestReg) {
  return BuildMI(*BB, BB->end(), MIMD, MCID, DestReg);
}

/// This version of the builder builds a DBG_VALUE intrinsic
/// for either a value in a register or a register-indirect
/// address.  The convention is that a DBG_VALUE is indirect iff the
/// second operand is an immediate.
LLVM_ABI MachineInstrBuilder BuildMI(MachineFunction &MF, const DebugLoc &DL,
                                     const MCInstrDesc &MCID, bool IsIndirect,
                                     Register Reg, const MDNode *Variable,
                                     const MDNode *Expr);

/// This version of the builder builds a DBG_VALUE or DBG_VALUE_LIST intrinsic
/// for a MachineOperand.
LLVM_ABI MachineInstrBuilder BuildMI(MachineFunction &MF, const DebugLoc &DL,
                                     const MCInstrDesc &MCID, bool IsIndirect,
                                     ArrayRef<MachineOperand> MOs,
                                     const MDNode *Variable,
                                     const MDNode *Expr);

/// This version of the builder builds a DBG_VALUE intrinsic
/// for either a value in a register or a register-indirect
/// address and inserts it at position I.
LLVM_ABI MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                     MachineBasicBlock::iterator I,
                                     const DebugLoc &DL,
                                     const MCInstrDesc &MCID, bool IsIndirect,
                                     Register Reg, const MDNode *Variable,
                                     const MDNode *Expr);

/// This version of the builder builds a DBG_VALUE, DBG_INSTR_REF, or
/// DBG_VALUE_LIST intrinsic for a machine operand and inserts it at position I.
LLVM_ABI MachineInstrBuilder BuildMI(
    MachineBasicBlock &BB, MachineBasicBlock::iterator I, const DebugLoc &DL,
    const MCInstrDesc &MCID, bool IsIndirect, ArrayRef<MachineOperand> MOs,
    const MDNode *Variable, const MDNode *Expr);

/// Clone a DBG_VALUE whose value has been spilled to FrameIndex.
LLVM_ABI MachineInstr *buildDbgValueForSpill(MachineBasicBlock &BB,
                                             MachineBasicBlock::iterator I,
                                             const MachineInstr &Orig,
                                             int FrameIndex, Register SpillReg);
LLVM_ABI MachineInstr *buildDbgValueForSpill(
    MachineBasicBlock &BB, MachineBasicBlock::iterator I,
    const MachineInstr &Orig, int FrameIndex,
    const SmallVectorImpl<const MachineOperand *> &SpilledOperands);

/// Update a DBG_VALUE whose value has been spilled to FrameIndex. Useful when
/// modifying an instruction in place while iterating over a basic block.
LLVM_ABI void updateDbgValueForSpill(MachineInstr &Orig, int FrameIndex,
                                     Register Reg);

inline RegState getDefRegState(bool B) {
  return B ? RegState::Define : RegState::NoFlags;
}
inline RegState getImplRegState(bool B) {
  return B ? RegState::Implicit : RegState::NoFlags;
}
inline RegState getKillRegState(bool B) {
  return B ? RegState::Kill : RegState::NoFlags;
}
inline RegState getDeadRegState(bool B) {
  return B ? RegState::Dead : RegState::NoFlags;
}
inline RegState getUndefRegState(bool B) {
  return B ? RegState::Undef : RegState::NoFlags;
}
inline RegState getEarlyClobberRegState(bool B) {
  return B ? RegState::EarlyClobber : RegState::NoFlags;
}
inline RegState getDebugRegState(bool B) {
  return B ? RegState::Debug : RegState::NoFlags;
}
inline RegState getInternalReadRegState(bool B) {
  return B ? RegState::InternalRead : RegState::NoFlags;
}
inline RegState getRenamableRegState(bool B) {
  return B ? RegState::Renamable : RegState::NoFlags;
}

/// Get all register state flags from machine operand \p RegOp.
inline RegState getRegState(const MachineOperand &RegOp) {
  assert(RegOp.isReg() && "Not a register operand");
  return getDefRegState(RegOp.isDef()) | getImplRegState(RegOp.isImplicit()) |
         getKillRegState(RegOp.isKill()) | getDeadRegState(RegOp.isDead()) |
         getUndefRegState(RegOp.isUndef()) |
         // TODO: this is missing, I'm not sure why:
         // getEarlyClobberRegState(RegOp.isEarlyClobber()) |
         getInternalReadRegState(RegOp.isInternalRead()) |
         getDebugRegState(RegOp.isDebug()) |
         getRenamableRegState(RegOp.getReg().isPhysical() &&
                              RegOp.isRenamable());
}

/// Helper class for constructing bundles of MachineInstrs.
///
/// MIBundleBuilder can create a bundle from scratch by inserting new
/// MachineInstrs one at a time, or it can create a bundle from a sequence of
/// existing MachineInstrs in a basic block.
class MIBundleBuilder {
  MachineBasicBlock &MBB;
  MachineBasicBlock::instr_iterator Begin;
  MachineBasicBlock::instr_iterator End;

public:
  /// Create an MIBundleBuilder that inserts instructions into a new bundle in
  /// BB above the bundle or instruction at Pos.
  MIBundleBuilder(MachineBasicBlock &BB, MachineBasicBlock::iterator Pos)
      : MBB(BB), Begin(Pos.getInstrIterator()), End(Begin) {}

  /// Create a bundle from the sequence of instructions between B and E.
  MIBundleBuilder(MachineBasicBlock &BB, MachineBasicBlock::iterator B,
                  MachineBasicBlock::iterator E)
      : MBB(BB), Begin(B.getInstrIterator()), End(E.getInstrIterator()) {
    assert(B != E && "No instructions to bundle");
    ++B;
    while (B != E) {
      MachineInstr &MI = *B;
      ++B;
      MI.bundleWithPred();
    }
  }

  /// Create an MIBundleBuilder representing an existing instruction or bundle
  /// that has MI as its head.
  explicit MIBundleBuilder(MachineInstr *MI)
      : MBB(*MI->getParent()), Begin(MI),
        End(getBundleEnd(MI->getIterator())) {}

  /// Return a reference to the basic block containing this bundle.
  MachineBasicBlock &getMBB() const { return MBB; }

  /// Return true if no instructions have been inserted in this bundle yet.
  /// Empty bundles aren't representable in a MachineBasicBlock.
  bool empty() const { return Begin == End; }

  /// Return an iterator to the first bundled instruction.
  MachineBasicBlock::instr_iterator begin() const { return Begin; }

  /// Return an iterator beyond the last bundled instruction.
  MachineBasicBlock::instr_iterator end() const { return End; }

  /// Insert MI into this bundle before I which must point to an instruction in
  /// the bundle, or end().
  MIBundleBuilder &insert(MachineBasicBlock::instr_iterator I,
                          MachineInstr *MI) {
    MBB.insert(I, MI);
    if (I == Begin) {
      if (!empty())
        MI->bundleWithSucc();
      Begin = MI->getIterator();
      return *this;
    }
    if (I == End) {
      MI->bundleWithPred();
      return *this;
    }
    // MI was inserted in the middle of the bundle, so its neighbors' flags are
    // already fine. Update MI's bundle flags manually.
    MI->setFlag(MachineInstr::BundledPred);
    MI->setFlag(MachineInstr::BundledSucc);
    return *this;
  }

  /// Insert MI into MBB by prepending it to the instructions in the bundle.
  /// MI will become the first instruction in the bundle.
  MIBundleBuilder &prepend(MachineInstr *MI) {
    return insert(begin(), MI);
  }

  /// Insert MI into MBB by appending it to the instructions in the bundle.
  /// MI will become the last instruction in the bundle.
  MIBundleBuilder &append(MachineInstr *MI) {
    return insert(end(), MI);
  }
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEINSTRBUILDER_H
