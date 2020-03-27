//===- AMDGPURegisterBankInfo.cpp -------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the RegisterBankInfo class for
/// AMDGPU.
///
/// \par
///
/// AMDGPU has unique register bank constraints that require special high level
/// strategies to deal with. There are two main true physical register banks
/// VGPR (vector), and SGPR (scalar). Additionally the VCC register bank is a
/// sort of pseudo-register bank needed to represent SGPRs used in a vector
/// boolean context. There is also the AGPR bank, which is a special purpose
/// physical register bank present on some subtargets.
///
/// Copying from VGPR to SGPR is generally illegal, unless the value is known to
/// be uniform. It is generally not valid to legalize operands by inserting
/// copies as on other targets. Operations which require uniform, SGPR operands
/// generally require scalarization by repeatedly executing the instruction,
/// activating each set of lanes using a unique set of input values. This is
/// referred to as a waterfall loop.
///
/// \par Booleans
///
/// Booleans (s1 values) requires special consideration. A vector compare result
/// is naturally a bitmask with one bit per lane, in a 32 or 64-bit
/// register. These are represented with the VCC bank. During selection, we need
/// to be able to unambiguously go back from a register class to a register
/// bank. To distinguish whether an SGPR should use the SGPR or VCC register
/// bank, we need to know the use context type. An SGPR s1 value always means a
/// VCC bank value, otherwise it will be the SGPR bank. A scalar compare sets
/// SCC, which is a 1-bit unaddressable register. This will need to be copied to
/// a 32-bit virtual register. Taken together, this means we need to adjust the
/// type of boolean operations to be regbank legal. All SALU booleans need to be
/// widened to 32-bits, and all VALU booleans need to be s1 values.
///
/// A noteworthy exception to the s1-means-vcc rule is for legalization artifact
/// casts. G_TRUNC s1 results, and G_SEXT/G_ZEXT/G_ANYEXT sources are never vcc
/// bank. A non-boolean source (such as a truncate from a 1-bit load from
/// memory) will require a copy to the VCC bank which will require clearing the
/// high bits and inserting a compare.
///
/// \par Constant bus restriction
///
/// VALU instructions have a limitation known as the constant bus
/// restriction. Most VALU instructions can use SGPR operands, but may read at
/// most 1 SGPR or constant literal value (this to 2 in gfx10 for most
/// instructions). This is one unique SGPR, so the same SGPR may be used for
/// multiple operands. From a register bank perspective, any combination of
/// operands should be legal as an SGPR, but this is contextually dependent on
/// the SGPR operands all being the same register. There is therefore optimal to
/// choose the SGPR with the most uses to minimize the number of copies.
///
/// We avoid trying to solve this problem in RegBankSelect. Any VALU G_*
/// operation should have its source operands all mapped to VGPRs (except for
/// VCC), inserting copies from any SGPR operands. This the most trival legal
/// mapping. Anything beyond the simplest 1:1 instruction selection would be too
/// complicated to solve here. Every optimization pattern or instruction
/// selected to multiple outputs would have to enforce this rule, and there
/// would be additional complexity in tracking this rule for every G_*
/// operation. By forcing all inputs to VGPRs, it also simplifies the task of
/// picking the optimal operand combination from a post-isel optimization pass.
///
//===----------------------------------------------------------------------===//

#include "AMDGPURegisterBankInfo.h"

#include "AMDGPUGlobalISelUtils.h"
#include "AMDGPUInstrInfo.h"
#include "AMDGPUSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/GlobalISel/LegalizationArtifactCombiner.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/RegisterBank.h"
#include "llvm/CodeGen/GlobalISel/RegisterBankInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Constants.h"

#define GET_TARGET_REGBANK_IMPL
#include "AMDGPUGenRegisterBank.inc"

// This file will be TableGen'ed at some point.
#include "AMDGPUGenRegisterBankInfo.def"

using namespace llvm;
using namespace MIPatternMatch;

namespace {

// Observer to apply a register bank to new registers created by LegalizerHelper.
class ApplyRegBankMapping final : public GISelChangeObserver {
private:
  const AMDGPURegisterBankInfo &RBI;
  MachineRegisterInfo &MRI;
  const RegisterBank *NewBank;
  SmallVector<MachineInstr *, 4> NewInsts;

public:
  ApplyRegBankMapping(const AMDGPURegisterBankInfo &RBI_,
                      MachineRegisterInfo &MRI_, const RegisterBank *RB)
    : RBI(RBI_), MRI(MRI_), NewBank(RB) {}

  ~ApplyRegBankMapping() {
    for (MachineInstr *MI : NewInsts)
      applyBank(*MI);
  }

  /// Set any registers that don't have a set register class or bank to SALU.
  void applyBank(MachineInstr &MI) {
    const unsigned Opc = MI.getOpcode();
    if (Opc == AMDGPU::G_ANYEXT || Opc == AMDGPU::G_ZEXT ||
        Opc == AMDGPU::G_SEXT) {
      // LegalizerHelper wants to use the basic legalization artifacts when
      // widening etc. We don't handle selection with vcc in artifact sources,
      // so we need to use a sslect instead to handle these properly.
      Register DstReg = MI.getOperand(0).getReg();
      Register SrcReg = MI.getOperand(1).getReg();
      const RegisterBank *SrcBank = RBI.getRegBank(SrcReg, MRI, *RBI.TRI);
      if (SrcBank == &AMDGPU::VCCRegBank) {
        const LLT S32 = LLT::scalar(32);
        assert(MRI.getType(SrcReg) == LLT::scalar(1));
        assert(MRI.getType(DstReg) == S32);
        assert(NewBank == &AMDGPU::VGPRRegBank);

        // Replace the extension with a select, which really uses the boolean
        // source.
        MachineIRBuilder B(MI);
        auto True = B.buildConstant(S32, Opc == AMDGPU::G_SEXT ? -1 : 1);
        auto False = B.buildConstant(S32, 0);
        B.buildSelect(DstReg, SrcReg, True, False);
        MRI.setRegBank(True.getReg(0), *NewBank);
        MRI.setRegBank(False.getReg(0), *NewBank);
        MI.eraseFromParent();
      }

      assert(!MRI.getRegClassOrRegBank(DstReg));
      MRI.setRegBank(DstReg, *NewBank);
      return;
    }

#ifndef NDEBUG
    if (Opc == AMDGPU::G_TRUNC) {
      Register DstReg = MI.getOperand(0).getReg();
      const RegisterBank *DstBank = RBI.getRegBank(DstReg, MRI, *RBI.TRI);
      assert(DstBank != &AMDGPU::VCCRegBank);
    }
#endif

    for (MachineOperand &Op : MI.operands()) {
      if (!Op.isReg())
        continue;

      // We may see physical registers if building a real MI
      Register Reg = Op.getReg();
      if (Reg.isPhysical() || MRI.getRegClassOrRegBank(Reg))
        continue;

      const RegisterBank *RB = NewBank;
      if (MRI.getType(Reg) == LLT::scalar(1)) {
        assert(NewBank == &AMDGPU::VGPRRegBank &&
               "s1 operands should only be used for vector bools");
        assert((MI.getOpcode() != AMDGPU::G_TRUNC &&
                MI.getOpcode() != AMDGPU::G_ANYEXT) &&
               "not expecting legalization artifacts here");
        RB = &AMDGPU::VCCRegBank;
      }

      MRI.setRegBank(Reg, *RB);
    }
  }

  void erasingInstr(MachineInstr &MI) override {}

  void createdInstr(MachineInstr &MI) override {
    // At this point, the instruction was just inserted and has no operands.
    NewInsts.push_back(&MI);
  }

  void changingInstr(MachineInstr &MI) override {}
  void changedInstr(MachineInstr &MI) override {}
};

}
AMDGPURegisterBankInfo::AMDGPURegisterBankInfo(const GCNSubtarget &ST)
    : AMDGPUGenRegisterBankInfo(),
      Subtarget(ST),
      TRI(Subtarget.getRegisterInfo()),
      TII(Subtarget.getInstrInfo()) {

  // HACK: Until this is fully tablegen'd.
  static llvm::once_flag InitializeRegisterBankFlag;

  static auto InitializeRegisterBankOnce = [this]() {
    assert(&getRegBank(AMDGPU::SGPRRegBankID) == &AMDGPU::SGPRRegBank &&
           &getRegBank(AMDGPU::VGPRRegBankID) == &AMDGPU::VGPRRegBank &&
           &getRegBank(AMDGPU::AGPRRegBankID) == &AMDGPU::AGPRRegBank);
    (void)this;
  };

  llvm::call_once(InitializeRegisterBankFlag, InitializeRegisterBankOnce);
}

static bool isVectorRegisterBank(const RegisterBank &Bank) {
  unsigned BankID = Bank.getID();
  return BankID == AMDGPU::VGPRRegBankID || BankID == AMDGPU::AGPRRegBankID;
}

unsigned AMDGPURegisterBankInfo::copyCost(const RegisterBank &Dst,
                                          const RegisterBank &Src,
                                          unsigned Size) const {
  // TODO: Should there be a UniformVGPRRegBank which can use readfirstlane?
  if (Dst.getID() == AMDGPU::SGPRRegBankID &&
      (isVectorRegisterBank(Src) || Src.getID() == AMDGPU::VCCRegBankID)) {
    return std::numeric_limits<unsigned>::max();
  }

  // Bool values are tricky, because the meaning is based on context. The SCC
  // and VCC banks are for the natural scalar and vector conditions produced by
  // a compare.
  //
  // Legalization doesn't know about the necessary context, so an s1 use may
  // have been a truncate from an arbitrary value, in which case a copy (lowered
  // as a compare with 0) needs to be inserted.
  if (Size == 1 &&
      (Dst.getID() == AMDGPU::SGPRRegBankID) &&
      (isVectorRegisterBank(Src) ||
       Src.getID() == AMDGPU::SGPRRegBankID ||
       Src.getID() == AMDGPU::VCCRegBankID))
    return std::numeric_limits<unsigned>::max();

  // There is no direct copy between AGPRs.
  if (Dst.getID() == AMDGPU::AGPRRegBankID &&
      Src.getID() == AMDGPU::AGPRRegBankID)
    return 4;

  return RegisterBankInfo::copyCost(Dst, Src, Size);
}

unsigned AMDGPURegisterBankInfo::getBreakDownCost(
  const ValueMapping &ValMapping,
  const RegisterBank *CurBank) const {
  // Check if this is a breakdown for G_LOAD to move the pointer from SGPR to
  // VGPR.
  // FIXME: Is there a better way to do this?
  if (ValMapping.NumBreakDowns >= 2 || ValMapping.BreakDown[0].Length >= 64)
    return 10; // This is expensive.

  assert(ValMapping.NumBreakDowns == 2 &&
         ValMapping.BreakDown[0].Length == 32 &&
         ValMapping.BreakDown[0].StartIdx == 0 &&
         ValMapping.BreakDown[1].Length == 32 &&
         ValMapping.BreakDown[1].StartIdx == 32 &&
         ValMapping.BreakDown[0].RegBank == ValMapping.BreakDown[1].RegBank);

  // 32-bit extract of a 64-bit value is just access of a subregister, so free.
  // TODO: Cost of 0 hits assert, though it's not clear it's what we really
  // want.

  // TODO: 32-bit insert to a 64-bit SGPR may incur a non-free copy due to SGPR
  // alignment restrictions, but this probably isn't important.
  return 1;
}

const RegisterBank &
AMDGPURegisterBankInfo::getRegBankFromRegClass(const TargetRegisterClass &RC,
                                               LLT Ty) const {
  if (&RC == &AMDGPU::SReg_1RegClass)
    return AMDGPU::VCCRegBank;

  // We promote real scalar booleans to SReg_32. Any SGPR using s1 is really a
  // VCC-like use.
  if (TRI->isSGPRClass(&RC)) {
    // FIXME: This probably came from a copy from a physical register, which
    // should be inferrrable from the copied to-type. We don't have many boolean
    // physical register constraints so just assume a normal SGPR for now.
    if (!Ty.isValid())
      return AMDGPU::SGPRRegBank;

    return Ty == LLT::scalar(1) ? AMDGPU::VCCRegBank : AMDGPU::SGPRRegBank;
  }

  return TRI->isAGPRClass(&RC) ? AMDGPU::AGPRRegBank : AMDGPU::VGPRRegBank;
}

template <unsigned NumOps>
RegisterBankInfo::InstructionMappings
AMDGPURegisterBankInfo::addMappingFromTable(
    const MachineInstr &MI, const MachineRegisterInfo &MRI,
    const std::array<unsigned, NumOps> RegSrcOpIdx,
    ArrayRef<OpRegBankEntry<NumOps>> Table) const {

  InstructionMappings AltMappings;

  SmallVector<const ValueMapping *, 10> Operands(MI.getNumOperands());

  unsigned Sizes[NumOps];
  for (unsigned I = 0; I < NumOps; ++I) {
    Register Reg = MI.getOperand(RegSrcOpIdx[I]).getReg();
    Sizes[I] = getSizeInBits(Reg, MRI, *TRI);
  }

  for (unsigned I = 0, E = MI.getNumExplicitDefs(); I != E; ++I) {
    unsigned SizeI = getSizeInBits(MI.getOperand(I).getReg(), MRI, *TRI);
    Operands[I] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, SizeI);
  }

  // getInstrMapping's default mapping uses ID 1, so start at 2.
  unsigned MappingID = 2;
  for (const auto &Entry : Table) {
    for (unsigned I = 0; I < NumOps; ++I) {
      int OpIdx = RegSrcOpIdx[I];
      Operands[OpIdx] = AMDGPU::getValueMapping(Entry.RegBanks[I], Sizes[I]);
    }

    AltMappings.push_back(&getInstructionMapping(MappingID++, Entry.Cost,
                                                 getOperandsMapping(Operands),
                                                 Operands.size()));
  }

  return AltMappings;
}

RegisterBankInfo::InstructionMappings
AMDGPURegisterBankInfo::getInstrAlternativeMappingsIntrinsic(
    const MachineInstr &MI, const MachineRegisterInfo &MRI) const {
  switch (MI.getIntrinsicID()) {
  case Intrinsic::amdgcn_readlane: {
    static const OpRegBankEntry<3> Table[2] = {
      // Perfectly legal.
      { { AMDGPU::SGPRRegBankID, AMDGPU::VGPRRegBankID, AMDGPU::SGPRRegBankID }, 1 },

      // Need a readfirstlane for the index.
      { { AMDGPU::SGPRRegBankID, AMDGPU::VGPRRegBankID, AMDGPU::VGPRRegBankID }, 2 }
    };

    const std::array<unsigned, 3> RegSrcOpIdx = { { 0, 2, 3 } };
    return addMappingFromTable<3>(MI, MRI, RegSrcOpIdx, makeArrayRef(Table));
  }
  case Intrinsic::amdgcn_writelane: {
    static const OpRegBankEntry<4> Table[4] = {
      // Perfectly legal.
      { { AMDGPU::VGPRRegBankID, AMDGPU::SGPRRegBankID, AMDGPU::SGPRRegBankID, AMDGPU::VGPRRegBankID }, 1 },

      // Need readfirstlane of first op
      { { AMDGPU::VGPRRegBankID, AMDGPU::VGPRRegBankID, AMDGPU::SGPRRegBankID, AMDGPU::VGPRRegBankID }, 2 },

      // Need readfirstlane of second op
      { { AMDGPU::VGPRRegBankID, AMDGPU::SGPRRegBankID, AMDGPU::VGPRRegBankID, AMDGPU::VGPRRegBankID }, 2 },

      // Need readfirstlane of both ops
      { { AMDGPU::VGPRRegBankID, AMDGPU::VGPRRegBankID, AMDGPU::VGPRRegBankID, AMDGPU::VGPRRegBankID }, 3 }
    };

    // rsrc, voffset, offset
    const std::array<unsigned, 4> RegSrcOpIdx = { { 0, 2, 3, 4 } };
    return addMappingFromTable<4>(MI, MRI, RegSrcOpIdx, makeArrayRef(Table));
  }
  default:
    return RegisterBankInfo::getInstrAlternativeMappings(MI);
  }
}

RegisterBankInfo::InstructionMappings
AMDGPURegisterBankInfo::getInstrAlternativeMappingsIntrinsicWSideEffects(
    const MachineInstr &MI, const MachineRegisterInfo &MRI) const {

  switch (MI.getIntrinsicID()) {
  case Intrinsic::amdgcn_s_buffer_load: {
    static const OpRegBankEntry<2> Table[4] = {
      // Perfectly legal.
      { { AMDGPU::SGPRRegBankID, AMDGPU::SGPRRegBankID }, 1 },

      // Only need 1 register in loop
      { { AMDGPU::SGPRRegBankID, AMDGPU::VGPRRegBankID }, 300 },

      // Have to waterfall the resource.
      { { AMDGPU::VGPRRegBankID, AMDGPU::SGPRRegBankID }, 1000 },

      // Have to waterfall the resource, and the offset.
      { { AMDGPU::VGPRRegBankID, AMDGPU::VGPRRegBankID }, 1500 }
    };

    // rsrc, offset
    const std::array<unsigned, 2> RegSrcOpIdx = { { 2, 3 } };
    return addMappingFromTable<2>(MI, MRI, RegSrcOpIdx, makeArrayRef(Table));
  }
  case Intrinsic::amdgcn_ds_ordered_add:
  case Intrinsic::amdgcn_ds_ordered_swap: {
    // VGPR = M0, VGPR
    static const OpRegBankEntry<3> Table[2] = {
      // Perfectly legal.
      { { AMDGPU::VGPRRegBankID, AMDGPU::SGPRRegBankID, AMDGPU::VGPRRegBankID  }, 1 },

      // Need a readfirstlane for m0
      { { AMDGPU::VGPRRegBankID, AMDGPU::VGPRRegBankID, AMDGPU::VGPRRegBankID }, 2 }
    };

    const std::array<unsigned, 3> RegSrcOpIdx = { { 0, 2, 3 } };
    return addMappingFromTable<3>(MI, MRI, RegSrcOpIdx, makeArrayRef(Table));
  }
  case Intrinsic::amdgcn_s_sendmsg:
  case Intrinsic::amdgcn_s_sendmsghalt: {
    // FIXME: Should have no register for immediate
    static const OpRegBankEntry<1> Table[2] = {
      // Perfectly legal.
      { { AMDGPU::SGPRRegBankID }, 1 },

      // Need readlane
      { { AMDGPU::VGPRRegBankID }, 3 }
    };

    const std::array<unsigned, 1> RegSrcOpIdx = { { 2 } };
    return addMappingFromTable<1>(MI, MRI, RegSrcOpIdx, makeArrayRef(Table));
  }
  default:
    return RegisterBankInfo::getInstrAlternativeMappings(MI);
  }
}

static bool memOpHasNoClobbered(const MachineMemOperand *MMO) {
  const Instruction *I = dyn_cast_or_null<Instruction>(MMO->getValue());
  return I && I->getMetadata("amdgpu.noclobber");
}

// FIXME: Returns uniform if there's no source value information. This is
// probably wrong.
static bool isScalarLoadLegal(const MachineInstr &MI) {
  if (!MI.hasOneMemOperand())
    return false;

  const MachineMemOperand *MMO = *MI.memoperands_begin();
  const unsigned AS = MMO->getAddrSpace();
  const bool IsConst = AS == AMDGPUAS::CONSTANT_ADDRESS ||
                       AS == AMDGPUAS::CONSTANT_ADDRESS_32BIT;

  // There are no extending SMRD/SMEM loads, and they require 4-byte alignment.
  return MMO->getSize() >= 4 && MMO->getAlign() >= Align(4) &&
         // Can't do a scalar atomic load.
         !MMO->isAtomic() &&
         // Don't use scalar loads for volatile accesses to non-constant address
         // spaces.
         (IsConst || !MMO->isVolatile()) &&
         // Memory must be known constant, or not written before this load.
         (IsConst || MMO->isInvariant() || memOpHasNoClobbered(MMO)) &&
         AMDGPUInstrInfo::isUniformMMO(MMO);
}

RegisterBankInfo::InstructionMappings
AMDGPURegisterBankInfo::getInstrAlternativeMappings(
    const MachineInstr &MI) const {

  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();


  InstructionMappings AltMappings;
  switch (MI.getOpcode()) {
  case TargetOpcode::G_CONSTANT: {
    unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
    if (Size == 1) {
      static const OpRegBankEntry<1> Table[3] = {
        { { AMDGPU::VGPRRegBankID }, 1 },
        { { AMDGPU::SGPRRegBankID }, 1 },
        { { AMDGPU::VCCRegBankID }, 1 }
      };

      return addMappingFromTable<1>(MI, MRI, {{ 0 }}, Table);
    }

    LLVM_FALLTHROUGH;
  }
  case TargetOpcode::G_FCONSTANT:
  case TargetOpcode::G_FRAME_INDEX:
  case TargetOpcode::G_GLOBAL_VALUE: {
    static const OpRegBankEntry<1> Table[2] = {
      { { AMDGPU::VGPRRegBankID }, 1 },
      { { AMDGPU::SGPRRegBankID }, 1 }
    };

    return addMappingFromTable<1>(MI, MRI, {{ 0 }}, Table);
  }
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR: {
    unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);

    if (Size == 1) {
      // s_{and|or|xor}_b32 set scc when the result of the 32-bit op is not 0.
      const InstructionMapping &SCCMapping = getInstructionMapping(
        1, 1, getOperandsMapping(
          {AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, 32),
           AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, 32),
           AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, 32)}),
        3); // Num Operands
      AltMappings.push_back(&SCCMapping);

      const InstructionMapping &VCCMapping0 = getInstructionMapping(
        2, 1, getOperandsMapping(
          {AMDGPU::getValueMapping(AMDGPU::VCCRegBankID, Size),
           AMDGPU::getValueMapping(AMDGPU::VCCRegBankID, Size),
           AMDGPU::getValueMapping(AMDGPU::VCCRegBankID, Size)}),
        3); // Num Operands
      AltMappings.push_back(&VCCMapping0);
      return AltMappings;
    }

    if (Size != 64)
      break;

    const InstructionMapping &SSMapping = getInstructionMapping(
      1, 1, getOperandsMapping(
        {AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size),
         AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size),
         AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size)}),
      3); // Num Operands
    AltMappings.push_back(&SSMapping);

    const InstructionMapping &VVMapping = getInstructionMapping(
      2, 2, getOperandsMapping(
        {AMDGPU::getValueMappingSGPR64Only(AMDGPU::VGPRRegBankID, Size),
         AMDGPU::getValueMappingSGPR64Only(AMDGPU::VGPRRegBankID, Size),
         AMDGPU::getValueMappingSGPR64Only(AMDGPU::VGPRRegBankID, Size)}),
      3); // Num Operands
    AltMappings.push_back(&VVMapping);
    break;
  }
  case TargetOpcode::G_LOAD:
  case TargetOpcode::G_ZEXTLOAD:
  case TargetOpcode::G_SEXTLOAD: {
    unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
    LLT PtrTy = MRI.getType(MI.getOperand(1).getReg());
    unsigned PtrSize = PtrTy.getSizeInBits();
    unsigned AS = PtrTy.getAddressSpace();
    LLT LoadTy = MRI.getType(MI.getOperand(0).getReg());

    if ((AS != AMDGPUAS::LOCAL_ADDRESS && AS != AMDGPUAS::REGION_ADDRESS &&
         AS != AMDGPUAS::PRIVATE_ADDRESS) &&
        isScalarLoadLegal(MI)) {
      const InstructionMapping &SSMapping = getInstructionMapping(
          1, 1, getOperandsMapping(
                    {AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size),
                     AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, PtrSize)}),
          2); // Num Operands
      AltMappings.push_back(&SSMapping);
    }

    const InstructionMapping &VVMapping = getInstructionMapping(
        2, 1, getOperandsMapping(
          {AMDGPU::getValueMappingLoadSGPROnly(AMDGPU::VGPRRegBankID, LoadTy),
           AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, PtrSize)}),
        2); // Num Operands
    AltMappings.push_back(&VVMapping);

    // It may be possible to have a vgpr = load sgpr mapping here, because
    // the mubuf instructions support this kind of load, but probably for only
    // gfx7 and older.  However, the addressing mode matching in the instruction
    // selector should be able to do a better job of detecting and selecting
    // these kinds of loads from the vgpr = load vgpr mapping.

    return AltMappings;

  }
  case TargetOpcode::G_SELECT: {
    unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
    const InstructionMapping &SSMapping = getInstructionMapping(1, 1,
      getOperandsMapping({AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size),
                          AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, 1),
                          AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size),
                          AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size)}),
      4); // Num Operands
    AltMappings.push_back(&SSMapping);

    const InstructionMapping &VVMapping = getInstructionMapping(2, 1,
      getOperandsMapping({AMDGPU::getValueMappingSGPR64Only(AMDGPU::VGPRRegBankID, Size),
                          AMDGPU::getValueMapping(AMDGPU::VCCRegBankID, 1),
                          AMDGPU::getValueMappingSGPR64Only(AMDGPU::VGPRRegBankID, Size),
                          AMDGPU::getValueMappingSGPR64Only(AMDGPU::VGPRRegBankID, Size)}),
      4); // Num Operands
    AltMappings.push_back(&VVMapping);

    return AltMappings;
  }
  case TargetOpcode::G_SMIN:
  case TargetOpcode::G_SMAX:
  case TargetOpcode::G_UMIN:
  case TargetOpcode::G_UMAX: {
    static const OpRegBankEntry<3> Table[2] = {
      { { AMDGPU::VGPRRegBankID, AMDGPU::VGPRRegBankID, AMDGPU::VGPRRegBankID }, 1 },

      // Scalar requires cmp+select, and extends if 16-bit.
      // FIXME: Should there be separate costs for 32 and 16-bit
      { { AMDGPU::SGPRRegBankID, AMDGPU::SGPRRegBankID, AMDGPU::SGPRRegBankID }, 3 }
    };

    const std::array<unsigned, 3> RegSrcOpIdx = { { 0, 1, 2 } };
    return addMappingFromTable<3>(MI, MRI, RegSrcOpIdx, makeArrayRef(Table));
  }
  case TargetOpcode::G_UADDE:
  case TargetOpcode::G_USUBE:
  case TargetOpcode::G_SADDE:
  case TargetOpcode::G_SSUBE: {
    unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
    const InstructionMapping &SSMapping = getInstructionMapping(1, 1,
      getOperandsMapping(
        {AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size),
         AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, 1),
         AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size),
         AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size),
         AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, 1)}),
      5); // Num Operands
    AltMappings.push_back(&SSMapping);

    const InstructionMapping &VVMapping = getInstructionMapping(2, 1,
      getOperandsMapping({AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Size),
                          AMDGPU::getValueMapping(AMDGPU::VCCRegBankID, 1),
                          AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Size),
                          AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Size),
                          AMDGPU::getValueMapping(AMDGPU::VCCRegBankID, 1)}),
      5); // Num Operands
    AltMappings.push_back(&VVMapping);
    return AltMappings;
  }
  case AMDGPU::G_BRCOND: {
    assert(MRI.getType(MI.getOperand(0).getReg()).getSizeInBits() == 1);

    // TODO: Change type to 32 for scalar
    const InstructionMapping &SMapping = getInstructionMapping(
      1, 1, getOperandsMapping(
        {AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, 1), nullptr}),
      2); // Num Operands
    AltMappings.push_back(&SMapping);

    const InstructionMapping &VMapping = getInstructionMapping(
      1, 1, getOperandsMapping(
        {AMDGPU::getValueMapping(AMDGPU::VCCRegBankID, 1), nullptr }),
      2); // Num Operands
    AltMappings.push_back(&VMapping);
    return AltMappings;
  }
  case AMDGPU::G_INTRINSIC:
    return getInstrAlternativeMappingsIntrinsic(MI, MRI);
  case AMDGPU::G_INTRINSIC_W_SIDE_EFFECTS:
    return getInstrAlternativeMappingsIntrinsicWSideEffects(MI, MRI);
  default:
    break;
  }
  return RegisterBankInfo::getInstrAlternativeMappings(MI);
}

void AMDGPURegisterBankInfo::split64BitValueForMapping(
  MachineIRBuilder &B,
  SmallVector<Register, 2> &Regs,
  LLT HalfTy,
  Register Reg) const {
  assert(HalfTy.getSizeInBits() == 32);
  MachineRegisterInfo *MRI = B.getMRI();
  Register LoLHS = MRI->createGenericVirtualRegister(HalfTy);
  Register HiLHS = MRI->createGenericVirtualRegister(HalfTy);
  const RegisterBank *Bank = getRegBank(Reg, *MRI, *TRI);
  MRI->setRegBank(LoLHS, *Bank);
  MRI->setRegBank(HiLHS, *Bank);

  Regs.push_back(LoLHS);
  Regs.push_back(HiLHS);

  B.buildInstr(AMDGPU::G_UNMERGE_VALUES)
    .addDef(LoLHS)
    .addDef(HiLHS)
    .addUse(Reg);
}

/// Replace the current type each register in \p Regs has with \p NewTy
static void setRegsToType(MachineRegisterInfo &MRI, ArrayRef<Register> Regs,
                          LLT NewTy) {
  for (Register Reg : Regs) {
    assert(MRI.getType(Reg).getSizeInBits() == NewTy.getSizeInBits());
    MRI.setType(Reg, NewTy);
  }
}

static LLT getHalfSizedType(LLT Ty) {
  if (Ty.isVector()) {
    assert(Ty.getNumElements() % 2 == 0);
    return LLT::scalarOrVector(Ty.getNumElements() / 2, Ty.getElementType());
  }

  assert(Ty.getSizeInBits() % 2 == 0);
  return LLT::scalar(Ty.getSizeInBits() / 2);
}

/// Legalize instruction \p MI where operands in \p OpIndices must be SGPRs. If
/// any of the required SGPR operands are VGPRs, perform a waterfall loop to
/// execute the instruction for each unique combination of values in all lanes
/// in the wave. The block will be split such that rest of the instructions are
/// moved to a new block.
///
/// Essentially performs this loop:
//
/// Save Execution Mask
/// For (Lane : Wavefront) {
///   Enable Lane, Disable all other lanes
///   SGPR = read SGPR value for current lane from VGPR
///   VGPRResult[Lane] = use_op SGPR
/// }
/// Restore Execution Mask
///
/// There is additional complexity to try for compare values to identify the
/// unique values used.
bool AMDGPURegisterBankInfo::executeInWaterfallLoop(
  MachineIRBuilder &B,
  iterator_range<MachineBasicBlock::iterator> Range,
  SmallSet<Register, 4> &SGPROperandRegs,
  MachineRegisterInfo &MRI) const {
  SmallVector<Register, 4> ResultRegs;
  SmallVector<Register, 4> InitResultRegs;
  SmallVector<Register, 4> PhiRegs;

  // Track use registers which have already been expanded with a readfirstlane
  // sequence. This may have multiple uses if moving a sequence.
  DenseMap<Register, Register> WaterfalledRegMap;

  MachineBasicBlock &MBB = B.getMBB();
  MachineFunction *MF = &B.getMF();

  const TargetRegisterClass *WaveRC = TRI->getWaveMaskRegClass();
  const unsigned WaveAndOpc = Subtarget.isWave32() ?
    AMDGPU::S_AND_B32 : AMDGPU::S_AND_B64;
  const unsigned MovTermOpc = Subtarget.isWave32() ?
    AMDGPU::S_MOV_B32_term : AMDGPU::S_MOV_B64_term;
  const unsigned XorTermOpc = Subtarget.isWave32() ?
    AMDGPU::S_XOR_B32_term : AMDGPU::S_XOR_B64_term;
  const unsigned AndSaveExecOpc =  Subtarget.isWave32() ?
    AMDGPU::S_AND_SAVEEXEC_B32 : AMDGPU::S_AND_SAVEEXEC_B64;
  const unsigned ExecReg =  Subtarget.isWave32() ?
    AMDGPU::EXEC_LO : AMDGPU::EXEC;

#ifndef NDEBUG
  const int OrigRangeSize = std::distance(Range.begin(), Range.end());
#endif

  for (MachineInstr &MI : Range) {
    for (MachineOperand &Def : MI.defs()) {
      LLT ResTy = MRI.getType(Def.getReg());
      const RegisterBank *DefBank = getRegBank(Def.getReg(), MRI, *TRI);
      ResultRegs.push_back(Def.getReg());
      Register InitReg = B.buildUndef(ResTy).getReg(0);
      Register PhiReg = MRI.createGenericVirtualRegister(ResTy);
      InitResultRegs.push_back(InitReg);
      PhiRegs.push_back(PhiReg);
      MRI.setRegBank(PhiReg, *DefBank);
      MRI.setRegBank(InitReg, *DefBank);
    }
  }

  Register SaveExecReg = MRI.createVirtualRegister(WaveRC);
  Register InitSaveExecReg = MRI.createVirtualRegister(WaveRC);

  // Don't bother using generic instructions/registers for the exec mask.
  B.buildInstr(TargetOpcode::IMPLICIT_DEF)
    .addDef(InitSaveExecReg);

  Register PhiExec = MRI.createVirtualRegister(WaveRC);
  Register NewExec = MRI.createVirtualRegister(WaveRC);

  // To insert the loop we need to split the block. Move everything before this
  // point to a new block, and insert a new empty block before this instruction.
  MachineBasicBlock *LoopBB = MF->CreateMachineBasicBlock();
  MachineBasicBlock *RemainderBB = MF->CreateMachineBasicBlock();
  MachineBasicBlock *RestoreExecBB = MF->CreateMachineBasicBlock();
  MachineFunction::iterator MBBI(MBB);
  ++MBBI;
  MF->insert(MBBI, LoopBB);
  MF->insert(MBBI, RestoreExecBB);
  MF->insert(MBBI, RemainderBB);

  LoopBB->addSuccessor(RestoreExecBB);
  LoopBB->addSuccessor(LoopBB);

  // Move the rest of the block into a new block.
  RemainderBB->transferSuccessorsAndUpdatePHIs(&MBB);
  RemainderBB->splice(RemainderBB->begin(), &MBB, Range.end(), MBB.end());

  MBB.addSuccessor(LoopBB);
  RestoreExecBB->addSuccessor(RemainderBB);

  B.setInsertPt(*LoopBB, LoopBB->end());

  B.buildInstr(TargetOpcode::PHI)
    .addDef(PhiExec)
    .addReg(InitSaveExecReg)
    .addMBB(&MBB)
    .addReg(NewExec)
    .addMBB(LoopBB);

  for (auto Result : zip(InitResultRegs, ResultRegs, PhiRegs)) {
    B.buildInstr(TargetOpcode::G_PHI)
      .addDef(std::get<2>(Result))
      .addReg(std::get<0>(Result)) // Initial value / implicit_def
      .addMBB(&MBB)
      .addReg(std::get<1>(Result)) // Mid-loop value.
      .addMBB(LoopBB);
  }

  const DebugLoc &DL = B.getDL();

  MachineInstr &FirstInst = *Range.begin();

  // Move the instruction into the loop. Note we moved everything after
  // Range.end() already into a new block, so Range.end() is no longer valid.
  LoopBB->splice(LoopBB->end(), &MBB, Range.begin(), MBB.end());

  // Figure out the iterator range after splicing the instructions.
  MachineBasicBlock::iterator NewBegin = FirstInst.getIterator();
  auto NewEnd = LoopBB->end();

  MachineBasicBlock::iterator I = Range.begin();
  B.setInsertPt(*LoopBB, I);

  Register CondReg;

  assert(std::distance(NewBegin, NewEnd) == OrigRangeSize);

  for (MachineInstr &MI : make_range(NewBegin, NewEnd)) {
    for (MachineOperand &Op : MI.uses()) {
      if (!Op.isReg() || Op.isDef())
        continue;

      Register OldReg = Op.getReg();
      if (!SGPROperandRegs.count(OldReg))
        continue;

      // See if we already processed this register in another instruction in the
      // sequence.
      auto OldVal = WaterfalledRegMap.find(OldReg);
      if (OldVal != WaterfalledRegMap.end()) {
        Op.setReg(OldVal->second);
        continue;
      }

      LLT OpTy = MRI.getType(Op.getReg());
      unsigned OpSize = OpTy.getSizeInBits();

      // Can only do a readlane of 32-bit pieces.
      if (OpSize == 32) {
        // Avoid extra copies in the simple case of one 32-bit register.
        Register CurrentLaneOpReg
          = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
        MRI.setType(CurrentLaneOpReg, OpTy);

        constrainGenericRegister(Op.getReg(), AMDGPU::VGPR_32RegClass, MRI);
        // Read the next variant <- also loop target.
        BuildMI(*LoopBB, I, DL, TII->get(AMDGPU::V_READFIRSTLANE_B32),
                CurrentLaneOpReg)
          .addReg(Op.getReg());

        Register NewCondReg = MRI.createVirtualRegister(WaveRC);
        bool First = CondReg == AMDGPU::NoRegister;
        if (First)
          CondReg = NewCondReg;

        // Compare the just read M0 value to all possible Idx values.
        B.buildInstr(AMDGPU::V_CMP_EQ_U32_e64)
          .addDef(NewCondReg)
          .addReg(CurrentLaneOpReg)
          .addReg(Op.getReg());
        Op.setReg(CurrentLaneOpReg);

        if (!First) {
          Register AndReg = MRI.createVirtualRegister(WaveRC);

          // If there are multiple operands to consider, and the conditions.
          B.buildInstr(WaveAndOpc)
            .addDef(AndReg)
            .addReg(NewCondReg)
            .addReg(CondReg);
          CondReg = AndReg;
        }
      } else {
        LLT S32 = LLT::scalar(32);
        SmallVector<Register, 8> ReadlanePieces;

        // The compares can be done as 64-bit, but the extract needs to be done
        // in 32-bit pieces.

        bool Is64 = OpSize % 64 == 0;

        LLT UnmergeTy = OpSize % 64 == 0 ? LLT::scalar(64) : LLT::scalar(32);
        unsigned CmpOp = OpSize % 64 == 0 ? AMDGPU::V_CMP_EQ_U64_e64
          : AMDGPU::V_CMP_EQ_U32_e64;

        // The compares can be done as 64-bit, but the extract needs to be done
        // in 32-bit pieces.

        // Insert the unmerge before the loop.

        B.setMBB(MBB);
        auto Unmerge = B.buildUnmerge(UnmergeTy, Op.getReg());
        B.setInstr(*I);

        unsigned NumPieces = Unmerge->getNumOperands() - 1;
        for (unsigned PieceIdx = 0; PieceIdx != NumPieces; ++PieceIdx) {
          Register UnmergePiece = Unmerge.getReg(PieceIdx);

          Register CurrentLaneOpReg;
          if (Is64) {
            Register CurrentLaneOpRegLo = MRI.createGenericVirtualRegister(S32);
            Register CurrentLaneOpRegHi = MRI.createGenericVirtualRegister(S32);

            MRI.setRegClass(UnmergePiece, &AMDGPU::VReg_64RegClass);
            MRI.setRegClass(CurrentLaneOpRegLo, &AMDGPU::SReg_32_XM0RegClass);
            MRI.setRegClass(CurrentLaneOpRegHi, &AMDGPU::SReg_32_XM0RegClass);

            // Read the next variant <- also loop target.
            BuildMI(*LoopBB, I, DL, TII->get(AMDGPU::V_READFIRSTLANE_B32),
                    CurrentLaneOpRegLo)
              .addReg(UnmergePiece, 0, AMDGPU::sub0);

            // Read the next variant <- also loop target.
            BuildMI(*LoopBB, I, DL, TII->get(AMDGPU::V_READFIRSTLANE_B32),
                    CurrentLaneOpRegHi)
              .addReg(UnmergePiece, 0, AMDGPU::sub1);

            CurrentLaneOpReg =
              B.buildMerge(LLT::scalar(64),
                           {CurrentLaneOpRegLo, CurrentLaneOpRegHi})
              .getReg(0);

            MRI.setRegClass(CurrentLaneOpReg, &AMDGPU::SReg_64_XEXECRegClass);

            if (OpTy.getScalarSizeInBits() == 64) {
              // If we need to produce a 64-bit element vector, so use the
              // merged pieces
              ReadlanePieces.push_back(CurrentLaneOpReg);
            } else {
              // 32-bit element type.
              ReadlanePieces.push_back(CurrentLaneOpRegLo);
              ReadlanePieces.push_back(CurrentLaneOpRegHi);
            }
          } else {
            CurrentLaneOpReg = MRI.createGenericVirtualRegister(S32);
            MRI.setRegClass(UnmergePiece, &AMDGPU::VGPR_32RegClass);
            MRI.setRegClass(CurrentLaneOpReg, &AMDGPU::SReg_32_XM0RegClass);

            // Read the next variant <- also loop target.
            BuildMI(*LoopBB, I, DL, TII->get(AMDGPU::V_READFIRSTLANE_B32),
                    CurrentLaneOpReg)
              .addReg(UnmergePiece);
            ReadlanePieces.push_back(CurrentLaneOpReg);
          }

          Register NewCondReg = MRI.createVirtualRegister(WaveRC);
          bool First = CondReg == AMDGPU::NoRegister;
          if (First)
            CondReg = NewCondReg;

          B.buildInstr(CmpOp)
            .addDef(NewCondReg)
            .addReg(CurrentLaneOpReg)
            .addReg(UnmergePiece);

          if (!First) {
            Register AndReg = MRI.createVirtualRegister(WaveRC);

            // If there are multiple operands to consider, and the conditions.
            B.buildInstr(WaveAndOpc)
              .addDef(AndReg)
              .addReg(NewCondReg)
              .addReg(CondReg);
            CondReg = AndReg;
          }
        }

        // FIXME: Build merge seems to switch to CONCAT_VECTORS but not
        // BUILD_VECTOR
        if (OpTy.isVector()) {
          auto Merge = B.buildBuildVector(OpTy, ReadlanePieces);
          Op.setReg(Merge.getReg(0));
        } else {
          auto Merge = B.buildMerge(OpTy, ReadlanePieces);
          Op.setReg(Merge.getReg(0));
        }

        MRI.setRegBank(Op.getReg(), AMDGPU::SGPRRegBank);
      }

      // Make sure we don't re-process this register again.
      WaterfalledRegMap.insert(std::make_pair(OldReg, Op.getReg()));
    }
  }

  B.setInsertPt(*LoopBB, LoopBB->end());

  // Update EXEC, save the original EXEC value to VCC.
  B.buildInstr(AndSaveExecOpc)
    .addDef(NewExec)
    .addReg(CondReg, RegState::Kill);

  MRI.setSimpleHint(NewExec, CondReg);

  // Update EXEC, switch all done bits to 0 and all todo bits to 1.
  B.buildInstr(XorTermOpc)
    .addDef(ExecReg)
    .addReg(ExecReg)
    .addReg(NewExec);

  // XXX - s_xor_b64 sets scc to 1 if the result is nonzero, so can we use
  // s_cbranch_scc0?

  // Loop back to V_READFIRSTLANE_B32 if there are still variants to cover.
  B.buildInstr(AMDGPU::S_CBRANCH_EXECNZ)
    .addMBB(LoopBB);

  // Save the EXEC mask before the loop.
  BuildMI(MBB, MBB.end(), DL, TII->get(MovTermOpc), SaveExecReg)
    .addReg(ExecReg);

  // Restore the EXEC mask after the loop.
  B.setMBB(*RestoreExecBB);
  B.buildInstr(MovTermOpc)
    .addDef(ExecReg)
    .addReg(SaveExecReg);

  // Set the insert point after the original instruction, so any new
  // instructions will be in the remainder.
  B.setInsertPt(*RemainderBB, RemainderBB->begin());

  return true;
}

// Return any unique registers used by \p MI at \p OpIndices that need to be
// handled in a waterfall loop. Returns these registers in \p
// SGPROperandRegs. Returns true if there are any operansd to handle and a
// waterfall loop is necessary.
bool AMDGPURegisterBankInfo::collectWaterfallOperands(
  SmallSet<Register, 4> &SGPROperandRegs, MachineInstr &MI,
  MachineRegisterInfo &MRI, ArrayRef<unsigned> OpIndices) const {
  for (unsigned Op : OpIndices) {
    assert(MI.getOperand(Op).isUse());
    Register Reg = MI.getOperand(Op).getReg();
    const RegisterBank *OpBank = getRegBank(Reg, MRI, *TRI);
    if (OpBank->getID() == AMDGPU::VGPRRegBankID)
      SGPROperandRegs.insert(Reg);
  }

  // No operands need to be replaced, so no need to loop.
  return !SGPROperandRegs.empty();
}

bool AMDGPURegisterBankInfo::executeInWaterfallLoop(
  MachineIRBuilder &B, MachineInstr &MI, MachineRegisterInfo &MRI,
  ArrayRef<unsigned> OpIndices) const {
  // Use a set to avoid extra readfirstlanes in the case where multiple operands
  // are the same register.
  SmallSet<Register, 4> SGPROperandRegs;

  if (!collectWaterfallOperands(SGPROperandRegs, MI, MRI, OpIndices))
    return false;

  MachineBasicBlock::iterator I = MI.getIterator();
  return executeInWaterfallLoop(B, make_range(I, std::next(I)),
                                SGPROperandRegs, MRI);
}

bool AMDGPURegisterBankInfo::executeInWaterfallLoop(
  MachineInstr &MI, MachineRegisterInfo &MRI,
  ArrayRef<unsigned> OpIndices) const {
  MachineIRBuilder B(MI);
  return executeInWaterfallLoop(B, MI, MRI, OpIndices);
}

// Legalize an operand that must be an SGPR by inserting a readfirstlane.
void AMDGPURegisterBankInfo::constrainOpWithReadfirstlane(
    MachineInstr &MI, MachineRegisterInfo &MRI, unsigned OpIdx) const {
  Register Reg = MI.getOperand(OpIdx).getReg();
  const RegisterBank *Bank = getRegBank(Reg, MRI, *TRI);
  if (Bank != &AMDGPU::VGPRRegBank)
    return;

  MachineIRBuilder B(MI);
  Register SGPR = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
  B.buildInstr(AMDGPU::V_READFIRSTLANE_B32)
    .addDef(SGPR)
    .addReg(Reg);

  MRI.setType(SGPR, MRI.getType(Reg));

  const TargetRegisterClass *Constrained =
      constrainGenericRegister(Reg, AMDGPU::VGPR_32RegClass, MRI);
  (void)Constrained;
  assert(Constrained && "Failed to constrain readfirstlane src reg");

  MI.getOperand(OpIdx).setReg(SGPR);
}

// When regbankselect repairs registers, it will insert a repair instruction
// which defines the repaired register.  Then it calls applyMapping and expects
// that the targets will either delete or rewrite the originally wrote to the
// repaired registers.  Beccause of this, we end up in a situation where
// we have 2 instructions defining the same registers.
static MachineInstr *getOtherVRegDef(const MachineRegisterInfo &MRI,
                                     Register Reg,
                                     const MachineInstr &MI) {
  // Is there some way we can assert that there are exactly 2 def instructions?
  for (MachineInstr &Other : MRI.def_instructions(Reg)) {
    if (&Other != &MI)
      return &Other;
  }

  return nullptr;
}

bool AMDGPURegisterBankInfo::applyMappingWideLoad(MachineInstr &MI,
                        const AMDGPURegisterBankInfo::OperandsMapper &OpdMapper,
                                              MachineRegisterInfo &MRI) const {
  Register DstReg = MI.getOperand(0).getReg();
  const LLT LoadTy =  MRI.getType(DstReg);
  unsigned LoadSize = LoadTy.getSizeInBits();
  const unsigned MaxNonSmrdLoadSize = 128;
  // 128-bit loads are supported for all instruction types.
  if (LoadSize <= MaxNonSmrdLoadSize)
    return false;

  SmallVector<unsigned, 16> DefRegs(OpdMapper.getVRegs(0));
  SmallVector<unsigned, 1> SrcRegs(OpdMapper.getVRegs(1));

  // If the pointer is an SGPR, we have nothing to do.
  if (SrcRegs.empty()) {
    const RegisterBank *PtrBank =
      OpdMapper.getInstrMapping().getOperandMapping(1).BreakDown[0].RegBank;
    if (PtrBank == &AMDGPU::SGPRRegBank)
      return false;
    SrcRegs.push_back(MI.getOperand(1).getReg());
  }

  assert(LoadSize % MaxNonSmrdLoadSize == 0);

  // We want to get the repair instruction now, because it will help us
  // determine which instruction the legalizer inserts that will also
  // write to DstReg.
  MachineInstr *RepairInst = getOtherVRegDef(MRI, DstReg, MI);

  // RegBankSelect only emits scalar types, so we need to reset the pointer
  // operand to a pointer type.
  Register BasePtrReg = SrcRegs[0];
  LLT PtrTy = MRI.getType(MI.getOperand(1).getReg());
  MRI.setType(BasePtrReg, PtrTy);

  MachineIRBuilder B(MI);

  unsigned SplitElts =
      MaxNonSmrdLoadSize / LoadTy.getScalarType().getSizeInBits();
  const LLT LoadSplitTy =  LLT::vector(SplitElts, LoadTy.getScalarType());
  ApplyRegBankMapping O(*this, MRI, &AMDGPU::VGPRRegBank);
  GISelObserverWrapper Observer(&O);
  B.setChangeObserver(Observer);
  LegalizerHelper Helper(B.getMF(), Observer, B);
  if (Helper.fewerElementsVector(MI, 0, LoadSplitTy) != LegalizerHelper::Legalized)
    return false;

  // At this point, the legalizer has split the original load into smaller
  // loads.  At the end of lowering, it inserts an instruction (LegalizedInst)
  // that combines the outputs of the lower loads and writes it to DstReg.
  // The register bank selector has also added the RepairInst which writes to
  // DstReg as well.

  MachineInstr *LegalizedInst = getOtherVRegDef(MRI, DstReg, *RepairInst);

  // Replace the output of the LegalizedInst with a temporary register, since
  // RepairInst already defines DstReg.
  Register TmpReg = MRI.createGenericVirtualRegister(MRI.getType(DstReg));
  LegalizedInst->getOperand(0).setReg(TmpReg);
  B.setInsertPt(*RepairInst->getParent(), RepairInst);

  for (unsigned DefIdx = 0, e = DefRegs.size(); DefIdx != e; ++DefIdx) {
    Register IdxReg = B.buildConstant(LLT::scalar(32), DefIdx).getReg(0);
    MRI.setRegBank(IdxReg, AMDGPU::VGPRRegBank);
    B.buildExtractVectorElement(DefRegs[DefIdx], TmpReg, IdxReg);
  }

  MRI.setRegBank(DstReg, AMDGPU::VGPRRegBank);
  return true;
}

bool AMDGPURegisterBankInfo::applyMappingImage(
    MachineInstr &MI, const AMDGPURegisterBankInfo::OperandsMapper &OpdMapper,
    MachineRegisterInfo &MRI, int RsrcIdx) const {
  const int NumDefs = MI.getNumExplicitDefs();

  // The reported argument index is relative to the IR intrinsic call arguments,
  // so we need to shift by the number of defs and the intrinsic ID.
  RsrcIdx += NumDefs + 1;

  // Insert copies to VGPR arguments.
  applyDefaultMapping(OpdMapper);

  // Fixup any SGPR arguments.
  SmallVector<unsigned, 4> SGPRIndexes;
  for (int I = NumDefs, NumOps = MI.getNumOperands(); I != NumOps; ++I) {
    if (!MI.getOperand(I).isReg())
      continue;

    // If this intrinsic has a sampler, it immediately follows rsrc.
    if (I == RsrcIdx || I == RsrcIdx + 1)
      SGPRIndexes.push_back(I);
  }

  executeInWaterfallLoop(MI, MRI, SGPRIndexes);
  return true;
}

static Register getSrcRegIgnoringCopies(const MachineRegisterInfo &MRI,
                                        Register Reg) {
  MachineInstr *Def = getDefIgnoringCopies(Reg, MRI);
  if (!Def)
    return Reg;

  // TODO: Guard against this being an implicit def
  return Def->getOperand(0).getReg();
}

// Analyze a combined offset from an llvm.amdgcn.s.buffer intrinsic and store
// the three offsets (voffset, soffset and instoffset)
static unsigned setBufferOffsets(MachineIRBuilder &B,
                                 const AMDGPURegisterBankInfo &RBI,
                                 Register CombinedOffset,
                                 Register &VOffsetReg,
                                 Register &SOffsetReg,
                                 int64_t &InstOffsetVal,
                                 unsigned Align) {
  const LLT S32 = LLT::scalar(32);
  MachineRegisterInfo *MRI = B.getMRI();

  if (Optional<int64_t> Imm = getConstantVRegVal(CombinedOffset, *MRI)) {
    uint32_t SOffset, ImmOffset;
    if (AMDGPU::splitMUBUFOffset(*Imm, SOffset, ImmOffset,
                                 &RBI.Subtarget, Align)) {
      VOffsetReg = B.buildConstant(S32, 0).getReg(0);
      SOffsetReg = B.buildConstant(S32, SOffset).getReg(0);
      InstOffsetVal = ImmOffset;

      B.getMRI()->setRegBank(VOffsetReg, AMDGPU::VGPRRegBank);
      B.getMRI()->setRegBank(SOffsetReg, AMDGPU::SGPRRegBank);
      return SOffset + ImmOffset;
    }
  }

  Register Base;
  unsigned Offset;
  MachineInstr *Unused;

  std::tie(Base, Offset, Unused)
    = AMDGPU::getBaseWithConstantOffset(*MRI, CombinedOffset);

  uint32_t SOffset, ImmOffset;
  if (Offset > 0 && AMDGPU::splitMUBUFOffset(Offset, SOffset, ImmOffset,
                                             &RBI.Subtarget, Align)) {
    if (RBI.getRegBank(Base, *MRI, *RBI.TRI) == &AMDGPU::VGPRRegBank) {
      VOffsetReg = Base;
      SOffsetReg = B.buildConstant(S32, SOffset).getReg(0);
      B.getMRI()->setRegBank(SOffsetReg, AMDGPU::SGPRRegBank);
      InstOffsetVal = ImmOffset;
      return 0; // XXX - Why is this 0?
    }

    // If we have SGPR base, we can use it for soffset.
    if (SOffset == 0) {
      VOffsetReg = B.buildConstant(S32, 0).getReg(0);
      B.getMRI()->setRegBank(VOffsetReg, AMDGPU::VGPRRegBank);
      SOffsetReg = Base;
      InstOffsetVal = ImmOffset;
      return 0; // XXX - Why is this 0?
    }
  }

  // Handle the variable sgpr + vgpr case.
  if (MachineInstr *Add = getOpcodeDef(AMDGPU::G_ADD, CombinedOffset, *MRI)) {
    Register Src0 = getSrcRegIgnoringCopies(*MRI, Add->getOperand(1).getReg());
    Register Src1 = getSrcRegIgnoringCopies(*MRI, Add->getOperand(2).getReg());

    const RegisterBank *Src0Bank = RBI.getRegBank(Src0, *MRI, *RBI.TRI);
    const RegisterBank *Src1Bank = RBI.getRegBank(Src1, *MRI, *RBI.TRI);

    if (Src0Bank == &AMDGPU::VGPRRegBank && Src1Bank == &AMDGPU::SGPRRegBank) {
      VOffsetReg = Src0;
      SOffsetReg = Src1;
      return 0;
    }

    if (Src0Bank == &AMDGPU::SGPRRegBank && Src1Bank == &AMDGPU::VGPRRegBank) {
      VOffsetReg = Src1;
      SOffsetReg = Src0;
      return 0;
    }
  }

  // Ensure we have a VGPR for the combined offset. This could be an issue if we
  // have an SGPR offset and a VGPR resource.
  if (RBI.getRegBank(CombinedOffset, *MRI, *RBI.TRI) == &AMDGPU::VGPRRegBank) {
    VOffsetReg = CombinedOffset;
  } else {
    VOffsetReg = B.buildCopy(S32, CombinedOffset).getReg(0);
    B.getMRI()->setRegBank(VOffsetReg, AMDGPU::VGPRRegBank);
  }

  SOffsetReg = B.buildConstant(S32, 0).getReg(0);
  B.getMRI()->setRegBank(SOffsetReg, AMDGPU::SGPRRegBank);
  return 0;
}

static LLT divideLLT(LLT Ty, int Factor) {
  if (Ty.isVector())
    return LLT::vector(Ty.getNumElements() / Factor, Ty.getElementType());
  return LLT::scalar(Ty.getSizeInBits() / Factor);
}

bool AMDGPURegisterBankInfo::applyMappingSBufferLoad(
  const OperandsMapper &OpdMapper) const {
  MachineInstr &MI = OpdMapper.getMI();
  MachineRegisterInfo &MRI = OpdMapper.getMRI();

  const LLT S32 = LLT::scalar(32);
  Register Dst = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(Dst);

  const RegisterBank *RSrcBank =
    OpdMapper.getInstrMapping().getOperandMapping(1).BreakDown[0].RegBank;
  const RegisterBank *OffsetBank =
    OpdMapper.getInstrMapping().getOperandMapping(2).BreakDown[0].RegBank;
  if (RSrcBank == &AMDGPU::SGPRRegBank &&
      OffsetBank == &AMDGPU::SGPRRegBank)
    return true; // Legal mapping

  // FIXME: 96-bit case was widened during legalize. We neeed to narrow it back
  // here but don't have an MMO.

  unsigned LoadSize = Ty.getSizeInBits();
  int NumLoads = 1;
  if (LoadSize == 256 || LoadSize == 512) {
    NumLoads = LoadSize / 128;
    Ty = divideLLT(Ty, NumLoads);
  }

  // Use the alignment to ensure that the required offsets will fit into the
  // immediate offsets.
  const unsigned Align = NumLoads > 1 ? 16 * NumLoads : 1;

  MachineIRBuilder B(MI);
  MachineFunction &MF = B.getMF();

  Register SOffset;
  Register VOffset;
  int64_t ImmOffset = 0;

  unsigned MMOOffset = setBufferOffsets(B, *this, MI.getOperand(2).getReg(),
                                        VOffset, SOffset, ImmOffset, Align);

  // TODO: 96-bit loads were widened to 128-bit results. Shrink the result if we
  // can, but we neeed to track an MMO for that.
  const unsigned MemSize = (Ty.getSizeInBits() + 7) / 8;
  const unsigned MemAlign = 4; // FIXME: ABI type alignment?
  MachineMemOperand *BaseMMO = MF.getMachineMemOperand(
    MachinePointerInfo(),
    MachineMemOperand::MOLoad | MachineMemOperand::MODereferenceable |
    MachineMemOperand::MOInvariant,
    MemSize, MemAlign);
  if (MMOOffset != 0)
    BaseMMO = MF.getMachineMemOperand(BaseMMO, MMOOffset, MemSize);

  // If only the offset is divergent, emit a MUBUF buffer load instead. We can
  // assume that the buffer is unswizzled.

  Register RSrc = MI.getOperand(1).getReg();
  Register VIndex = B.buildConstant(S32, 0).getReg(0);
  B.getMRI()->setRegBank(VIndex, AMDGPU::VGPRRegBank);

  SmallVector<Register, 4> LoadParts(NumLoads);

  MachineBasicBlock::iterator MII = MI.getIterator();
  MachineInstrSpan Span(MII, &B.getMBB());

  for (int i = 0; i < NumLoads; ++i) {
    if (NumLoads == 1) {
      LoadParts[i] = Dst;
    } else {
      LoadParts[i] = MRI.createGenericVirtualRegister(Ty);
      MRI.setRegBank(LoadParts[i], AMDGPU::VGPRRegBank);
    }

    MachineMemOperand *MMO = BaseMMO;
    if (i != 0)
      BaseMMO = MF.getMachineMemOperand(BaseMMO, MMOOffset + 16 * i, MemSize);

    B.buildInstr(AMDGPU::G_AMDGPU_BUFFER_LOAD)
      .addDef(LoadParts[i])       // vdata
      .addUse(RSrc)               // rsrc
      .addUse(VIndex)             // vindex
      .addUse(VOffset)            // voffset
      .addUse(SOffset)            // soffset
      .addImm(ImmOffset + 16 * i) // offset(imm)
      .addImm(0)                  // cachepolicy, swizzled buffer(imm)
      .addImm(0)                  // idxen(imm)
      .addMemOperand(MMO);
  }

  // TODO: If only the resource is a VGPR, it may be better to execute the
  // scalar load in the waterfall loop if the resource is expected to frequently
  // be dynamically uniform.
  if (RSrcBank != &AMDGPU::SGPRRegBank) {
    // Remove the original instruction to avoid potentially confusing the
    // waterfall loop logic.
    B.setInstr(*Span.begin());
    MI.eraseFromParent();

    SmallSet<Register, 4> OpsToWaterfall;

    OpsToWaterfall.insert(RSrc);
    executeInWaterfallLoop(B, make_range(Span.begin(), Span.end()),
                           OpsToWaterfall, MRI);
  }

  if (NumLoads != 1) {
    if (Ty.isVector())
      B.buildConcatVectors(Dst, LoadParts);
    else
      B.buildMerge(Dst, LoadParts);
  }

  // We removed the instruction earlier with a waterfall loop.
  if (RSrcBank == &AMDGPU::SGPRRegBank)
    MI.eraseFromParent();

  return true;
}

bool AMDGPURegisterBankInfo::applyMappingBFEIntrinsic(
  const OperandsMapper &OpdMapper, bool Signed) const {
  MachineInstr &MI = OpdMapper.getMI();
  MachineRegisterInfo &MRI = OpdMapper.getMRI();

  // Insert basic copies
  applyDefaultMapping(OpdMapper);

  Register DstReg = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(DstReg);

  const LLT S32 = LLT::scalar(32);

  const RegisterBank *DstBank =
    OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
  if (DstBank == &AMDGPU::VGPRRegBank) {
    if (Ty == S32)
      return true;

    // TODO: 64-bit version is scalar only, so we need to expand this.
    return false;
  }

  Register SrcReg = MI.getOperand(2).getReg();
  Register OffsetReg = MI.getOperand(3).getReg();
  Register WidthReg = MI.getOperand(4).getReg();

  // The scalar form packs the offset and width in a single operand.

  ApplyRegBankMapping ApplyBank(*this, MRI, &AMDGPU::SGPRRegBank);
  GISelObserverWrapper Observer(&ApplyBank);
  MachineIRBuilder B(MI);
  B.setChangeObserver(Observer);

  // Ensure the high bits are clear to insert the offset.
  auto OffsetMask = B.buildConstant(S32, maskTrailingOnes<unsigned>(6));
  auto ClampOffset = B.buildAnd(S32, OffsetReg, OffsetMask);

  // Zeros out the low bits, so don't bother clamping the input value.
  auto ShiftWidth = B.buildShl(S32, WidthReg, B.buildConstant(S32, 16));

  // Transformation function, pack the offset and width of a BFE into
  // the format expected by the S_BFE_I32 / S_BFE_U32. In the second
  // source, bits [5:0] contain the offset and bits [22:16] the width.
  auto MergedInputs = B.buildOr(S32, ClampOffset, ShiftWidth);

  // TODO: It might be worth using a pseudo here to avoid scc clobber and
  // register class constraints.
  unsigned Opc = Ty == S32 ? (Signed ? AMDGPU::S_BFE_I32 : AMDGPU::S_BFE_U32) :
                             (Signed ? AMDGPU::S_BFE_I64 : AMDGPU::S_BFE_U64);

  auto MIB = B.buildInstr(Opc, {DstReg}, {SrcReg, MergedInputs});
  if (!constrainSelectedInstRegOperands(*MIB, *TII, *TRI, *this))
    llvm_unreachable("failed to constrain BFE");

  MI.eraseFromParent();
  return true;
}

// FIXME: Duplicated from LegalizerHelper
static CmpInst::Predicate minMaxToCompare(unsigned Opc) {
  switch (Opc) {
  case TargetOpcode::G_SMIN:
    return CmpInst::ICMP_SLT;
  case TargetOpcode::G_SMAX:
    return CmpInst::ICMP_SGT;
  case TargetOpcode::G_UMIN:
    return CmpInst::ICMP_ULT;
  case TargetOpcode::G_UMAX:
    return CmpInst::ICMP_UGT;
  default:
    llvm_unreachable("not in integer min/max");
  }
}

static unsigned minMaxToExtend(unsigned Opc) {
  switch (Opc) {
  case TargetOpcode::G_SMIN:
  case TargetOpcode::G_SMAX:
    return TargetOpcode::G_SEXT;
  case TargetOpcode::G_UMIN:
  case TargetOpcode::G_UMAX:
    return TargetOpcode::G_ZEXT;
  default:
    llvm_unreachable("not in integer min/max");
  }
}

// Emit a legalized extension from <2 x s16> to 2 32-bit components, avoiding
// any illegal vector extend or unmerge operations.
static std::pair<Register, Register>
unpackV2S16ToS32(MachineIRBuilder &B, Register Src, unsigned ExtOpcode) {
  const LLT S32 = LLT::scalar(32);
  auto Bitcast = B.buildBitcast(S32, Src);

  if (ExtOpcode == TargetOpcode::G_SEXT) {
    auto ExtLo = B.buildSExtInReg(S32, Bitcast, 16);
    auto ShiftHi = B.buildAShr(S32, Bitcast, B.buildConstant(S32, 16));
    return std::make_pair(ExtLo.getReg(0), ShiftHi.getReg(0));
  }

  auto ShiftHi = B.buildLShr(S32, Bitcast, B.buildConstant(S32, 16));
  if (ExtOpcode == TargetOpcode::G_ZEXT) {
    auto ExtLo = B.buildAnd(S32, Bitcast, B.buildConstant(S32, 0xffff));
    return std::make_pair(ExtLo.getReg(0), ShiftHi.getReg(0));
  }

  assert(ExtOpcode == TargetOpcode::G_ANYEXT);
  return std::make_pair(Bitcast.getReg(0), ShiftHi.getReg(0));
}

static MachineInstr *buildExpandedScalarMinMax(MachineIRBuilder &B,
                                               CmpInst::Predicate Pred,
                                               Register Dst, Register Src0,
                                               Register Src1) {
  const LLT CmpType = LLT::scalar(32);
  auto Cmp = B.buildICmp(Pred, CmpType, Src0, Src1);
  return B.buildSelect(Dst, Cmp, Src0, Src1);
}

// FIXME: Duplicated from LegalizerHelper, except changing the boolean type.
void AMDGPURegisterBankInfo::lowerScalarMinMax(MachineIRBuilder &B,
                                               MachineInstr &MI) const {
  Register Dst = MI.getOperand(0).getReg();
  Register Src0 = MI.getOperand(1).getReg();
  Register Src1 = MI.getOperand(2).getReg();

  const CmpInst::Predicate Pred = minMaxToCompare(MI.getOpcode());
  MachineInstr *Sel = buildExpandedScalarMinMax(B, Pred, Dst, Src0, Src1);

  Register CmpReg = Sel->getOperand(1).getReg();
  B.getMRI()->setRegBank(CmpReg, AMDGPU::SGPRRegBank);
  MI.eraseFromParent();
}

// For cases where only a single copy is inserted for matching register banks.
// Replace the register in the instruction operand
static bool substituteSimpleCopyRegs(
  const AMDGPURegisterBankInfo::OperandsMapper &OpdMapper, unsigned OpIdx) {
  SmallVector<unsigned, 1> SrcReg(OpdMapper.getVRegs(OpIdx));
  if (!SrcReg.empty()) {
    assert(SrcReg.size() == 1);
    OpdMapper.getMI().getOperand(OpIdx).setReg(SrcReg[0]);
    return true;
  }

  return false;
}

/// Handle register layout difference for f16 images for some subtargets.
Register AMDGPURegisterBankInfo::handleD16VData(MachineIRBuilder &B,
                                                MachineRegisterInfo &MRI,
                                                Register Reg) const {
  if (!Subtarget.hasUnpackedD16VMem())
    return Reg;

  const LLT S16 = LLT::scalar(16);
  LLT StoreVT = MRI.getType(Reg);
  if (!StoreVT.isVector() || StoreVT.getElementType() != S16)
    return Reg;

  auto Unmerge = B.buildUnmerge(S16, Reg);


  SmallVector<Register, 4> WideRegs;
  for (int I = 0, E = Unmerge->getNumOperands() - 1; I != E; ++I)
    WideRegs.push_back(Unmerge.getReg(I));

  const LLT S32 = LLT::scalar(32);
  int NumElts = StoreVT.getNumElements();

  return B.buildMerge(LLT::vector(NumElts, S32), WideRegs).getReg(0);
}

static std::pair<Register, unsigned>
getBaseWithConstantOffset(MachineRegisterInfo &MRI, Register Reg) {
  int64_t Const;
  if (mi_match(Reg, MRI, m_ICst(Const)))
    return std::make_pair(Register(), Const);

  Register Base;
  if (mi_match(Reg, MRI, m_GAdd(m_Reg(Base), m_ICst(Const))))
    return std::make_pair(Base, Const);

  // TODO: Handle G_OR used for add case
  return std::make_pair(Reg, 0);
}

std::pair<Register, unsigned>
AMDGPURegisterBankInfo::splitBufferOffsets(MachineIRBuilder &B,
                                           Register OrigOffset) const {
  const unsigned MaxImm = 4095;
  Register BaseReg;
  unsigned ImmOffset;
  const LLT S32 = LLT::scalar(32);

  std::tie(BaseReg, ImmOffset) = getBaseWithConstantOffset(*B.getMRI(),
                                                           OrigOffset);

  unsigned C1 = 0;
  if (ImmOffset != 0) {
    // If the immediate value is too big for the immoffset field, put the value
    // and -4096 into the immoffset field so that the value that is copied/added
    // for the voffset field is a multiple of 4096, and it stands more chance
    // of being CSEd with the copy/add for another similar load/store.
    // However, do not do that rounding down to a multiple of 4096 if that is a
    // negative number, as it appears to be illegal to have a negative offset
    // in the vgpr, even if adding the immediate offset makes it positive.
    unsigned Overflow = ImmOffset & ~MaxImm;
    ImmOffset -= Overflow;
    if ((int32_t)Overflow < 0) {
      Overflow += ImmOffset;
      ImmOffset = 0;
    }

    C1 = ImmOffset;
    if (Overflow != 0) {
      if (!BaseReg)
        BaseReg = B.buildConstant(S32, Overflow).getReg(0);
      else {
        auto OverflowVal = B.buildConstant(S32, Overflow);
        BaseReg = B.buildAdd(S32, BaseReg, OverflowVal).getReg(0);
      }
    }
  }

  if (!BaseReg)
    BaseReg = B.buildConstant(S32, 0).getReg(0);

  return {BaseReg, C1};
}

static bool isZero(Register Reg, MachineRegisterInfo &MRI) {
  int64_t C;
  return mi_match(Reg, MRI, m_ICst(C)) && C == 0;
}

static unsigned extractGLC(unsigned CachePolicy) {
  return CachePolicy & 1;
}

static unsigned extractSLC(unsigned CachePolicy) {
  return (CachePolicy >> 1) & 1;
}

static unsigned extractDLC(unsigned CachePolicy) {
  return (CachePolicy >> 2) & 1;
}

MachineInstr *
AMDGPURegisterBankInfo::selectStoreIntrinsic(MachineIRBuilder &B,
                                             MachineInstr &MI) const {
   MachineRegisterInfo &MRI = *B.getMRI();
  executeInWaterfallLoop(B, MI, MRI, {2, 4});

  // FIXME: DAG lowering brokenly changes opcode based on FP vs. integer.

  Register VData = MI.getOperand(1).getReg();
  LLT Ty = MRI.getType(VData);

  int EltSize = Ty.getScalarSizeInBits();
  int Size = Ty.getSizeInBits();

  // FIXME: Broken integer truncstore.
  if (EltSize != 32)
    report_fatal_error("unhandled intrinsic store");

  // FIXME: Verifier should enforce 1 MMO for these intrinsics.
  const int MemSize = (*MI.memoperands_begin())->getSize();


  Register RSrc = MI.getOperand(2).getReg();
  Register VOffset = MI.getOperand(3).getReg();
  Register SOffset = MI.getOperand(4).getReg();
  unsigned CachePolicy = MI.getOperand(5).getImm();

  unsigned ImmOffset;
  std::tie(VOffset, ImmOffset) = splitBufferOffsets(B, VOffset);

  const bool Offen = !isZero(VOffset, MRI);

  unsigned Opc = AMDGPU::BUFFER_STORE_DWORD_OFFEN_exact;
  switch (8 * MemSize) {
  case 8:
    Opc = Offen ? AMDGPU::BUFFER_STORE_BYTE_OFFEN_exact :
                  AMDGPU::BUFFER_STORE_BYTE_OFFSET_exact;
    break;
  case 16:
    Opc = Offen ? AMDGPU::BUFFER_STORE_SHORT_OFFEN_exact :
                  AMDGPU::BUFFER_STORE_SHORT_OFFSET_exact;
    break;
  default:
    Opc = Offen ? AMDGPU::BUFFER_STORE_DWORD_OFFEN_exact :
                  AMDGPU::BUFFER_STORE_DWORD_OFFSET_exact;
    if (Size > 32)
      Opc = AMDGPU::getMUBUFOpcode(Opc, Size / 32);
    break;
  }


  // Set the insertion point back to the instruction in case it was moved into a
  // loop.
  B.setInstr(MI);

  MachineInstrBuilder MIB = B.buildInstr(Opc)
    .addUse(VData);

  if (Offen)
    MIB.addUse(VOffset);

  MIB.addUse(RSrc)
     .addUse(SOffset)
     .addImm(ImmOffset)
     .addImm(extractGLC(CachePolicy))
     .addImm(extractSLC(CachePolicy))
     .addImm(0) // tfe: FIXME: Remove from inst
     .addImm(extractDLC(CachePolicy))
     .cloneMemRefs(MI);

  // FIXME: We need a way to report failure from applyMappingImpl.
  // Insert constrain copies before inserting the loop.
  if (!constrainSelectedInstRegOperands(*MIB, *TII, *TRI, *this))
    report_fatal_error("failed to constrain selected store intrinsic");

  return MIB;
}

bool AMDGPURegisterBankInfo::buildVCopy(MachineIRBuilder &B, Register DstReg,
                                        Register SrcReg) const {
  MachineRegisterInfo &MRI = *B.getMRI();
  LLT SrcTy = MRI.getType(SrcReg);
  if (SrcTy.getSizeInBits() == 32) {
    // Use a v_mov_b32 here to make the exec dependency explicit.
    B.buildInstr(AMDGPU::V_MOV_B32_e32)
      .addDef(DstReg)
      .addUse(SrcReg);
    return constrainGenericRegister(DstReg, AMDGPU::VGPR_32RegClass, MRI) &&
           constrainGenericRegister(SrcReg, AMDGPU::SReg_32RegClass, MRI);
  }

  Register TmpReg0 = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register TmpReg1 = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);

  B.buildInstr(AMDGPU::V_MOV_B32_e32)
    .addDef(TmpReg0)
    .addUse(SrcReg, 0, AMDGPU::sub0);
  B.buildInstr(AMDGPU::V_MOV_B32_e32)
    .addDef(TmpReg1)
    .addUse(SrcReg, 0, AMDGPU::sub1);
  B.buildInstr(AMDGPU::REG_SEQUENCE)
    .addDef(DstReg)
    .addUse(TmpReg0)
    .addImm(AMDGPU::sub0)
    .addUse(TmpReg1)
    .addImm(AMDGPU::sub1);

  return constrainGenericRegister(SrcReg, AMDGPU::SReg_64RegClass, MRI) &&
         constrainGenericRegister(DstReg, AMDGPU::VReg_64RegClass, MRI);
}

/// Utility function for pushing dynamic vector indexes with a constant offset
/// into waterwall loops.
static void reinsertVectorIndexAdd(MachineIRBuilder &B,
                                   MachineInstr &IdxUseInstr,
                                   unsigned OpIdx,
                                   unsigned ConstOffset) {
  MachineRegisterInfo &MRI = *B.getMRI();
  const LLT S32 = LLT::scalar(32);
  Register WaterfallIdx = IdxUseInstr.getOperand(OpIdx).getReg();
  B.setInsertPt(*IdxUseInstr.getParent(), IdxUseInstr.getIterator());

  auto MaterializedOffset = B.buildConstant(S32, ConstOffset);

  auto Add = B.buildAdd(S32, WaterfallIdx, MaterializedOffset);
  MRI.setRegBank(MaterializedOffset.getReg(0), AMDGPU::SGPRRegBank);
  MRI.setRegBank(Add.getReg(0), AMDGPU::SGPRRegBank);
  IdxUseInstr.getOperand(OpIdx).setReg(Add.getReg(0));
}

void AMDGPURegisterBankInfo::applyMappingImpl(
    const OperandsMapper &OpdMapper) const {
  MachineInstr &MI = OpdMapper.getMI();
  unsigned Opc = MI.getOpcode();
  MachineRegisterInfo &MRI = OpdMapper.getMRI();
  switch (Opc) {
  case AMDGPU::G_PHI: {
    Register DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);
    if (DstTy != LLT::scalar(1))
      break;

    const LLT S32 = LLT::scalar(32);
    const RegisterBank *DstBank =
      OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
    if (DstBank == &AMDGPU::VCCRegBank) {
      applyDefaultMapping(OpdMapper);
      // The standard handling only considers the result register bank for
      // phis. For VCC, blindly inserting a copy when the phi is lowered will
      // produce an invalid copy. We can only copy with some kind of compare to
      // get a vector boolean result. Insert a regitser bank copy that will be
      // correctly lowered to a compare.
      MachineIRBuilder B(*MI.getParent()->getParent());

      for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2) {
        Register SrcReg = MI.getOperand(I).getReg();
        const RegisterBank *SrcBank = getRegBank(SrcReg, MRI, *TRI);

        if (SrcBank != &AMDGPU::VCCRegBank) {
          MachineBasicBlock *SrcMBB = MI.getOperand(I + 1).getMBB();
          B.setInsertPt(*SrcMBB, SrcMBB->getFirstTerminator());

          auto Copy = B.buildCopy(LLT::scalar(1), SrcReg);
          MRI.setRegBank(Copy.getReg(0), AMDGPU::VCCRegBank);
          MI.getOperand(I).setReg(Copy.getReg(0));
        }
      }

      return;
    }

    // Phi handling is strange and only considers the bank of the destination.
    substituteSimpleCopyRegs(OpdMapper, 0);

    // Promote SGPR/VGPR booleans to s32
    MachineFunction *MF = MI.getParent()->getParent();
    ApplyRegBankMapping ApplyBank(*this, MRI, DstBank);
    GISelObserverWrapper Observer(&ApplyBank);
    MachineIRBuilder B(MI);
    LegalizerHelper Helper(*MF, Observer, B);

    if (Helper.widenScalar(MI, 0, S32) != LegalizerHelper::Legalized)
      llvm_unreachable("widen scalar should have succeeded");

    return;
  }
  case AMDGPU::G_ICMP:
  case AMDGPU::G_UADDO:
  case AMDGPU::G_USUBO:
  case AMDGPU::G_UADDE:
  case AMDGPU::G_SADDE:
  case AMDGPU::G_USUBE:
  case AMDGPU::G_SSUBE: {
    unsigned BoolDstOp = Opc == AMDGPU::G_ICMP ? 0 : 1;
    Register DstReg = MI.getOperand(BoolDstOp).getReg();

    const RegisterBank *DstBank =
      OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
    if (DstBank != &AMDGPU::SGPRRegBank)
      break;

    const bool HasCarryIn = MI.getNumOperands() == 5;

    // If this is a scalar compare, promote the result to s32, as the selection
    // will end up using a copy to a 32-bit vreg.
    const LLT S32 = LLT::scalar(32);
    Register NewDstReg = MRI.createGenericVirtualRegister(S32);
    MRI.setRegBank(NewDstReg, AMDGPU::SGPRRegBank);
    MI.getOperand(BoolDstOp).setReg(NewDstReg);
    MachineIRBuilder B(MI);

    if (HasCarryIn) {
      Register NewSrcReg = MRI.createGenericVirtualRegister(S32);
      MRI.setRegBank(NewSrcReg, AMDGPU::SGPRRegBank);
      B.buildZExt(NewSrcReg, MI.getOperand(4).getReg());
      MI.getOperand(4).setReg(NewSrcReg);
    }

    MachineBasicBlock *MBB = MI.getParent();
    B.setInsertPt(*MBB, std::next(MI.getIterator()));

    // If we had a constrained VCC result register, a copy was inserted to VCC
    // from SGPR.
    SmallVector<Register, 1> DefRegs(OpdMapper.getVRegs(0));
    if (DefRegs.empty())
      DefRegs.push_back(DstReg);
    B.buildTrunc(DefRegs[0], NewDstReg);
    return;
  }
  case AMDGPU::G_SELECT: {
    Register DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);

    SmallVector<Register, 1> CondRegs(OpdMapper.getVRegs(1));
    if (CondRegs.empty())
      CondRegs.push_back(MI.getOperand(1).getReg());
    else {
      assert(CondRegs.size() == 1);
    }

    const RegisterBank *CondBank = getRegBank(CondRegs[0], MRI, *TRI);
    if (CondBank == &AMDGPU::SGPRRegBank) {
      MachineIRBuilder B(MI);
      const LLT S32 = LLT::scalar(32);
      Register NewCondReg = MRI.createGenericVirtualRegister(S32);
      MRI.setRegBank(NewCondReg, AMDGPU::SGPRRegBank);

      MI.getOperand(1).setReg(NewCondReg);
      B.buildZExt(NewCondReg, CondRegs[0]);
    }

    if (DstTy.getSizeInBits() != 64)
      break;

    MachineIRBuilder B(MI);
    LLT HalfTy = getHalfSizedType(DstTy);

    SmallVector<Register, 2> DefRegs(OpdMapper.getVRegs(0));
    SmallVector<Register, 2> Src1Regs(OpdMapper.getVRegs(2));
    SmallVector<Register, 2> Src2Regs(OpdMapper.getVRegs(3));

    // All inputs are SGPRs, nothing special to do.
    if (DefRegs.empty()) {
      assert(Src1Regs.empty() && Src2Regs.empty());
      break;
    }

    if (Src1Regs.empty())
      split64BitValueForMapping(B, Src1Regs, HalfTy, MI.getOperand(2).getReg());
    else {
      setRegsToType(MRI, Src1Regs, HalfTy);
    }

    if (Src2Regs.empty())
      split64BitValueForMapping(B, Src2Regs, HalfTy, MI.getOperand(3).getReg());
    else
      setRegsToType(MRI, Src2Regs, HalfTy);

    setRegsToType(MRI, DefRegs, HalfTy);

    B.buildSelect(DefRegs[0], CondRegs[0], Src1Regs[0], Src2Regs[0]);
    B.buildSelect(DefRegs[1], CondRegs[0], Src1Regs[1], Src2Regs[1]);

    MRI.setRegBank(DstReg, AMDGPU::VGPRRegBank);
    MI.eraseFromParent();
    return;
  }
  case AMDGPU::G_BRCOND: {
    Register CondReg = MI.getOperand(0).getReg();
    // FIXME: Should use legalizer helper, but should change bool ext type.
    const RegisterBank *CondBank =
      OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;

    if (CondBank == &AMDGPU::SGPRRegBank) {
      MachineIRBuilder B(MI);
      const LLT S32 = LLT::scalar(32);
      Register NewCondReg = MRI.createGenericVirtualRegister(S32);
      MRI.setRegBank(NewCondReg, AMDGPU::SGPRRegBank);

      MI.getOperand(0).setReg(NewCondReg);
      B.buildZExt(NewCondReg, CondReg);
      return;
    }

    break;
  }
  case AMDGPU::G_AND:
  case AMDGPU::G_OR:
  case AMDGPU::G_XOR: {
    // 64-bit and is only available on the SALU, so split into 2 32-bit ops if
    // there is a VGPR input.
    Register DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);

    if (DstTy.getSizeInBits() == 1) {
      const RegisterBank *DstBank =
        OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
      if (DstBank == &AMDGPU::VCCRegBank)
        break;

      MachineFunction *MF = MI.getParent()->getParent();
      ApplyRegBankMapping ApplyBank(*this, MRI, DstBank);
      GISelObserverWrapper Observer(&ApplyBank);
      MachineIRBuilder B(MI);
      LegalizerHelper Helper(*MF, Observer, B);

      if (Helper.widenScalar(MI, 0, LLT::scalar(32)) !=
          LegalizerHelper::Legalized)
        llvm_unreachable("widen scalar should have succeeded");
      return;
    }

    if (DstTy.getSizeInBits() != 64)
      break;

    LLT HalfTy = getHalfSizedType(DstTy);
    SmallVector<Register, 2> DefRegs(OpdMapper.getVRegs(0));
    SmallVector<Register, 2> Src0Regs(OpdMapper.getVRegs(1));
    SmallVector<Register, 2> Src1Regs(OpdMapper.getVRegs(2));

    // All inputs are SGPRs, nothing special to do.
    if (DefRegs.empty()) {
      assert(Src0Regs.empty() && Src1Regs.empty());
      break;
    }

    assert(DefRegs.size() == 2);
    assert(Src0Regs.size() == Src1Regs.size() &&
           (Src0Regs.empty() || Src0Regs.size() == 2));

    // Depending on where the source registers came from, the generic code may
    // have decided to split the inputs already or not. If not, we still need to
    // extract the values.
    MachineIRBuilder B(MI);

    if (Src0Regs.empty())
      split64BitValueForMapping(B, Src0Regs, HalfTy, MI.getOperand(1).getReg());
    else
      setRegsToType(MRI, Src0Regs, HalfTy);

    if (Src1Regs.empty())
      split64BitValueForMapping(B, Src1Regs, HalfTy, MI.getOperand(2).getReg());
    else
      setRegsToType(MRI, Src1Regs, HalfTy);

    setRegsToType(MRI, DefRegs, HalfTy);

    B.buildInstr(Opc)
      .addDef(DefRegs[0])
      .addUse(Src0Regs[0])
      .addUse(Src1Regs[0]);

    B.buildInstr(Opc)
      .addDef(DefRegs[1])
      .addUse(Src0Regs[1])
      .addUse(Src1Regs[1]);

    MRI.setRegBank(DstReg, AMDGPU::VGPRRegBank);
    MI.eraseFromParent();
    return;
  }
  case AMDGPU::G_ADD:
  case AMDGPU::G_SUB:
  case AMDGPU::G_MUL: {
    Register DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);
    const LLT S32 = LLT::scalar(32);
    if (DstTy == S32)
      break;

    const RegisterBank *DstBank =
      OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
    if (DstBank == &AMDGPU::VGPRRegBank)
      break;

    // 16-bit operations are VALU only, but can be promoted to 32-bit SALU.
    // Packed 16-bit operations need to be scalarized and promoted.

    MachineFunction *MF = MI.getParent()->getParent();
    MachineIRBuilder B(MI);
    ApplyRegBankMapping ApplySALU(*this, MRI, &AMDGPU::SGPRRegBank);
    GISelObserverWrapper Observer(&ApplySALU);

    if (DstTy.isVector()) {
      B.setChangeObserver(Observer);

      Register WideSrc0Lo, WideSrc0Hi;
      Register WideSrc1Lo, WideSrc1Hi;

      std::tie(WideSrc0Lo, WideSrc0Hi)
        = unpackV2S16ToS32(B, MI.getOperand(1).getReg(), AMDGPU::G_ANYEXT);
      std::tie(WideSrc1Lo, WideSrc1Hi)
        = unpackV2S16ToS32(B, MI.getOperand(2).getReg(), AMDGPU::G_ANYEXT);
      auto Lo = B.buildInstr(MI.getOpcode(), {S32}, {WideSrc0Lo, WideSrc1Lo});
      auto Hi = B.buildInstr(MI.getOpcode(), {S32}, {WideSrc0Hi, WideSrc1Hi});
      B.buildBuildVectorTrunc(DstReg, {Lo.getReg(0), Hi.getReg(0)});
      MI.eraseFromParent();
    } else {
      LegalizerHelper Helper(*MF, Observer, B);

      if (Helper.widenScalar(MI, 0, S32) != LegalizerHelper::Legalized)
        llvm_unreachable("widen scalar should have succeeded");
    }

    return;
  }
  case AMDGPU::G_SMIN:
  case AMDGPU::G_SMAX:
  case AMDGPU::G_UMIN:
  case AMDGPU::G_UMAX: {
    Register DstReg = MI.getOperand(0).getReg();
    const RegisterBank *DstBank =
      OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
    if (DstBank == &AMDGPU::VGPRRegBank)
      break;

    MachineFunction *MF = MI.getParent()->getParent();
    MachineIRBuilder B(MI);

    // Turn scalar min/max into a compare and select.
    LLT Ty = MRI.getType(DstReg);
    const LLT S32 = LLT::scalar(32);
    const LLT S16 = LLT::scalar(16);
    const LLT V2S16 = LLT::vector(2, 16);

    if (Ty == V2S16) {
      ApplyRegBankMapping ApplySALU(*this, MRI, &AMDGPU::SGPRRegBank);
      GISelObserverWrapper Observer(&ApplySALU);
      B.setChangeObserver(Observer);

      // Need to widen to s32, and expand as cmp + select, and avoid producing
      // illegal vector extends or unmerges that would need further
      // legalization.
      //
      // TODO: Should we just readfirstlane? That should probably be handled
      // with a UniformVGPR register bank that wouldn't need special
      // consideration here.

      Register Dst = MI.getOperand(0).getReg();
      Register Src0 = MI.getOperand(1).getReg();
      Register Src1 = MI.getOperand(2).getReg();

      Register WideSrc0Lo, WideSrc0Hi;
      Register WideSrc1Lo, WideSrc1Hi;

      unsigned ExtendOp = minMaxToExtend(MI.getOpcode());

      std::tie(WideSrc0Lo, WideSrc0Hi) = unpackV2S16ToS32(B, Src0, ExtendOp);
      std::tie(WideSrc1Lo, WideSrc1Hi) = unpackV2S16ToS32(B, Src1, ExtendOp);

      Register Lo = MRI.createGenericVirtualRegister(S32);
      Register Hi = MRI.createGenericVirtualRegister(S32);
      const CmpInst::Predicate Pred = minMaxToCompare(MI.getOpcode());
      buildExpandedScalarMinMax(B, Pred, Lo, WideSrc0Lo, WideSrc1Lo);
      buildExpandedScalarMinMax(B, Pred, Hi, WideSrc0Hi, WideSrc1Hi);

      B.buildBuildVectorTrunc(Dst, {Lo, Hi});
      MI.eraseFromParent();
    } else if (Ty == S16) {
      ApplyRegBankMapping ApplySALU(*this, MRI, &AMDGPU::SGPRRegBank);
      GISelObserverWrapper Observer(&ApplySALU);
      LegalizerHelper Helper(*MF, Observer, B);

      // Need to widen to s32, and expand as cmp + select.
      if (Helper.widenScalar(MI, 0, S32) != LegalizerHelper::Legalized)
        llvm_unreachable("widenScalar should have succeeded");

      // FIXME: This is relying on widenScalar leaving MI in place.
      lowerScalarMinMax(B, MI);
    } else
      lowerScalarMinMax(B, MI);

    return;
  }
  case AMDGPU::G_SEXT_INREG: {
    SmallVector<Register, 2> SrcRegs(OpdMapper.getVRegs(1));
    if (SrcRegs.empty())
      break; // Nothing to repair

    const LLT S32 = LLT::scalar(32);
    MachineIRBuilder B(MI);
    ApplyRegBankMapping O(*this, MRI, &AMDGPU::VGPRRegBank);
    GISelObserverWrapper Observer(&O);
    B.setChangeObserver(Observer);

    // Don't use LegalizerHelper's narrowScalar. It produces unwanted G_SEXTs
    // we would need to further expand, and doesn't let us directly set the
    // result registers.
    SmallVector<Register, 2> DstRegs(OpdMapper.getVRegs(0));

    int Amt = MI.getOperand(2).getImm();
    if (Amt <= 32) {
      if (Amt == 32) {
        // The low bits are unchanged.
        B.buildCopy(DstRegs[0], SrcRegs[0]);
      } else {
        // Extend in the low bits and propagate the sign bit to the high half.
        B.buildSExtInReg(DstRegs[0], SrcRegs[0], Amt);
      }

      B.buildAShr(DstRegs[1], DstRegs[0], B.buildConstant(S32, 31));
    } else {
      // The low bits are unchanged, and extend in the high bits.
      B.buildCopy(DstRegs[0], SrcRegs[0]);
      B.buildSExtInReg(DstRegs[1], DstRegs[0], Amt - 32);
    }

    Register DstReg = MI.getOperand(0).getReg();
    MRI.setRegBank(DstReg, AMDGPU::VGPRRegBank);
    MI.eraseFromParent();
    return;
  }
  case AMDGPU::G_CTPOP:
  case AMDGPU::G_CTLZ_ZERO_UNDEF:
  case AMDGPU::G_CTTZ_ZERO_UNDEF: {
    MachineIRBuilder B(MI);
    MachineFunction &MF = B.getMF();

    const RegisterBank *DstBank =
      OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
    if (DstBank == &AMDGPU::SGPRRegBank)
      break;

    Register SrcReg = MI.getOperand(1).getReg();
    const LLT S32 = LLT::scalar(32);
    LLT Ty = MRI.getType(SrcReg);
    if (Ty == S32)
      break;

    ApplyRegBankMapping ApplyVALU(*this, MRI, &AMDGPU::VGPRRegBank);
    GISelObserverWrapper Observer(&ApplyVALU);
    LegalizerHelper Helper(MF, Observer, B);

    if (Helper.narrowScalar(MI, 1, S32) != LegalizerHelper::Legalized)
      llvm_unreachable("narrowScalar should have succeeded");
    return;
  }
  case AMDGPU::G_SEXT:
  case AMDGPU::G_ZEXT:
  case AMDGPU::G_ANYEXT: {
    Register SrcReg = MI.getOperand(1).getReg();
    LLT SrcTy = MRI.getType(SrcReg);
    const bool Signed = Opc == AMDGPU::G_SEXT;

    assert(empty(OpdMapper.getVRegs(1)));

    MachineIRBuilder B(MI);
    const RegisterBank *SrcBank =
      OpdMapper.getInstrMapping().getOperandMapping(1).BreakDown[0].RegBank;

    Register DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);
    if (DstTy.isScalar() &&
        SrcBank != &AMDGPU::SGPRRegBank &&
        SrcBank != &AMDGPU::VCCRegBank &&
        // FIXME: Should handle any type that round to s64 when irregular
        // breakdowns supported.
        DstTy.getSizeInBits() == 64 &&
        SrcTy.getSizeInBits() <= 32) {
      const LLT S32 = LLT::scalar(32);
      SmallVector<Register, 2> DefRegs(OpdMapper.getVRegs(0));

      // Extend to 32-bit, and then extend the low half.
      if (Signed) {
        // TODO: Should really be buildSExtOrCopy
        B.buildSExtOrTrunc(DefRegs[0], SrcReg);

        // Replicate sign bit from 32-bit extended part.
        auto ShiftAmt = B.buildConstant(S32, 31);
        MRI.setRegBank(ShiftAmt.getReg(0), *SrcBank);
        B.buildAShr(DefRegs[1], DefRegs[0], ShiftAmt);
      } else if (Opc == AMDGPU::G_ZEXT) {
        B.buildZExtOrTrunc(DefRegs[0], SrcReg);
        B.buildConstant(DefRegs[1], 0);
      } else {
        B.buildAnyExtOrTrunc(DefRegs[0], SrcReg);
        B.buildUndef(DefRegs[1]);
      }

      MRI.setRegBank(DstReg, *SrcBank);
      MI.eraseFromParent();
      return;
    }

    if (SrcTy != LLT::scalar(1))
      return;

    // It is not legal to have a legalization artifact with a VCC source. Rather
    // than introducing a copy, insert the selcet we would have to select the
    // copy to.
    if (SrcBank == &AMDGPU::VCCRegBank) {
      SmallVector<Register, 2> DefRegs(OpdMapper.getVRegs(0));

      const RegisterBank *DstBank = &AMDGPU::VGPRRegBank;

      unsigned DstSize = DstTy.getSizeInBits();
      // 64-bit select is SGPR only
      const bool UseSel64 = DstSize > 32 &&
        SrcBank->getID() == AMDGPU::SGPRRegBankID;

      // TODO: Should s16 select be legal?
      LLT SelType = UseSel64 ? LLT::scalar(64) : LLT::scalar(32);
      auto True = B.buildConstant(SelType, Signed ? -1 : 1);
      auto False = B.buildConstant(SelType, 0);

      MRI.setRegBank(True.getReg(0), *DstBank);
      MRI.setRegBank(False.getReg(0), *DstBank);
      MRI.setRegBank(DstReg, *DstBank);

      if (DstSize > 32) {
        B.buildSelect(DefRegs[0], SrcReg, True, False);
        B.buildCopy(DefRegs[1], DefRegs[0]);
      } else if (DstSize < 32) {
        auto Sel = B.buildSelect(SelType, SrcReg, True, False);
        MRI.setRegBank(Sel.getReg(0), *DstBank);
        B.buildTrunc(DstReg, Sel);
      } else {
        B.buildSelect(DstReg, SrcReg, True, False);
      }

      MI.eraseFromParent();
      return;
    }

    break;
  }
  case AMDGPU::G_BUILD_VECTOR:
  case AMDGPU::G_BUILD_VECTOR_TRUNC: {
    Register DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);
    if (DstTy != LLT::vector(2, 16))
      break;

    assert(MI.getNumOperands() == 3 && OpdMapper.getVRegs(0).empty());
    substituteSimpleCopyRegs(OpdMapper, 1);
    substituteSimpleCopyRegs(OpdMapper, 2);

    const RegisterBank *DstBank =
      OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
    if (DstBank == &AMDGPU::SGPRRegBank)
      break; // Can use S_PACK_* instructions.

    MachineIRBuilder B(MI);

    Register Lo = MI.getOperand(1).getReg();
    Register Hi = MI.getOperand(2).getReg();
    const LLT S32 = LLT::scalar(32);

    const RegisterBank *BankLo =
      OpdMapper.getInstrMapping().getOperandMapping(1).BreakDown[0].RegBank;
    const RegisterBank *BankHi =
      OpdMapper.getInstrMapping().getOperandMapping(2).BreakDown[0].RegBank;

    Register ZextLo;
    Register ShiftHi;

    if (Opc == AMDGPU::G_BUILD_VECTOR) {
      ZextLo = B.buildZExt(S32, Lo).getReg(0);
      MRI.setRegBank(ZextLo, *BankLo);

      Register ZextHi = B.buildZExt(S32, Hi).getReg(0);
      MRI.setRegBank(ZextHi, *BankHi);

      auto ShiftAmt = B.buildConstant(S32, 16);
      MRI.setRegBank(ShiftAmt.getReg(0), *BankHi);

      ShiftHi = B.buildShl(S32, ZextHi, ShiftAmt).getReg(0);
      MRI.setRegBank(ShiftHi, *BankHi);
    } else {
      Register MaskLo = B.buildConstant(S32, 0xffff).getReg(0);
      MRI.setRegBank(MaskLo, *BankLo);

      auto ShiftAmt = B.buildConstant(S32, 16);
      MRI.setRegBank(ShiftAmt.getReg(0), *BankHi);

      ShiftHi = B.buildShl(S32, Hi, ShiftAmt).getReg(0);
      MRI.setRegBank(ShiftHi, *BankHi);

      ZextLo = B.buildAnd(S32, Lo, MaskLo).getReg(0);
      MRI.setRegBank(ZextLo, *BankLo);
    }

    auto Or = B.buildOr(S32, ZextLo, ShiftHi);
    MRI.setRegBank(Or.getReg(0), *DstBank);

    B.buildBitcast(DstReg, Or);
    MI.eraseFromParent();
    return;
  }
  case AMDGPU::G_EXTRACT_VECTOR_ELT: {
    SmallVector<Register, 2> DstRegs(OpdMapper.getVRegs(0));

    assert(OpdMapper.getVRegs(1).empty() && OpdMapper.getVRegs(2).empty());

    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = MI.getOperand(1).getReg();

    const LLT S32 = LLT::scalar(32);
    LLT DstTy = MRI.getType(DstReg);
    LLT SrcTy = MRI.getType(SrcReg);

    MachineIRBuilder B(MI);

    const ValueMapping &DstMapping
      = OpdMapper.getInstrMapping().getOperandMapping(0);
    const RegisterBank *DstBank = DstMapping.BreakDown[0].RegBank;
    const RegisterBank *SrcBank =
      OpdMapper.getInstrMapping().getOperandMapping(1).BreakDown[0].RegBank;
    const RegisterBank *IdxBank =
        OpdMapper.getInstrMapping().getOperandMapping(2).BreakDown[0].RegBank;

    Register BaseIdxReg;
    unsigned ConstOffset;
    MachineInstr *OffsetDef;
    std::tie(BaseIdxReg, ConstOffset, OffsetDef) =
        AMDGPU::getBaseWithConstantOffset(MRI, MI.getOperand(2).getReg());

    // See if the index is an add of a constant which will be foldable by moving
    // the base register of the index later if this is going to be executed in a
    // waterfall loop. This is essentially to reassociate the add of a constant
    // with the readfirstlane.
    bool ShouldMoveIndexIntoLoop = IdxBank != &AMDGPU::SGPRRegBank &&
                                   ConstOffset > 0 &&
                                   ConstOffset < SrcTy.getNumElements();

    // Move the base register. We'll re-insert the add later.
    if (ShouldMoveIndexIntoLoop)
      MI.getOperand(2).setReg(BaseIdxReg);

    // If this is a VGPR result only because the index was a VGPR result, the
    // actual indexing will be done on the SGPR source vector, which will
    // produce a scalar result. We need to copy to the VGPR result inside the
    // waterfall loop.
    const bool NeedCopyToVGPR = DstBank == &AMDGPU::VGPRRegBank &&
                                SrcBank == &AMDGPU::SGPRRegBank;
    if (DstRegs.empty()) {
      applyDefaultMapping(OpdMapper);

      executeInWaterfallLoop(MI, MRI, { 2 });

      if (NeedCopyToVGPR) {
        // We don't want a phi for this temporary reg.
        Register TmpReg = MRI.createGenericVirtualRegister(DstTy);
        MRI.setRegBank(TmpReg, AMDGPU::SGPRRegBank);
        MI.getOperand(0).setReg(TmpReg);
        B.setInsertPt(*MI.getParent(), ++MI.getIterator());

        // Use a v_mov_b32 here to make the exec dependency explicit.
        buildVCopy(B, DstReg, TmpReg);
      }

      // Re-insert the constant offset add inside the waterfall loop.
      if (ShouldMoveIndexIntoLoop)
        reinsertVectorIndexAdd(B, MI, 2, ConstOffset);

      return;
    }

    assert(DstTy.getSizeInBits() == 64);

    LLT Vec32 = LLT::vector(2 * SrcTy.getNumElements(), 32);

    auto CastSrc = B.buildBitcast(Vec32, SrcReg);
    auto One = B.buildConstant(S32, 1);

    MachineBasicBlock::iterator MII = MI.getIterator();

    // Split the vector index into 32-bit pieces. Prepare to move all of the
    // new instructions into a waterfall loop if necessary.
    //
    // Don't put the bitcast or constant in the loop.
    MachineInstrSpan Span(MII, &B.getMBB());

    // Compute 32-bit element indices, (2 * OrigIdx, 2 * OrigIdx + 1).
    auto IdxLo = B.buildShl(S32, BaseIdxReg, One);
    auto IdxHi = B.buildAdd(S32, IdxLo, One);

    auto Extract0 = B.buildExtractVectorElement(DstRegs[0], CastSrc, IdxLo);
    auto Extract1 = B.buildExtractVectorElement(DstRegs[1], CastSrc, IdxHi);

    MRI.setRegBank(DstReg, *DstBank);
    MRI.setRegBank(CastSrc.getReg(0), *SrcBank);
    MRI.setRegBank(One.getReg(0), AMDGPU::SGPRRegBank);
    MRI.setRegBank(IdxLo.getReg(0), AMDGPU::SGPRRegBank);
    MRI.setRegBank(IdxHi.getReg(0), AMDGPU::SGPRRegBank);

    SmallSet<Register, 4> OpsToWaterfall;
    if (!collectWaterfallOperands(OpsToWaterfall, MI, MRI, { 2 })) {
      MI.eraseFromParent();
      return;
    }

    // Remove the original instruction to avoid potentially confusing the
    // waterfall loop logic.
    B.setInstr(*Span.begin());
    MI.eraseFromParent();
    executeInWaterfallLoop(B, make_range(Span.begin(), Span.end()),
                           OpsToWaterfall, MRI);

    if (NeedCopyToVGPR) {
      MachineBasicBlock *LoopBB = Extract1->getParent();
      Register TmpReg0 = MRI.createGenericVirtualRegister(S32);
      Register TmpReg1 = MRI.createGenericVirtualRegister(S32);
      MRI.setRegBank(TmpReg0, AMDGPU::SGPRRegBank);
      MRI.setRegBank(TmpReg1, AMDGPU::SGPRRegBank);

      Extract0->getOperand(0).setReg(TmpReg0);
      Extract1->getOperand(0).setReg(TmpReg1);

      B.setInsertPt(*LoopBB, ++Extract1->getIterator());

      buildVCopy(B, DstRegs[0], TmpReg0);
      buildVCopy(B, DstRegs[1], TmpReg1);
    }

    if (ShouldMoveIndexIntoLoop)
      reinsertVectorIndexAdd(B, *IdxLo, 1, ConstOffset);

    return;
  }
  case AMDGPU::G_INSERT_VECTOR_ELT: {
    SmallVector<Register, 2> InsRegs(OpdMapper.getVRegs(2));

    Register DstReg = MI.getOperand(0).getReg();
    LLT VecTy = MRI.getType(DstReg);

    assert(OpdMapper.getVRegs(0).empty());
    assert(OpdMapper.getVRegs(3).empty());

    const RegisterBank *IdxBank =
      OpdMapper.getInstrMapping().getOperandMapping(3).BreakDown[0].RegBank;

    if (substituteSimpleCopyRegs(OpdMapper, 1))
      MRI.setType(MI.getOperand(1).getReg(), VecTy);

    Register SrcReg = MI.getOperand(1).getReg();
    Register InsReg = MI.getOperand(2).getReg();
    LLT InsTy = MRI.getType(InsReg);
    (void)InsTy;

    Register BaseIdxReg;
    unsigned ConstOffset;
    MachineInstr *OffsetDef;
    std::tie(BaseIdxReg, ConstOffset, OffsetDef) =
      AMDGPU::getBaseWithConstantOffset(MRI, MI.getOperand(3).getReg());

    // See if the index is an add of a constant which will be foldable by moving
    // the base register of the index later if this is going to be executed in a
    // waterfall loop. This is essentially to reassociate the add of a constant
    // with the readfirstlane.
    bool ShouldMoveIndexIntoLoop = IdxBank != &AMDGPU::SGPRRegBank &&
      ConstOffset > 0 &&
      ConstOffset < VecTy.getNumElements();

    // Move the base register. We'll re-insert the add later.
    if (ShouldMoveIndexIntoLoop)
      MI.getOperand(3).setReg(BaseIdxReg);


    if (InsRegs.empty()) {
      executeInWaterfallLoop(MI, MRI, { 3 });

      // Re-insert the constant offset add inside the waterfall loop.
      if (ShouldMoveIndexIntoLoop) {
        MachineIRBuilder B(MI);
        reinsertVectorIndexAdd(B, MI, 3, ConstOffset);
      }

      return;
    }


    assert(InsTy.getSizeInBits() == 64);

    const LLT S32 = LLT::scalar(32);
    LLT Vec32 = LLT::vector(2 * VecTy.getNumElements(), 32);

    MachineIRBuilder B(MI);
    auto CastSrc = B.buildBitcast(Vec32, SrcReg);
    auto One = B.buildConstant(S32, 1);

    // Split the vector index into 32-bit pieces. Prepare to move all of the
    // new instructions into a waterfall loop if necessary.
    //
    // Don't put the bitcast or constant in the loop.
    MachineInstrSpan Span(MachineBasicBlock::iterator(&MI), &B.getMBB());

    // Compute 32-bit element indices, (2 * OrigIdx, 2 * OrigIdx + 1).
    auto IdxLo = B.buildShl(S32, BaseIdxReg, One);
    auto IdxHi = B.buildAdd(S32, IdxLo, One);

    auto InsLo = B.buildInsertVectorElement(Vec32, CastSrc, InsRegs[0], IdxLo);
    auto InsHi = B.buildInsertVectorElement(Vec32, InsLo, InsRegs[1], IdxHi);

    const RegisterBank *DstBank =
      OpdMapper.getInstrMapping().getOperandMapping(0).BreakDown[0].RegBank;
    const RegisterBank *SrcBank =
      OpdMapper.getInstrMapping().getOperandMapping(1).BreakDown[0].RegBank;
    const RegisterBank *InsSrcBank =
      OpdMapper.getInstrMapping().getOperandMapping(2).BreakDown[0].RegBank;

    MRI.setRegBank(InsReg, *InsSrcBank);
    MRI.setRegBank(CastSrc.getReg(0), *SrcBank);
    MRI.setRegBank(InsLo.getReg(0), *DstBank);
    MRI.setRegBank(InsHi.getReg(0), *DstBank);
    MRI.setRegBank(One.getReg(0), AMDGPU::SGPRRegBank);
    MRI.setRegBank(IdxLo.getReg(0), AMDGPU::SGPRRegBank);
    MRI.setRegBank(IdxHi.getReg(0), AMDGPU::SGPRRegBank);


    SmallSet<Register, 4> OpsToWaterfall;
    if (!collectWaterfallOperands(OpsToWaterfall, MI, MRI, { 3 })) {
      B.setInsertPt(B.getMBB(), MI);
      B.buildBitcast(DstReg, InsHi);
      MI.eraseFromParent();
      return;
    }

    B.setInstr(*Span.begin());
    MI.eraseFromParent();

    // Figure out the point after the waterfall loop before mangling the control
    // flow.
    executeInWaterfallLoop(B, make_range(Span.begin(), Span.end()),
                           OpsToWaterfall, MRI);

    // The insertion point is now right after the original instruction.
    //
    // Keep the bitcast to the original vector type out of the loop. Doing this
    // saved an extra phi we don't need inside the loop.
    B.buildBitcast(DstReg, InsHi);

    // Re-insert the constant offset add inside the waterfall loop.
    if (ShouldMoveIndexIntoLoop)
      reinsertVectorIndexAdd(B, *IdxLo, 1, ConstOffset);

    return;
  }
  case AMDGPU::G_AMDGPU_BUFFER_LOAD:
  case AMDGPU::G_AMDGPU_BUFFER_LOAD_USHORT:
  case AMDGPU::G_AMDGPU_BUFFER_LOAD_SSHORT:
  case AMDGPU::G_AMDGPU_BUFFER_LOAD_UBYTE:
  case AMDGPU::G_AMDGPU_BUFFER_LOAD_SBYTE:
  case AMDGPU::G_AMDGPU_BUFFER_LOAD_FORMAT:
  case AMDGPU::G_AMDGPU_BUFFER_LOAD_FORMAT_D16:
  case AMDGPU::G_AMDGPU_TBUFFER_LOAD_FORMAT:
  case AMDGPU::G_AMDGPU_TBUFFER_LOAD_FORMAT_D16:
  case AMDGPU::G_AMDGPU_BUFFER_STORE:
  case AMDGPU::G_AMDGPU_BUFFER_STORE_BYTE:
  case AMDGPU::G_AMDGPU_BUFFER_STORE_SHORT:
  case AMDGPU::G_AMDGPU_BUFFER_STORE_FORMAT:
  case AMDGPU::G_AMDGPU_BUFFER_STORE_FORMAT_D16:
  case AMDGPU::G_AMDGPU_TBUFFER_STORE_FORMAT:
  case AMDGPU::G_AMDGPU_TBUFFER_STORE_FORMAT_D16: {
    applyDefaultMapping(OpdMapper);
    executeInWaterfallLoop(MI, MRI, {1, 4});
    return;
  }
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_SWAP:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_ADD:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_SUB:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_SMIN:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_UMIN:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_SMAX:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_UMAX:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_AND:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_OR:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_XOR:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_INC:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_DEC: {
    applyDefaultMapping(OpdMapper);
    executeInWaterfallLoop(MI, MRI, {2, 5});
    return;
  }
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_CMPSWAP: {
    applyDefaultMapping(OpdMapper);
    executeInWaterfallLoop(MI, MRI, {3, 6});
    return;
  }
  case AMDGPU::G_AMDGPU_S_BUFFER_LOAD: {
    applyMappingSBufferLoad(OpdMapper);
    return;
  }
  case AMDGPU::G_INTRINSIC: {
    switch (MI.getIntrinsicID()) {
    case Intrinsic::amdgcn_readlane: {
      substituteSimpleCopyRegs(OpdMapper, 2);

      assert(OpdMapper.getVRegs(0).empty());
      assert(OpdMapper.getVRegs(3).empty());

      // Make sure the index is an SGPR. It doesn't make sense to run this in a
      // waterfall loop, so assume it's a uniform value.
      constrainOpWithReadfirstlane(MI, MRI, 3); // Index
      return;
    }
    case Intrinsic::amdgcn_writelane: {
      assert(OpdMapper.getVRegs(0).empty());
      assert(OpdMapper.getVRegs(2).empty());
      assert(OpdMapper.getVRegs(3).empty());

      substituteSimpleCopyRegs(OpdMapper, 4); // VGPR input val
      constrainOpWithReadfirstlane(MI, MRI, 2); // Source value
      constrainOpWithReadfirstlane(MI, MRI, 3); // Index
      return;
    }
    case Intrinsic::amdgcn_interp_p1:
    case Intrinsic::amdgcn_interp_p2:
    case Intrinsic::amdgcn_interp_mov:
    case Intrinsic::amdgcn_interp_p1_f16:
    case Intrinsic::amdgcn_interp_p2_f16: {
      applyDefaultMapping(OpdMapper);

      // Readlane for m0 value, which is always the last operand.
      // FIXME: Should this be a waterfall loop instead?
      constrainOpWithReadfirstlane(MI, MRI, MI.getNumOperands() - 1); // Index
      return;
    }
    case Intrinsic::amdgcn_permlane16:
    case Intrinsic::amdgcn_permlanex16: {
      // Doing a waterfall loop over these wouldn't make any sense.
      substituteSimpleCopyRegs(OpdMapper, 2);
      substituteSimpleCopyRegs(OpdMapper, 3);
      constrainOpWithReadfirstlane(MI, MRI, 4);
      constrainOpWithReadfirstlane(MI, MRI, 5);
      return;
    }
    case Intrinsic::amdgcn_sbfe:
      applyMappingBFEIntrinsic(OpdMapper, true);
      return;
    case Intrinsic::amdgcn_ubfe:
      applyMappingBFEIntrinsic(OpdMapper, false);
      return;
    }
    break;
  }
  case AMDGPU::G_AMDGPU_INTRIN_IMAGE_LOAD:
  case AMDGPU::G_AMDGPU_INTRIN_IMAGE_STORE: {
    const AMDGPU::RsrcIntrinsic *RSrcIntrin
      = AMDGPU::lookupRsrcIntrinsic(MI.getIntrinsicID());
    assert(RSrcIntrin && RSrcIntrin->IsImage);
    // Non-images can have complications from operands that allow both SGPR
    // and VGPR. For now it's too complicated to figure out the final opcode
    // to derive the register bank from the MCInstrDesc.
    applyMappingImage(MI, OpdMapper, MRI, RSrcIntrin->RsrcArg);
    return;
  }
  case AMDGPU::G_INTRINSIC_W_SIDE_EFFECTS: {
    auto IntrID = MI.getIntrinsicID();
    switch (IntrID) {
    case Intrinsic::amdgcn_ds_ordered_add:
    case Intrinsic::amdgcn_ds_ordered_swap: {
      // This is only allowed to execute with 1 lane, so readfirstlane is safe.
      assert(OpdMapper.getVRegs(0).empty());
      substituteSimpleCopyRegs(OpdMapper, 3);
      constrainOpWithReadfirstlane(MI, MRI, 2); // M0
      return;
    }
    case Intrinsic::amdgcn_ds_gws_init:
    case Intrinsic::amdgcn_ds_gws_barrier:
    case Intrinsic::amdgcn_ds_gws_sema_br: {
      // Only the first lane is executes, so readfirstlane is safe.
      substituteSimpleCopyRegs(OpdMapper, 1);
      constrainOpWithReadfirstlane(MI, MRI, 2); // M0
      return;
    }
    case Intrinsic::amdgcn_ds_gws_sema_v:
    case Intrinsic::amdgcn_ds_gws_sema_p:
    case Intrinsic::amdgcn_ds_gws_sema_release_all: {
      // Only the first lane is executes, so readfirstlane is safe.
      constrainOpWithReadfirstlane(MI, MRI, 1); // M0
      return;
    }
    case Intrinsic::amdgcn_ds_append:
    case Intrinsic::amdgcn_ds_consume: {
      constrainOpWithReadfirstlane(MI, MRI, 2); // M0
      return;
    }
    case Intrinsic::amdgcn_s_sendmsg:
    case Intrinsic::amdgcn_s_sendmsghalt: {
      // FIXME: Should this use a waterfall loop?
      constrainOpWithReadfirstlane(MI, MRI, 2); // M0
      return;
    }
    default: {
      if (const AMDGPU::RsrcIntrinsic *RSrcIntrin =
              AMDGPU::lookupRsrcIntrinsic(IntrID)) {
        // Non-images can have complications from operands that allow both SGPR
        // and VGPR. For now it's too complicated to figure out the final opcode
        // to derive the register bank from the MCInstrDesc.
        if (RSrcIntrin->IsImage) {
          applyMappingImage(MI, OpdMapper, MRI, RSrcIntrin->RsrcArg);
          return;
        }
      }

      break;
    }
    }
    break;
  }
  case AMDGPU::G_LOAD:
  case AMDGPU::G_ZEXTLOAD:
  case AMDGPU::G_SEXTLOAD: {
    if (applyMappingWideLoad(MI, OpdMapper, MRI))
      return;
    break;
  }
  default:
    break;
  }

  return applyDefaultMapping(OpdMapper);
}

bool AMDGPURegisterBankInfo::isSALUMapping(const MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  for (unsigned i = 0, e = MI.getNumOperands();i != e; ++i) {
    if (!MI.getOperand(i).isReg())
      continue;
    Register Reg = MI.getOperand(i).getReg();
    if (const RegisterBank *Bank = getRegBank(Reg, MRI, *TRI)) {
      if (Bank->getID() != AMDGPU::SGPRRegBankID)
        return false;
    }
  }
  return true;
}

const RegisterBankInfo::InstructionMapping &
AMDGPURegisterBankInfo::getDefaultMappingSOP(const MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<const ValueMapping*, 8> OpdsMapping(MI.getNumOperands());

  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    const MachineOperand &SrcOp = MI.getOperand(i);
    if (!SrcOp.isReg())
      continue;

    unsigned Size = getSizeInBits(SrcOp.getReg(), MRI, *TRI);
    OpdsMapping[i] = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size);
  }
  return getInstructionMapping(1, 1, getOperandsMapping(OpdsMapping),
                               MI.getNumOperands());
}

const RegisterBankInfo::InstructionMapping &
AMDGPURegisterBankInfo::getDefaultMappingVOP(const MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<const ValueMapping*, 8> OpdsMapping(MI.getNumOperands());

  // Even though we technically could use SGPRs, this would require knowledge of
  // the constant bus restriction. Force all sources to VGPR (except for VCC).
  //
  // TODO: Unary ops are trivially OK, so accept SGPRs?
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    const MachineOperand &Src = MI.getOperand(i);
    if (!Src.isReg())
      continue;

    unsigned Size = getSizeInBits(Src.getReg(), MRI, *TRI);
    unsigned BankID = Size == 1 ? AMDGPU::VCCRegBankID : AMDGPU::VGPRRegBankID;
    OpdsMapping[i] = AMDGPU::getValueMapping(BankID, Size);
  }

  return getInstructionMapping(1, 1, getOperandsMapping(OpdsMapping),
                               MI.getNumOperands());
}

const RegisterBankInfo::InstructionMapping &
AMDGPURegisterBankInfo::getDefaultMappingAllVGPR(const MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<const ValueMapping*, 8> OpdsMapping(MI.getNumOperands());

  for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
    const MachineOperand &Op = MI.getOperand(I);
    if (!Op.isReg())
      continue;

    unsigned Size = getSizeInBits(Op.getReg(), MRI, *TRI);
    OpdsMapping[I] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Size);
  }

  return getInstructionMapping(1, 1, getOperandsMapping(OpdsMapping),
                               MI.getNumOperands());
}

const RegisterBankInfo::InstructionMapping &
AMDGPURegisterBankInfo::getImageMapping(const MachineRegisterInfo &MRI,
                                        const MachineInstr &MI,
                                        int RsrcIdx) const {
  // The reported argument index is relative to the IR intrinsic call arguments,
  // so we need to shift by the number of defs and the intrinsic ID.
  RsrcIdx += MI.getNumExplicitDefs() + 1;

  const int NumOps = MI.getNumOperands();
  SmallVector<const ValueMapping *, 8> OpdsMapping(NumOps);

  // TODO: Should packed/unpacked D16 difference be reported here as part of
  // the value mapping?
  for (int I = 0; I != NumOps; ++I) {
    if (!MI.getOperand(I).isReg())
      continue;

    Register OpReg = MI.getOperand(I).getReg();
    unsigned Size = getSizeInBits(OpReg, MRI, *TRI);

    // FIXME: Probably need a new intrinsic register bank searchable table to
    // handle arbitrary intrinsics easily.
    //
    // If this has a sampler, it immediately follows rsrc.
    const bool MustBeSGPR = I == RsrcIdx || I == RsrcIdx + 1;

    if (MustBeSGPR) {
      // If this must be an SGPR, so we must report whatever it is as legal.
      unsigned NewBank = getRegBankID(OpReg, MRI, *TRI, AMDGPU::SGPRRegBankID);
      OpdsMapping[I] = AMDGPU::getValueMapping(NewBank, Size);
    } else {
      // Some operands must be VGPR, and these are easy to copy to.
      OpdsMapping[I] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Size);
    }
  }

  return getInstructionMapping(1, 1, getOperandsMapping(OpdsMapping), NumOps);
}

/// Return the mapping for a pointer arugment.
const RegisterBankInfo::ValueMapping *
AMDGPURegisterBankInfo::getValueMappingForPtr(const MachineRegisterInfo &MRI,
                                              Register PtrReg) const {
  LLT PtrTy = MRI.getType(PtrReg);
  unsigned Size = PtrTy.getSizeInBits();
  if (Subtarget.useFlatForGlobal() ||
      !SITargetLowering::isFlatGlobalAddrSpace(PtrTy.getAddressSpace()))
    return AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Size);

  // If we're using MUBUF instructions for global memory, an SGPR base register
  // is possible. Otherwise this needs to be a VGPR.
  const RegisterBank *PtrBank = getRegBank(PtrReg, MRI, *TRI);
  return AMDGPU::getValueMapping(PtrBank->getID(), Size);
}

const RegisterBankInfo::InstructionMapping &
AMDGPURegisterBankInfo::getInstrMappingForLoad(const MachineInstr &MI) const {

  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<const ValueMapping*, 2> OpdsMapping(2);
  unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
  LLT LoadTy = MRI.getType(MI.getOperand(0).getReg());
  Register PtrReg = MI.getOperand(1).getReg();
  LLT PtrTy = MRI.getType(PtrReg);
  unsigned AS = PtrTy.getAddressSpace();
  unsigned PtrSize = PtrTy.getSizeInBits();

  const ValueMapping *ValMapping;
  const ValueMapping *PtrMapping;

  const RegisterBank *PtrBank = getRegBank(PtrReg, MRI, *TRI);

  if (PtrBank == &AMDGPU::SGPRRegBank &&
      SITargetLowering::isFlatGlobalAddrSpace(AS)) {
    if (isScalarLoadLegal(MI)) {
      // We have a uniform instruction so we want to use an SMRD load
      ValMapping = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size);
      PtrMapping = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, PtrSize);
    } else {
      ValMapping = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Size);

      // If we're using MUBUF instructions for global memory, an SGPR base
      // register is possible. Otherwise this needs to be a VGPR.
      unsigned PtrBankID = Subtarget.useFlatForGlobal() ?
        AMDGPU::VGPRRegBankID : AMDGPU::SGPRRegBankID;

      PtrMapping = AMDGPU::getValueMapping(PtrBankID, PtrSize);
      ValMapping = AMDGPU::getValueMappingLoadSGPROnly(AMDGPU::VGPRRegBankID,
                                                       LoadTy);
    }
  } else {
    ValMapping = AMDGPU::getValueMappingLoadSGPROnly(AMDGPU::VGPRRegBankID, LoadTy);
    PtrMapping = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, PtrSize);
  }

  OpdsMapping[0] = ValMapping;
  OpdsMapping[1] = PtrMapping;
  const RegisterBankInfo::InstructionMapping &Mapping = getInstructionMapping(
      1, 1, getOperandsMapping(OpdsMapping), MI.getNumOperands());
  return Mapping;

  // FIXME: Do we want to add a mapping for FLAT load, or should we just
  // handle that during instruction selection?
}

unsigned
AMDGPURegisterBankInfo::getRegBankID(Register Reg,
                                     const MachineRegisterInfo &MRI,
                                     const TargetRegisterInfo &TRI,
                                     unsigned Default) const {
  const RegisterBank *Bank = getRegBank(Reg, MRI, TRI);
  return Bank ? Bank->getID() : Default;
}


static unsigned regBankUnion(unsigned RB0, unsigned RB1) {
  return (RB0 == AMDGPU::SGPRRegBankID && RB1 == AMDGPU::SGPRRegBankID) ?
    AMDGPU::SGPRRegBankID : AMDGPU::VGPRRegBankID;
}

static int regBankBoolUnion(int RB0, int RB1) {
  if (RB0 == -1)
    return RB1;
  if (RB1 == -1)
    return RB0;

  // vcc, vcc -> vcc
  // vcc, sgpr -> vcc
  // vcc, vgpr -> vcc
  if (RB0 == AMDGPU::VCCRegBankID || RB1 == AMDGPU::VCCRegBankID)
    return AMDGPU::VCCRegBankID;

  // vcc, vgpr -> vgpr
  return regBankUnion(RB0, RB1);
}

const RegisterBankInfo::ValueMapping *
AMDGPURegisterBankInfo::getSGPROpMapping(Register Reg,
                                         const MachineRegisterInfo &MRI,
                                         const TargetRegisterInfo &TRI) const {
  // Lie and claim anything is legal, even though this needs to be an SGPR
  // applyMapping will have to deal with it as a waterfall loop.
  unsigned Bank = getRegBankID(Reg, MRI, TRI, AMDGPU::SGPRRegBankID);
  unsigned Size = getSizeInBits(Reg, MRI, TRI);
  return AMDGPU::getValueMapping(Bank, Size);
}

const RegisterBankInfo::ValueMapping *
AMDGPURegisterBankInfo::getVGPROpMapping(Register Reg,
                                         const MachineRegisterInfo &MRI,
                                         const TargetRegisterInfo &TRI) const {
  unsigned Size = getSizeInBits(Reg, MRI, TRI);
  return AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Size);
}

const RegisterBankInfo::ValueMapping *
AMDGPURegisterBankInfo::getAGPROpMapping(Register Reg,
                                         const MachineRegisterInfo &MRI,
                                         const TargetRegisterInfo &TRI) const {
  unsigned Size = getSizeInBits(Reg, MRI, TRI);
  return AMDGPU::getValueMapping(AMDGPU::AGPRRegBankID, Size);
}

///
/// This function must return a legal mapping, because
/// AMDGPURegisterBankInfo::getInstrAlternativeMappings() is not called
/// in RegBankSelect::Mode::Fast.  Any mapping that would cause a
/// VGPR to SGPR generated is illegal.
///
// Operands that must be SGPRs must accept potentially divergent VGPRs as
// legal. These will be dealt with in applyMappingImpl.
//
const RegisterBankInfo::InstructionMapping &
AMDGPURegisterBankInfo::getInstrMapping(const MachineInstr &MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  if (MI.isCopy()) {
    // The default logic bothers to analyze impossible alternative mappings. We
    // want the most straightforward mapping, so just directly handle this.
    const RegisterBank *DstBank = getRegBank(MI.getOperand(0).getReg(), MRI,
                                             *TRI);
    const RegisterBank *SrcBank = getRegBank(MI.getOperand(1).getReg(), MRI,
                                             *TRI);
    assert(SrcBank && "src bank should have been assigned already");
    if (!DstBank)
      DstBank = SrcBank;

    unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
    if (cannotCopy(*DstBank, *SrcBank, Size))
      return getInvalidInstructionMapping();

    const ValueMapping &ValMap = getValueMapping(0, Size, *DstBank);
    return getInstructionMapping(
        1, /*Cost*/ 1,
        /*OperandsMapping*/ getOperandsMapping({&ValMap}), 1);
  }

  if (MI.isRegSequence()) {
    // If any input is a VGPR, the result must be a VGPR. The default handling
    // assumes any copy between banks is legal.
    unsigned BankID = AMDGPU::SGPRRegBankID;

    for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2) {
      auto OpBank = getRegBankID(MI.getOperand(I).getReg(), MRI, *TRI);
      // It doesn't make sense to use vcc or scc banks here, so just ignore
      // them.
      if (OpBank != AMDGPU::SGPRRegBankID) {
        BankID = AMDGPU::VGPRRegBankID;
        break;
      }
    }
    unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);

    const ValueMapping &ValMap = getValueMapping(0, Size, getRegBank(BankID));
    return getInstructionMapping(
        1, /*Cost*/ 1,
        /*OperandsMapping*/ getOperandsMapping({&ValMap}), 1);
  }

  // The default handling is broken and doesn't handle illegal SGPR->VGPR copies
  // properly.
  //
  // TODO: There are additional exec masking dependencies to analyze.
  if (MI.getOpcode() == TargetOpcode::G_PHI) {
    // TODO: Generate proper invalid bank enum.
    int ResultBank = -1;
    Register DstReg = MI.getOperand(0).getReg();

    // Sometimes the result may have already been assigned a bank.
    if (const RegisterBank *DstBank = getRegBank(DstReg, MRI, *TRI))
      ResultBank = DstBank->getID();

    for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2) {
      Register Reg = MI.getOperand(I).getReg();
      const RegisterBank *Bank = getRegBank(Reg, MRI, *TRI);

      // FIXME: Assuming VGPR for any undetermined inputs.
      if (!Bank || Bank->getID() == AMDGPU::VGPRRegBankID) {
        ResultBank = AMDGPU::VGPRRegBankID;
        break;
      }

      // FIXME: Need to promote SGPR case to s32
      unsigned OpBank = Bank->getID();
      ResultBank = regBankBoolUnion(ResultBank, OpBank);
    }

    assert(ResultBank != -1);

    unsigned Size = MRI.getType(DstReg).getSizeInBits();

    const ValueMapping &ValMap =
        getValueMapping(0, Size, getRegBank(ResultBank));
    return getInstructionMapping(
        1, /*Cost*/ 1,
        /*OperandsMapping*/ getOperandsMapping({&ValMap}), 1);
  }

  const RegisterBankInfo::InstructionMapping &Mapping = getInstrMappingImpl(MI);
  if (Mapping.isValid())
    return Mapping;

  SmallVector<const ValueMapping*, 8> OpdsMapping(MI.getNumOperands());

  switch (MI.getOpcode()) {
  default:
    return getInvalidInstructionMapping();

  case AMDGPU::G_AND:
  case AMDGPU::G_OR:
  case AMDGPU::G_XOR: {
    unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    if (Size == 1) {
      const RegisterBank *DstBank
        = getRegBank(MI.getOperand(0).getReg(), MRI, *TRI);

      unsigned TargetBankID = -1;
      unsigned BankLHS = -1;
      unsigned BankRHS = -1;
      if (DstBank) {
        TargetBankID = DstBank->getID();
        if (DstBank == &AMDGPU::VCCRegBank) {
          TargetBankID = AMDGPU::VCCRegBankID;
          BankLHS = AMDGPU::VCCRegBankID;
          BankRHS = AMDGPU::VCCRegBankID;
        } else {
          BankLHS = getRegBankID(MI.getOperand(1).getReg(), MRI, *TRI,
                                 AMDGPU::SGPRRegBankID);
          BankRHS = getRegBankID(MI.getOperand(2).getReg(), MRI, *TRI,
                                 AMDGPU::SGPRRegBankID);
        }
      } else {
        BankLHS = getRegBankID(MI.getOperand(1).getReg(), MRI, *TRI,
                               AMDGPU::VCCRegBankID);
        BankRHS = getRegBankID(MI.getOperand(2).getReg(), MRI, *TRI,
                               AMDGPU::VCCRegBankID);

        // Both inputs should be true booleans to produce a boolean result.
        if (BankLHS == AMDGPU::VGPRRegBankID || BankRHS == AMDGPU::VGPRRegBankID) {
          TargetBankID = AMDGPU::VGPRRegBankID;
        } else if (BankLHS == AMDGPU::VCCRegBankID || BankRHS == AMDGPU::VCCRegBankID) {
          TargetBankID = AMDGPU::VCCRegBankID;
          BankLHS = AMDGPU::VCCRegBankID;
          BankRHS = AMDGPU::VCCRegBankID;
        } else if (BankLHS == AMDGPU::SGPRRegBankID && BankRHS == AMDGPU::SGPRRegBankID) {
          TargetBankID = AMDGPU::SGPRRegBankID;
        }
      }

      OpdsMapping[0] = AMDGPU::getValueMapping(TargetBankID, Size);
      OpdsMapping[1] = AMDGPU::getValueMapping(BankLHS, Size);
      OpdsMapping[2] = AMDGPU::getValueMapping(BankRHS, Size);
      break;
    }

    if (Size == 64) {

      if (isSALUMapping(MI)) {
        OpdsMapping[0] = getValueMappingSGPR64Only(AMDGPU::SGPRRegBankID, Size);
        OpdsMapping[1] = OpdsMapping[2] = OpdsMapping[0];
      } else {
        OpdsMapping[0] = getValueMappingSGPR64Only(AMDGPU::VGPRRegBankID, Size);
        unsigned Bank1 = getRegBankID(MI.getOperand(1).getReg(), MRI, *TRI/*, DefaultBankID*/);
        OpdsMapping[1] = AMDGPU::getValueMapping(Bank1, Size);

        unsigned Bank2 = getRegBankID(MI.getOperand(2).getReg(), MRI, *TRI/*, DefaultBankID*/);
        OpdsMapping[2] = AMDGPU::getValueMapping(Bank2, Size);
      }

      break;
    }

    LLVM_FALLTHROUGH;
  }
  case AMDGPU::G_PTR_ADD:
  case AMDGPU::G_ADD:
  case AMDGPU::G_SUB:
  case AMDGPU::G_MUL:
  case AMDGPU::G_SHL:
  case AMDGPU::G_LSHR:
  case AMDGPU::G_ASHR:
  case AMDGPU::G_UADDO:
  case AMDGPU::G_USUBO:
  case AMDGPU::G_UADDE:
  case AMDGPU::G_SADDE:
  case AMDGPU::G_USUBE:
  case AMDGPU::G_SSUBE:
  case AMDGPU::G_SMIN:
  case AMDGPU::G_SMAX:
  case AMDGPU::G_UMIN:
  case AMDGPU::G_UMAX:
  case AMDGPU::G_SHUFFLE_VECTOR:
    if (isSALUMapping(MI))
      return getDefaultMappingSOP(MI);
    LLVM_FALLTHROUGH;

  case AMDGPU::G_FADD:
  case AMDGPU::G_FSUB:
  case AMDGPU::G_FPTOSI:
  case AMDGPU::G_FPTOUI:
  case AMDGPU::G_FMUL:
  case AMDGPU::G_FMA:
  case AMDGPU::G_FMAD:
  case AMDGPU::G_FSQRT:
  case AMDGPU::G_FFLOOR:
  case AMDGPU::G_FCEIL:
  case AMDGPU::G_FRINT:
  case AMDGPU::G_SITOFP:
  case AMDGPU::G_UITOFP:
  case AMDGPU::G_FPTRUNC:
  case AMDGPU::G_FPEXT:
  case AMDGPU::G_FEXP2:
  case AMDGPU::G_FLOG2:
  case AMDGPU::G_FMINNUM:
  case AMDGPU::G_FMAXNUM:
  case AMDGPU::G_FMINNUM_IEEE:
  case AMDGPU::G_FMAXNUM_IEEE:
  case AMDGPU::G_FCANONICALIZE:
  case AMDGPU::G_INTRINSIC_TRUNC:
  case AMDGPU::G_BSWAP: // TODO: Somehow expand for scalar?
  case AMDGPU::G_AMDGPU_FFBH_U32:
  case AMDGPU::G_AMDGPU_FMIN_LEGACY:
  case AMDGPU::G_AMDGPU_FMAX_LEGACY:
  case AMDGPU::G_AMDGPU_RCP_IFLAG:
    return getDefaultMappingVOP(MI);
  case AMDGPU::G_UMULH:
  case AMDGPU::G_SMULH: {
    if (Subtarget.hasScalarMulHiInsts() && isSALUMapping(MI))
      return getDefaultMappingSOP(MI);
    return getDefaultMappingVOP(MI);
  }
  case AMDGPU::G_IMPLICIT_DEF: {
    unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size);
    break;
  }
  case AMDGPU::G_FCONSTANT:
  case AMDGPU::G_CONSTANT:
  case AMDGPU::G_GLOBAL_VALUE:
  case AMDGPU::G_BLOCK_ADDR:
  case AMDGPU::G_READCYCLECOUNTER: {
    unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size);
    break;
  }
  case AMDGPU::G_FRAME_INDEX: {
    // TODO: This should be the same as other constants, but eliminateFrameIndex
    // currently assumes VALU uses.
    unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Size);
    break;
  }
  case AMDGPU::G_INSERT: {
    unsigned BankID = isSALUMapping(MI) ? AMDGPU::SGPRRegBankID :
                                          AMDGPU::VGPRRegBankID;
    unsigned DstSize = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
    unsigned SrcSize = getSizeInBits(MI.getOperand(1).getReg(), MRI, *TRI);
    unsigned EltSize = getSizeInBits(MI.getOperand(2).getReg(), MRI, *TRI);
    OpdsMapping[0] = AMDGPU::getValueMapping(BankID, DstSize);
    OpdsMapping[1] = AMDGPU::getValueMapping(BankID, SrcSize);
    OpdsMapping[2] = AMDGPU::getValueMapping(BankID, EltSize);
    OpdsMapping[3] = nullptr;
    break;
  }
  case AMDGPU::G_EXTRACT: {
    unsigned BankID = getRegBankID(MI.getOperand(1).getReg(), MRI, *TRI);
    unsigned DstSize = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
    unsigned SrcSize = getSizeInBits(MI.getOperand(1).getReg(), MRI, *TRI);
    OpdsMapping[0] = AMDGPU::getValueMapping(BankID, DstSize);
    OpdsMapping[1] = AMDGPU::getValueMapping(BankID, SrcSize);
    OpdsMapping[2] = nullptr;
    break;
  }
  case AMDGPU::G_BUILD_VECTOR:
  case AMDGPU::G_BUILD_VECTOR_TRUNC: {
    LLT DstTy = MRI.getType(MI.getOperand(0).getReg());
    if (DstTy == LLT::vector(2, 16)) {
      unsigned DstSize = DstTy.getSizeInBits();
      unsigned SrcSize = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
      unsigned Src0BankID = getRegBankID(MI.getOperand(1).getReg(), MRI, *TRI);
      unsigned Src1BankID = getRegBankID(MI.getOperand(2).getReg(), MRI, *TRI);
      unsigned DstBankID = regBankUnion(Src0BankID, Src1BankID);

      OpdsMapping[0] = AMDGPU::getValueMapping(DstBankID, DstSize);
      OpdsMapping[1] = AMDGPU::getValueMapping(Src0BankID, SrcSize);
      OpdsMapping[2] = AMDGPU::getValueMapping(Src1BankID, SrcSize);
      break;
    }

    LLVM_FALLTHROUGH;
  }
  case AMDGPU::G_MERGE_VALUES:
  case AMDGPU::G_CONCAT_VECTORS: {
    unsigned Bank = isSALUMapping(MI) ?
      AMDGPU::SGPRRegBankID : AMDGPU::VGPRRegBankID;
    unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    unsigned SrcSize = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();

    OpdsMapping[0] = AMDGPU::getValueMapping(Bank, DstSize);
    // Op1 and Dst should use the same register bank.
    for (unsigned i = 1, e = MI.getNumOperands(); i != e; ++i)
      OpdsMapping[i] = AMDGPU::getValueMapping(Bank, SrcSize);
    break;
  }
  case AMDGPU::G_BITCAST:
  case AMDGPU::G_INTTOPTR:
  case AMDGPU::G_PTRTOINT:
  case AMDGPU::G_BITREVERSE:
  case AMDGPU::G_FABS:
  case AMDGPU::G_FNEG: {
    unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    unsigned BankID = getRegBankID(MI.getOperand(1).getReg(), MRI, *TRI);
    OpdsMapping[0] = OpdsMapping[1] = AMDGPU::getValueMapping(BankID, Size);
    break;
  }
  case AMDGPU::G_CTLZ_ZERO_UNDEF:
  case AMDGPU::G_CTTZ_ZERO_UNDEF:
  case AMDGPU::G_CTPOP: {
    unsigned Size = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
    unsigned BankID = getRegBankID(MI.getOperand(1).getReg(), MRI, *TRI);
    OpdsMapping[0] = AMDGPU::getValueMapping(BankID, 32);

    // This should really be getValueMappingSGPR64Only, but allowing the generic
    // code to handle the register split just makes using LegalizerHelper more
    // difficult.
    OpdsMapping[1] = AMDGPU::getValueMapping(BankID, Size);
    break;
  }
  case AMDGPU::G_TRUNC: {
    Register Dst = MI.getOperand(0).getReg();
    Register Src = MI.getOperand(1).getReg();
    unsigned Bank = getRegBankID(Src, MRI, *TRI);
    unsigned DstSize = getSizeInBits(Dst, MRI, *TRI);
    unsigned SrcSize = getSizeInBits(Src, MRI, *TRI);
    OpdsMapping[0] = AMDGPU::getValueMapping(Bank, DstSize);
    OpdsMapping[1] = AMDGPU::getValueMapping(Bank, SrcSize);
    break;
  }
  case AMDGPU::G_ZEXT:
  case AMDGPU::G_SEXT:
  case AMDGPU::G_ANYEXT:
  case AMDGPU::G_SEXT_INREG: {
    Register Dst = MI.getOperand(0).getReg();
    Register Src = MI.getOperand(1).getReg();
    unsigned DstSize = getSizeInBits(Dst, MRI, *TRI);
    unsigned SrcSize = getSizeInBits(Src, MRI, *TRI);

    unsigned DstBank;
    const RegisterBank *SrcBank = getRegBank(Src, MRI, *TRI);
    assert(SrcBank);
    switch (SrcBank->getID()) {
    case AMDGPU::SGPRRegBankID:
      DstBank = AMDGPU::SGPRRegBankID;
      break;
    default:
      DstBank = AMDGPU::VGPRRegBankID;
      break;
    }

    // Scalar extend can use 64-bit BFE, but VGPRs require extending to
    // 32-bits, and then to 64.
    OpdsMapping[0] = AMDGPU::getValueMappingSGPR64Only(DstBank, DstSize);
    OpdsMapping[1] = AMDGPU::getValueMappingSGPR64Only(SrcBank->getID(),
                                                       SrcSize);
    break;
  }
  case AMDGPU::G_FCMP: {
    unsigned Size = MRI.getType(MI.getOperand(2).getReg()).getSizeInBits();
    unsigned Op2Bank = getRegBankID(MI.getOperand(2).getReg(), MRI, *TRI);
    OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::VCCRegBankID, 1);
    OpdsMapping[1] = nullptr; // Predicate Operand.
    OpdsMapping[2] = AMDGPU::getValueMapping(Op2Bank, Size);
    OpdsMapping[3] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Size);
    break;
  }
  case AMDGPU::G_STORE: {
    assert(MI.getOperand(0).isReg());
    unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();

    // FIXME: We need to specify a different reg bank once scalar stores are
    // supported.
    const ValueMapping *ValMapping =
        AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Size);
    OpdsMapping[0] = ValMapping;
    OpdsMapping[1] = getValueMappingForPtr(MRI, MI.getOperand(1).getReg());
    break;
  }
  case AMDGPU::G_ICMP: {
    auto Pred = static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());
    unsigned Size = MRI.getType(MI.getOperand(2).getReg()).getSizeInBits();

    // See if the result register has already been constrained to vcc, which may
    // happen due to control flow intrinsic lowering.
    unsigned DstBank = getRegBankID(MI.getOperand(0).getReg(), MRI, *TRI,
                                    AMDGPU::SGPRRegBankID);
    unsigned Op2Bank = getRegBankID(MI.getOperand(2).getReg(), MRI, *TRI);
    unsigned Op3Bank = getRegBankID(MI.getOperand(3).getReg(), MRI, *TRI);

    bool CanUseSCC = DstBank == AMDGPU::SGPRRegBankID &&
                     Op2Bank == AMDGPU::SGPRRegBankID &&
                     Op3Bank == AMDGPU::SGPRRegBankID &&
      (Size == 32 || (Size == 64 &&
                      (Pred == CmpInst::ICMP_EQ || Pred == CmpInst::ICMP_NE) &&
                      Subtarget.hasScalarCompareEq64()));

    DstBank = CanUseSCC ? AMDGPU::SGPRRegBankID : AMDGPU::VCCRegBankID;
    unsigned SrcBank = CanUseSCC ? AMDGPU::SGPRRegBankID : AMDGPU::VGPRRegBankID;

    // TODO: Use 32-bit for scalar output size.
    // SCC results will need to be copied to a 32-bit SGPR virtual register.
    const unsigned ResultSize = 1;

    OpdsMapping[0] = AMDGPU::getValueMapping(DstBank, ResultSize);
    OpdsMapping[2] = AMDGPU::getValueMapping(SrcBank, Size);
    OpdsMapping[3] = AMDGPU::getValueMapping(SrcBank, Size);
    break;
  }
  case AMDGPU::G_EXTRACT_VECTOR_ELT: {
    // VGPR index can be used for waterfall when indexing a SGPR vector.
    unsigned SrcBankID = getRegBankID(MI.getOperand(1).getReg(), MRI, *TRI);
    unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    unsigned SrcSize = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
    unsigned IdxSize = MRI.getType(MI.getOperand(2).getReg()).getSizeInBits();
    unsigned IdxBank = getRegBankID(MI.getOperand(2).getReg(), MRI, *TRI);
    unsigned OutputBankID = regBankUnion(SrcBankID, IdxBank);

    OpdsMapping[0] = AMDGPU::getValueMappingSGPR64Only(OutputBankID, DstSize);
    OpdsMapping[1] = AMDGPU::getValueMapping(SrcBankID, SrcSize);

    // The index can be either if the source vector is VGPR.
    OpdsMapping[2] = AMDGPU::getValueMapping(IdxBank, IdxSize);
    break;
  }
  case AMDGPU::G_INSERT_VECTOR_ELT: {
    unsigned OutputBankID = isSALUMapping(MI) ?
      AMDGPU::SGPRRegBankID : AMDGPU::VGPRRegBankID;

    unsigned VecSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    unsigned InsertSize = MRI.getType(MI.getOperand(2).getReg()).getSizeInBits();
    unsigned IdxSize = MRI.getType(MI.getOperand(3).getReg()).getSizeInBits();
    unsigned InsertEltBankID = getRegBankID(MI.getOperand(2).getReg(),
                                            MRI, *TRI);
    unsigned IdxBankID = getRegBankID(MI.getOperand(3).getReg(), MRI, *TRI);

    OpdsMapping[0] = AMDGPU::getValueMapping(OutputBankID, VecSize);
    OpdsMapping[1] = AMDGPU::getValueMapping(OutputBankID, VecSize);

    // This is a weird case, because we need to break down the mapping based on
    // the register bank of a different operand.
    if (InsertSize == 64 && OutputBankID == AMDGPU::VGPRRegBankID) {
      OpdsMapping[2] = AMDGPU::getValueMappingSplit64(InsertEltBankID,
                                                      InsertSize);
    } else {
      assert(InsertSize == 32 || InsertSize == 64);
      OpdsMapping[2] = AMDGPU::getValueMapping(InsertEltBankID, InsertSize);
    }

    // The index can be either if the source vector is VGPR.
    OpdsMapping[3] = AMDGPU::getValueMapping(IdxBankID, IdxSize);
    break;
  }
  case AMDGPU::G_UNMERGE_VALUES: {
    unsigned Bank = isSALUMapping(MI) ? AMDGPU::SGPRRegBankID :
      AMDGPU::VGPRRegBankID;

    // Op1 and Dst should use the same register bank.
    // FIXME: Shouldn't this be the default? Why do we need to handle this?
    for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
      unsigned Size = getSizeInBits(MI.getOperand(i).getReg(), MRI, *TRI);
      OpdsMapping[i] = AMDGPU::getValueMapping(Bank, Size);
    }
    break;
  }
  case AMDGPU::G_AMDGPU_BUFFER_LOAD:
  case AMDGPU::G_AMDGPU_BUFFER_LOAD_UBYTE:
  case AMDGPU::G_AMDGPU_BUFFER_LOAD_SBYTE:
  case AMDGPU::G_AMDGPU_BUFFER_LOAD_USHORT:
  case AMDGPU::G_AMDGPU_BUFFER_LOAD_SSHORT:
  case AMDGPU::G_AMDGPU_BUFFER_LOAD_FORMAT:
  case AMDGPU::G_AMDGPU_BUFFER_LOAD_FORMAT_D16:
  case AMDGPU::G_AMDGPU_TBUFFER_LOAD_FORMAT:
  case AMDGPU::G_AMDGPU_TBUFFER_LOAD_FORMAT_D16:
  case AMDGPU::G_AMDGPU_BUFFER_STORE:
  case AMDGPU::G_AMDGPU_BUFFER_STORE_BYTE:
  case AMDGPU::G_AMDGPU_BUFFER_STORE_SHORT:
  case AMDGPU::G_AMDGPU_BUFFER_STORE_FORMAT:
  case AMDGPU::G_AMDGPU_BUFFER_STORE_FORMAT_D16: {
    OpdsMapping[0] = getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);

    // rsrc
    OpdsMapping[1] = getSGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);

    // vindex
    OpdsMapping[2] = getVGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);

    // voffset
    OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);

    // soffset
    OpdsMapping[4] = getSGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);

    // Any remaining operands are immediates and were correctly null
    // initialized.
    break;
  }
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_SWAP:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_ADD:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_SUB:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_SMIN:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_UMIN:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_SMAX:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_UMAX:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_AND:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_OR:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_XOR:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_INC:
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_DEC: {
    // vdata_out
    OpdsMapping[0] = getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);

    // vdata_in
    OpdsMapping[1] = getVGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);

    // rsrc
    OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);

    // vindex
    OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);

    // voffset
    OpdsMapping[4] = getVGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);

    // soffset
    OpdsMapping[5] = getSGPROpMapping(MI.getOperand(5).getReg(), MRI, *TRI);

    // Any remaining operands are immediates and were correctly null
    // initialized.
    break;
  }
  case AMDGPU::G_AMDGPU_BUFFER_ATOMIC_CMPSWAP: {
    // vdata_out
    OpdsMapping[0] = getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);

    // vdata_in
    OpdsMapping[1] = getVGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);

    // cmp
    OpdsMapping[2] = getVGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);

    // rsrc
    OpdsMapping[3] = getSGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);

    // vindex
    OpdsMapping[4] = getVGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);

    // voffset
    OpdsMapping[5] = getVGPROpMapping(MI.getOperand(5).getReg(), MRI, *TRI);

    // soffset
    OpdsMapping[6] = getSGPROpMapping(MI.getOperand(6).getReg(), MRI, *TRI);

    // Any remaining operands are immediates and were correctly null
    // initialized.
    break;
  }
  case AMDGPU::G_AMDGPU_S_BUFFER_LOAD: {
    // Lie and claim everything is legal, even though some need to be
    // SGPRs. applyMapping will have to deal with it as a waterfall loop.
    OpdsMapping[1] = getSGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);
    OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);

    // We need to convert this to a MUBUF if either the resource of offset is
    // VGPR.
    unsigned RSrcBank = OpdsMapping[1]->BreakDown[0].RegBank->getID();
    unsigned OffsetBank = OpdsMapping[2]->BreakDown[0].RegBank->getID();
    unsigned ResultBank = regBankUnion(RSrcBank, OffsetBank);

    unsigned Size0 = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    OpdsMapping[0] = AMDGPU::getValueMapping(ResultBank, Size0);
    break;
  }
  case AMDGPU::G_INTRINSIC: {
    switch (MI.getIntrinsicID()) {
    default:
      return getInvalidInstructionMapping();
    case Intrinsic::amdgcn_div_fmas:
    case Intrinsic::amdgcn_div_fixup:
    case Intrinsic::amdgcn_trig_preop:
    case Intrinsic::amdgcn_sin:
    case Intrinsic::amdgcn_cos:
    case Intrinsic::amdgcn_log_clamp:
    case Intrinsic::amdgcn_rcp:
    case Intrinsic::amdgcn_rcp_legacy:
    case Intrinsic::amdgcn_rsq:
    case Intrinsic::amdgcn_rsq_legacy:
    case Intrinsic::amdgcn_rsq_clamp:
    case Intrinsic::amdgcn_fmul_legacy:
    case Intrinsic::amdgcn_ldexp:
    case Intrinsic::amdgcn_frexp_mant:
    case Intrinsic::amdgcn_frexp_exp:
    case Intrinsic::amdgcn_fract:
    case Intrinsic::amdgcn_cvt_pkrtz:
    case Intrinsic::amdgcn_cvt_pknorm_i16:
    case Intrinsic::amdgcn_cvt_pknorm_u16:
    case Intrinsic::amdgcn_cvt_pk_i16:
    case Intrinsic::amdgcn_cvt_pk_u16:
    case Intrinsic::amdgcn_fmed3:
    case Intrinsic::amdgcn_cubeid:
    case Intrinsic::amdgcn_cubema:
    case Intrinsic::amdgcn_cubesc:
    case Intrinsic::amdgcn_cubetc:
    case Intrinsic::amdgcn_sffbh:
    case Intrinsic::amdgcn_fmad_ftz:
    case Intrinsic::amdgcn_mbcnt_lo:
    case Intrinsic::amdgcn_mbcnt_hi:
    case Intrinsic::amdgcn_mul_u24:
    case Intrinsic::amdgcn_mul_i24:
    case Intrinsic::amdgcn_lerp:
    case Intrinsic::amdgcn_sad_u8:
    case Intrinsic::amdgcn_msad_u8:
    case Intrinsic::amdgcn_sad_hi_u8:
    case Intrinsic::amdgcn_sad_u16:
    case Intrinsic::amdgcn_qsad_pk_u16_u8:
    case Intrinsic::amdgcn_mqsad_pk_u16_u8:
    case Intrinsic::amdgcn_mqsad_u32_u8:
    case Intrinsic::amdgcn_cvt_pk_u8_f32:
    case Intrinsic::amdgcn_alignbit:
    case Intrinsic::amdgcn_alignbyte:
    case Intrinsic::amdgcn_fdot2:
    case Intrinsic::amdgcn_sdot2:
    case Intrinsic::amdgcn_udot2:
    case Intrinsic::amdgcn_sdot4:
    case Intrinsic::amdgcn_udot4:
    case Intrinsic::amdgcn_sdot8:
    case Intrinsic::amdgcn_udot8:
      return getDefaultMappingVOP(MI);
    case Intrinsic::amdgcn_sbfe:
    case Intrinsic::amdgcn_ubfe:
      if (isSALUMapping(MI))
        return getDefaultMappingSOP(MI);
      return getDefaultMappingVOP(MI);
    case Intrinsic::amdgcn_ds_swizzle:
    case Intrinsic::amdgcn_ds_permute:
    case Intrinsic::amdgcn_ds_bpermute:
    case Intrinsic::amdgcn_update_dpp:
    case Intrinsic::amdgcn_mov_dpp8:
    case Intrinsic::amdgcn_mov_dpp:
    case Intrinsic::amdgcn_wwm:
    case Intrinsic::amdgcn_wqm:
    case Intrinsic::amdgcn_softwqm:
      return getDefaultMappingAllVGPR(MI);
    case Intrinsic::amdgcn_kernarg_segment_ptr:
    case Intrinsic::amdgcn_s_getpc:
    case Intrinsic::amdgcn_groupstaticsize: {
      unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size);
      break;
    }
    case Intrinsic::amdgcn_wqm_vote: {
      unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      OpdsMapping[0] = OpdsMapping[2]
        = AMDGPU::getValueMapping(AMDGPU::VCCRegBankID, Size);
      break;
    }
    case Intrinsic::amdgcn_ps_live: {
      OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::VCCRegBankID, 1);
      break;
    }
    case Intrinsic::amdgcn_div_scale: {
      unsigned Dst0Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      unsigned Dst1Size = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
      OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Dst0Size);
      OpdsMapping[1] = AMDGPU::getValueMapping(AMDGPU::VCCRegBankID, Dst1Size);

      unsigned SrcSize = MRI.getType(MI.getOperand(3).getReg()).getSizeInBits();
      OpdsMapping[3] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, SrcSize);
      OpdsMapping[4] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, SrcSize);
      break;
    }
    case Intrinsic::amdgcn_class: {
      Register Src0Reg = MI.getOperand(2).getReg();
      Register Src1Reg = MI.getOperand(3).getReg();
      unsigned Src0Size = MRI.getType(Src0Reg).getSizeInBits();
      unsigned Src1Size = MRI.getType(Src1Reg).getSizeInBits();
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::VCCRegBankID, DstSize);
      OpdsMapping[2] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Src0Size);
      OpdsMapping[3] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Src1Size);
      break;
    }
    case Intrinsic::amdgcn_icmp:
    case Intrinsic::amdgcn_fcmp: {
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      // This is not VCCRegBank because this is not used in boolean contexts.
      OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, DstSize);
      unsigned OpSize = MRI.getType(MI.getOperand(2).getReg()).getSizeInBits();
      OpdsMapping[2] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, OpSize);
      OpdsMapping[3] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, OpSize);
      break;
    }
    case Intrinsic::amdgcn_readlane: {
      // This must be an SGPR, but accept a VGPR.
      Register IdxReg = MI.getOperand(3).getReg();
      unsigned IdxSize = MRI.getType(IdxReg).getSizeInBits();
      unsigned IdxBank = getRegBankID(IdxReg, MRI, *TRI, AMDGPU::SGPRRegBankID);
      OpdsMapping[3] = AMDGPU::getValueMapping(IdxBank, IdxSize);
      LLVM_FALLTHROUGH;
    }
    case Intrinsic::amdgcn_readfirstlane: {
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      unsigned SrcSize = MRI.getType(MI.getOperand(2).getReg()).getSizeInBits();
      OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, DstSize);
      OpdsMapping[2] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, SrcSize);
      break;
    }
    case Intrinsic::amdgcn_writelane: {
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      Register SrcReg = MI.getOperand(2).getReg();
      unsigned SrcSize = MRI.getType(SrcReg).getSizeInBits();
      unsigned SrcBank = getRegBankID(SrcReg, MRI, *TRI, AMDGPU::SGPRRegBankID);
      Register IdxReg = MI.getOperand(3).getReg();
      unsigned IdxSize = MRI.getType(IdxReg).getSizeInBits();
      unsigned IdxBank = getRegBankID(IdxReg, MRI, *TRI, AMDGPU::SGPRRegBankID);
      OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, DstSize);

      // These 2 must be SGPRs, but accept VGPRs. Readfirstlane will be inserted
      // to legalize.
      OpdsMapping[2] = AMDGPU::getValueMapping(SrcBank, SrcSize);
      OpdsMapping[3] = AMDGPU::getValueMapping(IdxBank, IdxSize);
      OpdsMapping[4] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, SrcSize);
      break;
    }
    case Intrinsic::amdgcn_if_break: {
      unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
      OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size);
      OpdsMapping[2] = AMDGPU::getValueMapping(AMDGPU::VCCRegBankID, 1);
      OpdsMapping[3] = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size);
      break;
    }
    case Intrinsic::amdgcn_permlane16:
    case Intrinsic::amdgcn_permlanex16: {
      unsigned Size = getSizeInBits(MI.getOperand(0).getReg(), MRI, *TRI);
      OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Size);
      OpdsMapping[2] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Size);
      OpdsMapping[3] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, Size);
      OpdsMapping[4] = getSGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);
      OpdsMapping[5] = getSGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::amdgcn_mfma_f32_4x4x1f32:
    case Intrinsic::amdgcn_mfma_f32_4x4x4f16:
    case Intrinsic::amdgcn_mfma_i32_4x4x4i8:
    case Intrinsic::amdgcn_mfma_f32_4x4x2bf16:
    case Intrinsic::amdgcn_mfma_f32_16x16x1f32:
    case Intrinsic::amdgcn_mfma_f32_16x16x4f32:
    case Intrinsic::amdgcn_mfma_f32_16x16x4f16:
    case Intrinsic::amdgcn_mfma_f32_16x16x16f16:
    case Intrinsic::amdgcn_mfma_i32_16x16x4i8:
    case Intrinsic::amdgcn_mfma_i32_16x16x16i8:
    case Intrinsic::amdgcn_mfma_f32_16x16x2bf16:
    case Intrinsic::amdgcn_mfma_f32_16x16x8bf16:
    case Intrinsic::amdgcn_mfma_f32_32x32x1f32:
    case Intrinsic::amdgcn_mfma_f32_32x32x2f32:
    case Intrinsic::amdgcn_mfma_f32_32x32x4f16:
    case Intrinsic::amdgcn_mfma_f32_32x32x8f16:
    case Intrinsic::amdgcn_mfma_i32_32x32x4i8:
    case Intrinsic::amdgcn_mfma_i32_32x32x8i8:
    case Intrinsic::amdgcn_mfma_f32_32x32x2bf16:
    case Intrinsic::amdgcn_mfma_f32_32x32x4bf16: {
      // Default for MAI intrinsics.
      // srcC can also be an immediate which can be folded later.
      // FIXME: Should we eventually add an alternative mapping with AGPR src
      // for srcA/srcB?
      //
      // vdst, srcA, srcB, srcC
      OpdsMapping[0] = getAGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);
      OpdsMapping[2] = getVGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);
      OpdsMapping[4] = getAGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::amdgcn_interp_p1:
    case Intrinsic::amdgcn_interp_p2:
    case Intrinsic::amdgcn_interp_mov:
    case Intrinsic::amdgcn_interp_p1_f16:
    case Intrinsic::amdgcn_interp_p2_f16: {
      const int M0Idx = MI.getNumOperands() - 1;
      Register M0Reg = MI.getOperand(M0Idx).getReg();
      unsigned M0Bank = getRegBankID(M0Reg, MRI, *TRI, AMDGPU::SGPRRegBankID);
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();

      OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, DstSize);
      for (int I = 2; I != M0Idx && MI.getOperand(I).isReg(); ++I)
        OpdsMapping[I] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, 32);

      // Must be SGPR, but we must take whatever the original bank is and fix it
      // later.
      OpdsMapping[M0Idx] = AMDGPU::getValueMapping(M0Bank, 32);
      break;
    }
    }
    break;
  }
  case AMDGPU::G_AMDGPU_INTRIN_IMAGE_LOAD:
  case AMDGPU::G_AMDGPU_INTRIN_IMAGE_STORE: {
    auto IntrID = MI.getIntrinsicID();
    const AMDGPU::RsrcIntrinsic *RSrcIntrin = AMDGPU::lookupRsrcIntrinsic(IntrID);
    assert(RSrcIntrin && "missing RsrcIntrinsic for image intrinsic");
    // Non-images can have complications from operands that allow both SGPR
    // and VGPR. For now it's too complicated to figure out the final opcode
    // to derive the register bank from the MCInstrDesc.
    assert(RSrcIntrin->IsImage);
    return getImageMapping(MRI, MI, RSrcIntrin->RsrcArg);
  }
  case AMDGPU::G_INTRINSIC_W_SIDE_EFFECTS: {
    auto IntrID = MI.getIntrinsicID();
    switch (IntrID) {
    case Intrinsic::amdgcn_s_getreg:
    case Intrinsic::amdgcn_s_memtime:
    case Intrinsic::amdgcn_s_memrealtime:
    case Intrinsic::amdgcn_s_get_waveid_in_workgroup: {
      unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size);
      break;
    }
    case Intrinsic::amdgcn_ds_fadd:
    case Intrinsic::amdgcn_ds_fmin:
    case Intrinsic::amdgcn_ds_fmax:
      return getDefaultMappingAllVGPR(MI);
    case Intrinsic::amdgcn_ds_ordered_add:
    case Intrinsic::amdgcn_ds_ordered_swap: {
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, DstSize);
      unsigned M0Bank = getRegBankID(MI.getOperand(2).getReg(), MRI, *TRI,
                                 AMDGPU::SGPRRegBankID);
      OpdsMapping[2] = AMDGPU::getValueMapping(M0Bank, 32);
      OpdsMapping[3] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, 32);
      break;
    }
    case Intrinsic::amdgcn_ds_append:
    case Intrinsic::amdgcn_ds_consume: {
      unsigned DstSize = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
      OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, DstSize);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::amdgcn_exp_compr:
      OpdsMapping[3] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, 32);
      OpdsMapping[4] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, 32);
      break;
    case Intrinsic::amdgcn_exp:
      // FIXME: Could we support packed types here?
      OpdsMapping[3] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, 32);
      OpdsMapping[4] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, 32);
      OpdsMapping[5] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, 32);
      OpdsMapping[6] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, 32);
      break;
    case Intrinsic::amdgcn_s_sendmsg:
    case Intrinsic::amdgcn_s_sendmsghalt: {
      // This must be an SGPR, but accept a VGPR.
      unsigned Bank = getRegBankID(MI.getOperand(2).getReg(), MRI, *TRI,
                                   AMDGPU::SGPRRegBankID);
      OpdsMapping[2] = AMDGPU::getValueMapping(Bank, 32);
      break;
    }
    case Intrinsic::amdgcn_end_cf:
    case Intrinsic::amdgcn_init_exec: {
      unsigned Size = getSizeInBits(MI.getOperand(1).getReg(), MRI, *TRI);
      OpdsMapping[1] = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size);
      break;
    }
    case Intrinsic::amdgcn_else: {
      unsigned WaveSize = getSizeInBits(MI.getOperand(1).getReg(), MRI, *TRI);
      OpdsMapping[0] = AMDGPU::getValueMapping(AMDGPU::VCCRegBankID, 1);
      OpdsMapping[1] = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, WaveSize);
      OpdsMapping[3] = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, WaveSize);
      break;
    }
    case Intrinsic::amdgcn_kill: {
      OpdsMapping[1] = AMDGPU::getValueMapping(AMDGPU::VCCRegBankID, 1);
      break;
    }
    case Intrinsic::amdgcn_raw_buffer_load:
    case Intrinsic::amdgcn_raw_tbuffer_load: {
      // FIXME: Should make intrinsic ID the last operand of the instruction,
      // then this would be the same as store
      OpdsMapping[0] = getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);
      OpdsMapping[4] = getSGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::amdgcn_raw_buffer_store:
    case Intrinsic::amdgcn_raw_buffer_store_format:
    case Intrinsic::amdgcn_raw_tbuffer_store: {
      OpdsMapping[1] = getVGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);
      OpdsMapping[4] = getSGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::amdgcn_struct_buffer_load:
    case Intrinsic::amdgcn_struct_tbuffer_load: {
      OpdsMapping[0] = getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);
      OpdsMapping[4] = getVGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);
      OpdsMapping[5] = getSGPROpMapping(MI.getOperand(5).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::amdgcn_struct_buffer_store:
    case Intrinsic::amdgcn_struct_tbuffer_store: {
      OpdsMapping[1] = getVGPROpMapping(MI.getOperand(1).getReg(), MRI, *TRI);
      OpdsMapping[2] = getSGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
      OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);
      OpdsMapping[4] = getVGPROpMapping(MI.getOperand(4).getReg(), MRI, *TRI);
      OpdsMapping[5] = getSGPROpMapping(MI.getOperand(5).getReg(), MRI, *TRI);
      break;
    }
    case Intrinsic::amdgcn_init_exec_from_input: {
      unsigned Size = getSizeInBits(MI.getOperand(1).getReg(), MRI, *TRI);
      OpdsMapping[1] = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size);
      OpdsMapping[2] = AMDGPU::getValueMapping(AMDGPU::SGPRRegBankID, Size);
      break;
    }
    case Intrinsic::amdgcn_ds_gws_init:
    case Intrinsic::amdgcn_ds_gws_barrier:
    case Intrinsic::amdgcn_ds_gws_sema_br: {
      OpdsMapping[1] = AMDGPU::getValueMapping(AMDGPU::VGPRRegBankID, 32);

      // This must be an SGPR, but accept a VGPR.
      unsigned Bank = getRegBankID(MI.getOperand(2).getReg(), MRI, *TRI,
                                   AMDGPU::SGPRRegBankID);
      OpdsMapping[2] = AMDGPU::getValueMapping(Bank, 32);
      break;
    }
    case Intrinsic::amdgcn_ds_gws_sema_v:
    case Intrinsic::amdgcn_ds_gws_sema_p:
    case Intrinsic::amdgcn_ds_gws_sema_release_all: {
      // This must be an SGPR, but accept a VGPR.
      unsigned Bank = getRegBankID(MI.getOperand(1).getReg(), MRI, *TRI,
                                   AMDGPU::SGPRRegBankID);
      OpdsMapping[1] = AMDGPU::getValueMapping(Bank, 32);
      break;
    }
    default:
      return getInvalidInstructionMapping();
    }
    break;
  }
  case AMDGPU::G_SELECT: {
    unsigned Size = MRI.getType(MI.getOperand(0).getReg()).getSizeInBits();
    unsigned Op2Bank = getRegBankID(MI.getOperand(2).getReg(), MRI, *TRI,
                                    AMDGPU::SGPRRegBankID);
    unsigned Op3Bank = getRegBankID(MI.getOperand(3).getReg(), MRI, *TRI,
                                    AMDGPU::SGPRRegBankID);
    bool SGPRSrcs = Op2Bank == AMDGPU::SGPRRegBankID &&
                    Op3Bank == AMDGPU::SGPRRegBankID;

    unsigned CondBankDefault = SGPRSrcs ?
      AMDGPU::SGPRRegBankID : AMDGPU::VCCRegBankID;
    unsigned CondBank = getRegBankID(MI.getOperand(1).getReg(), MRI, *TRI,
                                     CondBankDefault);
    if (CondBank == AMDGPU::SGPRRegBankID)
      CondBank = SGPRSrcs ? AMDGPU::SGPRRegBankID : AMDGPU::VCCRegBankID;
    else if (CondBank == AMDGPU::VGPRRegBankID)
      CondBank = AMDGPU::VCCRegBankID;

    unsigned Bank = SGPRSrcs && CondBank == AMDGPU::SGPRRegBankID ?
      AMDGPU::SGPRRegBankID : AMDGPU::VGPRRegBankID;

    assert(CondBank == AMDGPU::VCCRegBankID || CondBank == AMDGPU::SGPRRegBankID);

    // TODO: Should report 32-bit for scalar condition type.
    if (Size == 64) {
      OpdsMapping[0] = AMDGPU::getValueMappingSGPR64Only(Bank, Size);
      OpdsMapping[1] = AMDGPU::getValueMapping(CondBank, 1);
      OpdsMapping[2] = AMDGPU::getValueMappingSGPR64Only(Bank, Size);
      OpdsMapping[3] = AMDGPU::getValueMappingSGPR64Only(Bank, Size);
    } else {
      OpdsMapping[0] = AMDGPU::getValueMapping(Bank, Size);
      OpdsMapping[1] = AMDGPU::getValueMapping(CondBank, 1);
      OpdsMapping[2] = AMDGPU::getValueMapping(Bank, Size);
      OpdsMapping[3] = AMDGPU::getValueMapping(Bank, Size);
    }

    break;
  }

  case AMDGPU::G_LOAD:
  case AMDGPU::G_ZEXTLOAD:
  case AMDGPU::G_SEXTLOAD:
    return getInstrMappingForLoad(MI);

  case AMDGPU::G_ATOMICRMW_XCHG:
  case AMDGPU::G_ATOMICRMW_ADD:
  case AMDGPU::G_ATOMICRMW_SUB:
  case AMDGPU::G_ATOMICRMW_AND:
  case AMDGPU::G_ATOMICRMW_OR:
  case AMDGPU::G_ATOMICRMW_XOR:
  case AMDGPU::G_ATOMICRMW_MAX:
  case AMDGPU::G_ATOMICRMW_MIN:
  case AMDGPU::G_ATOMICRMW_UMAX:
  case AMDGPU::G_ATOMICRMW_UMIN:
  case AMDGPU::G_ATOMICRMW_FADD:
  case AMDGPU::G_AMDGPU_ATOMIC_CMPXCHG:
  case AMDGPU::G_AMDGPU_ATOMIC_INC:
  case AMDGPU::G_AMDGPU_ATOMIC_DEC: {
    OpdsMapping[0] = getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);
    OpdsMapping[1] = getValueMappingForPtr(MRI, MI.getOperand(1).getReg());
    OpdsMapping[2] = getVGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
    break;
  }
  case AMDGPU::G_ATOMIC_CMPXCHG: {
    OpdsMapping[0] = getVGPROpMapping(MI.getOperand(0).getReg(), MRI, *TRI);
    OpdsMapping[1] = getValueMappingForPtr(MRI, MI.getOperand(1).getReg());
    OpdsMapping[2] = getVGPROpMapping(MI.getOperand(2).getReg(), MRI, *TRI);
    OpdsMapping[3] = getVGPROpMapping(MI.getOperand(3).getReg(), MRI, *TRI);
    break;
  }
  case AMDGPU::G_BRCOND: {
    unsigned Bank = getRegBankID(MI.getOperand(0).getReg(), MRI, *TRI,
                                 AMDGPU::SGPRRegBankID);
    assert(MRI.getType(MI.getOperand(0).getReg()).getSizeInBits() == 1);
    if (Bank != AMDGPU::SGPRRegBankID)
      Bank = AMDGPU::VCCRegBankID;

    OpdsMapping[0] = AMDGPU::getValueMapping(Bank, 1);
    break;
  }
  }

  return getInstructionMapping(/*ID*/1, /*Cost*/1,
                               getOperandsMapping(OpdsMapping),
                               MI.getNumOperands());
}
