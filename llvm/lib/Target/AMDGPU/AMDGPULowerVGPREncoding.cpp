//===- AMDGPULowerVGPREncoding.cpp - Set MODE & Lower idx Pseudos ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Lower VGPRs above first 256 on gfx1250+.
/// Also lowers dynamic VGPR indexing pseudo instructions to subtarget
/// instructions.
///
/// The pass scans used VGPRs and inserts S_SET_VGPR_MSB instructions on
/// gfx1250 (or S_SET_VGPR_FRAMES on gfx13+) to switch VGPR addressing mode. The
/// mode change is effective until the next change. This instruction provides
/// high bits of a VGPR address for four of the operands: vdst, src0, src1, and
/// src2, or other 4 operands depending on the instruction encoding. If bits are
/// set they are added as MSB to the corresponding operand VGPR number.
///
/// There is no need to replace actual register operands because encoding of the
/// high and low VGPRs is the same. I.e. v0 has the encoding 0x100, so does
/// v256. v1 has the encoding 0x101 and v257 has the same encoding. So high
/// VGPRs will survive until actual encoding and will result in a same actual
/// bit encoding.
///
/// The InstPrinter will take care of the printing a low VGPR instead of a high
/// one. In prinicple this shall be viable to print actual high VGPR numbers,
/// but that would disagree with a disasm printing and create a situation where
/// asm text is not deterministic.
///
/// Another part of the pass is lowering of dynamic VGPR indexing pseudo
/// instructions. V_LOAD/STORE_IDX are lowered to one or several V_MOV_B32,
/// and the index registers they use are encoded in a preceding update to the
/// index select bits in MODE using S_SET_VGPR_FRAMES. Dynamic indexing bundles
/// containing V_LOAD/STORE_IDXs and a CoreMI are lowered by folding
/// V_LOAD/STORE_IDX of CoreMI's operands into CoreMI, and inserting
/// S_SET_VGPR_FRAMES.
///
/// This pass creates a convention where non-fall through basic blocks shall
/// start with all MODE register bits 0. Otherwise a disassembly would not be
/// readable. An optimization here is possible but deemed not desirable because
/// of the readbility concerns.
///
/// Consequentially the ABI is set to expect all 16 MODE bits to be zero on
/// entry. The pass must run very late in the pipeline to make sure no changes
/// to VGPR operands will be made after it.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUResourceUsageAnalysis.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-lower-vgpr-encoding"

namespace {

class AMDGPULowerVGPREncoding : public MachineFunctionPass {
  static constexpr unsigned VSrc0 = 0, VDst = 3;
  static constexpr unsigned OpNum = 7;

  enum class EncodeType : unsigned {
    SET_VGPR_MSB = 0,
    SET_VGPR_FRAMES = 1,
    VOPM = 2
  };

  struct OpMode {
    // No MSBs or index register set means they are not required to be
    // of a particular value.
    std::optional<unsigned> MSBits;
    std::optional<MCRegister> IdxReg;

    bool update(const OpMode &New, bool &Rewritten) {
      bool Updated = false;
      if (New.MSBits) {
        if (*New.MSBits != MSBits.value_or(0)) {
          Updated = true;
          Rewritten |= MSBits.has_value();
        }
        MSBits = New.MSBits;
      }
      if (New.IdxReg) {
        if (*New.IdxReg != IdxReg.value_or(AMDGPU::IDX0)) {
          Updated = true;
          Rewritten |= IdxReg.has_value();
        }
        IdxReg = New.IdxReg;
      }
      return Updated;
    }
  };

  struct ModeTy {
    OpMode Ops[OpNum];

    bool update(const ModeTy &New, bool &Rewritten) {
      bool Updated = false;
      for (unsigned I : seq(OpNum))
        Updated |= Ops[I].update(New.Ops[I], Rewritten);
      return Updated;
    }

    bool isSet() const {
      return any_of(Ops, [](const OpMode &Op) {
        return Op.MSBits.value_or(0) != 0 ||
               Op.IdxReg.value_or(AMDGPU::IDX0) != AMDGPU::IDX0;
      });
    }

    unsigned encode(EncodeType type) const {
      switch (type) {
      case EncodeType::SET_VGPR_FRAMES: {
        // GFX13 layout:
        // [src0 idx_sel, src1 idx_sel, src2 idx_sel, dst idx_sel,
        //  src0 msb, src1 msb, src2 msb, dst msb]
        static constexpr unsigned BitsPerField = 2;
        static constexpr unsigned MSBFieldsPos = 8;
        unsigned V = 0;
        for (const auto &[I, Op] : enumerate(Ops)) {
          MCRegister R = Op.IdxReg.value_or(AMDGPU::IDX0);
          assert(AMDGPU::IDX0 <= R && R <= AMDGPU::IDX3);
          V |= (R - AMDGPU::IDX0) << (I * BitsPerField);
          V |= Op.MSBits.value_or(0) << (I * BitsPerField + MSBFieldsPos);
        }
        return V;
      }
      case EncodeType::VOPM: {
        static constexpr unsigned BitsPerField = 4;
        // GFX13 VOPM layout in idxs operand:
        // [dst idx_sel, src0 idx_sel, src1 idx_sel, ... ]
        unsigned V = 0;
        for (const auto &[I, Op] : enumerate(Ops)) {
          MCRegister R = Op.IdxReg.value_or(AMDGPU::IDX0);
          assert(AMDGPU::IDX0 <= R && R <= AMDGPU::IDX3);
          V |= (R - AMDGPU::IDX0) << (I * BitsPerField);
        }
        return V;
      }
      default: {
        assert(type == EncodeType::SET_VGPR_MSB);
        // GFX1250 layout: [src0 msb, src1 msb, src2 msb, dst msb]
        static constexpr unsigned BitsPerField = 2;
        unsigned V = 0;
        for (const auto &[I, Op] : enumerate(Ops))
          V |= Op.MSBits.value_or(0) << (I * BitsPerField);
        return V;
      }
      }
    }
  };

public:
  static char ID;

  AMDGPULowerVGPREncoding() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    // Preserving the resource usage analysis is required here, and in all
    // following passes, because the VGPR usage cannot be computed properly
    // after lowering dynamic indexing offsets to VGPRs (without a costly change
    // to the analysis).
    AU.addPreserved<AMDGPUResourceUsageAnalysis>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const GCNSubtarget *ST;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  const SIMachineFunctionInfo *MFI;

  /// Most recent s_set_* instruction.
  MachineInstr *MostRecentModeSet;

  /// Whether the current mode is known.
  bool CurrentModeKnown;

  /// Current mode values. The current mode is suitable for all instructions
  /// between the previous mode set, MostRecentModeSet, and the previous
  /// instruction. If it can be updated to include the current instruction we
  /// will do it, and if it can't we will insert a new mode set.
  ModeTy CurrentMode;

  /// Number of current hard clause instructions.
  unsigned ClauseLen;

  /// Number of hard clause instructions remaining.
  unsigned ClauseRemaining;

  /// Clause group breaks.
  unsigned ClauseBreaks;

  /// Last hard clause instruction.
  MachineInstr *Clause;

  /// Insert mode change before \p I. \returns true if mode was changed.
  bool setMode(ModeTy NewMode, MachineInstr *I);

  /// Reset mode to default.
  void resetMode(MachineInstr *I) {
    setMode(ModeTy{{{0, AMDGPU::IDX0},
                    {0, AMDGPU::IDX0},
                    {0, AMDGPU::IDX0},
                    {0, AMDGPU::IDX0}}},
            I);
  }

  /// If \p MO references VGPRs, return the MSBs. Otherwise, return nullopt.
  std::optional<unsigned> getMSBs(const MachineOperand &MO) const;

  /// Handle single \p MI. \return true if changed.
  /// Updates MII to point to the last instruction processed (existing or newly
  /// inserted) for mode update, so that upon increment in runOnMachineFunction
  /// MII is the correct value to process next.
  bool runOnMachineInstr(MachineBasicBlock::instr_iterator &MII);

  /// Compute and set the mode for a single \p MI given \p Ops
  /// operands bit mapping. If MI is BUNDLE, lowers the bundle by replacing the
  /// operands of CoreMI with the dynamic indices from the bundle. Optionally
  /// takes second array \p Ops2 for VOPD. If provided and an operand from \p
  /// Ops is not a VGPR, then \p Ops2 is checked.
  void lowerInstrOrBundle(MachineInstr &MI, MachineInstr *CoreMI,
                          const unsigned Ops[OpNum],
                          const unsigned *Ops2 = nullptr);

  /// Lower V_LOAD/STORE_IDX to one or several V_MOV_B32 instructions and update
  /// the mode. MII is updated to point to the last V_MOV inserted.
  void lowerIDX(MachineBasicBlock::instr_iterator &MII);

  /// Lower bundles which only contain V_LOAD/STORE_IDX, as would be used
  /// to move data, and update the mode. MII is updated to point to the
  /// last V_MOV inserted.
  void lowerMovBundle(MachineInstr &MI, MachineInstr &CoreMI,
                      MachineBasicBlock::instr_iterator &MII);

  /// Check if an instruction \p I is within a clause and returns a suitable
  /// iterator to insert mode change. It may also modify the S_CLAUSE
  /// instruction to extend it or drop the clause if it cannot be adjusted.
  MachineInstr *handleClause(MachineInstr *I);
};

bool AMDGPULowerVGPREncoding::setMode(ModeTy NewMode, MachineInstr *I) {

  if (AMDGPU::isVOPMPseudo(I->getOpcode())) {

    MachineOperand *IdxsOp = TII->getNamedOperand(*I, AMDGPU::OpName::idxs);
    assert(IdxsOp && IdxsOp->getImm() == 0);
    IdxsOp->setImm(NewMode.encode(EncodeType::VOPM));

    return true;
  }

  EncodeType encodeType = ST->hasVGPRIndexingRegisters()
                              ? EncodeType::SET_VGPR_FRAMES
                              : EncodeType::SET_VGPR_MSB;

  if (CurrentModeKnown) {
    bool Rewritten = false;
    if (!CurrentMode.update(NewMode, Rewritten))
      return false;

    if (MostRecentModeSet && !Rewritten) {
      MostRecentModeSet->getOperand(0).setImm(CurrentMode.encode(encodeType));
      return true;
    }
  }

  I = handleClause(I);
  MostRecentModeSet = BuildMI(*I->getParent(), I, {},
                              TII->get(ST->hasVGPRIndexingRegisters()
                                           ? AMDGPU::S_SET_VGPR_FRAMES
                                           : AMDGPU::S_SET_VGPR_MSB))
                          .addImm(NewMode.encode(encodeType));

  CurrentMode = NewMode;
  CurrentModeKnown = true;
  return true;
}

std::optional<unsigned>
AMDGPULowerVGPREncoding::getMSBs(const MachineOperand &MO) const {
  if (!MO.isReg())
    return std::nullopt;

  MCRegister Reg = MO.getReg();
  const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(Reg);
  if (!RC || !TRI->isVGPRClass(RC))
    return std::nullopt;

  unsigned Idx = TRI->getHWRegIndex(Reg);
  return Idx >> 8;
}

void AMDGPULowerVGPREncoding::lowerMovBundle(
    MachineInstr &MI, MachineInstr &CoreMI,
    MachineBasicBlock::instr_iterator &MII) {
  assert(CoreMI.getOpcode() == AMDGPU::V_STORE_IDX);

  // The RC in MachineInstrDesc for V_LOAD/STORE_IDX can contain many
  // possible register sizes, we need to use the MMO instead to determine size.
  assert(CoreMI.hasOneMemOperand() && "V_LOAD/STORE_IDX must have one MMO");
  MachineMemOperand *MMO = *CoreMI.memoperands_begin();
  auto Size = MMO->getSizeInBits().getValue();
  if (Size % 32 != 0)
    report_fatal_error(
        "TODO-GFX13 Support lowering non-multiple-of-32-bit sizes for "
        "V_LOAD/STORE_IDX");

  MachineInstr *LoadMI = CoreMI.getPrevNode();

#if !defined(NDEBUG)
  // Check if the value loaded by V_LOAD_IDX is the same as stored by
  // V_STORE_IDX
  assert(LoadMI->getOpcode() == AMDGPU::V_LOAD_IDX &&
         "V_LOAD_IDX + V_STORE_IDX Bundle was not created correctly");
  assert(LoadMI->hasOneMemOperand() && "V_LOAD/STORE_IDX must have one MMO");
  MachineMemOperand *LoadMMO = *LoadMI->memoperands_begin();
  auto LoadSize = LoadMMO->getSizeInBits().getValue();
  MachineOperand &DataOp = CoreMI.getOperand(
      AMDGPU::getNamedOperandIdx(AMDGPU::V_STORE_IDX, AMDGPU::OpName::data_op));
  const auto *TRI = ST->getRegisterInfo();
  unsigned StoreDataRegNum = TRI->getHWRegIndex(DataOp.getReg());
  MachineOperand &LoadDataOp = LoadMI->getOperand(
      AMDGPU::getNamedOperandIdx(AMDGPU::V_LOAD_IDX, AMDGPU::OpName::data_op));
  unsigned LoadDataRegNum = TRI->getHWRegIndex(LoadDataOp.getReg());
  assert(LoadSize == Size && LoadDataRegNum == StoreDataRegNum &&
         "V_LOAD_IDX + V_STORE_IDX Bundle was not created correctly");
#endif

  Register StoreIdxReg = CoreMI
                             .getOperand(AMDGPU::getNamedOperandIdx(
                                 AMDGPU::V_STORE_IDX, AMDGPU::OpName::idx))
                             .getReg();
  int StoreOffsetIdx =
      AMDGPU::getNamedOperandIdx(AMDGPU::V_STORE_IDX, AMDGPU::OpName::offset);
  unsigned StoreOffset = CoreMI.getOperand(StoreOffsetIdx).getImm();

  Register LoadIdxReg = LoadMI
                            ->getOperand(AMDGPU::getNamedOperandIdx(
                                AMDGPU::V_LOAD_IDX, AMDGPU::OpName::idx))
                            .getReg();
  int LoadOffsetIdx =
      AMDGPU::getNamedOperandIdx(AMDGPU::V_LOAD_IDX, AMDGPU::OpName::offset);
  unsigned LoadOffset = LoadMI->getOperand(LoadOffsetIdx).getImm();

  ModeTy NewMode;
  NewMode.Ops[VSrc0] = {0, LoadIdxReg.asMCReg()};
  NewMode.Ops[VDst] = {0, StoreIdxReg.asMCReg()};

  unsigned MaxVGPR = ST->getAddressableNumVGPRs() - 1;
  const MCInstrDesc &OpDesc = TII->get(AMDGPU::V_MOV_B32_e32);
  for (unsigned I = 0; I < Size / 32; ++I) {
    unsigned CurLoadOffset = (LoadOffset + I) & MaxVGPR;
    unsigned CurStoreOffset = (StoreOffset + I) & MaxVGPR;

    auto MIB = BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), OpDesc);
    MIB.addDef(AMDGPU::VGPR0 + CurStoreOffset)
        .addUse(AMDGPU::VGPR0 + CurLoadOffset, RegState::Undef);
    NewMode.Ops[VSrc0].MSBits = CurLoadOffset >> 8;
    NewMode.Ops[VDst].MSBits = CurStoreOffset >> 8;

    setMode(NewMode, &*MIB);
    MII = MachineBasicBlock::instr_iterator(MIB);
  }
  LoadMI->removeFromBundle();
  CoreMI.removeFromBundle();
  MI.eraseFromParent();
}

void AMDGPULowerVGPREncoding::lowerIDX(MachineBasicBlock::instr_iterator &MII) {
  MachineInstr &MI = *MII;
  unsigned Opc = MI.getOpcode();
  bool IsLoad = Opc == AMDGPU::V_LOAD_IDX;

  // The RC in MachineInstrDesc for V_LOAD/STORE_IDX can contain many
  // possible register sizes, we need to use the MMO instead to determine size.
  assert(MI.hasOneMemOperand() && "V_LOAD/STORE_IDX must have one MMO");
  MachineMemOperand *MMO = *MI.memoperands_begin();
  auto Size = MMO->getSizeInBits().getValue();
  assert((Size % 32) == 0 &&
         "TODO-GFX13 Support lowering non-multiple-of-32-bit sizes for "
         "V_LOAD/STORE_IDX");
  const auto *TRI = ST->getRegisterInfo();
  MachineOperand &DataOp =
      MI.getOperand(AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::data_op));
  unsigned DataRegNum = TRI->getHWRegIndex(DataOp.getReg());

  Register IdxReg =
      MI.getOperand(AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::idx))
          .getReg();

  int OffsetIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::offset);
  assert(OffsetIdx != -1 && "Malformed V_LOAD/STORE_IDX instruction");
  unsigned Offset = MI.getOperand(OffsetIdx).getImm();

  ModeTy NewMode;
  NewMode.Ops[VSrc0] = {0, AMDGPU::IDX0};
  NewMode.Ops[VDst] = {0, AMDGPU::IDX0};
  if (IsLoad)
    NewMode.Ops[VSrc0].IdxReg = IdxReg.asMCReg();
  else
    NewMode.Ops[VDst].IdxReg = IdxReg.asMCReg();

  unsigned MaxVGPR = ST->getAddressableNumVGPRs() - 1;
  const MCInstrDesc &OpDesc = TII->get(AMDGPU::V_MOV_B32_e32);
  for (unsigned i = 0; i < Size / 32; ++i) {
    unsigned CurOffset = (Offset + i) & MaxVGPR;
    unsigned CurData = DataRegNum + i;

    auto MIB = BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), OpDesc);
    if (IsLoad) {
      MIB.addDef(AMDGPU::VGPR0 + CurData)
          .addUse(AMDGPU::VGPR0 + CurOffset, RegState::Undef);

      NewMode.Ops[VSrc0].MSBits = CurOffset >> 8;
      NewMode.Ops[VDst].MSBits = CurData >> 8;
    } else {
      MIB.addDef(AMDGPU::VGPR0 + CurOffset)
          .addUse(AMDGPU::VGPR0 + CurData, DataOp.isUndef() ? RegState::Undef : 0);
      NewMode.Ops[VSrc0].MSBits = CurData >> 8;
      NewMode.Ops[VDst].MSBits = CurOffset >> 8;
    }

    setMode(NewMode, &*MIB);
    MII = MachineBasicBlock::instr_iterator(MIB);
  }

  MI.eraseFromParent();
}

void AMDGPULowerVGPREncoding::lowerInstrOrBundle(MachineInstr &MI,
                                                 MachineInstr *CoreMI,
                                                 const unsigned Ops[OpNum],
                                                 const unsigned *Ops2) {
  bool IsBundleWithGPRIndexing = CoreMI != nullptr;
  if (!CoreMI)
    CoreMI = &MI;
  ModeTy NewMode;

  for (unsigned I = 0; I < OpNum; ++I) {
    MachineOperand *CoreOp = TII->getNamedOperand(*CoreMI, Ops[I]);

    if (CoreOp && IsBundleWithGPRIndexing && CoreOp->isReg()) {
      MachineBasicBlock::instr_iterator II = MI.getIterator();
      MachineBasicBlock::instr_iterator E = MI.getParent()->instr_end();
      while (++II != E && II->isInsideBundle()) {
        if (&*II == CoreMI)
          continue;
        unsigned Opc = II->getOpcode();
        if (CoreOp->isDef() && Opc != AMDGPU::V_STORE_IDX)
          continue;
        if (CoreOp->isUse() &&
            !(CoreOp->isInternalRead() && Opc == AMDGPU::V_LOAD_IDX))
          continue;
        MachineOperand &DataOp = II->getOperand(
            AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::data_op));
        if (DataOp.getReg() != CoreOp->getReg())
          continue;

        // Replace CoreOp with a new register of the correct width and offset
        size_t ByteSize = AMDGPU::getRegOperandSize(TRI, CoreMI->getDesc(),
                                                    CoreOp->getOperandNo());
        int OffsetIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::offset);
        assert(OffsetIdx != -1 && "Malformed V_LOAD/STORE_IDX instruction");
        unsigned Offset = II->getOperand(OffsetIdx).getImm();
        assert(Offset < ST->getAddressableNumVGPRs() - ByteSize / 4);
        assert(
            !ST->needsAlignedVGPRs() || ByteSize <= 4 ||
            (Offset & 1) == 0 &&
                "Instructions with odd offsets should not have been bundled");
        CoreOp->setReg(
            TRI->getAnyVGPRClassForBitWidth(ByteSize * 8)->getRegister(Offset));
        CoreOp->setIsUndef();
        CoreOp->setIsInternalRead(false);

        NewMode.Ops[I].IdxReg =
            II->getOperand(AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::idx))
                .getReg()
                .asMCReg();

        --II;
        II->getNextNode()->removeFromBundle();
      }
    }

    // VOPM will not read or write the MODE register.
    if (AMDGPU::isVOPMPseudo(CoreMI->getOpcode()))
      continue;

    std::optional<unsigned> MSBits;
    if (CoreOp)
      MSBits = getMSBs(*CoreOp);

#if !defined(NDEBUG)
    if (MSBits.has_value() && Ops2) {
      auto Op2 = TII->getNamedOperand(*CoreMI, Ops2[I]);
      if (Op2) {
        std::optional<unsigned> MSBits2;
        MSBits2 = getMSBs(*Op2);
        if (MSBits2.has_value() && MSBits != MSBits2)
          llvm_unreachable("Invalid VOPD pair was created");
      }
    }
#endif

    if (!MSBits.has_value() && Ops2) {
      CoreOp = TII->getNamedOperand(*CoreMI, Ops2[I]);
      if (CoreOp)
        MSBits = getMSBs(*CoreOp);
    }

    if (!MSBits.has_value())
      continue;

    // Skip tied uses of src2 of VOP2, these will be handled along with defs and
    // only vdst bit affects these operands. We cannot skip tied uses of VOP3,
    // these uses are real even if must match the vdst.
    if (Ops[I] == AMDGPU::OpName::src2 && !CoreOp->isDef() &&
        CoreOp->isTied() &&
        (SIInstrInfo::isVOP2(*CoreMI) ||
         (SIInstrInfo::isVOP3(*CoreMI) &&
          TII->hasVALU32BitEncoding(CoreMI->getOpcode()))))
      continue;

    NewMode.Ops[I].MSBits = MSBits.value();

    if (ST->hasVGPRIndexingRegisters()) {
      if (!NewMode.Ops[I].IdxReg)
        NewMode.Ops[I].IdxReg = AMDGPU::IDX0;
    }
  }

  if (IsBundleWithGPRIndexing) {
    MachineBasicBlock::instr_iterator Start(MI.getIterator());
    for (MachineBasicBlock::instr_iterator I = ++Start,
                                           E = MI.getParent()->instr_end();
         I != E && I->isBundledWithPred(); ++I) {
      assert(I->getOpcode() != AMDGPU::V_LOAD_IDX &&
             I->getOpcode() != AMDGPU::V_STORE_IDX &&
             "Failed to lower bundled index instruction");
      I->unbundleFromPred();
    }
    MI.eraseFromParent();
  }
  setMode(NewMode, CoreMI);
}

bool AMDGPULowerVGPREncoding::runOnMachineInstr(
    MachineBasicBlock::instr_iterator &MII) {
  MachineInstr &MI = *MII;
  unsigned Opc = MI.getOpcode();
  if (Opc == AMDGPU::V_LOAD_IDX || Opc == AMDGPU::V_STORE_IDX) {
    lowerIDX(MII);
    return true;
  }

  MachineInstr *CoreMI = SIInstrInfo::bundleWithGPRIndexing(MI);
  auto Ops = AMDGPU::getVGPRLoweringOperandTables(CoreMI ? CoreMI->getDesc()
                                                         : MI.getDesc());
  if (Ops.first) {
    if (CoreMI)
      MII = MachineBasicBlock::instr_iterator(CoreMI);
    lowerInstrOrBundle(MI, CoreMI, Ops.first, Ops.second);
    return true;
  }
  if (CoreMI) {
    lowerMovBundle(MI, *CoreMI, MII);
    return true;
  }
  assert(!TII->hasVGPRUses(MI) || MI.isMetaInstruction() || MI.isPseudo());

  return false;
}

MachineInstr *AMDGPULowerVGPREncoding::handleClause(MachineInstr *I) {
  if (!ClauseRemaining)
    return I;

  // A clause cannot start with a special instruction, place it right before
  // the clause.
  if (ClauseRemaining == ClauseLen) {
    I = Clause->getPrevNode();
    assert(I->isBundle());
    return I;
  }

  // If a clause defines breaks each group cannot start with a mode change.
  // just drop the clause.
  if (ClauseBreaks) {
    Clause->eraseFromBundle();
    ClauseRemaining = 0;
    return I;
  }

  // Otherwise adjust a number of instructions in the clause if it fits.
  // If it does not clause will just become shorter. Since the length
  // recorded in the clause is one less, increment the length after the
  // update. Note that SIMM16[5:0] must be 1-62, not 0 or 63.
  if (ClauseLen < 63)
    Clause->getOperand(0).setImm(ClauseLen | (ClauseBreaks << 8));

  ++ClauseLen;

  return I;
}

bool AMDGPULowerVGPREncoding::runOnMachineFunction(MachineFunction &MF) {
  ST = &MF.getSubtarget<GCNSubtarget>();
  if (!ST->has1024AddressableVGPRs())
    return false;

  TII = ST->getInstrInfo();
  TRI = ST->getRegisterInfo();
  MFI = MF.getInfo<SIMachineFunctionInfo>();

  bool Changed = false;
  ClauseLen = ClauseRemaining = 0;
  CurrentMode = {};
  CurrentModeKnown = true;
  for (auto &MBB : MF) {
    MostRecentModeSet = nullptr;

    MachineBasicBlock::instr_iterator I = MBB.instr_begin();
    MachineBasicBlock::instr_iterator E = MBB.instr_end();
    for (; I != E; ++I) {
      if (I->isMetaInstruction())
        continue;

      if (I->isTerminator() || I->isCall()) {
        if (I->getOpcode() == AMDGPU::S_ENDPGM ||
            I->getOpcode() == AMDGPU::S_ENDPGM_SAVED) {
          CurrentMode = {};
          CurrentModeKnown = true;
        } else
          resetMode(&*I);
        continue;
      }

      if (I->isInlineAsm()) {
        if (TII->hasVGPRUses(*I))
          resetMode(&*I);
        continue;
      }

      if (I->getOpcode() == AMDGPU::S_CLAUSE) {
        assert(!ClauseRemaining && "Nested clauses are not supported");
        ClauseLen = I->getOperand(0).getImm();
        ClauseBreaks = (ClauseLen >> 8) & 15;
        ClauseLen = ClauseRemaining = (ClauseLen & 63) + 1;
        Clause = &*I;
        continue;
      }

      Changed |= runOnMachineInstr(I);

      if (ClauseRemaining)
        --ClauseRemaining;
    }

    // If we're falling through to a block that has at least one other
    // predecessor, we no longer know the mode.
    MachineBasicBlock *Next = MBB.getNextNode();
    if (Next && Next->pred_size() >= 2 &&
        llvm::is_contained(Next->predecessors(), &MBB)) {
      if (CurrentMode.isSet())
        CurrentModeKnown = false;
    }
  }

  return Changed;
}

} // namespace

char AMDGPULowerVGPREncoding::ID = 0;

char &llvm::AMDGPULowerVGPREncodingID = AMDGPULowerVGPREncoding::ID;

INITIALIZE_PASS(AMDGPULowerVGPREncoding, DEBUG_TYPE,
                "AMDGPU Lower VGPR Encoding", false, false)
