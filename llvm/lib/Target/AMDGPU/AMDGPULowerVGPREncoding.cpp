//===- AMDGPULowerVGPREncoding.cpp - Set MODE & Lower idx Pseudos ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Lower VGPRs above first 256 on gfx1210+.
/// Also lowers dynamic VGPR indexing pseudo instructions to subtarget
/// instructions.
///
/// The pass scans used VGPRs and inserts S_SET_VGPR_MSB instructions on
/// gfx1210 (or S_SET_VGPR_FRAMES on gfx13+) to switch VGPR addressing mode. The
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
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/PackedVector.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-lower-vgpr-encoding"

namespace {

class AMDGPULowerVGPREncoding : public MachineFunctionPass {
  static constexpr unsigned OpNum = 4;
  static constexpr unsigned BitsPerField = 2;
  static constexpr unsigned NumFields = 8;
  static constexpr unsigned FieldMask = (1 << BitsPerField) - 1;
  using ModeType = PackedVector<unsigned, BitsPerField,
                                std::bitset<BitsPerField * NumFields>>;

  /// GFX1210 layout: [src0 msb, src1 msb, src2 msb, dst msb]
  /// GFX13   layout: [src0 idx_sel, src1 idx_sel, src2 idx_sel, dst idx_sel,
  ///                  src0 msb, src1 msb, src2 msb, dst msb]
  class ModeTy : public ModeType {
  public:
    // bitset constructor will set all bits to zero
    ModeTy() : ModeType(0) {}

    operator int64_t() const { return raw_bits().to_ulong(); }

    static ModeTy fullMask() {
      ModeTy M;
      M.raw_bits().flip();
      return M;
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

  /// Current mode bits.
  ModeTy CurrentMode;

  /// Current mask of mode bits that instructions since MostRecentModeSet care
  /// about.
  ModeTy CurrentMask;

  /// Number of current hard clause instructions.
  unsigned ClauseLen;

  /// Number of hard clause instructions remaining.
  unsigned ClauseRemaining;

  /// Clause group breaks.
  unsigned ClauseBreaks;

  /// Last hard clause instruction.
  MachineInstr *Clause;

  /// Insert mode change before \p I. \returns true if mode was changed.
  bool setMode(ModeTy NewMode, ModeTy Mask, MachineInstr *I);

  /// Reset mode to default.
  void resetMode(MachineInstr *I) { setMode(ModeTy(), ModeTy::fullMask(), I); }

  /// If \p MO references VGPRs, return the MSBs. Otherwise, return nullopt.
  std::optional<unsigned> getMSBs(const MachineOperand &MO) const;

  /// Handle single \p MI. \return true if changed.
  /// Updates MII to point to the last instruction processed (existing or newly
  /// inserted) for mode update, so that upon increment in runOnMachineFunction
  /// MII is the correct value to process next.
  bool runOnMachineInstr(MachineBasicBlock::instr_iterator &MII);

  /// Compute and set the mode and mode mask for a single \p MI given \p Ops
  /// operands bit mapping. If MI is BUNDLE, lowers the bundle by replacing the
  /// operands of CoreMI with the dynamic indices from the bundle. Optionally
  /// takes second array \p Ops2 for VOPD. If provided and an operand from \p
  /// Ops is not a VGPR, then \p Ops2 is checked.
  void lowerInstrOrBundle(MachineInstr &MI, MachineInstr *CoreMI,
                          const unsigned Ops[OpNum],
                          const unsigned *Ops2 = nullptr);

  /// Lower V_LOAD/STORE_IDX to one or several V_MOV_B32 instructions and update
  /// the mode and mask. MII is updated to point to the last V_MOV inserted.
  void lowerIDX(MachineBasicBlock::instr_iterator &MII);

  /// Lower bundles which only contain V_LOAD/STORE_IDX, as would be used
  /// to move data, and update the mode and mask. MII is updated to point to the
  /// last V_MOV inserted.
  void lowerMovBundle(MachineInstr &MI, MachineInstr *CoreMI,
                      MachineBasicBlock::instr_iterator &MII);

  /// Check if an instruction \p I is within a clause and returns a suitable
  /// iterator to insert mode change. It may also modify the S_CLAUSE
  /// instruction to extend it or drop the clause if it cannot be adjusted.
  MachineInstr *handleClause(MachineInstr *I);
};

bool AMDGPULowerVGPREncoding::setMode(ModeTy NewMode, ModeTy Mask,
                                      MachineInstr *I) {
  assert((NewMode.raw_bits() & ~Mask.raw_bits()).none());

  if (CurrentModeKnown) {
    auto Delta = NewMode.raw_bits() ^ CurrentMode.raw_bits();

    if ((Delta & Mask.raw_bits()).none()) {
      CurrentMask |= Mask;
      return false;
    }

    if (MostRecentModeSet && (Delta & CurrentMask.raw_bits()).none()) {
      CurrentMode |= NewMode;
      CurrentMask |= Mask;

      MostRecentModeSet->getOperand(0).setImm(CurrentMode);
      return true;
    }
  }

  I = handleClause(I);
  MostRecentModeSet =
      BuildMI(*I->getParent(), I, {}, TII->get(ST->hasVGPRIndexingRegisters() ? AMDGPU::S_SET_VGPR_FRAMES
                                                  : AMDGPU::S_SET_VGPR_MSB))
          .addImm(NewMode);

  CurrentMode = NewMode;
  CurrentMask = Mask;
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
    MachineInstr &MI, MachineInstr *CoreMI,
    MachineBasicBlock::instr_iterator &MII) {
  assert(CoreMI->getOpcode() == AMDGPU::V_STORE_IDX);

  // The RC in MachineInstrDesc for V_LOAD/STORE_IDX can contain many
  // possible register sizes, we need to use the register info instead.
  const auto *TRI = ST->getRegisterInfo();
  MachineOperand &DataOp = MI.getOperand(
      AMDGPU::getNamedOperandIdx(AMDGPU::V_STORE_IDX, AMDGPU::OpName::data_op));
  const auto *RC = TRI->getPhysRegBaseClass(DataOp.getReg());
  auto Size = TRI->getRegSizeInBits(*RC);
  if (Size % 32 != 0)
    report_fatal_error(
        "TODO-GFX13 Support lowering non-multiple-of-32-bit sizes for "
        "V_LOAD/STORE_IDX");

  MachineInstr *LoadMI = CoreMI->getPrevNode();

#if !defined(NDEBUG)
  // Check if the value loaded by V_LOAD_IDX is the same as stored by
  // V_STORE_IDX
  assert(LoadMI->getOpcode() == AMDGPU::V_LOAD_IDX &&
         "V_LOAD_IDX + V_STORE_IDX Bundle was not created correctly");
  MachineOperand &LoadDataOp = MI.getOperand(
      AMDGPU::getNamedOperandIdx(AMDGPU::V_LOAD_IDX, AMDGPU::OpName::data_op));
  const auto *LoadRC = TRI->getPhysRegBaseClass(DataOp.getReg());
  auto LoadSize = TRI->getRegSizeInBits(*LoadRC);
  unsigned StoreDataRegNum = TRI->getHWRegIndex(DataOp.getReg());
  unsigned LoadDataRegNum = TRI->getHWRegIndex(LoadDataOp.getReg());
  assert(LoadSize == Size && LoadDataRegNum == StoreDataRegNum &&
         "V_LOAD_IDX + V_STORE_IDX Bundle was not created correctly");
#endif

  Register StoreIdxReg = CoreMI
                             ->getOperand(AMDGPU::getNamedOperandIdx(
                                 AMDGPU::V_STORE_IDX, AMDGPU::OpName::idx))
                             .getReg();
  unsigned StoreIdxRegVal = StoreIdxReg - AMDGPU::IDX0;
  int StoreOffsetIdx =
      AMDGPU::getNamedOperandIdx(AMDGPU::V_STORE_IDX, AMDGPU::OpName::offset);
  unsigned StoreOffset = CoreMI->getOperand(StoreOffsetIdx).getImm();

  Register LoadIdxReg = LoadMI
                            ->getOperand(AMDGPU::getNamedOperandIdx(
                                AMDGPU::V_LOAD_IDX, AMDGPU::OpName::idx))
                            .getReg();
  unsigned LoadIdxRegVal = LoadIdxReg - AMDGPU::IDX0;
  int LoadOffsetIdx =
      AMDGPU::getNamedOperandIdx(AMDGPU::V_LOAD_IDX, AMDGPU::OpName::offset);
  unsigned LoadOffset = LoadMI->getOperand(LoadOffsetIdx).getImm();

  ModeTy NewMode, Mask;
  using namespace llvm::AMDGPU::VGPRIndexMode;
  NewMode[ID_SRC0] = LoadIdxRegVal;
  NewMode[ID_DST] = StoreIdxRegVal;
  Mask[ID_SRC0] = FieldMask;
  Mask[ID_DST] = FieldMask;
  Mask[ID_SRC0 + OpNum] = FieldMask;
  Mask[ID_DST + OpNum] = FieldMask;

  unsigned MaxVGPR = ST->getAddressableNumVGPRs() - 1;
  const MCInstrDesc &OpDesc = TII->get(AMDGPU::V_MOV_B32_e32);
  for (unsigned I = 0; I < Size / 32; ++I) {
    unsigned CurLoadOffset = (LoadOffset + I) & MaxVGPR;
    unsigned CurStoreOffset = (StoreOffset + I) & MaxVGPR;

    auto MIB = BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), OpDesc);
    MIB.addDef(AMDGPU::VGPR0 + CurStoreOffset)
        .addUse(AMDGPU::VGPR0 + CurLoadOffset, RegState::Undef);
    NewMode[ID_SRC0 + OpNum] = CurLoadOffset >> 8;
    NewMode[ID_DST + OpNum] = CurStoreOffset >> 8;

    setMode(NewMode, Mask, &*MIB);
    MII = MachineBasicBlock::instr_iterator(MIB);
  }
  LoadMI->removeFromBundle();
  CoreMI->removeFromBundle();
  MI.eraseFromParent();
}

void AMDGPULowerVGPREncoding::lowerIDX(MachineBasicBlock::instr_iterator &MII) {
  MachineInstr &MI = *MII;
  unsigned Opc = MI.getOpcode();
  bool IsLoad = Opc == AMDGPU::V_LOAD_IDX;

  // The RC in MachineInstrDesc for V_LOAD/STORE_IDX can contain many
  // possible register sizes, we need to use the register info instead.
  const auto *TRI = ST->getRegisterInfo();
  MachineOperand &DataOp =
      MI.getOperand(AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::data_op));
  const auto *RC = TRI->getPhysRegBaseClass(DataOp.getReg());
  auto Size = TRI->getRegSizeInBits(*RC);
  assert((Size % 32) == 0 &&
         "TODO-GFX13 Support lowering non-multiple-of-32-bit sizes for "
         "V_LOAD/STORE_IDX");
  unsigned DataRegNum = TRI->getHWRegIndex(DataOp.getReg());

  Register IdxReg =
      MI.getOperand(AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::idx))
          .getReg();
  unsigned IdxRegVal = IdxReg - AMDGPU::IDX0;

  int OffsetIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::offset);
  assert(OffsetIdx != -1 && "Malformed V_LOAD/STORE_IDX instruction");
  unsigned Offset = MI.getOperand(OffsetIdx).getImm();

  ModeTy NewMode, Mask;
  using namespace llvm::AMDGPU::VGPRIndexMode;
  if (IsLoad)
    NewMode[ID_SRC0] = IdxRegVal;
  else
    NewMode[ID_DST] = IdxRegVal;
  Mask[ID_SRC0] = FieldMask;
  Mask[ID_DST] = FieldMask;
  Mask[ID_SRC0 + OpNum] = FieldMask;
  Mask[ID_DST + OpNum] = FieldMask;

  unsigned MaxVGPR = ST->getAddressableNumVGPRs() - 1;
  const MCInstrDesc &OpDesc = TII->get(AMDGPU::V_MOV_B32_e32);
  for (unsigned i = 0; i < Size / 32; ++i) {
    unsigned CurOffset = (Offset + i) & MaxVGPR;
    unsigned CurData = DataRegNum + i;

    auto MIB = BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), OpDesc);
    if (IsLoad) {
      MIB.addDef(AMDGPU::VGPR0 + CurData)
          .addUse(AMDGPU::VGPR0 + CurOffset, RegState::Undef);

      NewMode[ID_SRC0 + OpNum] = CurOffset >> 8;
      NewMode[ID_DST + OpNum] = CurData >> 8;
    } else {
      MIB.addDef(AMDGPU::VGPR0 + CurOffset)
          .addUse(AMDGPU::VGPR0 + CurData, DataOp.isUndef() ? RegState::Undef : 0);
      NewMode[ID_SRC0 + OpNum] = CurData >> 8;
      NewMode[ID_DST + OpNum] = CurOffset >> 8;
    }

    setMode(NewMode, Mask, &*MIB);
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
  ModeTy NewMode, Mask;

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
        if (CoreOp->isUse() && Opc != AMDGPU::V_LOAD_IDX)
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
        CoreOp->setReg(
            TRI->getAnyVGPRClassForBitWidth(ByteSize * 8)->getRegister(Offset));
        CoreOp->setIsUndef();
        CoreOp->setIsInternalRead(false);

        Register IdxReg =
            II->getOperand(AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::idx))
                .getReg();
        unsigned IdxRegVal = IdxReg - AMDGPU::IDX0;
        NewMode[I] = IdxRegVal;

        --II;
        II->getNextNode()->removeFromBundle();
      }
    }

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

    unsigned IdxOffset = ST->hasVGPRIndexingRegisters() ? 4 : 0;
    NewMode[I + IdxOffset] = MSBits.value();
    Mask[I + IdxOffset] = FieldMask;

    if (ST->hasVGPRIndexingRegisters())
      Mask[I] = FieldMask;
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
  setMode(NewMode, Mask, CoreMI);
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
    lowerMovBundle(MI, CoreMI, MII);
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
  CurrentMode.reset();
  CurrentMask.reset();
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
          CurrentMode.reset();
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
      if (CurrentMode.raw_bits().any())
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
