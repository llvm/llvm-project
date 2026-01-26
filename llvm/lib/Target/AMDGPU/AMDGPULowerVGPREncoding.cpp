//===- AMDGPULowerVGPREncoding.cpp - lower VGPRs above v255 ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Lower VGPRs above first 256 on gfx1250.
///
/// The pass scans used VGPRs and inserts S_SET_VGPR_MSB instructions to switch
/// VGPR addressing mode. The mode change is effective until the next change.
/// This instruction provides high bits of a VGPR address for four of the
/// operands: vdst, src0, src1, and src2, or other 4 operands depending on the
/// instruction encoding. If bits are set they are added as MSB to the
/// corresponding operand VGPR number.
///
/// There is no need to replace actual register operands because encoding of the
/// high and low VGPRs is the same. I.e. v0 has the encoding 0x100, so does
/// v256. v1 has the encoding 0x101 and v257 has the same encoding. So high
/// VGPRs will survive until actual encoding and will result in a same actual
/// bit encoding.
///
/// As a result the pass only inserts S_SET_VGPR_MSB to provide an actual offset
/// to a VGPR address of the subseqent instructions. The InstPrinter will take
/// care of the printing a low VGPR instead of a high one. In prinicple this
/// shall be viable to print actual high VGPR numbers, but that would disagree
/// with a disasm printing and create a situation where asm text is not
/// deterministic.
///
/// This pass creates a convention where non-fall through basic blocks shall
/// start with all 4 MSBs zero. Otherwise a disassembly would not be readable.
/// An optimization here is possible but deemed not desirable because of the
/// readbility concerns.
///
/// Consequentially the ABI is set to expect all 4 MSBs to be zero on entry.
/// The pass must run very late in the pipeline to make sure no changes to VGPR
/// operands will be made after it.
//
//===----------------------------------------------------------------------===//

#include "AMDGPULowerVGPREncoding.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-lower-vgpr-encoding"

namespace {

class AMDGPULowerVGPREncoding {
  static constexpr unsigned OpNum = 4;
  static constexpr unsigned BitsPerField = 2;
  static constexpr unsigned NumFields = 4;
  static constexpr unsigned ModeWidth = NumFields * BitsPerField;
  static constexpr unsigned ModeMask = (1 << ModeWidth) - 1;
  static constexpr unsigned VGPRMSBShift =
      llvm::countr_zero_constexpr<unsigned>(AMDGPU::Hwreg::DST_VGPR_MSB);

  struct OpMode {
    // No MSBs set means they are not required to be of a particular value.
    std::optional<unsigned> MSBits;

    bool update(const OpMode &New, bool &Rewritten) {
      bool Updated = false;
      if (New.MSBits) {
        if (*New.MSBits != MSBits.value_or(0)) {
          Updated = true;
          Rewritten |= MSBits.has_value();
        }
        MSBits = New.MSBits;
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

    unsigned encode() const {
      // Layout: [src0 msb, src1 msb, src2 msb, dst msb].
      unsigned V = 0;
      for (const auto &[I, Op] : enumerate(Ops))
        V |= Op.MSBits.value_or(0) << (I * 2);
      return V;
    }
  };

public:
  bool run(MachineFunction &MF);

private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;

  // Current basic block.
  MachineBasicBlock *MBB;

  /// Most recent s_set_* instruction.
  MachineInstr *MostRecentModeSet;

  /// Current mode bits.
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
  bool setMode(ModeTy NewMode, MachineBasicBlock::instr_iterator I);

  /// Reset mode to default.
  void resetMode(MachineBasicBlock::instr_iterator I) {
    ModeTy Mode;
    for (OpMode &Op : Mode.Ops)
      Op.MSBits = 0;
    setMode(Mode, I);
  }

  /// If \p MO references VGPRs, return the MSBs. Otherwise, return nullopt.
  std::optional<unsigned> getMSBs(const MachineOperand &MO) const;

  /// Handle single \p MI. \return true if changed.
  bool runOnMachineInstr(MachineInstr &MI);

  /// Compute the mode for a single \p MI given \p Ops operands
  /// bit mapping. Optionally takes second array \p Ops2 for VOPD.
  /// If provided and an operand from \p Ops is not a VGPR, then \p Ops2
  /// is checked.
  void computeMode(ModeTy &NewMode, MachineInstr &MI,
                   const AMDGPU::OpName Ops[OpNum],
                   const AMDGPU::OpName *Ops2 = nullptr);

  /// Check if an instruction \p I is within a clause and returns a suitable
  /// iterator to insert mode change. It may also modify the S_CLAUSE
  /// instruction to extend it or drop the clause if it cannot be adjusted.
  MachineBasicBlock::instr_iterator
  handleClause(MachineBasicBlock::instr_iterator I);

  /// Check if an instruction \p I is immediately after another program state
  /// instruction which it cannot coissue with. If so, insert before that
  /// instruction to encourage more coissuing.
  MachineBasicBlock::instr_iterator
  handleCoissue(MachineBasicBlock::instr_iterator I);

  /// Handle S_SETREG_IMM32_B32 targeting MODE register. On certain hardware,
  /// this instruction clobbers VGPR MSB bits[12:19], so we need to restore
  /// the current mode. \returns true if the instruction was modified or a
  /// new one was inserted.
  bool handleSetregMode(MachineInstr &MI);

  /// Update bits[12:19] of the imm operand in S_SETREG_IMM32_B32 to contain
  /// the VGPR MSB mode value. \returns true if the immediate was changed.
  bool updateSetregModeImm(MachineInstr &MI, int64_t ModeValue);
};

bool AMDGPULowerVGPREncoding::setMode(ModeTy NewMode,
                                      MachineBasicBlock::instr_iterator I) {
  // Record previous mode into high 8 bits of the immediate.
  int64_t OldModeBits = CurrentMode.encode() << ModeWidth;

  bool Rewritten = false;
  if (!CurrentMode.update(NewMode, Rewritten))
    return false;

  if (MostRecentModeSet && !Rewritten) {
    // Update MostRecentModeSet with the new mode. It can be either
    // S_SET_VGPR_MSB or S_SETREG_IMM32_B32 (with Size <= 12).
    if (MostRecentModeSet->getOpcode() == AMDGPU::S_SET_VGPR_MSB) {
      MachineOperand &Op = MostRecentModeSet->getOperand(0);
      // Carry old mode bits from the existing instruction.
      int64_t OldModeBits = Op.getImm() & (ModeMask << ModeWidth);
      Op.setImm(CurrentMode.encode() | OldModeBits);
    } else {
      assert(MostRecentModeSet->getOpcode() == AMDGPU::S_SETREG_IMM32_B32 &&
             "unexpected MostRecentModeSet opcode");
      updateSetregModeImm(*MostRecentModeSet, CurrentMode.encode());
    }

    return true;
  }

  I = handleClause(I);
  I = handleCoissue(I);
  MostRecentModeSet = BuildMI(*MBB, I, {}, TII->get(AMDGPU::S_SET_VGPR_MSB))
                          .addImm(NewMode.encode() | OldModeBits);

  CurrentMode = NewMode;
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

void AMDGPULowerVGPREncoding::computeMode(ModeTy &NewMode, MachineInstr &MI,
                                          const AMDGPU::OpName Ops[OpNum],
                                          const AMDGPU::OpName *Ops2) {
  NewMode = {};

  for (unsigned I = 0; I < OpNum; ++I) {
    MachineOperand *Op = TII->getNamedOperand(MI, Ops[I]);

    std::optional<unsigned> MSBits;
    if (Op)
      MSBits = getMSBs(*Op);

#if !defined(NDEBUG)
    if (MSBits.has_value() && Ops2) {
      auto Op2 = TII->getNamedOperand(MI, Ops2[I]);
      if (Op2) {
        std::optional<unsigned> MSBits2;
        MSBits2 = getMSBs(*Op2);
        if (MSBits2.has_value() && MSBits != MSBits2)
          llvm_unreachable("Invalid VOPD pair was created");
      }
    }
#endif

    if (!MSBits.has_value() && Ops2) {
      Op = TII->getNamedOperand(MI, Ops2[I]);
      if (Op)
        MSBits = getMSBs(*Op);
    }

    if (!MSBits.has_value())
      continue;

    // Skip tied uses of src2 of VOP2, these will be handled along with defs and
    // only vdst bit affects these operands. We cannot skip tied uses of VOP3,
    // these uses are real even if must match the vdst.
    if (Ops[I] == AMDGPU::OpName::src2 && !Op->isDef() && Op->isTied() &&
        (SIInstrInfo::isVOP2(MI) ||
         (SIInstrInfo::isVOP3(MI) &&
          TII->hasVALU32BitEncoding(MI.getOpcode()))))
      continue;

    NewMode.Ops[I].MSBits = MSBits.value();
  }
}

bool AMDGPULowerVGPREncoding::runOnMachineInstr(MachineInstr &MI) {
  auto Ops = AMDGPU::getVGPRLoweringOperandTables(MI.getDesc());
  if (Ops.first) {
    ModeTy NewMode;
    computeMode(NewMode, MI, Ops.first, Ops.second);
    return setMode(NewMode, MI.getIterator());
  }
  assert(!TII->hasVGPRUses(MI) || MI.isMetaInstruction() || MI.isPseudo());

  return false;
}

MachineBasicBlock::instr_iterator
AMDGPULowerVGPREncoding::handleClause(MachineBasicBlock::instr_iterator I) {
  if (!ClauseRemaining)
    return I;

  // A clause cannot start with a special instruction, place it right before
  // the clause.
  if (ClauseRemaining == ClauseLen) {
    I = Clause->getPrevNode()->getIterator();
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

MachineBasicBlock::instr_iterator
AMDGPULowerVGPREncoding::handleCoissue(MachineBasicBlock::instr_iterator I) {
  if (I.isEnd())
    return I;

  // "Program State instructions" are instructions which are used to control
  // operation of the GPU rather than performing arithmetic. Such instructions
  // have different coissuing rules w.r.t s_set_vgpr_msb.
  auto isProgramStateInstr = [this](MachineInstr *MI) {
    unsigned Opc = MI->getOpcode();
    return TII->isBarrier(Opc) || TII->isWaitcnt(Opc) ||
           Opc == AMDGPU::S_DELAY_ALU;
  };

  while (!I.isEnd() && I != I->getParent()->begin()) {
    auto Prev = std::prev(I);
    if (!isProgramStateInstr(&*Prev))
      return I;
    I = Prev;
  }

  return I;
}

/// Convert mode value from S_SET_VGPR_MSB format to MODE register format.
/// S_SET_VGPR_MSB uses: (src0[0-1], src1[2-3], src2[4-5], dst[6-7])
/// MODE register uses:  (dst[0-1], src0[2-3], src1[4-5], src2[6-7])
/// This is a left rotation by 2 bits on an 8-bit value.
static int64_t convertModeToSetregFormat(int64_t Mode) {
  assert(isUInt<8>(Mode) && "Mode expected to be 8-bit");
  return llvm::rotl<uint8_t>(static_cast<uint8_t>(Mode), /*R=*/2);
}

bool AMDGPULowerVGPREncoding::updateSetregModeImm(MachineInstr &MI,
                                                  int64_t ModeValue) {
  assert(MI.getOpcode() == AMDGPU::S_SETREG_IMM32_B32);

  // Convert from S_SET_VGPR_MSB format to MODE register format
  int64_t SetregMode = convertModeToSetregFormat(ModeValue);

  MachineOperand *ImmOp = TII->getNamedOperand(MI, AMDGPU::OpName::imm);
  int64_t OldImm = ImmOp->getImm();
  int64_t NewImm =
      (OldImm & ~AMDGPU::Hwreg::VGPR_MSB_MASK) | (SetregMode << VGPRMSBShift);
  ImmOp->setImm(NewImm);
  return NewImm != OldImm;
}

bool AMDGPULowerVGPREncoding::handleSetregMode(MachineInstr &MI) {
  using namespace AMDGPU::Hwreg;

  assert(MI.getOpcode() == AMDGPU::S_SETREG_IMM32_B32 &&
         "only S_SETREG_IMM32_B32 needs to be handled");

  MachineOperand *SIMM16Op = TII->getNamedOperand(MI, AMDGPU::OpName::simm16);
  assert(SIMM16Op && "SIMM16Op must be present");

  auto [HwRegId, Offset, Size] = HwregEncoding::decode(SIMM16Op->getImm());
  (void)Offset;
  if (HwRegId != ID_MODE)
    return false;

  int64_t ModeValue = CurrentMode.encode();

  // Case 1: Size <= 12 - the original instruction uses imm32[0:Size-1], so
  // imm32[12:19] is unused. Safe to set imm32[12:19] to the correct VGPR
  // MSBs.
  if (Size <= VGPRMSBShift) {
    // This instruction now acts as MostRecentModeSet so it can be updated if
    // CurrentMode changes via piggybacking.
    MostRecentModeSet = &MI;
    return updateSetregModeImm(MI, ModeValue);
  }

  // Case 2: Size > 12 - the original instruction uses bits beyond 11, so we
  // cannot arbitrarily modify imm32[12:19]. Check if it already matches VGPR
  // MSBs. Note: imm32[12:19] is in MODE register format, while ModeValue is
  // in S_SET_VGPR_MSB format, so we need to convert before comparing.
  MachineOperand *ImmOp = TII->getNamedOperand(MI, AMDGPU::OpName::imm);
  assert(ImmOp && "ImmOp must be present");
  int64_t ImmBits12To19 = (ImmOp->getImm() & VGPR_MSB_MASK) >> VGPRMSBShift;
  int64_t SetregModeValue = convertModeToSetregFormat(ModeValue);
  if (ImmBits12To19 == SetregModeValue) {
    // Already correct, but we must invalidate MostRecentModeSet because this
    // instruction will overwrite mode[12:19]. We can't update this instruction
    // via piggybacking (bits[12:19] are meaningful), so if CurrentMode changes,
    // a new s_set_vgpr_msb will be inserted after this instruction.
    MostRecentModeSet = nullptr;
    return false;
  }

  // imm32[12:19] doesn't match VGPR MSBs - insert s_set_vgpr_msb after
  // the original instruction to restore the correct value.
  MachineBasicBlock::iterator InsertPt = std::next(MI.getIterator());
  MostRecentModeSet = BuildMI(*MBB, InsertPt, MI.getDebugLoc(),
                              TII->get(AMDGPU::S_SET_VGPR_MSB))
                          .addImm(ModeValue);
  return true;
}

bool AMDGPULowerVGPREncoding::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.has1024AddressableVGPRs())
    return false;

  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();

  bool Changed = false;
  ClauseLen = ClauseRemaining = 0;
  CurrentMode = {};
  for (auto &MBB : MF) {
    MostRecentModeSet = nullptr;
    this->MBB = &MBB;

    for (auto &MI : llvm::make_early_inc_range(MBB.instrs())) {
      if (MI.isMetaInstruction())
        continue;

      if (MI.isTerminator() || MI.isCall()) {
        if (MI.getOpcode() == AMDGPU::S_ENDPGM ||
            MI.getOpcode() == AMDGPU::S_ENDPGM_SAVED)
          CurrentMode = {};
        else
          resetMode(MI.getIterator());
        continue;
      }

      if (MI.isInlineAsm()) {
        if (TII->hasVGPRUses(MI))
          resetMode(MI.getIterator());
        continue;
      }

      if (MI.getOpcode() == AMDGPU::S_CLAUSE) {
        assert(!ClauseRemaining && "Nested clauses are not supported");
        ClauseLen = MI.getOperand(0).getImm();
        ClauseBreaks = (ClauseLen >> 8) & 15;
        ClauseLen = ClauseRemaining = (ClauseLen & 63) + 1;
        Clause = &MI;
        continue;
      }

      if (MI.getOpcode() == AMDGPU::S_SETREG_IMM32_B32 &&
          ST.hasSetregVGPRMSBFixup()) {
        Changed |= handleSetregMode(MI);
        continue;
      }

      Changed |= runOnMachineInstr(MI);

      if (ClauseRemaining)
        --ClauseRemaining;
    }

    // Reset the mode if we are falling through.
    resetMode(MBB.instr_end());
  }

  return Changed;
}

class AMDGPULowerVGPREncodingLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPULowerVGPREncodingLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    return AMDGPULowerVGPREncoding().run(MF);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AMDGPULowerVGPREncodingLegacy::ID = 0;

char &llvm::AMDGPULowerVGPREncodingLegacyID = AMDGPULowerVGPREncodingLegacy::ID;

INITIALIZE_PASS(AMDGPULowerVGPREncodingLegacy, DEBUG_TYPE,
                "AMDGPU Lower VGPR Encoding", false, false)

PreservedAnalyses
AMDGPULowerVGPREncodingPass::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  if (!AMDGPULowerVGPREncoding().run(MF))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}
