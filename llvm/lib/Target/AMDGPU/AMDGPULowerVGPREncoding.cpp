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
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/PackedVector.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-lower-vgpr-encoding"

namespace {

class AMDGPULowerVGPREncoding {
  static constexpr unsigned OpNum = 4;
  static constexpr unsigned BitsPerField = 2;
  static constexpr unsigned NumFields = 4;
  static constexpr unsigned FieldMask = (1 << BitsPerField) - 1;
  using ModeType = PackedVector<unsigned, BitsPerField,
                                std::bitset<BitsPerField * NumFields>>;

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
  bool run(MachineFunction &MF);

private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;

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
  bool runOnMachineInstr(MachineInstr &MI);

  /// Compute the mode and mode mask for a single \p MI given \p Ops operands
  /// bit mapping. Optionally takes second array \p Ops2 for VOPD.
  /// If provided and an operand from \p Ops is not a VGPR, then \p Ops2
  /// is checked.
  void computeMode(ModeTy &NewMode, ModeTy &Mask, MachineInstr &MI,
                   const AMDGPU::OpName Ops[OpNum],
                   const AMDGPU::OpName *Ops2 = nullptr);

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
      BuildMI(*I->getParent(), I, {}, TII->get(AMDGPU::S_SET_VGPR_MSB))
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

void AMDGPULowerVGPREncoding::computeMode(ModeTy &NewMode, ModeTy &Mask,
                                          MachineInstr &MI,
                                          const AMDGPU::OpName Ops[OpNum],
                                          const AMDGPU::OpName *Ops2) {
  NewMode = {};
  Mask = {};

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

    NewMode[I] = MSBits.value();
    Mask[I] = FieldMask;
  }
}

bool AMDGPULowerVGPREncoding::runOnMachineInstr(MachineInstr &MI) {
  auto Ops = AMDGPU::getVGPRLoweringOperandTables(MI.getDesc());
  if (Ops.first) {
    ModeTy NewMode, Mask;
    computeMode(NewMode, Mask, MI, Ops.first, Ops.second);
    return setMode(NewMode, Mask, &MI);
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

bool AMDGPULowerVGPREncoding::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.has1024AddressableVGPRs())
    return false;

  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();

  bool Changed = false;
  ClauseLen = ClauseRemaining = 0;
  CurrentMode.reset();
  CurrentMask.reset();
  CurrentModeKnown = true;
  for (auto &MBB : MF) {
    MostRecentModeSet = nullptr;

    for (auto &MI : llvm::make_early_inc_range(MBB.instrs())) {
      if (MI.isMetaInstruction())
        continue;

      if (MI.isTerminator() || MI.isCall()) {
        if (MI.getOpcode() == AMDGPU::S_ENDPGM ||
            MI.getOpcode() == AMDGPU::S_ENDPGM_SAVED) {
          CurrentMode.reset();
          CurrentModeKnown = true;
        } else
          resetMode(&MI);
        continue;
      }

      if (MI.isInlineAsm()) {
        if (TII->hasVGPRUses(MI))
          resetMode(&MI);
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

      Changed |= runOnMachineInstr(MI);

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

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
