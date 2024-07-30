//===- AMDGPULowerVGPREncoding.cpp - Insert s_delay_alu instructions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Lower VGPRs above first 256 on gfx1210.
///
/// The pass scans used VGPRs and inserts S_SET_VGPR_MSB instructions to switch
/// VGPR addressing mode. The mode change is effective until the next change.
/// This instruction provides high bits of a VGPR address for four of the
/// operands: vdst, src0, src1, and src2, or other 4 operands depending on the
/// instruction encoding. If bits are set they are added as MSB to the
/// corresponding operand VGPR number.
///
/// There is no need to replace actual register operands because encoding of the
/// high and low VGPRs is the same. I.e. v0 has the encoding 0x100, so does v256.
/// v1 has the encoding 0x101 and v257 has the same encoding. So high VGPRs will
/// survive until actual encoding and will result in a same actual bit encoding.
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

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/PackedVector.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-lower-vgpr-encoding"

namespace {

class AMDGPULowerVGPREncoding : public MachineFunctionPass {
  static constexpr unsigned OpNum = 4;
  static constexpr unsigned BitsPerField = 2;
  static constexpr unsigned NumFields = 4;
  using ModeType = PackedVector<unsigned, BitsPerField,
                                std::bitset<BitsPerField * NumFields>>;

  class ModeTy : public ModeType {
  public:
    // bitset constructor will set all bits to zero
    ModeTy() : ModeType(0) {}

    operator int64_t() const { return raw_bits().to_ulong(); }
  };

public:
  static char ID;

  AMDGPULowerVGPREncoding() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;

  /// Current mode bits.
  ModeTy Mode;

  /// Number of current hard clause instructions.
  unsigned ClauseLen;

  /// Number of hard clause instructions remaining.
  unsigned ClauseRemaining;

  /// Clause group breaks.
  unsigned ClauseBreaks;

  /// Last hard clause instruction.
  MachineBasicBlock::instr_iterator Clause;

  /// Insert mode change before \p I. \returns true if mode was changed.
  bool setMode(ModeTy NewMode, MachineBasicBlock::instr_iterator I);

  /// Reset mode to default.
  void resetMode(MachineBasicBlock::instr_iterator I) { setMode(ModeTy(), I); }

  /// If \p MO references VGPRs, return the MSBs. Otherwise, return nullopt.
  std::optional<unsigned> getMSBs(const MachineOperand &MO) const;

  /// Handle single \p MI. \return true if changed.
  bool runOnMachineInstr(MachineInstr &MI);

  /// Handle single \p MI given \p Ops operands bit mapping.
  /// \return true if changed. Optionally takes second array \p Ops2.
  /// If provided and an operand from \p Ops is not a VGPR, then \p Ops2
  /// is checked.
  bool runOnMachineInstr(MachineInstr &MI, const unsigned Ops[OpNum],
                         const unsigned *Ops2 = nullptr);

  /// Check if an instruction \p I is within a clause and returns a suitable
  /// iterator to insert mode change. It may also modify the S_CLAUSE
  /// instruction to extend it or drop the clause if it cannot be adjusted.
  MachineBasicBlock::instr_iterator
  handleClause(MachineBasicBlock::instr_iterator I);
};

bool AMDGPULowerVGPREncoding::setMode(ModeTy NewMode,
                                      MachineBasicBlock::instr_iterator I) {
  if (NewMode == Mode)
    return false;

  I = handleClause(I);
  BuildMI(*I->getParent(), I, nullptr, TII->get(AMDGPU::S_SET_VGPR_MSB))
      .addImm(NewMode);

  Mode = NewMode;
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

bool AMDGPULowerVGPREncoding::runOnMachineInstr(MachineInstr &MI,
                                                const unsigned Ops[OpNum],
                                                const unsigned *Ops2) {
  ModeTy NewMode = Mode;

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

    // Keep unused bits from the old mask to minimize switches.
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
  }

  return setMode(NewMode, MI.getIterator());
}

bool AMDGPULowerVGPREncoding::runOnMachineInstr(MachineInstr &MI) {
  auto Ops = AMDGPU::getVGPRLoweringOperandTables(MI.getDesc());

  if (Ops.first)
    return runOnMachineInstr(MI, Ops.first, Ops.second);

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
    I = std::prev(Clause);
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
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.has1024AddressableVGPRs())
    return false;

  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();

  bool Changed = false;
  ClauseLen = ClauseRemaining = 0;
  Mode.reset();
  for (auto &MBB : MF) {
    for (auto &MI : llvm::make_early_inc_range(MBB.instrs())) {
      if (MI.isMetaInstruction())
        continue;

      if (MI.isTerminator() || MI.isCall()) {
        if (MI.getOpcode() == AMDGPU::S_ENDPGM ||
            MI.getOpcode() == AMDGPU::S_ENDPGM_SAVED)
          Mode.reset();
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
        Clause = MI.getIterator();
        continue;
      }

      Changed |= runOnMachineInstr(MI);

      if (ClauseRemaining)
        --ClauseRemaining;
    }
  }

  return Changed;
}

} // namespace

char AMDGPULowerVGPREncoding::ID = 0;

char &llvm::AMDGPULowerVGPREncodingID = AMDGPULowerVGPREncoding::ID;

INITIALIZE_PASS(AMDGPULowerVGPREncoding, DEBUG_TYPE,
                "AMDGPU Lower VGPR Encoding", false, false)
