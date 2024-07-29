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
/// instructions. V_LOAD/STORE_IDX are be lowered to V_MOV_B32, and the index
/// registers they use are encoded in a preceding update the index select bits
/// in MODE using S_SET_VGPR_FRAMES.
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

  /// If \p MO is a high VGPR \returns offset MSBs and a corresponding low VGPR.
  /// If \p MO is a low VGPR \returns 0 and that register.
  /// Otherwise \returns 0 and NoRegister.
  std::pair<unsigned, MCRegister>
  getLowRegister(const MachineOperand &MO) const;

  /// Handle single \p MI. \return true if changed.
  bool runOnMachineInstr(MachineInstr &MI);

  /// Handle single \p MI given \p Ops operands bit mapping.
  /// Optionally takes second array \p Ops2.
  /// If provided and an operand from \p Ops is not a VGPR, then \p Ops2
  /// is checked.
  /// \return true if any VGPRs are used in MI.
  bool computeModeForMSBs(ModeTy &NewMode, MachineInstr &MI,
                          const unsigned Ops[OpNum],
                          const unsigned *Ops2 = nullptr);

  /// Abstraction between which index register is used and where the signifying
  /// bits are stored.
  inline void updateModeForIDX(ModeTy &Mode, const ModeTy &Mask);

  bool lowerIDX(ModeTy &NewMode, MachineInstr &MI);

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
  BuildMI(*I->getParent(), I, nullptr,
          TII->get(ST->hasVGPRIndexingRegisters() ? AMDGPU::S_SET_VGPR_FRAMES
                                                  : AMDGPU::S_SET_VGPR_MSB))
      .addImm(NewMode);

  Mode = NewMode;
  return true;
}

std::pair<unsigned, MCRegister>
AMDGPULowerVGPREncoding::getLowRegister(const MachineOperand &MO) const {
  if (!MO.isReg())
    return std::pair(0, MCRegister());

  MCRegister Reg = MO.getReg();
  const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(Reg);
  if (!RC || !TRI->isVGPRClass(RC))
    return std::pair(0, MCRegister());

  unsigned Idx = TRI->getHWRegIndex(Reg);
  if (Idx <= 255)
    return std::pair(0, Reg);

  unsigned Align = TRI->getRegClassAlignmentNumBits(RC) / 32;
  assert(Align == 1 || Align == 2);
  unsigned RegNum = (Idx & 0xff) >> (Align - 1);
  return std::pair(Idx >> 8, RC->getRegister(RegNum));
}

void AMDGPULowerVGPREncoding::updateModeForIDX(ModeTy &Mode,
                                               const ModeTy &Mask) {
  for (unsigned I = 0; I < OpNum; ++I) {
    Mode[I] = Mask[I];
  }
}

bool AMDGPULowerVGPREncoding::lowerIDX(ModeTy &NewMode, MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  // The RC in MachineInstrDesc for V_LOAD/STORE_IDX can contain many
  // possible register sizes, we need to use the register info instead.
  const auto *TRI = ST->getRegisterInfo();
  auto Reg =
      MI.getOperand(AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::data_op))
          .getReg();
  const auto *RC = TRI->getPhysRegBaseClass(Reg);
  auto Size = TRI->getRegSizeInBits(*RC);
  assert(Size == 32 &&
         "TODO-GFX13 Support lowering non-32-bit sizes for V_LOAD/STORE_IDX");
  Register IdxReg =
      MI.getOperand(AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::idx))
          .getReg();
  unsigned IdxRegVal = IdxReg - AMDGPU::IDX0;
  bool IsLoad = Opc == AMDGPU::V_LOAD_IDX;
  // src0, src1, src2, dst
  ModeTy UsedIdxRegs;
  if (IsLoad)
    UsedIdxRegs[0] = IdxRegVal;
  else
    UsedIdxRegs[3] = IdxRegVal;
  updateModeForIDX(NewMode, UsedIdxRegs);

  // Synthesize the offset VGPR
  int OffsetIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::offset);
  assert(OffsetIdx != -1 && "Malformed V_LOAD/STORE_IDX instruction");
  unsigned Offset = MI.getOperand(OffsetIdx).getImm();
  assert(Offset < MFI->getLaneSharedVGPRSize() && "Offset out of range");
  // Laneshared allocation starts at VGPR0
  unsigned OffsetVGPRReg = AMDGPU::VGPR0 + Offset;
  MachineOperand OffsetVGPR = MachineOperand::CreateReg(OffsetVGPRReg, 0);

  // Insert V_MOV
  const MCInstrDesc &OpDesc = TII->get(AMDGPU::V_MOV_B32_e32);
  auto MIB = BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), OpDesc);
  if (IsLoad) {
    // IsUndef because there may be no writes to the register from this function
    OffsetVGPR.setIsUndef(true);
    MIB.add(MI.getOperand(0)) // loaded value
        .add(OffsetVGPR);
  } else {
    OffsetVGPR.setIsDef(true);
    // clang-format off
    MIB.add(OffsetVGPR)
        .add(MI.getOperand(0)); // stored value
    // clang-format on
  }

  // An earlier pass should have already inserted the SET_GPR_IDX before the
  // V_LOAD/STORE_IDX

  MI.eraseFromParent();

  auto Ops = AMDGPU::getVGPRLoweringOperandTables(MIB->getDesc());
  assert(Ops.first);
  computeModeForMSBs(NewMode, *MIB, Ops.first, Ops.second);

  return setMode(NewMode, MIB->getIterator());
}

bool AMDGPULowerVGPREncoding::computeModeForMSBs(ModeTy &Mode, MachineInstr &MI,
                                                 const unsigned Ops[OpNum],
                                                 const unsigned *Ops2) {
  bool RegUsed = false;
  ModeTy NewMode = Mode;
  for (unsigned I = 0; I < OpNum; ++I) {
    MachineOperand *Op = TII->getNamedOperand(MI, Ops[I]);

    MCRegister Reg;
    unsigned MSBits;
    if (Op)
      std::tie(MSBits, Reg) = getLowRegister(*Op);

#if !defined(NDEBUG)
    if (Reg && Ops2) {
      auto Op2 = TII->getNamedOperand(MI, Ops2[I]);
      if (Op2) {
        unsigned MSBits2;
        MCRegister Reg2;
        std::tie(MSBits2, Reg2) = getLowRegister(*Op2);
        if (Reg2 && MSBits != MSBits2)
          llvm_unreachable("Invalid VOPD pair was created");
      }
    }
#endif

    if (!Reg && Ops2) {
      Op = TII->getNamedOperand(MI, Ops2[I]);
      if (Op)
        std::tie(MSBits, Reg) = getLowRegister(*Op);
    }

    // Keep unused bits from the old mask to minimize switches.
    if (!Reg)
      continue;

    // Skip tied uses of src2 of VOP2, these will be handled along with defs and
    // only vdst bit affects these operands. We cannot skip tied uses of VOP3,
    // these uses are real even if must match the vdst.
    if (Ops[I] == AMDGPU::OpName::src2 && !Op->isDef() && Op->isTied() &&
        (SIInstrInfo::isVOP2(MI) ||
         (SIInstrInfo::isVOP3(MI) &&
          TII->hasVALU32BitEncoding(MI.getOpcode()))))
      continue;

    // If any registers are used, even if MSBs are unchanged, we need to update
    // idx select bits.
    RegUsed = true;

    unsigned IdxOffset = ST->hasVGPRIndexingRegisters() ? 4 : 0;
    NewMode[I + IdxOffset] = MSBits;
  }

  Mode = NewMode;
  return RegUsed;
}

bool AMDGPULowerVGPREncoding::runOnMachineInstr(MachineInstr &MI) {
  ModeTy NewMode = Mode;
  unsigned Opc = MI.getOpcode();
  // TODO-GFX13 Support BUNDLEs with multiple V_LOAD/STORE_IDX instructions
  if (Opc == AMDGPU::V_LOAD_IDX || Opc == AMDGPU::V_STORE_IDX)
    return lowerIDX(NewMode, MI);

  auto Ops = AMDGPU::getVGPRLoweringOperandTables(MI.getDesc());
  if (Ops.first) {
    bool VGPRAreUsed = computeModeForMSBs(NewMode, MI, Ops.first, Ops.second);
    if (VGPRAreUsed && ST->hasVGPRIndexingRegisters()) {
      // Idx registers are used, and we should reset them to 0.
      ModeTy IdxRegsUsed;
      updateModeForIDX(NewMode, IdxRegsUsed);
    }
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
  ST = &MF.getSubtarget<GCNSubtarget>();
  if (!ST->has1024AddressableVGPRs())
    return false;

  TII = ST->getInstrInfo();
  TRI = ST->getRegisterInfo();
  MFI = MF.getInfo<SIMachineFunctionInfo>();

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
