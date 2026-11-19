//===- SIFoldPackedMoves.cpp - Combine physical VGPR copies ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SIFoldPackedMoves.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/MathExtras.h"
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "si-fold-packed-moves"

STATISTIC(NumPackedMoves, "Number of copy groups combined into packed moves");
STATISTIC(NumWideCopies, "Number of copy groups combined into wide copies");

namespace {

class SIFoldPackedMoves {
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  const MCInstrDesc &PackedMove;
  const BitVector Reserved;

  MCRegister getPair(MCRegister Reg, unsigned OpIdx) const;
  MCRegister getCopyDstPair(const MachineInstr &MI) const;
  bool isBarrier(const MachineInstr &MI) const;
  void emitMove(MachineInstr &Anchor, MCRegister Dst, MCRegister SrcLo,
                MCRegister SrcHi, MCRegister PairLo, MCRegister PairHi) const;
  bool combine(MachineInstr &First, MachineInstr &Last, MCRegister Dst) const;
  bool foldBlock(MachineBasicBlock &MBB) const;

  explicit SIFoldPackedMoves(MachineFunction &MF, const GCNSubtarget &ST)
      : TII(*ST.getInstrInfo()), TRI(*ST.getRegisterInfo()),
        PackedMove(TII.get(AMDGPU::V_PK_MOV_B32)),
        Reserved(TRI.getReservedRegs(MF)) {}

public:
  static bool run(MachineFunction &MF);
};

class SIFoldPackedMovesLegacy : public MachineFunctionPass {
public:
  static char ID;
  SIFoldPackedMovesLegacy() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "SI Fold Packed Moves"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    return !skipFunction(MF.getFunction()) && SIFoldPackedMoves::run(MF);
  }
};

} // namespace

INITIALIZE_PASS(SIFoldPackedMovesLegacy, DEBUG_TYPE, "SI Fold Packed Moves",
                false, false)

char SIFoldPackedMovesLegacy::ID = 0;
char &llvm::SIFoldPackedMovesLegacyID = SIFoldPackedMovesLegacy::ID;

MCRegister SIFoldPackedMoves::getPair(MCRegister Reg, unsigned OpIdx) const {
  unsigned Half = TRI.getHWRegIndex(Reg) & 1 ? AMDGPU::sub1 : AMDGPU::sub0;
  MCRegister Pair =
      TRI.getMatchingSuperReg(Reg, Half, TII.getRegClass(PackedMove, OpIdx));
  if (!Pair || Reserved.test(Reg) || Reserved.test(Pair))
    return MCRegister();
  return Pair;
}

MCRegister SIFoldPackedMoves::getCopyDstPair(const MachineInstr &MI) const {
  if (!MI.isCopy() || MI.getFlags() || MI.peekDebugInstrNum() ||
      !MI.memoperands_empty() || MI.getPCSections() ||
      MI.getHeapAllocMarker() || MI.getMMRAMetadata() ||
      MI.getDeactivationSymbol() || MI.getCFIType())
    return MCRegister();
  const MachineOperand &Dst = MI.getOperand(0), &Src = MI.getOperand(1);
  if (!Dst.isReg() || !Dst.isDef() || !Src.isReg() || !Src.isUse() ||
      Dst.isImplicit() || Src.isImplicit() || Dst.getSubReg() ||
      Src.getSubReg() || Dst.isUndef() || Dst.isDead() || Src.isUndef() ||
      Dst.isEarlyClobber() || Dst.isInternalRead() || Src.isInternalRead() ||
      Dst.isTied() || Src.isTied() || Dst.getReg() == Src.getReg() ||
      !AMDGPU::VGPR_32RegClass.contains(Dst.getReg()) ||
      !(AMDGPU::VGPR_32RegClass.contains(Src.getReg()) ||
        AMDGPU::SGPR_32RegClass.contains(Src.getReg())))
    return MCRegister();
  for (const MachineOperand &MO : llvm::drop_begin(MI.operands(), 2))
    if (!MO.isReg() || !MO.isImplicit() || !MO.isUse() || MO.isUndef() ||
        MO.getReg() != AMDGPU::EXEC || MO.isInternalRead() || MO.isTied())
      return MCRegister();
  return getPair(Dst.getReg().asMCReg(), 0);
}

bool SIFoldPackedMoves::isBarrier(const MachineInstr &MI) const {
  return MI.isMetaInstruction() || MI.isBundled() || MI.isInlineAsm() ||
         MI.isCall() || MI.hasUnmodeledSideEffects() ||
         MI.getPreInstrSymbol() || MI.getPostInstrSymbol() ||
         TII.isSchedulingBoundary(MI, MI.getParent(), *MI.getMF()) ||
         llvm::any_of(MI.operands(),
                      [](const MachineOperand &MO) { return MO.isRegMask(); });
}

bool SIFoldPackedMoves::combine(MachineInstr &First, MachineInstr &Last,
                                MCRegister Dst) const {
  MCRegister FirstDst = First.getOperand(0).getReg().asMCReg();
  MCRegister FirstSrc = First.getOperand(1).getReg().asMCReg();
  MCRegister LastDst = Last.getOperand(0).getReg().asMCReg();
  MCRegister LastSrc = Last.getOperand(1).getReg().asMCReg();
  if (FirstDst == LastDst || TRI.regsOverlap(FirstDst, LastSrc))
    return false;

  bool FirstIsLow = FirstDst == TRI.getSubReg(Dst, AMDGPU::sub0);
  MCRegister SrcLo = FirstIsLow ? FirstSrc : LastSrc;
  MCRegister SrcHi = FirstIsLow ? LastSrc : FirstSrc;
  MCRegister PairLo = getPair(SrcLo, 2), PairHi = getPair(SrcHi, 4);
  if (!PairLo || !PairHi ||
      (PairLo != PairHi && AMDGPU::SGPR_32RegClass.contains(SrcLo) &&
       AMDGPU::SGPR_32RegClass.contains(SrcHi) &&
       TII.getSubtarget().getConstantBusLimit(PackedMove.getOpcode()) < 2))
    return false;

  auto Between = make_range(std::next(First.getIterator()), Last.getIterator());
  auto CanMove = [&](MCRegister MovingDst, MCRegister MovingSrc) {
    return llvm::none_of(Between, [&](const MachineInstr &MI) {
      return MI.readsRegister(MovingDst, &TRI) ||
             MI.modifiesRegister(MovingDst, &TRI) ||
             MI.modifiesRegister(MovingSrc, &TRI);
    });
  };
  bool Sink = CanMove(FirstDst, FirstSrc);
  if (!Sink && !CanMove(LastDst, LastSrc))
    return false;
  if (Sink) {
    for (MachineInstr &MI : Between)
      for (MachineOperand &MO : MI.operands())
        if (MO.isReg() && MO.isUse() && MO.isKill() &&
            MO.getReg().isPhysical() && TRI.regsOverlap(MO.getReg(), FirstSrc))
          MO.setIsKill(false);
  }

  emitMove(Sink ? Last : First, Dst, SrcLo, SrcHi, PairLo, PairHi);
  First.eraseFromParent();
  Last.eraseFromParent();
  return true;
}

void SIFoldPackedMoves::emitMove(MachineInstr &Anchor, MCRegister Dst,
                                 MCRegister SrcLo, MCRegister SrcHi,
                                 MCRegister PairLo, MCRegister PairHi) const {
  bool LoHigh = SrcLo == TRI.getSubReg(PairLo, AMDGPU::sub1);
  bool HiHigh = SrcHi == TRI.getSubReg(PairHi, AMDGPU::sub1);
  if (PairLo == PairHi && !LoHigh && HiHigh) {
    BuildMI(*Anchor.getParent(), Anchor, Anchor.getDebugLoc(),
            TII.get(TargetOpcode::COPY), Dst)
        .addReg(PairLo)
        .addReg(AMDGPU::EXEC, RegState::Implicit);
    ++NumWideCopies;
  } else {
    auto Mods = [](bool High) {
      return SISrcMods::OP_SEL_1 | (High ? SISrcMods::OP_SEL_0 : 0);
    };
    BuildMI(*Anchor.getParent(), Anchor, Anchor.getDebugLoc(), PackedMove, Dst)
        .addImm(Mods(LoHigh))
        .addReg(PairLo, RegState::Undef)
        .addImm(Mods(HiHigh))
        .addReg(PairHi, RegState::Undef)
        .addImm(0)
        .addImm(0)
        .addImm(0)
        .addImm(0)
        .addImm(0)
        .addReg(SrcLo, RegState::Implicit)
        .addReg(SrcHi, RegState::Implicit);
    ++NumPackedMoves;
  }
}

bool SIFoldPackedMoves::foldBlock(MachineBasicBlock &MBB) const {
  DenseMap<MCRegister, MachineInstr *> Pending;
  bool Changed = false;
  for (MachineInstr &MI : make_early_inc_range(MBB)) {
    if (isBarrier(MI)) {
      Pending.clear();
      continue;
    }
    if (MCRegister Dst = getCopyDstPair(MI)) {
      MachineInstr *First = std::exchange(Pending[Dst], &MI);
      if (First && combine(*First, MI, Dst)) {
        Pending.erase(Dst);
        Changed = true;
      }
    }
  }
  return Changed;
}

/// Indexed operands need not name the physical words they access. The mode can
/// survive a basic-block boundary without an explicit MODE use on the COPYs.
static bool mayEnableVGPRIndexing(const MachineInstr &ModeDef) {
  switch (ModeDef.getOpcode()) {
  case AMDGPU::S_SETREG_B32:
  case AMDGPU::S_SETREG_IMM32_B32: {
    auto [ID, Offset, Width] =
        AMDGPU::Hwreg::HwregEncoding::decode(ModeDef.getOperand(1).getImm());
    return ID == AMDGPU::Hwreg::ID_MODE &&
           ((maskTrailingOnes<uint32_t>(Width) << Offset) &
            AMDGPU::Hwreg::GPR_IDX_EN_MASK);
  }
  case AMDGPU::S_SETREG_B32_mode:
  case AMDGPU::S_SETREG_IMM32_B32_mode:
  case AMDGPU::S_ROUND_MODE:
  case AMDGPU::S_DENORM_MODE:
  case AMDGPU::S_SET_GPR_IDX_OFF:
    return false;
  default:
    return true;
  }
}

bool SIFoldPackedMoves::run(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasPkMovB32() || !MF.getProperties().hasNoVRegs() ||
      llvm::any_of(MF.getRegInfo().def_instructions(AMDGPU::MODE),
                   mayEnableVGPRIndexing))
    return false;
  SIFoldPackedMoves Folder(MF, ST);
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= Folder.foldBlock(MBB);
  return Changed;
}

PreservedAnalyses SIFoldPackedMovesPass::run(MachineFunction &MF,
                                             MachineFunctionAnalysisManager &) {
  if (MF.getFunction().hasOptNone() || !SIFoldPackedMoves::run(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
