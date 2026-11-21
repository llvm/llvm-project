//===- SIFoldPackedMoves.cpp - Fold VGPR copy pairs into packed moves -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// After register allocation, fold two 32-bit COPYs that write the two halves
/// of a VGPR pair into one instruction:
///
///   $vgpr0 = COPY $vgpr5
///   $vgpr1 = COPY $sgpr2
/// ->
///   $vgpr0_vgpr1 = V_PK_MOV_B32 <high half of> $vgpr4_vgpr5,
///                               <low half of> $sgpr2_sgpr3
///
/// V_PK_MOV_B32 takes each half of its result from either half of the matching
/// 64-bit source. When both halves come from one source pair in order, a 64-bit
/// COPY is emitted instead and copyPhysReg chooses the instruction.
///
/// The two COPYs need not be adjacent. The folded instruction replaces the
/// second COPY if the first one can be sunk down to it, or else the first COPY
/// if the second one can be hoisted up to it.
//
//===----------------------------------------------------------------------===//

#include "SIFoldPackedMoves.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "si-fold-packed-moves"

STATISTIC(NumPackedMoves, "Number of COPY pairs folded into V_PK_MOV_B32");
STATISTIC(NumWideCopies, "Number of COPY pairs folded into 64-bit COPYs");

namespace {

class SIFoldPackedMoves {
  MachineFunction &MF;
  const GCNSubtarget &ST;
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  const MachineRegisterInfo &MRI;

  MCRegister getPair(MCRegister Reg, AMDGPU::OpName PkMovOp) const;
  MCRegister getFoldableDstPair(const MachineInstr &MI) const;
  bool isBarrier(const MachineInstr &MI) const;
  bool
  canCross(MCRegister Dst, MCRegister Src,
           iterator_range<MachineBasicBlock::instr_iterator> Between) const;
  void updateMovedCopyMetadata(
      MCRegister FirstSrc, MCRegister MovedDst, bool Sink,
      iterator_range<MachineBasicBlock::instr_iterator> Between) const;
  void emitMove(MachineInstr &InsertPt, const DebugLoc &DL, MCRegister DstPair,
                MCRegister SrcLo, MCRegister SrcHi, MCRegister SrcLoPair,
                MCRegister SrcHiPair) const;
  bool tryFold(MachineInstr &First, MachineInstr &Second,
               MCRegister DstPair) const;

public:
  explicit SIFoldPackedMoves(MachineFunction &MF)
      : MF(MF), ST(MF.getSubtarget<GCNSubtarget>()), TII(*ST.getInstrInfo()),
        TRI(*ST.getRegisterInfo()), MRI(MF.getRegInfo()) {}

  bool run();
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

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    return SIFoldPackedMoves(MF).run();
  }
};

} // end anonymous namespace

INITIALIZE_PASS(SIFoldPackedMovesLegacy, DEBUG_TYPE, "SI Fold Packed Moves",
                false, false)

char SIFoldPackedMovesLegacy::ID = 0;

char &llvm::SIFoldPackedMovesLegacyID = SIFoldPackedMovesLegacy::ID;

/// Return the 64-bit register that has \p Reg as one of its halves and that
/// V_PK_MOV_B32 accepts as operand \p PkMovOp, if it may be used.
MCRegister SIFoldPackedMoves::getPair(MCRegister Reg,
                                      AMDGPU::OpName PkMovOp) const {
  int OpIdx = AMDGPU::getNamedOperandIdx(AMDGPU::V_PK_MOV_B32, PkMovOp);
  const TargetRegisterClass *RC =
      TII.getRegClass(TII.get(AMDGPU::V_PK_MOV_B32), OpIdx);

  // Only the even-aligned pair is tried. SGPR pairs always are, and so must
  // VGPR pairs be on every subtarget that has this instruction.
  unsigned Half = TRI.getHWRegIndex(Reg) % 2 ? AMDGPU::sub1 : AMDGPU::sub0;
  MCRegister Pair = TRI.getMatchingSuperReg(Reg, Half, RC);

  // Reserved registers can be read or written by instructions that do not name
  // them, so there is no telling what a copy of one can be moved across.
  if (!Pair || MRI.isReserved(Pair))
    return MCRegister();
  return Pair;
}

/// If \p MI is a COPY that can be folded with a COPY into the other half of
/// the same VGPR pair, return that pair.
MCRegister SIFoldPackedMoves::getFoldableDstPair(const MachineInstr &MI) const {
  if (!MI.isCopy())
    return MCRegister();

  const MachineOperand &Dst = MI.getOperand(0);
  const MachineOperand &Src = MI.getOperand(1);
  if (!AMDGPU::VGPR_32RegClass.contains(Dst.getReg()) ||
      (!AMDGPU::VGPR_32RegClass.contains(Src.getReg()) &&
       !AMDGPU::SGPR_32RegClass.contains(Src.getReg())))
    return MCRegister();

  // ExpandPostRAPseudos removes these instead of emitting a move.
  if (Dst.isDead() || Src.isUndef() || Dst.getReg() == Src.getReg())
    return MCRegister();

  // Any implicit operand other than the EXEC use of a VGPR copy would have to
  // be carried over to the folded instruction.
  for (const MachineOperand &MO : MI.implicit_operands())
    if (!MO.isReg() || !MO.isUse() || MO.getReg() != AMDGPU::EXEC)
      return MCRegister();

  return getPair(Dst.getReg(), AMDGPU::OpName::vdst);
}

/// Whether no COPY may be moved across \p MI, whatever its operands are.
bool SIFoldPackedMoves::isBarrier(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  case AMDGPU::IGLP_OPT:
  case AMDGPU::SCHED_BARRIER:
  case AMDGPU::SCHED_GROUP_BARRIER:
    // These direct how the instructions around them are scheduled.
    return true;
  default:
    break;
  }

  // Stay clear of calls, and of inline assembly and bundles, whose contents
  // are not examined.
  if (MI.isCall() || MI.isInlineAsm() || MI.isBundled())
    return true;

  // A VGPR copy depends on EXEC and on the VGPR indexing mode without naming
  // either. Every instruction that changes one is a scheduling boundary.
  return TII.isSchedulingBoundary(MI, MI.getParent(), MF);
}

/// Whether a COPY from \p Src to \p Dst can be moved across \p Between.
bool SIFoldPackedMoves::canCross(
    MCRegister Dst, MCRegister Src,
    iterator_range<MachineBasicBlock::instr_iterator> Between) const {
  return none_of(instructionsWithoutDebug(Between.begin(), Between.end()),
                 [&](const MachineInstr &MI) {
                   return MI.readsRegister(Dst, &TRI) ||
                          MI.modifiesRegister(Dst, &TRI) ||
                          MI.modifiesRegister(Src, &TRI);
                 });
}

/// Update kill flags and debug values in \p Between for a moved COPY.
void SIFoldPackedMoves::updateMovedCopyMetadata(
    MCRegister FirstSrc, MCRegister MovedDst, bool Sink,
    iterator_range<MachineBasicBlock::instr_iterator> Between) const {
  for (MachineInstr &MI : Between) {
    // The source of First is now read after MI.
    if (Sink)
      MI.clearRegisterKills(FirstSrc, &TRI);

    // MovedDst no longer holds the value that MI describes.
    if (MI.isDebugValue() &&
        any_of(MI.debug_operands(), [&](const MachineOperand &MO) {
          return MO.isReg() && MO.getReg().isPhysical() &&
                 TRI.regsOverlap(MO.getReg(), MovedDst);
        }))
      MI.setDebugValueUndef();
  }
}

/// Emit the folded 64-bit COPY or packed move before \p InsertPt.
void SIFoldPackedMoves::emitMove(MachineInstr &InsertPt, const DebugLoc &DL,
                                 MCRegister DstPair, MCRegister SrcLo,
                                 MCRegister SrcHi, MCRegister SrcLoPair,
                                 MCRegister SrcHiPair) const {
  MachineBasicBlock &MBB = *InsertPt.getParent();
  bool LoIsHiHalf = SrcLo == TRI.getSubReg(SrcLoPair, AMDGPU::sub1);
  bool HiIsHiHalf = SrcHi == TRI.getSubReg(SrcHiPair, AMDGPU::sub1);
  if (SrcLoPair == SrcHiPair && !LoIsHiHalf && HiIsHiHalf) {
    // The two halves of one source pair in order are a 64-bit copy.
    BuildMI(MBB, InsertPt, DL, TII.get(TargetOpcode::COPY), DstPair)
        .addReg(SrcLoPair)
        .addReg(AMDGPU::EXEC, RegState::Implicit);
    ++NumWideCopies;
  } else {
    // OP_SEL_0 selects the high half of a source. OP_SEL_1 does not affect the
    // result and is only set to keep op_sel_hi from being printed.
    auto SrcMods = [](bool HiHalf) {
      return SISrcMods::OP_SEL_1 | (HiHalf ? SISrcMods::OP_SEL_0 : 0);
    };
    // Only the selected half of each source pair is read, and the other half
    // need not even be defined: the pair is undef and the half an implicit use.
    BuildMI(MBB, InsertPt, DL, TII.get(AMDGPU::V_PK_MOV_B32), DstPair)
        .addImm(SrcMods(LoIsHiHalf))
        .addReg(SrcLoPair, RegState::Undef)
        .addImm(SrcMods(HiIsHiHalf))
        .addReg(SrcHiPair, RegState::Undef)
        .addImm(0) // clamp
        .addImm(0) // op_sel
        .addImm(0) // op_sel_hi
        .addImm(0) // neg_lo
        .addImm(0) // neg_hi
        .addReg(SrcLo, RegState::Implicit)
        .addReg(SrcHi, RegState::Implicit);
    ++NumPackedMoves;
  }
}

/// Try to replace \p First and the later \p Second, which are COPYs into halves
/// of \p DstPair, with a single instruction.
bool SIFoldPackedMoves::tryFold(MachineInstr &First, MachineInstr &Second,
                                MCRegister DstPair) const {
  MCRegister FirstDst = First.getOperand(0).getReg();
  MCRegister FirstSrc = First.getOperand(1).getReg();
  MCRegister SecondDst = Second.getOperand(0).getReg();
  MCRegister SecondSrc = Second.getOperand(1).getReg();

  // Two COPYs into the same half are not a pair.
  if (FirstDst == SecondDst)
    return false;

  // The folded instruction reads both sources before it writes either half, so
  // Second must not read the half that First wrote. Second overwriting the
  // source of First is fine.
  if (SecondSrc == FirstDst)
    return false;

  bool FirstIsLo = FirstDst == TRI.getSubReg(DstPair, AMDGPU::sub0);
  MCRegister SrcLo = FirstIsLo ? FirstSrc : SecondSrc;
  MCRegister SrcHi = FirstIsLo ? SecondSrc : FirstSrc;
  MCRegister SrcLoPair = getPair(SrcLo, AMDGPU::OpName::src0);
  MCRegister SrcHiPair = getPair(SrcHi, AMDGPU::OpName::src1);
  if (!SrcLoPair || !SrcHiPair)
    return false;

  // Each distinct SGPR pair takes a slot on the constant bus.
  auto IsSGPR = [](MCRegister Reg) {
    return AMDGPU::SGPR_32RegClass.contains(Reg);
  };
  unsigned NumSGPRPairs =
      SrcLoPair == SrcHiPair ? IsSGPR(SrcLo) : IsSGPR(SrcLo) + IsSGPR(SrcHi);
  if (NumSGPRPairs > ST.getConstantBusLimit(AMDGPU::V_PK_MOV_B32))
    return false;

  auto Between =
      make_range(std::next(First.getIterator()), Second.getIterator());

  // Sink First down to Second, or else hoist Second up to First.
  bool Sink = canCross(FirstDst, FirstSrc, Between);
  if (!Sink && !canCross(SecondDst, SecondSrc, Between))
    return false;
  MachineInstr &InsertPt = Sink ? Second : First;
  MCRegister MovedDst = Sink ? FirstDst : SecondDst;

  updateMovedCopyMetadata(FirstSrc, MovedDst, Sink, Between);

  DebugLoc DL =
      DebugLoc::getMergedLocation(First.getDebugLoc(), Second.getDebugLoc());
  emitMove(InsertPt, DL, DstPair, SrcLo, SrcHi, SrcLoPair, SrcHiPair);

  First.eraseFromParent();
  Second.eraseFromParent();
  return true;
}

bool SIFoldPackedMoves::run() {
  if (!ST.hasPkMovB32())
    return false;

  bool Changed = false;
  // For each VGPR pair, the latest COPY into one half of it that a COPY into
  // the other half could still be folded with.
  DenseMap<MCRegister, MachineInstr *> Pending;
  for (MachineBasicBlock &MBB : MF) {
    Pending.clear();
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (MI.isDebugInstr())
        continue;
      if (isBarrier(MI)) {
        Pending.clear();
        continue;
      }

      MCRegister DstPair = getFoldableDstPair(MI);
      if (!DstPair)
        continue;
      MachineInstr *&Waiting = Pending[DstPair];
      if (Waiting && tryFold(*Waiting, MI, DstPair)) {
        Waiting = nullptr;
        Changed = true;
      } else {
        Waiting = &MI;
      }
    }
  }
  return Changed;
}

PreservedAnalyses SIFoldPackedMovesPass::run(MachineFunction &MF,
                                             MachineFunctionAnalysisManager &) {
  MFPropsModifier _(*this, MF);

  if (MF.getFunction().hasOptNone() || !SIFoldPackedMoves(MF).run())
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
