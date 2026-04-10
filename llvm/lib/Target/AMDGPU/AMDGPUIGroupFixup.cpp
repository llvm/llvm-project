//===-- AMDGPUIGLPUnpack.cpp - AMDGPU IGLP unpack MIR cleanup -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Machine IR cleanup pass that runs after rename-independent-subregs and
// before the pre-RA machine scheduler.
//
// Schedule regions (for IGLP / packed-op analysis) match MachineScheduler:
// getSchedRegions (calls, SIInstrInfo::isSchedulingBoundary, FAKE_USE).
//
// V_PK * F32 unpacking lowers each packed op to two scalar VALUs that define
// the destination register by subregs (e.g. vreg_64: undef %dst.sub0 = …;
// %dst.sub1 = …), matching how wide accumulators are updated in real kernels
// (full wide def, then per-lane subreg writes — see incoming FMHA MIR). A
// single V_PK full-reg def leaves IsSSA set on the function; multiple subreg
// defs require clearing IsSSA. Subreg indices follow the V_PK dest (sub0/sub1
// for vreg_64; composite 64-bit lanes e.g. sub0_sub1 … sub6_sub7 on wide
// vectors via composeSubRegIndices). Unpacked sources use the same compose
// pattern as SIPreEmitPeephole (operand packed subreg composed with sub0/sub1
// for the scalar lane). Undef on the first unpacked def only when the
// destination vreg has no prior def in the block and the destination is not
// also a packed source (otherwise lanes already hold defined values). Emit low
// lane, then high lane (first BuildMI before V_PK is earliest in the block).
//
// COPY cleanup: when a packed source vreg is only used by this V_PK and is
// populated solely by COPYs from another register (e.g. assembling a vreg_64
// from two lanes of a wide vector), fold those COPY sources into the unpacked
// VALUs and erase the now-dead COPY defs.
//
// Same-vreg subreg COPY (e.g. %v.sub1 = COPY %v.sub0): when every read of the
// destination lane is an explicit DstSub use after the COPY, rewrite those to
// SrcSub and remove the COPY. Full-register uses cannot be folded.
//
// Sink: when the COPY cannot be removed, move it down in the same MBB to
// immediately before the first instruction that reads the destination lane
// (or immediately before the first def that blocks sinking past it).
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUIGroupLP.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <iterator>
#include <limits>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-iglp-unpack"

/// Max number of V_PK unpack attempts per schedule sub-region (program order
/// within the region). 0 means unlimited. Resets for each sub-region. Does not
/// affect --amdgpu-enable-iglp-unpack=false (full bypass).
static cl::opt<unsigned> AMDGPUIGLPUnpackMaxVPKPerRegion(
    "amdgpu-iglp-unpack-max-vpk-per-region",
    cl::desc("Unpack at most this many V_PK instructions per schedule region "
             "(0 = unlimited)"),
    cl::init(0), cl::Hidden);

/// After V_PK unpacking, optionally split 64-bit VGPR/AGPR vregs that are only
/// referenced via \p sub0 / \p sub1 into 32-bit vregs. 0 skips this cleanup.
/// Default is UINT_MAX (all eligible vregs). Unlike the V_PK-per-region cap,
/// 0 does not mean unlimited.
static cl::opt<unsigned> AMDGPUIGLPUnpackPostCleanupMaxVRegs(
    "amdgpu-iglp-unpack-post-cleanup-max-vregs",
    cl::desc("Post-unpack: split at most this many 64-bit VGPR/AGPR registers "
             "to 32-bit (0 = skip; default = all)"),
    cl::init(std::numeric_limits<unsigned>::max()), cl::Hidden);

// === AMDGPUIGLP_UNPACK_POLICY ==============================================
//
// kRequireMFMAValuSpacingIGLPOptForRegion
//   When true (default): only schedule sub-regions that contain IGLP_OPT with
//   immediate == MFMAValuSpacingOptID are candidates for V_PK unpack. That
//   pseudo must survive for the pre-RA scheduler (IGroupLP) to apply
//   MFMAValuSpacingOpt.
//   When false: any sub-region with at least one V_PK is a candidate; IGLP_OPT
//   is not required.
//
static constexpr bool kRequireMFMAValuSpacingIGLPOptForRegion = true;
// ============================================================================

namespace {

/// A schedule sub-region (between scheduling boundaries) that has at least one
/// V_PK, and when kRequireMFMAValuSpacingIGLPOptForRegion also an IGLP_OPT with
/// immediate MFMAValuSpacingOptID, for unpack / cleanup.
struct CandidateRegion {
  MachineBasicBlock *MBB = nullptr;
  MachineBasicBlock::iterator Begin;
  MachineBasicBlock::iterator End;
  SmallVector<MachineInstr *, 8> VPKInsts;
};

static bool isVPKOpcode(const MCInstrInfo &II, unsigned Opc) {
  return II.getName(Opc).starts_with("V_PK");
}

/// Log why a single V_PK was not unpacked. Enable with
/// -debug-only=amdgpu-iglp-unpack.
static void debugSkipVPKUnpack(const MachineInstr &MI, const SIInstrInfo *TII,
                               const TargetRegisterInfo *TRI,
                               const char *Reason, unsigned LoSeqIdx = 0,
                               unsigned HiSeqIdx = 0, bool LoRCValid = true,
                               bool HiRCValid = true) {
  LLVM_DEBUG({
    dbgs() << DEBUG_TYPE << ": skip unpack (" << Reason << ")";
    if (MI.getNumOperands() > 0 && MI.getOperand(0).isReg()) {
      dbgs() << " op=" << TII->getName(MI.getOpcode()) << " dst="
             << printReg(MI.getOperand(0).getReg(), TRI,
                         MI.getOperand(0).getSubReg());
    }
    if (LoSeqIdx && TRI) {
      if (const char *N = TRI->getSubRegIndexName(LoSeqIdx))
        dbgs() << " loSub=" << N;
      else
        dbgs() << " loSubIdx=" << LoSeqIdx;
      if (!LoRCValid)
        dbgs() << "[!validRC]";
    }
    if (HiSeqIdx && TRI) {
      if (const char *N = TRI->getSubRegIndexName(HiSeqIdx))
        dbgs() << " hiSub=" << N;
      else
        dbgs() << " hiSubIdx=" << HiSeqIdx;
      if (!HiRCValid)
        dbgs() << "[!validRC]";
    }
    dbgs() << "\n";
  });
}

static bool findCandidateRegion(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator RegionBegin,
                                MachineBasicBlock::iterator RegionEnd,
                                const MCInstrInfo &II, CandidateRegion &Out) {
  Out = CandidateRegion{};
  bool HasMFMAValuSpacingIGLP = false;

  for (MachineBasicBlock::iterator It = RegionBegin; It != RegionEnd; ++It) {
    MachineInstr &MI = *It;
    if (MI.getOpcode() == AMDGPU::IGLP_OPT && MI.getNumOperands() >= 1 &&
        MI.getOperand(0).isImm() &&
        MI.getOperand(0).getImm() ==
            static_cast<int64_t>(AMDGPU::IGLPStrategyID::MFMAValuSpacingOptID))
      HasMFMAValuSpacingIGLP = true;

    if (isVPKOpcode(II, MI.getOpcode()))
      Out.VPKInsts.push_back(&MI);
  }

  if (Out.VPKInsts.empty())
    return false;
  if (kRequireMFMAValuSpacingIGLPOptForRegion && !HasMFMAValuSpacingIGLP)
    return false;

  Out.MBB = &MBB;
  Out.Begin = RegionBegin;
  Out.End = RegionEnd;
  return true;
}

// --- F32 unpack (aligned with SIPreEmitPeephole) ----------------------------

/// Skip unpacking for a schedule sub-region if any instruction has an explicit
/// operand using an allocatable physical register (pre-RA MIR is virtual; phys
/// operands indicate an unusual/late state we do not transform).
static bool
schedRegionHasExplicitAllocatablePhysReg(MachineBasicBlock::iterator Begin,
                                         MachineBasicBlock::iterator End,
                                         const MachineRegisterInfo &MRI) {
  for (MachineBasicBlock::iterator It = Begin; It != End; ++It) {
    for (const MachineOperand &MO : It->operands()) {
      if (!MO.isReg() || !MO.getReg().isPhysical())
        continue;
      if (MO.isImplicit())
        continue;
      if (!MRI.isAllocatable(MO.getReg()))
        continue;
      return true;
    }
  }
  return false;
}

static uint32_t mapToUnpackedOpcode(const MachineInstr &I) {
  switch (I.getOpcode()) {
  case AMDGPU::V_PK_ADD_F32:
  case AMDGPU::V_PK_ADD_F32_gfx12:
    return AMDGPU::V_ADD_F32_e64;
  case AMDGPU::V_PK_MUL_F32:
  case AMDGPU::V_PK_MUL_F32_gfx12:
    return AMDGPU::V_MUL_F32_e64;
  case AMDGPU::V_PK_FMA_F32:
  case AMDGPU::V_PK_FMA_F32_gfx12:
    return AMDGPU::V_FMA_F32_e64;
  default:
    return std::numeric_limits<uint32_t>::max();
  }
}

static bool canUnpackingClobberRegister(const MachineInstr &MI,
                                        const SIInstrInfo *TII,
                                        const SIRegisterInfo *TRI) {
  Register DstReg = MI.getOperand(0).getReg();
  // Virtual unpack lowers to subreg VALU defs; the classic post-RA clobber case
  // only applies to physical destinations.
  if (DstReg.isVirtual())
    return false;

  unsigned OpCode = MI.getOpcode();
  Register UnpackedDstReg = TRI->getSubReg(DstReg, AMDGPU::sub0);

  const MachineOperand *Src0MO = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
  if (Src0MO && Src0MO->isReg()) {
    Register SrcReg0 = Src0MO->getReg();
    unsigned Src0Mods =
        TII->getNamedOperand(MI, AMDGPU::OpName::src0_modifiers)->getImm();
    Register HiSrc0Reg = (Src0Mods & SISrcMods::OP_SEL_1)
                             ? TRI->getSubReg(SrcReg0, AMDGPU::sub1)
                             : TRI->getSubReg(SrcReg0, AMDGPU::sub0);
    if (TRI->regsOverlap(UnpackedDstReg, HiSrc0Reg))
      return true;
  }

  const MachineOperand *Src1MO = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
  if (Src1MO && Src1MO->isReg()) {
    Register SrcReg1 = Src1MO->getReg();
    unsigned Src1Mods =
        TII->getNamedOperand(MI, AMDGPU::OpName::src1_modifiers)->getImm();
    Register HiSrc1Reg = (Src1Mods & SISrcMods::OP_SEL_1)
                             ? TRI->getSubReg(SrcReg1, AMDGPU::sub1)
                             : TRI->getSubReg(SrcReg1, AMDGPU::sub0);
    if (TRI->regsOverlap(UnpackedDstReg, HiSrc1Reg))
      return true;
  }

  if (AMDGPU::hasNamedOperand(OpCode, AMDGPU::OpName::src2)) {
    const MachineOperand *Src2MO =
        TII->getNamedOperand(MI, AMDGPU::OpName::src2);
    if (Src2MO && Src2MO->isReg()) {
      Register SrcReg2 = Src2MO->getReg();
      unsigned Src2Mods =
          TII->getNamedOperand(MI, AMDGPU::OpName::src2_modifiers)->getImm();
      Register HiSrc2Reg = (Src2Mods & SISrcMods::OP_SEL_1)
                               ? TRI->getSubReg(SrcReg2, AMDGPU::sub1)
                               : TRI->getSubReg(SrcReg2, AMDGPU::sub0);
      if (TRI->regsOverlap(UnpackedDstReg, HiSrc2Reg))
        return true;
    }
  }
  return false;
}

/// True if any packed source uses the same vreg as the destination
/// (dst-as-src).
static bool vpkAnySrcUsesDst(const MachineInstr &MI, const SIInstrInfo *TII,
                             Register DstReg) {
  unsigned Opc = MI.getOpcode();
  const MachineOperand *S0 = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
  if (S0 && S0->isReg() && S0->getReg() == DstReg)
    return true;
  const MachineOperand *S1 = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
  if (S1 && S1->isReg() && S1->getReg() == DstReg)
    return true;
  if (AMDGPU::hasNamedOperand(Opc, AMDGPU::OpName::src2)) {
    const MachineOperand *S2 = TII->getNamedOperand(MI, AMDGPU::OpName::src2);
    if (S2 && S2->isReg() && S2->getReg() == DstReg)
      return true;
  }
  return false;
}

/// True if Reg has an explicit def in the same MBB strictly before I.
static bool hasExplicitDefOfRegBefore(const MachineInstr &I, Register Reg) {
  const MachineBasicBlock *MBB = I.getParent();
  for (MachineBasicBlock::const_iterator It = MBB->begin(), E = I.getIterator();
       It != E; ++It) {
    for (const MachineOperand &MO : It->operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg() == Reg)
        return true;
    }
  }
  return false;
}

/// True if every non-debug use of R is on \p VPK (same instruction may use R
/// more than once, e.g. src0 and src1).
static bool isVirtualSrcRegOnlyUsedByThisVPK(Register R, MachineInstr &VPK,
                                             MachineRegisterInfo &MRI) {
  if (!R.isVirtual())
    return false;
  if (MRI.use_nodbg_empty(R))
    return false;
  for (MachineInstr &U : MRI.use_nodbg_instructions(R)) {
    if (&U != &VPK)
      return false;
  }
  return true;
}

/// If a COPY in the same MBB strictly before VPK defines SrcReg with subreg
/// NeedSubIdx from a register source, return true and set OutReg/OutSub.
static bool tryFoldThroughSameBlockCopyDef(const MachineInstr &VPK,
                                           Register SrcReg, unsigned NeedSubIdx,
                                           Register &OutReg, unsigned &OutSub) {
  const MachineBasicBlock *MBB = VPK.getParent();
  for (MachineBasicBlock::const_iterator It = MBB->begin(),
                                         E = VPK.getIterator();
       It != E; ++It) {
    if (It->getOpcode() != AMDGPU::COPY)
      continue;
    const MachineOperand &DefMO = It->getOperand(0);
    if (!DefMO.isReg() || !DefMO.isDef() || DefMO.getReg() != SrcReg)
      continue;
    if (DefMO.getSubReg() != NeedSubIdx)
      continue;
    const MachineOperand &SrcMO = It->getOperand(1);
    if (!SrcMO.isReg())
      return false;
    OutReg = SrcMO.getReg();
    OutSub = SrcMO.getSubReg();
    return true;
  }
  return false;
}

static void addOperandAndMods(MachineInstrBuilder &NewMI, unsigned SrcMods,
                              bool IsHiBits, const MachineOperand &SrcMO,
                              Register DstReg, Register LaneSrcBase,
                              unsigned DstPackSub, MachineInstr &VPK,
                              const SIRegisterInfo *TRI) {
  unsigned NewSrcMods = 0;
  unsigned NegModifier = IsHiBits ? SISrcMods::NEG_HI : SISrcMods::NEG;
  unsigned OpSelModifier = IsHiBits ? SISrcMods::OP_SEL_1 : SISrcMods::OP_SEL_0;
  if (SrcMods & NegModifier)
    NewSrcMods |= SISrcMods::NEG;
  NewMI.addImm(NewSrcMods);
  if (SrcMO.isImm()) {
    NewMI.addImm(SrcMO.getImm());
    return;
  }
  Register OrigSrcReg = SrcMO.getReg();
  unsigned SrcPackSub = SrcMO.getSubReg();

  const bool UseHiOfPair = (SrcMods & OpSelModifier) != 0;
  const unsigned PairLane = UseHiOfPair ? AMDGPU::sub1 : AMDGPU::sub0;

  // Packed subreg on the operand (e.g. sub6_sub7); when src is the same
  // super-register as dst (dst-as-src), MO may omit the subreg — use the V_PK
  // destination packed subreg.
  unsigned BasePack = SrcPackSub;
  if (!BasePack && OrigSrcReg == DstReg)
    BasePack = DstPackSub;

  unsigned FinalSubIdx =
      BasePack ? TRI->composeSubRegIndices(BasePack, PairLane) : PairLane;

  Register SrcReg = OrigSrcReg;
  if (OrigSrcReg.isVirtual() && OrigSrcReg != DstReg) {
    Register FoldReg;
    unsigned FoldSub = 0;
    if (tryFoldThroughSameBlockCopyDef(VPK, OrigSrcReg, FinalSubIdx, FoldReg,
                                       FoldSub) &&
        FoldSub) {
      // COPY source subreg is the exact read for this packed lane; use it
      // directly for scalar lanes (<=32b). Wider composed subregs need
      // compose(FoldSub, PairLane).
      SrcReg = FoldReg;
      const unsigned Sz = TRI->getSubRegIdxSize(FoldSub);
      if (Sz <= 32)
        FinalSubIdx = FoldSub;
      else if (unsigned Composed = TRI->composeSubRegIndices(FoldSub, PairLane))
        FinalSubIdx = Composed;
      else
        FinalSubIdx = FoldSub;
    }
  }

  if (OrigSrcReg == DstReg)
    SrcReg = LaneSrcBase;

  bool KillState = false;
  if (SrcMO.isKill()) {
    bool OpSel = SrcMods & SISrcMods::OP_SEL_0;
    bool OpSelHi = SrcMods & SISrcMods::OP_SEL_1;
    KillState = true;
    if ((OpSel == OpSelHi) && !IsHiBits)
      KillState = false;
  }
  if (SrcReg.isPhysical()) {
    Register Phys = TRI->getSubReg(SrcReg, FinalSubIdx);
    if (KillState)
      NewMI.addReg(Phys, RegState::Kill);
    else
      NewMI.addReg(Phys);
  } else {
    if (KillState)
      NewMI.addReg(SrcReg, RegState::Kill, FinalSubIdx);
    else
      NewMI.addReg(SrcReg, {}, FinalSubIdx);
  }
}

static MachineInstrBuilder
createUnpackedMI(MachineInstr &I, const SIInstrInfo *TII,
                 const SIRegisterInfo *TRI, uint32_t UnpackedOpcode,
                 bool IsHiBits, Register DstReg, unsigned DefSubIdx,
                 Register LaneSrcBase, unsigned DstPackSub, bool UndefOnDef) {
  MachineBasicBlock &MBB = *I.getParent();
  const DebugLoc &DL = I.getDebugLoc();
  const MachineOperand *SrcMO0 = TII->getNamedOperand(I, AMDGPU::OpName::src0);
  const MachineOperand *SrcMO1 = TII->getNamedOperand(I, AMDGPU::OpName::src1);
  unsigned OpCode = I.getOpcode();

  int64_t ClampVal = TII->getNamedOperand(I, AMDGPU::OpName::clamp)->getImm();
  unsigned Src0Mods =
      TII->getNamedOperand(I, AMDGPU::OpName::src0_modifiers)->getImm();
  unsigned Src1Mods =
      TII->getNamedOperand(I, AMDGPU::OpName::src1_modifiers)->getImm();

  MachineInstrBuilder NewMI = BuildMI(MBB, I, DL, TII->get(UnpackedOpcode));
  NewMI.addDef(DstReg, RegState::Define, DefSubIdx);
  if (UndefOnDef)
    NewMI->getOperand(0).setIsUndef(true);
  addOperandAndMods(NewMI, Src0Mods, IsHiBits, *SrcMO0, DstReg, LaneSrcBase,
                    DstPackSub, I, TRI);
  addOperandAndMods(NewMI, Src1Mods, IsHiBits, *SrcMO1, DstReg, LaneSrcBase,
                    DstPackSub, I, TRI);

  if (AMDGPU::hasNamedOperand(OpCode, AMDGPU::OpName::src2)) {
    const MachineOperand *SrcMO2 =
        TII->getNamedOperand(I, AMDGPU::OpName::src2);
    unsigned Src2Mods =
        TII->getNamedOperand(I, AMDGPU::OpName::src2_modifiers)->getImm();
    addOperandAndMods(NewMI, Src2Mods, IsHiBits, *SrcMO2, DstReg, LaneSrcBase,
                      DstPackSub, I, TRI);
  }
  NewMI.addImm(ClampVal);
  NewMI.addImm(0);
  return NewMI;
}

static void
recomputeIntervalsAfterVirtualUnpack(ArrayRef<MachineInstr *> MIs,
                                     LiveIntervals &LIS,
                                     ArrayRef<Register> ExtraRegs = {}) {
  SmallVector<Register, 16> Regs;
  for (MachineInstr *MI : MIs) {
    if (!MI)
      continue;
    for (MachineOperand &MO : MI->operands()) {
      if (MO.isReg() && MO.getReg().isVirtual())
        Regs.push_back(MO.getReg());
    }
  }
  Regs.append(ExtraRegs.begin(), ExtraRegs.end());
  llvm::sort(Regs);
  Regs.erase(llvm::unique(Regs), Regs.end());
  for (Register R : Regs) {
    if (LIS.hasInterval(R))
      LIS.removeInterval(R);
    LIS.createAndComputeVirtRegInterval(R);
  }
}

/// Erase COPY defs of \p R in \p MBB when \p R has no non-debug uses (e.g. temp
/// only fed this V_PK). Collect virtual regs touched for LIS recomputation.
static void eraseRedundantCopyDefsForRegIfUnused(
    Register R, MachineBasicBlock &MBB, MachineRegisterInfo &MRI,
    LiveIntervals &LIS, SmallVectorImpl<Register> &RegsToRecompute) {
  if (!R.isVirtual() || !MRI.use_nodbg_empty(R))
    return;

  SmallVector<MachineInstr *, 8> ToErase;
  for (MachineInstr &MI : MBB) {
    if (MI.getOpcode() != AMDGPU::COPY)
      continue;
    MachineOperand &Def = MI.getOperand(0);
    if (!Def.isReg() || !Def.isDef() || Def.getReg() != R)
      continue;
    ToErase.push_back(&MI);
  }
  for (MachineInstr *MI : ToErase) {
    for (MachineOperand &MO : MI->operands()) {
      if (MO.isReg() && MO.getReg().isVirtual())
        RegsToRecompute.push_back(MO.getReg());
    }
    LIS.RemoveMachineInstrFromMaps(*MI);
    MI->eraseFromParent();
  }
}

/// True iff \p A is ordered before \p B in the same block (linear scan).
static bool instrIsBeforeInSameBB(const MachineInstr *A,
                                  const MachineInstr *B) {
  assert(A->getParent() == B->getParent());
  for (const MachineInstr &MI : *A->getParent()) {
    if (&MI == A)
      return true;
    if (&MI == B)
      return false;
  }
  llvm_unreachable("instructions not in same block");
}

static LaneBitmask laneMaskForRegOperand(const MachineOperand &MO, Register R,
                                         const MachineRegisterInfo &MRI,
                                         const SIRegisterInfo &TRI) {
  if (!MO.isReg() || MO.getReg() != R)
    return LaneBitmask::getNone();
  const TargetRegisterClass *RC = MRI.getRegClass(R);
  unsigned Sub = MO.getSubReg();
  if (Sub)
    return TRI.getSubRegIndexLaneMask(Sub);
  // No subreg: operand names the full virtual register. RC->getLaneMask() can
  // equal a single 32-bit lane for 64-bit vreg_64 classes; OR in the paired
  // lane so full-reg reads overlap sub1 when querying sub1's mask.
  LaneBitmask M = RC->getLaneMask();
  if (TRI.getRegSizeInBits(*RC) == 64) {
    LaneBitmask S0 = TRI.getSubRegIndexLaneMask(AMDGPU::sub0);
    LaneBitmask S1 = TRI.getSubRegIndexLaneMask(AMDGPU::sub1);
    if (M == S0 || M == S1)
      return S0 | S1;
  }
  return M;
}

/// True if \p Blocker defines \p R on a lane that overlaps the COPY source or
/// (non-COPY) destination lane, so the same-vreg COPY cannot be sunk past it.
static bool instrBlocksSinkOfSameVRegCopy(const MachineInstr &Blocker,
                                          const MachineInstr &CopyMI,
                                          Register R, LaneBitmask SMask,
                                          LaneBitmask DMask,
                                          const MachineRegisterInfo &MRI,
                                          const SIRegisterInfo &TRI) {
  for (const MachineOperand &MO : Blocker.all_defs()) {
    if (!MO.isReg() || MO.getReg() != R)
      continue;
    unsigned Sub = MO.getSubReg();
    const TargetRegisterClass *RC = MRI.getRegClass(R);
    LaneBitmask FullMask = RC->getLaneMask();
    LaneBitmask DefM = Sub ? TRI.getSubRegIndexLaneMask(Sub) : FullMask;
    if ((DefM & SMask).any())
      return true;
    if ((DefM & DMask).any() && &Blocker != &CopyMI)
      return true;
  }
  return false;
}

/// First instruction in the same MBB strictly after \p CopyMI in program order,
/// before \p SearchEnd (exclusive), that has an operand on \p R touching a
/// lane overlapping \p DMask.
static MachineInstr *findFirstDependentUseAfterSameVRegCopy(
    MachineInstr &CopyMI, Register R, LaneBitmask DMask,
    const MachineRegisterInfo &MRI, const SIRegisterInfo &TRI,
    MachineBasicBlock::iterator SearchEnd) {
  for (auto It = std::next(CopyMI.getIterator()); It != SearchEnd; ++It) {
    MachineInstr &MI = *It;
    for (MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || MO.getReg() != R || MO.isDebug())
        continue;
      // Ignore operands that do not read the register (e.g. S_NOP may carry
      // reg operands that are not real uses for liveness).
      if (!MO.readsReg())
        continue;
      LaneBitmask M = laneMaskForRegOperand(MO, R, MRI, TRI);
      if ((M & DMask).none())
        continue;
      return &MI;
    }
  }
  return nullptr;
}

/// Sink COPY %R:DstSub = %R:SrcSub toward its first dependent use, without
/// crossing \p RegionEnd (the exclusive end of the current schedule region —
/// e.g. the next scheduling boundary). If the first dependent use is in a later
/// region, sink to immediately before \p RegionEnd instead.
static bool trySinkSameVRegSubregCopy(MachineInstr &CopyMI,
                                      MachineRegisterInfo &MRI,
                                      LiveIntervals &LIS,
                                      const SIRegisterInfo &TRI,
                                      MachineBasicBlock::iterator RegionEnd) {
  if (CopyMI.getOpcode() != AMDGPU::COPY || CopyMI.getNumOperands() < 2)
    return false;
  if (CopyMI.isBundled())
    return false;

  MachineOperand &DefMO = CopyMI.getOperand(0);
  MachineOperand &SrcMO = CopyMI.getOperand(1);
  if (!DefMO.isReg() || !DefMO.isDef() || !SrcMO.isReg())
    return false;
  if (!SrcMO.readsReg())
    return false;
  Register R = DefMO.getReg();
  if (R != SrcMO.getReg() || !R.isVirtual())
    return false;
  unsigned DSub = DefMO.getSubReg();
  unsigned SSub = SrcMO.getSubReg();
  if (!DSub || !SSub || DSub == SSub)
    return false;

  LaneBitmask DMask = TRI.getSubRegIndexLaneMask(DSub);
  LaneBitmask SMask = TRI.getSubRegIndexLaneMask(SSub);

  MachineBasicBlock *MBB = CopyMI.getParent();
  MachineBasicBlock::iterator CopyIt = CopyMI.getIterator();

  if (std::next(CopyIt) == RegionEnd)
    return false;

  MachineInstr *FirstUseInRegion = findFirstDependentUseAfterSameVRegCopy(
      CopyMI, R, DMask, MRI, TRI, RegionEnd);

  MachineBasicBlock::iterator InsertPt;

  if (FirstUseInRegion) {
    MachineBasicBlock::iterator FirstUseIt = FirstUseInRegion->getIterator();
    InsertPt = FirstUseIt;
    for (MachineBasicBlock::iterator It = std::next(CopyIt); It != FirstUseIt;
         ++It) {
      if (instrBlocksSinkOfSameVRegCopy(*It, CopyMI, R, SMask, DMask, MRI,
                                        TRI)) {
        InsertPt = It;
        break;
      }
    }
  } else {
    // Dependent use is only past this schedule region — sink to the region
    // end (before RegionEnd), not to the first use in the next region.
    if (RegionEnd == MBB->end())
      return false;
    MachineInstr *LaterUse = findFirstDependentUseAfterSameVRegCopy(
        CopyMI, R, DMask, MRI, TRI, MBB->end());
    if (!LaterUse)
      return false;
    InsertPt = RegionEnd;
    for (MachineBasicBlock::iterator It = std::next(CopyIt); It != RegionEnd;
         ++It) {
      if (instrBlocksSinkOfSameVRegCopy(*It, CopyMI, R, SMask, DMask, MRI,
                                        TRI)) {
        InsertPt = It;
        break;
      }
    }
  }

  if (InsertPt == std::next(CopyIt))
    return false;

  CopyMI.moveBefore(&*InsertPt);
  LIS.handleMove(CopyMI);
  return true;
}

/// COPY DstReg:DstSub = SrcReg:SrcSub with DstReg==SrcReg (virtual): redirect
/// explicit uses of DstSub to SrcSub and remove the COPY.
///
/// Does not remove the COPY when any use reads the destination lane without an
/// explicit DstSub (e.g. full-register %v:vreg_64) — those operands cannot be
/// rewritten to SrcSub without changing semantics (sub1 would become undef).
static bool tryFoldSameVRegSubregCopy(MachineInstr &CopyMI,
                                      MachineRegisterInfo &MRI,
                                      LiveIntervals &LIS,
                                      const SIRegisterInfo &TRI) {
  if (CopyMI.getOpcode() != AMDGPU::COPY || CopyMI.getNumOperands() < 2)
    return false;
  MachineOperand &DefMO = CopyMI.getOperand(0);
  MachineOperand &SrcMO = CopyMI.getOperand(1);
  if (!DefMO.isReg() || !DefMO.isDef() || !SrcMO.isReg())
    return false;
  if (!SrcMO.readsReg())
    return false;
  Register R = DefMO.getReg();
  if (R != SrcMO.getReg() || !R.isVirtual())
    return false;
  unsigned DSub = DefMO.getSubReg();
  unsigned SSub = SrcMO.getSubReg();
  if (!DSub || !SSub || DSub == SSub)
    return false;

  LaneBitmask DMask = TRI.getSubRegIndexLaneMask(DSub);

  SmallVector<MachineOperand *, 16> UseOps;
  for (MachineOperand &MO : MRI.use_nodbg_operands(R)) {
    if (!MO.readsReg())
      continue;
    if (MO.getParent() == &CopyMI)
      continue;

    LaneBitmask M = laneMaskForRegOperand(MO, R, MRI, TRI);
    if ((M & DMask).none())
      continue;

    unsigned Sub = MO.getSubReg();

    if (Sub != DSub)
      return false;

    MachineInstr *UseMI = MO.getParent();
    if (UseMI->getParent() != CopyMI.getParent())
      return false;
    if (!instrIsBeforeInSameBB(&CopyMI, UseMI))
      return false;

    UseOps.push_back(&MO);
  }

  for (MachineOperand *MO : UseOps)
    MO->setSubReg(SSub);

  LIS.RemoveMachineInstrFromMaps(CopyMI);
  CopyMI.eraseFromParent();

  if (LIS.hasInterval(R)) {
    LIS.removeInterval(R);
    LIS.createAndComputeVirtRegInterval(R);
  }
  return true;
}

/// Walk [Begin, End) and fold / sink same-vreg subreg COPYs (iterator-safe).
static bool cleanupSameVRegSubregCopiesInRange(
    MachineBasicBlock::iterator Begin, MachineBasicBlock::iterator End,
    MachineRegisterInfo &MRI, LiveIntervals &LIS, const SIRegisterInfo &TRI) {
  bool Changed = false;
  SmallVector<MachineInstr *, 16> Copies;
  for (auto I = Begin; I != End; ++I) {
    if (I->getOpcode() == AMDGPU::COPY)
      Copies.push_back(&*I);
  }
  for (MachineInstr *MI : Copies) {
    if (tryFoldSameVRegSubregCopy(*MI, MRI, LIS, TRI))
      Changed = true;
  }
  Copies.clear();
  for (auto I = Begin; I != End; ++I) {
    if (I->getOpcode() == AMDGPU::COPY)
      Copies.push_back(&*I);
  }
  for (MachineInstr *MI : Copies) {
    if (trySinkSameVRegSubregCopy(*MI, MRI, LIS, TRI, End))
      Changed = true;
  }
  return Changed;
}

static bool performF32Unpacking(MachineInstr &I, const SIInstrInfo *TII,
                                const SIRegisterInfo *TRI, LiveIntervals &LIS,
                                DenseSet<Register> *InvolvedUnpackRegs) {
  uint32_t UnpackedOpcode = mapToUnpackedOpcode(I);
  if (UnpackedOpcode == std::numeric_limits<uint32_t>::max()) {
    debugSkipVPKUnpack(I, TII, TRI, "unsupported_opcode");
    return false;
  }
  if (canUnpackingClobberRegister(I, TII, TRI)) {
    debugSkipVPKUnpack(I, TII, TRI, "would_clobber_overlapping_src");
    return false;
  }

  const MachineOperand &DstMO = I.getOperand(0);
  Register DstReg = DstMO.getReg();
  if (!DstReg.isVirtual()) {
    debugSkipVPKUnpack(I, TII, TRI, "physical_dst");
    return false;
  }

  unsigned DstSub = DstMO.getSubReg();

  MachineFunction *MF = I.getMF();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetRegisterClass *DstRC = MRI.getRegClass(DstReg);

  // Lane indices for subreg defs: full vreg_64 uses sub0/sub1. For composite
  // 64-bit packed destinations (sub0_sub1, sub2_sub3, sub4_sub5, sub6_sub7,
  // … on wide vectors), decompose with composeSubRegIndices — not getSubReg,
  // whose first operand is an MCRegister, not a SubRegIndex (SIInstrInfo uses
  // the same compose pattern for partial subregs).
  unsigned LoSeqIdx = DstSub ? TRI->composeSubRegIndices(DstSub, AMDGPU::sub0)
                             : static_cast<unsigned>(AMDGPU::sub0);
  unsigned HiSeqIdx = DstSub ? TRI->composeSubRegIndices(DstSub, AMDGPU::sub1)
                             : static_cast<unsigned>(AMDGPU::sub1);
  if (DstSub && (!LoSeqIdx || !HiSeqIdx)) {
    debugSkipVPKUnpack(I, TII, TRI, "bad_composite_subreg_decompose", LoSeqIdx,
                       HiSeqIdx);
    return false;
  }

  const bool LoRCValid = TRI->isSubRegValidForRegClass(DstRC, LoSeqIdx);
  const bool HiRCValid = TRI->isSubRegValidForRegClass(DstRC, HiSeqIdx);
  if (!LoRCValid || !HiRCValid) {
    debugSkipVPKUnpack(I, TII, TRI, "invalid_subreg_for_regclass", LoSeqIdx,
                       HiSeqIdx, LoRCValid, HiRCValid);
    return false;
  }

  Register LaneSrcBase = DstReg;

  const bool UndefOnFirstLane = !vpkAnySrcUsesDst(I, TII, DstReg) &&
                                !hasExplicitDefOfRegBefore(I, DstReg);

  SmallVector<Register, 4> CleanCandidates;
  auto AddSrcCandidate = [&](const MachineOperand *MO) {
    if (!MO || !MO->isReg())
      return;
    Register R = MO->getReg();
    if (!R.isVirtual() || R == DstReg)
      return;
    if (!isVirtualSrcRegOnlyUsedByThisVPK(R, I, MRI))
      return;
    CleanCandidates.push_back(R);
  };
  AddSrcCandidate(TII->getNamedOperand(I, AMDGPU::OpName::src0));
  AddSrcCandidate(TII->getNamedOperand(I, AMDGPU::OpName::src1));
  if (AMDGPU::hasNamedOperand(I.getOpcode(), AMDGPU::OpName::src2))
    AddSrcCandidate(TII->getNamedOperand(I, AMDGPU::OpName::src2));
  llvm::sort(CleanCandidates);
  CleanCandidates.erase(llvm::unique(CleanCandidates), CleanCandidates.end());

  SmallVector<Register, 4> SrcRegsForInvolvedSet;
  auto AddInvolvedSrc = [&](const MachineOperand *MO) {
    if (!MO || !MO->isReg())
      return;
    Register R = MO->getReg();
    if (R.isVirtual())
      SrcRegsForInvolvedSet.push_back(R);
  };
  AddInvolvedSrc(TII->getNamedOperand(I, AMDGPU::OpName::src0));
  AddInvolvedSrc(TII->getNamedOperand(I, AMDGPU::OpName::src1));
  if (AMDGPU::hasNamedOperand(I.getOpcode(), AMDGPU::OpName::src2))
    AddInvolvedSrc(TII->getNamedOperand(I, AMDGPU::OpName::src2));

  MachineBasicBlock *MBB = I.getParent();

  LIS.RemoveMachineInstrFromMaps(I);

  // BuildMI(MBB, I, ...) prepends each instruction immediately before I, so the
  // first BuildMI call ends up furthest from I in program order: low lane, then
  // high lane.
  MachineInstrBuilder Op0L = createUnpackedMI(I, TII, TRI, UnpackedOpcode,
                                              /*IsHiBits=*/false, DstReg,
                                              LoSeqIdx, LaneSrcBase, DstSub,
                                              /*UndefOnDef=*/UndefOnFirstLane);
  LIS.InsertMachineInstrInMaps(*Op0L);
  MachineInstrBuilder Op0H =
      createUnpackedMI(I, TII, TRI, UnpackedOpcode,
                       /*IsHiBits=*/true, DstReg, HiSeqIdx, LaneSrcBase, DstSub,
                       /*UndefOnDef=*/false);
  LIS.InsertMachineInstrInMaps(*Op0H);

  uint32_t IFlags = I.getFlags();
  Op0L->setFlags(IFlags);
  Op0H->setFlags(IFlags);

  I.eraseFromParent();

  MF->getProperties().reset(MachineFunctionProperties::Property::IsSSA);

  SmallVector<Register, 16> ExtraRegs;
  for (Register R : CleanCandidates)
    eraseRedundantCopyDefsForRegIfUnused(R, *MBB, MRI, LIS, ExtraRegs);

  SmallVector<MachineInstr *, 5> ToRecompute;
  ToRecompute.push_back(&*Op0L);
  ToRecompute.push_back(&*Op0H);
  recomputeIntervalsAfterVirtualUnpack(ToRecompute, LIS, ExtraRegs);
  if (InvolvedUnpackRegs) {
    InvolvedUnpackRegs->insert(DstReg);
    for (Register R : SrcRegsForInvolvedSet)
      InvolvedUnpackRegs->insert(R);
  }
  return true;
}

/// True if \p R is a 64-bit VGPR/AGPR virtual register and every register
/// operand uses only \p sub0 or \p sub1 (no full-register or other subregs).
static bool is64BitOnlySub0Sub1(Register R, const MachineRegisterInfo &MRI,
                                const SIRegisterInfo &TRI) {
  if (!R.isVirtual())
    return false;
  const TargetRegisterClass *RC = MRI.getRegClass(R);
  if (!RC || TRI.getRegSizeInBits(*RC) != 64)
    return false;
  if (!TRI.isVGPR(MRI, R) && !TRI.isAGPR(MRI, R))
    return false;

  bool Any = false;
  for (const MachineOperand &MO : MRI.reg_operands(R)) {
    Any = true;
    unsigned S = MO.getSubReg();
    if (S == AMDGPU::NoSubRegister)
      return false;
    if (S != AMDGPU::sub0 && S != AMDGPU::sub1)
      return false;
  }
  return Any;
}

static bool split64BitOnlySub0Sub1ToV32(Register R, MachineRegisterInfo &MRI,
                                        const SIRegisterInfo &TRI,
                                        LiveIntervals &LIS) {
  const TargetRegisterClass *RC = MRI.getRegClass(R);
  const TargetRegisterClass *LoRC = TRI.getSubRegisterClass(RC, AMDGPU::sub0);
  const TargetRegisterClass *HiRC = TRI.getSubRegisterClass(RC, AMDGPU::sub1);
  if (!LoRC || !HiRC)
    return false;

  bool HasLo = false, HasHi = false;
  for (const MachineOperand &MO : MRI.reg_operands(R)) {
    unsigned S = MO.getSubReg();
    if (S == AMDGPU::sub0)
      HasLo = true;
    else if (S == AMDGPU::sub1)
      HasHi = true;
  }
  if (!HasLo && !HasHi)
    return false;

  Register NewLo, NewHi;
  if (HasLo)
    NewLo = MRI.createVirtualRegister(LoRC);
  if (HasHi)
    NewHi = MRI.createVirtualRegister(HiRC);

  SmallVector<MachineOperand *, 32> Ops;
  for (MachineOperand &MO : MRI.reg_operands(R))
    Ops.push_back(&MO);

  for (MachineOperand *MO : Ops) {
    unsigned S = MO->getSubReg();
    Register NR = (S == AMDGPU::sub0) ? NewLo : NewHi;
    MO->setReg(NR);
    MO->setSubReg(AMDGPU::NoSubRegister);
    // Partial subreg defs used UndefOnDef for the wide vreg; each new vreg_32
    // is fully written by its instruction — drop stale undef on defs.
    if (MO->isDef())
      MO->setIsUndef(false);
  }

  if (LIS.hasInterval(R))
    LIS.removeInterval(R);
  if (HasLo)
    LIS.createAndComputeVirtRegInterval(NewLo);
  if (HasHi)
    LIS.createAndComputeVirtRegInterval(NewHi);

  LLVM_DEBUG({
    dbgs() << DEBUG_TYPE << ": post-unpack split " << printReg(R, &TRI)
           << " -> ";
    if (HasLo)
      dbgs() << printReg(NewLo, &TRI) << '(' << TRI.getRegClassName(LoRC)
             << ')';
    if (HasLo && HasHi)
      dbgs() << ", ";
    if (HasHi)
      dbgs() << printReg(NewHi, &TRI) << '(' << TRI.getRegClassName(HiRC)
             << ')';
    dbgs() << '\n';
  });

  return true;
}

/// Split 64-bit VGPR/AGPR vregs that only use \p sub0 / \p sub1 into 32-bit
/// vregs. Only considers virtual registers that appeared as a V_PK
/// destination or as a virtual src0/src1/src2 on a successful unpack in this
/// pass (\p InvolvedUnpackRegs). Only the first \p MaxRegsToProcess candidates
/// in virtual-register index order are considered (UINT_MAX = all).
static bool postUnpackSplit64BitSubregsToV32(
    MachineFunction &MF, LiveIntervals &LIS, const SIRegisterInfo &TRI,
    unsigned MaxRegsToProcess, const DenseSet<Register> &InvolvedUnpackRegs) {
  assert(MaxRegsToProcess > 0);
  if (InvolvedUnpackRegs.empty())
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const unsigned NumVR = MRI.getNumVirtRegs();
  SmallVector<Register, 64> Candidates;
  Candidates.reserve(InvolvedUnpackRegs.size());
  for (unsigned I = 0; I < NumVR; ++I) {
    Register R = Register::index2VirtReg(I);
    if (!InvolvedUnpackRegs.contains(R))
      continue;
    if (is64BitOnlySub0Sub1(R, MRI, TRI))
      Candidates.push_back(R);
  }
  llvm::sort(Candidates, [](Register A, Register B) {
    return A.virtRegIndex() < B.virtRegIndex();
  });

  const bool Unlimited =
      MaxRegsToProcess == std::numeric_limits<unsigned>::max();

  bool Changed = false;
  unsigned Seen = 0;
  for (Register R : Candidates) {
    if (!Unlimited && Seen >= MaxRegsToProcess)
      break;
    ++Seen;
    if (split64BitOnlySub0Sub1ToV32(R, MRI, TRI, LIS))
      Changed = true;
  }
  return Changed;
}

class AMDGPUIGLPUnpackImpl {
  LiveIntervals *LIS;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;

public:
  explicit AMDGPUIGLPUnpackImpl(LiveIntervals *L) : LIS(L) {}

  bool run(MachineFunction &MF);
};

bool AMDGPUIGLPUnpackImpl::run(MachineFunction &MF) {
  if (!LIS)
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  const MCInstrInfo &II = *TII;

  MBBRegionsVector SubRegions;

  bool Changed = false;
  bool AnyVPKUnpacked = false;
  DenseSet<Register> InvolvedUnpackRegs;

  for (MachineBasicBlock &MBB : MF) {
    SubRegions.clear();
    getSchedRegions(&MBB, SubRegions, /*RegionsTopDown=*/true);

    for (const SchedRegion &SR : SubRegions) {
      MachineBasicBlock::iterator Beg = SR.RegionBegin;
      MachineBasicBlock::iterator End = SR.RegionEnd;
      CandidateRegion CR;
      if (!findCandidateRegion(MBB, Beg, End, II, CR)) {
        LLVM_DEBUG({
          unsigned NumVPK = 0;
          bool HasValuSpacingIGLP = false;
          for (auto It = Beg; It != End; ++It) {
            if (isVPKOpcode(II, It->getOpcode()))
              ++NumVPK;
            if (It->getOpcode() == AMDGPU::IGLP_OPT &&
                It->getNumOperands() >= 1 && It->getOperand(0).isImm() &&
                It->getOperand(0).getImm() ==
                    static_cast<int64_t>(
                        AMDGPU::IGLPStrategyID::MFMAValuSpacingOptID))
              HasValuSpacingIGLP = true;
          }
          if (NumVPK > 0 && kRequireMFMAValuSpacingIGLPOptForRegion &&
              !HasValuSpacingIGLP)
            dbgs() << DEBUG_TYPE << ": skip region MBB#" << MBB.getNumber()
                   << " (" << NumVPK << " V_PK, missing IGLP_OPT imm="
                   << static_cast<int>(
                          AMDGPU::IGLPStrategyID::MFMAValuSpacingOptID)
                   << ")\n";
        });
        continue;
      }

      if (schedRegionHasExplicitAllocatablePhysReg(Beg, End, MF.getRegInfo())) {
        LLVM_DEBUG({
          dbgs() << DEBUG_TYPE << ": skip region MBB#" << MBB.getNumber()
                 << " (explicit allocatable physical register operand)\n";
        });
        continue;
      }

      const unsigned MaxVPK = AMDGPUIGLPUnpackMaxVPKPerRegion;
      LLVM_DEBUG({
        dbgs() << DEBUG_TYPE << ": " << MF.getName() << " MBB#"
               << MBB.getNumber()
               << " region instrs=" << std::distance(Beg, End)
               << " v_pk=" << CR.VPKInsts.size();
        if (MaxVPK > 0)
          dbgs() << " max_vpk_unpack=" << MaxVPK;
        dbgs() << "\n";
      });

      unsigned UnpackSlot = 0;
      for (MachineInstr *MI : CR.VPKInsts) {
        if (MaxVPK > 0 && UnpackSlot >= MaxVPK) {
          LLVM_DEBUG({
            dbgs() << DEBUG_TYPE << ": skip remaining V_PK in region (limit "
                   << MaxVPK << ")\n";
          });
          break;
        }
        ++UnpackSlot;
        if (performF32Unpacking(*MI, TII, TRI, *LIS, &InvolvedUnpackRegs)) {
          Changed = true;
          AnyVPKUnpacked = true;
        }
      }

      if (cleanupSameVRegSubregCopiesInRange(CR.Begin, CR.End, MF.getRegInfo(),
                                             *LIS, *TRI))
        Changed = true;
    }
  }

  if (AnyVPKUnpacked) {
    const unsigned PostMax = AMDGPUIGLPUnpackPostCleanupMaxVRegs;
    if (PostMax > 0) {
      if (postUnpackSplit64BitSubregsToV32(MF, *LIS, *TRI, PostMax,
                                           InvolvedUnpackRegs))
        Changed = true;
    }
  }

  return Changed;
}

class AMDGPUIGLPUnpackLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUIGLPUnpackLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    LiveIntervals &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    return AMDGPUIGLPUnpackImpl(&LIS).run(MF);
  }

  StringRef getPassName() const override { return "AMDGPU IGLP unpack"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // Like GCNRewritePartialRegUses: preserving LIS/SlotIndexes avoids the
    // legacy PM recomputing them before MachineScheduler when we make no MIR
    // changes.
    AU.setPreservesCFG();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AMDGPUIGLPUnpackLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUIGLPUnpackLegacy, DEBUG_TYPE, "AMDGPU IGLP unpack",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AMDGPUIGLPUnpackLegacy, DEBUG_TYPE, "AMDGPU IGLP unpack",
                    false, false)

char &llvm::AMDGPUIGLPUnpackID = AMDGPUIGLPUnpackLegacy::ID;

PreservedAnalyses
AMDGPUIGLPUnpackPass::run(MachineFunction &MF,
                          MachineFunctionAnalysisManager &MFAM) {
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  AMDGPUIGLPUnpackImpl Impl(&LIS);
  if (!Impl.run(MF))
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
