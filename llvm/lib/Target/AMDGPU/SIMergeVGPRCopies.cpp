//===-- SIMergeVGPRCopies.cpp - Merge VGPR move pairs ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Merge pairs of 32-bit VGPR moves into a single 64-bit move
/// (V_PK_MOV_B32 or V_MOV_B64).
///
/// Both sources and destinations must be consecutive and potentially
/// even-aligned, but the source and destination pairs can be in either order.
/// The sources can be SGPRs if the target supports V_MOV_B64, but not if the
/// merged copy needs to use V_PK_MOV_B32 (e.g. if the source pairs are inversed
/// relative to the destinations).
///
/// The pass uses a two-phase algorithm per basic block:
///   Phase 1 - lookahead scan: for each V_MOV_B32_e32, scan forward through
///   the block to find a non-adjacent partner that can be safely merged. Pairs
///   are recorded without modifying the instruction stream.
///   Phase 2 - apply: emit the merged 64-bit instruction at the first move's
///   position and erase both originals.
///
//===----------------------------------------------------------------------===//

#include "SIMergeVGPRCopies.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "si-merge-vgpr-copies"

namespace {

class SIMergeVGPRCopiesLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIMergeVGPRCopiesLegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "SI Merge VGPR copies"; }
};

class SIMergeVGPRCopies {
public:
  bool run(MachineFunction &MF);
};

} // End anonymous namespace.

INITIALIZE_PASS(SIMergeVGPRCopiesLegacy, DEBUG_TYPE, "SI Merge VGPR copies",
                false, false)

char SIMergeVGPRCopiesLegacy::ID = 0;

char &llvm::SIMergeVGPRCopiesID = SIMergeVGPRCopiesLegacy::ID;

PreservedAnalyses SIMergeVGPRCopiesPass::run(MachineFunction &MF,
                                             MachineFunctionAnalysisManager &) {
  SIMergeVGPRCopies().run(MF);
  return PreservedAnalyses::all();
}

bool SIMergeVGPRCopiesLegacy::runOnMachineFunction(MachineFunction &MF) {
  return SIMergeVGPRCopies().run(MF);
}

static bool getBaseIfConsecutive(const Register &Reg1, const Register &Reg2,
                                 const SIRegisterInfo *TRI, Register &Base) {
  long Dist =
      static_cast<long>(TRI->getHWRegIndex(Reg2)) - TRI->getHWRegIndex(Reg1);
  if (std::abs(Dist) != 1)
    return false;
  Base = (Dist == 1) ? Reg1 : Reg2;
  return true;
}

namespace {
struct MergePair {
  MachineInstr *First;
  MachineInstr *Second;
  Register Dst64;
  Register Src64;
  bool SrcIsInversed;
  bool KillSrc;
  bool UsesSGPRSources;
};

// Tracks hazards between a candidate first move MI and lookahead instructions.
// Valid partner destinations are MI.dst's two HW-index neighbours, and valid
// partner sources are MI.src's two HW-index neighbours (same consecutiveness
// requirement). All def/use tracking therefore reduces to six booleans.
struct HazardTracker {
  const SIRegisterInfo *TRI;
  const unsigned MIDstHWIdx;
  const unsigned MISrcHWIdx;
  const bool SrcIsSGPR;
  bool LoDstNBDefined = false; // I.dst-1 defined by an intermediate instr
  bool HiDstNBDefined = false; // I.dst+1 defined by an intermediate instr
  bool LoDstNBUsed = false;    // I.dst-1 read by an intermediate instr
  bool HiDstNBUsed = false;    // I.dst+1 read by an intermediate instr
  bool LoSrcNBDefined = false; // I.src-1 defined by an intermediate instr
  bool HiSrcNBDefined = false; // I.src+1 defined by an intermediate instr

  HazardTracker(Register MIDstReg, Register MISrcReg, bool SrcIsSGPR,
                const SIRegisterInfo *TRI)
      : TRI(TRI), MIDstHWIdx(TRI->getHWRegIndex(MIDstReg)),
        MISrcHWIdx(TRI->getHWRegIndex(MISrcReg)), SrcIsSGPR(SrcIsSGPR) {}

  void update(const MachineInstr &K) {
    for (const MachineOperand &MO : K.operands()) {
      if (!MO.isReg() || !MO.getReg().isPhysical())
        continue;
      Register R = MO.getReg();
      unsigned HWIdx = TRI->getHWRegIndex(R);
      if (MO.isDef()) {
        if (AMDGPU::VGPR_32RegClass.contains(R)) {
          LoDstNBDefined |= (HWIdx + 1 == MIDstHWIdx);
          HiDstNBDefined |= (HWIdx == MIDstHWIdx + 1);
          if (!SrcIsSGPR) {
            LoSrcNBDefined |= (HWIdx + 1 == MISrcHWIdx);
            HiSrcNBDefined |= (HWIdx == MISrcHWIdx + 1);
          }
        } else if (SrcIsSGPR && AMDGPU::SGPR_32RegClass.contains(R)) {
          LoSrcNBDefined |= (HWIdx + 1 == MISrcHWIdx);
          HiSrcNBDefined |= (HWIdx == MISrcHWIdx + 1);
        }
      } else if (MO.isUse() && AMDGPU::VGPR_32RegClass.contains(R)) {
        LoDstNBUsed |= (HWIdx + 1 == MIDstHWIdx);
        HiDstNBUsed |= (HWIdx == MIDstHWIdx + 1);
      }
    }
  }

  // Returns true when no future candidate can form a valid pair with MI.
  bool isDeadEnd() const {
    return (LoDstNBDefined && HiDstNBDefined) ||
           (LoSrcNBDefined && HiSrcNBDefined);
  }

  bool hasSourceHazard(Register NextMISrcReg) const {
    unsigned HWIdx = TRI->getHWRegIndex(NextMISrcReg);
    if (HWIdx + 1 == MISrcHWIdx)
      return LoSrcNBDefined;
    if (HWIdx == MISrcHWIdx + 1)
      return HiSrcNBDefined;
    return false;
  }

  bool hasDestHazard(Register NextMIDstReg) const {
    unsigned HWIdx = TRI->getHWRegIndex(NextMIDstReg);
    if (HWIdx + 1 == MIDstHWIdx)
      return LoDstNBDefined || LoDstNBUsed;
    if (HWIdx == MIDstHWIdx + 1)
      return HiDstNBDefined || HiDstNBUsed;
    return true;
  }
};

std::optional<MergePair> areMergablePartners(MachineInstr &MI,
                                             MachineInstr &NextMI,
                                             const GCNSubtarget &ST,
                                             const HazardTracker &Tracker) {
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();

  if (NextMI.getOpcode() != AMDGPU::V_MOV_B32_e32)
    return std::nullopt;

  // Get destination registers.
  Register MIDstReg = MI.getOperand(0).getReg();
  Register NextMIDstReg = NextMI.getOperand(0).getReg();
  if (!AMDGPU::VGPR_32RegClass.contains(NextMIDstReg))
    return std::nullopt;

  // Get source registers.
  const MachineOperand *NextMISrcOp =
      TII->getNamedOperand(NextMI, AMDGPU::OpName::src0);
  const MachineOperand *MISrcOp =
      TII->getNamedOperand(MI, AMDGPU::OpName::src0);
  if (!MISrcOp || !NextMISrcOp || !MISrcOp->isReg() || !NextMISrcOp->isReg())
    return std::nullopt;
  Register MISrcReg = MISrcOp->getReg();
  Register NextMISrcReg = NextMISrcOp->getReg();

  // Hazard checks: is it safe to move NextMI's read/write to I's position?
  if (Tracker.hasSourceHazard(NextMISrcReg) ||
      Tracker.hasDestHazard(NextMIDstReg))
    return std::nullopt;

  // Sources may always be VGPRs, but can also be SGPRs if the target supports
  // V_MOV_B64. Either way, the sources must be the same class.
  bool UsesSGPRSources = ST.hasMovB64() &&
                         AMDGPU::SGPR_32RegClass.contains(MISrcReg) &&
                         AMDGPU::SGPR_32RegClass.contains(NextMISrcReg);
  if (!UsesSGPRSources && !(AMDGPU::VGPR_32RegClass.contains(MISrcReg) &&
                            AMDGPU::VGPR_32RegClass.contains(NextMISrcReg)))
    return std::nullopt;

  // Ensure the first copy is the one with the lower-numbered destination
  // register. This simplifies checking for consecutive sources and ensures
  // the merged copy reads the same registers as the original pair.
  Register DstBase;
  Register SrcBase;
  if (!getBaseIfConsecutive(MIDstReg, NextMIDstReg, TRI, DstBase) ||
      !getBaseIfConsecutive(MISrcReg, NextMISrcReg, TRI, SrcBase))
    return std::nullopt;

  // The sources must also be consecutive and even-aligned, but they can be in
  // either order. If they are inversed, the merged copy needs to use
  // V_PK_MOV_B32 which in turn cannot use SGPR sources, otherwise it can use
  // V_MOV_B64.
  bool SrcIsInversed = (DstBase == NextMIDstReg) != (SrcBase == NextMISrcReg);
  if (SrcIsInversed && (UsesSGPRSources || !ST.hasPkMovB32()))
    return std::nullopt;

  const TargetRegisterClass *VGPR64RC = TRI->getVGPR64Class();
  const TargetRegisterClass *SGPR64RC = &AMDGPU::SReg_64RegClass;
  Register Dst64 = TRI->getMatchingSuperReg(DstBase, AMDGPU::sub0, VGPR64RC);
  Register Src64 = TRI->getMatchingSuperReg(
      SrcBase, AMDGPU::sub0, UsesSGPRSources ? SGPR64RC : VGPR64RC);
  if (!Dst64 || !Src64)
    return std::nullopt;

  bool KillSrc = MISrcOp->isKill() && NextMISrcOp->isKill();
  return MergePair{&MI,           &NextMI, Dst64,          Src64,
                   SrcIsInversed, KillSrc, UsesSGPRSources};
}

std::optional<MergePair>
findMergeablePartner(MachineBasicBlock::iterator I,
                     MachineBasicBlock::iterator E, const GCNSubtarget &ST,
                     const SmallPtrSetImpl<MachineInstr *> &Claimed) {
  // Lookahead: scan instructions after I for a partner.
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();
  Register MIDstReg = I->getOperand(0).getReg();
  const MachineOperand *MISrcOp =
      TII->getNamedOperand(*I, AMDGPU::OpName::src0);
  if (!MISrcOp || !MISrcOp->isReg())
    return std::nullopt;
  Register MISrcReg = MISrcOp->getReg();
  bool SrcIsSGPR = AMDGPU::SGPR_32RegClass.contains(MISrcReg);
  HazardTracker Tracker(MIDstReg, MISrcReg, SrcIsSGPR, TRI);

  for (MachineInstr &NextMI : instructionsWithoutDebug(std::next(I), E)) {
    if (Claimed.count(&NextMI))
      continue;

    // An EXEC write invalidates the lane mask assumption — stop entirely.
    // A write to I's destination means the merged write (placed at I's
    // position) would be clobbered; no later NextMI can help either.
    if (NextMI.modifiesRegister(AMDGPU::EXEC, TRI) ||
        NextMI.modifiesRegister(MIDstReg, TRI))
      return std::nullopt;

    if (std::optional<MergePair> MaybePair =
            areMergablePartners(*I, NextMI, ST, Tracker))
      return MaybePair;

    Tracker.update(NextMI);

    if (Tracker.isDeadEnd())
      return std::nullopt;
  }

  return std::nullopt; // no partner found
}
} // namespace

bool SIMergeVGPRCopies::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  // Only merge when the target has a 64-bit move instruction.
  if (!ST.hasPkMovB32() && !ST.hasMovB64())
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    SmallVector<MergePair, 8> Pairs;
    SmallPtrSet<MachineInstr *, 16> Claimed;

    // Phase 1: scan forward from each unclaimed V_MOV_B32_e32 to find a
    // mergeable partner later in the block.
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E;
         ++I) {
      if (I->getOpcode() != AMDGPU::V_MOV_B32_e32 || Claimed.count(&*I))
        continue;

      if (std::optional<MergePair> MaybePair =
              findMergeablePartner(I, E, ST, Claimed)) {
        MergePair &P = Pairs.emplace_back(std::move(*MaybePair));
        // Since we only look forward, we only need to claim the
        // second move in the pair.
        Claimed.insert(P.Second);
      }
    }

    // Phase 2: apply all recorded pairs in forward order.
    for (MergePair &P : Pairs) {
      MachineInstr &MI = *P.First;
      MachineInstr &NextMI = *P.Second;

      if (P.SrcIsInversed || !ST.hasMovB64()) {
        int64_t OpSelImm0 = SISrcMods::OP_SEL_1;
        int64_t OpSelImm1 = SISrcMods::OP_SEL_0 | SISrcMods::OP_SEL_1;
        if (P.SrcIsInversed)
          std::swap(OpSelImm0, OpSelImm1);
        BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(AMDGPU::V_PK_MOV_B32),
                P.Dst64)
            .addImm(OpSelImm0)
            .addReg(P.Src64)
            .addImm(OpSelImm1)
            .addReg(P.Src64)
            .addImm(0) // op_sel_lo
            .addImm(0) // op_sel_hi
            .addImm(0) // neg_lo
            .addImm(0) // neg_hi
            .addImm(0) // clamp
            .addReg(P.Src64, getKillRegState(P.KillSrc) | RegState::Implicit);
      } else {
        BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(AMDGPU::V_MOV_B64_e32),
                P.Dst64)
            .addReg(P.Src64, getKillRegState(P.KillSrc));
      }

      LLVM_DEBUG(dbgs() << "Merged VGPR32 move pair into 64-bit move from "
                        << MI << " and " << NextMI);
      MI.eraseFromParent();
      NextMI.eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}
