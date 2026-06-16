//===-- SIPinVGPR.cpp - Bias allocator to keep values VGPR-resident -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pre-RA pass for `llvm.amdgcn.internal.vgpr.pin`. For every SI_VGPR_PIN
// marker, marks the LiveInterval of each VGPR-class vreg in the forward
// COPY/INSERT_SUBREG closure of its operand not-spillable, then erases the
// marker.
//
// Not-spillable (huge_valf weight) is binding, not advisory: RAGreedy will not
// spill a pinned interval, so it redirects spilling to other (unpinned) values
// under pressure.
//
// If the pinned values' own peak VGPR demand already exceeds the function's
// VGPR budget, honoring the pins cannot succeed. The pass detects this up
// front, emits an optimization remark, and leaves the intervals spillable so
// allocation degrades gracefully. This is only a necessary condition, not a
// sufficient one: other non-spillable demand (live-in args, fixed inline-asm
// regs, reserved registers) can still make allocation fail, so this check
// reduces - but does not eliminate - the chance of a hard "ran out of
// registers" abort. The intended producer (AMDGPUPromoteAlloca) is still
// expected to keep the pinned set within budget; this is a backstop, not a
// substitute for that.
//
//===----------------------------------------------------------------------===//

#include "SIPinVGPR.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-pin-vgpr"

namespace {

class SIPinVGPRImpl {
public:
  bool run(MachineFunction &MF, LiveIntervals &LIS);
};

class SIPinVGPRLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIPinVGPRLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    LiveIntervals &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    return SIPinVGPRImpl().run(MF, LIS);
  }

  StringRef getPassName() const override {
    return "AMDGPU Pin VGPR Live Intervals";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(SIPinVGPRLegacy, DEBUG_TYPE,
                      "AMDGPU Pin VGPR Live Intervals", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(SIPinVGPRLegacy, DEBUG_TYPE,
                    "AMDGPU Pin VGPR Live Intervals", false, false)

char SIPinVGPRLegacy::ID = 0;

char &llvm::SIPinVGPRLegacyID = SIPinVGPRLegacy::ID;
// If `VReg` is a pinnable VGPR-class vreg with a live interval, append it to
// `Pinnable`.
static void collectPinnable(Register VReg, const SIRegisterInfo *TRI,
                            MachineRegisterInfo &MRI, LiveIntervals &LIS,
                            SmallVectorImpl<Register> &Pinnable) {
  if (!VReg.isVirtual())
    return;
  const TargetRegisterClass *RC = MRI.getRegClassOrNull(VReg);
  if (!RC)
    return;
  // Skip AGPR-class vregs (a "VGPR pin" on an AGPR is meaningless); the
  // use-chain walk still follows them, so an AGPR -> VGPR detour (e.g.
  // around MFMA) re-acquires the pin on the downstream VGPR view.
  if (!TRI->hasVectorRegisters(RC) || TRI->isAGPRClass(RC))
    return;

  if (!LIS.hasInterval(VReg)) {
    LLVM_DEBUG(dbgs() << "[si-pin-vgpr]   " << printReg(VReg, nullptr)
                      << " has no live interval; skipping\n");
    return;
  }
  Pinnable.push_back(VReg);
}

// Returns the peak simultaneous VGPR demand (in 32-bit units) of the pinned
// set.
//
// This counts each interval's full register width over its whole main range and
// ignores subregister liveness, so a value with dead lanes at the peak point is
// over-counted: the result is an over-approximation that can occasionally fire
// the remark for a set that would in fact have fit.
// FIXME: walk subranges (when shouldTrackSubRegLiveness) for a tighter bound.
static unsigned peakPinnedVGPRPressure(ArrayRef<Register> Pinnable,
                                       const SIRegisterInfo *TRI,
                                       MachineRegisterInfo &MRI,
                                       LiveIntervals &LIS) {
  struct Event {
    SlotIndex Idx;
    int Delta;
  };
  SmallVector<Event, 32> Events;
  for (Register R : Pinnable) {
    int Lanes = TRI->getRegSizeInBits(*MRI.getRegClass(R)) / 32;
    for (const LiveInterval::Segment &S : LIS.getInterval(R)) {
      Events.push_back({S.start, Lanes});
      Events.push_back({S.end, -Lanes});
    }
  }
  llvm::sort(Events, [](const Event &A, const Event &B) {
    if (A.Idx < B.Idx)
      return true;
    if (B.Idx < A.Idx)
      return false;
    return A.Delta < B.Delta;
  });
  int Cur = 0, Peak = 0;
  for (const Event &E : Events) {
    Cur += E.Delta;
    Peak = std::max(Peak, Cur);
  }
  return Peak;
}

bool SIPinVGPRImpl::run(MachineFunction &MF, LiveIntervals &LIS) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo *TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();

  // Phase 1: collect every SI_VGPR_PIN, seed the worklist from its operand,
  // then erase the marker (keeping LiveIntervals/SlotIndexes consistent).
  SmallVector<Register, 8> Worklist;
  SmallVector<MachineInstr *, 8> PinPseudos;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.getOpcode() == AMDGPU::SI_VGPR_PIN)
        PinPseudos.push_back(&MI);

  if (PinPseudos.empty())
    return false;

  // Keep a location/block from a marker for any remark emitted below; the
  // markers themselves are erased before we know whether the pins fit. (Loc is
  // arbitrary among multiple pins, but the remark is about the whole set.)
  const DebugLoc PinDL = PinPseudos.front()->getDebugLoc();
  MachineBasicBlock *const PinMBB = PinPseudos.front()->getParent();

  SmallDenseSet<Register, 8> ToShrink;
  for (MachineInstr *MI : PinPseudos) {
    // SI_VGPR_PIN is use-only: operand 0 is the pinned value. ISel always emits
    // a vreg, but guard against a future folder producing something else.
    MachineOperand &Src = MI->getOperand(0);
    if (Src.isReg() && Src.getReg().isVirtual()) {
      Worklist.push_back(Src.getReg());
      ToShrink.insert(Src.getReg());
    }
    LIS.RemoveMachineInstrFromMaps(*MI);
    MI->eraseFromParent();
  }

  // Erasing the marker dropped a use, so retighten the affected intervals
  // before they are (possibly) marked not-spillable below.
  for (Register VReg : ToShrink)
    if (LIS.hasInterval(VReg))
      LIS.shrinkToUses(&LIS.getInterval(VReg));

  // Phase 2: forward-only walk through whole-register COPYs and the tied
  // input of INSERT_SUBREG, marking each VGPR-class LiveInterval not-spillable.
  //
  // FIXME: does not walk REG_SEQUENCE / SUBREG_TO_REG / EXTRACT_SUBREG, so a
  // pin is lost when a value is split or re-assembled across subregisters.
  // Generalize once a real consumer lands.
  SmallDenseSet<Register, 16> Seen;
  SmallVector<Register, 16> Pinnable;
  while (!Worklist.empty()) {
    Register VReg = Worklist.pop_back_val();
    if (!Seen.insert(VReg).second)
      continue;
    collectPinnable(VReg, TRI, MRI, LIS, Pinnable);

    for (MachineInstr &UseMI : MRI.use_instructions(VReg)) {
      // A whole-register COPY (no sub-register index on either operand)
      // forwards the entire value; sub-register slices don't.
      if (UseMI.isCopy() && UseMI.getOperand(0).getSubReg() == 0 &&
          UseMI.getOperand(1).getSubReg() == 0) {
        if (UseMI.getOperand(1).getReg() == VReg)
          Worklist.push_back(UseMI.getOperand(0).getReg());
        continue;
      }
      if (UseMI.getOpcode() == TargetOpcode::INSERT_SUBREG) {
        // Only propagate via operand 1 (the tied whole-value input);
        // operand 2 is a sub-slice.
        if (UseMI.getOperand(1).getReg() == VReg)
          Worklist.push_back(UseMI.getOperand(0).getReg());
      }
    }
  }

  // Feasibility backstop: if the pinned set's own peak demand already exceeds
  // the VGPR budget, marking it not-spillable would force the allocator to
  // abort. Emit an optimization remark and leave the intervals spillable so
  // allocation degrades to a normal (spilled) result instead of failing.
  //
  // Use the VGPR-only share of the vector-register budget: on gfx90a/gfx942 the
  // file is split between VGPRs and AGPRs, and the pinned count is VGPR-only.
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  unsigned Budget = ST.getMaxNumVectorRegs(MF.getFunction()).first;
  unsigned Peak = peakPinnedVGPRPressure(Pinnable, TRI, MRI, LIS);
  if (Peak > Budget) {
    std::string Msg =
        ("VGPR pinning could not be honored: pinned values need " +
         Twine(Peak) + " VGPRs but only " + Twine(Budget) +
         " are available; affected values will be spilled")
            .str();
    MachineOptimizationRemarkEmitter MORE(MF, /*MBFI=*/nullptr);
    MORE.emit([&]() {
      return MachineOptimizationRemarkMissed(DEBUG_TYPE, "VGPRPinOverBudget",
                                             PinDL, PinMBB)
             << Msg;
    });
    LLVM_DEBUG(dbgs() << "[si-pin-vgpr] pinned peak " << Peak
                      << " exceeds budget " << Budget
                      << "; leaving intervals spillable\n");
    return true;
  }

  for (Register VReg : Pinnable) {
    LiveInterval &LI = LIS.getInterval(VReg);
    if (LI.isSpillable())
      LI.markNotSpillable();
    LLVM_DEBUG(dbgs() << "[si-pin-vgpr]   pinned vreg "
                      << printReg(VReg, nullptr)
                      << " (live interval marked not-spillable)\n");
  }

  return true;
}

PreservedAnalyses SIPinVGPRPass::run(MachineFunction &MF,
                                     MachineFunctionAnalysisManager &MFAM) {
  // Match the legacy pass's skipFunction(): leave optnone functions untouched.
  if (MF.getFunction().hasOptNone())
    return PreservedAnalyses::all();
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  SIPinVGPRImpl().run(MF, LIS);
  return PreservedAnalyses::all();
}
