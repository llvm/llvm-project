#include "AMDGPU.h"
#include "AMDGPUSSARAUtils.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"

#include <algorithm>
#include <stack>

#include "VRegMaskPair.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-rebuild-ssa"

namespace {

class AMDGPURebuildSSALegacy : public MachineFunctionPass {
  LiveIntervals *LIS = nullptr;
  MachineDominatorTree *MDT = nullptr;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineLoopInfo *MLI = nullptr;

  // Optional scratch; currently unused but kept for parity with your header.
  DenseMap<MachineOperand *, std::pair<MachineInstr *, LaneBitmask>>
      RegSeqences;

  //===--------------------------------------------------------------------===//
  // Internal helpers (now class methods)
  //===--------------------------------------------------------------------===//

  /// Return the VNInfo reaching this PHI operand along its predecessor edge.
  VNInfo *incomingOnEdge(LiveInterval &LI, MachineInstr *Phi,
                         MachineOperand &PhiOp) {
    unsigned OpIdx = Phi->getOperandNo(&PhiOp);
    MachineBasicBlock *Pred = Phi->getOperand(OpIdx + 1).getMBB();
    SlotIndex EndB = LIS->getMBBEndIdx(Pred);
    return LI.getVNInfoBefore(EndB);
  }

  /// True if \p UseMI’s operand is reached by \p VNI (PHIs, same-block order,
  /// cross-block dominance).
  bool reachedByThisVNI(LiveInterval &LI, MachineInstr *DefMI,
                        MachineInstr *UseMI, MachineOperand &UseOp,
                        VNInfo *VNI) {
    if (UseMI->isPHI())
      return incomingOnEdge(LI, UseMI, UseOp) == VNI;

    if (UseMI->getParent() == DefMI->getParent()) {
      SlotIndex DefIdx = LIS->getInstructionIndex(*DefMI);
      SlotIndex UseIdx = LIS->getInstructionIndex(*UseMI);
      return DefIdx < UseIdx; // strict within-block order
    }
    return MDT->dominates(DefMI->getParent(), UseMI->getParent());
  }

  /// What lanes does this operand read?
  LaneBitmask operandLaneMask(const MachineOperand &MO) const {
    if (unsigned Sub = MO.getSubReg())
      return TRI->getSubRegIndexLaneMask(Sub);
    return MRI->getMaxLaneMaskForVReg(MO.getReg());
  }

  /// Build a REG_SEQUENCE to materialize a super-reg/mixed-lane use.
  /// Inserts at the PHI predecessor terminator (for PHI uses) or right before
  /// UseMI otherwise. Returns the new full-width vreg, the RS index via OutIdx,
  /// and the subrange lane masks that should be extended to that point.
  Register buildRSForSuperUse(MachineInstr *UseMI, MachineOperand &MO,
                              Register OldVR, Register NewVR,
                              LaneBitmask MaskToRewrite, LiveInterval &LI,
                              const TargetRegisterClass *OpRC,
                              SlotIndex &OutIdx,
                              SmallVectorImpl<LaneBitmask> &LanesToExtend) {
    MachineBasicBlock *InsertBB = UseMI->getParent();
    MachineBasicBlock::iterator IP(UseMI);
    SlotIndex QueryIdx;

    if (UseMI->isPHI()) {
      unsigned OpIdx = UseMI->getOperandNo(&MO);
      MachineBasicBlock *Pred = UseMI->getOperand(OpIdx + 1).getMBB();
      InsertBB = Pred;
      IP = Pred->getFirstTerminator(); // ok if == end()
      QueryIdx = LIS->getMBBEndIdx(Pred).getPrevSlot();
    } else {
      QueryIdx = LIS->getInstructionIndex(*UseMI);
    }

    Register Dest = MRI->createVirtualRegister(OpRC);
    auto RS = BuildMI(*InsertBB, IP,
                      (IP != InsertBB->end() ? IP->getDebugLoc() : DebugLoc()),
                      TII->get(TargetOpcode::REG_SEQUENCE), Dest);

    SmallDenseSet<unsigned, 8> AddedSubIdxs;
    SmallDenseSet<LaneBitmask::Type, 8> AddedMasks;

    for (const LiveInterval::SubRange &SR : LI.subranges()) {
      if (!SR.getVNInfoAt(QueryIdx))
        continue;
      LaneBitmask Lane = SR.LaneMask;
      if (!AddedMasks.insert(Lane.getAsInteger()).second)
        continue;

      unsigned SubIdx = getSubRegIndexForLaneMask(Lane, TRI);
      if (!SubIdx || !AddedSubIdxs.insert(SubIdx).second)
        continue;

      if (Lane == MaskToRewrite)
        RS.addReg(NewVR).addImm(SubIdx);
      else
        RS.addReg(OldVR, 0, SubIdx).addImm(SubIdx);

      LanesToExtend.push_back(Lane);
    }

    // Fallback: ensure at least the rewritten lane appears.
    if (AddedSubIdxs.empty()) {
      unsigned SubIdx = getSubRegIndexForLaneMask(MaskToRewrite, TRI);
      RS.addReg(NewVR).addImm(SubIdx);
      LanesToExtend.push_back(MaskToRewrite);
    }

    LIS->InsertMachineInstrInMaps(*RS);
    OutIdx = LIS->getInstructionIndex(*RS);

#ifndef NDEBUG
    LLVM_DEBUG({
      dbgs() << "  [RS] inserted ";
      RS->print(dbgs());
    });
#endif
    return Dest;
  }

  /// Extend LI (and only the specified subranges) at Idx.
  void extendAt(LiveInterval &LI, SlotIndex Idx, ArrayRef<LaneBitmask> Lanes) {
    SmallVector<SlotIndex, 1> P{Idx};
    LIS->extendToIndices(LI, P);
    for (auto &SR : LI.subranges())
      for (LaneBitmask L : Lanes)
        if (SR.LaneMask == L)
          LIS->extendToIndices(SR, P);
  }

  //===--------------------------------------------------------------------===//
  // Public interface
  //===--------------------------------------------------------------------===//

  void buildRealPHI(VNInfo *VNI, LiveInterval &LI, Register OldVR);
  void splitNonPhiValue(VNInfo *VNI, LiveInterval &LI, Register OldVR);
  void rewriteUses(MachineInstr *DefMI, Register OldVR,
                   LaneBitmask MaskToRewrite, Register NewVR, LiveInterval &LI,
                   VNInfo *VNI);

public:
  static char ID;
  AMDGPURebuildSSALegacy() : MachineFunctionPass(ID) {
    initializeAMDGPURebuildSSALegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredTransitiveID(MachineDominatorsID);
    AU.addPreservedID(MachineDominatorsID);
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// buildRealPHI
// Create a whole- or sub-reg PHI for VNI at its block begin, then rewrite
// dominated uses to the PHI result. We require a *uniform* lane mask across
// all predecessors; if none found we treat it as full-width.
//===----------------------------------------------------------------------===//

void AMDGPURebuildSSALegacy::buildRealPHI(VNInfo *VNI, LiveInterval &LI,
                                          Register OldVR) {
  MachineBasicBlock *DefMBB = LIS->getMBBFromIndex(VNI->def);
  SmallVector<MachineOperand> Ops;
  const LaneBitmask FullMask = MRI->getMaxLaneMaskForVReg(OldVR);

  LaneBitmask CommonMask = LaneBitmask::getAll(); // intersection across preds
  LaneBitmask UnionMask = LaneBitmask::getNone();

#ifndef NDEBUG
  LLVM_DEBUG(dbgs() << "\n[PHI] Build PHI for " << printReg(OldVR) << " at MBB_"
                    << DefMBB->getNumber() << '\n');
#endif

  for (auto *Pred : DefMBB->predecessors()) {
    SlotIndex EndB = LIS->getMBBEndIdx(Pred);
    LaneBitmask EdgeMask = LaneBitmask::getNone();

    for (const LiveInterval::SubRange &SR : LI.subranges())
      if (SR.getVNInfoBefore(EndB))
        EdgeMask |= SR.LaneMask;

#ifndef NDEBUG
    const bool HasSubranges = !LI.subranges().empty();
    VNInfo *MainOut = LI.getVNInfoBefore(EndB); // whole-reg live-out?
    LLVM_DEBUG({
      dbgs() << "    subranges: " << (HasSubranges ? "yes" : "no")
             << ", main-range live-out: " << (MainOut ? "yes" : "no") << '\n';
    });
#endif

    if (EdgeMask.none()) {
#ifndef NDEBUG
      LLVM_DEBUG({
        dbgs() << "    EdgeMask is NONE; reason: ";
        if (LI.subranges().empty())
          dbgs() << "no subranges for this vreg";
        else if (LI.getVNInfoBefore(EndB))
          dbgs() << "subranges exist but none live at edge; main-range is "
                    "live-out";
        else
          dbgs() << "subranges exist and main-range not live-out (treating as "
                    "undef edge)";
        dbgs() << "\n";
      });
#endif

      // Current policy: treat “no subrange info” or “main-range live-out” as
      // full.
      if (LI.subranges().empty() || LI.getVNInfoBefore(EndB))
        EdgeMask = FullMask;
      else {
        // TODO:
        // Optional: if we prefer making the PHI operand explicitly undef on
        // this edge: keep EdgeMask == NONE and later add OldVR with
        // RegState::Undef or insert an IMPLICIT_DEF.
      }
    }
    CommonMask &= EdgeMask;
    UnionMask |= EdgeMask;

    unsigned SubIdx = AMDGPU::NoRegister;
    if ((FullMask & ~EdgeMask).any()) // partial register incoming
      SubIdx = getSubRegIndexForLaneMask(EdgeMask, TRI);

    Ops.push_back(MachineOperand::CreateReg(OldVR, /*isDef*/ false,
                                            /*isImp*/ false, /*isKill*/ false,
                                            /*isDead*/ false, /*isUndef*/ false,
                                            /*isEarlyClobber*/ false, SubIdx));
    Ops.push_back(MachineOperand::CreateMBB(Pred));
  }

  // Decide the lanes this PHI represents. If preds disagree, conservatively
  // use the union; otherwise the intersection equals the union.
  LaneBitmask PhiMask = (CommonMask.none() ? UnionMask : CommonMask);
  if (PhiMask.none())
    PhiMask = FullMask;

#ifndef NDEBUG
  LLVM_DEBUG(dbgs() << "  [PHI] final mask=" << PrintLaneMask(PhiMask) << '\n');
#endif

  const TargetRegisterClass *RC =
      TRI->getRegClassForOperandReg(*MRI, Ops.front());
  Register DestReg = MRI->createVirtualRegister(RC);

  auto PHINode = BuildMI(*DefMBB, DefMBB->begin(), DebugLoc(),
                         TII->get(TargetOpcode::PHI), DestReg)
                     .add(ArrayRef(Ops));
  MachineInstr *PHI = PHINode.getInstr();
  LIS->InsertMachineInstrInMaps(*PHI);

#ifndef NDEBUG
  LLVM_DEBUG({
    dbgs() << "  [PHI] inserted ";
    PHI->print(dbgs());
  });
#endif

  // Rewrite dominated uses to the PHI’s value.
  rewriteUses(PHI, OldVR, PhiMask, DestReg, LI, VNI);
  LIS->createAndComputeVirtRegInterval(DestReg);
}

//===----------------------------------------------------------------------===//
// splitNonPhiValue
// Turn a (non-PHI) value number into a new vreg definition, then rewrite
// dominated uses of the affected lanes to that new vreg.
//===----------------------------------------------------------------------===//

void AMDGPURebuildSSALegacy::splitNonPhiValue(VNInfo *VNI, LiveInterval &LI,
                                              Register OldVR) {
  MachineInstr *DefMI = LIS->getInstructionFromIndex(VNI->def);
  int OpIdx = DefMI->findRegisterDefOperandIdx(OldVR, TRI, /*IsDead*/ false,
                                               /*Overlaps*/ true);
  MachineOperand &MO = DefMI->getOperand(OpIdx);
  unsigned SubRegIdx = MO.getSubReg();

  LaneBitmask Mask = SubRegIdx ? TRI->getSubRegIndexLaneMask(SubRegIdx)
                               : MRI->getMaxLaneMaskForVReg(MO.getReg());
  const TargetRegisterClass *RC = TRI->getRegClassForOperandReg(*MRI, MO);

  Register NewVR = MRI->createVirtualRegister(RC);
  MO.setReg(NewVR);
  MO.setSubReg(AMDGPU::NoRegister);
  MO.setIsUndef(false); // keep partial-def semantics via subranges/uses
  LIS->ReplaceMachineInstrInMaps(*DefMI, *DefMI);

#ifndef NDEBUG
  LLVM_DEBUG({
    dbgs() << "[SPLIT] def ";
    DefMI->print(dbgs());
    dbgs() << "        lanes=" << PrintLaneMask(Mask) << " -> new vreg "
           << printReg(NewVR) << '\n';
  });
#endif

  rewriteUses(DefMI, OldVR, Mask, NewVR, LI, VNI);
  LIS->createAndComputeVirtRegInterval(NewVR);
}

//===----------------------------------------------------------------------===//
// rewriteUses
// For each use of OldVR reached by VNI:
//  * exact lane match → replace with NewVR,
//  * strict subset    → keep subindex, swap vreg,
//  * super/mixed      → build REG_SEQUENCE (OldVR for untouched lanes,
//                        NewVR for rewritten lanes), extend liveness,
//                        swap the operand.
//===----------------------------------------------------------------------===//

void AMDGPURebuildSSALegacy::rewriteUses(MachineInstr *DefMI, Register OldVR,
                                         LaneBitmask MaskToRewrite,
                                         Register NewVR, LiveInterval &LI,
                                         VNInfo *VNI) {
  const TargetRegisterClass *NewRC = TRI->getRegClassForReg(*MRI, NewVR);

#ifndef NDEBUG
  LLVM_DEBUG(dbgs() << "[RW] rewriting uses of " << printReg(OldVR)
                    << " lanes=" << PrintLaneMask(MaskToRewrite) << " with "
                    << printReg(NewVR) << '\n');
#endif

  for (MachineOperand &MO :
       llvm::make_early_inc_range(MRI->use_operands(OldVR))) {
    MachineInstr *UseMI = MO.getParent();
    if (UseMI == DefMI)
      continue;

    if (!reachedByThisVNI(LI, DefMI, UseMI, MO, VNI))
      continue;

    LaneBitmask OpMask = operandLaneMask(MO);
    if ((OpMask & MaskToRewrite).none())
      continue;

    const TargetRegisterClass *OpRC = TRI->getRegClassForOperandReg(*MRI, MO);

    // 1) Exact match fast path.
    if (OpMask == MaskToRewrite &&
        isOfRegClass(getRegSubRegPair(MO), *NewRC, *MRI)) {
#ifndef NDEBUG
      LLVM_DEBUG(dbgs() << "  [RW] exact -> " << printReg(NewVR) << " at ";
                 UseMI->print(dbgs()));
#endif
      MO.setReg(NewVR);
      MO.setSubReg(AMDGPU::NoRegister);
      continue;
    }

    // 2) Super/mixed vs subset split.
    if ((OpMask & ~MaskToRewrite).any()) {
      // SUPER/MIXED: build RS and swap.
      SmallVector<LaneBitmask, 4> LanesToExtend;
      SlotIndex RSIdx;
      Register RSv = buildRSForSuperUse(UseMI, MO, OldVR, NewVR, MaskToRewrite,
                                        LI, OpRC, RSIdx, LanesToExtend);
      extendAt(LI, RSIdx, LanesToExtend);
      MO.setReg(RSv);
      MO.setSubReg(AMDGPU::NoRegister);
    } else {
      // SUBSET: keep subindex, swap vreg.
      unsigned Sub = MO.getSubReg();
      assert(Sub && "subset path requires a subregister use");
#ifndef NDEBUG
      LLVM_DEBUG(dbgs() << "  [RW] subset sub" << Sub << " -> "
                        << printReg(NewVR) << " at ";
                 UseMI->print(dbgs()));
#endif
      MO.setReg(NewVR);
      MO.setSubReg(Sub);
    }
  }
}

//===----------------------------------------------------------------------===//
// runOnMachineFunction
// Walk all vregs, build a dominance-ordered worklist of main-range VNs.
// First materialize PHIs (post-dominance order), then split non-PHI values.
// Optionally prune LI afterwards.
//===----------------------------------------------------------------------===//

bool AMDGPURebuildSSALegacy::runOnMachineFunction(MachineFunction &MF) {
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  if (MRI->isSSA())
    return false;

#ifndef NDEBUG
  LLVM_DEBUG(dbgs() << "\n=== AMDGPURebuildSSALegacy on " << MF.getName()
                    << " ===\n");
#endif

  DenseSet<Register> Processed;

  for (auto &B : MF) {
    for (auto &I : B) {
      for (auto Def : I.defs()) {
        if (!Def.isReg() || !Def.getReg().isVirtual())
          continue;

        Register VReg = Def.getReg();
        if (!LIS->hasInterval(VReg) || !Processed.insert(VReg).second)
          continue;

        LiveInterval &LI = LIS->getInterval(VReg);
        if (LI.getNumValNums() == 1)
          continue;

#ifndef NDEBUG
        LLVM_DEBUG(dbgs() << "\n[VREG] " << printReg(VReg) << " has "
                          << LI.getNumValNums() << " VNs\n");
#endif

        // 1) Build worklist from the main range (1 VN per def site).
        SmallVector<VNInfo *, 8> WorkList;
        for (VNInfo *V : LI.vnis())
          if (V && !V->isUnused())
            WorkList.push_back(V);

        // 2) Sort by (dom-preorder, SlotIndex).
        auto DomKey = [&](VNInfo *V) {
          MachineBasicBlock *BB = LIS->getMBBFromIndex(V->def);
          static DenseMap<MachineBasicBlock *, unsigned> Num;
          if (Num.empty()) {
            unsigned N = 0;
            for (auto *Node : depth_first(MDT->getRootNode()))
              Num[Node->getBlock()] = N++;
          }
          return std::pair{Num[BB], V->def};
        };
        llvm::sort(WorkList,
                   [&](VNInfo *A, VNInfo *B) { return DomKey(A) < DomKey(B); });

#ifndef NDEBUG
        LLVM_DEBUG({
          dbgs() << "  [WL] order:\n";
          for (VNInfo *V : WorkList)
            dbgs() << "    id=" << V->id << " def=" << V->def
                   << (V->isPHIDef() ? " (phi)\n" : "\n");
        });
#endif

        // 3) Root dominates all others. Process PHIs first (post-dominating
        // order).
        VNInfo *Root = WorkList.front();
        auto IsPhi = [&](VNInfo *V) { return V != Root && V->isPHIDef(); };
        auto Mid =
            std::stable_partition(WorkList.begin(), WorkList.end(), IsPhi);

        auto PHISlice =
            llvm::ArrayRef(WorkList).take_front(Mid - WorkList.begin());
        for (auto It = PHISlice.rbegin(); It != PHISlice.rend(); ++It)
          buildRealPHI(*It, LI, VReg);

        // 4) Then split remaining non-PHI values, skipping the dominating root.
        for (VNInfo *VNI :
             llvm::ArrayRef(WorkList).slice(Mid - WorkList.begin())) {
          if (VNI == Root)
            continue;
          splitNonPhiValue(VNI, LI, VReg);
        }

        // 5) Single clean-up. (Keep prune optional; leave IsUndef on partial
        // defs.) LIS->shrinkToUses(&LI);
        // FIXME: For some reason shrinkToUses makes REG_SEQUENCE use
        // definitions dead!
        LI.RenumberValues();
      }
    }
  }

  Processed.clear();

  MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);
  MF.getProperties().reset(MachineFunctionProperties::Property::NoPHIs);

#ifndef NDEBUG
  LLVM_DEBUG(dbgs() << "=== verify ===\n");
#endif
  MF.verify();
  return MRI->isSSA();
}

char AMDGPURebuildSSALegacy::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPURebuildSSALegacy, DEBUG_TYPE, "AMDGPU Rebuild SSA",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AMDGPURebuildSSALegacy, DEBUG_TYPE, "AMDGPU Rebuild SSA",
                    false, false)

// Legacy PM registration
FunctionPass *llvm::createAMDGPURebuildSSALegacyPass() {
  return new AMDGPURebuildSSALegacy();
}

PreservedAnalyses
llvm::AMDGPURebuildSSAPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  AMDGPURebuildSSALegacy Impl;
  bool Changed = Impl.runOnMachineFunction(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  // TODO: We could detect CFG changed.
  auto PA = getMachineFunctionPassPreservedAnalyses();
  return PA;
}

llvm::PassPluginLibraryInfo getAMDGPURebuildSSAPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "AMDGPURebuildSSA", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, MachineFunctionPassManager &PM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "amdgpu-rebuild-ssa") {
                    PM.addPass(AMDGPURebuildSSAPass());
                    return true;
                  }
                  return false;
                });
          }};
}

// Expose the pass to LLVM’s pass manager infrastructure
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getAMDGPURebuildSSAPassPluginInfo();
}