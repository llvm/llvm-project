//===- SIWaitcntBranchPadding.cpp - Balance branch waitcnt histories -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "si-waitcnt-branch-padding"

static cl::opt<bool> EnableWaitcntBranchPadding(
    "amdgpu-waitcnt-branch-padding",
    cl::desc("Balance branch-local waitcnt histories at joins"),
    cl::init(false), cl::Hidden);

// This pass inserts NOPs to balance VM_CNT across branches. It only supports
// gfx942/gfx950 targets. The NOPs improve performance by allowing more relaxed
// `s_waitcnt vmcnt` instructions after branches. For example, prior to this
// pass the CFG might look like:
//
//             global_load_dword v0
//                /          \
//   global_load_dword v1   no VMEM
//                \          /
//              s_waitcnt vmcnt(0)
//                   use v0
//
// This pass then introduces `buffer_inv 0` to the no VMEM branch, allowing a
// relaxed s_waitcnt:
//
//             global_load_dword v0
//                /             \
//   global_load_dword v1   buffer_inv 0
//                \             /
//              s_waitcnt vmcnt(1)
//                   use v0
//
// This works because `buffer_inv 0` increments VM_CNT by 1 and is otherwise a
// NOP.

namespace {

// Generation identifies the common baseline for EventCount; only generation
// equality is meaningful.
//
//   common load             G1/E1
//       /    \
//    load    none           G1/E2, G1/E1: comparable
//
//   common load             G1/E1
//       /    \
// vmcnt(0); load  load      G2/E1, G1/E2: not comparable
struct WaitcntPaddingState {
  unsigned Generation = 0;
  unsigned EventCount = 0;
};

struct WaitcntEdgePadding {
  MachineBasicBlock *Pred = nullptr;
  MachineBasicBlock *Succ = nullptr;
  unsigned Count = 0;
};

struct WaitcntJoinPadding {
  SmallVector<WaitcntEdgePadding, 2> Edges;
};

class SIWaitcntBranchPadding {
  MachineFunction &MF;
  MachineLoopInfo &MLI;
  const GCNSubtarget &ST;
  const SIInstrInfo &TII;
  SmallVector<WaitcntJoinPadding, 4> Padding;
  unsigned NextGeneration = 1;
  bool ChangedCFG = false;

  WaitcntPaddingState startNewGeneration() { return {NextGeneration++, 0}; }

  unsigned getCounterMax() const {
    AMDGPU::HardwareLimits Limits(AMDGPU::getIsaVersion(ST.getCPU()));
    return Limits.LoadcntMax;
  }

  bool incrementsCounter(const MachineInstr &MI) const {
    if (TII.isFLAT(MI))
      return TII.mayAccessVMEMThroughFlat(MI);
    if (!SIInstrInfo::isVMEM(MI) || !SIInstrInfo::usesVM_CNT(MI))
      return false;
    return !AMDGPU::getMUBUFIsBufferInv(MI.getOpcode()) ||
           MI.getOpcode() == AMDGPU::BUFFER_INV ||
           MI.getOpcode() == AMDGPU::BUFFER_WBL2;
  }

  bool startsNewGeneration(const MachineInstr &MI) const;
  WaitcntPaddingState transferBlock(MachineBasicBlock &MBB,
                                    WaitcntPaddingState State,
                                    unsigned MaxEventCount);
  void emitPadding(MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
                   unsigned Count) const;
  bool plan();
  bool materialize();

public:
  SIWaitcntBranchPadding(MachineFunction &MF, MachineLoopInfo &MLI)
      : MF(MF), MLI(MLI), ST(MF.getSubtarget<GCNSubtarget>()),
        TII(*ST.getInstrInfo()) {}

  bool run() { return plan() && materialize(); }
  bool changedCFG() const { return ChangedCFG; }
};

class SIWaitcntBranchPaddingLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIWaitcntBranchPaddingLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    return SIWaitcntBranchPadding(MF, MLI).run();
  }

  StringRef getPassName() const override { return "SI Waitcnt Branch Padding"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

// Waits establish a new relative baseline; calls and inline asm make the prior
// LOAD_CNT history unknowable.
bool SIWaitcntBranchPadding::startsNewGeneration(const MachineInstr &MI) const {
  if (MI.isCall() || MI.isInlineAsm())
    return true;

  unsigned Opcode = SIInstrInfo::getNonSoftWaitcntOpcode(MI.getOpcode());
  if (Opcode == AMDGPU::WAIT_ASYNCMARK)
    return true;
  if (Opcode == AMDGPU::S_WAITCNT_lds_direct)
    return true;
  if (Opcode == AMDGPU::S_WAITCNT) {
    AMDGPU::Waitcnt Wait = AMDGPU::decodeWaitcnt(
        AMDGPU::getIsaVersion(ST.getCPU()), MI.getOperand(0).getImm());
    return Wait.get(AMDGPU::LOAD_CNT) != ~0u;
  }

  auto WaitCounter = AMDGPU::counterTypeForInstr(Opcode);
  return WaitCounter && *WaitCounter == AMDGPU::LOAD_CNT;
}

WaitcntPaddingState SIWaitcntBranchPadding::transferBlock(
    MachineBasicBlock &MBB, WaitcntPaddingState State, unsigned MaxEventCount) {
  for (MachineInstr &MI : MBB) {
    if (startsNewGeneration(MI)) {
      State = startNewGeneration();
      continue;
    }
    if (!incrementsCounter(MI))
      continue;
    if (State.EventCount == MaxEventCount) {
      // Make the overflowing event the new generation's implicit baseline. Its
      // event count remains zero, while paths that did not execute it retain a
      // different generation.
      State = startNewGeneration();
      continue;
    }
    ++State.EventCount;
  }
  return State;
}

void SIWaitcntBranchPadding::emitPadding(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator InsertPt,
                                         unsigned Count) const {
  DebugLoc DL = MBB.findDebugLoc(InsertPt);
  for (unsigned I = 0; I != Count; ++I)
    BuildMI(MBB, InsertPt, DL, TII.get(AMDGPU::BUFFER_INV)).addImm(0);
}

bool SIWaitcntBranchPadding::plan() {
  const bool IsEnabled =
      EnableWaitcntBranchPadding.getNumOccurrences()
          ? EnableWaitcntBranchPadding
          : MF.getFunction()
                .getFnAttribute("amdgpu-waitcnt-branch-padding")
                .getValueAsBool();
  if (!IsEnabled || !ST.hasGFX940Insts() || ST.hasVscnt() || !ST.isWave64() ||
      ST.isPreciseMemoryEnabled() || MF.hasBBSections())
    return false;

  StringRef CPU = ST.getCPU();
  if (CPU != "gfx942" && CPU != "gfx950")
    return false;

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  if (!MFI->isEntryFunction())
    return false;

  // Traverse the CFG once in reverse postorder and record edge-local LOAD_CNT
  // padding without mutating the CFG. At joins whose incoming states share a
  // generation and loop scope, pad each path to the largest event count.
  //
  // TODO: Extend this analysis and its edge plans to balance DS_CNT with DS_NOP
  // in the same RPO traversal.
  const unsigned MaxEventCount = getCounterMax() - 1;
  Padding.clear();
  NextGeneration = 1;

  DenseMap<MachineBasicBlock *, WaitcntPaddingState> Outgoing;
  for (MachineBasicBlock *MBB :
       ReversePostOrderTraversal<MachineFunction *>(&MF)) {
    SmallVector<MachineBasicBlock *, 4> Preds;
    SmallVector<WaitcntPaddingState, 4> Incoming;
    MachineLoop *Loop = MLI.getLoopFor(MBB);

    for (MachineBasicBlock *Pred : MBB->predecessors()) {
      auto It = Outgoing.find(Pred);
      if (It == Outgoing.end())
        continue;
      Preds.push_back(Pred);
      Incoming.push_back(It->second);
    }

    // Compare predecessor event counts only when RPO has already produced a
    // state for every predecessor and no incoming edge crosses a loop boundary.
    // Loop headers are excluded because their backedges are visited later.
    bool Comparable = MBB != &MF.front() && !MLI.isLoopHeader(MBB) &&
                      !Incoming.empty() && Incoming.size() == MBB->pred_size();
    if (Comparable) {
      unsigned Generation = Incoming.front().Generation;
      for (unsigned I = 0, E = Incoming.size(); I != E; ++I)
        Comparable &= Incoming[I].Generation == Generation &&
                      MLI.getLoopFor(Preds[I]) == Loop;
    }

    WaitcntPaddingState State;
    if (!Comparable) {
      // Establish a local baseline when no incoming state can be propagated,
      // letting downstream paths become comparable again.
      State = startNewGeneration();
    } else {
      State = Incoming.front();
      unsigned MinEventCount = State.EventCount;
      unsigned TargetEventCount = State.EventCount;
      for (const WaitcntPaddingState &In : drop_begin(Incoming)) {
        MinEventCount = std::min(MinEventCount, In.EventCount);
        TargetEventCount = std::max(TargetEventCount, In.EventCount);
      }

      if (Preds.size() > 1 && MinEventCount != TargetEventCount) {
        WaitcntJoinPadding JoinPlan;
        bool CanPad = true;
        // Validate every required edge split before recording the join so
        // padding is applied to all shorter paths or none of them.
        for (unsigned I = 0, E = Preds.size(); I != E; ++I) {
          unsigned Count = TargetEventCount - Incoming[I].EventCount;
          if (!Count)
            continue;
          MachineBasicBlock *Pred = Preds[I];
          if (Pred->succ_size() > 1 && !Pred->canSplitCriticalEdge(MBB, &MLI)) {
            CanPad = false;
            break;
          }
          JoinPlan.Edges.push_back({Pred, MBB, Count});
        }

        if (CanPad) {
          LLVM_DEBUG(dbgs()
                     << "Waitcnt branch padding join bb." << MBB->getNumber()
                     << ": target=" << TargetEventCount
                     << ", padded_edges=" << JoinPlan.Edges.size() << '\n');
          Padding.push_back(std::move(JoinPlan));
          // Propagate the virtual post-padding event count to downstream
          // blocks.
          State.EventCount = TargetEventCount;
        } else {
          State = startNewGeneration();
        }
      }
    }

    Outgoing[MBB] = transferBlock(*MBB, State, MaxEventCount);
  }

  return !Padding.empty();
}

bool SIWaitcntBranchPadding::materialize() {
  // Materialize the preflighted join plans with edge-local counter events.
  // Split multi-successor predecessor edges, then share padding suffixes
  // between edge-only blocks.
  MachineBasicBlock::SplitCriticalEdgeAnalyses Analyses{
      /*LIS=*/nullptr, /*SI=*/nullptr, /*LV=*/nullptr, /*MLI=*/&MLI};

  for (WaitcntJoinPadding &JoinPlan : Padding) {
    struct SharedPaddingBlock {
      WaitcntEdgePadding *Edge;
      MachineBasicBlock *MBB;
    };
    SmallVector<SharedPaddingBlock, 2> SharedBlocks;

    MachineBasicBlock *Join = JoinPlan.Edges.front().Succ;
    // Chaining padding blocks changes the join's direct predecessors. Avoid
    // rewriting machine PHIs by retaining independent blocks in that case.
    bool CanSharePadding = Join->empty() || !Join->front().isPHI();
    if (CanSharePadding)
      llvm::stable_sort(JoinPlan.Edges, [](const WaitcntEdgePadding &LHS,
                                           const WaitcntEdgePadding &RHS) {
        return LHS.Count > RHS.Count;
      });

    for (WaitcntEdgePadding &Edge : JoinPlan.Edges) {
      MachineBasicBlock *InsertBB = Edge.Pred;
      if (Edge.Pred->succ_size() > 1) {
        bool WasFallthrough = Edge.Pred->isLayoutSuccessor(Edge.Succ);
        InsertBB = Edge.Pred->SplitCriticalEdge(Edge.Succ, Analyses,
                                                /*LiveInSets=*/nullptr,
                                                /*MDTU=*/nullptr);
        if (!InsertBB) {
          // Downstream plans already include this join's virtual padding, so
          // continuing without the preflighted split would make their counts
          // unsound.
          report_fatal_error(
              "failed to split preflighted waitcnt padding edge");
        }
        ChangedCFG = true;

        if (!WasFallthrough) {
          // SplitCriticalEdge initially places the new block after Pred. Move
          // it away when the original edge was taken so Pred keeps its
          // fallthrough layout.
          MF.splice(MF.end(), InsertBB);
          Edge.Pred->updateTerminator(InsertBB);
        }
      }

      MachineBasicBlock::iterator InsertPt = InsertBB->getFirstTerminator();
      // Only edge-only blocks can be chained without executing another
      // predecessor's original instructions.
      if (CanSharePadding && InsertBB != Edge.Pred)
        SharedBlocks.push_back({&Edge, InsertBB});
      else
        emitPadding(*InsertBB, InsertPt, Edge.Count);
    }

    // A larger padding delta can reuse the next smaller suffix.
    for (unsigned I = 0, E = SharedBlocks.size(); I != E; ++I) {
      SharedPaddingBlock &PaddingBlock = SharedBlocks[I];
      unsigned SuffixCount = I + 1 == E ? 0 : SharedBlocks[I + 1].Edge->Count;
      MachineBasicBlock::iterator InsertPt =
          PaddingBlock.MBB->getFirstTerminator();
      unsigned Count = PaddingBlock.Edge->Count - SuffixCount;
      emitPadding(*PaddingBlock.MBB, InsertPt, Count);
      if (I + 1 != E) {
        MachineBasicBlock *NextMBB = SharedBlocks[I + 1].MBB;
        PaddingBlock.MBB->ReplaceUsesOfBlockWith(Join, NextMBB);
        PaddingBlock.MBB->updateTerminator(NextMBB);
      }
    }
  }

  return true;
}

INITIALIZE_PASS_BEGIN(SIWaitcntBranchPaddingLegacy, DEBUG_TYPE,
                      "SI Waitcnt Branch Padding", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(SIWaitcntBranchPaddingLegacy, DEBUG_TYPE,
                    "SI Waitcnt Branch Padding", false, false)

char SIWaitcntBranchPaddingLegacy::ID = 0;

char &llvm::SIWaitcntBranchPaddingID = SIWaitcntBranchPaddingLegacy::ID;

FunctionPass *llvm::createSIWaitcntBranchPaddingPass() {
  return new SIWaitcntBranchPaddingLegacy();
}

PreservedAnalyses
SIWaitcntBranchPaddingPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  auto &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
  SIWaitcntBranchPadding Padding(MF, MLI);
  if (!Padding.run())
    return PreservedAnalyses::all();

  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  if (!Padding.changedCFG())
    PA.preserveSet<CFGAnalyses>();
  return PA;
}
