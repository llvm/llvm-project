//===- AMDGPUVGPRMSBAffinity.cpp - VGPR MSB-group allocation hints --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// On gfx1250 a wave may use 1024 VGPRs, but an instruction only addresses
/// v0-v255; VGPRs 256-1023 are reached via per-slot MSB bits set by
/// S_SET_VGPR_MSB, which AMDGPULowerVGPREncoding emits whenever a slot's MSB
/// group changes between consecutive instructions.
///
/// This pre-RA pass (run after the scheduler fixes the order) records a desired
/// MSB group per virtual register; SIRegisterInfo's allocation-hint hook then
/// biases the greedy allocator toward it. The hint is soft, so it can never
/// make allocation fail. Steps: build a schedule-driven affinity graph (edges
/// between vregs that would cause a mode switch), cluster it under a per-group
/// register-pressure cap, and pack the clusters into MSB groups.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUVGPRMSBAffinity.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <climits>
#include <limits>
#include <optional>
#include <queue>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-vgpr-msb-affinity-hints"

static cl::opt<bool> EnableVGPRMSBAffinity(
    "amdgpu-vgpr-msb-affinity", cl::Hidden, cl::init(false),
    cl::desc("Bias VGPR allocation into 256-VGPR MSB groups to reduce "
             "S_SET_VGPR_MSB insertions (gfx1250)"));

static cl::opt<unsigned> BenefitPct(
    "amdgpu-vgpr-msb-affinity-benefit-pct", cl::Hidden, cl::init(75),
    cl::desc("Commit only if predicted plan switches < this % of the no-hint "
             "switches (self-benefit gate; 0 disables it)"));

static cl::opt<unsigned> MinBaseSwitch(
    "amdgpu-vgpr-msb-affinity-min-base-switch", cl::Hidden, cl::init(500),
    cl::desc("Skip when the naive baseline switch weight is below "
             "this"));

namespace {

constexpr unsigned MSBGroupSize = 256;
constexpr unsigned NumMSBGroups = 4;
// Skip the plan when a group's planned load exceeds this percent of its cap.
// Mild overflow is realizable (RA spills a few, most hints honored); severe
// overflow is not.
constexpr unsigned OverflowPct = 125;

// Weighted, undirected affinity graph over virtual-register indices. An edge's
// weight is the S_SET_VGPR_MSB cost paid if its two vregs land in different MSB
// groups; FirstOrdinal (earliest program point) is a deterministic clustering
// tie-break.
class AffinityGraph {
public:
  // Canonical (order-independent) key for the edge between vreg indices A and
  // B.
  static uint64_t makeKey(uint32_t A, uint32_t B) {
    if (A > B)
      std::swap(A, B);
    return (static_cast<uint64_t>(A) << HalfWidth) | B;
  }
  static uint32_t lowEnd(uint64_t Key) { return Key >> HalfWidth; }
  static uint32_t highEnd(uint64_t Key) { return Key & HalfMask; }

  // Accumulate Weight on the edge (A, B) and remember its earliest Ordinal.
  void addEdge(uint32_t A, uint32_t B, uint64_t Weight, unsigned Ordinal) {
    if (A == B)
      return;
    uint64_t Key = makeKey(A, B);
    Weights[Key] += Weight;
    FirstOrdinal.try_emplace(Key, Ordinal);
  }

  bool empty() const { return Weights.empty(); }
  unsigned size() const { return Weights.size(); }
  unsigned firstOrdinal(uint64_t Key) const { return FirstOrdinal.lookup(Key); }
  const DenseMap<uint64_t, uint64_t> &edges() const { return Weights; }

private:
  static constexpr unsigned HalfWidth = sizeof(uint32_t) * CHAR_BIT;
  static constexpr uint64_t HalfMask = (UINT64_C(1) << HalfWidth) - 1;

  DenseMap<uint64_t, uint64_t> Weights;
  DenseMap<uint64_t, unsigned> FirstOrdinal;
};

// Union-find over vreg indices that greedily merges the heaviest affinity
// edges, refusing a merge whose footprint would exceed a cap (one MSB group's
// pressure limit). Footprints come from an injected functor so it stays
// decoupled from LiveIntervals.
class ClusterForest {
public:
  using FootprintFn = function_ref<int(ArrayRef<Register>)>;

  ClusterForest(unsigned NumNodes, FootprintFn ComputeFootprint)
      : Parent(NumNodes), Rank(NumNodes, 0), Epoch(NumNodes, 0),
        Footprint(NumNodes, -1), Nodes(NumNodes),
        ComputeFootprint(ComputeFootprint) {
    for (unsigned I = 0; I < NumNodes; ++I)
      Parent[I] = I;
  }

  // Seed the singleton cluster for node Idx with register Reg.
  void addNode(unsigned Idx, Register Reg) { Nodes[Idx].push_back(Reg); }

  // Path compression only caches, hence const.
  unsigned find(unsigned X) const {
    while (Parent[X] != X) {
      Parent[X] = Parent[Parent[X]];
      X = Parent[X];
    }
    return X;
  }

  ArrayRef<Register> nodes(unsigned Root) const { return Nodes[Root]; }

  // Cached simultaneously-live footprint of the cluster rooted at Root.
  int footprintOf(unsigned Root) {
    if (Footprint[Root] < 0)
      Footprint[Root] = ComputeFootprint(Nodes[Root]);
    return Footprint[Root];
  }

  // Greedily merge edges heaviest-first (ties broken by smallest footprint
  // delta, then earliest ordinal, then key for determinism), refusing a merge
  // that would push the merged footprint past MergeCap. Each refused edge is a
  // cut.
  void clusterByWeight(const AffinityGraph &Graph, unsigned MergeCap) {
    struct Item {
      uint64_t Weight;
      int Delta;
      unsigned Ordinal;
      uint64_t Key;
      unsigned RootA, RootB;
      uint64_t EpochA, EpochB;
      // Max-heap ordering: an Item that should be processed first must compare
      // "greater" than the others.
      bool operator<(const Item &O) const {
        if (Weight != O.Weight)
          return Weight < O.Weight; // higher weight first
        if (Delta != O.Delta)
          return Delta > O.Delta; // smaller footprint delta first
        if (Ordinal != O.Ordinal)
          return Ordinal > O.Ordinal; // earlier program order first
        return Key > O.Key;           // lower key first (determinism)
      }
    };
    std::priority_queue<Item> Queue;
    // (Re-)evaluate the merge for an edge against the current forest and
    // enqueue it. Endpoints already in the same cluster are dropped.
    auto PushEdge = [&](uint64_t Key, uint64_t Weight) {
      unsigned RootA = find(AffinityGraph::lowEnd(Key));
      unsigned RootB = find(AffinityGraph::highEnd(Key));
      if (RootA == RootB)
        return;
      int Merged = unionFootprint(RootA, RootB);
      int Delta = Merged - std::max(footprintOf(RootA), footprintOf(RootB));
      Queue.push({Weight, Delta, Graph.firstOrdinal(Key), Key, RootA, RootB,
                  Epoch[RootA], Epoch[RootB]});
    };
    for (auto &[Key, Weight] : Graph.edges())
      PushEdge(Key, Weight);
    while (!Queue.empty()) {
      Item Top = Queue.top();
      Queue.pop();
      unsigned RootA = find(AffinityGraph::lowEnd(Top.Key));
      unsigned RootB = find(AffinityGraph::highEnd(Top.Key));
      if (RootA == RootB)
        continue;
      // A touched cluster changed since this item was pushed -> its delta/roots
      // are stale, so re-evaluate and re-enqueue rather than act on it.
      if (RootA != Top.RootA || RootB != Top.RootB ||
          Epoch[RootA] != Top.EpochA || Epoch[RootB] != Top.EpochB) {
        PushEdge(Top.Key, Top.Weight);
        continue;
      }
      int Merged = unionFootprint(RootA, RootB);
      if (Merged > static_cast<int>(MergeCap))
        continue; // Refuse: this edge becomes a cut.
      mergeInto(RootA, RootB, Merged);
    }
  }

private:
  // Exact union footprint of two clusters (time-aware peak of their nodes).
  int unionFootprint(unsigned RootA, unsigned RootB) {
    SmallVector<Register, 16> Both(Nodes[RootA].begin(), Nodes[RootA].end());
    Both.append(Nodes[RootB].begin(), Nodes[RootB].end());
    return ComputeFootprint(Both);
  }
  // Rank-union RootB into RootA (the higher-rank root is kept), fold nodes and
  // bump the kept root's epoch so stale queue items are detected.
  void mergeInto(unsigned RootA, unsigned RootB, int MergedFootprint) {
    if (Rank[RootA] < Rank[RootB])
      std::swap(RootA, RootB);
    Parent[RootB] = RootA;
    if (Rank[RootA] == Rank[RootB])
      ++Rank[RootA];
    Nodes[RootA].append(Nodes[RootB].begin(), Nodes[RootB].end());
    Nodes[RootB].clear();
    Footprint[RootA] = MergedFootprint;
    Footprint[RootB] = -1;
    ++Epoch[RootA];
  }

  mutable SmallVector<unsigned, 0> Parent;
  SmallVector<unsigned, 0> Rank;
  SmallVector<uint64_t, 0> Epoch;
  SmallVector<int, 0> Footprint; // cached per-root footprint, -1 = stale.
  SmallVector<SmallVector<Register, 4>, 0> Nodes;
  FootprintFn ComputeFootprint;
};

class AMDGPUVGPRMSBAffinity {
public:
  bool run(MachineFunction &MF, LiveIntervals *LIS, MachineLoopInfo *MLI);

private:
  // Cluster packing strategy: pack hottest-first into the lowest group that
  // fits (Compact) or into the least-loaded group so clusters spread and every
  // used group keeps slack for the soft hints (Balanced).
  enum class PackMode { Compact, Balanced };
  // Scope the self-benefit gate is scored over: the whole function, or only the
  // in-loop (recurring) switches.
  enum class GateScope { WholeFunction, LoopOnly };

  // Build the affinity graph, cluster, pack into MSB groups and commit hints
  // for one region (a set of blocks). Vregs already in \p Assigned (hinted by a
  // hotter region) are skipped; newly hinted vregs are added to it.
  void processRegion(ArrayRef<MachineBasicBlock *> Blocks,
                     ArrayRef<Register> AllVGPRs, unsigned EffMSBGroups,
                     unsigned VGPRBudget, PackMode Mode, GateScope Scope,
                     DenseSet<unsigned> &Assigned, SIMachineFunctionInfo *MFI);

  AffinityGraph buildAffinityGraph(ArrayRef<MachineBasicBlock *> Blocks) const;

  SmallVector<unsigned, 0> collectHotRoots(const ClusterForest &Forest,
                                           const AffinityGraph &Graph) const;

  void packClusters(const ClusterForest &Forest, ArrayRef<unsigned> Roots,
                    unsigned EffMSBGroups, unsigned VGPRBudget, PackMode Mode,
                    MutableArrayRef<unsigned> MSBLoad,
                    DenseMap<unsigned, unsigned> &ClusterMSB) const;

  // Per-group register cap. For power-of-two occupancy VGPRBudget is a whole
  // number of 256-groups so every cap is 256; at a fractional occupancy the
  // last group holds only VGPRBudget - (EffMSBGroups-1)*256 registers. Clamp to
  // [1, 256] so a fractional group is never over-packed (which drops a wave).
  static unsigned groupCap(unsigned Group, unsigned VGPRBudget) {
    int Cap =
        static_cast<int>(VGPRBudget) - static_cast<int>(Group * MSBGroupSize);
    return static_cast<unsigned>(std::max(1, std::min<int>(MSBGroupSize, Cap)));
  }

  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  LiveIntervals *LIS = nullptr;
  MachineLoopInfo *MLI = nullptr;
  const GCNSubtarget *STI = nullptr;

  // Per-block edge weight: a loop-depth proxy for trip count, so an
  // innermost-loop transition outweighs straight-line code by orders of
  // magnitude.
  uint64_t blockFreq(const MachineBasicBlock &MBB) const {
    unsigned Depth = MLI ? MLI->getLoopDepth(&MBB) : 0;
    return 1ull << std::min(4u * Depth, 40u);
  }

  // Value-group union-find (mutable for path compression): vregs that coalesce
  // to one physreg must be counted once in the footprint. See buildValueGroups.
  mutable SmallVector<unsigned, 0> VGParent;

  bool isVGPRVirtReg(Register Reg) const {
    return Reg.isVirtual() && TRI->isVGPRClass(MRI->getRegClass(Reg));
  }

  unsigned dwords(Register Reg) const {
    // Integer-divide by 32 (a 16-bit vreg -> 0): rounding up over-counts
    // lo16/hi16 pairs that share a dword and worsens True16 plans. The
    // footprint only feeds soft hints, so the undercount is acceptable.
    return TRI->getRegSizeInBits(*MRI->getRegClass(Reg)) / 32;
  }

  // Record the MSB-group affinity and also a concrete physreg hint in that
  // group: the latter marks a known preference so greedy colors the vreg early
  // and it claims its group before contention. Existing (copy) hints win.
  void recordMSB(SIMachineFunctionInfo *MFI, Register Reg, unsigned MSB) {
    MFI->setVGPRMSBAffinity(Reg, MSB);
    if (MRI->getRegAllocationHint(Reg).second)
      return;
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    for (MCPhysReg P : *RC) {
      if (!MRI->isReserved(P) && (TRI->getHWRegIndex(P) >> 8) == MSB) {
        MRI->setRegAllocationHint(Reg, 0, P);
        return;
      }
    }
  }

  unsigned vgFind(unsigned X) const {
    while (VGParent[X] != X) {
      VGParent[X] = VGParent[VGParent[X]];
      X = VGParent[X];
    }
    return X;
  }

  void buildValueGroups(MachineFunction &MF) {
    unsigned N = MRI->getNumVirtRegs();
    VGParent.resize(N);
    for (unsigned I = 0; I < N; ++I)
      VGParent[I] = I;
    auto UnionVGroup = [&](Register A, Register B) {
      if (!isVGPRVirtReg(A) || !isVGPRVirtReg(B))
        return;
      unsigned RootA = vgFind(A.virtRegIndex()),
               RootB = vgFind(B.virtRegIndex());
      if (RootA != RootB)
        VGParent[RootA] = RootB;
    };
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        // Coalesce tied def/use pairs (e.g. the WMMA accumulator src2 tied to
        // dst). General COPYs are intentionally *not* unioned: they connect
        // distinct values and would collapse unrelated footprints.
        for (unsigned I = 0, E = MI.getNumOperands(); I < E; ++I) {
          const MachineOperand &MO = MI.getOperand(I);
          if (MO.isReg() && MO.isUse() && MO.isTied()) {
            unsigned DefIdx = MI.findTiedOperandIdx(I);
            const MachineOperand &Def = MI.getOperand(DefIdx);
            if (Def.isReg())
              UnionVGroup(MO.getReg(), Def.getReg());
          }
        }
        // Coalesce the accumulator chain dst <- src2: across an unrolled K-loop
        // this chains acc0->acc1->... into one value group so the footprint
        // counts the accumulator once. Disjoint output tiles never merge.
        if (SIInstrInfo::isWMMA(MI) || TII->isMAI(MI)) {
          const MachineOperand *D =
              TII->getNamedOperand(MI, AMDGPU::OpName::vdst);
          const MachineOperand *S2 =
              TII->getNamedOperand(MI, AMDGPU::OpName::src2);
          if (D && D->isReg() && S2 && S2->isReg())
            UnionVGroup(D->getReg(), S2->getReg());
        }
      }
    }
  }

  // Peak simultaneously-live VGPR dwords in \p Regs, merging the live ranges of
  // a value group so a coalescing value is counted once.
  unsigned maxSimultaneousDwords(ArrayRef<Register> Regs) const {
    DenseMap<unsigned, SmallVector<std::pair<SlotIndex, SlotIndex>, 2>> ByGroup;
    DenseMap<unsigned, int> GroupSize;
    for (Register Reg : Regs) {
      if (!LIS->hasInterval(Reg))
        continue;
      unsigned G = vgFind(Reg.virtRegIndex());
      GroupSize[G] = std::max<int>(GroupSize[G], dwords(Reg));
      auto &Segs = ByGroup[G];
      for (const LiveRange::Segment &S : LIS->getInterval(Reg))
        Segs.emplace_back(S.start, S.end);
    }
    SmallVector<std::pair<SlotIndex, int>, 64> Events;
    for (auto &[G, Segs] : ByGroup) {
      llvm::sort(Segs);
      int Sz = GroupSize[G];
      SlotIndex CurS, CurE;
      bool Open = false;
      auto Flush = [&] {
        Events.emplace_back(CurS, Sz);
        Events.emplace_back(CurE, -Sz);
      };
      for (auto &[S, E] : Segs) {
        if (Open && S <= CurE) {
          CurE = std::max(CurE, E); // overlaps: extend the open interval
        } else {
          if (Open) // gap: close the previous interval
            Flush();
          CurS = S;
          CurE = E;
          Open = true;
        }
      }
      if (Open)
        Flush();
    }
    llvm::sort(Events, [](const std::pair<SlotIndex, int> &A,
                          const std::pair<SlotIndex, int> &B) {
      return A.first < B.first || (A.first == B.first && A.second < B.second);
    });
    int Cur = 0, Max = 0;
    for (auto &[Idx, Delta] : Events) {
      Cur += Delta;
      Max = std::max(Max, Cur);
    }
    return Max;
  }

  // Natural (no-hint) MSB-group assignment for the self-benefit baseline: a
  // linear scan placing each vreg in the lowest free column run and freeing
  // columns as live ranges end -- an approximation of what the allocator does
  // unhinted.
  DenseMap<unsigned, int> computeNaiveMSB(ArrayRef<Register> Regs,
                                          unsigned EffMSBGroups) const {
    DenseMap<unsigned, int> MSB;
    const unsigned Cols = EffMSBGroups * MSBGroupSize;
    SmallVector<Register, 0> Order(Regs.begin(), Regs.end());
    llvm::stable_sort(Order, [&](Register A, Register B) {
      return LIS->getInterval(A).beginIndex() <
             LIS->getInterval(B).beginIndex();
    });
    SmallVector<bool, 0> Free(Cols, true);
    // Active allocations: (endIndex, startCol, width) to reclaim columns.
    SmallVector<std::tuple<SlotIndex, unsigned, unsigned>, 0> Active;
    for (Register R : Order) {
      SlotIndex Begin = LIS->getInterval(R).beginIndex();
      // Reclaim columns of ranges that ended before this def.
      for (unsigned I = 0; I < Active.size();) {
        if (std::get<0>(Active[I]) <= Begin) {
          unsigned StartCol = std::get<1>(Active[I]);
          unsigned RunWidth = std::get<2>(Active[I]);
          for (unsigned Col = StartCol; Col < StartCol + RunWidth; ++Col)
            Free[Col] = true;
          Active[I] = Active.back();
          Active.pop_back();
        } else
          ++I;
      }
      unsigned Width = dwords(R);
      // Lowest free run of Width columns.
      int Start = -1;
      for (unsigned Col = 0, Run = 0; Col < Cols; ++Col) {
        Run = Free[Col] ? Run + 1 : 0;
        if (Run == Width) {
          Start = static_cast<int>(Col + 1 - Width);
          break;
        }
      }
      int Group;
      if (Start < 0) {
        // No contiguous run fits: reserve Width columns in the least-occupied
        // MSB group so this vreg's footprint stays visible (otherwise the
        // baseline looks artificially uncongested and skews the self-benefit
        // comparison).
        unsigned BestGroup = 0, BestFreeCount = 0;
        for (unsigned Cand = 0; Cand < EffMSBGroups; ++Cand) {
          unsigned FreeCount = 0;
          for (unsigned Col = Cand * MSBGroupSize;
               Col < (Cand + 1) * MSBGroupSize; ++Col)
            FreeCount += Free[Col];
          if (FreeCount >= BestFreeCount) {
            BestFreeCount = FreeCount;
            BestGroup = Cand;
          }
        }
        for (unsigned Col = BestGroup * MSBGroupSize, Reserved = 0;
             Col < (BestGroup + 1) * MSBGroupSize && Reserved < Width; ++Col)
          if (Free[Col]) {
            Free[Col] = false;
            ++Reserved;
          }
        Group = static_cast<int>(BestGroup);
      } else {
        for (unsigned Col = Start; Col < Start + Width; ++Col)
          Free[Col] = false;
        Active.emplace_back(LIS->getInterval(R).endIndex(),
                            static_cast<unsigned>(Start), Width);
        Group = Start / static_cast<int>(MSBGroupSize);
      }
      MSB[R.virtRegIndex()] = Group;
    }
    return MSB;
  }

  // Predicted freq-weighted s_set_vgpr_msb count for a vreg->MSB-group map,
  // simulated like AMDGPULowerVGPREncoding: walk the stream with sticky
  // per-slot state (reset per block), charging blockFreq per instruction that
  // changes a slot's group.
  uint64_t simSwitchWeight(ArrayRef<MachineBasicBlock *> Blocks,
                           function_ref<int(Register)> MsbOf,
                           bool LoopOnly = false) const {
    uint64_t Sw = 0;
    for (MachineBasicBlock *MBBp : Blocks) {
      MachineBasicBlock &MBB = *MBBp;
      // Realizability/relevance: only in-loop switches recur every iteration
      // and dominate runtime cost; prologue/epilogue switches fire once.
      // Scoring the gate on loop blocks only keeps the plan from trading a loop
      // win for one-time out-of-loop churn (which the whole-function total
      // misranks).
      if (LoopOnly && (!MLI || MLI->getLoopDepth(&MBB) == 0))
        continue;
      uint64_t Freq = blockFreq(MBB);
      // Mode is reset to group 0 at a block header (and again at a call /
      // terminator / VGPR inline asm), matching AMDGPULowerVGPREncoding.
      int Last[4] = {0, 0, 0, 0};
      for (MachineInstr &MI : MBB) {
        if (MI.isMetaInstruction())
          continue;
        if (MI.isTerminator() || MI.isCall() ||
            (MI.isInlineAsm() && TII->hasVGPRUses(MI))) {
          Last[0] = Last[1] = Last[2] = Last[3] = 0;
          continue;
        }
        auto Ops = AMDGPU::getVGPRLoweringOperandTables(MI.getDesc());
        if (!Ops.first)
          continue;
        int Need[4] = {-1, -1, -1, -1};
        for (unsigned S = 0; S < 4; ++S) {
          const MachineOperand *MO = TII->getNamedOperand(MI, Ops.first[S]);
          if ((!MO || !MO->isReg() || !MO->getReg()) && Ops.second)
            MO = TII->getNamedOperand(MI, Ops.second[S]);
          if (!MO || !MO->isReg() || !MO->getReg())
            continue;
          Register R = MO->getReg();
          if (isVGPRVirtReg(R))
            Need[S] = std::max(0, MsbOf(R));
          else if (R.isPhysical() && TRI->isVGPR(*MRI, R))
            Need[S] = static_cast<int>(TRI->getHWRegIndex(R) >> 8);
        }
        bool Changed = false;
        for (unsigned S = 0; S < 4; ++S)
          if (Need[S] >= 0 && Last[S] != Need[S])
            Changed = true;
        if (Changed)
          Sw += Freq;
        for (unsigned S = 0; S < 4; ++S)
          if (Need[S] >= 0)
            Last[S] = Need[S];
      }
    }
    return Sw;
  }
};

class AMDGPUVGPRMSBAffinityLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUVGPRMSBAffinityLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto *LISW = getAnalysisIfAvailable<LiveIntervalsWrapperPass>();
    auto *MLIW = getAnalysisIfAvailable<MachineLoopInfoWrapperPass>();
    return AMDGPUVGPRMSBAffinity().run(MF, LISW ? &LISW->getLIS() : nullptr,
                                       MLIW ? &MLIW->getLI() : nullptr);
  }

  StringRef getPassName() const override { return "AMDGPU VGPR MSB Affinity"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

bool AMDGPUVGPRMSBAffinity::run(MachineFunction &MF, LiveIntervals *LISIn,
                                MachineLoopInfo *MLIIn) {
  if (!EnableVGPRMSBAffinity)
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.has1024AddressableVGPRs() || !LISIn)
    return false;

  // Only steer compute kernels; graphics shaders are out of scope for the
  // 1024-VGPR / s_set_vgpr_msb MSB grouping this pass targets.
  if (!AMDGPU::isCompute(MF.getFunction().getCallingConv()))
    return false;

  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();
  LIS = LISIn;
  MLI = MLIIn;
  STI = &ST;

  LLVM_DEBUG(dbgs() << "*** AMDGPUVGPRMSBAffinity on " << MF.getName()
                    << " ***\n");

  // Coalesce vregs that will share one physreg so the footprint counts them
  // once.
  buildValueGroups(MF);

  // If the whole function fits one 256-VGPR group, no S_SET_VGPR_MSB is ever
  // needed and partitioning would only inflate the VGPR count.
  SmallVector<Register, 0> AllVGPRs;
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (!MRI->reg_nodbg_empty(R) && isVGPRVirtReg(R))
      AllVGPRs.push_back(R);
  }
  unsigned GlobalFP = maxSimultaneousDwords(AllVGPRs);
  LLVM_DEBUG(dbgs() << "  early-check GlobalFP=" << GlobalFP << "\n");
  if (GlobalFP <= MSBGroupSize) {
    LLVM_DEBUG(dbgs() << "  -> return: footprint fits one group\n");
    return false;
  }

  // Baseline occupancy: min of the VGPR-limited estimate and MFI's non-VGPR
  // limit.
  const SIMachineFunctionInfo *MFIOcc = MF.getInfo<SIMachineFunctionInfo>();
  unsigned VOcc = STI->getOccupancyWithNumVGPRs(
      GlobalFP, MFIOcc->getDynamicVGPRBlockSize());
  unsigned BaseOcc = std::min(VOcc, MFIOcc->getOccupancy());
  LLVM_DEBUG(dbgs() << "  early-check BaseOcc=" << BaseOcc << " (VOcc=" << VOcc
                    << " MFIOcc=" << MFIOcc->getOccupancy() << ")\n");
  if (BaseOcc == 0)
    return false;

  // As many MSB groups as occupancy allows (NumMSBGroups/BaseOcc), but at least
  // what the footprint needs; extra groups cost only VGPRs, free under the occ
  // limit.
  unsigned Needed = (GlobalFP + MSBGroupSize - 1) / MSBGroupSize;
  const unsigned EffMSBGroups =
      std::min(NumMSBGroups, std::max(Needed, NumMSBGroups / BaseOcc));

  unsigned VGPRBudget =
      STI->getMaxNumVGPRs(BaseOcc, MFIOcc->getDynamicVGPRBlockSize());

  LLVM_DEBUG(dbgs() << "  GlobalFP(true RP)=" << GlobalFP << " BaseOcc="
                    << BaseOcc << " EffMSBGroups=" << EffMSBGroups
                    << " VGPRBudget=" << VGPRBudget << "\n");

  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  DenseSet<unsigned> Assigned;
  SmallVector<MachineBasicBlock *, 16> Blocks;
  for (MachineBasicBlock &MBB : MF)
    Blocks.push_back(&MBB);

  auto TryStage = [&](PackMode Mode, GateScope Scope) {
    processRegion(Blocks, AllVGPRs, EffMSBGroups, VGPRBudget, Mode, Scope,
                  Assigned, MFI);
    return !Assigned.empty();
  };

  // Compact packing handles most kernels; fall back to balanced packing with an
  // in-loop gate only when compact commits nothing (it recovers near-full occ-1
  // and fractional occ-3 kernels without disturbing the rest).
  if (!TryStage(PackMode::Compact, GateScope::WholeFunction))
    TryStage(PackMode::Balanced, GateScope::LoopOnly);

  return false;
}

// An edge weight is the block frequency summed over the points where the two
// vregs occupy the same MSB slot back to back -- where a mode switch is paid
// unless they share a MSB group.
AffinityGraph AMDGPUVGPRMSBAffinity::buildAffinityGraph(
    ArrayRef<MachineBasicBlock *> Blocks) const {
  AffinityGraph Graph;

  // Add the edge (A, B), scaling its weight by the operand width (capped at 8)
  // so a wide value -- e.g. the WMMA accumulator -- that alternates in a slot
  // outweighs a scalar doing the same.
  auto AddScaledEdge = [&](Register A, Register B, uint64_t Weight,
                           unsigned Ordinal) {
    if (A.virtRegIndex() == B.virtRegIndex())
      return;
    unsigned Width = std::min({dwords(A), dwords(B), 8u});
    Weight *= std::max(1u, Width);
    Graph.addEdge(A.virtRegIndex(), B.virtRegIndex(), Weight, Ordinal);
  };

  unsigned Ordinal = 0; // Monotonic instruction ordinal (program order).
  for (MachineBasicBlock *MBBp : Blocks) {
    MachineBasicBlock &MBB = *MBBp;
    uint64_t Freq = blockFreq(MBB);

    // Sticky per-slot state, reset at each block (the lowering pass resets the
    // mode at block boundaries).
    Register LastInSlot[4];
    bool PrevDsRead = false;   // Previous real instr was a ds_read.
    unsigned PrevDsDstLen = 0; // That ds_read's vdst tuple width (dwords).

    for (MachineInstr &MI : MBB) {
      if (MI.isMetaInstruction())
        continue;
      ++Ordinal;
      bool ThisDsRead = TII->isDS(MI) && MI.mayLoad();

      // A VGPR-to-VGPR COPY is likely coalesced by the allocator (its copy hint
      // outranks ours), landing both in one MSB group; add an edge so our plan
      // agrees. Weight Freq*144 matches a single-slot boundary so the copy is
      // not under-weighted.
      if (MI.isCopy()) {
        const MachineOperand &Dst = MI.getOperand(0), &Src = MI.getOperand(1);
        if (Dst.isReg() && Src.isReg() && isVGPRVirtReg(Dst.getReg()) &&
            isVGPRVirtReg(Src.getReg()))
          AddScaledEdge(Dst.getReg(), Src.getReg(), Freq * 144, Ordinal);
      }

      auto Ops = AMDGPU::getVGPRLoweringOperandTables(MI.getDesc());
      if (!Ops.first) {
        PrevDsRead = ThisDsRead;
        continue;
      }

      // Width of this instruction's dst tuple (vdst = slot 3), used to scale
      // the src0 boost at a ds_read boundary ("boost to dst len").
      unsigned ThisDstLen = 0;
      if (const MachineOperand *D = TII->getNamedOperand(MI, Ops.first[3])) {
        if ((!D->isReg() || !D->getReg()) && Ops.second)
          D = TII->getNamedOperand(MI, Ops.second[3]);
        if (D && D->isReg() && D->getReg() && isVGPRVirtReg(D->getReg()))
          ThisDstLen = dwords(D->getReg());
      }

      // Per-slot edges, not deduplicated: a tied accumulator drives both src2
      // and dst and so contributes 2*Freq to its pair, intentionally
      // emphasizing the accumulator/dst chain over single-slot src0/src1 edges.
      SmallVector<std::tuple<Register, Register, unsigned>, 4> Changed;
      for (unsigned Slot = 0; Slot < 4; ++Slot) {
        const MachineOperand *MO = TII->getNamedOperand(MI, Ops.first[Slot]);
        if ((!MO || !MO->isReg() || !MO->getReg()) && Ops.second)
          MO = TII->getNamedOperand(MI, Ops.second[Slot]);
        if (!MO || !MO->isReg() || !MO->getReg())
          continue; // Slot not constrained: stays sticky.

        Register R = MO->getReg();
        if (isVGPRVirtReg(R)) {
          if (LastInSlot[Slot] && LastInSlot[Slot] != R)
            Changed.emplace_back(LastInSlot[Slot], R, Slot);
          LastInSlot[Slot] = R;
        } else if (R.isPhysical() && TRI->isVGPR(*MRI, R)) {
          // A physical VGPR pins the slot to a fixed MSB group; break the run
          // so we do not attract vregs across it.
          LastInSlot[Slot] = Register();
        }
        // SGPR / immediate operands leave the slot sticky.
      }
      // Charge the boundary once (one s_set_vgpr_msb covers all changed slots)
      // and distribute its cost across the k changed slots as weight/k, so
      // co-locating a slot that is the sole changer (k=1) is valued fully while
      // batched multi-slot boundaries are discounted. Base 144 = 12^2 keeps the
      // per-slot weight integral for k in 1..4.
      if (!Changed.empty()) {
        uint64_t BoundaryWeight = (Freq * 144) / Changed.size();
        bool DsBoundary = PrevDsRead || ThisDsRead;
        // Boost factor for the src0 edge at a ds_read boundary: the ds_read's
        // dst tuple width ("boost to dst len").
        unsigned DstLen = ThisDsRead ? ThisDstLen : PrevDsDstLen;
        // Isolation count for the src0 gate excludes the dst slot: a ds_read
        // writes a fresh tile in the dst slot, so a WMMA->ds_read boundary
        // looks like it changes dst (different vregs) even though both tiles
        // usually land in the same group post-RA. Counting dst would
        // misclassify these as multi-slot and skip the boost, leaving the
        // ds_read address stranded in another group (the src0 g0<->g1 flip).
        // Count only src0/src1/src2.
        unsigned ChangedNonDst = 0;
        for (auto &[LhsReg, RhsReg, Slot] : Changed)
          if (Slot != 3)
            ++ChangedNonDst;
        for (auto &[LhsReg, RhsReg, Slot] : Changed) {
          uint64_t SlotWeight = BoundaryWeight;
          // Only boost src0 when it is the sole non-dst changer at the
          // boundary: co-locating the address removes a switch only then; on a
          // boundary where src1/src2 also change, the switch is paid anyway.
          // Boost by the ds_read dst width, capped at 2.
          if (Slot == 0 && DsBoundary && ChangedNonDst <= 1)
            SlotWeight *= std::max(1u, std::min(DstLen, 2u));
          AddScaledEdge(LhsReg, RhsReg, SlotWeight, Ordinal);
        }
      }
      PrevDsRead = ThisDsRead;
      if (ThisDsRead)
        PrevDsDstLen = ThisDstLen;
    }
  }
  return Graph;
}

// Cluster roots carrying real (loop-level) affinity, hottest-first.
SmallVector<unsigned, 0>
AMDGPUVGPRMSBAffinity::collectHotRoots(const ClusterForest &Forest,
                                       const AffinityGraph &Graph) const {
  // Cluster weight = total internal affinity (how costly it is to split).
  DenseMap<unsigned, uint64_t> ClusterWeight;
  for (auto &[Key, Weight] : Graph.edges()) {
    unsigned RootA = Forest.find(AffinityGraph::lowEnd(Key));
    unsigned RootB = Forest.find(AffinityGraph::highEnd(Key));
    if (RootA == RootB)
      ClusterWeight[RootA] += Weight;
  }

  // Only steer clusters that carry real (loop-level) affinity. Cold registers
  // with no significant same-slot neighbours are left unhinted so the allocator
  // packs them naturally instead of being forced into a MSB group.
  uint64_t MaxWeight = 0;
  for (const auto &[_, Weight] : ClusterWeight)
    MaxWeight = std::max(MaxWeight, Weight);
  uint64_t WeightCutoff = MaxWeight / 4;

  SmallVector<unsigned, 0> Roots;
  const unsigned N = MRI->getNumVirtRegs();
  for (unsigned I = 0; I < N; ++I)
    if (Forest.find(I) == I && !Forest.nodes(I).empty() &&
        ClusterWeight.lookup(I) > WeightCutoff)
      Roots.push_back(I);
  // Sort hottest-first: the most important clusters are placed first. First-fit
  // then puts them in the low MSB groups -- which matters because group 0 is
  // special: AMDGPULowerVGPREncoding resets the mode to all-zero at
  // non-fall-through block entries (including the loop header every iteration),
  // so a value in group 0 needs no switch right after a reset. Keeping the
  // hottest clusters in group 0 therefore minimizes switches.
  llvm::stable_sort(Roots, [&](unsigned A, unsigned B) {
    return ClusterWeight.lookup(A) > ClusterWeight.lookup(B);
  });
  return Roots;
}

// Pack the hottest-first Roots into EffMSBGroups groups. A group's load is the
// peak simultaneously-live footprint of its clusters, not the sum of their
// peaks, so time-disjoint clusters (e.g. successive prefetch tiles) share a
// group instead of each reserving 256 registers -- this is what keeps the VGPR
// count from ballooning on software-pipelined kernels.
void AMDGPUVGPRMSBAffinity::packClusters(
    const ClusterForest &Forest, ArrayRef<unsigned> Roots,
    unsigned EffMSBGroups, unsigned VGPRBudget, PackMode Mode,
    MutableArrayRef<unsigned> MSBLoad,
    DenseMap<unsigned, unsigned> &ClusterMSB) const {
  SmallVector<SmallVector<Register, 0>, 8> GroupNodes(EffMSBGroups);
  for (unsigned Root : Roots) {
    ArrayRef<Register> RootNodes = Forest.nodes(Root);
    // Compact: first-fit into the lowest group that fits (keeps hot clusters in
    // group 0, which is reset-free after the loop header). Balanced: place in
    // the group that minimizes the resulting load, so clusters spread across
    // all groups and every used group keeps slack for the soft hints.
    std::optional<unsigned> Best;
    unsigned BestLoad = 0;
    unsigned BestResult = std::numeric_limits<unsigned>::max();
    for (unsigned Group = 0; Group < EffMSBGroups; ++Group) {
      SmallVector<Register, 16> Combined(GroupNodes[Group].begin(),
                                         GroupNodes[Group].end());
      Combined.append(RootNodes.begin(), RootNodes.end());
      unsigned Load = maxSimultaneousDwords(Combined);
      if (Mode == PackMode::Compact) {
        if (Load <= groupCap(Group, VGPRBudget)) {
          Best = Group;
          BestLoad = Load;
          break; // lowest MSB group that fits
        }
      } else if (Load <= groupCap(Group, VGPRBudget) && Load < BestResult) {
        BestResult = Load;
        Best = Group;
        BestLoad = Load;
      }
    }
    if (!Best) {
      // No MSB group fits this cluster within capacity; over-subscribe the
      // least-loaded one (the soft hint may spill).
      Best = llvm::min_element(MSBLoad) - MSBLoad.begin();
      GroupNodes[*Best].append(RootNodes.begin(), RootNodes.end());
      BestLoad = maxSimultaneousDwords(GroupNodes[*Best]);
    } else {
      GroupNodes[*Best].append(RootNodes.begin(), RootNodes.end());
    }
    MSBLoad[*Best] = BestLoad;
    ClusterMSB[Root] = *Best;
  }
}

void AMDGPUVGPRMSBAffinity::processRegion(
    ArrayRef<MachineBasicBlock *> Blocks, ArrayRef<Register> AllVGPRs,
    unsigned EffMSBGroups, unsigned VGPRBudget, PackMode Mode, GateScope Scope,
    DenseSet<unsigned> &Assigned, SIMachineFunctionInfo *MFI) {
  const unsigned N = MRI->getNumVirtRegs();

  // Cluster capacity: normally a full group (256). With balanced packing, when
  // the footprint leaves spare room across the groups, cap at
  // ceil(FP/EffGroups) so clusters spread and every used group keeps slack for
  // the soft hints.
  unsigned MergeCap = MSBGroupSize;
  if (Mode == PackMode::Balanced) {
    unsigned FP = maxSimultaneousDwords(AllVGPRs);
    unsigned Balanced = (FP + EffMSBGroups - 1) / std::max(1u, EffMSBGroups);
    MergeCap = std::min<unsigned>(MSBGroupSize, std::max(1u, Balanced));
  }

  AffinityGraph Graph = buildAffinityGraph(Blocks);
  if (Graph.empty())
    return;

  auto ComputeFootprintFn = [this](ArrayRef<Register> Regs) {
    return static_cast<int>(maxSimultaneousDwords(Regs));
  };
  ClusterForest Forest(N, ComputeFootprintFn);
  for (unsigned I = 0; I < N; ++I) {
    Register R = Register::index2VirtReg(I);
    if (!MRI->reg_nodbg_empty(R) && isVGPRVirtReg(R))
      Forest.addNode(I, R);
  }
  Forest.clusterByWeight(Graph, MergeCap);

  SmallVector<unsigned, 0> Roots = collectHotRoots(Forest, Graph);
  SmallVector<unsigned, 8> MSBLoad(EffMSBGroups, 0);
  DenseMap<unsigned, unsigned> ClusterMSB;
  packClusters(Forest, Roots, EffMSBGroups, VGPRBudget, Mode, MSBLoad,
               ClusterMSB);
  unsigned NumClusters = Roots.size();

  // Skip a *severely* over-subscribed plan. Slightly over cap still helps: RA
  // honors most hints and spills at most a few more than it would have. Past
  // cap*OverflowPct/100 the hints are just dropped. Per group so a fractional
  // last group is not over-packed.
  bool Infeasible = llvm::any_of(llvm::seq(0u, EffMSBGroups), [&](unsigned G) {
    return static_cast<uint64_t>(MSBLoad[G]) * 100 >
           static_cast<uint64_t>(groupCap(G, VGPRBudget)) * OverflowPct;
  });

  // Self-benefit gate: on an already MSB-coherent schedule our partition can
  // *raise* the switch count, so commit only if the plan's predicted switches
  // beat the natural no-hint layout on this same schedule. Both sims share the
  // naive groups for unhinted vregs, isolating the effect of our hints.
  bool LoopOnly = Scope == GateScope::LoopOnly;
  bool NoBenefit = false;
  uint64_t PlanSw = 0, BaseSw = 0;
  if (!Infeasible && BenefitPct) {
    DenseMap<unsigned, int> NaiveMSB = computeNaiveMSB(AllVGPRs, EffMSBGroups);
    DenseMap<unsigned, int> PlanOverride;
    for (unsigned Root : Roots)
      for (Register R : Forest.nodes(Root))
        PlanOverride[R.virtRegIndex()] = ClusterMSB[Root];
    auto NaiveOf = [&](Register R) {
      return NaiveMSB.lookup(R.virtRegIndex());
    };
    auto PlanOf = [&](Register R) {
      auto It = PlanOverride.find(R.virtRegIndex());
      return It != PlanOverride.end() ? It->second : NaiveOf(R);
    };
    BaseSw = simSwitchWeight(Blocks, NaiveOf, LoopOnly);
    PlanSw = simSwitchWeight(Blocks, PlanOf, LoopOnly);
    LLVM_DEBUG(dbgs() << "  gate(" << (LoopOnly ? "loop-only" : "whole-fn")
                      << ") planSw=" << PlanSw << " baseSw=" << BaseSw << "\n");
    // Commit only if the plan predicts a large enough win over the naive layout
    // on this exact schedule. Compute the products in 128 bits so the
    // comparison can't overflow on a huge function where the freq-weighted sums
    // are large.
    APInt PlanScaled = APInt(128, PlanSw) * APInt(128, 100);
    APInt BaseScaled = APInt(128, BaseSw) * APInt(128, BenefitPct.getValue());
    NoBenefit = PlanScaled.uge(BaseScaled);
    // Require a minimum absolute baseline cost. A small baseline means the loop
    // is already near-coherent; the predictor over-estimates its few switches
    // and the plan's "win" does not survive real allocation, so committing
    // regresses (e.g. an already-coherent occ-1 GEMM loop). Leave those to the
    // allocator.
    if (MinBaseSwitch && BaseSw < MinBaseSwitch)
      NoBenefit = true;
  }

  unsigned NumAssigned = 0;
  if (!Infeasible && !NoBenefit) {
    for (unsigned Root : Roots)
      for (Register R : Forest.nodes(Root)) {
        // A hotter (deeper) region already fixed this vreg's MSB group; don't
        // re-hint it to a different MSB group.
        if (!Assigned.insert(R.virtRegIndex()).second)
          continue;
        recordMSB(MFI, R, ClusterMSB[Root]);
        ++NumAssigned;
      }
  }

  LLVM_DEBUG({
    SmallVector<int, 0> FPs;
    for (unsigned Root : Roots)
      FPs.push_back(Forest.footprintOf(Root));
    llvm::sort(FPs, std::greater<int>());
    int Over = 0;
    for (int F : FPs)
      if (F > static_cast<int>(MSBGroupSize))
        ++Over;
    dbgs() << "  edges=" << Graph.size() << " clusters=" << NumClusters
           << " vregsAssigned=" << NumAssigned << " clustersOver256=" << Over
           << " planSw=" << PlanSw << " baseSw=" << BaseSw
           << (Infeasible ? " INFEASIBLE(skipped)"
                          : (NoBenefit ? " NO-BENEFIT(skipped)" : ""))
           << "\n  cluster FPs:";
    for (int F : FPs)
      dbgs() << ' ' << F;
    dbgs() << "\n  MSB group loads:";
    for (unsigned Group = 0; Group < EffMSBGroups; ++Group)
      dbgs() << " [" << Group << "]=" << MSBLoad[Group];
    dbgs() << "\n";
  });
}

char AMDGPUVGPRMSBAffinityLegacy::ID = 0;

char &llvm::AMDGPUVGPRMSBAffinityLegacyID = AMDGPUVGPRMSBAffinityLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUVGPRMSBAffinityLegacy, DEBUG_TYPE,
                      "AMDGPU VGPR MSB Affinity", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUVGPRMSBAffinityLegacy, DEBUG_TYPE,
                    "AMDGPU VGPR MSB Affinity", false, false)

FunctionPass *llvm::createAMDGPUVGPRMSBAffinityLegacyPass() {
  return new AMDGPUVGPRMSBAffinityLegacy();
}

PreservedAnalyses
AMDGPUVGPRMSBAffinityPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  auto *LIS = &MFAM.getResult<LiveIntervalsAnalysis>(MF);
  auto *MLI = &MFAM.getResult<MachineLoopAnalysis>(MF);
  AMDGPUVGPRMSBAffinity().run(MF, LIS, MLI);
  // Only allocation hints are recorded; no IR change.
  return PreservedAnalyses::all();
}
