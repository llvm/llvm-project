//===---------------------- AMDGPUNextUseAnalysis.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUNextUseAnalysis.h"
#include "AMDGPU.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"

#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"

#include <cmath>
#include <limits>
#include <queue>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-next-use-analysis"

namespace {

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Options
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cl::opt<bool> DumpNextUseDistance("amdgpu-next-use-analysis-dump-distance",
                                  cl::init(false), cl::Hidden);

cl::opt<std::string>
    DumpNextUseDistanceAsJson("amdgpu-next-use-analysis-dump-distance-as-json",
                              cl::Hidden);
cl::opt<bool>
    DumpNextUseDistanceVerbose("amdgpu-next-use-analysis-dump-distance-verbose",
                               cl::init(false), cl::Hidden);

cl::opt<AMDGPUNextUseAnalysis::CompatibilityMode> CompatModeOpt(
    "amdgpu-next-use-analysis-compatibility-mode", cl::Hidden,
    cl::init(AMDGPUNextUseAnalysis::CompatibilityMode::Graphics),
    cl::values(clEnumValN(AMDGPUNextUseAnalysis::CompatibilityMode::Graphics,
                          "graphics", "TBD"),
               clEnumValN(AMDGPUNextUseAnalysis::CompatibilityMode::Compute,
                          "compute", "TBD")));

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// String helpers
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename T> inline std::string printToString(T &X) {
  std::string S;
  raw_string_ostream OS(S);
  X.print(OS);
  return StringRef(OS.str()).trim().str();
}

template <typename T> inline std::string printToString(T *X) {
  return X ? printToString(*X) : "null";
}

inline std::string printToString(const MachineInstr &MI,
                                 ModuleSlotTracker &MST) {
  std::string S;
  raw_string_ostream OS(S);
  MI.print(OS, MST,
           /* IsStandalone    */ false,
           /* SkipOpers       */ false,
           /* SkipDebugLoc    */ false,
           /* AddNewLine      */ false,
           /* TargetInstrInfo */ nullptr);
  return StringRef(OS.str()).trim().str();
}

std::string printRegToString(Register Reg, unsigned SubRegIdx,
                             const MachineRegisterInfo *MRI,
                             const SIRegisterInfo *TRI) {
  std::string S;
  raw_string_ostream OS(S);
  OS << printReg(Reg, TRI, SubRegIdx, MRI);
  return OS.str();
}

std::string printRegToString(Register Reg, LaneBitmask LaneMask,
                             const MachineRegisterInfo *MRI,
                             const SIRegisterInfo *TRI) {
  unsigned SubRegIdx = 0;
  if (!Reg.isVirtual() || LaneMask != MRI->getMaxLaneMaskForVReg(Reg))
    SubRegIdx = TRI->getSubRegIndexForLaneMask(LaneMask);
  return printRegToString(Reg, SubRegIdx, MRI, TRI);
}

std::string nameForMBB(const MachineBasicBlock &BB, ModuleSlotTracker &MST) {
  std::string S;
  raw_string_ostream OS(S);
  BB.printName(OS, MachineBasicBlock::PrintNameIr, &MST);
  return OS.str();
}

struct InstructionInfo {
  std::string MIStr; // Backing storage for StringRefs
  StringRef DefName;
  StringRef DefType;
  StringRef Instr;
};

InstructionInfo parseInstructionString(const MachineInstr &MI,
                                       ModuleSlotTracker &MST) {
  InstructionInfo Info;
  Info.MIStr = printToString(MI, MST);
  StringRef MIRef(Info.MIStr);
  StringRef Def;
  std::tie(Def, Info.Instr) = MIRef.split('=');
  if (Info.Instr.empty()) {
    Def = "%void:void";
    Info.Instr = MIRef;
  }
  Info.Instr = Info.Instr.trim();
  std::tie(Info.DefName, Info.DefType) = Def.trim().split(":");
  return Info;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// MBBDistPair - Represents a distance to a machine basic block.
/// Used for returning both the distance and the target block together.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
struct MBBDistPair {
  double Distance;
  const MachineBasicBlock *MBB;
  constexpr MBBDistPair()
      : Distance(std::numeric_limits<double>::max()), MBB(nullptr) {}
  MBBDistPair(double D, const MachineBasicBlock *B) : Distance(D), MBB(B) {}

  MBBDistPair operator+(double D) { return {Distance + D, MBB}; }
  MBBDistPair &operator+=(double D) {
    Distance += D;
    return *this;
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// LiveRegUse - Represents a live register use with its distance.
/// Used for tracking and sorting register uses by distance.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
struct LiveRegUse {
  const MachineOperand *Use = nullptr;
  double Dist = 0.0;
  LiveRegUse() = default;
  LiveRegUse(const MachineOperand *Use, double Dist) : Use(Use), Dist(Dist) {}

  std::string toString() const {
    if (!valid())
      return "<invalid>";
    return std::to_string(Dist) + "@" + printToString(Use) + "*" +
           printToString(Use->getParent());
  }
  bool valid() const { return Use; }

  Register getReg() const { return Use->getReg(); }
  LaneBitmask getLaneMask(const SIRegisterInfo *TRI) const {
    return TRI->getSubRegIndexLaneMask(Use->getSubReg());
  }

  bool operator<(const LiveRegUse &Other) const {
    if (!Use)
      return true; // Other is better

    if (Dist < Other.Dist)
      return true; // Other is better

    if (Dist != Other.Dist)
      return false; // this is better

    if (Use == Other.Use)
      return false; // this is better

    // Ugh. In computeMode PHIs and the first non-PHI instruction have id 0. In
    // this case, consider PHIs as less than the first non-PHI instruction.
    const MachineInstr *MI = Use->getParent();
    const MachineInstr *OtherMI = Other.Use->getParent();
    const MachineBasicBlock *MBB = MI->getParent();
    if (MBB == OtherMI->getParent()) {
      bool IsPhiOp = MI->isPHI();
      bool OtherIsPhiOp = OtherMI->isPHI();
      if (IsPhiOp && !OtherIsPhiOp && OtherMI == &(*MBB->getFirstNonPHI()))
        return true;
    }

    // Ensure deterministic results (that match v1)
    return Other.getReg() < getReg();
  }
};

} // namespace

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// AMDGPUNextUseAnalysisImpl
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class llvm::AMDGPUNextUseAnalysisImpl {
  using CompatibilityMode = AMDGPUNextUseAnalysis::CompatibilityMode;
  const MachineFunction *MF = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  const MachineLoopInfo *MLI = nullptr;
  const MachineDominatorTree *DT = nullptr;
  const MachineRegisterInfo *MRI = nullptr;
  ModuleSlotTracker *MST = nullptr;

  DenseMap<const MachineInstr *, double> InstrToId;
  CompatibilityMode CompatMode;

  void initializeTables() {
    for (const MachineBasicBlock &BB : *MF)
      calcInstrIds(&BB, InstrToId);
    initializePathsFromMF();
  }

  void clearTables() {
    InstrToId.clear();
    RegUseMap.clear();
    Paths.clear();
  }

  bool computeMode() const { return CompatMode == CompatibilityMode::Compute; }

  bool graphicsMode() const {
    return CompatMode == CompatibilityMode::Graphics;
  }

  //----------------------------------------------------------------------------
  // Instruction Ids
  //----------------------------------------------------------------------------
private:
  void calcInstrIds(const MachineBasicBlock *BB,
                    DenseMap<const MachineInstr *, double> &InstrToId) const {
    double Id = 0.0;
    for (auto &MI : BB->instrs()) {
      InstrToId[&MI] = Id;
      if (!computeMode() || !MI.isPHI())
        ++Id;
    }
  }

  /// Returns MI's instruction Id. It renumbers (part of) the BB if MI is not
  /// found in the map.
  double getInstrId(const MachineInstr *MI) const {
    auto It = InstrToId.find(MI);
    if (It != InstrToId.end())
      return It->second;

    // Renumber the MBB.
    // TODO: Renumber from MI onwards.
    auto MutInstrToId =
        const_cast<DenseMap<const MachineInstr *, double> &>(InstrToId);
    calcInstrIds(MI->getParent(), MutInstrToId);
    return InstrToId.find(MI)->second;
  }
  double getInstrId(MachineBasicBlock::const_instr_iterator I) const {
    return getInstrId(&*I);
  }

  // Length of the segment from MI (inclusive) to the first instruction of the
  // basic block.
  double getHeadLen(const MachineInstr *MI) const {
    const MachineBasicBlock *MBB = MI->getParent();
    return getInstrId(MI) + getInstrId(&MBB->instr_front()) + 1;
  }

  // Length of the segment from MI (exclusive) to the last instruction of the
  // basic block.
  double getTailLen(const MachineInstr *MI) const {
    const MachineBasicBlock *MBB = MI->getParent();
    return getInstrId(&MBB->instr_back()) - getInstrId(MI);
  }

  // Length of the segment from 'From' to 'To' (exclusive). Both instructions
  // must in the same basic block.
  double getDistance(const MachineInstr *From, const MachineInstr *To) const {
    assert(From->getParent() == To->getParent());
    return getInstrId(To) - getInstrId(From);
  }

  //----------------------------------------------------------------------------
  // RegUses
  //----------------------------------------------------------------------------
private:
  DenseMap<unsigned, SmallVector<const MachineOperand *>> RegUseMap;

  const SmallVector<const MachineOperand *> &getRegisterUses(unsigned Reg) {
    auto I = RegUseMap.find(Reg);
    if (I != RegUseMap.end())
      return I->second;

    SmallVector<const MachineOperand *> &Uses = RegUseMap[Reg];
    for (const MachineOperand &UseMO : MRI->use_nodbg_operands(Reg)) {
      if (!UseMO.isUndef())
        Uses.push_back(&UseMO);
    }
    return Uses;
  }

  //----------------------------------------------------------------------------
  // Paths
  //----------------------------------------------------------------------------
private:
  class Path {
  private:
    const MachineBasicBlock *Src;
    const MachineBasicBlock *Dst;

  public:
    Path(const MachineBasicBlock *Src, const MachineBasicBlock *Dst)
        : Src(Src), Dst(Dst) {}
    Path() : Src(nullptr), Dst(nullptr) {}
    Path(const Path &Other) = default;
    Path &operator=(const Path &Other) = default;

    bool operator==(const Path &Other) const {
      return Src == Other.Src && Dst == Other.Dst;
    }
    bool operator!=(const Path &Other) const {
      return !this->operator==(Other);
    }

    const MachineBasicBlock *src() const { return Src; }
    const MachineBasicBlock *dst() const { return Dst; }
  };

  struct PathDenseMapInfo {
    using MBBPtrInfo = DenseMapInfo<MachineBasicBlock *>;

    static inline Path getEmptyKey() {
      return Path(MBBPtrInfo::getEmptyKey(), MBBPtrInfo::getEmptyKey());
    }
    static inline Path getTombstoneKey() {
      return Path(MBBPtrInfo::getTombstoneKey(), MBBPtrInfo::getTombstoneKey());
    }

    static unsigned getHashValue(const Path &Val) {
      return detail::combineHashValue(MBBPtrInfo::getHashValue(Val.src()),
                                      MBBPtrInfo::getHashValue(Val.dst()));
    }
    static bool isEqual(const Path &LHS, const Path &RHS) { return LHS == RHS; }
  };

  enum EdgeKind { Back = -1, None = 0, Tree = 1, Forward = 2, Cross = 3 };
  struct PathInfo {
    EdgeKind EK;
    bool Reachable;
    bool ForwardReachable;
    double LoopWeight;
    std::optional<double> ShortestDistance;
    std::optional<double> ShortestUnweightedDistance;
    double Size;

    bool isBackedge() const { return EK == EdgeKind::Back; }
  };
  DenseMap<Path, PathInfo, PathDenseMapInfo> Paths;

  void initializePathInfo(PathInfo &Slot, Path P, EdgeKind EK) const {
    Slot.EK = EK;
    Slot.Reachable = calcIsReachable(P.src(), P.dst());
    if (EK == EdgeKind::None)
      Slot.ForwardReachable =
          Slot.Reachable &&
          calcIsReachable(P.src(), P.dst(), /*ForwardOnly*/ true);
    else
      Slot.ForwardReachable = 0 < EK;

    Slot.LoopWeight = calcLoopWeight(P.src(), P.dst());
    Slot.Size = P.src() == P.dst() ? calcSize(P.src())
                                   : std::numeric_limits<double>::max();
  }

  void initializePathsFromMF() {
    Paths.clear();

    int TS = 0;
    struct Timestamps {
      int Discovered;
      int Visited;
      int Finished;
    };
    DenseMap<const MachineBasicBlock *, Timestamps> Time;

    SmallVector<const MachineBasicBlock *> Work;
    Work.emplace_back(&MF->front());
    Time[&MF->front()].Discovered = ++TS;

    while (!Work.empty()) {

      const MachineBasicBlock *Src = Work.back();
      Timestamps &SrcTime = Time[Src];

      if (SrcTime.Visited) {
        Work.pop_back();
        SrcTime.Finished = ++TS;
        continue;
      }

      SrcTime.Visited = ++TS;
      for (const MachineBasicBlock *Dst : Src->successors()) {
        EdgeKind EK = EdgeKind::None;
        Timestamps &DstTime = Time[Dst];
        if (!DstTime.Discovered) {
          EK = EdgeKind::Tree;
          Work.emplace_back(Dst);
          DstTime.Discovered = ++TS;
        } else if (DstTime.Visited && !DstTime.Finished) {
          EK = EdgeKind::Back;
        } else if (SrcTime.Discovered < DstTime.Discovered) {
          EK = EdgeKind::Forward;
        } else {
          EK = EdgeKind::Cross;
        }

        Path P(Src, Dst);
        assert(!Paths.contains(P));
        PathInfo &Slot = Paths[P];
        initializePathInfo(Slot, P, EK);
      }
    }
  }

  PathInfo &mutPathInfoFor(const MachineBasicBlock *From,
                           const MachineBasicBlock *To) const {
    auto &MutPaths = const_cast<AMDGPUNextUseAnalysisImpl *>(this)->Paths;
    Path P(From, To);
    auto I = MutPaths.find(P);
    if (I != MutPaths.end())
      return I->second;

    PathInfo &Slot = MutPaths[P];
    initializePathInfo(Slot, P, EdgeKind::None);
    return Slot;
  }

  const PathInfo &pathInfoFor(const MachineBasicBlock *From,
                              const MachineBasicBlock *To) const {
    return mutPathInfoFor(From, To);
  }

  //----------------------------------------------------------------------------
  // Calculate features
  //----------------------------------------------------------------------------
private:
  double calcSize(const MachineBasicBlock *BB) const {
    double Size = BB->size();
    if (computeMode())
      Size -= std::distance(BB->begin(), BB->getFirstNonPHI());
    return Size;
  }

  double calcWeightedSize(const MachineBasicBlock *From,
                          const MachineBasicBlock *To) const {
    double LoopWeight = getLoopWeight(From, To);
    if (LoopWeight == 0.0)
      LoopWeight = 1.0;
    return getSize(From) * LoopWeight;
  }

  static double getEffectiveLoopDepth(MachineLoop *Loop,
                                      const MachineBasicBlock *To,
                                      const MachineLoopInfo *MLI) {
    double LoopDepth = 0.0;
    MachineLoop *const End = Loop->getOutermostLoop()->getParentLoop();
    for (MachineLoop *TmpLoop = Loop; TmpLoop != End;
         TmpLoop = TmpLoop->getParentLoop()) {
      if (TmpLoop->contains(To))
        continue;
      LoopDepth++;
    }
    return LoopDepth;
  }

  static double encodeLoopDepth(double Depth) {
    constexpr double LoopWeight = 1000.0;
    return std::pow(LoopWeight, Depth);
  }

  double calcLoopWeight(const MachineBasicBlock *From,
                        const MachineBasicBlock *To) const {
    MachineLoop *LoopFrom = MLI->getLoopFor(From);
    MachineLoop *LoopTo = MLI->getLoopFor(To);

    if (!LoopFrom)
      return 0.0;

    if (!LoopTo)
      return encodeLoopDepth(getEffectiveLoopDepth(LoopFrom, To, MLI));

    if (LoopFrom->contains(LoopTo)) // covers LoopFrom == LoopTo
      return 1.0;

    if (LoopTo->contains(LoopFrom))
      return encodeLoopDepth(MLI->getLoopDepth(From) - MLI->getLoopDepth(To));

    return encodeLoopDepth(getEffectiveLoopDepth(LoopFrom, To, MLI));
  }

  bool calcIsReachable(const MachineBasicBlock *From,
                       const MachineBasicBlock *To,
                       bool ForwardOnly = false) const {
    SmallVector<const MachineBasicBlock *> Work;
    DenseSet<const MachineBasicBlock *> Visited;

    Work.push_back(From);
    Visited.insert(From);

    while (!Work.empty()) {
      const MachineBasicBlock *Current = Work.pop_back_val();

      for (const MachineBasicBlock *Succ : Current->successors()) {
        if (ForwardOnly && isBackedge(Current, Succ))
          continue;

        if (Succ == To)
          return true;

        Path P(Succ, To);
        auto I = Paths.find(P);
        if (I != Paths.end()) {
          if (ForwardOnly && I->second.ForwardReachable)
            return true;
          if (!ForwardOnly && I->second.Reachable)
            return true;
          continue;
        }

        if (Visited.insert(Succ).second)
          Work.push_back(Succ);
      }
    }
    return false;
  }

  double calcShortestPath(const MachineBasicBlock *FromMBB,
                          const MachineBasicBlock *ToMBB,
                          bool Unweighted) const {

    assert(FromMBB != ToMBB && "The basic blocks should be different.");
    DenseSet<const MachineBasicBlock *> Visited;
    struct Data {
      const MachineBasicBlock *BestPred = nullptr;
      double ShortestDistance = std::numeric_limits<double>::max();
    };
    DenseMap<const MachineBasicBlock *, Data> MBBData;

    auto Cmp = [&MBBData](const MachineBasicBlock *MBB1,
                          const MachineBasicBlock *MBB2) {
      return MBBData[MBB1].ShortestDistance > MBBData[MBB2].ShortestDistance;
    };
    std::priority_queue<const MachineBasicBlock *,
                        std::vector<const MachineBasicBlock *>, decltype(Cmp)>
        Worklist(Cmp);

    Worklist.push(FromMBB);
    MBBData[FromMBB] = {nullptr, 0.0};

    while (!Worklist.empty()) {
      const MachineBasicBlock *CurMBB = Worklist.top();
      Worklist.pop();

      if (!Visited.insert(CurMBB).second)
        continue;

      if (CurMBB == ToMBB) {
        auto *Pred = MBBData[CurMBB].BestPred;
        return MBBData[Pred].ShortestDistance -
               MBBData[FromMBB].ShortestDistance;
      }

      auto Pair = MBBData.try_emplace(
          CurMBB, Data{nullptr, std::numeric_limits<double>::max()});
      double CurrMBBDist = Pair.first->second.ShortestDistance;

      for (MachineBasicBlock *Succ : CurMBB->successors()) {
        const PathInfo &PI = pathInfoFor(CurMBB, Succ);
        if (PI.isBackedge() && graphicsMode())
          continue;

        double AB = Unweighted ? getSize(Succ) : calcWeightedSize(Succ, ToMBB);
        double NewSuccDist = CurrMBBDist + AB;

        auto &[SuccPred, SuccDist] = MBBData[Succ];
        if (NewSuccDist < SuccDist) {
          // We found a better path to Succ, update best predecessor and
          // distance
          SuccPred = CurMBB;
          SuccDist = NewSuccDist;
        }

        Worklist.push(Succ);
      }
    }
    return std::numeric_limits<double>::max();
  }

  /// If the path from \p MI to \p UseMI does not cross any loops, then this
  /// \returns the shortest instruction distance between them.
  double calcShortestDistance(const MachineInstr *CurMI,
                              const MachineInstr *UseMI) const {
    const MachineBasicBlock *CurMBB = CurMI->getParent();
    const MachineBasicBlock *UseMBB = UseMI->getParent();

    static auto check = [](double D) {
      assert(D >= 0);
      return D;
    };

    if (CurMBB == UseMBB)
      return check(getDistance(CurMI, UseMI));

    double CurMITailLen = getTailLen(CurMI);
    double UseHeadLen = getHeadLen(UseMI);
    double Dst = getShortestPath(CurMBB, UseMBB);
    assert(Dst != std::numeric_limits<double>::max() &&
           "calcShortestDistance called for instructions in non-reachable"
           " basic blocks!");
    return check(CurMITailLen + Dst + UseHeadLen);
  }

  double calcShortestUnweightedDistance(const MachineInstr *CurMI,
                                        const MachineInstr *UseMI) const {
    const MachineBasicBlock *CurMBB = CurMI->getParent();
    const MachineBasicBlock *UseMBB = UseMI->getParent();

    if (CurMBB == UseMBB)
      return getDistance(CurMI, UseMI);

    double CurMITailLen = getTailLen(CurMI);
    double UseHeadLen = getHeadLen(UseMI);
    double Dst = getShortestUnweightedPath(CurMBB, UseMBB);
    assert(Dst != std::numeric_limits<double>::max() &&
           "calcShortestUnweightedDistance called for instructions in"
           " non-reachable basic blocks!");
    return CurMITailLen + Dst + UseHeadLen;
  }

  //----------------------------------------------------------------------------
  // Feature getters. Use cached results if available. If not calculate.
  //----------------------------------------------------------------------------
private:
  double getSize(const MachineBasicBlock *BB) const {
    return pathInfoFor(BB, BB).Size;
  }

  bool isReachable(const MachineBasicBlock *From,
                   const MachineBasicBlock *To) const {
    return pathInfoFor(From, To).Reachable;
  }

  bool isReachableOrSame(const MachineBasicBlock *From,
                         const MachineBasicBlock *To) const {
    return From == To || pathInfoFor(From, To).Reachable;
  }

  bool isForwardReachable(const MachineBasicBlock *From,
                          const MachineBasicBlock *To) const {
    return pathInfoFor(From, To).ForwardReachable;
  }

  bool isBackedge(const MachineBasicBlock *From,
                  const MachineBasicBlock *To) const {
    return pathInfoFor(From, To).isBackedge();
  }

  bool isDistanceFinite(const MachineBasicBlock *From,
                        const MachineBasicBlock *To) const {
    if (From == To)
      return false;
    return getShortestPath(From, To) != std::numeric_limits<double>::max();
  }

  // Can be used as a substitute for DT->dominates(A, B) if A and B are in the
  // same basic block.
  bool instrsAreInOrder(const MachineInstr *A, const MachineInstr *B) const {
    assert(A->getParent() == B->getParent() &&
           "instructions must be in the same basic block!");
    if (A == B || getInstrId(A) < getInstrId(B))
      return true;
    if (!A->isPHI())
      return false;
    if (!B->isPHI())
      return true;
    for (auto &PHI : A->getParent()->phis()) {
      if (&PHI == A)
        return true;
      if (&PHI == B)
        return false;
    }
    return false;
  }

  double getLoopWeight(const MachineBasicBlock *From,
                       const MachineBasicBlock *To) const {
    return pathInfoFor(From, To).LoopWeight;
  }

  /// Calculates the shortest distance and caches it.
  double getShortestPath(const MachineBasicBlock *From,
                         const MachineBasicBlock *To) const {
    std::optional<double> &D = mutPathInfoFor(From, To).ShortestDistance;
    if (!D.has_value())
      D = calcShortestPath(From, To, /* Unweighted */ false);
    return D.value();
  }

  double getShortestUnweightedPath(const MachineBasicBlock *From,
                                   const MachineBasicBlock *To) const {
    std::optional<double> &D =
        mutPathInfoFor(From, To).ShortestUnweightedDistance;
    if (!D.has_value())
      D = calcShortestPath(From, To, /* Unweighted */ true);
    return D.value();
  }

  //----------------------------------------------------------------------------
  // Loop helpers
  //----------------------------------------------------------------------------
private:
  static bool isUseOutsideOfTheCurrentLoopNest(const MachineLoop *UseLoop,
                                               const MachineLoop *CurLoop) {
    if (CurLoop && !UseLoop)
      return true;

    if (!CurLoop || !UseLoop)
      return false;

    return !UseLoop->contains(CurLoop) && !CurLoop->contains(UseLoop);
  }

  static bool isUseOutsideOfTheCurrentLoop(const MachineLoop *UseLoop,
                                           const MachineLoop *CurLoop) {
    if (CurLoop && !UseLoop)
      return true;

    if (!CurLoop || !UseLoop)
      return false;

    if (!UseLoop->contains(CurLoop) && !CurLoop->contains(UseLoop))
      return true;

    return UseLoop->contains(CurLoop) && UseLoop != CurLoop;
  }

  static bool isUseInParentLoop(const MachineLoop *UseLoop,
                                const MachineLoop *CurLoop) {
    if (!CurLoop || !UseLoop)
      return false;

    return UseLoop->contains(CurLoop) && UseLoop != CurLoop;
  }

  static bool isStandAloneLoop(const MachineLoop *Loop) {
    return Loop->getSubLoops().empty() && Loop->isOutermost();
  }

  static const MachineBasicBlock *
  getOutermostPreheader(const MachineLoop *Loop) {
    return Loop->getOutermostLoop()->getLoopPreheader();
  }

  static const MachineBasicBlock *mbbForPhiOp(const MachineOperand *MO) {
    return MO->getParent()->getOperand(MO->getOperandNo() + 1).getMBB();
  }

  //----------------------------------------------------------------------------
  // CFG Helpers
  //----------------------------------------------------------------------------
private:
  // Return the shortest distance to a latch
  MBBDistPair calcShortestDistanceToLatch(const MachineBasicBlock *CurMBB,
                                          const MachineLoop *CurLoop) const {
    SmallVector<MachineBasicBlock *, 2> Latches;
    CurLoop->getLoopLatches(Latches);
    MBBDistPair LD;

    for (MachineBasicBlock *LMBB : Latches) {
      if (LMBB == CurMBB)
        return {0.0, CurMBB};

      double Dst = getShortestPath(CurMBB, LMBB);
      if (Dst < LD.Distance) {
        LD.Distance = Dst;
        LD.MBB = LMBB;
      }
    }
    return LD;
  }

  // Return the shortest distance through a loop (header to latch) that goes
  // through CurMBB.
  MBBDistPair calcShortestDistanceThroughLoop(const MachineBasicBlock *CurMBB,
                                              MachineLoop *CurLoop) const {

    // This is a hot spot. Check it before doing anything else.
    if (CurLoop->getNumBlocks() == 1)
      return {getSize(CurMBB), CurMBB};

    MachineBasicBlock *LoopHeader = CurLoop->getHeader();
    MBBDistPair LD{0.0, nullptr};

    LD += getSize(LoopHeader);

    if (CurMBB != LoopHeader)
      LD += getShortestPath(LoopHeader, CurMBB);

    if (CurLoop->isLoopLatch(CurMBB))
      LD.MBB = CurMBB;
    else
      LD = calcShortestDistanceToLatch(CurMBB, CurLoop) + LD.Distance;

    if (CurMBB != LoopHeader && CurMBB != LD.MBB)
      LD += getSize(CurMBB);

    LD += getSize(LD.MBB);

    return LD;
  }

  // Similar to calcShortestDistanceThroughLoop with LoopWeight applied to the
  // returned distance.
  MBBDistPair calcWeightedDistanceThroughLoop(const MachineBasicBlock *CurMBB,
                                              MachineLoop *CurLoop) const {
    MBBDistPair LD = calcShortestDistanceThroughLoop(CurMBB, CurLoop);
    LD.Distance *= encodeLoopDepth(1);
    return LD;
  }

  // Return the weighted, shortest distance through a loop (header to latch).
  // If ParentLoop is provided, use it to adjust the loop depth.
  MBBDistPair calcWeightedDistanceThroughLoop(
      MachineLoop *CurLoop, const MachineLoop *ParentLoop = nullptr) const {

    const MachineBasicBlock *Hdr = CurLoop->getHeader();
    if (CurLoop->getNumBlocks() != 1)
      return calcWeightedDistanceThroughLoop(Hdr, CurLoop);

    unsigned LoopDepth = MLI->getLoopDepth(Hdr);
    if (ParentLoop)
      LoopDepth -= ParentLoop->getLoopDepth();

    return {getSize(Hdr) * encodeLoopDepth(LoopDepth), CurLoop->getLoopLatch()};
  }

  // Calculate total distance from exit point to use instruction
  double appendDistanceToUse(const MBBDistPair &Exit, const MachineInstr *UseMI,
                             const MachineBasicBlock *UseMBB) const {
    return Exit.Distance + getShortestPath(Exit.MBB, UseMBB) +
           getHeadLen(UseMI);
  }

  // Return the weighted, shortest distance through the sub-loop of CurLoop
  // containing UseLoop.
  MBBDistPair calcDistanceThroughSubLoopUse(MachineLoop *CurLoop,
                                            MachineLoop *UseLoop) const {

    assert(UseLoop->contains(CurLoop) && "CurLoop should be nested in UseLoop");

    // All the sub-loops of the UseLoop will be executed before the use.
    // Hence, we should take this into consideration in distance calculation.
    MachineLoop *UseLoopSubLoop = CurLoop;
    while (UseLoopSubLoop->getParentLoop() != UseLoop)
      UseLoopSubLoop = UseLoopSubLoop->getParentLoop();
    return calcWeightedDistanceThroughLoop(UseLoopSubLoop, UseLoop);
  }

  // Similar to calcDistanceThroughSubLoopUse, adding the distance to 'UseMI'.
  double calcDistanceThroughSubLoopToUse(const MachineBasicBlock *CurMBB,
                                         MachineLoop *CurLoop,
                                         const MachineInstr *UseMI,
                                         const MachineBasicBlock *UseMBB,
                                         MachineLoop *UseLoop) const {
    return appendDistanceToUse(calcDistanceThroughSubLoopUse(CurLoop, UseLoop),
                               UseMI, UseMBB);
  }

  // Return the weighted distance through a loop to an outside use loop.
  // Differentiates between uses inside or outside of the current loop nest.
  MBBDistPair calcDistanceThroughLoopToOutsideLoopUse(
      const MachineBasicBlock *CurMBB, MachineLoop *CurLoop,
      const MachineBasicBlock *UseMBB, MachineLoop *UseLoop) const {

    assert(!CurLoop->contains(UseLoop));

    if (isStandAloneLoop(CurLoop))
      return calcWeightedDistanceThroughLoop(CurMBB, CurLoop);

    MachineLoop *OutermostLoop = CurLoop->getOutermostLoop();
    if (!OutermostLoop->contains(UseLoop)) {
      // We should take into consideration the whole loop nest in the
      // calculation of the distance because we will reach the use after
      // executing the whole loop nest.
      return calcWeightedDistanceThroughLoop(OutermostLoop);
    }

    // At this point we know that CurLoop and UseLoop are independent and they
    // are in the same loop nest.

    if (MLI->getLoopDepth(CurMBB) <= MLI->getLoopDepth(UseMBB)) {
      if (computeMode() && (CurLoop->getNumBlocks() == 1))
        return calcWeightedDistanceThroughLoop(CurLoop->getHeader(), CurLoop);
      return calcWeightedDistanceThroughLoop(CurLoop);
    }

    assert(CurLoop != OutermostLoop && "The loop cannot be the outermost.");
    const unsigned UseLoopDepth = MLI->getLoopDepth(UseMBB);
    for (;;) {
      if (CurLoop->getLoopDepth() == UseLoopDepth)
        break;
      CurLoop = CurLoop->getParentLoop();
      if (CurLoop == OutermostLoop)
        break;
    }
    return calcWeightedDistanceThroughLoop(CurLoop);
  }

  // Similar to calcDistanceThroughLoopToOutsideLoopUse but adds the distance to
  // an instruction in the loop.
  double calcDistanceThroughLoopToOutsideLoopUseMI(
      const MachineBasicBlock *CurMBB, MachineLoop *CurLoop,
      const MachineInstr *UseMI, const MachineBasicBlock *UseMBB,
      MachineLoop *UseLoop) const {
    return appendDistanceToUse(calcDistanceThroughLoopToOutsideLoopUse(
                                   CurMBB, CurLoop, UseMBB, UseLoop),
                               UseMI, UseMBB);
  }

  // Return true if 'MO' is covered by 'LaneMask'
  bool machineOperandCoveredBy(const MachineOperand &MO,
                               LaneBitmask LaneMask) const {
    LaneBitmask Mask = TRI->getSubRegIndexLaneMask(MO.getSubReg());
    return (Mask & LaneMask) == Mask;
  }

  // Returns true iff uses of LiveReg/LiveLaneMask in PHI UseMI are coming from
  // a backedge when starting at CurMI.
  bool isIncomingValFromBackedge(Register LiveReg, LaneBitmask LiveLaneMask,
                                 const MachineInstr *CurMI,
                                 const MachineInstr *UseMI) const {
    if (!UseMI->isPHI())
      return false;

    MachineLoop *CurLoop = MLI->getLoopFor(CurMI->getParent());
    MachineLoop *UseLoop = MLI->getLoopFor(UseMI->getParent());

    // Not a backedge if ...
    // A: not in a loop at all
    // B: or CurMI is in a loop outside of UseLoop
    // C: or UseMI is not in the UseLoop header
    if (/*A:*/ !UseLoop ||
        /*B:*/ (CurLoop && !UseLoop->contains(CurLoop)) ||
        /*C:*/ UseMI->getParent() != UseLoop->getHeader())
      return false;

    SmallVector<MachineBasicBlock *, 2> Latches;
    UseLoop->getLoopLatches(Latches);

    auto Ops = UseMI->operands();
    for (auto It = std::next(Ops.begin()), ItE = Ops.end(); It != ItE;
         It = std::next(It, 2)) {
      auto &RegMO = *It;
      auto &MBBMO = *std::next(It);
      assert(RegMO.isReg() && "Expected register operand of PHI");
      assert(MBBMO.isMBB() && "Expected MBB operand of PHI");
      if (RegMO.getReg() == LiveReg &&
          machineOperandCoveredBy(RegMO, LiveLaneMask)) {
        MachineBasicBlock *IncomingBB = MBBMO.getMBB();
        if (llvm::find(Latches, IncomingBB) != Latches.end())
          return true;
      }
    }
    return false;
  }

  // Return the distance from 'CurMI' through a backedge PHI Use
  // ('UseMI'). Handles various loop configurations.
  double calcBackedgeDistance(const MachineInstr *CurMI,
                              const MachineBasicBlock *CurMBB,
                              MachineLoop *CurLoop, const MachineInstr *UseMI,
                              const MachineBasicBlock *UseMBB,
                              MachineLoop *UseLoop) const {

    assert(UseLoop && "There is no backedge.");
    double CurMITailLen = getTailLen(CurMI);
    double UseHeadLen = getHeadLen(UseMI);

    if (!CurLoop)
      return CurMITailLen + getShortestPath(CurMBB, UseMBB) + UseHeadLen;

    if (CurLoop == UseLoop) {
      MBBDistPair LD = calcShortestDistanceToLatch(CurMBB, CurLoop);
      if (LD.MBB == CurMBB)
        return CurMITailLen + UseHeadLen;
      return UseHeadLen + CurMITailLen + LD.Distance + getSize(LD.MBB);
    }

    if (!CurLoop->contains(UseLoop) && !UseLoop->contains(CurLoop)) {
      MBBDistPair LD = calcShortestDistanceThroughLoop(CurMBB, CurLoop);
      return LD.Distance + getShortestPath(LD.MBB, UseMBB) + UseHeadLen;
    }

    if (!CurLoop->contains(UseLoop)) {
      MBBDistPair InnerLoopLD = calcDistanceThroughSubLoopUse(CurLoop, UseLoop);
      MBBDistPair LD = calcShortestDistanceToLatch(InnerLoopLD.MBB, UseLoop);
      return InnerLoopLD.Distance + LD.Distance + getSize(LD.MBB) + UseHeadLen;
    }

    llvm_unreachable("The backedge distance has not been calculated!");
  }

  // Optimized version of calcBackedgeDistance when we already know that CurMI
  // and UseMI are in the same basic block
  double calcBackedgeDistance(const MachineInstr *CurMI,
                              const MachineBasicBlock *CurMBB,
                              MachineLoop *CurLoop,
                              const MachineInstr *UseMI) const {
    // use is in the next loop iteration
    double CurTailLen = getTailLen(CurMI);
    double UseHeadLen = getHeadLen(UseMI);
    MBBDistPair LD = calcShortestDistanceToLatch(CurMBB, CurLoop);
    const MachineBasicBlock *HdrMBB = CurLoop->getHeader();
    double HdrSize = getSize(HdrMBB);
    double Dst = CurMBB == HdrMBB ? 0.0 : getShortestPath(HdrMBB, CurMBB);
    return CurTailLen + LD.Distance + HdrSize + Dst + UseHeadLen;
  }

  // Return the distance from CurMI inside of a loop to UseMI outside of that
  // loop. 'LiveReg' and 'LiveLaneMask' are used to identify relevant backedges
  // if needed.
  double calcInsideToOutsideLoopDistance(
      Register LiveReg, LaneBitmask LiveLaneMask, const MachineInstr *CurMI,
      const MachineBasicBlock *CurMBB, MachineLoop *CurLoop,
      const MachineInstr *UseMI, const MachineBasicBlock *UseMBB,
      MachineLoop *UseLoop) const {

    if (isUseOutsideOfTheCurrentLoopNest(UseLoop, CurLoop))
      return calcDistanceThroughLoopToOutsideLoopUseMI(CurMBB, CurLoop, UseMI,
                                                       UseMBB, UseLoop);

    if (isUseInParentLoop(UseLoop, CurLoop)) {
      assert(MLI->getLoopDepth(UseMBB) < MLI->getLoopDepth(CurMBB) &&
             "The loop depth of the current instruction must be bigger than "
             "these.\n");
      if (isIncomingValFromBackedge(LiveReg, LiveLaneMask, CurMI, UseMI))
        return calcBackedgeDistance(CurMI, CurMBB, CurLoop, UseMI, UseMBB,
                                    UseLoop);

      return calcDistanceThroughSubLoopToUse(CurMBB, CurLoop, UseMI, UseMBB,
                                             UseLoop);
    }

    llvm_unreachable("Unexpected loop configuration");
  }

  //----------------------------------------------------------------------------
  // calcDistanceToUse*
  //----------------------------------------------------------------------------
private:
  // Return the distance from CurMI to a use (UseMO or UseMI) - graphics edition
  double calcDistanceToUseForGraphics(
      Register LiveReg, LaneBitmask LiveLaneMask, const MachineInstr &CurMI,
      const MachineBasicBlock *CurMBB, MachineLoop *CurLoop,
      const MachineInstr *UseMI, const MachineBasicBlock *UseMBB,
      MachineLoop *UseLoop, const MachineBasicBlock *PhiUseEdge) const {
    if (isUseOutsideOfTheCurrentLoop(UseLoop, CurLoop))
      return calcInsideToOutsideLoopDistance(LiveReg, LiveLaneMask, &CurMI,
                                             CurMBB, CurLoop, UseMI, UseMBB,
                                             UseLoop);

    if (isIncomingValFromBackedge(LiveReg, LiveLaneMask, &CurMI, UseMI))
      return calcBackedgeDistance(&CurMI, CurMBB, CurLoop, UseMI, UseMBB,
                                  UseLoop);

    return calcShortestDistance(&CurMI, UseMI);
  }

  // Return the distance from CurMI to a use (UseMO or UseMI) - compute
  // edition
  double calcDistanceToUseForCompute(
      Register LiveReg, LaneBitmask LiveLaneMask, const MachineInstr &CurMI,
      const MachineBasicBlock *CurMBB, MachineLoop *CurLoop,
      const MachineInstr *UseMI, const MachineBasicBlock *UseMBB,
      MachineLoop *UseLoop, const MachineBasicBlock *PhiUseEdge) const {
    if (PhiUseEdge)
      return calcDistanceToUse(LiveReg, LiveLaneMask, CurMI,
                               &PhiUseEdge->back(), nullptr);

    // No loops involved
    if (!CurLoop && !UseLoop) {
      if (CurMBB != UseMBB) {
        // -1 for PHIs so that they appear closer than non-PHIs.
        return calcShortestUnweightedDistance(&CurMI, UseMI) - UseMI->isPHI();
      }
      return calcShortestDistance(&CurMI, UseMI);
    }

    // From non-loop to inside loop use
    if (!CurLoop && UseLoop) {
      // Reset to UseLoop preheader position. This models: if spilled before
      // loop, reload at preheader
      const MachineBasicBlock *PreHdr = getOutermostPreheader(UseLoop);
      return calcShortestUnweightedDistance(&CurMI, &PreHdr->back());
    }

    // From loop to non-loop use
    if (CurLoop && !UseLoop)
      return calcDistanceThroughLoopToOutsideLoopUseMI(CurMBB, CurLoop, UseMI,
                                                       UseMBB, UseLoop);

    // Both in loops
    if (CurLoop == UseLoop) {
      if (CurMBB == UseMBB && !instrsAreInOrder(&CurMI, UseMI))
        return calcBackedgeDistance(&CurMI, CurMBB, CurLoop, UseMI);
      return calcShortestDistance(&CurMI, UseMI);
    }

    if (CurLoop->contains(UseLoop))
      return calcShortestDistance(&CurMI, UseMI);

    if (UseLoop->contains(CurLoop))
      return calcDistanceThroughSubLoopToUse(CurMBB, CurLoop, UseMI, UseMBB,
                                             UseLoop);

    // Loops are unrelated
    if (isStandAloneLoop(CurLoop)) {
      // Reset to UseLoop preheader position. This models: if spilled before
      // loop, reload at preheader
      MBBDistPair LD = calcWeightedDistanceThroughLoop(CurMBB, CurLoop);
      const MachineBasicBlock *PreHdr = getOutermostPreheader(UseLoop);
      return appendDistanceToUse(LD, &PreHdr->back(), PreHdr);
    }
    return calcDistanceThroughLoopToOutsideLoopUseMI(CurMBB, CurLoop, UseMI,
                                                     UseMBB, UseLoop);
  }

  // Return the distance from CurMI to a use (UseMO or UseMI).
  double calcDistanceToUse(Register LiveReg, LaneBitmask LiveLaneMask,
                           const MachineInstr &CurMI, const MachineInstr *UseMI,
                           const MachineBasicBlock *PhiUseEdge) const {

    const MachineBasicBlock *CurMBB = CurMI.getParent();
    const MachineBasicBlock *UseMBB = UseMI->getParent();
    MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
    MachineLoop *UseLoop = MLI->getLoopFor(UseMBB);

    if (graphicsMode()) {
      return calcDistanceToUseForGraphics(LiveReg, LiveLaneMask, CurMI, CurMBB,
                                          CurLoop, UseMI, UseMBB, UseLoop,
                                          PhiUseEdge);
    }

    if (computeMode()) {
      return calcDistanceToUseForCompute(LiveReg, LiveLaneMask, CurMI, CurMBB,
                                         CurLoop, UseMI, UseMBB, UseLoop,
                                         PhiUseEdge);
    }

    llvm_unreachable("not handled: CurLoop && !UseLoop");
  }

  double calcDistanceToUse(Register LiveReg, LaneBitmask LiveLaneMask,
                           const MachineInstr &CurMI,
                           const MachineOperand *UseMO) const {

    const MachineInstr *UseMI = UseMO->getParent();
    const MachineBasicBlock *PhiUseEdge =
        UseMI->isPHI() ? mbbForPhiOp(UseMO) : nullptr;
    return calcDistanceToUse(LiveReg, LiveLaneMask, CurMI, UseMI, PhiUseEdge);
  }

  //----------------------------------------------------------------------------
  // getUses helpers (compute mode)
  //----------------------------------------------------------------------------
private:
  bool isUseReachableForCompute(const MachineInstr &MI,
                                const MachineBasicBlock *MBB,
                                const MachineOperand *UseMO,
                                const MachineInstr *UseMI,
                                const MachineBasicBlock *UseMBB) const {

    // Filter out uses that are clearly unreachable
    if (MBB != UseMBB && !isReachable(MBB, UseMBB))
      return false;

    // PHI uses are considered part of the incoming BB. Check for reachability
    // at the edge.
    if (UseMI->isPHI()) {
      const MachineBasicBlock *EdgeMBB = mbbForPhiOp(UseMO);
      if (!isReachableOrSame(MBB, EdgeMBB))
        return false;
    }

    // Filter out uses with an intermediate def.
    const MachineInstr *DefMI = MRI->getUniqueVRegDef(UseMO->getReg());
    const MachineBasicBlock *DefMBB = DefMI->getParent();
    if (MBB == UseMBB) {
      if (UseMI->isPHI() && MBB == DefMBB)
        return true;

      if (instrsAreInOrder(&MI, UseMI))
        return true;

      // A Def in the loop means that the value at MI will not survive through
      // to this use.
      MachineLoop *UseLoop = MLI->getLoopFor(UseMBB);
      return UseLoop && !UseLoop->contains(DefMBB);
    }

    if (MBB == DefMBB)
      return instrsAreInOrder(DefMI, &MI);

    MachineLoop *Loop = MLI->getLoopFor(MBB);
    if (!Loop)
      return true;

    MachineLoop *TopLoop = Loop->getOutermostLoop();
    return !TopLoop->contains(DefMBB) || !isReachable(MBB, DefMBB) ||
           !isForwardReachable(UseMBB, MBB);
  }

  //----------------------------------------------------------------------------
  // Debug/Developer Helpers
  //----------------------------------------------------------------------------
private:
  /// Goes over all MBB pairs in \p MF, calculates the shortest path between
  /// them and fills in \p ShortestPathTable.
  void populatePathTable() {
    for (const MachineBasicBlock &MBB1 : *MF) {
      for (const MachineBasicBlock &MBB2 : *MF) {
        if (&MBB1 == &MBB2)
          continue;
        getShortestPath(&MBB1, &MBB2);
      }
    }
  }

  void dumpShortestPaths() const {
    for (const auto &P : Paths) {
      const MachineBasicBlock *From = P.first.src();
      const MachineBasicBlock *To = P.first.dst();
      std::optional<double> Dist = P.second.ShortestDistance;
      errs() << "From: " << From->getName() << "-> To:" << To->getName()
             << " = " << Dist.value_or(-1.0) << "\n";
    }
  }

  void printAllDistances() {
    auto getRegNextUseDistance =
        [this](Register DefReg) -> std::optional<double> {
      const MachineInstr &DefMI = *MRI->def_instr_begin(DefReg);

      SmallVector<const MachineOperand *> Uses;
      for (MachineOperand &UseMO : MRI->use_nodbg_operands(DefReg))
        Uses.push_back(&UseMO);

      return getNextUseDistance(DefReg, DefMI, Uses);
    };

    for (const MachineBasicBlock &MBB : *MF) {
      for (const MachineInstr &MI : *&MBB) {
        for (const MachineOperand &MO : MI.operands()) {
          if (!MO.isReg() || MO.isUse())
            continue;

          Register Reg = MO.getReg();
          if (Reg.isPhysical() || TRI->isAGPR(*MRI, Reg))
            continue;

          std::optional<double> NextUseDistance = getRegNextUseDistance(Reg);
          errs() << "Next-use distance of Register " << printReg(Reg, TRI)
                 << " = ";
          if (NextUseDistance)
            errs() << Fmt(*NextUseDistance);
          else
            errs() << "null";
          errs() << "\n";
        }
      }
    }
  }

  //----------------------------------------------------------------------------
  // Helper methods for printFurthestDistancesAsJson
  //----------------------------------------------------------------------------
private:
  void collectDefinedRegisters(const MachineInstr &MI,
                               SmallSet<unsigned, 4> &Defs) const {
    for (const MachineOperand &MO : MI.all_defs())
      if (MO.isReg() && MO.getReg().isValid())
        Defs.insert(MO.getReg());
  }

  void processLiveRegUses(
      const MachineInstr &MI, const GCNRPTracker::LiveRegSet &LiveRegs,
      const SmallSet<unsigned, 4> &Defs,
      DenseMap<const MachineOperand *, LiveRegUse> &RelevantUses,
      LiveRegUse &Furthest, LiveRegUse *FurthestSubreg = nullptr) {

    SmallVector<const MachineOperand *> Uses;
    SmallVector<double> Distances;
    std::map<LaneBitmask, SmallVector<LiveRegUse>> UsesByMask;

    for (auto &KV : LiveRegs) {
      const unsigned Reg = KV.first;
      const LaneBitmask LaneMask = KV.second;
      if (Defs.contains(Reg))
        continue;

      Uses.clear();
      UsesByMask.clear();

      this->getUses(Reg, LaneMask, MI, Uses);
      if (Uses.empty())
        continue;

      const MachineOperand *NextUse = nullptr;
      std::optional<double> Dist;
      Dist = this->getNextUseDistance(Reg, LaneMask, MI, Uses, &Distances,
                                      &NextUse);
      if (!Dist.has_value())
        continue;

      LiveRegUse U{NextUse, Dist.value()};
      RelevantUses.try_emplace(NextUse, U);
      // if U is better than Furthest
      // if distances are equal U is better if it's reg is < Furthest reg
      if (Furthest < U) {
        Furthest = U;
      }

      // Determine furthest sub-register if requested
      if (!FurthestSubreg)
        return;

      assert(Uses.size() == Distances.size());
      SmallVector<unsigned> Indexes;
      for (size_t I = 0; I < Uses.size(); ++I) {
        const MachineOperand *MO = Uses[I];

        Indexes.clear();
        if (MO->getSubReg()) {
          Indexes.push_back(MO->getSubReg());
        } else {
          const TargetRegisterClass *RC = MRI->getRegClass(MO->getReg());
          TRI->getCoveringSubRegIndexes(RC, LaneBitmask::getAll(), Indexes);
        }
        for (unsigned Idx : Indexes) {

          LaneBitmask Mask = TRI->getSubRegIndexLaneMask(Idx);
          if (Mask.all() || Mask == LaneMask) {
            continue;
          }

          // FIXME: Integrate loop over UsesByMask here.
          UsesByMask[Mask].push_back({MO, Distances[I]});
        }
      }

      if (UsesByMask.empty()) {
        if (*FurthestSubreg < U) {
          *FurthestSubreg = U;
        }
        continue;
      }

      for (auto &KV : UsesByMask) {
        SmallVector<LiveRegUse> &SubregUses = KV.second;
        LiveRegUse SubregU;
        for (LiveRegUse &LRU : SubregUses) {
          if (!SubregU.Use || LRU < SubregU)
            SubregU = LRU;
        }

        RelevantUses.try_emplace(SubregU.Use, SubregU);
        if (*FurthestSubreg < SubregU) {
          *FurthestSubreg = SubregU;
        }
      }
    }
  }

  static std::string Quote(StringRef S) { return "\"" + S.str() + "\""; }
  static std::string Sep(bool Final) { return std::string(Final ? "" : ","); }
  static format_object<double> Fmt(double Dist) { return format("%.1f", Dist); }

  void printInstructionHeader(raw_ostream &OS, const MachineInstr &MI,
                              ModuleSlotTracker &MST) const {
    InstructionInfo Info = parseInstructionString(MI, MST);
    OS << "    {\n";
    OS << "      " << Quote("name") << ": " << Quote(Info.DefName) << ",\n";
    OS << "      " << Quote("type") << ": " << Quote(Info.DefType) << ",\n";
    OS << "      " << Quote("instr") << ": " << Quote(Info.Instr) << ",\n";
    if (DumpNextUseDistanceVerbose) {
      OS << "      " << Quote("id") << ": " << Fmt(getInstrId(&MI)) << ",\n";
      OS << "      " << Quote("head-len") << ": " << Fmt(getHeadLen(&MI))
         << ",\n";
      OS << "      " << Quote("tail-len") << ": " << Fmt(getTailLen(&MI))
         << ",\n";
    }
  }

  void printDistances(
      raw_ostream &OS,
      const DenseMap<const MachineOperand *, LiveRegUse> &Uses) const {
    OS << "      " << Quote("distances") << ": {\n";

    // Sorting isn't necessary for the purposes of JSON, but it reduces
    // FileCheck differences.
    SmallVector<const MachineOperand *> Keys;
    for (const MachineOperand *K : Uses.keys())
      Keys.push_back(K);
    std::sort(Keys.begin(), Keys.end(), [](const auto &A, const auto &B) {
      return A->getReg() < B->getReg() ||
             (A->getReg() == B->getReg() && A->getSubReg() < B->getSubReg());
    });

    unsigned rem = Uses.size();
    for (const MachineOperand *K : Keys) {
      const bool FinalUse = --rem == 0;
      const LiveRegUse &U = Uses.at(K);
      std::string RegStr =
          printRegToString(U.getReg(), U.getLaneMask(TRI), MRI, TRI);
      OS << "        ";
      OS << Quote(RegStr) << ": " << Fmt(U.Dist) << Sep(FinalUse) << "\n";
    }
    OS << "      },\n";
  }

  void printFurthestUse(raw_ostream &OS, const LiveRegUse &Furthest,
                        bool Subreg = false, bool Last = false) const {
    OS << "      " << Quote(Subreg ? "furthest-subreg" : "furthest") << ": {\n";
    if (Furthest.Use) {
      std::string RegStr = printRegToString(
          Furthest.getReg(),
          Subreg ? Furthest.getLaneMask(TRI) : LaneBitmask::getAll(), MRI, TRI);
      OS << "        " << Quote("register") << ": " << Quote(RegStr) << ",\n";
      if (DumpNextUseDistanceVerbose) {
        std::string UseStr = printToString(Furthest.Use);
        std::string UseMIStr = printToString(Furthest.Use->getParent());
        OS << "        " << Quote("use") << ": " << Quote(UseStr) << ",\n";
        OS << "        " << Quote("use-mi") << ": " << Quote(UseMIStr) << ",\n";
      }
      OS << "        " << Quote("distance") << ": " << Fmt(Furthest.Dist)
         << "\n";
    }
    OS << "      }" << (Last ? "\n" : ",\n");
  }

public:
  AMDGPUNextUseAnalysisImpl() = default;
  ~AMDGPUNextUseAnalysisImpl() { clearTables(); }

  void initialize(const MachineFunction *, const MachineLoopInfo *,
                  const MachineDominatorTree *);

  CompatibilityMode getCompatibilityMode() { return CompatMode; }
  void setCompatibilityMode(CompatibilityMode Mode) {
    CompatMode = Mode;
    clearTables();
    initializeTables();
  }

  /// \Returns the next-use distance for \p LiveReg.
  std::optional<double>
  getNextUseDistance(Register LiveReg, LaneBitmask LaneMask,
                     const MachineInstr &FromMI,
                     const SmallVector<const MachineOperand *> &Uses,
                     SmallVector<double> *Distances = nullptr,
                     const MachineOperand **UseOut = nullptr);

  std::optional<double>
  getNextUseDistance(Register LiveReg, const MachineInstr &FromMI,
                     const SmallVector<const MachineOperand *> &Uses,
                     SmallVector<double> *Distances = nullptr,
                     const MachineOperand **UseOut = nullptr) {
    return getNextUseDistance(LiveReg, LaneBitmask::getAll(), FromMI, Uses,
                              Distances, UseOut);
  }

  void getUses(unsigned Register, LaneBitmask LaneMask, const MachineInstr &MI,
               SmallVector<const MachineOperand *> &Uses);

  void printFurthestDistancesAsJson(raw_ostream &OS, const LiveIntervals *LIS);
};

void AMDGPUNextUseAnalysisImpl::initialize(const MachineFunction *MF,
                                           const MachineLoopInfo *ML,
                                           const MachineDominatorTree *DT) {

  this->MF = MF;
  this->MLI = ML;
  this->DT = DT;

  const Function *F = &MF->getFunction();
  const Module *M = F->getParent();
  ModuleSlotTracker MST(M);
  MST.incorporateFunction(*F);
  this->MST = &MST;

  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF->getRegInfo();

  if (CompatModeOpt.getNumOccurrences()) {
    CompatMode = CompatModeOpt;
  } else {
    // TODO: Set default based on subtarget?
    CompatMode = CompatibilityMode::Graphics;
  }

  initializeTables();

  if (DumpNextUseDistance) {
    populatePathTable();
    MF->print(errs());
    printAllDistances();
  }
  this->MST = nullptr;
}

std::optional<double> AMDGPUNextUseAnalysisImpl::getNextUseDistance(
    Register LiveReg, LaneBitmask LaneMask, const MachineInstr &CurMI,
    const SmallVector<const MachineOperand *> &Uses,
    SmallVector<double> *Distances, const MachineOperand **UseOut) {

  assert(!LiveReg.isPhysical() && !TRI->isAGPR(*MRI, LiveReg) &&
         "Next-use distance is calculated for SGPRs and VGPRs");
  const MachineOperand *NextUse = nullptr;
  double NextUseDistance = std::numeric_limits<double>::max();

  if (Distances) {
    Distances->clear();
    Distances->reserve(Uses.size());
  }
  for (auto *UseMO : Uses) {
    double D = calcDistanceToUse(LiveReg, LaneMask, CurMI, UseMO);
    if (D < NextUseDistance) {
      NextUseDistance = D;
      NextUse = UseMO;
    }
    if (Distances)
      Distances->push_back(D);
  }
  if (UseOut)
    *UseOut = NextUse;
  return NextUseDistance != std::numeric_limits<double>::max()
             ? std::optional<double>(NextUseDistance)
             : std::nullopt;
}

void AMDGPUNextUseAnalysisImpl::getUses(
    unsigned Reg, LaneBitmask LaneMask, const MachineInstr &MI,
    SmallVector<const MachineOperand *> &Uses) {

  const bool CheckMask = LaneMask != LaneBitmask::getAll() &&
                         LaneMask != MRI->getMaxLaneMaskForVReg(Reg);
  const MachineBasicBlock *MBB = MI.getParent();

  for (const MachineOperand *UseMO : getRegisterUses(Reg)) {
    const MachineInstr *UseMI = UseMO->getParent();
    const MachineBasicBlock *UseMBB = UseMI->getParent();

    if (CheckMask && !machineOperandCoveredBy(*UseMO, LaneMask))
      continue;

    bool Reachable;
    if (computeMode())
      Reachable = isUseReachableForCompute(MI, MBB, UseMO, UseMI, UseMBB);
    else if (MBB == UseMBB)
      Reachable = instrsAreInOrder(&MI, UseMI);
    else
      Reachable = isDistanceFinite(MBB, UseMBB);

    if (Reachable)
      Uses.push_back(UseMO);
  }
}

void AMDGPUNextUseAnalysisImpl::printFurthestDistancesAsJson(
    raw_ostream &OS, const LiveIntervals *LIS) {
  const Function *F = &MF->getFunction();
  const Module *M = F->getParent();

  GCNDownwardRPTracker RPTracker(*LIS);
  ModuleSlotTracker MST(M);
  MST.incorporateFunction(*F);

  SmallSet<unsigned, 4> Defs;
  DenseMap<const MachineOperand *, LiveRegUse> RelevantUses;

  OS << "{\n";
  for (const MachineBasicBlock &MBB : *MF) {
    const bool FinalMBB = &MBB == &MF->back();
    std::string MBBName = nameForMBB(MBB, MST);

    OS << "  " << Quote(MBBName) << ": [\n";
    const MachineInstr *PrevMI = nullptr;
    for (const MachineInstr &MI : MBB) {
      const bool FinalMI = &MI == &MBB.back();

      // Update register pressure tracker
      if (!PrevMI || PrevMI->getOpcode() == AMDGPU::PHI)
        RPTracker.reset(MI);
      RPTracker.advance();

      Defs.clear();
      collectDefinedRegisters(MI, Defs);

      LiveRegUse Furthest;
      LiveRegUse FurthestSubreg;
      RelevantUses.clear();
      processLiveRegUses(MI, RPTracker.getLiveRegs(), Defs, RelevantUses,
                         Furthest, &FurthestSubreg);

      // Print instruction JSON
      printInstructionHeader(OS, MI, MST);
      printDistances(OS, RelevantUses);
      printFurthestUse(OS, Furthest);
      printFurthestUse(OS, FurthestSubreg, /*Subreg*/ true, /*Last*/ true);

      OS << "    }" << Sep(FinalMI) << "\n";
      PrevMI = &MI;
    }
    OS << "  ]" << Sep(FinalMBB) << "\n";
  }
  OS << "}";
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// AMDGPUNextUseAnalysis
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void AMDGPUNextUseAnalysis::initialize(const MachineFunction *MF,
                                       const MachineLoopInfo *MLI,
                                       const MachineDominatorTree *DT) {
  Impl = std::make_unique<AMDGPUNextUseAnalysisImpl>();
  Impl->initialize(MF, MLI, DT);
}

AMDGPUNextUseAnalysis::CompatibilityMode
AMDGPUNextUseAnalysis::getCompatibilityMode() {
  return Impl->getCompatibilityMode();
}

void AMDGPUNextUseAnalysis::setCompatibilityMode(CompatibilityMode M) {
  Impl->setCompatibilityMode(M);
}

/// \Returns the next-use distance for \p DefReg.
std::optional<double> AMDGPUNextUseAnalysis::getNextUseDistance(
    Register LiveReg, const MachineInstr &FromMI,
    const SmallVector<const MachineOperand *> &Uses,
    SmallVector<double> *Distances, const MachineOperand **UseOut) {
  return Impl->getNextUseDistance(LiveReg, FromMI, Uses, Distances, UseOut);
}

void AMDGPUNextUseAnalysis::getUses(unsigned Register, LaneBitmask LaneMask,
                                    const MachineInstr &MI,
                                    SmallVector<const MachineOperand *> &Uses) {
  return Impl->getUses(Register, LaneMask, MI, Uses);
}

void AMDGPUNextUseAnalysis::printFurthestDistancesAsJson(
    raw_ostream &OS, const LiveIntervals *LIS) {
  Impl->printFurthestDistancesAsJson(OS, LIS);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// AMDGPUNextUseAnalysisPass
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bool AMDGPUNextUseAnalysisPass::runOnMachineFunction(MachineFunction &MF) {
  const MachineLoopInfo *MLI =
      &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  const MachineDominatorTree *DT =
      &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  NUA = std::make_unique<AMDGPUNextUseAnalysis>();
  NUA->initialize(&MF, MLI, DT);

  if (DumpNextUseDistanceAsJson.getNumOccurrences()) {
    const LiveIntervals *LIS =
        &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    std::string FN = DumpNextUseDistanceAsJson;
    if (FN.empty() || FN == "-") {
      NUA->printFurthestDistancesAsJson(outs(), LIS);
    } else {
      std::error_code EC;
      ToolOutputFile OutF(FN, EC, sys::fs::OF_None);
      NUA->printFurthestDistancesAsJson(OutF.os(), LIS);
      OutF.keep();
    }
  }

  return true;
}

void AMDGPUNextUseAnalysisPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LiveVariablesWrapperPass>();
  AU.addRequired<MachineLoopInfoWrapperPass>();

  AU.addRequired<LiveIntervalsWrapperPass>();
  AU.addRequired<SlotIndexesWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();

  AU.addPreserved<LiveVariablesWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();

  MachineFunctionPass::getAnalysisUsage(AU);
}

char AMDGPUNextUseAnalysisPass::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUNextUseAnalysisPass, DEBUG_TYPE,
                      "Next Use Analysis", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveVariablesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)

INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)

INITIALIZE_PASS_END(AMDGPUNextUseAnalysisPass, DEBUG_TYPE, "Next Use Analysis",
                    false, false)

char &llvm::AMDGPUNextUseAnalysisID = AMDGPUNextUseAnalysisPass::ID;

FunctionPass *llvm::createAMDGPUNextUseAnalysisPass() {
  return new AMDGPUNextUseAnalysisPass();
}
