//===---------------------- AMDGPUNextUseAnalysis.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AMDGPUNextUseAnalysis pass, a machine-level analysis
// that computes the distance from each instruction to the "nearest" next use of
// every live virtual register. These distances guide register spilling
// decisions by identifying which live values are furthest from their next use
// and are therefore the best candidates to spill.
//
// The analysis is based on the Braun & Hack CC'09 paper "Register Spilling and
// Live-Range Splitting for SSA-Form Programs."
//
// Key concepts:
//
//   NextUseDistance     A loop-depth-weighted instruction count representing
//                       how far away a register's next use is. Distances
//                       through deeper loops are scaled by fromLoopDepth() so
//                       that uses inside hot loops appear closer.
//
//   Inter-block         Pre-computed shortest weighted distances between all
//   distances           pairs of basic blocks, used to efficiently answer
//                       cross-block next-use queries. Each intermediate block
//                       is weighted by fromLoopDepth() applied once per loop
//                       boundary crossing relative to the destination.
//
// Configuration flags (see Config struct in the header):
//
//   CountPhis           Count PHI instructions toward distance and block size.
//   ForwardOnly         Restrict inter-block distances to forward-reachable
//                       paths.
//   PreciseUseModeling  Model PHI uses at their incoming edge block and filter
//                       uses with intermediate redefinitions.
//   PromoteToPreheader  Route loop-entry and inner-loop uses to the preheader.
//
// This file contains:
//
//   - Command-line options for configuration and debug output
//   - LiveRegUse / JSON helpers
//   - AMDGPUNextUseAnalysisImpl (the main analysis implementation)
//       - Instruction ID assignment and block size computation
//       - CFG path pre-computation (reachability, loop depth, back-edges)
//       - Inter-block distance computation
//       - Per-register next-use distance queries and caching
//   - AMDGPUNextUseAnalysis (public facade, pimpl)
//   - Legacy and new pass manager wrappers
//
//===----------------------------------------------------------------------===//

#include "AMDGPUNextUseAnalysis.h"
#include "AMDGPU.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-next-use-analysis"

//==============================================================================
// Options etc
//==============================================================================
namespace {

cl::opt<bool>
    DistanceCacheEnabled("amdgpu-next-use-analysis-distance-cache",
                         cl::init(true), cl::Hidden,
                         cl::desc("Enable live-reg-use distance cache"));

cl::opt<std::string>
    DumpNextUseDistanceAsJson("amdgpu-next-use-analysis-dump-distance-as-json",
                              cl::Hidden);

cl::opt<bool> DumpNextUseDistanceDefToUse(
    "amdgpu-next-use-analysis-dump-distance-def-to-use", cl::init(false),
    cl::Hidden);

cl::opt<bool>
    DumpNextUseDistanceVerbose("amdgpu-next-use-analysis-dump-distance-verbose",
                               cl::init(false), cl::Hidden);

// 'graphics' and 'compute' modes arose due to initial competing implementations
// of next-use analysis that emphasized different types of workloads. This
// implementation is a compromise that combines aspects of both. Over time, the
// hope is we will be able to remove some of these differences and settle on a
// more unified implementation.
cl::opt<std::string>
    ConfigPresetOpt("amdgpu-next-use-analysis-config", cl::Hidden,
                    cl::init("graphics"),
                    cl::desc("Config preset: 'graphics' or 'compute'"));

cl::opt<bool> ConfigCountPhisOpt(
    "amdgpu-next-use-analysis-count-phis", cl::Hidden,
    cl::desc("Count PHI instructions toward distance and block size"));
cl::opt<bool> ConfigForwardOnlyOpt(
    "amdgpu-next-use-analysis-forward-only", cl::Hidden,
    cl::desc("Restrict inter-block distances to forward-reachable paths"));
cl::opt<bool> ConfigPreciseUseModelingOpt(
    "amdgpu-next-use-analysis-precise-use-modeling", cl::Hidden,
    cl::desc("Model PHI uses via incoming edge block with loop-aware "
             "reachability filtering"));
cl::opt<bool> ConfigPromoteToPreheaderOpt(
    "amdgpu-next-use-analysis-use-preheader-model", cl::Hidden,
    cl::desc("Promote loop-entry and inner-loop uses to the loop preheader"));
} // namespace

//==============================================================================
// LiveRegUse - Represents a live register use with its distance. Used for
// tracking and sorting register uses by distance.
//==============================================================================
namespace {
using UseDistancePair = AMDGPUNextUseAnalysis::UseDistancePair;
struct LiveRegUse : public UseDistancePair {
  // 'nullptr' indicates an unset/invalid state.
  LiveRegUse() : UseDistancePair(nullptr, 0) {}
  LiveRegUse(const MachineOperand *Use, NextUseDistance Dist)
      : UseDistancePair(Use, Dist) {}
  LiveRegUse(const UseDistancePair &P) : UseDistancePair(P) {}

  bool isUnset() const { return Use == nullptr; }

  Register getReg() const { return Use->getReg(); }
  unsigned getSubReg() const { return Use->getSubReg(); }
  LaneBitmask getLaneMask(const SIRegisterInfo *TRI) const {
    return TRI->getSubRegIndexLaneMask(Use->getSubReg());
  }

  bool isCloserThan(const LiveRegUse &X) const {
    if (Dist < X.Dist)
      return true;

    if (Dist > X.Dist)
      return false;

    if (Use == X.Use)
      return false;

    // Ugh. When !CountPhis, PHIs and the first non-PHI instruction have id
    // 0. In this case, consider PHIs as less than the first non-PHI
    // instruction.
    const MachineInstr *ThisMI = Use->getParent();
    const MachineInstr *XMI = X.Use->getParent();
    const MachineBasicBlock *ThisMBB = ThisMI->getParent();
    if (ThisMBB == XMI->getParent()) {
      if (ThisMI->isPHI() && !XMI->isPHI() &&
          XMI == &(*ThisMBB->getFirstNonPHI()))
        return true;
    }

    // Ensure deterministic results
    return X.getReg() < getReg();
  }

  void print(raw_ostream &OS, const TargetRegisterInfo *TRI = nullptr,
             const MachineRegisterInfo *MRI = nullptr) const {
    if (isUnset()) {
      OS << "<unset>";
      return;
    }
    Dist.print(OS);
    OS << " [" << printReg(getReg(), TRI, getSubReg(), MRI) << "]";
  }

  LLVM_DUMP_METHOD void dump() const {
    print(dbgs());
    dbgs() << '\n';
  }
};

inline bool updateClosest(LiveRegUse &Closest, const LiveRegUse &X) {
  if (!Closest.Use || X.isCloserThan(Closest)) {
    Closest = X;
    return true;
  }
  return false;
}

inline bool updateFurthest(LiveRegUse &Furthest, const LiveRegUse &X) {
  if (!Furthest.Use || Furthest.isCloserThan(X)) {
    Furthest = X;
    return true;
  }
  return false;
}
} // namespace

//==============================================================================
// JSON helpers
//==============================================================================
namespace {
template <typename Lambda>
void printStringAttr(json::OStream &J, const char *Name, Lambda L) {
  J.attributeBegin(Name);
  raw_ostream &OS = J.rawValueBegin();
  OS << '"';
  L(OS);
  OS << '"';
  J.rawValueEnd();
  J.attributeEnd();
}
void printStringAttr(json::OStream &J, const char *Name, Printable P) {
  printStringAttr(J, Name, [&](raw_ostream &OS) { OS << P; });
}

void printStringAttr(json::OStream &J, const char *Name, const MachineInstr &MI,
                     ModuleSlotTracker &MST) {
  printStringAttr(J, Name, [&](raw_ostream &OS) {
    MI.print(OS, MST,
             /* IsStandalone    */ false,
             /* SkipOpers       */ false,
             /* SkipDebugLoc    */ false,
             /* AddNewLine ---> */ false,
             /* TargetInstrInfo */ nullptr);
  });
}

void printMBBNameAttr(json::OStream &J, const char *Name,
                      const MachineBasicBlock &MBB, ModuleSlotTracker &MST) {
  printStringAttr(J, Name, [&](raw_ostream &OS) {
    MBB.printName(OS, MachineBasicBlock::PrintNameIr, &MST);
  });
}

template <typename NameLambda, typename ValueT>
void printAttr(json::OStream &J, NameLambda NL, ValueT V) {
  std::string Name;
  raw_string_ostream NameOS(Name);
  NL(NameOS);
  J.attribute(NameOS.str(), V);
}

template <typename ValueT>
void printAttr(json::OStream &J, const Printable &P, ValueT V) {
  printAttr(J, [&](raw_ostream &OS) { OS << P; }, V);
}

} // namespace

//==============================================================================
// AMDGPUNextUseAnalysisImpl
//==============================================================================
class llvm::AMDGPUNextUseAnalysisImpl {
public:
  struct CacheableNextUseDistance {
    bool IsInstrRelative;
    NextUseDistance Distance;
  };
  static constexpr bool InstrRelative = true;
  static constexpr bool InstrInvariant = false;

private:
  const MachineFunction *MF = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  const MachineLoopInfo *MLI = nullptr;
  const MachineRegisterInfo *MRI = nullptr;

  using InstrIdTy = unsigned;
  using InstrToIdMap = DenseMap<const MachineInstr *, InstrIdTy>;
  InstrToIdMap InstrToId;
  AMDGPUNextUseAnalysis::Config Cfg;

  void initializeTables() {
    for (const MachineBasicBlock &BB : *MF)
      calcInstrIds(&BB, InstrToId);
    initializeCfgPaths();
    initializeInterBlockDistances();
  }

  void clearTables() {
    InstrToId.clear();
    RegUseMap.clear();
    Paths.clear();

    resetDistanceCache();
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Instruction Ids
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
private:
  unsigned sizeOf(const MachineInstr &MI) const {
    // When !Cfg.CountPhis, PHIs do not contribute to distances/sizes since they
    // generally don't result in the generation of a machine instruction.
    // FIXME: Consider using MI.isPseudo() or maybe MI.isMetaInstruction().
    return Cfg.CountPhis ? 1 : !MI.isPHI();
  }

  void calcInstrIds(const MachineBasicBlock *BB,
                    InstrToIdMap &MutableInstrToId) const {
    InstrIdTy Id = 0;
    for (auto &MI : BB->instrs()) {
      MutableInstrToId[&MI] = Id;
      Id += sizeOf(MI);
    }
  }

  /// Returns MI's instruction Id. It renumbers (part of) the BB if MI is not
  /// found in the map.
  InstrIdTy getInstrId(const MachineInstr *MI) const {
    auto It = InstrToId.find(MI);
    if (It != InstrToId.end())
      return It->second;

    // Renumber the MBB.
    // TODO: Renumber from MI onwards.
    auto &MutableInstrToId = const_cast<InstrToIdMap &>(InstrToId);
    calcInstrIds(MI->getParent(), MutableInstrToId);
    return InstrToId.find(MI)->second;
  }

  // Length of the segment from MI (inclusive) to the first instruction of the
  // basic block.
  InstrIdTy getHeadLen(const MachineInstr *MI) const {
    const MachineBasicBlock *MBB = MI->getParent();
    return getInstrId(MI) + getInstrId(&MBB->instr_front()) + 1;
  }

  // Length of the segment from MI (exclusive) to the last instruction of the
  // basic block.
  InstrIdTy getTailLen(const MachineInstr *MI) const {
    const MachineBasicBlock *MBB = MI->getParent();
    return getInstrId(&MBB->instr_back()) - getInstrId(MI);
  }

  // Length of the segment from 'From' to 'To' (exclusive). Both instructions
  // must be in the same basic block.
  InstrIdTy getDistance(const MachineInstr *From,
                        const MachineInstr *To) const {
    assert(From->getParent() == To->getParent());
    return getInstrId(To) - getInstrId(From);
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // RegUses - cache of uses by register
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
private:
  DenseMap<Register, SmallVector<const MachineOperand *>> RegUseMap;

  const SmallVector<const MachineOperand *> &
  getRegisterUses(Register Reg) const {
    auto I = RegUseMap.find(Reg);
    if (I != RegUseMap.end())
      return I->second;

    auto *NonConstThis = const_cast<AMDGPUNextUseAnalysisImpl *>(this);
    SmallVector<const MachineOperand *> &Uses = NonConstThis->RegUseMap[Reg];
    for (const MachineOperand &UseMO : MRI->use_nodbg_operands(Reg)) {
      if (!UseMO.isUndef())
        Uses.push_back(&UseMO);
    }
    return Uses;
  }

  bool hasAtLeastOneUse(Register Reg) const {
    return !getRegisterUses(Reg).empty();
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Paths
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
private:
  class Path {
    using StorageTy =
        std::pair<const MachineBasicBlock *, const MachineBasicBlock *>;
    StorageTy P;

  public:
    constexpr Path() : P(nullptr, nullptr) {}
    constexpr Path(const MachineBasicBlock *Src, const MachineBasicBlock *Dst)
        : P(Src, Dst) {}
    Path(const StorageTy &Pair) : P(Pair) {}

    constexpr operator const StorageTy &() const { return P; }
    using DenseMapInfo = llvm::DenseMapInfo<StorageTy>;

    const MachineBasicBlock *src() const { return P.first; }
    const MachineBasicBlock *dst() const { return P.second; }
  };

  enum class EdgeKind { Back = -1, None = 0, Forward = 1 };
  static constexpr StringRef toString(EdgeKind EK) {
    if (EK == EdgeKind::Back)
      return "back";
    if (EK == EdgeKind::Forward)
      return "fwd";
    return "none";
  }

  struct PathInfo {
    EdgeKind EK;
    bool Reachable;
    int ForwardReachable;
    unsigned RelativeLoopDepth;
    std::optional<NextUseDistance> ShortestDistance;
    std::optional<NextUseDistance> ShortestUnweightedDistance;
    InstrIdTy Size;

    PathInfo()
        : EK(EdgeKind::None), Reachable(false), ForwardReachable(-1),
          RelativeLoopDepth(0), Size(0) {}

    bool isBackedge() const { return EK == EdgeKind::Back; }

    bool isForwardReachableSet() const { return 0 <= ForwardReachable; }
    bool isForwardReachableUnset() const { return ForwardReachable < 0; }
    bool isForwardReachable() const { return ForwardReachable == 1; }
    bool isNotForwardReachable() const { return ForwardReachable == 0; }

    void print(raw_ostream &OS) const {
      OS << "{ek=" << toString(EK) << " reach=" << Reachable
         << " fwd-reach=" << ForwardReachable
         << " loop-depth=" << RelativeLoopDepth << " size=" << Size;
      if (ShortestDistance) {
        OS << " shortest-dist=";
        ShortestDistance->print(OS);
      }
      if (ShortestUnweightedDistance) {
        OS << " shortest-unweighted-dist=";
        ShortestUnweightedDistance->print(OS);
      }
      OS << "}";
    }

    LLVM_DUMP_METHOD void dump() const {
      print(dbgs());
      dbgs() << '\n';
    }
  };

  //----------------------------------------------------------------------------
  // Path Storage - 'Paths' is lazily populated and some members are lazily
  // computed. All mutations should go through one of the 'initializePathInfo*'
  // flavors below.
  //----------------------------------------------------------------------------
  DenseMap<Path, PathInfo, Path::DenseMapInfo> Paths;

  const PathInfo *maybePathInfoFor(const MachineBasicBlock *From,
                                   const MachineBasicBlock *To) const {
    auto I = Paths.find({From, To});
    return I == Paths.end() ? nullptr : &I->second;
  }

  PathInfo &getOrInitPathInfo(const MachineBasicBlock *From,
                              const MachineBasicBlock *To) const {
    auto *NonConstThis = const_cast<AMDGPUNextUseAnalysisImpl *>(this);
    auto &MutablePaths = NonConstThis->Paths;

    Path P(From, To);
    auto [I, Inserted] = MutablePaths.try_emplace(P);
    if (!Inserted)
      return I->second;

    bool Reachable = calcIsReachable(P.src(), P.dst());

    // Iterator may have been invalidated by calcIsReachable, so get a fresh
    // reference to the slot.
    return NonConstThis->initializePathInfo(MutablePaths.at(P), P,
                                            EdgeKind::None, Reachable);
  }

  const PathInfo &pathInfoFor(const MachineBasicBlock *From,
                              const MachineBasicBlock *To) const {
    return getOrInitPathInfo(From, To);
  }

  //----------------------------------------------------------------------------
  // initializePathInfo* - various flavors of PathInfo initialization. They
  // (should) always funnel to the first flavor below.
  //----------------------------------------------------------------------------
  PathInfo &initializePathInfo(PathInfo &Slot, Path P, EdgeKind EK,
                               bool Reachable) {
    Slot.EK = EK;
    Slot.Reachable = Reachable;
    Slot.ForwardReachable = EK == EdgeKind::None ? -1 : EK == EdgeKind::Forward;
    Slot.RelativeLoopDepth =
        Slot.Reachable ? calcRelativeLoopDepth(P.src(), P.dst()) : 0;
    Slot.Size = P.src() == P.dst() ? calcSize(P.src()) : 0;
    if (EK != EdgeKind::None)
      Slot.ShortestUnweightedDistance = 0;
    return Slot;
  }

  PathInfo &initializePathInfo(Path P, EdgeKind EK, bool Reachable) const {
    auto *NonConstThis = const_cast<AMDGPUNextUseAnalysisImpl *>(this);
    auto &MutablePaths = NonConstThis->Paths;
    return NonConstThis->initializePathInfo(MutablePaths[P], P, EK, Reachable);
  }

  std::pair<PathInfo *, bool> maybeInitializePathInfo(Path P, EdgeKind EK,
                                                      bool Reachable) const {
    auto *NonConstThis = const_cast<AMDGPUNextUseAnalysisImpl *>(this);
    auto &MutablePaths = NonConstThis->Paths;
    auto [I, Inserted] = MutablePaths.try_emplace(P);
    if (Inserted)
      NonConstThis->initializePathInfo(I->second, P, EK, Reachable);
    return {&I->second, Inserted};
  }

  bool initializePathInfoForwardReachable(const MachineBasicBlock *From,
                                          const MachineBasicBlock *To,
                                          bool Value) const {
    PathInfo &Slot = getOrInitPathInfo(From, To);
    assert(Slot.isForwardReachableUnset());
    Slot.ForwardReachable = Value;
    return Value;
  }

  NextUseDistance
  initializePathInfoShortestDistance(const MachineBasicBlock *From,
                                     const MachineBasicBlock *To,
                                     NextUseDistance Value) const {
    PathInfo &Slot = getOrInitPathInfo(From, To);
    assert(!Slot.ShortestDistance.has_value());
    Slot.ShortestDistance = Value;
    return Value;
  }

  NextUseDistance
  initializePathInfoShortestUnweightedDistance(const MachineBasicBlock *From,
                                               const MachineBasicBlock *To,
                                               NextUseDistance Value) const {
    PathInfo &Slot = getOrInitPathInfo(From, To);
    assert(!Slot.ShortestUnweightedDistance.has_value());
    Slot.ShortestUnweightedDistance = Value;
    return Value;
  }

  //----------------------------------------------------------------------------
  // initialize*Paths
  //----------------------------------------------------------------------------
private:
  void initializePaths(const SmallVector<Path> &ReachablePaths,
                       const SmallVector<Path> &UnreachablePaths) const {
    for (const Path &P : ReachablePaths)
      initializePathInfo(P, EdgeKind::None, true);
    for (const Path &P : UnreachablePaths)
      initializePathInfo(P, EdgeKind::None, false);
  }

  void
  initializeForwardOnlyPaths(const SmallVector<Path> &ReachablePaths,
                             const SmallVector<Path> &UnreachablePaths) const {
    for (bool R : {true, false}) {
      const auto &ToInit = R ? ReachablePaths : UnreachablePaths;
      for (const Path &P : ToInit) {
        PathInfo &Slot = getOrInitPathInfo(P.src(), P.dst());
        assert(Slot.isForwardReachableUnset() || Slot.ForwardReachable == R);
        Slot.ForwardReachable = R;
      }
    }
  }

  // Follow the control flow graph starting at the entry block until all blocks
  // have been visited. Along the way, initialize the PathInfo for each edge
  // traversed.
  void initializeCfgPaths() {
    Paths.clear();

    enum VisitState { Undiscovered, Visiting, Finished };
    DenseMap<const MachineBasicBlock *, VisitState> State;

    SmallVector<const MachineBasicBlock *> Work{&MF->front()};
    State[&MF->front()] = Undiscovered;

    while (!Work.empty()) {
      const MachineBasicBlock *Src = Work.back();
      VisitState &SrcState = State[Src];

      // A block may already be 'Finished' if it is reachable from multiple
      // predecessors causing it to be pushed more than once while still
      // 'Undiscovered'.
      if (SrcState == Visiting || SrcState == Finished) {
        Work.pop_back();
        SrcState = Finished;
        continue;
      }

      SrcState = Visiting;
      for (const MachineBasicBlock *Dst : Src->successors()) {
        const VisitState DstState = State.lookup(Dst);

        EdgeKind EK;
        if (DstState == Undiscovered) {
          EK = EdgeKind::Forward;
          Work.push_back(Dst);
        } else if (DstState == Visiting) {
          EK = EdgeKind::Back;
        } else {
          EK = EdgeKind::Forward;
        }

        Path P(Src, Dst);
        assert(!Paths.contains(P));
        initializePathInfo(P, EK, /*Reachable*/ true);
      }
    }

    LLVM_DEBUG(dumpPaths());
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Loop helpers
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
private:
  static bool isStandAloneLoop(const MachineLoop *Loop) {
    return Loop->getSubLoops().empty() && Loop->isOutermost();
  }

  static MachineLoop *findChildLoop(MachineLoop *const Parent,
                                    MachineLoop *Descendant) {
    for (MachineLoop *L = Descendant; L != Parent; L = L->getParentLoop()) {
      if (L->getParentLoop() == Parent)
        return L;
    }
    return nullptr;
  }

  // If loops 'A' and 'B' share a common parent loop, return that loop and the
  // depth of 'A' relative to it. Otherwise return nullptr and the loop depth of
  // 'A'.
  static std::pair<MachineLoop *, unsigned>
  findCommonParent(MachineLoop *A, const MachineLoop *B) {
    unsigned Depth = 0;
    for (; A != nullptr; A = A->getParentLoop(), ++Depth) {
      if (A->contains(B))
        break;
    }
    return {A, Depth};
  }

  static const MachineBasicBlock *
  getOutermostPreheader(const MachineLoop *Loop) {
    return Loop ? Loop->getOutermostLoop()->getLoopPreheader() : nullptr;
  }

  static MachineBasicBlock *findChildPreheader(MachineLoop *const Parent,
                                               MachineLoop *Descendant) {
    MachineLoop *ChildLoop = findChildLoop(Parent, Descendant);
    return ChildLoop ? ChildLoop->getLoopPreheader() : nullptr;
  }

  static const MachineBasicBlock *
  getIncomingBlockIfPhiUse(const MachineInstr *MI, const MachineOperand *MO) {
    return MI->isPHI() ? MI->getOperand(MO->getOperandNo() + 1).getMBB()
                       : nullptr;
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Calculate features
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
private:
  InstrIdTy calcSize(const MachineBasicBlock *BB) const {
    InstrIdTy Size = BB->size();
    if (!Cfg.CountPhis)
      Size -= std::distance(BB->begin(), BB->getFirstNonPHI());
    return Size;
  }

  NextUseDistance calcWeightedSize(const MachineBasicBlock *From,
                                   const MachineBasicBlock *To) const {
    return NextUseDistance::fromSize(getSize(From),
                                     getRelativeLoopDepth(From, To));
  }

  // Return the loop depth of 'From' relative to 'To'.
  unsigned calcRelativeLoopDepth(const MachineBasicBlock *From,
                                 const MachineBasicBlock *To) const {
    MachineLoop *LoopFrom = MLI->getLoopFor(From);
    MachineLoop *LoopTo = MLI->getLoopFor(To);

    if (!LoopFrom)
      return 0;

    if (!LoopTo)
      return LoopFrom->getLoopDepth();

    if (LoopFrom->contains(LoopTo)) // covers LoopFrom == LoopTo
      return 0;

    if (LoopTo->contains(LoopFrom))
      return LoopFrom->getLoopDepth() - LoopTo->getLoopDepth();

    // Loops are siblings of some sort.
    return findCommonParent(LoopFrom, LoopTo).second;
  }

  // Attempt to find a path from 'From' to 'To' using a depth first search. If
  // 'ForwardOnly' is true, do not follow backedges. As a performance
  // improvement, this may initialize reachable intermediate paths or paths we
  // determine are unreachable.
  bool calcIsReachable(const MachineBasicBlock *From,
                       const MachineBasicBlock *To,
                       bool ForwardOnly = false) const {
    if (From == To && !MLI->getLoopFor(From))
      return false;

    if (!ForwardOnly && interBlockDistanceExists(From, To))
      return true;

    enum { VisitOp, PopOp };
    using MBBOpPair = std::pair<const MachineBasicBlock *, int>;
    SmallVector<MBBOpPair> Work{{From, VisitOp}};
    DenseSet<const MachineBasicBlock *> Visited{From};

    SmallVector<Path> IntermediatePath;
    SmallVector<Path> Unreachable;

    // Should be run at every function exit point.
    auto Finally = [&](bool Reachable) {
      // This is an optimization. For intermediate paths we found while
      // calculating reachability for 'From' --> 'To', remember their
      // reachability.
      if (!Reachable) {
        IntermediatePath.clear();
        for (const MachineBasicBlock *MBB : Visited) {
          if (MBB != From)
            Unreachable.emplace_back(MBB, To);
        }
      }

      if (ForwardOnly)
        initializeForwardOnlyPaths(IntermediatePath, Unreachable);
      else
        initializePaths(IntermediatePath, Unreachable);

      return Reachable;
    };

    while (!Work.empty()) {
      auto [Current, Op] = Work.pop_back_val();

      // Backtracking
      if (Op == PopOp) {
        IntermediatePath.pop_back();
        if (ForwardOnly)
          Unreachable.emplace_back(Current, To);
        continue;
      }

      if (Current->succ_empty())
        continue;

      if (Current != From) {
        IntermediatePath.emplace_back(Current, To);
        Work.emplace_back(Current, PopOp);
      }

      for (const MachineBasicBlock *Succ : Current->successors()) {
        if (ForwardOnly && isBackedge(Current, Succ))
          continue;

        if (Succ == To)
          return Finally(true);

        if (auto CachedReachable = isMaybeReachable(Succ, To, ForwardOnly)) {
          if (CachedReachable.value())
            return Finally(true);
          Visited.insert(Succ);
          continue;
        }

        if (Visited.insert(Succ).second)
          Work.emplace_back(Succ, VisitOp);
      }
    }

    return Finally(false);
  }

  //----------------------------------------------------------------------------
  // Inter-block distance - the weighted and unweighted cost (i.e. "distance")
  // to travel from one MachineBasicBlock to another.
  //
  // Values are pre-computed and stored in 'InterBlockDistances' using a
  // backwards data-flow algorithm similar to the one described in 4.1 of a
  // "Register Spilling and Live-Range Splitting for SSA-Form Programs" by
  // Matthias Braun and Sebastian Hack, CC'09. This replaced a prior
  // implementation based on Dijkstra's shortest path algorithm.
  //----------------------------------------------------------------------------
private:
  struct InterBlockDistance {
    NextUseDistance Weighted;
    NextUseDistance Unweighted;
    InterBlockDistance() : Weighted(-1), Unweighted(-1) {}
    InterBlockDistance(NextUseDistance W, NextUseDistance UW)
        : Weighted(W), Unweighted(UW) {}
    bool operator==(const InterBlockDistance &Other) const {
      return Weighted == Other.Weighted && Unweighted == Other.Unweighted;
    }
    bool operator!=(const InterBlockDistance &Other) const {
      return !(*this == Other);
    }

    void print(raw_ostream &OS) const {
      OS << "{W=";
      Weighted.print(OS);
      OS << " U=";
      Unweighted.print(OS);
      OS << "}";
    }

    LLVM_DUMP_METHOD void dump() const {
      print(dbgs());
      dbgs() << '\n';
    }
  };
  using InterBlockDistanceMap =
      DenseMap<unsigned, DenseMap<unsigned, InterBlockDistance>>;
  InterBlockDistanceMap InterBlockDistances;

  void initializeInterBlockDistances() {
    InterBlockDistanceMap Distances;

    bool Changed;
    do {
      Changed = false;
      for (const MachineBasicBlock *MBB : post_order(MF)) {
        unsigned MBBNum = MBB->getNumber();

        // Save previous state for convergence check
        InterBlockDistanceMap::mapped_type Prev = std::move(Distances[MBBNum]);
        InterBlockDistanceMap::mapped_type Curr;
        Curr.reserve(Prev.size());

        // Direct successors are distance 0 by definition: no instructions are
        // executed between exiting MBB and entering Succ.
        for (const MachineBasicBlock *Succ : MBB->successors())
          Curr[Succ->getNumber()] = InterBlockDistance(0, 0);

        // Propagate further destinations through each successor.
        for (const MachineBasicBlock *Succ : MBB->successors()) {
          unsigned SuccNum = Succ->getNumber();
          const unsigned UnweightedSize{getSize(Succ)};

          for (const auto &[DestBlockNum, DestDist] : Distances[SuccNum]) {
            // MBB -> MBB is considered unreachable (getInterBlockDistance
            // asserts From != To).
            if (DestBlockNum == MBBNum)
              continue;

            const MachineBasicBlock *DestMBB =
                MF->getBlockNumbered(DestBlockNum);

            const NextUseDistance UnweightedDist{UnweightedSize +
                                                 DestDist.Unweighted};

            unsigned SuccToDestLoopDepth = calcRelativeLoopDepth(Succ, DestMBB);

            const NextUseDistance WeightedDist =
                DestDist.Weighted +
                NextUseDistance::fromSize(UnweightedSize, SuccToDestLoopDepth);

            // Insert or update distances (take minimum)
            auto [I, First] =
                Curr.try_emplace(DestBlockNum, WeightedDist, UnweightedDist);
            if (!First) {
              InterBlockDistance &Slot = I->second;
              Slot.Weighted = min(Slot.Weighted, WeightedDist);
              Slot.Unweighted = min(Slot.Unweighted, UnweightedDist);
            }
          }
        }
        Changed |= (Prev != Curr);
        Distances[MBBNum] = std::move(Curr);
      }
    } while (Changed);

    InterBlockDistances = std::move(Distances);
    LLVM_DEBUG(dumpInterBlockDistances());
  }

  const InterBlockDistance *
  getInterBlockDistanceMapValue(const MachineBasicBlock *From,
                                const MachineBasicBlock *To) const {
    auto I = InterBlockDistances.find(From->getNumber());
    if (I == InterBlockDistances.end())
      return nullptr;
    const InterBlockDistanceMap::mapped_type &FromSlot = I->second;
    auto J = FromSlot.find(To->getNumber());
    return J == FromSlot.end() ? nullptr : &J->second;
  }

  bool interBlockDistanceExists(const MachineBasicBlock *From,
                                const MachineBasicBlock *To) const {
    return getInterBlockDistanceMapValue(From, To);
  }

  NextUseDistance getInterBlockDistance(const MachineBasicBlock *From,
                                        const MachineBasicBlock *To,
                                        bool Unweighted) const {

    assert(From != To && "The basic blocks should be different.");
    if (!From || !To)
      return NextUseDistance::unreachable();

    if (Cfg.ForwardOnly && !isForwardReachable(From, To))
      return NextUseDistance::unreachable();

    const InterBlockDistance *BD = getInterBlockDistanceMapValue(From, To);
    if (!BD)
      return NextUseDistance::unreachable();

    return Unweighted ? BD->Unweighted : BD->Weighted;
  }

  NextUseDistance
  getWeightedInterBlockDistance(const MachineBasicBlock *From,
                                const MachineBasicBlock *To) const {
    return getInterBlockDistance(From, To, false);
  }

  NextUseDistance
  getUnweightedInterBlockDistance(const MachineBasicBlock *From,
                                  const MachineBasicBlock *To) const {
    return getInterBlockDistance(From, To, true);
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Feature getters. Use cached results if available. If not calculate.
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
private:
  InstrIdTy getSize(const MachineBasicBlock *BB) const {
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
    const PathInfo &PI = pathInfoFor(From, To);
    if (PI.isForwardReachableSet())
      return PI.isForwardReachable();

    return initializePathInfoForwardReachable(
        From, To,
        PI.Reachable && calcIsReachable(From, To, /*ForwardOnly*/ true));
  }

  // Return true/false if we know that 'To' is reachable or not from
  // 'From'. Otherwise return 'std::nullopt'.
  std::optional<bool> isMaybeReachable(const MachineBasicBlock *From,
                                       const MachineBasicBlock *To,
                                       bool ForwardOnly) const {
    const PathInfo *PI = maybePathInfoFor(From, To);
    if (!PI)
      return std::nullopt;

    if (ForwardOnly) {
      if (PI->isForwardReachable())
        return true;

      if (PI->isNotForwardReachable())
        return false;
      return std::nullopt;
    }
    return PI->Reachable;
  }

  bool isBackedge(const MachineBasicBlock *From,
                  const MachineBasicBlock *To) const {
    return pathInfoFor(From, To).isBackedge();
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

  unsigned getRelativeLoopDepth(const MachineBasicBlock *From,
                                const MachineBasicBlock *To) const {
    return pathInfoFor(From, To).RelativeLoopDepth;
  }

  NextUseDistance getShortestPath(const MachineBasicBlock *From,
                                  const MachineBasicBlock *To) const {
    std::optional<NextUseDistance> MaybeD =
        pathInfoFor(From, To).ShortestDistance;
    if (MaybeD.has_value())
      return MaybeD.value();

    NextUseDistance Dist = getWeightedInterBlockDistance(From, To);
    return initializePathInfoShortestDistance(From, To, Dist);
  }

  NextUseDistance getShortestUnweightedPath(const MachineBasicBlock *From,
                                            const MachineBasicBlock *To) const {
    std::optional<NextUseDistance> MaybeD =
        pathInfoFor(From, To).ShortestUnweightedDistance;
    if (MaybeD.has_value())
      return MaybeD.value();

    return initializePathInfoShortestUnweightedDistance(
        From, To, getUnweightedInterBlockDistance(From, To));
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  /// MBBDistPair - Represents the distance to a machine basic block.
  /// Used for returning both the distance and the target block together.
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
private:
  struct MBBDistPair {
    NextUseDistance Distance;
    const MachineBasicBlock *MBB;
    MBBDistPair() : Distance(NextUseDistance::unreachable()), MBB(nullptr) {}
    MBBDistPair(NextUseDistance D, const MachineBasicBlock *B)
        : Distance(D), MBB(B) {}

    MBBDistPair operator+(NextUseDistance D) { return {Distance + D, MBB}; }
    MBBDistPair &operator+=(NextUseDistance D) {
      Distance += D;
      return *this;
    }

    void print(raw_ostream &OS) const {
      OS << "{";
      Distance.print(OS);
      if (MBB)
        OS << " " << printMBBReference(*MBB);
      else
        OS << " <null>";
      OS << "}";
    }

    LLVM_DUMP_METHOD void dump() const {
      print(dbgs());
      dbgs() << '\n';
    }
  };

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CFG Helpers
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
private:
  // Return the shortest distance to a latch
  MBBDistPair calcShortestDistanceToLatch(const MachineBasicBlock *CurMBB,
                                          const MachineLoop *CurLoop) const {
    SmallVector<MachineBasicBlock *, 2> Latches;
    CurLoop->getLoopLatches(Latches);
    MBBDistPair LD;

    for (MachineBasicBlock *LMBB : Latches) {
      if (LMBB == CurMBB)
        return {0, CurMBB};

      NextUseDistance Dst = getShortestPath(CurMBB, LMBB);
      if (Dst < LD.Distance) {
        LD.Distance = Dst;
        LD.MBB = LMBB;
      }
    }
    return LD;
  }

  // Return the shortest unweighted distance to a latch
  MBBDistPair
  calcShortestUnweightedDistanceToLatch(const MachineBasicBlock *CurMBB,
                                        const MachineLoop *CurLoop) const {
    SmallVector<MachineBasicBlock *, 2> Latches;
    CurLoop->getLoopLatches(Latches);
    MBBDistPair LD;

    for (MachineBasicBlock *LMBB : Latches) {
      if (LMBB == CurMBB)
        return {0, CurMBB};

      NextUseDistance Dst = getShortestUnweightedPath(CurMBB, LMBB);
      if (Dst < LD.Distance) {
        LD.Distance = Dst;
        LD.MBB = LMBB;
      }
    }
    return LD;
  }

  // Return the shortest distance to an exit
  MBBDistPair calcShortestDistanceToExit(const MachineBasicBlock *CurMBB,
                                         const MachineLoop *CurLoop) const {
    SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock *>> ExitEdges;
    CurLoop->getExitEdges(ExitEdges);
    MBBDistPair LD;

    for (auto [Exit, Dest] : ExitEdges) {
      if (Exit == CurMBB)
        return {0, CurMBB};

      NextUseDistance Dst = getShortestPath(CurMBB, Exit);
      if (Dst < LD.Distance) {
        LD.Distance = Dst;
        LD.MBB = Exit;
      }
    }
    return LD;
  }

  // Return the shortest distance through a loop (header to latch) that goes
  // through CurMBB.
  MBBDistPair
  calcShortestDistanceThroughInnermostLoop(const MachineBasicBlock *CurMBB,
                                           MachineLoop *CurLoop) const {
    assert(MLI->getLoopFor(CurMBB) == CurLoop);

    // This is a hot spot. Check it before doing anything else.
    if (CurLoop->getNumBlocks() == 1)
      return {getSize(CurMBB), CurMBB};

    MachineBasicBlock *LoopHeader = CurLoop->getHeader();
    MBBDistPair LD{0, nullptr};

    LD += getSize(LoopHeader);

    if (CurMBB != LoopHeader)
      LD += getShortestPath(LoopHeader, CurMBB);

    if (CurLoop->isLoopExiting(CurMBB))
      LD.MBB = CurMBB;
    else
      LD = calcShortestDistanceToExit(CurMBB, CurLoop) + LD.Distance;

    if (CurMBB != LoopHeader && CurMBB != LD.MBB)
      LD += getSize(CurMBB);

    if (LD.MBB != LoopHeader)
      LD += getSize(LD.MBB);

    return LD;
  }

  // Return the shortest distance through a loop (header to latch) that goes
  // through CurMBB.
  MBBDistPair calcShortestDistanceThroughLoop(const MachineBasicBlock *CurMBB,
                                              MachineLoop *OuterLoop) const {
    MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
    MBBDistPair CurLD =
        calcShortestDistanceThroughInnermostLoop(CurMBB, CurLoop);

    MachineBasicBlock *CurHdr = CurLoop->getHeader();
    for (;;) {
      if (OuterLoop == CurLoop)
        return CurLD;

      MachineLoop *ParentLoop = CurLoop->getParentLoop();
      MachineBasicBlock *ParentHdr = ParentLoop->getHeader();

      MBBDistPair LD{0, nullptr};
      LD += getSize(ParentHdr);
      LD += getShortestPath(ParentHdr, CurHdr);
      LD += CurLD.Distance.applyLoopWeight();
      LD = calcShortestDistanceToExit(CurLD.MBB, ParentLoop) + LD.Distance;
      LD += getSize(LD.MBB);
      CurLD = LD;
      CurLoop = ParentLoop;
      CurHdr = ParentHdr;
    }
    llvm_unreachable("CurMBB not contained in OuterLoop");
  }

  // Similar to calcShortestDistanceThroughLoop with LoopWeight applied to the
  // returned distance.
  MBBDistPair
  calcWeightedDistanceThroughLoopViaMBB(const MachineBasicBlock *CurMBB,
                                        MachineLoop *CurLoop) const {
    MBBDistPair LD = calcShortestDistanceThroughLoop(CurMBB, CurLoop);
    LD.Distance = LD.Distance.applyLoopWeight();
    return LD;
  }

  // Return the weighted, shortest distance through a loop (header to latch).
  // If ParentLoop is provided, use it to adjust the loop depth.
  MBBDistPair calcWeightedDistanceThroughLoop(
      const MachineBasicBlock *CurMBB, MachineLoop *CurLoop,
      const MachineLoop *ParentLoop = nullptr) const {
    if (CurLoop->getNumBlocks() != 1)
      return calcWeightedDistanceThroughLoopViaMBB(CurMBB, CurLoop);

    unsigned LoopDepth = MLI->getLoopDepth(CurMBB);
    if (ParentLoop)
      LoopDepth -= ParentLoop->getLoopDepth();

    return {NextUseDistance::fromSize(getSize(CurMBB), LoopDepth),
            CurLoop->getLoopLatch()};
  }

  // Calculate total distance from exit point to use instruction
  NextUseDistance appendDistanceToUse(const MBBDistPair &Exit,
                                      const MachineInstr *UseMI,
                                      const MachineBasicBlock *UseMBB) const {
    return Exit.Distance + getShortestPath(Exit.MBB, UseMBB) +
           getHeadLen(UseMI);
  }

  // Return the weighted, shortest distance through the CurLoop which is a
  // sub-loop of UseLoop.
  MBBDistPair calcDistanceThroughSubLoopUse(const MachineBasicBlock *CurMBB,
                                            MachineLoop *CurLoop,
                                            MachineLoop *UseLoop) const {
    // All the sub-loops of the UseLoop will be executed before the use.
    // Hence, we should take this into consideration in distance calculation.
    MachineLoop *UseLoopSubLoop = findChildLoop(UseLoop, CurLoop);
    assert(UseLoopSubLoop && "CurLoop should be nested in UseLoop");
    return calcWeightedDistanceThroughLoop(CurMBB, UseLoopSubLoop, UseLoop);
  }

  // Similar to calcDistanceThroughSubLoopUse, adding the distance to 'UseMI'.
  NextUseDistance calcDistanceThroughSubLoopToUseMI(
      const MachineBasicBlock *CurMBB, MachineLoop *CurLoop,
      const MachineInstr *UseMI, const MachineBasicBlock *UseMBB,
      MachineLoop *UseLoop) const {
    return appendDistanceToUse(
        calcDistanceThroughSubLoopUse(CurMBB, CurLoop, UseLoop), UseMI, UseMBB);
  }

  // Return the weighted distance through a loop to an outside use loop.
  // Differentiates between uses inside or outside of the current loop nest.
  MBBDistPair calcDistanceThroughLoopToOutsideLoopUse(
      const MachineBasicBlock *CurMBB, MachineLoop *CurLoop,
      const MachineBasicBlock *UseMBB, MachineLoop *UseLoop) const {
    assert(!CurLoop->contains(UseLoop));

    if (isStandAloneLoop(CurLoop))
      return calcWeightedDistanceThroughLoopViaMBB(CurMBB, CurLoop);

    MachineLoop *OutermostLoop = CurLoop->getOutermostLoop();
    if (!OutermostLoop->contains(UseLoop)) {
      // We should take into consideration the whole loop nest in the
      // calculation of the distance because we will reach the use after
      // executing the whole loop nest.

      // ... But make sure that we pick a route that goes through CurMBB
      return calcWeightedDistanceThroughLoopViaMBB(CurMBB, OutermostLoop);
    }

    // At this point we know that CurLoop and UseLoop are independent and they
    // are in the same loop nest.

    if (MLI->getLoopDepth(CurMBB) <= MLI->getLoopDepth(UseMBB))
      return calcWeightedDistanceThroughLoop(CurMBB, CurLoop);

    assert(CurLoop != OutermostLoop && "The loop cannot be the outermost.");
    const unsigned UseLoopDepth = MLI->getLoopDepth(UseMBB);
    for (;;) {
      if (CurLoop->getLoopDepth() == UseLoopDepth)
        break;
      CurLoop = CurLoop->getParentLoop();
      if (CurLoop == OutermostLoop)
        break;
    }
    return calcWeightedDistanceThroughLoop(CurMBB, CurLoop);
  }

  // Similar to calcDistanceThroughLoopToOutsideLoopUse but adds the distance to
  // an instruction in the loop.
  NextUseDistance calcDistanceThroughLoopToOutsideLoopUseMI(
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

    const unsigned NumOps = UseMI->getNumOperands();
    for (unsigned I = 1; I < NumOps; I += 2) {
      const MachineOperand &RegMO = UseMI->getOperand(I - 1);
      const MachineOperand &MBBMO = UseMI->getOperand(I);
      assert(RegMO.isReg() && "Expected register operand of PHI");
      assert(MBBMO.isMBB() && "Expected MBB operand of PHI");
      if (RegMO.getReg() == LiveReg &&
          machineOperandCoveredBy(RegMO, LiveLaneMask)) {
        MachineBasicBlock *IncomingBB = MBBMO.getMBB();
        if (llvm::is_contained(Latches, IncomingBB))
          return true;
      }
    }
    return false;
  }

  // Return the distance from 'CurMI' through a parent loop backedge PHI Use
  // ('UseMI').
  CacheableNextUseDistance calcDistanceViaEnclosingBackedge(
      const MachineInstr *CurMI, const MachineBasicBlock *CurMBB,
      MachineLoop *CurLoop, const MachineInstr *UseMI,
      const MachineBasicBlock *UseMBB, MachineLoop *UseLoop) const {
    assert(UseLoop && "There is no backedge.");
    assert(CurLoop && (UseLoop != CurLoop) && UseLoop->contains(CurLoop) &&
           "Unexpected loop configuration");

    InstrIdTy UseHeadLen = getHeadLen(UseMI);
    MBBDistPair InnerLoopLD =
        calcDistanceThroughSubLoopUse(CurMBB, CurLoop, UseLoop);
    MBBDistPair LD = calcShortestDistanceToLatch(InnerLoopLD.MBB, UseLoop);
    return {InstrInvariant,
            InnerLoopLD.Distance + LD.Distance + getSize(LD.MBB) + UseHeadLen};
  }

  // Optimized version of calcBackedgeDistance when we already know that CurMI
  // and UseMI are in the same basic block
  NextUseDistance calcBackedgeDistance(const MachineInstr *CurMI,
                                       const MachineBasicBlock *CurMBB,
                                       MachineLoop *CurLoop,
                                       const MachineInstr *UseMI) const {
    // use is in the next loop iteration
    InstrIdTy CurTailLen = getTailLen(CurMI);
    InstrIdTy UseHeadLen = getHeadLen(UseMI);
    MBBDistPair LD = calcShortestUnweightedDistanceToLatch(CurMBB, CurLoop);
    const MachineBasicBlock *HdrMBB = CurLoop->getHeader();
    NextUseDistance Hdr = CurMBB == HdrMBB ? 0 : getSize(HdrMBB);
    NextUseDistance Dst =
        CurMBB == HdrMBB ? 0 : getShortestUnweightedPath(HdrMBB, CurMBB);

    return CurTailLen + LD.Distance + getSize(LD.MBB) + Hdr + Dst + UseHeadLen;
  }

  //----------------------------------------------------------------------------
  // Calculate inter-instruction distances
  //----------------------------------------------------------------------------
private:
  // Calculate the shortest weighted path from MachineInstruction 'FromMI' to
  // 'ToMI'. It is weighted distance in that paths that exit loops are made to
  // look much further away.
  NextUseDistance calcShortestDistance(const MachineInstr *FromMI,
                                       const MachineInstr *ToMI) const {
    const MachineBasicBlock *FromMBB = FromMI->getParent();
    const MachineBasicBlock *ToMBB = ToMI->getParent();

    if (FromMBB == ToMBB) {
      NextUseDistance RV = getDistance(FromMI, ToMI);
      assert(RV >= 0 && "unexpected negative distance from getDistance");
      return RV;
    }

    InstrIdTy FromTailLen = getTailLen(FromMI);
    InstrIdTy ToHeadLen = getHeadLen(ToMI);
    NextUseDistance Dst = getShortestPath(FromMBB, ToMBB);
    assert(Dst.isReachable() &&
           "calcShortestDistance called for instructions in non-reachable"
           " basic blocks!");
    NextUseDistance RV = FromTailLen + Dst + ToHeadLen;
    assert(RV >= 0 && "unexpected negative distance");
    return RV;
  }

  // Calculate the shortest unweighted path from MachineInstruction 'FromMI' to
  // 'ToMI'. In contrast with 'calcShortestDistance', distances are based solely
  // on basic block instruction counts and traversing a loop exit does not
  // affect the value.
  NextUseDistance
  calcShortestUnweightedDistance(const MachineInstr *FromMI,
                                 const MachineInstr *ToMI) const {
    const MachineBasicBlock *FromMBB = FromMI->getParent();
    const MachineBasicBlock *ToMBB = ToMI->getParent();

    if (FromMBB == ToMBB)
      return getDistance(FromMI, ToMI);

    InstrIdTy FromTailLen = getTailLen(FromMI);
    InstrIdTy ToHeadLen = getHeadLen(ToMI);
    NextUseDistance Dst = getShortestUnweightedPath(FromMBB, ToMBB);
    assert(Dst.isReachable() &&
           "calcShortestUnweightedDistance called for instructions in"
           " non-reachable basic blocks!");
    return FromTailLen + Dst + ToHeadLen;
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // calcDistanceToUse* - various flavors of calculating the distance from an
  // instruction 'CurMI' to the use of a live [sub]register.
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
private:
  // Return the distance from 'CurMI' to a live [sub]register use ('UseMI').
  //
  // Cfg flags controlling behavior:
  //   PreciseUseModeling — rewrite PHI uses to their incoming edge block;
  //                        also selects unweighted cross-block distance
  //   PromoteToPreheader — route loop-entry / inner-loop uses to the preheader
  CacheableNextUseDistance
  calcDistanceToUse(Register LiveReg, LaneBitmask LiveLaneMask,
                    const MachineInstr &CurMI,
                    const MachineOperand *UseMO) const {
    const MachineInstr *UseMI = UseMO->getParent();
    const MachineBasicBlock *CurMBB = CurMI.getParent();
    const MachineBasicBlock *UseMBB = UseMI->getParent();
    MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
    MachineLoop *UseLoop = MLI->getLoopFor(UseMBB);

    if (Cfg.PreciseUseModeling) {
      // Map PHI use to the end of its incoming edge block.
      if (auto *PhiUseEdge = getIncomingBlockIfPhiUse(UseMI, UseMO)) {
        UseMI = &PhiUseEdge->back();
        UseMBB = PhiUseEdge;
        UseLoop = MLI->getLoopFor(PhiUseEdge);
      }
    }

    enum class LoopConfig {
      NoCur,
      Same,
      CurContainsUse,
      UseContainsCur,
      Siblings,
      Unrelated
    };
    auto [LpCfg, PreHdr, CommonParent] = [&]()
        -> std::tuple<LoopConfig, const MachineBasicBlock *, MachineLoop *> {
      if (!CurLoop) {
        return {LoopConfig::NoCur, getOutermostPreheader(UseLoop), nullptr};
      }
      if (CurLoop->contains(UseLoop)) {
        return {CurMBB == UseMBB ? LoopConfig::Same
                                 : LoopConfig::CurContainsUse,
                findChildPreheader(CurLoop, UseLoop), nullptr};
      }

      if (MachineLoop *P = findCommonParent(UseLoop, CurLoop).first) {
        if (P != UseLoop)
          return {LoopConfig::Siblings, findChildPreheader(P, UseLoop), P};
        return {LoopConfig::UseContainsCur, nullptr, nullptr};
      }
      return {LoopConfig::Unrelated, getOutermostPreheader(UseLoop), nullptr};
    }();

    //--------------------------------------------------------------------------
    // Don't PromoteToPreheader
    //--------------------------------------------------------------------------
    if (!Cfg.PromoteToPreheader) {
      switch (LpCfg) {
      case LoopConfig::NoCur:
      case LoopConfig::Same:
      case LoopConfig::CurContainsUse:
        return {InstrRelative, calcShortestDistance(&CurMI, UseMI)};

      case LoopConfig::UseContainsCur: {
        if (isIncomingValFromBackedge(LiveReg, LiveLaneMask, &CurMI, UseMI)) {
          return calcDistanceViaEnclosingBackedge(&CurMI, CurMBB, CurLoop,
                                                  UseMI, UseMBB, UseLoop);
        }

        return {InstrInvariant, calcDistanceThroughSubLoopToUseMI(
                                    CurMBB, CurLoop, UseMI, UseMBB, UseLoop)};
      }
      case LoopConfig::Siblings:
      case LoopConfig::Unrelated:
        return {InstrInvariant, calcDistanceThroughLoopToOutsideLoopUseMI(
                                    CurMBB, CurLoop, UseMI, UseMBB, UseLoop)};
      }
      llvm_unreachable("unexpected loop configuration!");
    }

    //--------------------------------------------------------------------------
    // PromoteToPreheader
    //--------------------------------------------------------------------------
    if (PreHdr) {
      UseMI = &PreHdr->back();
      UseMBB = PreHdr;
      UseLoop = CommonParent;
    }

    switch (LpCfg) {
    case LoopConfig::NoCur:
      return {InstrRelative, calcShortestUnweightedDistance(&CurMI, UseMI) -
                                 (sizeOf(*UseMI) ? 0 : 1)};

    case LoopConfig::Same:
    case LoopConfig::CurContainsUse:
      if (CurMBB == UseMBB && !instrsAreInOrder(&CurMI, UseMI))
        return {InstrRelative,
                calcBackedgeDistance(&CurMI, CurMBB, CurLoop, UseMI)};

      return {InstrRelative, calcShortestUnweightedDistance(&CurMI, UseMI)};

    case LoopConfig::UseContainsCur:
    case LoopConfig::Siblings:
      return {InstrInvariant, calcDistanceThroughSubLoopToUseMI(
                                  CurMBB, CurLoop, UseMI, UseMBB, UseLoop)};

    case LoopConfig::Unrelated:
      return {InstrInvariant, calcDistanceThroughLoopToOutsideLoopUseMI(
                                  CurMBB, CurLoop, UseMI, UseMBB, UseLoop)};
    }
    llvm_unreachable("unexpected loop configuration!");
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // getUses helpers (compute mode)
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
private:
  // Returns true if Use is reachable from MI. Handles backedges and intervening
  // defs.
  bool isUseReachablePrecise(const MachineInstr &MI,
                             const MachineBasicBlock *MBB,
                             const MachineOperand *UseMO,
                             const MachineInstr *UseMI,
                             const MachineBasicBlock *UseMBB) const {

    // Filter out uses that are clearly unreachable
    if (MBB != UseMBB && !isReachable(MBB, UseMBB))
      return false;

    // PHI uses are considered part of the incoming BB. Check for reachability
    // at the edge.
    if (auto *PhiUseEdge = getIncomingBlockIfPhiUse(UseMI, UseMO)) {
      if (!isReachableOrSame(MBB, PhiUseEdge))
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

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Debug/Developer Helpers
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
private:
  /// Goes over all MBB pairs in \p MF, calculates the shortest path between
  /// them.
  void populatePathTable() {
    for (const MachineBasicBlock &MBB1 : *MF) {
      for (const MachineBasicBlock &MBB2 : *MF) {
        if (&MBB1 == &MBB2)
          continue;
        getShortestPath(&MBB1, &MBB2);
      }
    }
  }

  void printPaths(raw_ostream &OS) const {
    OS << "\n---------------- Paths --------------- {\n";
    for (const auto &[P, PI] : Paths) {
      OS << "  " << printMBBReference(*P.src()) << " -> "
         << printMBBReference(*P.dst()) << ": ";
      PI.print(OS);
      OS << '\n';
    }
    OS << "}\n";
  }

  LLVM_DUMP_METHOD void dumpPaths() const { printPaths(dbgs()); }

  // Legacy alias kept for existing call sites.
  void dumpShortestPaths() const {
    for (const auto &P : Paths) {
      const MachineBasicBlock *From = P.first.src();
      const MachineBasicBlock *To = P.first.dst();
      std::optional<NextUseDistance> Dist = P.second.ShortestDistance;
      dbgs() << "From: " << printMBBReference(*From)
             << "-> To:" << printMBBReference(*To) << " = "
             << Dist.value_or(-1).fmt() << "\n";
    }
  }

  void printInterBlockDistances(raw_ostream &OS) const {
    using MBBPair = std::pair<unsigned, unsigned>;
    using Elem = std::pair<NextUseDistance, MBBPair>;
    std::vector<Elem> SortedDistances;

    for (const auto &[FromNum, Dsts] : InterBlockDistances) {
      for (const auto &[ToNum, Dist] : Dsts) {
        SortedDistances.emplace_back(Dist.Weighted, MBBPair(FromNum, ToNum));
      }
    }
    llvm::sort(SortedDistances, [](const auto &A, const auto &B) {
      if (A.first != B.first)
        return A.first < B.first;

      if (A.second.first != B.second.first)
        return A.second.first < B.second.first;

      return A.second.second < B.second.second;
    });

    OS << "\n--------- InterBlockDistances -------- {\n";
    for (const Elem &E : SortedDistances) {

      OS << "  bb." << E.second.first << " -> bb." << E.second.second << ": ";
      E.first.print(OS);
      OS << '\n';
    }
    OS << "}\n";
  }

  LLVM_DUMP_METHOD void dumpInterBlockDistances() const {
    printInterBlockDistances(dbgs());
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // LiveRegUse Caching - A cache of the distances for the last
  // MachineInstruction. When getting the distances for a MachineInstruction, if
  // it is the same basic block as the cached instruction, we can generally use
  // an offset from the cached values to compute the distances. There are some
  // exceptions - see 'cacheLiveRegUse'.
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
private:
  struct LiveRegToUseMapElem {
    LiveRegUse Use;
    bool MIDependent;
    LiveRegToUseMapElem() : Use(), MIDependent(false) {}
    LiveRegToUseMapElem(LiveRegUse U, bool MIDep)
        : Use(U), MIDependent(MIDep) {}

    void print(raw_ostream &OS) const {
      Use.print(OS);
      OS << (MIDependent ? " [mi-dep]" : " [mi-indep]");
    }

    LLVM_DUMP_METHOD void dump() const {
      print(dbgs());
      dbgs() << '\n';
    }
  };

  // Using std::map because LaneBitmask does not work out-of-the-box as a
  // DenseMap key and I did not see a performance benefit over std::map.
  using LaneBitmaskToUseMap = std::map<LaneBitmask, LiveRegToUseMapElem>;
  using LiveRegToUseMap = DenseMap<Register, LaneBitmaskToUseMap>;

  const MachineInstr *CachedDistancesMI = nullptr;
  LiveRegToUseMap CachedDistances;
  LiveRegToUseMap PendingCachedDistances;
  unsigned DistanceCacheHits = 0;
  unsigned DistanceCacheMisses = 0;

  void resetDistanceCache() {
    CachedDistancesMI = nullptr;
    CachedDistances.clear();
    DistanceCacheHits = 0;
    DistanceCacheMisses = 0;
  }

  void maybeClearCachedLiveRegUses(const MachineInstr &MI) {
    if (CachedDistancesMI &&
        (CachedDistancesMI->getParent() != MI.getParent() ||
         !instrsAreInOrder(CachedDistancesMI, &MI))) {
      CachedDistancesMI = nullptr;
      CachedDistances.clear();
    }
  }

  bool okToUseCacheElem(const LiveRegToUseMapElem &CacheElem,
                        const MachineInstr &MI, const InstrIdTy LastDelta) {
    if (!CacheElem.MIDependent)
      return true;

    const LiveRegUse &U = CacheElem.Use;

    // Never okay to produce a negative distance
    if (U.Dist < LastDelta)
      return false;

    const MachineInstr *UseMI = U.Use->getParent();

    // Always okay if use is in another basic block or UseMI is MI
    if (UseMI->getParent() != MI.getParent() || UseMI == &MI)
      return true;

    // If CachedDistancesMI <= Use < MI we could have a problem since we don't
    // know if Use is still reachable.
    return !instrsAreInOrder(CachedDistancesMI, UseMI) ||
           !instrsAreInOrder(UseMI, &MI);
  }

  std::pair<const LaneBitmaskToUseMap *, const LiveRegToUseMapElem *>
  findCachedLiveRegUse(Register Reg, LaneBitmask LaneMask,
                       const MachineInstr &MI, const InstrIdTy LastDelta) {
    if (!DistanceCacheEnabled)
      return {nullptr, nullptr};

    ++DistanceCacheMisses; // Assume miss
    auto I = CachedDistances.find(Reg);
    if (I == CachedDistances.end())
      return {nullptr, nullptr};
    const LaneBitmaskToUseMap &RegSlot = I->second;
    if (RegSlot.empty())
      return {nullptr, nullptr};

    auto J = RegSlot.find(LaneMask);
    if (J == RegSlot.end())
      return {nullptr, nullptr};

    const LiveRegToUseMapElem &MaskSlot = J->second;
    if (!okToUseCacheElem(MaskSlot, MI, LastDelta))
      return {nullptr, nullptr};

    --DistanceCacheMisses;
    ++DistanceCacheHits;
    return {&RegSlot, &MaskSlot};
  }

  void cacheLiveRegUse(const MachineInstr &MI, Register Reg, LaneBitmask Mask,
                       LiveRegUse U, bool MIDependent) {
    if (!DistanceCacheEnabled)
      return;

    auto I = PendingCachedDistances.try_emplace(Reg).first;
    LaneBitmaskToUseMap &RegSlot = I->second;
    RegSlot.try_emplace(Mask, U, MIDependent);
  }

  void updateCachedLiveRegUses(const MachineInstr &MI) {
    if (!DistanceCacheEnabled)
      return;

    CachedDistancesMI = &MI;
    CachedDistances = std::move(PendingCachedDistances);
    PendingCachedDistances.clear();
    LLVM_DEBUG(dumpDistanceCache());
  }

  void printDistanceCache(raw_ostream &OS) const {
    OS << "\n----------- Distance Cache ----------- {\n";
    OS << "  CachedAt: ";
    if (CachedDistancesMI)
      OS << *CachedDistancesMI;
    else
      OS << "<none>\n";

    constexpr size_t RegNameWidth = 20;
    for (const auto &[Reg, ByMask] : CachedDistances) {
      const TargetRegisterClass *RC = MRI->getRegClass(Reg);
      LaneBitmask AllLanes = MRI->getMaxLaneMaskForVReg(Reg);

      for (const auto &[Mask, Elem] : ByMask) {
        std::string RegName;
        raw_string_ostream KOS(RegName);
        if (Mask == AllLanes) {
          KOS << printReg(Reg);
        } else {
          SmallVector<unsigned> Indexes;
          TRI->getCoveringSubRegIndexes(RC, Mask, Indexes);
          if (Indexes.size() == 1)
            KOS << printReg(Reg, TRI, Indexes.front(), MRI);
          else
            KOS << printReg(Reg) << " mask=" << Mask.getAsInteger();
        }
        OS << "  " << left_justify(RegName, RegNameWidth) << " : ";
        Elem.print(OS);
        OS << '\n';
      }
    }
    OS << "  (hits=" << DistanceCacheHits << " misses=" << DistanceCacheMisses
       << ")\n";
    OS << "}\n";
  }

  LLVM_DUMP_METHOD void dumpDistanceCache() const {
    printDistanceCache(dbgs());
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Processing Live Reg Uses
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
private:
  // Decompose each use in 'Uses' by sub-reg and store the nearest one in
  // 'UseByMask'. Ignores subregs matching 'LiveRegLaneMask' - these are handled
  // as registers, not sub-regs.
  DenseMap<const TargetRegisterClass *, SmallVector<unsigned>>
      SubRegIndexesForRegClass;
  void collectSubRegUsesByMask(
      const SmallVectorImpl<const MachineOperand *> &Uses,
      const SmallVectorImpl<CacheableNextUseDistance> &Distances,
      LaneBitmask LiveRegLaneMask, LaneBitmaskToUseMap &UseByMask) {

    assert(Uses.size());
    assert(Uses.size() == Distances.size());

    const TargetRegisterClass *RC = MRI->getRegClass(Uses.front()->getReg());
    auto [SRI, Inserted] = SubRegIndexesForRegClass.try_emplace(RC);
    if (Inserted)
      TRI->getCoveringSubRegIndexes(RC, LaneBitmask::getAll(), SRI->second);
    const SmallVector<unsigned> &RCSubRegIndexes = SRI->second;

    unsigned OneIndex; // Backing store for 'Indexes' below when 1 index
    for (size_t I = 0; I < Uses.size(); ++I) {
      const MachineOperand *MO = Uses[I];
      auto [SubRegMIDep, Dist] = Distances[I];
      const LiveRegUse LRU{MO, Dist};

      ArrayRef<unsigned> Indexes;
      if (MO->getSubReg()) {
        OneIndex = MO->getSubReg();
        Indexes = ArrayRef(OneIndex);
      } else {
        Indexes = RCSubRegIndexes;
      }

      for (unsigned Idx : Indexes) {
        LaneBitmask Mask = TRI->getSubRegIndexLaneMask(Idx);
        if (Mask.all() || Mask == LiveRegLaneMask)
          continue;

        auto &[SlotU, SlotMIDep] = UseByMask[Mask];
        if (updateClosest(SlotU, LRU))
          SlotMIDep = SubRegMIDep;
      }
    }
  }

  // Similar to 'collectSubRegUsesByMask' above, but uses cached distances.
  void collectSubRegUsesByMaskFromCache(const LaneBitmaskToUseMap &CachedMap,
                                        LaneBitmask LiveRegLaneMask,
                                        const MachineInstr *MI,
                                        InstrIdTy LastDelta,
                                        LaneBitmaskToUseMap &UseByMask) {

    for (const auto &KV : CachedMap) {
      LaneBitmask SubregLaneMask = KV.first;
      if (SubregLaneMask.all() || SubregLaneMask == LiveRegLaneMask)
        continue;

      const LiveRegToUseMapElem &SubregE = KV.second;
      if (!okToUseCacheElem(SubregE, *MI, LastDelta))
        continue;

      const bool MIDep = SubregE.MIDependent;
      LiveRegUse U = SubregE.Use;
      if (MIDep)
        U.Dist -= LastDelta;

      auto &[SlotU, SlotMIDep] = UseByMask[SubregLaneMask];
      if (updateClosest(SlotU, U))
        SlotMIDep = MIDep;
    }
  }

  // Loops through 'UseByMask' finding the furthest sub-register and updating
  // 'FurthestSubreg' accordingly.
  void updateFurthestSubReg(
      const MachineInstr &MI, const LiveRegUse &U,
      const LaneBitmaskToUseMap &UseByMask,
      DenseMap<const MachineOperand *, UseDistancePair> *RelevantUses,
      LiveRegUse &FurthestSubreg) {

    if (UseByMask.empty()) {
      updateFurthest(FurthestSubreg, U);
      return;
    }

    for (const auto &KV : UseByMask) {
      const LiveRegUse &SubregU = KV.second.Use;
      const bool SubregMIDep = KV.second.MIDependent;

      if (RelevantUses)
        RelevantUses->try_emplace(SubregU.Use, SubregU);
      cacheLiveRegUse(MI, SubregU.Use->getReg(), KV.first, SubregU,
                      SubregMIDep);
      updateFurthest(FurthestSubreg, SubregU);
    }
  }

  // Used to populate 'MIDefs' to be passed to 'getNextUseDistances'.
  SmallSet<Register, 4> collectDefinedRegisters(const MachineInstr &MI) const {
    SmallSet<Register, 4> MIDefs;

    for (const MachineOperand &MO : MI.all_defs()) {
      if (MO.isReg() && MO.getReg().isValid() && hasAtLeastOneUse(MO.getReg()))
        MIDefs.insert(MO.getReg());
    }
    return MIDefs;
  }

  // Computes distances from 'MI' to each registers in 'LiveRegs'. Returns the
  // furthest register and (optionally) sub-register in 'Furthest' and
  // 'FurthestSubreg' respectively.
public:
  void getNextUseDistances(const GCNRPTracker::LiveRegSet &LiveRegs,
                           const MachineInstr &MI, LiveRegUse &Furthest,
                           LiveRegUse *FurthestSubreg = nullptr,
                           DenseMap<const MachineOperand *, UseDistancePair>
                               *RelevantUses = nullptr) {
    const SmallSet<Register, 4> MIDefs(collectDefinedRegisters(MI));

    SmallVector<const MachineOperand *> Uses;
    SmallVector<CacheableNextUseDistance> Distances;
    LaneBitmaskToUseMap UseByMask;

    maybeClearCachedLiveRegUses(MI);
    const InstrIdTy LastDelta =
        CachedDistancesMI ? getDistance(CachedDistancesMI, &MI) : 0;

    for (auto &KV : LiveRegs) {
      const Register Reg = KV.first;
      const LaneBitmask LaneMask = KV.second;

      if (MIDefs.contains(Reg))
        continue;

      Uses.clear();
      UseByMask.clear();

      LiveRegUse U;
      bool MIDependent = false;
      auto [CacheMap, CacheElem] =
          findCachedLiveRegUse(Reg, LaneMask, MI, LastDelta);
      if (CacheMap && CacheElem) {
        MIDependent = CacheElem->MIDependent;
        U = CacheElem->Use;
        if (MIDependent)
          U.Dist -= LastDelta;
      } else {
        getReachableUses(Reg, LaneMask, MI, Uses);
        if (Uses.empty())
          continue;

        const MachineOperand *NextUse = nullptr;
        NextUseDistance Dist = getShortestDistance(
            Reg, LaneMask, MI, Uses, &NextUse, &MIDependent, &Distances);
        U = LiveRegUse{NextUse, Dist};
      }

      if (RelevantUses)
        RelevantUses->try_emplace(U.Use, U);
      cacheLiveRegUse(MI, Reg, LaneMask, U, MIDependent);

      updateFurthest(Furthest, U);

      if (!FurthestSubreg)
        continue;

      if (CacheMap) {
        collectSubRegUsesByMaskFromCache(*CacheMap, LaneMask, &MI, LastDelta,
                                         UseByMask);
      } else {
        collectSubRegUsesByMask(Uses, Distances, LaneMask, UseByMask);
      }
      updateFurthestSubReg(MI, U, UseByMask, RelevantUses, *FurthestSubreg);
    }
    updateCachedLiveRegUses(MI);
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Helper methods for printAsJson
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
private:
  static format_object<unsigned> Fmt(unsigned Id) { return format("%u", Id); }

public:
  void printVerboseInstrFields(json::OStream &J, const MachineInstr &MI) const {
    J.attribute("id", getInstrId(&MI));
    J.attribute("head-len", getHeadLen(&MI));
    J.attribute("tail-len", getTailLen(&MI));
  }

  void printPaths(json::OStream &J, ModuleSlotTracker &MST) const {
    J.attributeBegin("paths");
    J.arrayBegin();
    for (const auto &KV : Paths) {
      const Path &P = KV.first;
      const PathInfo &PI = KV.second;

      J.objectBegin();

      printMBBNameAttr(J, "src", *P.src(), MST);
      printMBBNameAttr(J, "dst", *P.dst(), MST);

      if (PI.ShortestDistance.has_value()) {
        J.attribute("shortest-distance",
                    PI.ShortestDistance.value().toJsonValue());
      } else {
        J.attribute("shortest-distance", nullptr);
      }

      if (PI.ShortestUnweightedDistance.has_value()) {
        J.attribute("shortest-unweighted-distance",
                    PI.ShortestUnweightedDistance.value().toJsonValue());
      } else {
        J.attribute("shortest-unweighted-distance", nullptr);
      }

      J.attribute("edge-kind", static_cast<int>(PI.EK));
      J.attribute("reachable", PI.Reachable);
      J.attribute("forward-reachable", PI.ForwardReachable);

      J.objectEnd();
    }
    J.arrayEnd();
    J.attributeEnd();
  }

public:
  AMDGPUNextUseAnalysisImpl(const MachineFunction *, const MachineLoopInfo *);
  ~AMDGPUNextUseAnalysisImpl() { clearTables(); }

  AMDGPUNextUseAnalysis::Config getConfig() const { return Cfg; }
  void setConfig(AMDGPUNextUseAnalysis::Config NewCfg) {
    Cfg = NewCfg;
    clearTables();
    initializeTables();
  }

  unsigned getDistanceCacheHits() const { return DistanceCacheHits; }
  unsigned getDistanceCacheMisses() const { return DistanceCacheMisses; }

  void getReachableUses(Register LiveReg, LaneBitmask LaneMask,
                        const MachineInstr &MI,
                        SmallVector<const MachineOperand *> &Uses) const;

  /// \Returns the shortest next-use distance for \p LiveReg.
  NextUseDistance
  getShortestDistance(Register LiveReg, LaneBitmask LaneMask,
                      const MachineInstr &FromMI,
                      const SmallVector<const MachineOperand *> &Uses,
                      const MachineOperand **ShortestUseOut, bool *MIDependent,
                      SmallVector<CacheableNextUseDistance> *Distances) const;

  NextUseDistance
  getShortestDistance(Register LiveReg, const MachineInstr &FromMI,
                      const SmallVector<const MachineOperand *> &Uses) const {
    return getShortestDistance(LiveReg, LaneBitmask::getAll(), FromMI, Uses,
                               nullptr, nullptr, nullptr);
  }
};

AMDGPUNextUseAnalysisImpl::AMDGPUNextUseAnalysisImpl(
    const MachineFunction *MF, const MachineLoopInfo *ML) {

  this->MF = MF;
  this->MLI = ML;

  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF->getRegInfo();

  // FIXME: Hopefully we will soon converge on a single way of calculating
  // next-use distance and remove these presets.
  if (ConfigPresetOpt == "compute")
    Cfg = AMDGPUNextUseAnalysis::Config::Compute();
  else
    Cfg = AMDGPUNextUseAnalysis::Config::Graphics();

  if (ConfigCountPhisOpt.getNumOccurrences())
    Cfg.CountPhis = ConfigCountPhisOpt;
  if (ConfigForwardOnlyOpt.getNumOccurrences())
    Cfg.ForwardOnly = ConfigForwardOnlyOpt;
  if (ConfigPreciseUseModelingOpt.getNumOccurrences())
    Cfg.PreciseUseModeling = ConfigPreciseUseModelingOpt;
  if (ConfigPromoteToPreheaderOpt.getNumOccurrences())
    Cfg.PromoteToPreheader = ConfigPromoteToPreheaderOpt;

  initializeTables();
}

NextUseDistance AMDGPUNextUseAnalysisImpl::getShortestDistance(
    Register LiveReg, LaneBitmask LaneMask, const MachineInstr &CurMI,
    const SmallVector<const MachineOperand *> &Uses,
    const MachineOperand **ShortestUseOut, bool *CurMIDependentOut,
    SmallVector<CacheableNextUseDistance> *Distances) const {

  assert(!LiveReg.isPhysical() && !TRI->isAGPR(*MRI, LiveReg) &&
         "Next-use distance is calculated for SGPRs and VGPRs");
  const MachineOperand *NextUse = nullptr;
  auto NextUseDist = NextUseDistance::unreachable();
  bool CurMIDependent = false;

  if (Distances) {
    Distances->clear();
    Distances->reserve(Uses.size());
  }
  for (auto *UseMO : Uses) {
    auto [Dep, D] = calcDistanceToUse(LiveReg, LaneMask, CurMI, UseMO);

    if (D < NextUseDist) {
      NextUseDist = D;
      NextUse = UseMO;
      CurMIDependent = Dep;
    }

    if (Distances)
      Distances->push_back({Dep, D});
  }
  if (ShortestUseOut)
    *ShortestUseOut = NextUse;
  if (CurMIDependentOut)
    *CurMIDependentOut = CurMIDependent;

  assert(NextUseDist.isReachable() &&
         "getShortestDistance called with no reachable uses");
  return NextUseDist;
}

void AMDGPUNextUseAnalysisImpl::getReachableUses(
    Register Reg, LaneBitmask LaneMask, const MachineInstr &MI,
    SmallVector<const MachineOperand *> &Uses) const {
  const bool CheckMask = LaneMask != LaneBitmask::getAll() &&
                         LaneMask != MRI->getMaxLaneMaskForVReg(Reg);
  const MachineBasicBlock *MBB = MI.getParent();

  for (const MachineOperand *UseMO : getRegisterUses(Reg)) {
    const MachineInstr *UseMI = UseMO->getParent();
    const MachineBasicBlock *UseMBB = UseMI->getParent();

    if (CheckMask && !machineOperandCoveredBy(*UseMO, LaneMask))
      continue;

    bool Reachable;
    if (Cfg.PreciseUseModeling)
      Reachable = isUseReachablePrecise(MI, MBB, UseMO, UseMI, UseMBB);
    else if (MBB == UseMBB)
      Reachable = instrsAreInOrder(&MI, UseMI);
    else
      Reachable = isForwardReachable(MBB, UseMBB);

    if (Reachable)
      Uses.push_back(UseMO);
  }
}

//==============================================================================
// AMDGPUNextUseAnalysis
//==============================================================================
AMDGPUNextUseAnalysis::AMDGPUNextUseAnalysis(const MachineFunction *MF,
                                             const MachineLoopInfo *MLI) {
  Impl = std::make_unique<AMDGPUNextUseAnalysisImpl>(MF, MLI);
}
AMDGPUNextUseAnalysis::AMDGPUNextUseAnalysis(AMDGPUNextUseAnalysis &&Other)
    : Impl(std::move(Other.Impl)) {}
AMDGPUNextUseAnalysis::~AMDGPUNextUseAnalysis() {}

AMDGPUNextUseAnalysis &
AMDGPUNextUseAnalysis::operator=(AMDGPUNextUseAnalysis &&Other) {
  if (this != &Other)
    Impl = std::move(Other.Impl);
  return *this;
}

AMDGPUNextUseAnalysis::Config AMDGPUNextUseAnalysis::getConfig() const {
  return Impl->getConfig();
}

void AMDGPUNextUseAnalysis::setConfig(Config Cfg) { Impl->setConfig(Cfg); }

/// \Returns the next-use distance for \p LiveReg.
NextUseDistance AMDGPUNextUseAnalysis::getShortestDistance(
    Register LiveReg, const MachineInstr &FromMI,
    const SmallVector<const MachineOperand *> &Uses,
    const MachineOperand **ShortestUseOut,
    SmallVector<NextUseDistance> *DistancesOut) const {

  SmallVector<AMDGPUNextUseAnalysisImpl::CacheableNextUseDistance> Distances;
  auto Dist = Impl->getShortestDistance(LiveReg, LaneBitmask::getAll(), FromMI,
                                        Uses, ShortestUseOut, nullptr,
                                        DistancesOut ? &Distances : nullptr);
  if (DistancesOut) {
    for (auto [MIDep, D] : Distances)
      DistancesOut->push_back(D);
  }
  return Dist;
}

void AMDGPUNextUseAnalysis::getNextUseDistances(
    const DenseMap<unsigned, LaneBitmask> &LiveRegs, const MachineInstr &MI,
    UseDistancePair &FurthestOut, UseDistancePair *FurthestSubregOut,
    DenseMap<const MachineOperand *, UseDistancePair> *RelevantUses) const {

  LiveRegUse Furthest;
  LiveRegUse FurthestSubreg;
  Impl->getNextUseDistances(LiveRegs, MI, Furthest,
                            FurthestSubregOut ? &FurthestSubreg : nullptr,
                            RelevantUses);
  FurthestOut = Furthest;
  if (FurthestSubregOut)
    *FurthestSubregOut = FurthestSubreg;
}
void AMDGPUNextUseAnalysis::getReachableUses(
    Register LiveReg, LaneBitmask LaneMask, const MachineInstr &MI,
    SmallVector<const MachineOperand *> &Uses) const {
  return Impl->getReachableUses(LiveReg, LaneMask, MI, Uses);
}

//==============================================================================
// AMDGPUNextUseAnalysisLegacyPass
//==============================================================================

//------------------------------------------------------------------------------
// Legacy Analysis Pass
//------------------------------------------------------------------------------
AMDGPUNextUseAnalysisLegacyPass::AMDGPUNextUseAnalysisLegacyPass()
    : MachineFunctionPass(ID) {}
StringRef AMDGPUNextUseAnalysisLegacyPass::getPassName() const {
  return "Next Use Analysis";
}

bool AMDGPUNextUseAnalysisLegacyPass::runOnMachineFunction(
    MachineFunction &MF) {
  const MachineLoopInfo *MLI =
      &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  NUA.reset(new AMDGPUNextUseAnalysis(&MF, MLI));
  return false;
}

void AMDGPUNextUseAnalysisLegacyPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

char AMDGPUNextUseAnalysisLegacyPass::ID = 0;
char &llvm::AMDGPUNextUseAnalysisLegacyID = AMDGPUNextUseAnalysisLegacyPass::ID;

INITIALIZE_PASS_BEGIN(AMDGPUNextUseAnalysisLegacyPass, DEBUG_TYPE,
                      "Next Use Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUNextUseAnalysisLegacyPass, DEBUG_TYPE,
                    "Next Use Analysis", false, true)

FunctionPass *llvm::createAMDGPUNextUseAnalysisLegacyPass() {
  return new AMDGPUNextUseAnalysisLegacyPass();
}

//------------------------------------------------------------------------------
// New Pass Manager Analysis Pass
//------------------------------------------------------------------------------
AnalysisKey AMDGPUNextUseAnalysisPass::Key;

AMDGPUNextUseAnalysisPass::Result
AMDGPUNextUseAnalysisPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  const MachineLoopInfo &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
  return AMDGPUNextUseAnalysis(&MF, &MLI);
}

//==============================================================================
// AMDGPUNextUseAnalysisPrinterLegacyPass
//==============================================================================
namespace {
void printInstrMember(json::OStream &J, ModuleSlotTracker &MST,
                      const MachineInstr &MI,
                      const AMDGPUNextUseAnalysisImpl &NUA) {
  printStringAttr(J, "instr", MI, MST);
  if (DumpNextUseDistanceVerbose)
    NUA.printVerboseInstrFields(J, MI);
}

void printDistances(
    json::OStream &J, const MachineRegisterInfo &MRI, const SIRegisterInfo &TRI,
    ModuleSlotTracker &MST,
    const DenseMap<const MachineOperand *, UseDistancePair> &Uses) {
  if (!DumpNextUseDistanceVerbose)
    return;

  // Sorting isn't necessary for the purposes of JSON, but it reduces
  // FileCheck differences.
  SmallVector<const MachineOperand *> Keys;
  for (const MachineOperand *K : Uses.keys())
    Keys.push_back(K);
  llvm::sort(Keys, [](const auto &A, const auto &B) {
    return A->getReg() < B->getReg() ||
           (A->getReg() == B->getReg() && A->getSubReg() < B->getSubReg());
  });

  J.attributeBegin("distances");
  J.objectBegin();

  for (const MachineOperand *K : Keys) {
    const LiveRegUse U = Uses.at(K);
    printAttr(J, printReg(U.getReg(), &TRI, U.getSubReg(), &MRI),
              U.Dist.toJsonValue());
  }

  J.objectEnd();
  J.attributeEnd();
}

void printFurthestUse(json::OStream &J, const MachineRegisterInfo &MRI,
                      const SIRegisterInfo &TRI, ModuleSlotTracker &MST,
                      const LiveRegUse F, bool Subreg = false) {
  J.attributeBegin(Subreg ? "furthest-subreg" : "furthest");
  J.objectBegin();

  if (F.Use) {
    printStringAttr(
        J, "register",
        printReg(F.getReg(), &TRI, Subreg ? F.getSubReg() : 0, &MRI));

    if (DumpNextUseDistanceVerbose) {
      printStringAttr(J, "use", [&](raw_ostream &OS) { OS << (*F.Use); });
      printStringAttr(J, "use-mi", *F.Use->getParent(), MST);
    }
    J.attribute("distance", F.Dist.toJsonValue());
  }

  J.objectEnd();
  J.attributeEnd();
}

void printDistanceFromDefToUse(json::OStream &J, const MachineFunction &MF,
                               const AMDGPUNextUseAnalysis &NUA,
                               const SIRegisterInfo &TRI,
                               const MachineRegisterInfo &MRI) {
  auto getRegNextUseDistance = [&](Register DefReg) {
    const MachineInstr &DefMI = *MRI.def_instr_begin(DefReg);

    SmallVector<const MachineOperand *> Uses;
    NUA.getReachableUses(DefReg, LaneBitmask::getAll(), DefMI, Uses);
    if (Uses.empty())
      return NextUseDistance::unreachable();
    return NUA.getShortestDistance(DefReg, DefMI, Uses);
  };

  J.attributeBegin("distance-from-def-to-closest-use");
  J.objectBegin();

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.all_defs()) {
        Register Reg = MO.getReg();
        if (Reg.isPhysical())
          continue;
        NextUseDistance D = getRegNextUseDistance(Reg);
        printAttr(J, printReg(Reg, &TRI, 0, &MRI), D.toJsonValue());
      }
    }
  }

  J.objectEnd();
  J.attributeEnd();
}

void printNextUseDistancesAsJson(json::OStream &J, const MachineFunction &MF,
                                 const AMDGPUNextUseAnalysis &NUA,
                                 const AMDGPUNextUseAnalysisImpl &NUAImpl,
                                 const LiveIntervals &LIS) {
  using UseDistancePair = AMDGPUNextUseAnalysis::UseDistancePair;
  const Function &F = MF.getFunction();
  const Module *M = F.getParent();

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  // We don't actually care about register pressure here - just using
  // GCNDownwardRPTracker as a convenient way of getting the set of live
  // registers at a given instruction.
  GCNDownwardRPTracker RPTracker(LIS);
  ModuleSlotTracker MST(M);
  MST.incorporateFunction(F);

  DenseMap<const MachineOperand *, UseDistancePair> RelevantUses;

  J.attributeBegin("furthest-distances");
  J.objectBegin();

  for (const MachineBasicBlock &MBB : MF) {
    std::string BBName;
    raw_string_ostream BBOS(BBName);
    MBB.printName(BBOS, MachineBasicBlock::PrintNameIr, &MST);

    J.attributeBegin(BBOS.str());
    J.arrayBegin();

    const MachineInstr *PrevMI = nullptr;
    for (const MachineInstr &MI : MBB) {
      // Update register pressure tracker
      if (!PrevMI || PrevMI->getOpcode() == AMDGPU::PHI)
        RPTracker.reset(MI, MBB.end());
      RPTracker.advance();

      UseDistancePair Furthest;
      UseDistancePair FurthestSubreg;
      RelevantUses.clear();
      NUA.getNextUseDistances(RPTracker.getLiveRegs(), MI, Furthest,
                              &FurthestSubreg, &RelevantUses);

      J.objectBegin();
      printInstrMember(J, MST, MI, NUAImpl);
      printDistances(J, MRI, TRI, MST, RelevantUses);
      printFurthestUse(J, MRI, TRI, MST, Furthest);
      printFurthestUse(J, MRI, TRI, MST, FurthestSubreg, /*Subreg*/ true);
      J.objectEnd();

      PrevMI = &MI;
    }

    J.arrayEnd();
    J.attributeEnd();
  }

  J.objectEnd();
  J.attributeEnd();

  if (DumpNextUseDistanceVerbose || DumpNextUseDistanceDefToUse)
    printDistanceFromDefToUse(J, MF, NUA, TRI, MRI);

  if (DumpNextUseDistanceVerbose)
    NUAImpl.printPaths(J, MST);

  if (DistanceCacheEnabled) {
    J.attributeBegin("metrics");
    J.objectBegin();
    {
      J.attributeBegin("distance-cache");
      J.objectBegin();
      {
        J.attribute("hits", NUAImpl.getDistanceCacheHits());
        J.attribute("misses", NUAImpl.getDistanceCacheMisses());
      }
      J.objectEnd();
      J.attributeEnd(); // distance-cache
    }
    J.objectEnd();
    J.attributeEnd(); // metrics
  }
}

void printAsJson(raw_ostream &FallbackOS, TimerGroup &JsonTimerGroup,
                 Timer &JsonTimer, const MachineFunction &MF,
                 const AMDGPUNextUseAnalysis &NUA,
                 const AMDGPUNextUseAnalysisImpl &NUAImpl,
                 const LiveIntervals &LIS) {
  std::string FN = DumpNextUseDistanceAsJson;

  auto dump = [&](raw_ostream &OS) {
    json::OStream J(OS, 2);
    J.objectBegin();

    J.attributeBegin("next-use-analysis");
    J.objectBegin();
    printNextUseDistancesAsJson(J, MF, NUA, NUAImpl, LIS);
    J.objectEnd();
    J.attributeEnd();

    JsonTimer.stopTimer();
    JsonTimerGroup.printJSONValues(OS, ",\n");

    J.objectEnd();
  };

  if (!DumpNextUseDistanceAsJson.getNumOccurrences()) {
    dump(FallbackOS);
  } else if (FN.empty() || FN == "-") {
    dump(outs());
  } else {
    std::error_code EC;
    ToolOutputFile OutF(FN, EC, sys::fs::OF_None);
    dump(OutF.os());
    OutF.keep();
  }
}
} // namespace

//------------------------------------------------------------------------------
// Legacy Printer Pass
//------------------------------------------------------------------------------
AMDGPUNextUseAnalysisPrinterLegacyPass::AMDGPUNextUseAnalysisPrinterLegacyPass()
    : MachineFunctionPass(ID) {}

StringRef AMDGPUNextUseAnalysisPrinterLegacyPass::getPassName() const {
  return "AMDGPU Next Use Analysis Printer";
}

bool AMDGPUNextUseAnalysisPrinterLegacyPass::runOnMachineFunction(
    MachineFunction &MF) {
  TimerGroup JsonTimerGroup("amdgpu-next-use-analysis-json",
                            "AMDGPU Next Use Analysis JSON Printer", false);
  Timer JsonTimer("json", "Total time spent generating json", JsonTimerGroup);
  JsonTimer.startTimer();

  const LiveIntervals &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  const AMDGPUNextUseAnalysis &NUA =
      getAnalysis<AMDGPUNextUseAnalysisLegacyPass>().getNextUseAnalysis();

  printAsJson(errs(), JsonTimerGroup, JsonTimer, MF, NUA, *NUA.Impl, LIS);

  return false;
}

void AMDGPUNextUseAnalysisPrinterLegacyPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addRequired<LiveIntervalsWrapperPass>();
  AU.addRequired<AMDGPUNextUseAnalysisLegacyPass>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

char AMDGPUNextUseAnalysisPrinterLegacyPass::ID = 0;
char &AMDGPUNextUseAnalysisPrinterLegacyID =
    AMDGPUNextUseAnalysisPrinterLegacyPass::ID;

INITIALIZE_PASS_BEGIN(AMDGPUNextUseAnalysisPrinterLegacyPass,
                      "amdgpu-next-use-printer",
                      "AMDGPU Next Use Analysis Printer", false, false)

INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)

INITIALIZE_PASS_END(AMDGPUNextUseAnalysisPrinterLegacyPass,
                    "amdgpu-next-use-printer",
                    "AMDGPU Next Use Analysis Printer", false, false)

FunctionPass *llvm::createAMDGPUNextUseAnalysisPrinterLegacyPass() {
  return new AMDGPUNextUseAnalysisPrinterLegacyPass();
}

//------------------------------------------------------------------------------
// New Pass Manager Printer Pass
//------------------------------------------------------------------------------
PreservedAnalyses
AMDGPUNextUseAnalysisPrinterPass::run(MachineFunction &MF,
                                      MachineFunctionAnalysisManager &MFAM) {

  TimerGroup JsonTimerGroup("amdgpu-next-use-analysis-json",
                            "AMDGPU Next Use Analysis JSON Printer", false);
  Timer JsonTimer("json", "Total time spent generating json", JsonTimerGroup);
  JsonTimer.startTimer();

  const LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  const AMDGPUNextUseAnalysis &NUA =
      MFAM.getResult<AMDGPUNextUseAnalysisPass>(MF);

  printAsJson(OS, JsonTimerGroup, JsonTimer, MF, NUA, *NUA.Impl, LIS);

  return PreservedAnalyses::all();
}
