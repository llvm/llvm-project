//===- AMDGPUHotBlockRematerialize.cpp - AMDGPU Hot Block Rematerialize ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief AMDGPU hot block rematerialization. Reduces SGPR pressure at hot
/// blocks by sinking (moving) or cloning the defs of long-lived scalar values
/// closer to their uses, so the register allocator can reach a higher
/// occupancy tier. Runs pre-RA on SSA MachineIR.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUMIRUtils.h"
#include "AMDGPUOccupancyAndLatencyHelper.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <vector>

#define DEBUG_TYPE "amdgpu-hot-block-remat"

using namespace llvm;

static cl::opt<bool> EnableAggressive(
    "amdgpu-remat-aggressive",
    cl::desc("Aggressive SGPR hot block rematerialization: shared sources are "
             "treated as free and all candidates are collected."),
    cl::init(false), cl::Hidden);

static cl::opt<int> DebugMBBIndex(
    "amdgpu-remat-debug-mbb",
    cl::desc("If non-negative, restrict hot block remat debug dumps to the "
             "block with this number."),
    cl::init(-1), cl::Hidden);

namespace {

using InstSet = DenseSet<MachineInstr *>;
using BlockSet = SmallPtrSet<MachineBasicBlock *, 8>;
template <typename T> using BlockMap = MapVector<MachineBasicBlock *, T>;

enum class RematKind {
  Candidate, // Not yet classified; holds a gain estimate only.
  Sink,      // Single-use: physically move DefMI to InsertBlock.
  Clone,     // Multi-use: copy DefMI into each user's block.
};

struct RematNode {
  Register Reg;
  MachineInstr *DefMI = nullptr;
  MachineBasicBlock *InsertBlock = nullptr;
  MachineInstr *InsertPointMI = nullptr;
  RematKind Kind = RematKind::Candidate;
  unsigned GainInDwords = 0;

  RematNode() = default;
  RematNode(Register R, MachineInstr *MI, unsigned Gain)
      : Reg(R), DefMI(MI), GainInDwords(Gain) {}
};

struct BlockLiveInfo {
  MachineBasicBlock *BB;
  unsigned MaxSPressure;
  unsigned MaxVPressure;
  // Cross-block virtual registers live at block entry.
  GCNRPTracker::LiveRegSet InputLive;
};

struct RematStatus {
  unsigned TargetOcc;
  unsigned TargetSLimit;
  unsigned TargetVLimit;
  unsigned MaxSPressure;
  unsigned MaxVPressure;
  unsigned InputPhysicalSPressure;
  unsigned InputPhysicalVPressure;
  bool MemBound;
  bool SGPRBound;
  DenseMap<MachineBasicBlock *, GCNRegPressure> MBBPressureMap;
  DenseMap<MachineBasicBlock *, GCNRPTracker::LiveRegSet> MBBInputLiveMap;
  DenseMap<MachineBasicBlock *, GCNRPTracker::LiveRegSet> MBBOutputLiveMap;
  DenseSet<MachineBasicBlock *> MemWriteMBBSet;
};

struct CandidateEntry {
  RematNode Node;
  int GainInDwords;
  bool IsSink;
};

class AMDGPUHotBlockRematerialize : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUHotBlockRematerialize() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "AMDGPU rematerialize"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachinePostDominatorTreeWrapperPass>();
    AU.addRequired<SlotIndexesWrapperPass>();
    AU.addRequired<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  bool hotBlockRemat(MachineFunction &MF, MachineLoopInfo *MLI,
                     LiveIntervals *LIS, MachineDominatorTree *DT,
                     MachinePostDominatorTree *PDT, bool &IsNearTarget);

  void applyRemat(MapVector<Register, RematNode> &RematMap,
                  std::vector<BlockLiveInfo> &HotBlocks,
                  MachineDominatorTree *DT, SlotIndexes *SlotIndexes,
                  MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                  const SIInstrInfo *SIII, LiveIntervals *LIS,
                  MachineFunction &MF);
  void applyCloneRemat(RematNode &Node, std::vector<BlockLiveInfo> &HotBlocks,
                       MachineDominatorTree *DT, MachineRegisterInfo &MRI,
                       SlotIndexes *SlotIndexes, const SIRegisterInfo *SIRI,
                       const SIInstrInfo *SIII, LiveIntervals *LIS,
                       MachineFunction &MF);
};

// A heuristic bias (in dwords) kept between the pass's SGPR budget and the hard
// limit, to absorb allocation losses. Defaults to the subtarget granule.
static unsigned getSgprBias(const GCNSubtarget *ST) {
  return ST->getSGPRAllocGranule();
}

//===----------------------------------------------------------------------===//
// Small shared helpers.
//===----------------------------------------------------------------------===//

static MachineBasicBlock::iterator
adjustInsertPointToAvoidSccSmash(MachineInstr *InstructionToMove,
                                 MachineBasicBlock *MBB,
                                 MachineBasicBlock::iterator CurrentInsertPoint,
                                 MachineRegisterInfo &MRI,
                                 const SIRegisterInfo *SIRI,
                                 const SIInstrInfo *SIII, LiveIntervals *LIS) {
  const bool WillSmashScc =
      InstructionToMove->modifiesRegister(AMDGPU::SCC, SIRI);
  if (WillSmashScc)
    CurrentInsertPoint = findOrCreateInsertionPointForSccDef(
        MBB, CurrentInsertPoint, SIRI, SIII, &MRI, LIS);
  return CurrentInsertPoint;
}

static bool isImplicitDefUse(MachineInstr *DefMI, MachineInstr *UseMI) {
  if (DefMI->getDesc().NumImplicitDefs == 0)
    return false;
  const TargetRegisterInfo *TRI =
      DefMI->getMF()->getSubtarget().getRegisterInfo();
  for (const MachineOperand &MO : DefMI->implicit_operands()) {
    if (!MO.isReg() || !MO.isDef())
      continue;
    if (UseMI->readsRegister(MO.getReg(), TRI))
      return true;
  }
  return false;
}

static bool isUsedByPhi(MachineInstr *DefMI, MachineRegisterInfo &MRI) {
  for (const MachineOperand &Def : DefMI->defs()) {
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Def.getReg())) {
      if (UseMI.isPHI())
        return true;
    }
  }
  return false;
}

static bool isSafeToMoveOrClone(MachineInstr *DefMI, MachineRegisterInfo &MRI) {
  if (isUsedByPhi(DefMI, MRI))
    return false;

  const TargetRegisterInfo *TRI =
      DefMI->getMF()->getSubtarget().getRegisterInfo();
  for (const MachineOperand &Op : DefMI->operands()) {
    if (!Op.isReg())
      continue;
    Register Reg = Op.getReg();
    if (Reg.isPhysical())
      continue;
    if (!MRI.getUniqueVRegDef(Reg) && !isConsistentMultiDef(Reg, MRI, TRI))
      return false;
  }
  return true;
}

// Net SGPR pressure reduction (in dwords) from moving/cloning DefMI: the size
// of the produced value minus the sizes of the source operands it drags along
// (unless those are shared, in aggressive mode). Returns 0 when unsafe.
static int computeGainInDwords(MachineInstr *DefMI, Register Reg,
                               const MachineRegisterInfo &MRI,
                               const SIRegisterInfo *SIRI) {
  int RematSize = SIRI->getRegSizeInBits(*MRI.getRegClass(Reg));
  for (MachineOperand &MO : DefMI->operands()) {
    if (MO.isImm())
      continue;
    if (!MO.isReg())
      continue;
    if (MO.isDef())
      continue;
    if (MO.isTied())
      continue;
    if (MO.getReg() == AMDGPU::EXEC)
      continue;
    // Do not move users of VCC or physical registers.
    if (MO.getReg() == AMDGPU::VCC || MO.getReg() == AMDGPU::VCC_LO)
      return 0;
    Register SrcReg = MO.getReg();
    if (SrcReg.isPhysical())
      return 0;
    // Only scalar remat is supported; mixing register files is not.
    if (SIRI->isVGPR(MRI, SrcReg))
      return 0;

    bool IsSingleDef = MRI.hasOneDef(SrcReg);
    if (!IsSingleDef)
      IsSingleDef = isConsistentMultiDef(
          SrcReg, MRI, MRI.getTargetRegisterInfo());
    if (!IsSingleDef)
      return 0;

    // In aggressive mode a multi-use source is treated as shared and its cost
    // is not charged against this candidate's gain.
    if (EnableAggressive && !MRI.hasOneNonDBGUse(SrcReg))
      continue;

    const TargetRegisterClass *OpRC = MRI.getRegClass(SrcReg);
    if (unsigned SubIdx = MO.getSubReg())
      OpRC = SIRI->getSubRegisterClass(OpRC, SubIdx);
    int InputSize = SIRI->getRegSizeInBits(*OpRC);
    if (RematSize > InputSize) {
      RematSize -= InputSize;
      continue;
    }
    return 0;
  }
  return RematSize;
}

//===----------------------------------------------------------------------===//
// Function-level pressure profiling.
//===----------------------------------------------------------------------===//

static unsigned collectMBBPressure(MachineBasicBlock &MBB, LiveIntervals *LIS,
                                   const GCNSubtarget *ST, unsigned DynVGPR,
                                   unsigned &MaxVPressure, unsigned &MaxSPressure,
                                   RematStatus &Status) {
  if (MBB.getFirstNonDebugInstr() == MBB.end())
    return ST->getOccupancyWithNumVGPRs(0, DynVGPR);

  auto BBEnd = MBB.rbegin();
  GCNUpwardRPTracker RPTracker(*LIS);
  if (!getNonDebugMBBEnd(BBEnd, MBB))
    return ST->getOccupancyWithNumVGPRs(0, DynVGPR);

  GCNRPTracker::LiveRegSet OutputLive = Status.MBBOutputLiveMap[&MBB];
  RPTracker.reset(MBB.getParent()->getRegInfo(), OutputLive);

  for (auto I = MBB.rbegin(), B = MBB.rend(); I != B; ++I) {
    MachineInstr &MI = *I;
    RPTracker.recede(MI);
    // Record blocks that write memory so loads are not sunk across them.
    // Control-flow terminators (S_BRANCH, S_CBRANCH_EXECZ) and S_ENDPGM are
    // barriers but not memory writes, so gate on mayStore only.
    if (MI.mayStore())
      Status.MemWriteMBBSet.insert(&MBB);
  }

  GCNRegPressure RP = RPTracker.getMaxPressureAndReset();
  unsigned SPressure = RP.getMaxSGPR();
  if (SPressure > MaxSPressure)
    MaxSPressure = SPressure;
  if (RP.getVGPRNum(ST->hasGFX90AInsts()) > MaxVPressure)
    MaxVPressure = RP.getVGPRNum(ST->hasGFX90AInsts());
  Status.MBBPressureMap[&MBB] = RP;
  return RP.getOccupancy(*ST, DynVGPR);
}

static unsigned collectFnPressure(MachineFunction &MF, LiveIntervals *LIS,
                                  const MachineRegisterInfo &MRI,
                                  const GCNSubtarget *ST, unsigned DynVGPR,
                                  unsigned &MaxVPressure, unsigned &MaxSPressure,
                                  RematStatus &Status) {
  unsigned TgtOcc = ST->getOccupancyWithWorkGroupSizes(MF).second;

  if (MF.size() > 1) {
    auto *SlotIndexes = LIS->getSlotIndexes();
    DenseMap<MachineBasicBlock *, SlotIndex> MBBInputSlotMap;
    DenseMap<MachineBasicBlock *, SlotIndex> MBBOutputSlotMap;
    for (MachineBasicBlock &MBB : MF) {
      auto BBBegin = MBB.getFirstNonDebugInstr();
      if (BBBegin != MBB.end())
        MBBInputSlotMap[&MBB] = SlotIndexes->getInstructionIndex(*BBBegin);

      auto BBEnd = MBB.rbegin();
      if (getNonDebugMBBEnd(BBEnd, MBB))
        MBBOutputSlotMap[&MBB] = SlotIndexes->getInstructionIndex(*BBEnd);
    }

    for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
      Register Reg = Register::index2VirtReg(I);
      if (!LIS->hasInterval(Reg))
        continue;
      const LiveInterval &LI = LIS->getInterval(Reg);
      // Local intervals cannot contribute to cross-block pressure.
      if (LIS->intervalIsInOneMBB(LI))
        continue;

      for (auto &InputIt : MBBInputSlotMap) {
        auto LiveMask = getLiveLaneMask(Reg, InputIt.second, *LIS, MRI);
        if (LiveMask.any())
          Status.MBBInputLiveMap[InputIt.first][Reg] |= LiveMask;
      }
      for (auto &OutputIt : MBBOutputSlotMap) {
        auto LiveMask = getLiveLaneMask(Reg, OutputIt.second, *LIS, MRI);
        if (LiveMask.any())
          Status.MBBOutputLiveMap[OutputIt.first][Reg] |= LiveMask;
      }
    }
  }

  for (MachineBasicBlock &MBB : MF) {
    unsigned Occ = collectMBBPressure(MBB, LIS, ST, DynVGPR, MaxVPressure,
                                      MaxSPressure, Status);
    if (TgtOcc > Occ)
      TgtOcc = Occ;
  }
  return TgtOcc;
}

// Physical SGPR live-in footprint, rounded up to the allocator granule. This
// models allocation granularity (the allocator cannot pack unrelated values
// into a partially-used granule); it is not spill alignment.
static unsigned computePhysicalSInputPressure(const MachineRegisterInfo &MRI,
                                              const SIRegisterInfo *SIRI,
                                              const GCNSubtarget *ST) {
  unsigned Count = 0;
  for (const auto &Livein : MRI.liveins()) {
    Register Reg = Livein.first;
    if (SIRI->isVGPR(MRI, Reg))
      continue;
    const TargetRegisterClass *RC = SIRI->getRegClassForReg(MRI, Reg);
    Count += RC->getLaneMask().getNumLanes();
  }
  unsigned Granule = ST->getSGPRAllocGranule();
  if (Granule == 0)
    return Count;
  return alignTo(Count, Granule);
}

static bool nearSgprSpill(unsigned MaxSPressure, const GCNSubtarget *ST,
                          MachineFunction &MF) {
  unsigned MaxSGPR = ST->getAddressableNumSGPRs();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  if (MFI->getScratchRSrcReg())
    MaxSGPR -= 4;
  MaxSGPR -= ST->getSGPRAllocGranule();
  return MaxSPressure > MaxSGPR;
}

static RematStatus getRematStatus(MachineFunction &MF, MachineLoopInfo *MLI,
                                  LiveIntervals *LIS,
                                  const MachineRegisterInfo &MRI,
                                  const GCNSubtarget *ST) {
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  unsigned DynVGPR = MFI->getDynamicVGPRBlockSize();
  const SIRegisterInfo *SIRI = ST->getRegisterInfo();

  unsigned MaxSPressure = 0;
  unsigned MaxVPressure = 0;
  RematStatus Status;
  unsigned TgtOcc = collectFnPressure(MF, LIS, MRI, ST, DynVGPR, MaxVPressure,
                                      MaxSPressure, Status);

  const unsigned MaxOcc = ST->getWavesPerEU(MF.getFunction()).second;
  if (TgtOcc >= MaxOcc) {
    Status.TargetOcc = TgtOcc;
    Status.TargetVLimit = 0;
    Status.TargetSLimit = 0;
    Status.MaxVPressure = 0;
    Status.MaxSPressure = 0;
    Status.InputPhysicalVPressure = 0;
    Status.InputPhysicalSPressure = 0;
    Status.MemBound = false;
    Status.SGPRBound = false;
    return Status;
  }

  MaxSPressure += RegForVCC;
  MaxVPressure = std::min(MaxVPressure, ST->getMaxNumVGPRs(MF));
  unsigned STgtOcc = ST->getOccupancyWithNumSGPRs(MaxSPressure);
  unsigned VTgtOcc = ST->getOccupancyWithNumVGPRs(MaxVPressure, DynVGPR);

  SchedScore TotalScore = collectLatency(MF, *ST, MLI);
  bool MemBound =
      TotalScore.isMemBound(TgtOcc, std::max(STgtOcc, VTgtOcc) - TgtOcc);

  bool SGPRBound = false;
  const unsigned MaxOccupancy = ST->getMaxWavesPerEU();
  // Only SGPR pressure can be fixed with remat, so if SGPR is the bottleneck,
  // aim directly for the occupancy VGPR would allow.
  if (STgtOcc < VTgtOcc) {
    unsigned BigOcc = std::max(STgtOcc, VTgtOcc);
    if (BigOcc > TgtOcc) {
      TgtOcc = BigOcc;
      SGPRBound = true;
      if (TgtOcc >= MaxOccupancy)
        TgtOcc = MaxOccupancy - 1;
    }
  }

  unsigned SInputPressure = computePhysicalSInputPressure(MRI, SIRI, ST);

  // If balanced, aim one occupancy tier higher.
  TgtOcc = SGPRBound ? TgtOcc : (TgtOcc + 1);

  auto CC = MF.getFunction().getCallingConv();
  bool IsPsCs =
      CC == CallingConv::AMDGPU_CS || CC == CallingConv::AMDGPU_PS;
  if (!IsPsCs)
    TgtOcc = TgtOcc > 4 ? 4 : TgtOcc;

  Status.TargetOcc = TgtOcc;
  Status.TargetSLimit = ST->getMaxNumSGPRs(TgtOcc, true);
  Status.TargetVLimit = ST->getMaxNumVGPRs(TgtOcc, DynVGPR);
  Status.MaxVPressure = MaxVPressure;
  Status.MaxSPressure = MaxSPressure;
  Status.InputPhysicalVPressure = 0;
  Status.InputPhysicalSPressure = SInputPressure;
  Status.MemBound = MemBound;
  Status.SGPRBound = SGPRBound;
  return Status;
}

//===----------------------------------------------------------------------===//
// Live-info bookkeeping for already-planned remats.
//===----------------------------------------------------------------------===//

static void updateLiveInfo(
    const MapVector<Register, RematNode> &RematMap,
    GCNRPTracker::LiveRegSet &LiveSet,
    const GCNRPTracker::LiveRegSet &InputLive, const MachineBasicBlock *CurBB,
    DenseMap<const MachineBasicBlock *, unsigned> &RPOTIndexMap) {
  for (auto &It : RematMap) {
    Register Reg = It.first;
    if (!LiveSet.count(Reg))
      continue;
    // Input-set registers are accounted for in getReducedSize instead.
    if (InputLive.count(Reg))
      continue;

    const RematNode &Node = It.second;
    if (Node.Kind == RematKind::Sink) {
      unsigned LiveBBIndex = RPOTIndexMap[CurBB];
      unsigned InsertBBIndex = RPOTIndexMap[Node.InsertBlock];
      // Def still precedes CurBB, so the value is still live here.
      if (LiveBBIndex > InsertBBIndex)
        continue;
    }
    LiveSet.erase(Reg);
  }
}

static int getReducedSize(const MapVector<Register, RematNode> &RematMap,
                          GCNRPTracker::LiveRegSet &CandidateSet,
                          const MachineRegisterInfo &MRI,
                          const BlockLiveInfo &LiveInfo,
                          DenseMap<const MachineBasicBlock *, unsigned>
                              &RPOTIndexMap) {
  int ReducedSize = 0;
  for (const auto &It : RematMap) {
    Register Reg = It.first;
    if (!CandidateSet.count(Reg))
      continue;

    bool IsReduced = false;
    const RematNode &Node = It.second;
    if (Node.Kind == RematKind::Sink) {
      unsigned LiveBBIndex = RPOTIndexMap[LiveInfo.BB];
      unsigned InsertBBIndex = RPOTIndexMap[Node.InsertBlock];
      if (LiveBBIndex < InsertBBIndex)
        IsReduced = true;
    } else {
      IsReduced = true;
      for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
        if (UseMI.getParent() == LiveInfo.BB) {
          IsReduced = false;
          break;
        }
      }
    }
    if (IsReduced)
      ReducedSize += Node.GainInDwords;

    CandidateSet.erase(Reg);
  }
  return ReducedSize;
}

//===----------------------------------------------------------------------===//
// Candidate building and unified selection.
//===----------------------------------------------------------------------===//

static void buildCandidates(std::vector<RematNode> &OutCandidates,
                            DenseSet<Register> &PinnedRegSet,
                            GCNRPTracker::LiveRegSet &CandidateRegSet,
                            MachineRegisterInfo &MRI, const SIInstrInfo *SIII,
                            const SIRegisterInfo *SIRI) {
  for (const auto &LiveRegIt : CandidateRegSet) {
    Register Reg = LiveRegIt.first;
    if (PinnedRegSet.count(Reg))
      continue;
    // SGPR only.
    if (SIRI->isVGPR(MRI, Reg))
      continue;

    bool IsSafeCandidate = true;
    MachineInstr *MI = MRI.getUniqueVRegDef(Reg);
    if (!MI || MI->getDesc().NumDefs > 1)
      IsSafeCandidate = false;

    if (IsSafeCandidate) {
      int Gain = computeGainInDwords(MI, Reg, MRI, SIRI);
      if (Gain > 0)
        OutCandidates.emplace_back(RematNode(Reg, MI, (unsigned)Gain >> 5));
      else
        IsSafeCandidate = false;
    }
    if (!IsSafeCandidate)
      PinnedRegSet.insert(Reg);
  }

  llvm::stable_sort(OutCandidates, [](const RematNode &I, const RematNode &J) {
    return I.GainInDwords > J.GainInDwords;
  });
}

static MachineBasicBlock *findSinkTarget(MachineInstr *DefMI, Register Reg,
                                         MachineDominatorTree *DT,
                                         MachinePostDominatorTree *PDT,
                                         MachineLoopInfo *MLI,
                                         MachineRegisterInfo &MRI,
                                         bool MemBound) {
  BlockSet BBSet;
  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg))
    BBSet.insert(UseMI.getParent());
  if (BBSet.empty())
    return nullptr;

  MachineBasicBlock *BB = *BBSet.begin();
  if (BBSet.size() > 1) {
    BB = nearestCommonDominator(DT, BBSet);
    if (!BB)
      return nullptr;
  }
  if (!MemBound)
    BB = findNonLoopDominator(BB, DT, MLI);
  if (!BB)
    return nullptr;
  if (!reachBlock(DefMI->getParent(), DT, PDT, MLI, BB))
    return nullptr;
  return BB;
}

// True if a non-invariant load would cross a memory-writing block on the
// dominator path between FromBB and ToBB.
static bool hasInterleavedMemoryWrite(MachineInstr *DefMI,
                                      MachineBasicBlock *FromBB,
                                      MachineBasicBlock *ToBB,
                                      RematStatus &Status,
                                      MachineDominatorTree *DT) {
  if (!DefMI->mayLoad() || DefMI->isDereferenceableInvariantLoad())
    return false;

  MachineBasicBlock *BB = ToBB;
  while (BB && BB != FromBB) {
    if (Status.MemWriteMBBSet.count(BB))
      return true;
    MachineDomTreeNode *N = DT->getNode(BB);
    if (!N || !N->getIDom())
      break;
    BB = N->getIDom()->getBlock();
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Clone site reduction.
//===----------------------------------------------------------------------===//

static void updateUsers(Register Reg, Register NewReg, bool IsSubRegDef,
                        SmallVectorImpl<MachineInstr *> &UserMIs) {
  for (MachineInstr *UseMI : UserMIs) {
    for (MachineOperand &MO : UseMI->operands()) {
      if (!MO.isReg())
        continue;
      if (MO.getReg() == Reg) {
        MO.setReg(NewReg);
        if (IsSubRegDef)
          MO.setSubReg(0);
      }
    }
  }
}

// Prunes redundant clone sites using dominator relationships and returns a map
// from a clone-site block to the after-hot-range blocks that can share it.
static DenseMap<MachineBasicBlock *, BlockSet> reduceCloneSites(
    Register Reg, BlockMap<SmallVector<MachineInstr *, 2>> &UserBlocks,
    BlockSet &UserMBBSet, std::vector<BlockLiveInfo> &HotBlocks,
    MachineDominatorTree *DT,
    DenseMap<const MachineBasicBlock *, unsigned> &RPOTIndexMap) {
  BlockSet HotBlockSet;
  for (BlockLiveInfo &HotBlock : HotBlocks)
    if (HotBlock.InputLive.count(Reg))
      HotBlockSet.insert(HotBlock.BB);

  SmallVector<MachineBasicBlock *, 8> AfterHotRangeMBBs;
  for (MachineBasicBlock *MBB : UserMBBSet) {
    if (HotBlockSet.count(MBB))
      continue;

    bool IsDomAllHotBlocks = true;
    bool IsDomedByAllHotBlocks = true;
    for (MachineBasicBlock *HotMBB : HotBlockSet) {
      if (!DT->dominates(MBB, HotMBB))
        IsDomAllHotBlocks = false;
      if (!DT->dominates(HotMBB, MBB))
        IsDomedByAllHotBlocks = false;
      if (!IsDomAllHotBlocks && !IsDomedByAllHotBlocks)
        break;
    }
    if (IsDomAllHotBlocks)
      UserBlocks.erase(MBB);
    else if (IsDomedByAllHotBlocks)
      AfterHotRangeMBBs.push_back(MBB);
  }

  DenseMap<MachineBasicBlock *, BlockSet> DomMap;
  if (!AfterHotRangeMBBs.empty()) {
    // Deterministic transitive closure: sort by RPO, process leaves first.
    llvm::stable_sort(AfterHotRangeMBBs,
                      [&RPOTIndexMap](MachineBasicBlock *A,
                                      MachineBasicBlock *B) {
                        return RPOTIndexMap[A] < RPOTIndexMap[B];
                      });
    for (int I = (int)AfterHotRangeMBBs.size() - 1; I >= 0; --I) {
      MachineBasicBlock *MBB = AfterHotRangeMBBs[I];
      for (int J = I + 1; J < (int)AfterHotRangeMBBs.size(); ++J) {
        MachineBasicBlock *MBB2 = AfterHotRangeMBBs[J];
        if (DT->dominates(MBB, MBB2)) {
          DomMap[MBB].insert(MBB2);
          BlockSet &Dom2 = DomMap[MBB2];
          DomMap[MBB].insert(Dom2.begin(), Dom2.end());
        }
      }
    }
    for (MachineBasicBlock *MBB : AfterHotRangeMBBs) {
      for (MachineBasicBlock *DomedMBB : DomMap[MBB])
        UserMBBSet.erase(DomedMBB);
    }
  }

  return DomMap;
}

//===----------------------------------------------------------------------===//
// Applying rematerializations.
//===----------------------------------------------------------------------===//

static void applyOneDefOneUseRemat(RematNode &Node, MachineRegisterInfo &MRI,
                                   SlotIndexes *SlotIndexes,
                                   const SIRegisterInfo *SIRI,
                                   const SIInstrInfo *SIII, LiveIntervals *LIS) {
  MachineInstr *DefMI = Node.DefMI;
  MachineInstr *InsertPointMI = Node.InsertPointMI;
  MachineBasicBlock *MBB = nullptr;

  MachineBasicBlock::iterator InsertPoint;
  if (InsertPointMI) {
    InsertPoint = InsertPointMI->getIterator();
    MBB = InsertPointMI->getParent();
  } else {
    InsertPoint = Node.InsertBlock->getFirstTerminator();
    MBB = Node.InsertBlock;
  }

  InsertPoint = adjustInsertPointToAvoidSccSmash(DefMI, MBB, InsertPoint, MRI,
                                                 SIRI, SIII, LIS);

  DefMI->removeFromParent();
  InsertPoint->getParent()->insert(InsertPoint, DefMI);

  SlotIndexes->removeSingleMachineInstrFromMaps(*DefMI);
  SlotIndexes->insertMachineInstrInMaps(*DefMI);
}

void AMDGPUHotBlockRematerialize::applyCloneRemat(
    RematNode &Node, std::vector<BlockLiveInfo> &HotBlocks,
    MachineDominatorTree *DT, MachineRegisterInfo &MRI,
    SlotIndexes *SlotIndexes, const SIRegisterInfo *SIRI,
    const SIInstrInfo *SIII, LiveIntervals *LIS, MachineFunction &MF) {
  Register Reg = Node.Reg;
  MachineInstr *DefMI = MRI.getUniqueVRegDef(Reg);

  const MCInstrDesc &Desc = DefMI->getDesc();
  const TargetRegisterClass *RC =
      SIRI->getAllocatableClass(SIII->getOpRegClass(*DefMI, 0));
  const bool IsSubRegDef = DefMI->getOperand(0).getSubReg() != 0;
  const DebugLoc &DL = DefMI->getDebugLoc();
  const unsigned OpNum = Desc.getNumOperands();

  // RPO index map for deterministic clone reduction.
  DenseMap<const MachineBasicBlock *, unsigned> RPOTIndexMap;
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (MachineBasicBlock *MBB : RPOT)
    RPOTIndexMap[MBB] = RPOTIndexMap.size();

  BlockMap<SmallVector<MachineInstr *, 2>> UserMap;
  BlockSet UserMBBSet;
  for (auto UseIt = MRI.use_instr_nodbg_begin(Reg);
       UseIt != MRI.use_instr_nodbg_end();) {
    MachineInstr &UseMI = *(UseIt++);
    UserMap[UseMI.getParent()].emplace_back(&UseMI);
    UserMBBSet.insert(UseMI.getParent());
  }

  DenseMap<MachineBasicBlock *, BlockSet> DomMap =
      reduceCloneSites(Reg, UserMap, UserMBBSet, HotBlocks, DT, RPOTIndexMap);

  for (auto &UseIt : UserMap) {
    MachineBasicBlock *MBB = UseIt.first;
    if (MBB == DefMI->getParent())
      continue;
    if (UserMBBSet.count(MBB) == 0)
      continue;

    Register NewReg = MRI.createVirtualRegister(RC);
    auto NewDef = BuildMI(MF, DL, Desc).addDef(NewReg);
    // Copy explicit operands only; implicit operands are regenerated by BuildMI
    // from the MCInstrDesc.
    for (unsigned I = 1; I < OpNum; ++I)
      NewDef = NewDef.add(DefMI->getOperand(I));
    for (MachineMemOperand *MO : DefMI->memoperands())
      NewDef->addMemOperand(MF, MO);

    MachineInstr *InsertPointMI = UseIt.second.front();
    SlotIndex LastSlot = SlotIndexes->getInstructionIndex(*InsertPointMI);
    for (MachineInstr *UseMI : UseIt.second) {
      SlotIndex Slot = SlotIndexes->getInstructionIndex(*UseMI);
      if (LastSlot > Slot) {
        LastSlot = Slot;
        InsertPointMI = UseMI;
      }
    }

    MachineBasicBlock::iterator InsertPoint =
        adjustInsertPointToAvoidSccSmash(DefMI, InsertPointMI->getParent(),
                                         InsertPointMI, MRI, SIRI, SIII, LIS);

    MBB->insert(InsertPoint, NewDef);
    SlotIndexes->insertMachineInstrInMaps(*NewDef);

    updateUsers(Reg, NewReg, IsSubRegDef, UseIt.second);

    auto DomMapIt = DomMap.find(MBB);
    if (DomMapIt != DomMap.end()) {
      for (MachineBasicBlock *UpdateMBB : DomMapIt->second)
        updateUsers(Reg, NewReg, IsSubRegDef, UserMap[UpdateMBB]);
    }
  }

  if (MRI.use_empty(Reg)) {
    SlotIndexes->removeSingleMachineInstrFromMaps(*DefMI);
    DefMI->eraseFromParent();
  }
}

void AMDGPUHotBlockRematerialize::applyRemat(
    MapVector<Register, RematNode> &RematMap,
    std::vector<BlockLiveInfo> &HotBlocks, MachineDominatorTree *DT,
    SlotIndexes *SlotIndexes, MachineRegisterInfo &MRI,
    const SIRegisterInfo *SIRI, const SIInstrInfo *SIII, LiveIntervals *LIS,
    MachineFunction &MF) {
  std::vector<RematNode> UpdateList;
  for (auto &It : RematMap)
    UpdateList.emplace_back(It.second);

  // Move defs before uses: sort by def slot index so a sunk def is relocated
  // before any use that may also be sunk.
  llvm::stable_sort(UpdateList,
                    [&SlotIndexes](const RematNode &I, const RematNode &J) {
                      return SlotIndexes->getInstructionIndex(*I.DefMI) <
                             SlotIndexes->getInstructionIndex(*J.DefMI);
                    });

  for (RematNode &Node : UpdateList) {
    if (Node.Kind == RematKind::Sink)
      applyOneDefOneUseRemat(Node, MRI, SlotIndexes, SIRI, SIII, LIS);
    else if (Node.Kind == RematKind::Clone)
      applyCloneRemat(Node, HotBlocks, DT, MRI, SlotIndexes, SIRI, SIII, LIS,
                      MF);
  }
}

//===----------------------------------------------------------------------===//
// Main pass logic.
//===----------------------------------------------------------------------===//

bool AMDGPUHotBlockRematerialize::hotBlockRemat(
    MachineFunction &MF, MachineLoopInfo *MLI, LiveIntervals *LIS,
    MachineDominatorTree *DT, MachinePostDominatorTree *PDT,
    bool &IsNearTarget) {
  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *SIII = ST->getInstrInfo();
  const SIRegisterInfo *SIRI = ST->getRegisterInfo();
  auto &MRI = MF.getRegInfo();

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  DenseMap<const MachineBasicBlock *, unsigned> RPOTIndexMap;
  for (const MachineBasicBlock *MBB : RPOT)
    RPOTIndexMap[MBB] = RPOTIndexMap.size();

  bool IsUpdated = false;
  RematStatus Status = getRematStatus(MF, MLI, LIS, MRI, ST);

  const unsigned MaxOcc = ST->getWavesPerEU(MF.getFunction()).second;
  LLVM_DEBUG(dbgs() << "remat: TargetOcc=" << Status.TargetOcc
                    << " MaxOcc=" << MaxOcc
                    << " TargetSLimit=" << Status.TargetSLimit
                    << " TargetVLimit=" << Status.TargetVLimit
                    << " MaxSPressure=" << Status.MaxSPressure
                    << " InputPhysS=" << Status.InputPhysicalSPressure
                    << " SGPRBound=" << Status.SGPRBound
                    << " MemBound=" << Status.MemBound << "\n");
  if (Status.TargetOcc >= MaxOcc)
    return false;

  const unsigned SgprBias = getSgprBias(ST);

  {
    int InitialRematSCnt = Status.MaxSPressure - Status.TargetSLimit;
    if (EnableAggressive)
      InitialRematSCnt += SgprBias;

    bool InitialIsSGPRSpill = false;
    if (InitialRematSCnt > 0)
      InitialIsSGPRSpill = nearSgprSpill(Status.MaxSPressure, ST, MF);

    const bool InitialIsForceRematSgpr =
        InitialIsSGPRSpill || Status.SGPRBound;

    if (Status.TargetOcc > ST->getOccupancyWithWorkGroupSizes(MF).second &&
        !InitialIsForceRematSgpr)
      return false;
  }

  MachineBasicBlock *EntryMBB = &MF.front();
  auto *SlotIndexes = LIS->getSlotIndexes();

  MapVector<Register, RematNode> SRematMap;
  DenseSet<Register> PinnedRegSet;
  std::vector<BlockLiveInfo> HotBlocks;

  for (MachineBasicBlock *MBB : post_order(EntryMBB)) {
    auto &RP = Status.MBBPressureMap[MBB];

    // Stage 1: coarse filter.
    if (RP.getVGPRNum(ST->hasGFX90AInsts()) < Status.TargetVLimit &&
        (RP.getMaxSGPR() + RegForVCC + Status.InputPhysicalSPressure) <
            Status.TargetSLimit)
      continue;

    // Stage 2: fine forward scan.
    unsigned MaxVPressure = 0;
    unsigned MaxSPressure = 0;
    const GCNRPTracker::LiveRegSet InputLive = Status.MBBInputLiveMap[MBB];
    const GCNRPTracker::LiveRegSet OutputLive = Status.MBBOutputLiveMap[MBB];

    // Drive the tracker from the first non-PHI instruction: PHI incoming
    // operands are not live at the block-entry slot, so feeding PHIs to the
    // downward tracker would trip its "register isn't live" invariant. Let the
    // tracker self-compute its live-in set from LiveIntervals.
    auto ScanBegin = MBB->getFirstNonPHI();
    ScanBegin = skipDebugInstructionsForward(ScanBegin, MBB->end());
    if (ScanBegin != MBB->end()) {
      GCNDownwardRPTracker Tracker(*LIS);
      Tracker.reset(*ScanBegin, MBB->end());
      while (true) {
        GCNRPTracker::LiveRegSet LISLR = Tracker.getLiveRegs();
        updateLiveInfo(SRematMap, LISLR, InputLive, MBB, RPOTIndexMap);

        unsigned VPressure = 0;
        unsigned SPressure = 0;
        collectLiveSetPressure(LISLR, MRI, SIRI, VPressure, SPressure);
        MaxVPressure = std::max(MaxVPressure, VPressure);
        MaxSPressure = std::max(MaxSPressure, SPressure);

        if (!Tracker.advance())
          break;
      }
    }
    MaxSPressure += RegForVCC + Status.InputPhysicalSPressure;
    LLVM_DEBUG(dbgs() << "remat: bb." << MBB->getNumber()
                      << " MaxSPressure=" << MaxSPressure
                      << " MaxVPressure=" << MaxVPressure << "\n");
    if (MaxVPressure <= Status.TargetVLimit &&
        MaxSPressure <= Status.TargetSLimit)
      continue;

    BlockLiveInfo LiveInfo = {MBB, MaxSPressure, MaxVPressure,
                              MBB != EntryMBB ? InputLive : OutputLive};
    if (MBB != EntryMBB)
      HotBlocks.emplace_back(LiveInfo);

    GCNRPTracker::LiveRegSet CandidateRegs = LiveInfo.InputLive;

    int SReduced =
        getReducedSize(SRematMap, CandidateRegs, MRI, LiveInfo, RPOTIndexMap);

    const int RematSCnt = MaxSPressure - SReduced - Status.TargetSLimit;
    LLVM_DEBUG(dbgs() << "remat: bb." << MBB->getNumber()
                      << " HOT, SReduced=" << SReduced
                      << " RematSCnt=" << RematSCnt << "\n");

    bool IsSGPRSpill = false;
    if (RematSCnt > 0)
      IsSGPRSpill = nearSgprSpill(MaxSPressure, ST, MF);
    bool IsForceRematSgpr = IsSGPRSpill || Status.SGPRBound;

    int NewRematSCnt = 0;
    if (RematSCnt > 0) {
      std::vector<RematNode> Candidates;
      buildCandidates(Candidates, PinnedRegSet, CandidateRegs, MRI, SIII, SIRI);

      // Unified selection: classify each safe candidate as Sink or Clone,
      // gain-order them together, and greedily select.
      std::vector<CandidateEntry> AllEntries;
      for (RematNode &Node : Candidates) {
        Register Reg = Node.Reg;
        if (!isSafeToMoveOrClone(Node.DefMI, MRI)) {
          LLVM_DEBUG(dbgs() << "remat:   reject " << printReg(Reg, SIRI)
                            << " unsafe\n");
          PinnedRegSet.insert(Reg);
          continue;
        }
        if (MRI.hasOneNonDBGUse(Reg)) {
          MachineInstr &UseMI = *MRI.use_instr_nodbg_begin(Reg);
          if (isImplicitDefUse(Node.DefMI, &UseMI)) {
            LLVM_DEBUG(dbgs() << "remat:   reject " << printReg(Reg, SIRI)
                              << " implicit-def-use\n");
            continue;
          }
          MachineBasicBlock *InsertBB = findSinkTarget(
              Node.DefMI, Reg, DT, PDT, MLI, MRI, Status.MemBound);
          if (!InsertBB) {
            LLVM_DEBUG(dbgs() << "remat:   reject " << printReg(Reg, SIRI)
                              << " no sink target\n");
            continue;
          }
          if (hasInterleavedMemoryWrite(Node.DefMI, Node.DefMI->getParent(),
                                        InsertBB, Status, DT)) {
            LLVM_DEBUG(dbgs() << "remat:   reject " << printReg(Reg, SIRI)
                              << " interleaved mem write\n");
            continue;
          }
          RematNode SinkNode = Node;
          SinkNode.Kind = RematKind::Sink;
          SinkNode.InsertBlock = InsertBB;
          SinkNode.InsertPointMI =
              InsertBB == UseMI.getParent() ? &UseMI : nullptr;
          AllEntries.push_back({SinkNode, (int)Node.GainInDwords, true});
        } else {
          RematNode CloneNode = Node;
          CloneNode.Kind = RematKind::Clone;
          AllEntries.push_back({CloneNode, (int)Node.GainInDwords, false});
        }
      }

      llvm::stable_sort(AllEntries,
                        [](const CandidateEntry &A, const CandidateEntry &B) {
                          return A.GainInDwords > B.GainInDwords;
                        });

      int AccumulatedGain = 0;
      for (CandidateEntry &E : AllEntries) {
        if (AccumulatedGain >= RematSCnt && !EnableAggressive)
          break;
        SRematMap[E.Node.Reg] = E.Node;
        AccumulatedGain += E.GainInDwords;
      }
      NewRematSCnt = RematSCnt - AccumulatedGain;
      LLVM_DEBUG(dbgs() << "remat: bb." << MBB->getNumber() << " candidates="
                        << Candidates.size() << " entries=" << AllEntries.size()
                        << " selected(gain)=" << AccumulatedGain
                        << " NewRematSCnt=" << NewRematSCnt << "\n");
    }

    if (RematSCnt > 0)
      IsUpdated = true;

    if (NewRematSCnt > 0) {
      if ((unsigned)NewRematSCnt <= ST->getSGPRAllocGranule()) {
        IsNearTarget = true;
      } else if (!IsForceRematSgpr) {
        // Cannot fully relieve this block; keep what we accumulated and move
        // on rather than discarding the plan for other blocks.
        continue;
      }
    }
  }

  if (SRematMap.empty())
    return IsUpdated;

  IsUpdated = true;
  applyRemat(SRematMap, HotBlocks, DT, SlotIndexes, MRI, SIRI, SIII, LIS, MF);
  LLVM_DEBUG(llvm::dbgs() << "after hot block remat:\n"; MF.print(dbgs()));

  return IsUpdated;
}

bool AMDGPUHotBlockRematerialize::runOnMachineFunction(MachineFunction &MF) {
  if (MF.size() < 2)
    return false;
  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MachineDominatorTree *DT =
      &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MachinePostDominatorTree *PDT =
      &getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();
  MachineLoopInfo *MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  bool IsNearTarget = false;
  return hotBlockRemat(MF, MLI, LIS, DT, PDT, IsNearTarget);
}

} // namespace

INITIALIZE_PASS_BEGIN(AMDGPUHotBlockRematerialize, DEBUG_TYPE,
                      "AMDGPU rematerialize", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AMDGPUHotBlockRematerialize, DEBUG_TYPE,
                    "AMDGPU rematerialize", false, false)

char AMDGPUHotBlockRematerialize::ID = 0;
char &llvm::AMDGPUHotBlockRematerializeID = AMDGPUHotBlockRematerialize::ID;

FunctionPass *llvm::createAMDGPUHotBlockRematerializePass() {
  return new AMDGPUHotBlockRematerialize();
}
