//===- AMDGPUHotBlockRematerialize.cpp - AMDGPU Hot BlockRematerialize ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief AMDGPU hot block Rematerialize
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUMIRUtils.h"
#include "AMDGPUOccupancyAndLatencyHelper.h"
#include "AMDGPUSubExpDag.h"
#include "AMDGPUSubtarget.h"
#include "AMDGPUVMemDegreeDAG.h"
#include "GCNRegPressure.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/SlotIndexes.h"

#include "llvm/CodeGen/MachineCycleAnalysis.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"

#include <unordered_set>
#define DEBUG_TYPE "amdgpu-hot-block-remat"

using namespace llvm;

static cl::opt<unsigned> TargetOccupancy("amdgpu-remat-target-occupancy");
static cl::opt<bool>
    EnableAggressive("amdgpu-remat-enable-hot-block-remat-aggressive");
static cl::opt<bool>
    EnableSubExpAggressive("amdgpu-remat-enable-sub-exp-remat-aggressive");
static cl::opt<bool>
    EnableSubExpClone("amdgpu-remat-enable-sub-exp-remat-clone");
static cl::opt<bool> EnableVmemDegree("amdgpu-remat-enable-vmem-degree");
static cl::opt<bool> EnableInBlockRemat("amdgpu-remat-enable-in-blk-remat");
static cl::opt<bool> EnableSubExp("amdgpu-remat-enable-sub-exp-remat");
static cl::opt<bool>
    EnableUniformVectorToScalar("amdgpu-remat-enable-late-float-vtos");
static cl::opt<bool>
    EnableSubExpMinReg("amdgpu-remat-enable-sub-exp-remat-min-reg");

namespace {
typedef DenseSet<MachineInstr *> InstSet;
typedef DenseSet<MachineBasicBlock *> BlockSet;
template <typename T> using BlockMap = MapVector<MachineBasicBlock *, T>;

// Rematerialize in a single pass instead of doing in register allcation.
// If in register allocation, fail to rematerialize will cause spill.
class AMDGPUHotBlockRematerialize : public MachineFunctionPass {

public:
  static char ID;

  DenseSet<const MachineInstr *> TotalUniformInsts;
  DenseSet<const MachineInstr *> SafeToRemoveInsts;
  DenseSet<const MachineInstr *> DivergentInsts;
  void removeInst(const MachineInstr *MI) {
    TotalUniformInsts.erase(MI);
    SafeToRemoveInsts.erase(MI);
    DivergentInsts.erase(MI);
  }

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
    AU.addRequired<AAResultsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

typedef AMDGPUHotBlockRematerialize Remat;

} // end anonymous namespace

// Util functions.
namespace {

MachineBasicBlock *nearestCommonDominator(MachineDominatorTree *DT,
                                          BlockSet &Blocks) {
  auto I = Blocks.begin(), E = Blocks.end();

  MachineBasicBlock *DomB = cast<MachineBasicBlock>(*(I++));
  while (I != E) {
    MachineBasicBlock *B = cast<MachineBasicBlock>(*(I++));
    DomB = DT->findNearestCommonDominator(DomB, B);
    if (DomB == nullptr)
      return nullptr;
  }
  // For split block like:
  // bb.42:
  //    %632.sub2:vreg_128 = V_MOV_B32_e32 %717.sub2:vreg_128, implicit $exec,
  //    //    implicit $exec
  //  %130:sreg_64 = S_AND_SAVEEXEC_B64 %533:sreg_64, implicitdef $exec,
  //  implicitdef $scc, implicit $exec
  //
  // bb.68:
  //; predecessors: %bb.42
  //  successors: %bb.45(0x40000000), %bb.43(0x40000000); %bb.45(50.00%),
  //  %bb.43(50.00%)
  //
  //  SI_MASK_BRANCH %bb.43, implicit $exec
  //  S_BRANCH %bb.45
  // which is from
  // bb.42:
  //%129:vgpr_32 = V_MOV_B32_e32 killed %548:vgpr_32, implicit $exec, implicit
  //$exec %130:sreg_64 = S_AND_SAVEEXEC_B64 %533:sreg_64, implicitdef $exec,
  // SI_MASK_BRANCH %bb.43, implicit $exec
  // S_BRANCH %bb.45
  // The real common dom is bb.42.
  // TODO: use _term version of exec update instructions so don't need this
  // anymore.
  if (DomB && DomB->pred_size() == 1 && !DomB->empty()) {
    // Upstreaming note: This used to be SI_MASK_BRANCH
    if (DomB->begin()->getOpcode() == AMDGPU::S_CBRANCH_EXECZ) {
      MachineBasicBlock *Pred = *DomB->pred_begin();
      if (Pred->succ_size() == 1 &&
          (Pred->empty() || !Pred->back().isBranch())) {
        DomB = Pred;
      }
    }
  }

  return DomB;
}

MachineBasicBlock *findNonLoopDominator(MachineBasicBlock *BB,
                                        MachineDominatorTree *DT,
                                        MachineLoopInfo *LI) {
  while (LI->getLoopDepth(BB) > 0) {
    MachineDomTreeNode *N = DT->getNode(BB);
    if (N == nullptr)
      return nullptr;
    MachineDomTreeNode *IDom = N->getIDom();
    if (IDom == nullptr)
      return nullptr;

    BB = IDom->getBlock();
  }

  return BB;
}

MachineBasicBlock *
findInsertBlock(MachineInstr &DefMI, unsigned Reg, MachineDominatorTree *DT,
                MachinePostDominatorTree *PDT, MachineLoopInfo *MLI,
                const MachineRegisterInfo &MRI, bool MemBound) {

  BlockSet BBSet;
  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
    BBSet.insert(UseMI.getParent());
  }
  if (BBSet.size() == 0)
    return nullptr;

  MachineBasicBlock *BB = *BBSet.begin();
  if (BBSet.size() > 1) {
    MachineBasicBlock *BDom = nearestCommonDominator(DT, BBSet);
    if (!BDom)
      return nullptr;
    BB = BDom;
  }
  // Try to find non loop dominator.
  if (!MemBound) {
    BB = findNonLoopDominator(BB, DT, MLI);
  }
  if (!BB)
    return nullptr;

  // If BB is already a hot block, move to BB will not help.
  // hotBlockRemat will fail It when process BB.

  // Must reachable from DefMI.
  if (!llvm::reach_block(DefMI.getParent(), DT, PDT, MLI, BB))
    return nullptr;

  return BB;
}

// Maybe expensive to be called all over the place
bool isUsedByPhi(MachineInstr *DefMI, MachineRegisterInfo &MRI) {
  for (auto &Def : DefMI->defs()) {
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Def.getReg())) {
      if (UseMI.isPHI())
        return true;
    }
  }
  return false;
}

bool isSafeToMove(MachineInstr *DefMI, MachineRegisterInfo &MRI) {
  // Do not move PHI nodes
  if (isUsedByPhi(DefMI, MRI))
    return false;

  unsigned OpNum = DefMI->getNumOperands();
  // Only move DefMI which all operand is unique def.
  for (unsigned I = 0; I < OpNum; I++) {
    MachineOperand &Op = DefMI->getOperand(I);
    if (!Op.isReg())
      continue;
    if (!MRI.getUniqueVRegDef(Op.getReg()) &&
        !llvm::IsSub0Sub1SingleDef(Op.getReg(), MRI)) {
      return false;
    }
  }
  return true;
}

// SGPR has alignment requirment, cannot get accurate reg number.
const unsigned NearTargetRegLimit = 10;
bool nearSgprSpill(unsigned MaxSPressure, const GCNSubtarget *ST,
                   MachineFunction &MF) {
  unsigned MaxSGPR = ST->getAddressableNumSGPRs();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  Register ScratchRSrcReg = MFI->getScratchRSrcReg();
  if (ScratchRSrcReg)
    MaxSGPR -= 4;

  const unsigned AlignmentDelta = 3;
  MaxSGPR -= AlignmentDelta;

  return MaxSPressure > MaxSGPR;
}

struct RematStatus {
  unsigned TargetOcc;
  unsigned TargetVLimit;
  unsigned TargetSLimit;
  unsigned MaxVPressure;
  unsigned MaxSPressure;
  unsigned InputPhysicalVPressure;
  unsigned InputPhysicalSPressure;
  // More occupancy can help more than latency cost to reach It.
  bool MemBound;
  // abs(VTargetOcc-STargetOcc) > 1.
  bool NotBalance;
  DenseMap<MachineBasicBlock *, GCNRegPressure> MBBPressureMap;
  DenseMap<MachineBasicBlock *, GCNRPTracker::LiveRegSet> MBBInputLiveMap;
  DenseMap<MachineBasicBlock *, GCNRPTracker::LiveRegSet> MBBOutputLiveMap;
  // Collect MBBs which has memory write. When move instructions cross MBB, skip
  // mem inst if the MBB has memory write. To make things fast, just check
  // mayStore and isBarrier.
  DenseSet<MachineBasicBlock *> MemWriteMBBSet;
};

unsigned collectMBBPressure(MachineBasicBlock &MBB, LiveIntervals *LIS,
                            const GCNSubtarget *ST, unsigned &MaxVPressure,
                            unsigned &MaxSPressure, RematStatus &Status) {
  // Skip processing current block if It has only debug instructions
  if (MBB.getFirstNonDebugInstr() == MBB.end())
    return ST->getOccupancyWithNumVGPRs(0);
  auto BBEnd = MBB.rbegin();
  GCNUpwardRPTracker RPTracker(*LIS);
  // R.End doesn't point to the boundary instruction.
  // Skip Debug instr.
  if (!llvm::GetNonDebugMBBEnd(BBEnd, MBB))
    return ST->getOccupancyWithNumVGPRs(0);

  GCNRPTracker::LiveRegSet OutputLive = Status.MBBOutputLiveMap[&MBB];
  RPTracker.reset(*BBEnd, &OutputLive, true);

  for (auto I = MBB.rbegin(), B = MBB.rend(); I != B;) {
    MachineInstr &MI = (*I++);
    RPTracker.recede(MI);
    if (MI.mayStore() || (MI.isBarrier() && MI.getOpcode() != AMDGPU::S_BRANCH))
      Status.MemWriteMBBSet.insert(&MBB);
  }

  GCNRegPressure RP = RPTracker.getMaxPressureAndReset();
  unsigned SPressure = RP.getMaxSGPR();
  if (SPressure > MaxSPressure) {
    MaxSPressure = SPressure;
  }
  if (RP.getVGPRNum(ST->hasGFX90AInsts()) > MaxVPressure) {
    MaxVPressure = RP.getVGPRNum(ST->hasGFX90AInsts());
  }
  Status.MBBPressureMap[&MBB] = RP;
  return RP.getOccupancy(*ST);
}

unsigned collectFnPressure(MachineFunction &MF, LiveIntervals *LIS,
                           const MachineRegisterInfo &MRI,
                           const GCNSubtarget *ST, unsigned &MaxVPressure,
                           unsigned &MaxSPressure, RematStatus &Status) {
  unsigned TgtOcc = ST->getOccupancyWithWorkGroupSizes(MF).second;
  // If only have one block, input/ouput virtual live set are empty.
  if (MF.size() > 1) {
    // Build input output live reg first.
    auto *SlotIndexes = LIS->getSlotIndexes();
    DenseMap<MachineBasicBlock *, SlotIndex> MBBInputSlotMap;
    DenseMap<MachineBasicBlock *, SlotIndex> MBBOutputSlotMap;
    for (MachineBasicBlock &MBB : MF) {
      auto BBBegin = MBB.getFirstNonDebugInstr();
      if (BBBegin != MBB.end()) {
        auto SI = SlotIndexes->getInstructionIndex(*BBBegin);
        MBBInputSlotMap[&MBB] = SI;
      }

      auto BBEnd = MBB.rbegin();

      // R.End doesn't point to the boundary instruction.
      // Skip Debug instr.
      if (llvm::GetNonDebugMBBEnd(BBEnd, MBB)) {
        auto SI = SlotIndexes->getInstructionIndex(*BBEnd);
        MBBOutputSlotMap[&MBB] = SI;
      }
    }

    for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
      auto Reg = Register::index2VirtReg(I);
      if (!LIS->hasInterval(Reg))
        continue;

      const auto &LI = LIS->getInterval(Reg);

      // Skip local live interval to make live input/ouput faster.
      if (llvm::isLocalLiveInterval(LI, SlotIndexes))
        continue;

      for (auto InputIt : MBBInputSlotMap) {
        MachineBasicBlock *MBB = InputIt.first;
        auto SI = InputIt.second;

        auto LiveMask = getLiveLaneMask(Reg, SI, *LIS, MRI);
        if (LiveMask.any())
          Status.MBBInputLiveMap[MBB][Reg] |= LiveMask;
      }

      for (auto OutputIt : MBBOutputSlotMap) {
        MachineBasicBlock *MBB = OutputIt.first;
        auto SI = OutputIt.second;

        auto LiveMask = getLiveLaneMask(Reg, SI, *LIS, MRI);
        if (LiveMask.any())
          Status.MBBOutputLiveMap[MBB][Reg] |= LiveMask;
      }
    }
  }

  LLVM_DEBUG(
      const SIRegisterInfo *SIRI = ST->getRegisterInfo();
      dbgs() << "output live"; for (auto &It
                                    : Status.MBBOutputLiveMap) {
        unsigned Idx = It.first->getNumber();
        auto LiveReg = It.second;
        dbgs() << "MBB" << Idx << ":";
        llvm::dumpLiveSet(LiveReg, SIRI);
      } dbgs() << "input live";
      for (auto &It
           : Status.MBBInputLiveMap) {
        unsigned Idx = It.first->getNumber();
        auto LiveReg = It.second;
        dbgs() << "MBB" << Idx << ":";
        llvm::dumpLiveSet(LiveReg, SIRI);
      });

  for (auto It = MF.begin(); It != MF.end(); ++It) {
    MachineBasicBlock &MBB = *It;
    unsigned Occ =
        collectMBBPressure(MBB, LIS, ST, MaxVPressure, MaxSPressure, Status);
    if (TgtOcc > Occ)
      TgtOcc = Occ;
  }
  return TgtOcc;
}
RematStatus getRematStatus(MachineFunction &MF, MachineLoopInfo *MLI,
                           LiveIntervals *LIS, const MachineRegisterInfo &MRI,
                           const GCNSubtarget *ST) {
  unsigned MaxSPressure = 0;
  unsigned MaxVPressure = 0;
  RematStatus Status;
  unsigned TgtOcc =
      collectFnPressure(MF, LIS, MRI, ST, MaxVPressure, MaxSPressure, Status);
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
    Status.NotBalance = false;
    return Status;
  }

  MaxSPressure += RegForVCC;
  MaxVPressure = std::min(MaxVPressure, ST->getMaxNumVGPRs(MF));
  unsigned STgtOcc = ST->getOccupancyWithNumSGPRs(MaxSPressure);
  unsigned VTgtOcc = ST->getOccupancyWithNumVGPRs(MaxVPressure);

  llvm::SchedScore TotalScore = llvm::CollectLatency(MF, *ST, MLI);
  bool MemBound =
      TotalScore.isMemBound(TgtOcc, std::max(STgtOcc, VTgtOcc) - TgtOcc);

  bool NotBalance = false;

  const unsigned MaxOccupancy = ST->AMDGPUSubtarget::getMaxWavesPerEU();
  // Currently, only sgpr bound can be fixed with remat.
  if (STgtOcc < VTgtOcc) {
    unsigned BigOcc = std::max(STgtOcc, VTgtOcc);
    // Change TgtOcc to  in case sgpr and vgpr is not balance.
    if (BigOcc > TgtOcc) {
      TgtOcc = BigOcc;
      NotBalance = true;
      if (TgtOcc >= MaxOccupancy)
        TgtOcc = MaxOccupancy - 1;
    }
  }

  // Collect input physical pressure.
  const SIRegisterInfo *SIRI = ST->getRegisterInfo();

  unsigned VInputPressure = 0;
  uint64_t SInputMask = 0;
  for (const auto &Livein : MRI.liveins()) {
    const Register Reg = Livein.first;
    const TargetRegisterClass *RC = SIRI->getRegClassForReg(MRI, Reg);
    assert(Reg.isPhysical() && "input must be physical reg");
    unsigned RegSize = RC->getLaneMask().getNumLanes();
    if (SIRI->isVGPR(MRI, Reg)) {
      VInputPressure += RegSize;
    } else {
      unsigned RegIndex = SIRI->getHWRegIndex(Reg);
      uint64_t Mask = ((1 << RegSize) - 1) << RegIndex;
      SInputMask |= Mask;
    }
  }
  // SGPR need to align to 4 for the 4dowrd/8dword descriptors which cause high
  // pressure.
  unsigned SInputPressure = 0;
  uint64_t Mask = 0xf;
  while (Mask != 0) {
    if (Mask & SInputMask) {
      SInputPressure += 4;
    }
    Mask = Mask << 4;
  }

  // If balanced, try next occupancy.
  TgtOcc = NotBalance ? TgtOcc : (TgtOcc + 1);

  auto CC = MF.getFunction().getCallingConv();
  bool IsPsCs = CC == CallingConv::AMDGPU_CS || CC == CallingConv::AMDGPU_PS;
  // For shader profiles other than ps/cs, set target profile max as 4.
  if (!IsPsCs) {
    TgtOcc = TgtOcc > 4 ? 4 : TgtOcc;
  }
  if (TargetOccupancy)
    TgtOcc = TargetOccupancy;

  unsigned SLimit = ST->getMaxNumSGPRs(TgtOcc, true);
  unsigned VLimit = ST->getMaxNumVGPRs(TgtOcc);

  Status.TargetOcc = TgtOcc;
  Status.TargetVLimit = VLimit;
  Status.TargetSLimit = SLimit;
  Status.MaxVPressure = MaxVPressure;
  Status.MaxSPressure = MaxSPressure;
  Status.InputPhysicalVPressure = VInputPressure;
  Status.InputPhysicalSPressure = SInputPressure;
  Status.MemBound = MemBound;
  Status.NotBalance = NotBalance;
  return Status;
}

} // namespace

// Remat.
namespace {

struct RematNode {
  enum class RematKind {
    Candidate, // Not ready yet.
    OneDefOneUse,
    Clone,
  };
  RematNode()
      : Reg(0), DefMI(nullptr), InsertBlock(nullptr), InsertPointMI(nullptr),
        Kind(RematKind::Candidate), Size(0) {}
  RematNode(unsigned R, MachineInstr *MI, unsigned S)
      : Reg(R), DefMI(MI), InsertBlock(nullptr), InsertPointMI(nullptr),
        Kind(RematKind::Candidate), Size(S) {}
  unsigned Reg;
  MachineInstr *DefMI;
  MachineBasicBlock *InsertBlock;
  union {
    MachineInstr *InsertPointMI;
    unsigned UserCount;
  };
  RematKind Kind;
  unsigned Size;
};

struct BlockLiveInfo {
  MachineBasicBlock *BB;
  unsigned MaxSReg;
  unsigned MaxVReg;
  // Input live is the live reg which cross block.
  const GCNRPTracker::LiveRegSet InputLive;
};

// Skip live reg remated to other block.
void updateLiveInfo(MapVector<Register, RematNode> &RematMap,
                    GCNRPTracker::LiveRegSet &LiveSet,
                    const GCNRPTracker::LiveRegSet &InputLive,
                    MachineBasicBlock *CurBB,
                    DenseMap<MachineBasicBlock *, unsigned> &RPOTIndexMap) {
  for (auto &It : RematMap) {
    unsigned Reg = It.first;
    // Skip reg not in live set.
    if (!LiveSet.count(Reg))
      continue;
    // Skip reg already in input set.
    // Input set will be taken care in getReducedSize.
    if (InputLive.count(Reg))
      continue;

    auto &Node = It.second;
    if (Node.Kind == RematNode::RematKind::OneDefOneUse) {
      MachineBasicBlock *InsertBB = Node.InsertBlock;
      // If LiveInfo.BB is after InsertBB in Reverse post order, the def is
      // still before LiveInfo.BB, It is still live.
      unsigned LiveBBIndex = RPOTIndexMap[CurBB];
      unsigned InsertBBIndex = RPOTIndexMap[InsertBB];
      if (LiveBBIndex > InsertBBIndex) {
        continue;
      }
    }
    // Already in remat map, don't need to check again, remove from
    // candidate.
    LiveSet.erase(Reg);
  }
}

int getSharedReducedSize(InstSet &ReducedInsts, bool IsVGPR,
                         const MachineRegisterInfo &MRI,
                         const SIRegisterInfo *SIRI) {

  // Find shared operand in ReducedInsts.
  int SharedSize = 0;
  DenseMap<unsigned, LaneBitmask> SharedRegMaskMap;
  for (MachineInstr *DefMI : ReducedInsts) {
    for (MachineOperand &MO : DefMI->operands()) {
      if (MO.isImm())
        continue;
      if (!MO.isReg())
        continue;
      if (MO.isDef())
        continue;
      if (MO.isTied())
        continue;
      Register Reg = MO.getReg();

      if (Reg == AMDGPU::EXEC)
        continue;
      if (!Reg.isVirtual())
        continue;

      if (IsVGPR != SIRI->isVGPR(MRI, MO.getReg())) {
        // Not support mix of v and s when remat now.
        continue;
      }

      const TargetRegisterClass *OpRC = MRI.getRegClass(Reg);
      int MOSize = SIRI->getRegSizeInBits(*OpRC) >> 5;
      unsigned Mask;
      if (unsigned SubIdx = MO.getSubReg()) {
        OpRC = SIRI->getSubRegisterClass(OpRC, SubIdx);
        int SubMOSize = SIRI->getRegSizeInBits(*OpRC) >> 5;
        Mask = (1 << SubMOSize) - 1;
      } else {
        Mask = (1 << MOSize) - 1;
      }
      auto SharedRegIt = SharedRegMaskMap.find(Reg);
      if (SharedRegIt == SharedRegMaskMap.end()) {
        SharedRegMaskMap[Reg] = LaneBitmask(Mask);
      } else {
        unsigned PrevMask = SharedRegIt->second.getAsInteger();
        if (unsigned SharedMask = (PrevMask & Mask)) {
          // Some thing is shared.
          for (int I = 0; I < MOSize; I++) {
            if (SharedMask & (1 << I)) {
              SharedSize += 1;
            }
          }
        }
        LaneBitmask MoMask = LaneBitmask(Mask | PrevMask);
        SharedRegMaskMap[Reg] = MoMask;
      }
    }
  }
  return SharedSize;
}

int getReducedSize(MapVector<Register, RematNode> &RematMap,
                   GCNRPTracker::LiveRegSet &CanidateSet, InstSet &ReducedInsts,
                   const MachineRegisterInfo &MRI, BlockLiveInfo &LiveInfo,
                   DenseMap<MachineBasicBlock *, unsigned> &RPOTIndexMap) {
  int ReducedSize = 0;
  for (auto &It : RematMap) {
    Register Reg = It.first;

    if (!CanidateSet.count(Reg))
      continue;

    bool IsReduced = false;
    auto &Node = It.second;
    if (Node.Kind == RematNode::RematKind::OneDefOneUse) {
      MachineBasicBlock *InsertBB = Node.InsertBlock;
      // If LiveInfo.BB is before InsertBB in Reverse post order, the def is
      // moved after LiveInfo.BB, It is not live anymore.
      unsigned LiveBBIndex = RPOTIndexMap[LiveInfo.BB];
      unsigned InsertBBIndex = RPOTIndexMap[InsertBB];
      if (LiveBBIndex < InsertBBIndex)
        IsReduced = true;
    } else {
      // Clone.
      IsReduced = true;
      // If has use in LiveInfo.BB, could not reduce from input live.
      for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
        if (UseMI.getParent() == LiveInfo.BB) {
          IsReduced = false;
          break;
        }
      }
    }
    if (IsReduced) {
      ReducedSize += Node.Size;
      ReducedInsts.insert(Node.DefMI);
    }

    // Already in remat map, don't need to check again, remove from candidate.
    CanidateSet.erase(Reg);
  }

  return ReducedSize;
}

int rematGain(MachineInstr *DefMI, unsigned Reg, const MachineRegisterInfo &MRI,
              const SIRegisterInfo *SIRI, bool IsVGPR) {
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

    // Don't move user of VCC.
    if (MO.getReg() == AMDGPU::VCC) {
      RematSize = 0;
      break;
    }
    Register Reg = MO.getReg();

    // Don't move physical register use.
    if (Reg.isPhysical()) {
      RematSize = 0;
      break;
    }

    if (IsVGPR != SIRI->isVGPR(MRI, Reg)) {
      // Not support mix of v and s when remat now.
      // TODO: count possible pressure change here.
      RematSize = 0;
      break;
    }
    bool IsSingleDef = MRI.hasOneDef(Reg);
    if (!IsSingleDef) {
      IsSingleDef = llvm::IsSub0Sub1SingleDef(Reg, MRI);
    }

    if (IsSingleDef) {
      // The reg might share with other candidates,  check It here.
      // Count share reg in getReducedSize.
      if (EnableAggressive) {
        // In case of aggressive remat, treat multi use reg as shared reg and
        // ignore size of shared reg.
        if (!MRI.hasOneNonDBGUse(Reg))
          continue;
      }
      const TargetRegisterClass *OpRC = MRI.getRegClass(Reg);
      if (unsigned SubIdx = MO.getSubReg()) {
        if (OpRC)
          OpRC = SIRI->getSubRegisterClass(OpRC, SubIdx);
      }
      int InputSize = SIRI->getRegSizeInBits(*OpRC);
      // If input not live in hotspot, move It cross hotspot should have
      // less reg then DefMi.
      if (RematSize > InputSize) {
        RematSize -= InputSize;
        continue;
      }
    }

    RematSize = 0;
    break;
  }
  return RematSize;
}

void buildRematCandiates(std::vector<RematNode> &Candidates,
                         GCNRPTracker::LiveRegSet &CandidateRegSet,
                         DenseSet<unsigned> &PinnedRegSet,
                         const MachineRegisterInfo &MRI,
                         const SIInstrInfo *SIII, const SIRegisterInfo *SIRI,
                         bool IsVGPR) {

  for (auto LiveRegIt : CandidateRegSet) {
    unsigned Reg = LiveRegIt.first;
    // Skip unsafe reg.
    if (PinnedRegSet.count(Reg))
      continue;

    if (SIRI->isVGPR(MRI, Reg) != IsVGPR)
      continue;
    bool IsSafeCandidate = true;
    MachineInstr *MI = MRI.getUniqueVRegDef(Reg);
    if (MI) {
      if (IsVGPR) {
        // Only remat valu now.
        if (!SIII->isVALU(MI->getOpcode()) && MI->getOpcode() != AMDGPU::COPY)
          IsSafeCandidate = false;
        if (MI->getOpcode() == AMDGPU::COPY) {
          // Make sure src is unique define.
          if (MI->getOperand(1).isReg() &&
              nullptr == MRI.getUniqueVRegDef(MI->getOperand(1).getReg()))
            IsSafeCandidate = false;
        } else {
          // Skip convergent valu.
          if (MI->isConvergent())
            IsSafeCandidate = false;
        }
      }
      // Skip inst has more than 1 def.
      if (MI->getDesc().NumDefs > 1)
        IsSafeCandidate = false;
    } else {
      IsSafeCandidate = false;
    }

    if (IsSafeCandidate) {
      int Gain = rematGain(MI, Reg, MRI, SIRI, IsVGPR);
      if (Gain > 0) {
        Candidates.emplace_back(RematNode(Reg, MI, Gain >> 5));
      } else {
        IsSafeCandidate = false;
      }
    }
    // Save unsafe reg.
    if (!IsSafeCandidate)
      PinnedRegSet.insert(Reg);
  }

  // Sort by gain.
  std::sort(Candidates.begin(), Candidates.end(),
            [](RematNode &I, RematNode &J) { return I.Size > J.Size; });
}

// For case like
//   %477:sreg_32_xm0 = S_AND_B32 %472.sub0:sreg_64_xexec, %304:sreg_32_xm0,
//   implicit-def dead $scc; xb.uniform
//  S_CMP_EQ_U32 %302:sreg_32_xm0, %475:sreg_32_xm0, implicit-def $scc;
//  xb.uniform %2489:sreg_32_xm0 = S_CSELECT_B32 %477:sreg_32_xm0, 16, implicit
//  killed $scc; xb.uniform
// Sink S_AND right before S_CSELECT will overwrite SCC.
// To avoid It, skip case when DefMI and UseMI has implicit define use.
bool isImplicitDefUse(MachineInstr *DefMI, MachineInstr *UseMI) {
  if (DefMI->getDesc().NumImplicitDefs == 0)
    return false;

  auto *TRI = DefMI->getMF()->getSubtarget().getRegisterInfo();
  for (MachineOperand &Def : DefMI->implicit_operands()) {
    if (!Def.isReg())
      continue;
    if (Def.isUse())
      continue;
    Register Reg = Def.getReg();
    if (UseMI->readsRegister(Reg, TRI))
      return true;
  }
  return false;
}

void addOneDefOneUseCandidate(RematNode &Node,
                              std::vector<RematNode> &RematList,
                              MachineRegisterInfo &MRI, int &RematCnt,
                              MachineDominatorTree *DT,
                              MachinePostDominatorTree *PDT,
                              MachineLoopInfo *MLI, bool IsVGPR,
                              bool MemBound) {
  unsigned Reg = Node.Reg;
  MachineInstr *DefMI = Node.DefMI;

  unsigned Size = Node.Size;
  MachineInstr *UseMI = &*MRI.use_nodbg_instructions(Reg).begin();
  MachineBasicBlock *InsertBB = UseMI->getParent();

  // For VGPR, always move next to the only user to avoid wqm or exec issue.
  // But doing this will cause issue when DefMI is in wqm  user not in
  // wqm. Disable VGPR remat for now.
  // TODO: make sure single user don't need wqm.
  if (!IsVGPR) {
    if (MachineBasicBlock *NewInsertBB =
            findInsertBlock(*DefMI, Reg, DT, PDT, MLI, MRI, MemBound)) {
      if (InsertBB != NewInsertBB) {
        InsertBB = NewInsertBB;
        // If can find a non-loop insert block, go to the insert block.
        if (DefMI->getParent() != InsertBB) {
          if (!InsertBB->empty()) {
            auto It = InsertBB->getFirstNonPHI();
            It = skipDebugInstructionsForward(It, InsertBB->end());
            if (It == InsertBB->end())
              UseMI = nullptr;
            else
              UseMI = &*It;
          }
        }
      }
    }
  }

  if (IsVGPR) {
    // Don't count reg in same block for valu.
    if (UseMI->getParent() == DefMI->getParent())
      return;
  }

  // Skip case when DefMI has implicit define which used by UseMI.
  if (isImplicitDefUse(DefMI, UseMI)) {
    return;
  }

  Node.InsertBlock = InsertBB;
  Node.InsertPointMI = UseMI;
  Node.Kind = RematNode::RematKind::OneDefOneUse;
  RematList.emplace_back(Node);
  RematCnt += Size;
}

void addCloneCandidate(std::vector<RematNode *> &CloneList,
                       std::vector<RematNode> &RematList,
                       DenseSet<unsigned> &PinnedRegSet,
                       MachineRegisterInfo &MRI, int &RematCnt) {
  // Group user in same blocks.
  std::vector<BlockSet> UserSetList(CloneList.size());

  for (size_t I = 0; I < CloneList.size(); I++) {
    auto *Node = CloneList[I];
    unsigned Reg = Node->Reg;
    MachineInstr *DefMI = Node->DefMI;
    // Group user in same blocks.
    BlockSet &UserSet = UserSetList[I];

    for (auto UseIt = MRI.use_instr_nodbg_begin(Reg);
         UseIt != MRI.use_instr_nodbg_end();) {
      MachineInstr &UseMI = *(UseIt++);
      UserSet.insert(UseMI.getParent());
    }

    if (UserSet.size() == 1) {
      // All users are in same block with DefMI.
      if (*UserSet.begin() == DefMI->getParent()) {
        // Mark cannot remat for now.
        // TODO: try to split if is bigger than 4 and only used once per
        // channel.
        PinnedRegSet.insert(Reg);
        continue;
      }
    }

    int Size = Node->Size;
    Size <<= 16;
    // Pack userSet size to size.
    Size |= UserSet.size();
    Node->UserCount = Size;
  }

  std::sort(CloneList.begin(), CloneList.end(),
            // Sort based on userSet size.
            [](const RematNode *A, const RematNode *B) {
              static constexpr int Mask = 0xffff;
              return (A->UserCount & Mask) < (B->UserCount & Mask);
            });

  for (RematNode *Node : CloneList) {
    Node->Kind = RematNode::RematKind::Clone;
    RematList.emplace_back(*Node);
    RematCnt += Node->Size;
  }
}

int filterRematCandiates(std::vector<RematNode> &Candidates,
                         std::vector<RematNode> &RematList,
                         DenseSet<unsigned> &PinnedRegSet,
                         MachineDominatorTree *DT,
                         MachinePostDominatorTree *PDT, MachineLoopInfo *MLI,
                         MachineRegisterInfo &MRI, bool IsVGPR, bool MemBound) {
  int RematCnt = 0;
  // Work one def one use first.
  for (auto &Node : Candidates) {
    unsigned Reg = Node.Reg;
    if (!MRI.hasOneNonDBGUse(Reg)) {
      continue;
    }
    MachineInstr *DefMI = Node.DefMI;
    if (!isSafeToMove(DefMI, MRI)) {
      PinnedRegSet.insert(Reg);
      continue;
    }

    addOneDefOneUseCandidate(Node, RematList, MRI, RematCnt, DT, PDT, MLI,
                             IsVGPR, MemBound);
  }

  if (!IsVGPR) {
    std::vector<RematNode *> CloneList;
    // Try multi use case.
    for (auto &Node : Candidates) {
      unsigned Reg = Node.Reg;
      if (MRI.hasOneNonDBGUse(Reg)) {
        continue;
      }
      MachineInstr *DefMI = Node.DefMI;
      if (!isSafeToMove(DefMI, MRI)) {
        PinnedRegSet.insert(Reg);
        continue;
      }

      // Clone for each user.
      CloneList.emplace_back(&Node);
    }

    addCloneCandidate(CloneList, RematList, PinnedRegSet, MRI, RematCnt);
  }

  return RematCnt;
}

void updateUsers(unsigned Reg, unsigned NewReg, bool IsSubRegDef,
                 SmallVector<MachineInstr *, 2> &UserMIs) {
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

DenseMap<MachineBasicBlock *, BlockSet> reduceClonedMBBs(
    unsigned Reg, BlockMap<SmallVector<MachineInstr *, 2>> &UserBlocks,
    DenseSet<MachineBasicBlock *> &UserMBBSet,
    std::vector<BlockLiveInfo> &HotBlocks, MachineDominatorTree *DT) {
  // Collect hot blocks which Exp is live in.
  DenseSet<MachineBasicBlock *> HotBlockSet;
  for (BlockLiveInfo &HotBlock : HotBlocks) {
    if (HotBlock.InputLive.count(Reg)) {
      HotBlockSet.insert(HotBlock.BB);
    }
  }

  // For userBlocks which dominate all hotBlocks, don't need to clone because
  // the value not cross hotBlocks when later blocks are cloned.
  // For userBlocks which dominated by all hotBlocks, they could share clones
  // because once after hot block, the pressure is OK.
  DenseSet<MachineBasicBlock *> AfterHotRangeMBBs;
  for (MachineBasicBlock *MBB : UserMBBSet) {
    // Always clone in hot block.
    if (HotBlockSet.count(MBB))
      continue;

    bool IsDomAllHotBlocks = true;
    bool IsDomedByAllHotBlocks = true;
    for (MachineBasicBlock *HotMBB : HotBlockSet) {
      if (!DT->dominates(MBB, HotMBB)) {
        IsDomAllHotBlocks = false;
      }
      if (!DT->dominates(HotMBB, MBB)) {
        IsDomedByAllHotBlocks = false;
      }
      if (!IsDomAllHotBlocks && !IsDomedByAllHotBlocks) {
        break;
      }
    }
    if (IsDomAllHotBlocks) {
      UserBlocks.erase(MBB);
    } else if (IsDomedByAllHotBlocks) {
      AfterHotRangeMBBs.insert(MBB);
    }
  }

  // Split after hotRange block set by domtree.
  DenseMap<MachineBasicBlock *, BlockSet> DomMap;
  if (!AfterHotRangeMBBs.empty()) {
    for (MachineBasicBlock *MBB : AfterHotRangeMBBs) {
      for (MachineBasicBlock *MBB2 : AfterHotRangeMBBs) {
        if (MBB == MBB2)
          continue;
        if (DT->dominates(MBB, MBB2)) {
          auto &Dom = DomMap[MBB];
          Dom.insert(MBB2);
          auto &Dom2 = DomMap[MBB2];
          Dom.insert(Dom2.begin(), Dom2.end());
        }
      }
    }
    for (MachineBasicBlock *MBB : AfterHotRangeMBBs) {
      auto &Dom = DomMap[MBB];
      for (MachineBasicBlock *DomedMBB : Dom) {
        // Remove domedMBB.
        DomMap.erase(DomedMBB);
        UserMBBSet.erase(DomedMBB);
      }
    }
  }

  return DomMap;
}

// Look for an earlier insert point if the InstructionToMove
// writes to scc and scc is live at the CurrentInsertPoint.
static MachineBasicBlock::iterator adjustInsertPointToAvoidSccSmash(
    MachineInstr *InstructionToMove, MachineBasicBlock *MBB,
    MachineBasicBlock::iterator CurrentInsertPoint, MachineRegisterInfo &MRI,
    const SIRegisterInfo *SIRI, const SIInstrInfo *SIII) {
  const bool WillSmashScc =
      InstructionToMove->modifiesRegister(AMDGPU::SCC, SIRI);
  if (WillSmashScc) {
    CurrentInsertPoint = llvm::FindOrCreateInsertionPointForSccDef(
        MBB, CurrentInsertPoint, SIRI, SIII, &MRI);
  }

  return CurrentInsertPoint;
}

// Look for an earlier insert point if the SubExp
// writes to scc and scc is live at the CurrentInsertPoint.
static MachineBasicBlock::iterator adjustInsertPointForSubExpToAvoidSccSmash(
    const SubExp &SubExpToMove, MachineBasicBlock *MBB,
    MachineBasicBlock::iterator CurrentInsertPoint, MachineRegisterInfo &MRI,
    const SIRegisterInfo *SIRI, const SIInstrInfo *SIII) {
  const bool WillSmashScc = SubExpToMove.modifiesRegister(AMDGPU::SCC, SIRI);
  if (WillSmashScc) {
    CurrentInsertPoint = llvm::FindOrCreateInsertionPointForSccDef(
        MBB, CurrentInsertPoint, SIRI, SIII, &MRI);
  }

  return CurrentInsertPoint;
}

// Return trun if moving MI to Location will smash a live scc value.
static bool willSmashSccAtLocation(MachineInstr *MI, MachineBasicBlock *MBB,
                                   MachineBasicBlock::iterator Location) {
  // It is ok to pass nullptr to `modifiesRegister` for TRI here since
  // SCC has no subreg/suprereg relationships.
  return MI->modifiesRegister(AMDGPU::SCC, nullptr) &&
         llvm::IsSccLiveAt(MBB, Location);
}

void applyCloneRemat(Remat *Remat, RematNode &Node,
                     std::vector<BlockLiveInfo> &HotBlocks,
                     MachineDominatorTree *DT, MachineRegisterInfo &MRI,
                     SlotIndexes *SlotIndexes, const SIRegisterInfo *SIRI,
                     const SIInstrInfo *SIII, MachineFunction &MF) {
  unsigned Reg = Node.Reg;

  MachineInstr *DefMI = MRI.getUniqueVRegDef(Reg);
  auto DefOp = DefMI->getOperand(0);
  const MCInstrDesc &Desc = DefMI->getDesc();
  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  // When the unique def has subReg, just create newReg for the subReg part.
  bool IsSubRegDef = false;
  if (DefOp.getSubReg() != 0) {
    RC = SIRI->getSubRegisterClass(RC, DefOp.getSubReg());
    IsSubRegDef = true;
  }
  const DebugLoc DL = DefMI->getDebugLoc();
  unsigned OpNum = DefMI->getNumOperands();

  Node.Kind = RematNode::RematKind::Clone;

  // Group user in same blocks.
  BlockMap<SmallVector<MachineInstr *, 2>> UserMap;
  DenseSet<MachineBasicBlock *> UserMBBSet;
  for (auto UseIt = MRI.use_instr_nodbg_begin(Reg);
       UseIt != MRI.use_instr_nodbg_end();) {
    MachineInstr &UseMI = *(UseIt++);
    UserMap[UseMI.getParent()].emplace_back(&UseMI);
    UserMBBSet.insert(UseMI.getParent());
  }

  DenseMap<MachineBasicBlock *, BlockSet> DomMap =
      reduceClonedMBBs(Reg, UserMap, UserMBBSet, HotBlocks, DT);

  for (auto UseIt : UserMap) {
    MachineBasicBlock *MBB = UseIt.first;
    // Skip same block uses.
    if (MBB == DefMI->getParent()) {
      continue;
    }
    // Skip MBB which share clone from other MBBs.
    if (UserMBBSet.count(MBB) == 0)
      continue;

    Register NewReg = MRI.createVirtualRegister(RC);
    auto NewDef = BuildMI(MF, DL, Desc).addDef(NewReg);
    for (unsigned I = 1; I < OpNum; I++) {
      NewDef = NewDef.add(DefMI->getOperand(I));
    }

    MachineInstr *InsertPointMI = UseIt.second.front();
    SlotIndex LastSlot = SlotIndexes->getInstructionIndex(*InsertPointMI);

    for (MachineInstr *UseMI : UseIt.second) {
      SlotIndex Slot = SlotIndexes->getInstructionIndex(*UseMI);
      if (LastSlot > Slot) {
        LastSlot = Slot;
        InsertPointMI = UseMI;
      }
    }

    MachineBasicBlock::iterator InsertPoint = adjustInsertPointToAvoidSccSmash(
        DefMI, InsertPointMI->getParent(), InsertPointMI, MRI, SIRI, SIII);

    for (MachineMemOperand *MO : DefMI->memoperands()) {
      NewDef->addMemOperand(MF, MO);
    }

    MBB->insert(InsertPoint, NewDef);

    SlotIndexes->insertMachineInstrInMaps(*NewDef);

    SmallVector<MachineInstr *, 2> &UserMIs = UseIt.second;
    updateUsers(Reg, NewReg, IsSubRegDef, UserMIs);

    // update users in dom MBBs.
    auto DomMapIt = DomMap.find(MBB);
    if (DomMapIt != DomMap.end()) {
      for (MachineBasicBlock *UpdateMBB : DomMapIt->second) {
        SmallVector<MachineInstr *, 2> &UserMIs = UserMap[UpdateMBB];
        updateUsers(Reg, NewReg, IsSubRegDef, UserMIs);
      }
    }

    llvm::removeUnusedLanes(*NewDef.getInstr(), MRI, SIRI, SIII, SlotIndexes);
  }
  if (MRI.use_empty(Reg)) {
    SlotIndexes->removeSingleMachineInstrFromMaps(*DefMI);
    Remat->removeInst(DefMI);
    DefMI->eraseFromParent();
  }
}

void applyOneDefOneUseRemat(RematNode &Node, MachineRegisterInfo &MRI,
                            SlotIndexes *SlotIndexes,
                            const SIRegisterInfo *SIRI,
                            const SIInstrInfo *SIII) {
  MachineInstr *DefMI = Node.DefMI;
  MachineInstr *InsertPointMI = Node.InsertPointMI;
  MachineBasicBlock *MBB = nullptr;

  // Find a valid insert point.
  MachineBasicBlock::iterator InsertPoint;
  if (InsertPointMI) {
    InsertPoint = InsertPointMI->getIterator();
    MBB = InsertPointMI->getParent();
  } else {
    InsertPoint = Node.InsertBlock->getFirstTerminator();
    MBB = Node.InsertBlock;
  }

  InsertPoint = adjustInsertPointToAvoidSccSmash(DefMI, MBB, InsertPoint, MRI,
                                                 SIRI, SIII);

  // Move instruction to new location.
  DefMI->removeFromParent();
  InsertPoint->getParent()->insert(InsertPoint, DefMI);

  // Update slot index.
  SlotIndexes->removeSingleMachineInstrFromMaps(*DefMI);
  SlotIndexes->insertMachineInstrInMaps(*DefMI);
}

void applyRemat(Remat *Remat, MapVector<Register, RematNode> &RematMap,
                std::vector<BlockLiveInfo> &HotBlocks, MachineDominatorTree *DT,
                SlotIndexes *SlotIndexes, MachineRegisterInfo &MRI,
                const SIRegisterInfo *SIRI, const SIInstrInfo *SIII,
                MachineFunction &MF) {
  std::vector<RematNode> UpdateList;
  for (auto &It : RematMap) {
    UpdateList.emplace_back(It.second);
  }
  // Sort update list with slotIndex to make sure def moved before use.
  // If use moved before def, It might not be the first use anymore.
  std::sort(UpdateList.begin(), UpdateList.end(),
            [&SlotIndexes](RematNode &I, RematNode &J) {
              SlotIndex A = SlotIndexes->getInstructionIndex(*I.DefMI);
              SlotIndex B = SlotIndexes->getInstructionIndex(*J.DefMI);
              return A < B;
            });

  for (RematNode &Node : UpdateList) {
    if (Node.Kind == RematNode::RematKind::OneDefOneUse) {
      applyOneDefOneUseRemat(Node, MRI, SlotIndexes, SIRI, SIII);
    } else if (Node.Kind == RematNode::RematKind::Clone) {
      applyCloneRemat(Remat, Node, HotBlocks, DT, MRI, SlotIndexes, SIRI, SIII,
                      MF);
    }
  }
}

void dumpRematMap(MapVector<Register, RematNode> &RematMap,
                  const SIRegisterInfo *SIRI) {
  dbgs() << "\n rematMap: \n";
  for (auto It : RematMap) {
    int Reg = It.first;
    dbgs() << printReg(Reg, SIRI);
    dbgs() << "\n";
  }
}

int DebugBlockIndex = 42;

void dumpHotBlock(const GCNRPTracker::LiveRegSet &LiveSet,
                  MapVector<Register, RematNode> &VRematMap,
                  MapVector<Register, RematNode> &SRematMap, int BlockIndex,
                  const SIRegisterInfo *SIRI) {
  if (DebugBlockIndex != BlockIndex)
    return;
  llvm::dumpLiveSet(LiveSet, SIRI);
  dumpRematMap(VRematMap, SIRI);
  dumpRematMap(SRematMap, SIRI);
}

void dumpCandidates(std::vector<RematNode> &RematCandidates, int BlockIndex,
                    const SIRegisterInfo *SIRI) {
  if (DebugBlockIndex != BlockIndex)
    return;
  dbgs() << "\n Candidates: \n";
  unsigned TotalSize = 0;
  for (RematNode &Node : RematCandidates) {
    dbgs() << printReg(Node.Reg, SIRI) << " size:" << Node.Size;
    dbgs() << "\n";
    TotalSize += Node.Size;
  }
  dbgs() << "Total Size:" << TotalSize << "\n";
}

} // namespace

bool hotBlockRemat(Remat *Remat, MachineFunction &MF, MachineLoopInfo *MLI,
                   LiveIntervals *LIS, MachineDominatorTree *DT,
                   MachinePostDominatorTree *PDT, bool &IsNearTarget) {
  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();

  const SIInstrInfo *SIII = ST->getInstrInfo();
  const SIRegisterInfo *SIRI = ST->getRegisterInfo();

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  DenseMap<MachineBasicBlock *, unsigned> RPOTIndexMap;
  for (MachineBasicBlock *MBB : RPOT) {
    RPOTIndexMap[MBB] = RPOTIndexMap.size();
  }

  auto &MRI = MF.getRegInfo();

  bool IsUpdated = false;
  RematStatus Status = getRematStatus(MF, MLI, LIS, MRI, ST);

  const unsigned MaxOcc = ST->getWavesPerEU(MF.getFunction()).second;
  if (Status.TargetOcc >= MaxOcc)
    return false;

  unsigned VLimit = Status.TargetVLimit;
  unsigned SLimit = Status.TargetSLimit;

  int RematSCnt = Status.MaxSPressure - SLimit;
  // when agressive sgpr remat, reserve some for allocation lost.
  if (EnableAggressive)
    RematSCnt += NearTargetRegLimit;

  bool IsSGPRSpill = false;
  if (RematSCnt > 0) {
    IsSGPRSpill = nearSgprSpill(Status.MaxSPressure, ST, MF);
  }

  const bool IsForceRematSgpr = IsSGPRSpill || Status.NotBalance;

  // If bound by lds, skip.
  if (Status.TargetOcc > ST->getOccupancyWithWorkGroupSizes(MF).second &&
      !IsForceRematSgpr)
    return false;

  MachineBasicBlock *EntryMBB = &MF.front();

  auto *SlotIndexes = LIS->getSlotIndexes();

  // Reg which already marked remat.
  MapVector<Register, RematNode> VRematMap;
  MapVector<Register, RematNode> SRematMap;
  // Reg which cannot move around to remat.
  DenseSet<unsigned> PinnedRegSet;
  std::vector<BlockLiveInfo> HotBlocks;
  for (auto It = po_begin(EntryMBB); It != po_end(EntryMBB); It++) {
    MachineBasicBlock *MBB = *It;
    auto &RP = Status.MBBPressureMap[MBB];
    // ignore block not hot.
    if (RP.getVGPRNum(ST->hasGFX90AInsts()) < Status.TargetVLimit &&
        (RP.getMaxSGPR() + RegForVCC + Status.InputPhysicalSPressure) <
            Status.TargetSLimit)
      continue;
    // Collect reg pressure.
    unsigned MaxVPressure = 0;
    unsigned MaxSPressure = 0;
    const GCNRPTracker::LiveRegSet InputLive = Status.MBBInputLiveMap[MBB];

    const GCNRPTracker::LiveRegSet OutputLive = Status.MBBOutputLiveMap[MBB];
    LLVM_DEBUG(
        dumpHotBlock(InputLive, VRematMap, SRematMap, MBB->getNumber(), SIRI));

    GCNDownwardRPTracker Tracker(*LIS);

    Tracker.reset(*MBB->begin(), &InputLive);

    for (MachineInstr &MI : *MBB) {
      if (MI.isDebugInstr())
        continue;
      Tracker.advance();
      auto LISLR = Tracker.getLiveRegs();
      // Update live set for things already remated.
      updateLiveInfo(VRematMap, LISLR, InputLive, MBB, RPOTIndexMap);
      updateLiveInfo(SRematMap, LISLR, InputLive, MBB, RPOTIndexMap);

      const GCNRPTracker::LiveRegSet &LiveSet = LISLR;
      unsigned VPressure = 0;
      unsigned SPressure = 0;
      CollectLiveSetPressure(LiveSet, MRI, SIRI, VPressure, SPressure);
      if (MaxVPressure < VPressure)
        MaxVPressure = VPressure;
      if (MaxSPressure < SPressure)
        MaxSPressure = SPressure;
    }
    MaxSPressure += RegForVCC + Status.InputPhysicalSPressure;
    if (MaxVPressure <= VLimit && MaxSPressure <= SLimit)
      continue;

    // Build block live info.
    // Use outputLive for EntryMBB.
    BlockLiveInfo LiveInfo = {MBB, MaxSPressure, MaxVPressure,
                              MBB != EntryMBB ? InputLive : OutputLive};
    // Skip entry block when save hotBlock to reduce clone because not clone in
    // entry block.
    if (MBB != EntryMBB)
      HotBlocks.emplace_back(LiveInfo);
    GCNRPTracker::LiveRegSet CandidateRegs = LiveInfo.InputLive;

    // Update reg pressure based on remat list.
    InstSet VReducedInsts;
    InstSet SReducedInsts;
    int VReduced = getReducedSize(VRematMap, CandidateRegs, VReducedInsts, MRI,
                                  LiveInfo, RPOTIndexMap);
    int SReduced = getReducedSize(SRematMap, CandidateRegs, SReducedInsts, MRI,
                                  LiveInfo, RPOTIndexMap);

    // Calculate size need to be remat.
    int RematVCnt = MaxVPressure - VReduced - VLimit;
    int RematSCnt = MaxSPressure - SReduced - SLimit;

    bool IsSGPRSpill = false;
    if (RematSCnt > 0) {
      IsSGPRSpill = nearSgprSpill(MaxSPressure, ST, MF);
    }
    bool IsForceRematSgpr = IsSGPRSpill || Status.NotBalance;
    // Try to add candidates into remat list.

    int NewRematSCnt = 0;
    if (RematSCnt > 0) {
      // Build candidate nodes.
      std::vector<RematNode> SRematCandidates;
      buildRematCandiates(SRematCandidates, CandidateRegs, PinnedRegSet, MRI,
                          SIII, SIRI, /*IsVGPR*/ false);

      LLVM_DEBUG(dumpCandidates(SRematCandidates, MBB->getNumber(), SIRI));
      std::vector<RematNode> SRematList;
      // Filter candidates.
      NewRematSCnt = filterRematCandiates(SRematCandidates, SRematList,
                                          PinnedRegSet, DT, PDT, MLI, MRI,
                                          /*IsVGPR*/ false, Status.MemBound);
      if (NewRematSCnt > RematSCnt) {
        // Has enough remat node to cover rematCnt.
        int RematCnt = 0;
        for (RematNode &Node : SRematList) {
          SRematMap[Node.Reg] = Node;
          RematCnt += Node.Size;
          if (RematCnt > RematSCnt && !EnableAggressive)
            break;
        }
        NewRematSCnt = 0;
      } else {

        for (RematNode &Node : SRematList) {
          SReducedInsts.insert(Node.DefMI);
        }
        // Check shared size.
        int SharedReducedSize =
            getSharedReducedSize(SReducedInsts, /*IsVGPR*/ false, MRI, SIRI);
        if (((NewRematSCnt + SharedReducedSize) + (int)NearTargetRegLimit) >=
            RematSCnt) {
          for (RematNode &Node : SRematList) {
            SRematMap[Node.Reg] = Node;
          }
        } else {
          if (!IsForceRematSgpr)
            return false;
          for (RematNode &Node : SRematList) {
            SRematMap[Node.Reg] = Node;
          }
          // Find local one def one use candidates.
          for (MachineInstr &MI : *MBB) {
            if (MI.isDebugInstr())
              continue;
            if (MI.getDesc().NumDefs != 1)
              continue;
            MachineOperand &DstMO = MI.getOperand(0);
            Register Reg = DstMO.getReg();
            if (!SIRI->isSGPRReg(MRI, Reg))
              continue;
            if (!MRI.hasOneNonDBGUse(Reg))
              continue;
            if (!MRI.hasOneDef(Reg))
              continue;
            if (Reg.isPhysical())
              continue;
            MachineInstr &UseMI = *MRI.use_instr_nodbg_begin(Reg);
            if (UseMI.getParent() != MBB)
              continue;
            int Gain = rematGain(&MI, Reg, MRI, SIRI,
                                 /*IsVGPR*/ false);
            if (Gain > 0) {
              // Skip case when DefMI has implicit define which used by UseMI.
              if (isImplicitDefUse(&MI, &UseMI)) {
                continue;
              }
              RematNode Node = {Reg, &MI, (unsigned)Gain >> 5};
              Node.InsertPointMI = &UseMI;
              Node.Kind = RematNode::RematKind::OneDefOneUse;
              SRematMap[Reg] = Node;
              SharedReducedSize += Node.Size;
            }
          }
        }
        NewRematSCnt = RematSCnt - NewRematSCnt - SharedReducedSize;
      }
    }
    // If works, continue.

    // Collect live range from hot inst.
    // find common live range in hot insts.
    // Remat these common live range.
    // Apply the remat.

    int NewRematVCnt = 0;
    if (RematVCnt > 0) {
      // TODO: V remat.
    }

    bool NeedSRemat = RematSCnt > 0;
    bool NeedVRemat = RematVCnt > 0;
    // If sgpr spill, always do remat.
    bool IsSRematOK =
        (NewRematSCnt <= 0 && !SRematMap.empty()) || IsForceRematSgpr;
    bool IsVRematOK =
        (Status.NotBalance || NewRematVCnt <= 0) && !VRematMap.empty();
    if (NeedSRemat && NeedVRemat) {
      if (IsVRematOK && IsSRematOK) {
        IsUpdated = true;
      } else if (IsSGPRSpill) {
        IsUpdated = true;
      }
    } else if (NeedSRemat) {
      if (IsSRematOK) {
        IsUpdated = true;
      }
    } else if (NeedVRemat) {
      if (IsVRematOK) {
        IsUpdated = true;
      }
    }
    // TODO: what to do when cannot reach target?
    if (NewRematSCnt > 0) {
      if ((unsigned)NewRematSCnt <= NearTargetRegLimit) {
        IsNearTarget = true;
      } else {
        if (!IsSGPRSpill)
          return false;
      }
    }
  }

  if (SRematMap.empty() && VRematMap.empty()) {
    return IsUpdated;
  }

  if (!SRematMap.empty()) {
    IsUpdated = true;
    applyRemat(Remat, SRematMap, HotBlocks, DT, SlotIndexes, MRI, SIRI, SIII,
               MF);
    LLVM_DEBUG(llvm::dbgs() << "after hotremat"; MF.print(dbgs()););
  }

  // Balance between vector and scalar if possible.
  return IsUpdated;
}

namespace {
bool isPhyRegUniqueDef(unsigned Reg, const MachineRegisterInfo &MRI) {
  DenseSet<MachineInstr *> DefMIs;
  for (MachineInstr &DefMI : MRI.def_instructions(Reg)) {
    // skip implicit def.
    if (DefMI.getOpcode() == AMDGPU::IMPLICIT_DEF)
      continue;
    DefMIs.insert(&DefMI);
  }
  return DefMIs.size() == 1;
}

static bool isImplicitUseOfReg(const MachineOperand &MO, unsigned Reg) {
  if (!MO.isImplicit() || !MO.isUse() || !MO.isReg()) {
    return false;
  }

  return MO.getReg() == Reg;
}

static bool isSafeRematCandidateUser(const MachineInstr *UseMI,
                                     const SIInstrInfo *SIII) {
  // Make sure UseMI is not wqm like sample.
  if (SIII->isWQM(UseMI->getOpcode()))
    return false;
  if (UseMI->getOpcode() == AMDGPU::PHI)
    return false;

  return true;
}

static bool isConvergent(Remat *Remat, const MachineInstr &MI) {
  return MI.isConvergent() &&
         // This flag is set on readfirstlane's to indicate that they
         // are redundant (the value being read is already uniform).
         // Normally, readfirstlanes are convergent, because different exec
         // will cause a different value to be read; a known uniform
         // readfirstlane is safe to move or clone and not actually convergent.
         !Remat->TotalUniformInsts.count(&MI);
}

bool isSafeCandidate(Remat *Remat, Register Reg, const MachineRegisterInfo &MRI,
                     const SIRegisterInfo *SIRI, const SIInstrInfo *SIII,
                     bool IsSink) {
  if (Reg.isPhysical())
    return false;
  bool IsVGPR = SIRI->isVGPR(MRI, Reg);

  MachineInstr *DefMI = MRI.getUniqueVRegDef(Reg);
  if (!DefMI)
    return false;
  if (DefMI->getOpcode() == AMDGPU::PHI)
    return false;

  // Skip convergent.
  if (isConvergent(Remat, *DefMI))
    return false;

  // Skip inst has more than 1 def.
  if (DefMI->getDesc().NumDefs > 1)
    return false;

  unsigned OpNum = DefMI->getNumOperands();

  // Only move DefMI which all operand is unique def.
  for (unsigned I = 0; I < OpNum; I++) {
    MachineOperand &Op = DefMI->getOperand(I);
    if (!Op.isReg())
      continue;
    Register OpReg = Op.getReg();
    if (isImplicitUseOfReg(Op, AMDGPU::EXEC) ||
        isImplicitUseOfReg(Op, AMDGPU::EXEC_LO))
      continue;
    if (isImplicitUseOfReg(Op, AMDGPU::MODE))
      continue;
    if (isImplicitUseOfReg(Op, AMDGPU::M0) && isPhyRegUniqueDef(OpReg, MRI))
      continue;
    // Alow unused scc define.
    if (Op.isImplicit() && Op.isDead() && Op.isDef())
      continue;
    if (OpReg.isPhysical())
      return false;
    if (!MRI.getUniqueVRegDef(OpReg) &&
        !llvm::IsSub0Sub1SingleDef(OpReg, MRI)) {
      return false;
    }
  }

  if (IsVGPR && IsSink) {
    // Skip mem related inst.
    if (DefMI->mayLoadOrStore()) {
      return false;
    }

    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
      if (!isSafeRematCandidateUser(&UseMI, SIII))
        return false;
    }
  }

  return true;
}

std::vector<SubExp> buildSubExpFromCandidates(
    Remat *Remat, GCNRPTracker::LiveRegSet &Candidates, MachineBasicBlock *MBB,
    const SIRegisterInfo *SIRI, const SIInstrInfo *SIII,
    const MachineRegisterInfo &MRI, SlotIndexes *SlotIndexes,
    GCNRPTracker::LiveRegSet &UnusedPassThrus, bool AllowPartialUseInSubExp) {
  InstSet CandidateDefs;
  DenseSet<unsigned> RemovedCandidates;
  std::vector<unsigned> CandidateRegs;
  CandidateRegs.reserve(Candidates.size());
  for (auto It : Candidates) {
    unsigned Reg = It.first;
    CandidateRegs.emplace_back(Reg);
  }
  // Sort candidate by defMI order to make sure defMI has dependent check after
  // all its dependent node.
  std::sort(CandidateRegs.begin(), CandidateRegs.end(),
            [&MRI, &SlotIndexes](const unsigned A, unsigned B) {
              MachineInstr *MIa = MRI.getUniqueVRegDef(A);

              MachineInstr *MIb = MRI.getUniqueVRegDef(B);
              // Later instr first.
              return !SlotIndex::isEarlierInstr(
                  SlotIndexes->getInstructionIndex(*MIa),
                  SlotIndexes->getInstructionIndex(*MIb));
            });

  // If Candidate def has user in MBB, add It when allow partial candidates.
  // And the subExp has the define could only be clone, cannot move cross blocks
  // because user in MBB.
  DenseSet<MachineInstr *> PartialCandidates;
  LLVM_DEBUG(dbgs() << "\nCandidate Defs:\n";);
  for (unsigned Reg : CandidateRegs) {
    MachineInstr *MI = MRI.getUniqueVRegDef(Reg);
    bool IsHasNoCandidatesSameBlockUser = false;
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
      if (UseMI.getParent() == MI->getParent()) {
        if (UseMI.getNumExplicitDefs() == 1) {
          // Skip user which already in Candidates.
          Register UserDefReg = UseMI.getOperand(0).getReg();
          if (Candidates.count(UserDefReg) > 0 &&
              RemovedCandidates.count(UserDefReg) == 0)
            continue;
        }
        if (!AllowPartialUseInSubExp)
          IsHasNoCandidatesSameBlockUser = true;
        else
          PartialCandidates.insert(MI);
        break;
      }
    }
    if (IsHasNoCandidatesSameBlockUser) {
      RemovedCandidates.insert(Reg);
      continue;
    }
    LLVM_DEBUG(MI->dump());
    CandidateDefs.insert(MI);
  }
  LLVM_DEBUG(dbgs() << "\nCandidate Defs End\n";);

  if (CandidateDefs.empty())
    return std::vector<SubExp>();
  for (unsigned Reg : RemovedCandidates) {
    UnusedPassThrus[Reg] = Candidates[Reg];
    Candidates.erase(Reg);
  }

  // iterate MBB backward.
  // add inst which only used for candidate defines.
  for (auto It = MBB->rbegin(); It != MBB->rend(); It++) {
    MachineInstr &MI = *It;
    if (CandidateDefs.count(&MI) > 0) {
      continue;
    }

    if (isConvergent(Remat, MI))
      continue;
    // Skip if MI is not safe to move.
    if (MI.getNumDefs() != 1) {
      // allow to move unused implicit def.
      bool IsDeadImplictDef = false;
      for (MachineOperand &MO : MI.implicit_operands()) {
        if (!MO.isReg())
          continue;
        if (!MO.isDef())
          continue;
        IsDeadImplictDef = MO.isDead();
      }
      if (!IsDeadImplictDef)
        continue;
    }

    unsigned Reg = -1;
    for (MachineOperand &MO : MI.operands()) {
      if (!MO.isReg())
        continue;
      if (!MO.isDef())
        continue;
      Reg = MO.getReg();
      break;
    }

    if (!isSafeCandidate(Remat, Reg, MRI, SIRI, SIII, /*IsSink*/ true))
      continue;

    // If all users of MI are in candidate defs, add MI into candidate defs.
    // If part of user of MI is in candidate defs, add MI into candidate defs
    // when allow partialUse.
    bool IsAllUserInCandidate = true;
    bool IsHasCandidateUser = false;
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
      if (CandidateDefs.count(&UseMI) == 0)
        IsAllUserInCandidate = false;
      else
        IsHasCandidateUser = true;
    }
    if (!IsHasCandidateUser)
      continue;
    if (!IsAllUserInCandidate) {
      if (!AllowPartialUseInSubExp)
        continue;
      PartialCandidates.insert(&MI);
    }

    CandidateDefs.insert(&MI);
  }

  // Collect input for CandidateDefs.
  GCNRPTracker::LiveRegSet CandidateInput;
  for (MachineInstr *MI : CandidateDefs) {
    for (MachineOperand &MO : MI->operands()) {
      if (!MO.isReg())
        continue;
      if (MO.isDef())
        continue;
      Register Reg = MO.getReg();
      if (MO.isImplicit() && Reg.isPhysical())
        continue;

      MachineInstr *DefMI = MRI.getUniqueVRegDef(Reg);
      assert((DefMI || llvm::IsSub0Sub1SingleDef(Reg, MRI)) &&
             "UseMI should be safe to move");
      if (DefMI && CandidateDefs.count(DefMI) > 0)
        continue;
      // Add to input.
      CandidateInput[Reg] |= llvm::getRegMask(MO, MRI);
    }
  }

  // Build defs in order.
  std::vector<MachineInstr *> Defs;
  Defs.reserve(CandidateDefs.size());
  for (MachineInstr &MI : *MBB) {
    if (CandidateDefs.count(&MI) == 0)
      continue;
    Defs.emplace_back(&MI);
  }

  LLVM_DEBUG(dbgs() << "\nFinished Candidate Defs:\n"; for (MachineInstr *MI
                                                            : Defs) {
    MI->dump();
  } dbgs() << "\nFinished Candidate Defs End\n";);

  // Build SubExp with CandidateDefs as Nodes, CandidateInput as input
  // Candidates as output.
  ExpDag Dag(MRI, SIRI, SIII, /*IsJoinInput*/ true);
  Dag.build(CandidateInput, Candidates, Defs);
  if (AllowPartialUseInSubExp) {
    for (auto &SubExp : Dag.SubExps) {
      for (auto *MI : SubExp.SUnits) {
        if (PartialCandidates.count(MI)) {
          SubExp.IsCloneOnly = true;
          break;
        }
      }
    }
  }
  return Dag.SubExps;
}

std::vector<SubExp> buildSubExpFromCandidatesTopBottom(
    Remat *Remat, GCNRPTracker::LiveRegSet &Candidates, MachineBasicBlock *MBB,
    const SIRegisterInfo *SIRI, const SIInstrInfo *SIII,
    const MachineRegisterInfo &MRI) {
  InstSet CandidateDefs;

  LLVM_DEBUG(dbgs() << "\nCandidate Defs:\n";);
  for (auto It : Candidates) {
    unsigned Reg = It.first;
    MachineInstr *MI = MRI.getUniqueVRegDef(Reg);

    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
      if (isConvergent(Remat, UseMI))
        continue;
      MachineBasicBlock *UseMBB = UseMI.getParent();
      if (UseMBB == MI->getParent())
        continue;
      assert(UseMBB == MBB && "block mismatch");
      // If all operands in CandidateRegs, add to candidateDefs.
      bool IsHasOpRegNotInCandidates = false;
      for (MachineOperand &MO : UseMI.operands()) {
        if (!MO.isReg())
          continue;
        if (MO.isDef())
          continue;
        Register OpReg = MO.getReg();
        if (MO.isImplicit() && OpReg.isPhysical())
          continue;
        if (Candidates.count(OpReg) == 0) {
          IsHasOpRegNotInCandidates = true;
          break;
        }
      }
      if (IsHasOpRegNotInCandidates)
        continue;

      LLVM_DEBUG(UseMI.dump());
      CandidateDefs.insert(&UseMI);
    }
  }
  LLVM_DEBUG(dbgs() << "\nCandidate Defs End\n";);

  if (CandidateDefs.empty())
    return std::vector<SubExp>();

  // iterate MBB.
  GCNRPTracker::LiveRegSet LocalCandidates = Candidates;
  // add inst which only used by candidate defines.
  for (auto It = MBB->begin(); It != MBB->end(); It++) {
    MachineInstr &MI = *It;
    if (CandidateDefs.count(&MI) > 0) {
      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isReg())
          continue;
        if (!MO.isDef())
          continue;
        Register Reg = MO.getReg();
        if (Reg.isPhysical())
          continue;
        LocalCandidates[Reg];
      }
      continue;
    }

    // Skip if MI is not safe to move.
    if (isConvergent(Remat, MI))
      continue;

    if (MI.getNumDefs() != 1)
      continue;

    if (MI.mayLoadOrStore()) {
      continue;
    }

    unsigned Reg = -1;
    for (MachineOperand &MO : MI.operands()) {
      if (!MO.isReg())
        continue;
      if (!MO.isDef())
        continue;
      Reg = MO.getReg();
      break;
    }

    // Still use bsink to skip mem load/store.
    // if (!isSafeCandidate(Reg, MRI, SIRI, SIII, /*IsSink*/true))
    //  continue;

    // If all user of MI is in candidate defs, add MI into candidate defs.
    bool IsAllOperandInCandidate = true;
    for (MachineOperand &MO : MI.operands()) {
      if (!MO.isReg())
        continue;
      if (MO.isDef())
        continue;
      Register OpReg = MO.getReg();
      if (LocalCandidates.count(OpReg))
        continue;

      if (MO.isImplicit() &&
          (OpReg == AMDGPU::EXEC || OpReg == AMDGPU::EXEC_LO))
        continue;
      if (OpReg.isPhysical()) {
        IsAllOperandInCandidate = false;
        break;
      }
      MachineInstr *OpMI = MRI.getUniqueVRegDef(OpReg);
      if (!OpMI) {
        IsAllOperandInCandidate = false;
        break;
      }
      if (CandidateDefs.count(OpMI) == 0) {
        IsAllOperandInCandidate = false;
        break;
      }
      if (MO.isTied())
        continue;
    }
    if (!IsAllOperandInCandidate)
      continue;
    LLVM_DEBUG(llvm::dbgs() << "Add local candidates:";
               pressure::print_reg(Reg, MRI, SIRI, llvm::dbgs()););
    LocalCandidates[Reg];
    CandidateDefs.insert(&MI);
  }

  // Collect input for CandidateDefs.
  GCNRPTracker::LiveRegSet CandidateInput;
  for (MachineInstr *MI : CandidateDefs) {
    for (MachineOperand &MO : MI->operands()) {
      if (!MO.isReg())
        continue;
      if (MO.isDef())
        continue;
      Register Reg = MO.getReg();
      if (MO.isImplicit() && (Reg == AMDGPU::EXEC || Reg == AMDGPU::EXEC_LO))
        continue;
      if (Reg.isPhysical())
        continue;
      MachineInstr *DefMI = MRI.getUniqueVRegDef(Reg);
      if (!DefMI) {
        // Skip local def which is not unique.
        if (MO.isTied())
          continue;
        if (Candidates.count(Reg) == 0 && LocalCandidates.count(Reg) != 0)
          continue;
      }
      assert((DefMI || llvm::IsSub0Sub1SingleDef(Reg, MRI)) &&
             "UseMI should be safe to move");
      if (DefMI && CandidateDefs.count(DefMI) > 0)
        continue;
      // Add to input.
      CandidateInput[Reg] = llvm::getRegMask(MO, MRI);
    }
  }

  // Build defs in order.
  std::vector<MachineInstr *> Defs;
  Defs.reserve(CandidateDefs.size());
  for (MachineInstr &MI : *MBB) {
    if (CandidateDefs.count(&MI) == 0)
      continue;
    Defs.emplace_back(&MI);
  }

  LLVM_DEBUG(dbgs() << "\nFinished Candidate Defs:\n"; for (MachineInstr *MI
                                                            : Defs) {
    MI->dump();
  } dbgs() << "\nFinished Candidate Defs End\n";);

  LLVM_DEBUG(dbgs() << "\nLocalCandidates:\n"; for (auto It
                                                    : LocalCandidates) {
    pressure::print_reg(It.first, MRI, SIRI, llvm::dbgs());
  } dbgs() << "\nLocalCandidates End\n";);
  // Make sure all input reg are uniqueDef.
  // Input is Candidates, output is?
  // Build SubExp with CandidateDefs as Nodes, CandidateInput as input
  // Candidates as output.
  ExpDag Dag(MRI, SIRI, SIII, /*IsJoinInput*/ true);
  Dag.build(Candidates, LocalCandidates, Defs);
  return Dag.SubExps;
}

void printVreg(Register Reg, const MachineRegisterInfo &MRI) {
  if (Reg.isVirtual()) {
    StringRef Name = MRI.getVRegName(Reg);
    if (Name != "") {
      dbgs() << '%' << Name;
    } else {
      dbgs() << '%' << Register::virtReg2Index(Reg);
    }
  }
}

MachineBasicBlock *findTargetBlock(unsigned Reg, MachineBasicBlock *FromBB,
                                   const MachineRegisterInfo &MRI,
                                   MachineDominatorTree *DT) {
  BlockSet UserBlocks;
  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
    MachineBasicBlock *UserBB = UseMI.getParent();
    // Skip current BB.
    if (UserBB != FromBB)
      UserBlocks.insert(UserBB);
    else
      // When has user in FromBB, userBlock will be FromBB.
      return nullptr;
  }
  if (UserBlocks.empty())
    return nullptr;
  MachineBasicBlock *UserBlock = nearestCommonDominator(DT, UserBlocks);
  if (!DT->dominates(FromBB, UserBlock)) {
    return nullptr;
  }
  if (UserBlock == FromBB)
    return nullptr;
  return UserBlock;
}

void applySubExpMoveNearUser(SubExp &Exp, const MachineRegisterInfo &MRI,
                             MachineDominatorTree *DT,
                             SlotIndexes *SlotIndexes) {
  // Move from bottom.
  MachineBasicBlock *FromBB = Exp.FromBB;
  for (auto It = Exp.SUnits.rbegin(); It != Exp.SUnits.rend(); It++) {
    MachineInstr *DefMI = *It;
    if (DefMI->getNumExplicitDefs() != 1)
      continue;

    Register Reg = DefMI->getOperand(0).getReg();
    MachineBasicBlock *ToBB = findTargetBlock(Reg, FromBB, MRI, DT);
    if (!ToBB)
      continue;

    // Do not overwrite a live scc.
    MachineBasicBlock::iterator InsertPoint =
        ToBB->SkipPHIsAndLabels(ToBB->begin());
    if (willSmashSccAtLocation(DefMI, ToBB, InsertPoint))
      continue;

    DefMI->removeFromParent();
    assert(!llvm::isExecUpdateForControlFlow(*InsertPoint) &&
           "invalid insert point");
    ToBB->insert(InsertPoint, DefMI);
    // Debug insts don't need slot index.
    if (DefMI->isDebugInstr())
      continue;
    // Update slot index.
    SlotIndexes->removeSingleMachineInstrFromMaps(*DefMI);
    SlotIndexes->insertMachineInstrInMaps(*DefMI);
  }
}

void applySubExpMoveNearDefine(SubExp &Exp, MachineRegisterInfo &MRI,
                               SlotIndexes *SlotIndexes,
                               const SIInstrInfo *SIII,
                               const SIRegisterInfo *SIRI) {
  // Move from top.
  // Find lowest input def.
  MachineBasicBlock *ToBB = Exp.ToBB;
  assert(!ToBB->empty() && "ToBB have instructions for define of input nodes");
  auto Terminator = ToBB->getFirstTerminator();
  if (Terminator == ToBB->end() && ToBB->succ_size() == 1) {
    MachineInstr &EndMI = *ToBB->rbegin();
    if (SIII->isSchedulingBoundary(EndMI, ToBB, *ToBB->getParent()))
      // Insert before the scheduling boundary instruction.
      Terminator = EndMI.getIterator();
    else
      // No boundary so just insert inst at the end of the block.
      Terminator = ToBB->end();
  }

  Terminator = adjustInsertPointForSubExpToAvoidSccSmash(Exp, ToBB, Terminator,
                                                         MRI, SIRI, SIII);

  for (auto It = Exp.SUnits.begin(); It != Exp.SUnits.end(); It++) {
    MachineInstr *DefMI = *It;
    if (DefMI->getNumExplicitDefs() != 1)
      continue;
    if (SIII->isEXP(DefMI->getOpcode()))
      continue;
    if (DefMI->mayStore())
      continue;
    // Find def for DefMI operands as insert point.
    DefMI->removeFromParent();
    ToBB->insert(Terminator, DefMI);

    // Debug insts don't need slot index.
    if (DefMI->isDebugInstr())
      continue;
    // Update slot index.
    SlotIndexes->removeSingleMachineInstrFromMaps(*DefMI);
    SlotIndexes->insertMachineInstrInMaps(*DefMI);
  }
}

DenseSet<MachineInstr *> buildCloneSet(ExpDag &Dag,
                                       DenseSet<SUnit *> &DagBottoms,
                                       GCNRPTracker::LiveRegSet &UsedOutput) {
  DenseSet<MachineInstr *> CopySet;
  for (auto It = Dag.SUnits.rbegin(); It != Dag.SUnits.rend(); It++) {
    SUnit &SU = *It;
    // Skip non-inst node.
    if (!SU.isInstr())
      continue;
    MachineInstr *MI = SU.getInstr();
    if (DagBottoms.find(&SU) != DagBottoms.end()) {
      bool IsUsed = false;
      // For bottom SU, if in usedOutput, add to copySet;
      for (MachineOperand &DefMO : MI->defs()) {
        if (!DefMO.isReg())
          continue;
        Register Reg = DefMO.getReg();
        if (UsedOutput.count(Reg) > 0) {
          IsUsed = true;
          break;
        }
      }
      if (IsUsed) {
        CopySet.insert(MI);
        continue;
      }
      // bottom SU may still have succNode when It used both inExp and outExp.
      // So continue check succNode.
    }

    // If any SuccNode is in copySet, add to copySet.
    bool IsSuccCopied = false;
    for (SDep &SucDep : SU.Succs) {
      SUnit *SucSU = SucDep.getSUnit();
      MachineInstr *SuccMI = SucSU->getInstr();
      if (CopySet.count(SuccMI) > 0) {
        IsSuccCopied = true;
        break;
      }
    }
    if (IsSuccCopied)
      CopySet.insert(MI);
  }
  return CopySet;
}

void updateUsers(SmallVector<MachineInstr *, 2> &UserMIs,
                 DenseMap<unsigned, unsigned> &RegMap) {

  for (MachineInstr *UserMI : UserMIs) {
    for (MachineOperand &MO : UserMI->uses()) {
      if (!MO.isReg())
        continue;
      Register Reg = MO.getReg();
      auto It = RegMap.find(Reg);
      if (It == RegMap.end())
        continue;
      unsigned NewReg = It->second;
      MO.setReg(NewReg);
    }
  }
}

struct HotBlock {
  MachineBasicBlock *MBB = nullptr;
  GCNRPTracker::LiveRegSet InputLive;
  std::pair<unsigned, unsigned> MaxPressures;
  // Info about vmemLd.
  int VmemLdInputSize;
  int VmemLdOutputSize;
};

DenseMap<MachineBasicBlock *, BlockSet> reduceClonedMBBs(
    SubExp &Exp,
    MapVector<MachineBasicBlock *, SmallVector<MachineInstr *, 2>> &UserBlocks,
    DenseMap<MachineBasicBlock *, GCNRPTracker::LiveRegSet> &UserBlocksLiveRegs,
    std::vector<HotBlock> &HotBlocks, MachineDominatorTree *DT) {
  // Collect hot blocks which Exp is live in.
  DenseSet<MachineBasicBlock *> HotBlockSet;
  for (HotBlock &HotBlock : HotBlocks) {
    for (unsigned Reg : Exp.BottomRegs) {
      if (HotBlock.InputLive.count(Reg)) {
        HotBlockSet.insert(HotBlock.MBB);
        break;
      }
    }
  }

  // For userBlocks which dominate all hotBlocks, don't need to clone because
  // the value not cross hotBlocks when later blocks are cloned.
  // For userBlocks which dominated by all hotBlocks, they could share clones
  // because once after hot block, the pressure is OK.
  DenseSet<MachineBasicBlock *> AfterHotRangeMBBs;
  for (auto It : UserBlocksLiveRegs) {
    MachineBasicBlock *MBB = It.first;
    // Always clone in hot block.
    if (HotBlockSet.count(MBB))
      continue;

    bool IsDomAllHotBlocks = true;
    bool IsDomedByAllHotBlocks = true;
    for (MachineBasicBlock *HotMBB : HotBlockSet) {
      if (!DT->dominates(MBB, HotMBB)) {
        IsDomAllHotBlocks = false;
      }
      if (!DT->dominates(HotMBB, MBB)) {
        IsDomedByAllHotBlocks = false;
      }
      if (!IsDomAllHotBlocks && !IsDomedByAllHotBlocks) {
        break;
      }
    }
    if (IsDomAllHotBlocks) {
      UserBlocks.erase(MBB);
    } else if (IsDomedByAllHotBlocks) {
      AfterHotRangeMBBs.insert(MBB);
    }
  }

  // Split after hotRange block set by domtree.
  DenseMap<MachineBasicBlock *, BlockSet> DomMap;
  if (!AfterHotRangeMBBs.empty()) {
    for (MachineBasicBlock *MBB : AfterHotRangeMBBs) {
      for (MachineBasicBlock *MBB2 : AfterHotRangeMBBs) {
        if (MBB == MBB2)
          continue;
        if (DT->dominates(MBB, MBB2)) {
          auto &Dom = DomMap[MBB];
          Dom.insert(MBB2);
          auto &Dom2 = DomMap[MBB2];
          Dom.insert(Dom2.begin(), Dom2.end());
        }
      }
    }
    for (MachineBasicBlock *MBB : AfterHotRangeMBBs) {
      auto &UsedOutput = UserBlocksLiveRegs[MBB];
      auto &Dom = DomMap[MBB];
      for (MachineBasicBlock *DomedMBB : Dom) {
        // Merge domed use to MBB use.
        mergeLiveRegSet(UsedOutput, UserBlocksLiveRegs[DomedMBB]);
        // Remove domedMBB.
        DomMap.erase(DomedMBB);
        UserBlocksLiveRegs.erase(DomedMBB);
      }
    }
  }

  return DomMap;
}

void applySubExpCloneNearUser(SubExp &Exp, std::vector<HotBlock> &HotBlocks,
                              MachineDominatorTree *DT,
                              MachineRegisterInfo &MRI,
                              SlotIndexes *SlotIndexes, const SIInstrInfo *SIII,
                              const SIRegisterInfo *SIRI) {
  MapVector<MachineBasicBlock *, SmallVector<MachineInstr *, 2>> UserBlocks;
  DenseMap<MachineBasicBlock *, GCNRPTracker::LiveRegSet> UserBlocksLiveRegs;
  for (unsigned Reg : Exp.BottomRegs) {
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
      MachineBasicBlock *UserBB = UseMI.getParent();
      // Skip current BB.
      if (UserBB == Exp.FromBB)
        continue;

      UserBlocks[UserBB].emplace_back(&UseMI);
      auto &UserLives = UserBlocksLiveRegs[UserBB];
      for (MachineOperand &MO : UseMI.uses()) {
        if (!MO.isReg())
          continue;
        Register UseReg = MO.getReg();
        if (Reg != UseReg)
          continue;
        UserLives[Reg] |= getRegMask(MO, MRI);
      }
    }
  }
  // Build dag for SubExp to help remove unused inst when clone.
  ExpDag Dag(MRI, SIRI, SIII, /*IsJoinInput*/ true);
  Dag.build(Exp.InputLive, Exp.OutputLive, Exp.SUnits);
  DenseSet<SUnit *> DagBottoms;
  for (SUnit &SU : Dag.SUnits) {
    if (!SU.isInstr())
      continue;
    if (SU.NumSuccs == 0) {
      DagBottoms.insert(&SU);
    } else {
      MachineInstr *MI = SU.getInstr();
      // Add SU which def value in Exp.outputLive.
      for (MachineOperand &DefMO : MI->defs()) {
        if (!DefMO.isReg())
          continue;
        Register Reg = DefMO.getReg();
        if (Exp.BottomRegs.count(Reg) > 0) {
          DagBottoms.insert(&SU);
          break;
        }
      }
    }
  }

  // For userBlocks which dominate all hotBlocks, don't need to clone because
  // the value not cross hotBlocks when later blocks are cloned.
  // For userBlocks which dominated by all hotBlocks, they could share clones
  // because once after hot block, the pressure is OK.
  DenseMap<MachineBasicBlock *, BlockSet> DomMap =
      reduceClonedMBBs(Exp, UserBlocks, UserBlocksLiveRegs, HotBlocks, DT);

  // Sort to make stable order.
  std::sort(
      UserBlocks.begin(), UserBlocks.end(),
      [](std::pair<MachineBasicBlock *, SmallVector<MachineInstr *, 2>> &It0,
         std::pair<MachineBasicBlock *, SmallVector<MachineInstr *, 2>> &It1) {
        return It0.first->getNumber() < It1.first->getNumber();
      });

  const bool IsModifiesScc = Exp.modifiesRegister(AMDGPU::SCC, SIRI);

  // Clone for each userBlocks. Not share clone thru dom tree which cannot help
  // reg pressure.
  for (auto It : UserBlocks) {
    MachineBasicBlock *MBB = It.first;
    // Skip MBB which share clone from other MBBs.
    if (UserBlocksLiveRegs.count(MBB) == 0)
      continue;
    auto &UsedOutput = UserBlocksLiveRegs[MBB];
    auto CopySet = buildCloneSet(Dag, DagBottoms, UsedOutput);
    // Clone to MBB.
    // Create new regs first.
    DenseMap<unsigned, unsigned> RegMap;
    auto InsertPtr = MBB->getFirstNonPHI();
    // If Exp has scc read/write, make sure MBB not have scc in liveins.
    if (IsModifiesScc && llvm::IsSccLiveAt(MBB, InsertPtr))
      continue;
    MachineFunction *MF = MBB->getParent();
    for (auto It = Exp.SUnits.begin(); It != Exp.SUnits.end(); It++) {
      MachineInstr *DefMI = *It;
      // Not clone if already in MBB.
      if (DefMI->getParent() == MBB)
        continue;
      // Not clone if not used for MBB.
      if (CopySet.count(DefMI) == 0)
        continue;

      auto ClonedMI =
          BuildMI(*MBB, InsertPtr, DefMI->getDebugLoc(), DefMI->getDesc());

      for (MachineOperand &Def : DefMI->defs()) {
        Register Reg = Def.getReg();
        if (Reg.isPhysical()) {
          if (Def.isImplicit())
            continue;
          ClonedMI.addDef(Reg, 0, Def.getSubReg());
        } else {
          Register NewReg = MRI.createVirtualRegister(MRI.getRegClass(Reg));
          RegMap[Reg] = NewReg;
          ClonedMI.addDef(NewReg, 0, Def.getSubReg());
        }
      }

      for (MachineOperand &MO : DefMI->uses()) {
        if (MO.isReg()) {
          Register Reg = MO.getReg();
          if (Reg.isPhysical()) {
            if (MO.isImplicit())
              continue;
            ClonedMI.addReg(Reg, 0, MO.getSubReg());
          } else {
            auto It = RegMap.find(Reg);
            if (It == RegMap.end()) {
              ClonedMI.addReg(Reg, 0, MO.getSubReg());
            } else {
              ClonedMI.addReg(It->second, 0, MO.getSubReg());
            }
          }
        } else {
          ClonedMI.add(MO);
        }
      }

      MachineInstr *NewDef = ClonedMI.getInstr();
      SlotIndexes->insertMachineInstrInMaps(*NewDef);
      // Set mem operand
      for (MachineMemOperand *MO : DefMI->memoperands()) {
        NewDef->addMemOperand(*MF, MO);
      }
    }

    // update users in MBB.
    SmallVector<MachineInstr *, 2> &UserMIs = It.second;
    updateUsers(UserMIs, RegMap);

    // update users in dom MBBs.
    auto DomMapIt = DomMap.find(MBB);
    if (DomMapIt != DomMap.end()) {
      for (MachineBasicBlock *UpdateMBB : DomMapIt->second) {
        SmallVector<MachineInstr *, 2> &UserMIs = UserBlocks[UpdateMBB];
        updateUsers(UserMIs, RegMap);
      }
    }
  }
}

void applySubExpCloneNearUserInBlock(
    SubExp &Exp,
    DenseMap<MachineBasicBlock *, MachineInstr *> &InBlockHotVInstMap,
    DenseMap<MachineBasicBlock *, MachineInstr *> &InBlockHotSInstMap,
    MachineRegisterInfo &MRI, SlotIndexes *SlotIndexes,
    const SIRegisterInfo *SIRI) {
  MachineBasicBlock *MBB = Exp.FromBB;
  MachineFunction *MF = MBB->getParent();
  MachineInstr *HotVMI = InBlockHotVInstMap[MBB];
  MachineInstr *HotSMI = InBlockHotSInstMap[MBB];
  // Exp is build with hotVMI or hotSMI, cannot mix.
  assert(!(HotVMI && HotSMI) && "cannot mix hot MI");
  MachineInstr *HotMI = HotVMI;
  if (!HotMI) {
    HotMI = HotSMI;
  }

  SlotIndex HotSlot = SlotIndexes->getInstructionIndex(*HotMI).getBaseIndex();
  const bool IsModifiesScc = Exp.modifiesRegister(AMDGPU::SCC, SIRI);

  for (unsigned Reg : Exp.BottomRegs) {

    SmallVector<MachineInstr *, 2> UseMIs;
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
      MachineBasicBlock *UserBB = UseMI.getParent();
      // Skip current BB.
      if (UserBB != Exp.FromBB)
        continue;
      // Skip inst in Exp.
      if (Exp.BottomRoots.find(&UseMI) != Exp.BottomRoots.end())
        continue;
      SlotIndex UseSlot =
          SlotIndexes->getInstructionIndex(UseMI).getBaseIndex();
      // Only clone for use after hot slot.
      if (UseSlot < HotSlot)
        continue;

      // Do not overwrite a live scc.
      if (IsModifiesScc && llvm::IsSccLiveAt(UserBB, &UseMI))
        continue;

      UseMIs.emplace_back(&UseMI);
    }
    if (UseMIs.empty())
      continue;
    DenseMap<unsigned, unsigned> RegMap;

    std::sort(UseMIs.begin(), UseMIs.end(),
              [&SlotIndexes](const MachineInstr *MIa, const MachineInstr *MIb) {
                return SlotIndexes->getInstructionIndex(*MIa).getBaseIndex() <
                       SlotIndexes->getInstructionIndex(*MIb).getBaseIndex();
              });
    auto InsertPtr = UseMIs.front()->getIterator();

    for (auto It = Exp.SUnits.begin(); It != Exp.SUnits.end(); It++) {
      MachineInstr *DefMI = *It;
      auto ClonedMI =
          BuildMI(*MBB, InsertPtr, DefMI->getDebugLoc(), DefMI->getDesc());

      for (MachineOperand &Def : DefMI->defs()) {
        Register Reg = Def.getReg();
        if (Reg.isPhysical()) {
          ClonedMI.addDef(Reg, 0, Def.getSubReg());
        } else {
          Register NewReg = MRI.createVirtualRegister(MRI.getRegClass(Reg));
          RegMap[Reg] = NewReg;
          ClonedMI.addDef(NewReg, 0, Def.getSubReg());
        }
      }

      for (MachineOperand &MO : DefMI->uses()) {
        if (MO.isReg()) {
          if (MO.isImplicit()) {
            continue;
          }
          Register Reg = MO.getReg();
          if (Reg.isPhysical()) {
            ClonedMI.addReg(Reg, 0, MO.getSubReg());
          } else {
            auto It = RegMap.find(Reg);
            if (It == RegMap.end()) {
              ClonedMI.addReg(Reg, 0, MO.getSubReg());
            } else {
              ClonedMI.addReg(It->second, 0, MO.getSubReg());
            }
          }
        } else {
          ClonedMI.add(MO);
        }
      }

      MachineInstr *NewDef = ClonedMI.getInstr();
      SlotIndexes->insertMachineInstrInMaps(*NewDef);
      // Set mem operand
      for (MachineMemOperand *MO : DefMI->memoperands()) {
        NewDef->addMemOperand(*MF, MO);
      }
    }
    // TODO: only clone to cross hot range.
    for (MachineInstr *UseMI : UseMIs) {
      for (MachineOperand &MO : UseMI->uses()) {
        if (!MO.isReg())
          continue;
        Register Reg = MO.getReg();
        auto It = RegMap.find(Reg);
        if (It == RegMap.end())
          continue;
        Register NewReg = It->second;
        MO.setReg(NewReg);
      }
    }
  }
}

bool isInLiveSet(unsigned Reg, LaneBitmask Mask,
                 const GCNRPTracker::LiveRegSet &Live) {
  auto It = Live.find(Reg);
  if (It == Live.end())
    return false;

  LaneBitmask LiveMask = It->second;
  return (LiveMask | Mask) == LiveMask;
}

unsigned getPacifistLevel(unsigned Reg,
                          DenseMap<MachineInstr *, unsigned> &PacifistLevels,
                          const MachineRegisterInfo &MRI) {
  unsigned Level = 0;
  for (MachineInstr &MI : MRI.def_instructions(Reg)) {
    auto It = PacifistLevels.find(&MI);
    if (It == PacifistLevels.end())
      continue;
    Level = It->second;
  }
  return Level;
}

bool hasInBlockDef(unsigned Reg, MachineBasicBlock *MBB,
                   const MachineRegisterInfo &MRI) {
  for (MachineInstr &Def : MRI.def_instructions(Reg)) {
    if (Def.getParent() != MBB)
      continue;
    return true;
  }
  return false;
}

MachineInstr *getInBlockUniqueDef(unsigned Reg, MachineBasicBlock *MBB,
                                  const GCNRPTracker::LiveRegSet &InputLive,
                                  const MachineRegisterInfo &MRI) {
  MachineInstr *DefMI = nullptr;
  // If live as input for MBB, cannot be unique def.
  if (InputLive.count(Reg))
    return DefMI;
  for (MachineInstr &Def : MRI.def_instructions(Reg)) {
    if (Def.getParent() != MBB)
      continue;
    if (DefMI) {
      // Not unique.
      DefMI = nullptr;
      break;
    }
    DefMI = &Def;
  }
  return DefMI;
}

bool isPassThru(unsigned Reg, const GCNRPTracker::LiveRegSet &InputLive,
                const GCNRPTracker::LiveRegSet &OutputLive) {
  return InputLive.count(Reg) && OutputLive.count(Reg);
}

// Instructions which only use imm/passThru reg/output only reg will not kill
// any live reg, so name them pacifist here.
bool collectPacifist(MachineInstr &MI,
                     const GCNRPTracker::LiveRegSet &InputLive,
                     const GCNRPTracker::LiveRegSet &OutputLive,
                     const MachineRegisterInfo &MRI) {
  // If has implicit def, not move.
  if (MI.getDesc().NumImplicitDefs != 0)
    return false;

  for (MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    if (MO.isDef())
      continue;

    Register Reg = MO.getReg();
    if (MO.isImplicit() &&
        (Reg == AMDGPU::EXEC || Reg == AMDGPU::EXEC_LO || Reg == AMDGPU::MODE))
      continue;
    if (Reg.isPhysical())
      return false;
    // The def for reg must be unique def in block or pass thru which not has
    // def in block. If not, It is not safe to move.
    if (!(nullptr != getInBlockUniqueDef(Reg, MI.getParent(), InputLive, MRI) ||
          (isPassThru(Reg, InputLive, OutputLive) &&
           !hasInBlockDef(Reg, MI.getParent(), MRI))))
      return false;

    LaneBitmask Mask = llvm::getRegMask(MO, MRI);

    if (isInLiveSet(Reg, Mask, OutputLive))
      continue;

    return false;
  }
  bool IsHasDef = false;
  for (MachineOperand &MO : MI.defs()) {
    Register Reg = MO.getReg();

    if (Reg.isPhysical())
      return false;

    if (nullptr == getInBlockUniqueDef(Reg, MI.getParent(), InputLive, MRI))
      return false;

    IsHasDef = true;
  }
  // If no def, It will not increase pressure, don't mark It.
  return IsHasDef;
}

static MachineInstr *findFirstAliasingLoadOrStoreInMBB(MachineInstr &MI,
                                                       MachineBasicBlock &MBB,
                                                       AliasAnalysis *AA) {
  if (MI.mayLoadOrStore()) {
    for (MachineBasicBlock::iterator I = MI.getIterator(), E = MBB.end();
         I != E; ++I) {
      const bool UseTBAA = false;
      if (MI.mayAlias(AA, *I, UseTBAA)) {
        return &*I;
      }
    }
  }

  return nullptr;
}

static MachineInstr *findPacifistInsertPoint(MachineInstr &MI,
                                             MachineBasicBlock &MBB,
                                             MachineRegisterInfo &MRI,
                                             AliasAnalysis *AA,
                                             SlotIndexes *SlotIndexes) {

  SmallVector<MachineInstr *, 2> Users;

  // We cannot move the pacifist instruction past any memory
  // op with which It aliases. Find the first instruction
  // that aliases the pacifist MI (if any) and add It to the list
  // of users. The sort() below will select the earliest user instruction.
  if (MachineInstr *AliasMI = findFirstAliasingLoadOrStoreInMBB(MI, MBB, AA)) {
    Users.push_back(AliasMI);
  }

  for (MachineOperand &MO : MI.defs()) {
    Register Reg = MO.getReg();
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
      if (&MBB != UseMI.getParent())
        continue;
      Users.emplace_back(&UseMI);
    }
  }
  if (Users.empty())
    return nullptr;

  std::sort(Users.begin(), Users.end(),
            [&SlotIndexes](const MachineInstr *MIa, MachineInstr *MIb) {
              // Early instr first.
              return SlotIndex::isEarlierInstr(
                  SlotIndexes->getInstructionIndex(*MIa),
                  SlotIndexes->getInstructionIndex(*MIb));
            });
  return Users.front();
}

// Pacifist inst will only add pressure since they don't kill.
// Try to hold them as late as possible in a MBB to help pressure.
bool tryHoldPacifist(MachineBasicBlock &MBB, LiveIntervals *LIS,
                     MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                     AliasAnalysis *AA, RematStatus &Status) {
  const GCNRPTracker::LiveRegSet InputLive = Status.MBBInputLiveMap[&MBB];
  const GCNRPTracker::LiveRegSet OutputLive = Status.MBBOutputLiveMap[&MBB];

  SmallVector<MachineInstr *, 32> PacifistList;
  LLVM_DEBUG(dbgs() << "pacifist begin\n");
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    if (collectPacifist(MI, InputLive, OutputLive, MRI)) {
      PacifistList.emplace_back(&MI);
      LLVM_DEBUG(MI.dump());
    }
  }
  LLVM_DEBUG(dbgs() << "pacifist end\n");

  SlotIndexes *SlotIndexes = LIS->getSlotIndexes();
  bool IsUpdated = false;

  // Move pacifist to its first user.
  // for (MachineInstr *MI : pacifistList) {
  for (auto It = PacifistList.rbegin(); It != PacifistList.rend(); It++) {
    MachineInstr *MI = *It;
    MachineInstr *FirstUser =
        findPacifistInsertPoint(*MI, MBB, MRI, AA, SlotIndexes);
    if (FirstUser == MI)
      continue;
    if (FirstUser == MI->getNextNode())
      continue;

    auto InsertPoint = MBB.getFirstInstrTerminator();
    if (FirstUser) {
      InsertPoint = FirstUser->getIterator();
    } else {
      // When there's no terminator.
      if (InsertPoint == MBB.end())
        InsertPoint--;
      else
        // BRANCH may have exec update before It.
        InsertPoint--;

      InsertPoint =
          llvm::skipDebugInstructionsBackward(InsertPoint, MBB.instr_begin());

      while ((InsertPoint->definesRegister(AMDGPU::EXEC, SIRI) ||
              InsertPoint->definesRegister(AMDGPU::EXEC_LO, SIRI)) &&
             InsertPoint != MI->getIterator()) {
        InsertPoint--;
        InsertPoint =
            llvm::skipDebugInstructionsBackward(InsertPoint, MBB.instr_begin());
      }
      if (InsertPoint == MI->getIterator())
        continue;
    }
    // Do not overwrite a live scc.
    if (willSmashSccAtLocation(MI, &MBB, InsertPoint))
      continue;
    MI->removeFromParent();
    MBB.insert(InsertPoint, MI);

    LIS->handleMove(*MI);
    IsUpdated = true;
  }

  return IsUpdated;
}

DenseMap<unsigned, MachineInstr *>
collectUniformVgprs(Remat *Remat, MachineFunction &MF, MachineRegisterInfo &MRI,
                    const SIRegisterInfo *SIRI) {
  DenseMap<unsigned, MachineInstr *> UniformMap;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      if (!Remat->TotalUniformInsts.count(&MI))
        continue;
      if (MI.getNumDefs() != 1)
        continue;
      unsigned DstIdx =
          AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::vdst);
      if (DstIdx == (unsigned)-1)
        continue;
      MachineOperand &DstMO = MI.getOperand(DstIdx);
      if (DstMO.getSubReg() != 0)
        continue;
      if (DstMO.isTied())
        continue;
      Register Reg = DstMO.getReg();
      if (MRI.getUniqueVRegDef(Reg) == nullptr)
        continue;

      auto *VRC = SIRI->getRegClassForReg(MRI, Reg);
      if (SIRI->isSGPRClass(VRC))
        continue;
      // TODO: Support more reg class.
      if (VRC != &AMDGPU::VGPR_32RegClass)
        continue;

      UniformMap[Reg] = &MI;
    }
  }
  return UniformMap;
}

// Try insert readfirstlane on uniform vgpr to turn It in sgpr and save vgpr
// pressure.
bool collectVToSCrossHotSpot(
    MachineBasicBlock &MBB, RematStatus &Status,
    DenseMap<unsigned, MachineInstr *> &UniformMap,
    SmallMapVector<unsigned, MachineInstr *, 4> &VToSMap, LiveIntervals *LIS) {
  unsigned VLimit = Status.TargetVLimit;
  unsigned SLimit = Status.TargetSLimit;
  auto &ST = MBB.getParent()->getSubtarget<GCNSubtarget>();

  GCNDownwardRPTracker Tracker(*LIS);

  bool IsUpdated = false;
  const auto InputLive = Status.MBBInputLiveMap[&MBB];
  Tracker.reset(*MBB.begin(), &InputLive);
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr()) {
      continue;
    }

    unsigned VPressure = Tracker.getPressure().getVGPRNum(ST.hasGFX90AInsts());
    unsigned SPressure = Tracker.getPressure().getMaxSGPR();

    SPressure += RegForVCC;

    Tracker.advance();
    // Sgpr bound, vtos cannot help.
    if (SPressure > SLimit)
      return false;

    if (VPressure <= VLimit) {
      continue;
    }

    // Try to make all possible vtos to reduce vpressure.
    const GCNRPTracker::LiveRegSet &CurLives = Tracker.getLiveRegs();
    for (auto It : CurLives) {
      unsigned Reg = It.first;
      auto UniformIt = UniformMap.find(Reg);
      if (UniformIt == UniformMap.end())
        continue;
      VToSMap[UniformIt->first] = UniformIt->second;
      IsUpdated = true;
    }
  }
  return IsUpdated;
}

// Return true if the user is outside of the def's loop.
static bool isCrossLoopUse(MachineInstr *Def, MachineInstr *User,
                           MachineLoopInfo *MLI) {
  MachineLoop *L = MLI->getLoopFor(Def->getParent());
  return L && !L->contains(User->getParent());
}

bool rematUniformVgprToSgpr(Remat *Remat, MachineFunction &MF,
                            RematStatus &Status,
                            std::vector<HotBlock> &HotBlocks,
                            LiveIntervals *LIS, MachineRegisterInfo &MRI,
                            const SIRegisterInfo *SIRI, const SIInstrInfo *SIII,
                            MachineLoopInfo *MLI) {
  DenseMap<unsigned, MachineInstr *> UniformVgprMap =
      collectUniformVgprs(Remat, MF, MRI, SIRI);

  SmallMapVector<unsigned, MachineInstr *, 4> VToSMap;

  for (auto &HotBlock : HotBlocks) {
    MachineBasicBlock &MBB = *HotBlock.MBB;
    collectVToSCrossHotSpot(MBB, Status, UniformVgprMap, VToSMap, LIS);
  }

  if (VToSMap.empty())
    return false;
  SlotIndexes *SlotIndexes = LIS->getSlotIndexes();
  const MCInstrDesc &ReadFirstLaneDesc = SIII->get(AMDGPU::V_READFIRSTLANE_B32);
  for (auto It : VToSMap) {
    unsigned Reg = It.first;
    MachineInstr *MI = It.second;

    auto *VRC = SIRI->getRegClassForReg(MRI, Reg);
    // TODO: support bigger vgpr to sgpr.
    if (VRC != &AMDGPU::VGPR_32RegClass)
      continue;
    auto *NewRC = SIRI->getEquivalentSGPRClass(VRC);
    Register NewDst = MRI.createVirtualRegister(NewRC);

    auto ReadFirstLane =
        BuildMI(MF, MI->getDebugLoc(), ReadFirstLaneDesc, NewDst);
    SmallVector<MachineInstr *, 2> UserMIs;
    for (MachineInstr &UserMI : MRI.use_nodbg_instructions(Reg)) {
      // Do not replace v->s across loops. Even if the value is uniform
      // branch divergence can cause a uniform value in a loop to be
      // non-uniform when used outside a loop.
      if (isSafeRematCandidateUser(&UserMI, SIII) &&
          !isCrossLoopUse(MI, &UserMI, MLI))
        UserMIs.emplace_back(&UserMI);
    }

    // Finish readfirstlane
    ReadFirstLane.addReg(Reg);
    MachineInstr *VToSMI = ReadFirstLane.getInstr();
    Remat->TotalUniformInsts.insert(VToSMI);
    Remat->SafeToRemoveInsts.insert(VToSMI);
    MachineBasicBlock *MBB = MI->getParent();
    MBB->insertAfter(MI->getIterator(), VToSMI);
    SlotIndexes->insertMachineInstrInMaps(*VToSMI);

    for (MachineInstr *UserMI : UserMIs) {
      const auto &Desc = UserMI->getDesc();
      bool IsIllegal = false;
      for (unsigned I = 0; I < UserMI->getNumOperands(); I++) {
        MachineOperand &MO = UserMI->getOperand(I);
        if (!MO.isReg())
          continue;
        if (MO.isDef())
          continue;
        if (MO.getReg() != Reg)
          continue;
        if (I >= Desc.getNumOperands()) {
          IsIllegal = true;
          break;
        }

        MO.setReg(NewDst);
        if (UserMI->getDesc().operands()[I].RegClass != -1) {
          if (!SIII->isOperandLegal(*UserMI, I, &MO)) {
            SIII->legalizeOperands(*UserMI);
            // In case legalizeOperands not help, just legalize with mov.
            if (UserMI->getDesc().operands()[I].RegClass != -1 &&
                !SIII->isOperandLegal(*UserMI, I)) {
              SIII->legalizeOpWithMove(*UserMI, I);
            }
          }
        } else {
          // consider not have limit on reg class.
        }
      }
      if (IsIllegal)
        continue;

      auto RIt = UserMI->getReverseIterator();
      RIt++;
      auto EndIt = UserMI->getParent()->rend();
      while (RIt != EndIt && !RIt->isDebugInstr() &&
             !SlotIndexes->hasIndex(*RIt))
        SlotIndexes->insertMachineInstrInMaps(*(RIt++));
    }
  }

  return true;
}

bool collectRematableHotReg(
    MachineInstr &MI, const GCNRPTracker::LiveRegSet &HotLive,
    GCNRPTracker::LiveRegSet &PureHotRematSet,
    DenseMap<MachineInstr *, unsigned> &PureHotRematLevels, unsigned &DefReg,
    const GCNRPTracker::LiveRegSet &InputLive, const MachineRegisterInfo &MRI) {
  // Ignore inst not have def or more than 1 def.
  if (MI.getDesc().getNumDefs() != 1)
    return false;

  DefReg = MI.defs().begin()->getReg();

  unsigned Level = 0;
  for (MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    if (MO.isDef())
      continue;

    Register Reg = MO.getReg();

    // If user is in same MI like
    //  %4:vgpr_32 = V_MAD_LEGACY_F32 %2:vgpr_32, %3:vgpr_32, %4:vgpr_32
    // remat It will not help.
    if (Reg == DefReg) {
      return false;
    }

    if (MO.isImplicit() && (Reg == AMDGPU::EXEC || Reg == AMDGPU::EXEC_LO))
      continue;
    if (Reg.isPhysical())
      return false;

    if (nullptr == getInBlockUniqueDef(Reg, MI.getParent(), InputLive, MRI))
      return false;

    LaneBitmask Mask = llvm::getRegMask(MO, MRI);

    if (isInLiveSet(Reg, Mask, HotLive))
      continue;

    if (isInLiveSet(Reg, Mask, PureHotRematSet)) {
      unsigned RegLevel = getPacifistLevel(Reg, PureHotRematLevels, MRI);
      Level = std::max(Level, RegLevel);
      continue;
    }

    return false;
  }

  for (MachineOperand &MO : MI.defs()) {
    Register Reg = MO.getReg();

    if (Reg.isPhysical())
      return false;

    if (nullptr == getInBlockUniqueDef(Reg, MI.getParent(), InputLive, MRI))
      return false;

    LaneBitmask Mask = llvm::getRegMask(MO, MRI);
    PureHotRematSet[Reg] |= Mask;
  }

  PureHotRematLevels[&MI] = Level + 1;
  // If no def, It will not increase pressure, don't mark It.
  return true;
}

bool tryRemat(MachineBasicBlock &MBB, MachineInstr *HotMi,
              std::vector<SubExp> &InBlockCloneSubExps, bool IsVGPR,
              const GCNRPTracker::LiveRegSet &InputLive,
              DenseSet<MachineInstr *> &HotSet, int VDistance, int SDistance,
              unsigned VLimit, unsigned SLimit,
              const DenseSet<MachineBasicBlock *> &MemWriteMBBSet,
              LiveIntervals *LIS, const MachineRegisterInfo &MRI,
              const SIRegisterInfo *SIRI, const SIInstrInfo *SIII) {
  auto &ST = MBB.getParent()->getSubtarget<GCNSubtarget>();
  const auto &SI = LIS->getInstructionIndex(*HotMi).getBaseIndex();
  const auto LISLR = llvm::getLiveRegs(SI, *LIS, MRI);

  GCNRPTracker::LiveRegSet HotLive = LISLR;

  GCNRPTracker::LiveRegSet PureHotRematSet;
  std::vector<MachineInstr *> PureHotRematList;
  DenseMap<MachineInstr *, unsigned> PureHotRematLevels;

  GCNRPTracker::LiveRegSet OutputSet;
  LLVM_DEBUG(dbgs() << "pure hot remat begin\n");
  // Find reg which could remat from other reg in liveSet.
  const unsigned KMaxRematLevel = 6;
  GCNDownwardRPTracker Tracker(*LIS);
  Tracker.reset(*MBB.begin(), &InputLive);
  for (auto It = MBB.begin(); It != MBB.end(); It++) {
    MachineInstr &MI = *It;
    const GCNRegPressure &RP = Tracker.getPressure();

    if (MI.isDebugInstr())
      continue;

    // Igonre inst in hot range.
    if (RP.getVGPRNum(ST.hasGFX90AInsts()) > VLimit ||
        RP.getMaxSGPR() > SLimit) {
      Tracker.advance();
      continue;
    }

    // Stop at hotMI.
    if (&MI == HotMi)
      break;

    Tracker.advance();

    unsigned DefReg = 0;
    if (collectRematableHotReg(MI, HotLive, PureHotRematSet, PureHotRematLevels,
                               DefReg, InputLive, MRI)) {
      unsigned Level = PureHotRematLevels[&MI];
      if (Level >= KMaxRematLevel)
        continue;

      // If the def reg is in hot reg.
      // Add to output.
      if (HotLive.find(DefReg) != HotLive.end()) {
        bool IsUserIsHot = false;
        for (MachineInstr &UseMI : MRI.use_nodbg_instructions(DefReg)) {
          if (UseMI.getParent() != &MBB)
            continue;
          if (0 == HotSet.count(&UseMI))
            continue;

          const auto &UseSI = LIS->getInstructionIndex(UseMI).getBaseIndex();
          // When has a hot user after hotMI, remat It may not help.
          if (UseSI > SI) {
            IsUserIsHot = true;
            break;
          }
        }

        if (IsUserIsHot)
          continue;
        OutputSet[DefReg];
        LLVM_DEBUG(dbgs() << "hotRemat:");
        LLVM_DEBUG(MI.getOperand(0).dump());
        // remove It from hotLive to avoid It as input when build dag.
        HotLive.erase(DefReg);
      }
      PureHotRematList.emplace_back(&MI);
      LLVM_DEBUG(dbgs() << "level:" << Level);
      LLVM_DEBUG(MI.dump());
    }
  }

  LLVM_DEBUG(dbgs() << "pure hot remat end\n");

  // Create input/output for pure hot remat.
  // Input is things hot reg in level 1 and output is things level > 1.
  // Build SubExp with pureHotRematList as Nodes, hotLive as input
  // rematHot as output.
  // Not join input when build ExpDag to get small subExps.
  ExpDag Dag(MRI, SIRI, SIII, /*IsJoinInput*/ false);
  Dag.build(HotLive, OutputSet, PureHotRematList);
  // Find best subExp add to inBlockCloneSubExps.
  // Sort by size of subExp.
  std::sort(Dag.SubExps.begin(), Dag.SubExps.end(),
            [](const SubExp &A, const SubExp &B) {
              return A.SUnits.size() < B.SUnits.size();
            });
  std::vector<SubExp> CloneSubExps;
  int Distance = IsVGPR ? VDistance : SDistance;
  for (SubExp &SubExp : Dag.SubExps) {
    if (SubExp.IsNotSafeToCopy)
      continue;
    if (IsVGPR) {
      if (SubExp.VOutputSize == 0)
        continue;
    } else {
      if (SubExp.SOutputSize == 0)
        continue;
    }
    if (!SubExp.isSafeToMove(MRI, /*IsMoveUp*/ false))
      continue;
    // Not clone .
    if (SubExp.SUnits.size() > 10)
      continue;
    // Do not allow remat in the block when the expression has a memory op and
    // the block has a write. We could allow this in some cases with better
    // analysis.
    if (SubExp.IsHasMemInst && MemWriteMBBSet.count(&MBB))
      continue;
    if (IsVGPR) {
      Distance -= SubExp.VOutputSize;
    } else {
      Distance -= SubExp.SOutputSize;
    }
    CloneSubExps.emplace_back(SubExp);
    if (Distance <= 0)
      break;
  }
  if (Distance <= 0) {
    InBlockCloneSubExps.insert(InBlockCloneSubExps.end(), CloneSubExps.begin(),
                               CloneSubExps.end());
  }
  return Distance <= 0;
}

// Try to remat live reg in hot spot from other live reg in hot spot.
//
bool tryRematInHotSpot(
    MachineBasicBlock &MBB, RematStatus &Status, int VDistance, int SDistance,
    int VSaved, int SSaved, std::vector<SubExp> &InBlockCloneSubExps,
    DenseMap<MachineBasicBlock *, MachineInstr *> &InBlockHotVInstMap,
    DenseMap<MachineBasicBlock *, MachineInstr *> &InBlockHotSInstMap,
    LiveIntervals *LIS, const MachineRegisterInfo &MRI,
    const SIRegisterInfo *SIRI, const SIInstrInfo *SIII) {
  unsigned VLimit = Status.TargetVLimit;
  unsigned SLimit = Status.TargetSLimit;

  auto &ST = MBB.getParent()->getSubtarget<GCNSubtarget>();
  const GCNRPTracker::LiveRegSet InputLive = Status.MBBInputLiveMap[&MBB];

  const GCNRPTracker::LiveRegSet OutputLive = Status.MBBOutputLiveMap[&MBB];

  // Collect reg pressure.
  unsigned MaxLocalVPressure = 0;
  unsigned MaxLocalSPressure = 0;
  // Build a DAG or only on demand?
  MachineInstr *HotVMI = nullptr;
  MachineInstr *HotSMI = nullptr;
  DenseSet<MachineInstr *> HotSet;

  GCNDownwardRPTracker Tracker(*LIS);

  Tracker.reset(*MBB.begin(), &InputLive);
  for (auto It = MBB.begin(); It != MBB.end(); It++) {
    MachineInstr &MI = *It;
    if (MI.isDebugInstr()) {
      continue;
    }

    unsigned VPressure = Tracker.getPressure().getVGPRNum(ST.hasGFX90AInsts());
    unsigned SPressure = Tracker.getPressure().getMaxSGPR();

    SPressure += RegForVCC;

    VPressure -= VSaved;
    SPressure -= SSaved;
    Tracker.advance();

    if (VPressure <= VLimit && SPressure <= SLimit) {
      continue;
    }
    HotSet.insert(&MI);
    if (MaxLocalVPressure < VPressure) {
      MaxLocalVPressure = VPressure;
      HotVMI = &MI;
    }
    if (MaxLocalSPressure < SPressure) {
      MaxLocalSPressure = SPressure;
      HotSMI = &MI;
    }
  }

  InBlockHotVInstMap[&MBB] = HotVMI;
  InBlockHotSInstMap[&MBB] = HotSMI;
  if (VDistance > 0 && HotVMI) {
    // Use hotVMI when apply.
    InBlockHotSInstMap[&MBB] = nullptr;
    if (tryRemat(MBB, HotVMI, InBlockCloneSubExps, /*IsVGPR*/ true, InputLive,
                 HotSet, VDistance, SDistance, VLimit, SLimit,
                 Status.MemWriteMBBSet, LIS, MRI, SIRI, SIII))
      return true;
  }

  if (SDistance > 0 && HotSMI) {
    // Use hotSMI when apply.
    InBlockHotSInstMap[&MBB] = HotSMI;
    InBlockHotVInstMap[&MBB] = nullptr;
    return tryRemat(MBB, HotSMI, InBlockCloneSubExps, /*IsVGPR*/ false,
                    InputLive, HotSet, VDistance, VDistance, VLimit, SLimit,
                    Status.MemWriteMBBSet, LIS, MRI, SIRI, SIII);
  }
  return false;
}
// Sort subExpCandidates to make sure deeper subExp apply first.
// If subExp0 use result of subExp1, subExp0 is deeper than subExp1.
// When apply subExp1 before subExp0, new clone of subExp0 which use result of
// subExp1 will have old reg of subExp1. And reg pressure will not be reduced.
void sortSubExpCandidates(std::vector<SubExp> &SubExpCandidates) {
  MapVector<Register, SetVector<SubExp *>> InputMap;
  MapVector<Register, SetVector<SubExp *>> OutputMap;
  struct SortNode {
    SubExp Exp;
    unsigned Depth;
    bool IsDepthDirty;
    SmallDenseSet<SubExp *, 2> Preds;
    SmallDenseSet<SubExp *, 2> Succs;
  };

  {
    SmallVector<unsigned, 10> RegSortStorage;
    for (SubExp &Exp : SubExpCandidates) {
      RegSortStorage.assign(Exp.TopRegs.begin(), Exp.TopRegs.end());
      std::sort(RegSortStorage.begin(), RegSortStorage.end());
      for (auto It : RegSortStorage) {
        unsigned Reg = It;
        InputMap[Reg].insert(&Exp);
      }

      RegSortStorage.assign(Exp.BottomRegs.begin(), Exp.BottomRegs.end());
      std::sort(RegSortStorage.begin(), RegSortStorage.end());
      for (auto It : RegSortStorage) {
        unsigned Reg = It;
        OutputMap[Reg].insert(&Exp);
      }
    }
  }

  MapVector<SubExp *, SortNode> SortMap;
  for (auto It : InputMap) {
    unsigned Reg = It.first;
    MapVector<Register, SetVector<SubExp *>>::iterator OutIt = OutputMap.find(Reg);
    if (OutIt == OutputMap.end())
      continue;
    auto &InExps = It.second;
    auto &OutExps = OutIt->second;
    for (SubExp *InExp : InExps) {
      for (SubExp *OutExp : OutExps) {
        if (InExp->IsHoist != OutExp->IsHoist) {
          // Different direction.
          // If output (def) move up, input(use) move down, nothing happens.
          if (OutExp->IsHoist)
            continue;
          // Canot input(use) move up, output(def) move down.
          // Choose the exp which save more.
          int InExpGain = InExp->VOutputSize - InExp->VInputSize;
          int OutExpGain = OutExp->VInputSize - InExp->VOutputSize;
          if (InExpGain >= OutExpGain) {
            OutExp->SUnits.clear();
          } else {
            InExp->SUnits.clear();
          }
          continue;
        }
        // Link outExp to inExp.
        if (InExp->IsHoist) {
          SortMap[OutExp].Preds.insert(InExp);
          SortMap[InExp].Succs.insert(OutExp);
        } else {
          SortMap[InExp].Preds.insert(OutExp);
          SortMap[OutExp].Succs.insert(InExp);
        }
      }
    }
  }

  if (SortMap.empty())
    return;

  SmallVector<SubExp *, 8> WorkList;
  for (SubExp &Exp : SubExpCandidates) {
    SortNode &Node = SortMap[&Exp];
    Node.Depth = 0;
    Node.Exp = Exp;
    Node.IsDepthDirty = !Node.Preds.empty();
    if (!Node.IsDepthDirty)
      WorkList.emplace_back(&Exp);
  }
  // Calc depth.
  while (!WorkList.empty()) {
    SubExp *Exp = WorkList.pop_back_val();
    SortNode &Node = SortMap[Exp];
    for (SubExp *Succ : Node.Succs) {
      SortNode &SuccNode = SortMap[Succ];
      SuccNode.Depth = std::max(SuccNode.Depth, Node.Depth + 1);
      bool IsAllPrevClean = true;
      for (SubExp *Prev : SuccNode.Preds) {
        SortNode &PrevNode = SortMap[Prev];
        if (PrevNode.IsDepthDirty) {
          IsAllPrevClean = false;
          break;
        }
      }
      if (IsAllPrevClean) {
        SuccNode.IsDepthDirty = false;
        WorkList.push_back(Succ);
      }
    }
  }

  std::vector<SortNode *> Nodes;
  for (auto &It : SortMap) {
    SortNode &Node = It.second;
    Nodes.emplace_back(&Node);
  }

  struct Sorter {
    bool operator()(const SortNode *A, const SortNode *B) {
      return A->Depth > B->Depth;
    }
  };

  // subExp deeper should be apply first.
  std::sort(Nodes.begin(), Nodes.end(), Sorter());

  SubExpCandidates.clear();
  for (auto &Node : Nodes) {
    SubExpCandidates.emplace_back(Node->Exp);
  }
}

// Compare pressure, return ture if maxV0/maxS0 pressure is higher than
// maxV1/maxS1.
bool pressureHigher(unsigned MaxV0, unsigned MaxS0, unsigned MaxV1,
                    unsigned MaxS1, const GCNSubtarget *ST) {
  unsigned VTgtOcc0 = ST->getOccupancyWithNumVGPRs(MaxV0);
  unsigned VTgtOcc1 = ST->getOccupancyWithNumVGPRs(MaxV1);
  unsigned STgtOcc0 = ST->getOccupancyWithNumSGPRs(MaxS0);
  unsigned STgtOcc1 = ST->getOccupancyWithNumSGPRs(MaxS1);
  unsigned Occ0 = std::min(VTgtOcc0, STgtOcc0);
  unsigned Occ1 = std::min(VTgtOcc1, STgtOcc1);
  //  is low pressure.
  if (Occ0 > Occ1)
    return false;
  if (Occ0 < Occ1)
    return true;
  // When sgpr bound,  is high pressure.
  if (VTgtOcc0 > STgtOcc0 && VTgtOcc1 > STgtOcc1) {
    return MaxS0 > MaxS1;
  }
  // When vgpr bound or mix, vgpr higher is higher pressure.
  return MaxV0 > MaxV1;
}

// Return true if the subExp can help pressure for passThrus.
bool canHelpPressureWhenSink(SubExp &SubExp,
                             const GCNRPTracker::LiveRegSet &PassThrus,
                             const MachineRegisterInfo &MRI,
                             const SIRegisterInfo *SIRI,
                             const MachineLoopInfo *MLI,
                             MachineDominatorTree *DT, bool IsCanClone,
                             bool IsSgprBound) {
  LLVM_DEBUG(SubExp.dump(MRI, SIRI));
  if (!SubExp.isSafeToMove(MRI, /*IsMoveUp*/ false))
    return false;

  // Update input size to ignore lives in which already in
  // passThrus.
  for (auto It : SubExp.InputLive) {
    unsigned Reg = It.first;
    if (PassThrus.count(Reg) == 0)
      continue;
    unsigned Size = getRegSize(Reg, It.second, MRI, SIRI);
    if (SIRI->isVGPR(MRI, Reg)) {
      SubExp.VInputSize -= Size;
    } else {
      SubExp.SInputSize -= Size;
    }
  }

  if (SubExp.VInputSize > SubExp.VOutputSize)
    return false;

  if (SubExp.SInputSize > SubExp.SOutputSize && IsSgprBound)
    return false;

  if (SubExp.SInputSize >= SubExp.SOutputSize &&
      SubExp.VInputSize == SubExp.VOutputSize)
    return false;

  // Try to find a Insert Block.
  // Skip multi def output sub exp.
  // Collect user blocks, find common dom.
  BlockSet UserBlocks;
  for (unsigned Reg : SubExp.BottomRegs) {
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
      MachineBasicBlock *UserBB = UseMI.getParent();
      // Skip current BB.
      if (UserBB != SubExp.FromBB)
        UserBlocks.insert(UserBB);
    }
  }
  if (UserBlocks.empty())
    return false;
  MachineBasicBlock *UserBlock = nearestCommonDominator(DT, UserBlocks);
  if (!DT->dominates(SubExp.FromBB, UserBlock)) {
    return false;
  }
  if (UserBlock == SubExp.FromBB &&
      // When allow clone, could go clone path if cannot move subExp.
      !IsCanClone)
    return false;

  SubExp.ToBB = UserBlock;
  if (auto *ToLoop = MLI->getLoopFor(UserBlock)) {
    auto *FromLoop = MLI->getLoopFor(SubExp.FromBB);
    if (!FromLoop || FromLoop->getLoopDepth() < ToLoop->getLoopDepth())
      SubExp.IsMoveIntoLoop = true;
  } else if (auto *FromLoop = MLI->getLoopFor(SubExp.FromBB)) {
    auto *ToLoop = MLI->getLoopFor(UserBlock);
    // not safe to move out of loop.
    if (!ToLoop || FromLoop->getLoopDepth() > ToLoop->getLoopDepth() ||
        ToLoop != FromLoop)
      return false;
  }
  return true;
}

bool canHelpPressureWhenHoist(SubExp &SubExp, const MachineRegisterInfo &MRI,
                              const MachineLoopInfo *MLI, bool IsSgprBound) {
  if (!SubExp.isSafeToMove(MRI, /*IsMoveUp*/ true))
    return false;
  if (SubExp.VInputSize < SubExp.VOutputSize)
    return false;
  if (SubExp.SInputSize < SubExp.SOutputSize && IsSgprBound)
    return false;

  if (SubExp.SInputSize <= SubExp.SOutputSize &&
      SubExp.VInputSize == SubExp.VOutputSize)
    return false;

  // Try to find a Insert Block.
  // Skip multi def output sub exp.
  // Collect user blocks, find common dom.
  BlockSet DefBlocks;
  for (unsigned Reg : SubExp.TopRegs) {
    MachineInstr *DefMI = MRI.getUniqueVRegDef(Reg);
    if (!DefMI)
      continue;
    DefBlocks.insert(DefMI->getParent());
  }
  if (DefBlocks.size() != 1)
    return false;
  MachineBasicBlock *DefBlock = *DefBlocks.begin();
  SubExp.ToBB = DefBlock;
  // Not do same block hoist.
  if (SubExp.ToBB == SubExp.FromBB)
    return false;

  if (auto *ToLoop = MLI->getLoopFor(DefBlock)) {
    auto *FromLoop = MLI->getLoopFor(SubExp.FromBB);
    // TODO: enable move into loop when hoist.
    if (!FromLoop || FromLoop->getLoopDepth() < ToLoop->getLoopDepth())
      return false;
  } else if (auto *FromLoop = MLI->getLoopFor(SubExp.FromBB)) {
    auto *ToLoop = MLI->getLoopFor(DefBlock);
    // not safe to move out of loop.
    if (!ToLoop || FromLoop->getLoopDepth() > ToLoop->getLoopDepth() ||
        ToLoop != FromLoop)
      return false;
  }
  return true;
}

SmallVector<std::pair<MachineBasicBlock *, GCNRPTracker::LiveRegSet>>
groupPassThruByDefBlock(Remat *Remat, const GCNRPTracker::LiveRegSet &PassThrus,
                        GCNRPTracker::LiveRegSet &UsedPassThrus,
                        MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                        const SIInstrInfo *SIII) {
  MapVector<MachineBasicBlock *, GCNRPTracker::LiveRegSet> Candidates;

  // Group safe candidates by define block.
  for (auto It : PassThrus) {
    Register Reg = It.first;
    // Skip used pass thru reg to avoid count It twice for different hot block.
    if (UsedPassThrus.count(Reg))
      continue;
    LLVM_DEBUG(printVreg(Reg, MRI));
    LLVM_DEBUG(if (SIRI->isSGPRReg(MRI, Reg)) dbgs() << " sgpr ";
               else dbgs() << " vgpr ";);
    if (!isSafeCandidate(Remat, Reg, MRI, SIRI, SIII, /*IsSink*/ true)) {
      LLVM_DEBUG(dbgs() << " is not safe\n");
      continue;
    }
    LLVM_DEBUG(dbgs() << " is safe\n");
    // DefMI is already checked in isSafeCandidate.
    MachineInstr *DefMI = MRI.getUniqueVRegDef(Reg);

    GCNRPTracker::LiveRegSet &DefInMBB = Candidates[DefMI->getParent()];
    DefInMBB[Reg] = It.second;
  }

  llvm::SmallVector<std::pair<MachineBasicBlock *, GCNRPTracker::LiveRegSet>>
      Result = Candidates.takeVector();

  LLVM_DEBUG(llvm::dbgs() << "Before sort candidates\n"; for (auto It
                                                              : Result) {
    MachineBasicBlock *MBB = It.first;
    auto &defInMBB = It.second;
    MBB->dump();
    llvm::dumpLiveSet(defInMBB, SIRI);
  } llvm::dbgs() << "end of candidates\n";);

  std::sort(Result.begin(), Result.end(),
            [](std::pair<MachineBasicBlock *, GCNRPTracker::LiveRegSet> &It0,
               std::pair<MachineBasicBlock *, GCNRPTracker::LiveRegSet> &It1) {
              return It0.first->getNumber() < It1.first->getNumber();
            });

  LLVM_DEBUG(llvm::dbgs() << "After sort candidates\n"; for (auto It
                                                             : Result) {
    MachineBasicBlock *MBB = It.first;
    auto &defInMBB = It.second;
    MBB->dump();
    llvm::dumpLiveSet(defInMBB, SIRI);
  } llvm::dbgs() << "end of candidates\n";);

  return Result;
}

// collect pass thru regs of MBB.
GCNRPTracker::LiveRegSet
collectPassThrus(MachineBasicBlock *MBB,
                 const GCNRPTracker::LiveRegSet &InputLive,
                 const GCNRPTracker::LiveRegSet &OutputLive,
                 const GCNRPTracker::LiveRegSet &LiveRegCandidates,
                 MachineRegisterInfo &MRI, bool IsCanClone) {
  GCNRPTracker::LiveRegSet PassThrus;
  llvm::mergeLiveRegSet(PassThrus, InputLive);
  llvm::andLiveRegSet(PassThrus, OutputLive);

  // Remove reg which not in liveRegCandidates.
  GCNRPTracker::LiveRegSet TmpPassThrus = PassThrus;
  for (auto It : TmpPassThrus) {
    unsigned Reg = It.first;
    if (!LiveRegCandidates.count(Reg)) {
      PassThrus.erase(Reg);
    }
  }
  TmpPassThrus = PassThrus;
  // Remove reg which has read/write in MBB.
  for (auto It : TmpPassThrus) {
    unsigned Reg = It.first;
    DenseSet<MachineBasicBlock *> DefMBBs;
    for (MachineInstr &DefMI : MRI.def_instructions(Reg)) {
      MachineBasicBlock *MBB = DefMI.getParent();
      DefMBBs.insert(MBB);
    }
    DenseSet<MachineBasicBlock *> UseMBBs;
    // Allow use for pass thru if clone is OK.
    if (!IsCanClone) {
      for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
        MachineBasicBlock *UserMBB = UseMI.getParent();
        UseMBBs.insert(UserMBB);
      }
    }
    bool IsW = DefMBBs.count(MBB) > 0;
    bool IsR = UseMBBs.count(MBB) > 0;

    bool IsPassThru = !IsW && !IsR;
    if (!IsPassThru)
      PassThrus.erase(Reg);
  }
  return PassThrus;
}
// Try to build a free subExp which all input is passThrus.
SubExp buildFreeSubExp(SubExp &Exp,
                       GCNRPTracker::LiveRegSet &PassThrus,
                       MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI) {
  SubExp FreeExp;
  // Try to split the subExp to find a help case.
  // Scan all inst in subExp, propagate free inst which input is from
  // passThrus.
  SmallDenseSet<Register, 4> FreeRegs;
  SmallDenseSet<Register, 8> FreeInstUseRegs;
  SmallVector<MachineInstr *, 4> FreeInsts;
  for (MachineInstr *MI : Exp.SUnits) {
    bool IsFree = true;
    // Check all use regs are free.
    for (MachineOperand &MO : MI->uses()) {
      if (!MO.isReg())
        continue;
      Register Reg = MO.getReg();
      if (MO.isImplicit() && Reg == AMDGPU::EXEC)
        continue;
      if (MRI.getUniqueVRegDef(Reg) == nullptr) {
        IsFree = false;
        break;
      }
      // Skip local pass thrus unless It is free.
      if (PassThrus.count(Reg) && Exp.TopRegs.count(Reg))
        continue;
      if (FreeRegs.count(Reg))
        continue;
      IsFree = false;
      break;
    }
    // Check def is unique.
    for (MachineOperand &MO : MI->defs()) {
      Register Reg = MO.getReg();
      if (MRI.getUniqueVRegDef(Reg) == nullptr) {
        IsFree = false;
        break;
      }
    }
    if (!IsFree)
      continue;
    // Save inst as free inst.
    FreeInsts.emplace_back(MI);
    // Save def as free reg.
    for (MachineOperand &MO : MI->defs()) {
      Register Reg = MO.getReg();
      FreeRegs.insert(Reg);
    }
    // Save use regs as free use reg.
    for (MachineOperand &MO : MI->uses()) {
      if (!MO.isReg())
        continue;
      Register Reg = MO.getReg();

      FreeInstUseRegs.insert(Reg);
    }
  }
  // Then remove local inst has no output use.
  for (MachineInstr *MI : FreeInsts) {
    bool IsFreeUsed = false;
    for (MachineOperand &MO : MI->defs()) {
      Register Reg = MO.getReg();
      // Used as freeInst or output.
      IsFreeUsed |= FreeInstUseRegs.count(Reg) > 0 || Exp.BottomRegs.count(Reg);
    }
    if (!IsFreeUsed)
      continue;
    FreeExp.SUnits.emplace_back(MI);
  }
  if (FreeExp.SUnits.empty()) {
    // mark has terminator to make It unsafe.
    FreeExp.IsHasTerminatorInst = true;
    return FreeExp;
  }
  // Build BottomRegs and TopRegs for freeExp.
  // BottomRegs is freeRegs in subExp.BottomRegs.
  for (Register FreeReg : FreeRegs) {
    if (Exp.BottomRegs.count(FreeReg))
      FreeExp.BottomRegs.insert(FreeReg);
  }
  // TopRegs is freeInstUseRegs in subExp.TopRegs.
  for (Register FreeInstUseReg : FreeInstUseRegs) {
    if (Exp.TopRegs.count(FreeInstUseReg))
      FreeExp.TopRegs.insert(FreeInstUseReg);
  }
  FreeExp.FromBB = Exp.FromBB;
  FreeExp.ToBB = Exp.ToBB;
  // must be clone since is partial of subExp.
  FreeExp.IsCloneOnly = true;

  // Calc reg for freeExp.
  for (unsigned Reg : FreeExp.TopRegs) {
    FreeExp.InputLive[Reg];
  }

  for (unsigned Reg : FreeExp.BottomRegs) {
    FreeExp.OutputLive[Reg];
  }

  CollectLiveSetPressure(FreeExp.InputLive, MRI, SIRI, FreeExp.VInputSize,
                         FreeExp.SInputSize);
  CollectLiveSetPressure(FreeExp.OutputLive, MRI, SIRI, FreeExp.VOutputSize,
                         FreeExp.SOutputSize);
  return FreeExp;
}

std::vector<SubExp> buildSubExpCandidates(
    Remat *Remat,
    SmallVector<std::pair<MachineBasicBlock *, GCNRPTracker::LiveRegSet>>
        &Candidates,
    GCNRPTracker::LiveRegSet &PassThrus, MachineRegisterInfo &MRI,
    const SIRegisterInfo *SIRI, const SIInstrInfo *SIII,
    const MachineLoopInfo *MLI, SlotIndexes *SlotIndexes,
    MachineDominatorTree *DT, bool IsCanClone, bool IsSgprBound,
    GCNRPTracker::LiveRegSet &UnusedPassThrus,
    DenseSet<MachineBasicBlock *> &MemWriteMBBSet,
    bool AllowPartialUseInSubExp) {
  std::vector<SubExp> SubExpCandidates;
  // Build exp dag on define blocks.
  // Save profit candidates into list.
  for (auto &It : Candidates) {
    MachineBasicBlock *DefMBB = It.first;
    // Try to remove out reg def sub exp from DefMBB.
    GCNRPTracker::LiveRegSet &DefInMBB = It.second;
    // Go up on the dag until reach share node.
    auto SubExps = buildSubExpFromCandidates(
        Remat, DefInMBB, DefMBB, SIRI, SIII, MRI, SlotIndexes, UnusedPassThrus,
        AllowPartialUseInSubExp);
    for (SubExp &Exp : SubExps) {
      if (Exp.IsHasMemInst) {
        // Skip when memory ld/st inst need to cross MBB which write memory.
        // TODO: check all MBBs in between FromBB and ToBB not write memory.
        // Currently just skip when any memory write exist.
        if (!MemWriteMBBSet.empty()) {
          MachineBasicBlock *FromBB = Exp.FromBB;
          MachineBasicBlock *ToBB = Exp.ToBB;
          if (Exp.IsHoist) {
            FromBB = Exp.ToBB;
            ToBB = Exp.FromBB;
          }
          bool IsCrossMemWriteMBB = false;
          for (MachineBasicBlock *MemMBB : MemWriteMBBSet) {
            if (DT->dominates(ToBB, MemMBB))
              continue;
            if (DT->dominates(MemMBB, FromBB))
              continue;
            IsCrossMemWriteMBB = true;
            break;
          }
          if (IsCrossMemWriteMBB)
            continue;
        }
      }
      if (!canHelpPressureWhenSink(Exp, PassThrus, MRI, SIRI, MLI, DT,
                                   IsCanClone, IsSgprBound)) {
        if (AllowPartialUseInSubExp &&
            Exp.isSafeToMove(MRI, /*IsMoveUp*/ false)) {
          SubExp FreeSubExp = buildFreeSubExp(Exp, PassThrus, MRI, SIRI);
          if (canHelpPressureWhenSink(FreeSubExp, PassThrus, MRI, SIRI, MLI, DT,
                                      IsCanClone, IsSgprBound)) {
            SubExpCandidates.emplace_back(FreeSubExp);
          }
        }
        continue;
      }

      SubExpCandidates.emplace_back(Exp);
    }
  }
  return SubExpCandidates;
}

std::pair<int, int>
calculateSaving(HotBlock &HotBb, std::vector<SubExp> &SubExpCandidates,
                GCNRPTracker::LiveRegSet &InputLive,
                GCNRPTracker::LiveRegSet &OutputLive, bool IsVOutBound,
                bool IsSOutBound, bool IsCanClone, MachineDominatorTree *DT,
                const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI) {
  int Vgpr = 0;
  int Sgpr = 0;
  MachineBasicBlock *MBB = HotBb.MBB;
  // Sink saving.
  for (SubExp &Exp : SubExpCandidates) {
    if (Exp.IsHoist) {
      // ToMBB -> MBB -> FromMBB.
      // If ToMBB not dom hot block, reg will not live in MBB.
      if (!DT->dominates(Exp.ToBB, MBB))
        continue;
    } else {
      // If FromBB not dom hot block, reg will not live in MBB.
      if (!DT->dominates(Exp.FromBB, MBB))
        continue;
      // When subExp is from hotBB, check output instead of input.
      if (Exp.FromBB == MBB) {
        if (IsVOutBound && Exp.VOutputSize < Exp.VInputSize)
          continue;
        if (IsSOutBound && Exp.SOutputSize < Exp.SInputSize)
          continue;
        Vgpr += Exp.VInputSize;
        Vgpr -= Exp.VOutputSize;
        Sgpr += Exp.SInputSize;
        Sgpr -= Exp.SOutputSize;
        continue;
      }
    }
    int VgprDiff = 0;
    int SgprDiff = 0;
    MachineBasicBlock *ToMBB = Exp.ToBB;
    // If subExp is to hotBB, It is crossing output instead of input.
    GCNRPTracker::LiveRegSet &CrossLive = MBB == ToMBB ? OutputLive : InputLive;

    bool IsClone = false;
    GCNRPTracker::LiveRegSet NewInput;
    if (!Exp.IsMoveIntoLoop) {
      if (Exp.IsHoist) {
        // If FromBB dom hot block, It will not change live for MBB.
        if (Exp.FromBB != MBB && DT->dominates(Exp.FromBB, MBB))
          continue;
      } else {
        // If ToBB dom hot block, It will not change live for MBB.
        if (ToMBB != MBB && DT->dominates(ToMBB, MBB)) {
          if (IsCanClone && !Exp.IsNotSafeToCopy) {
            IsClone = true;
          } else {
            continue;
          }
        }
      }

      for (auto OutIt : Exp.OutputLive) {
        unsigned Reg = OutIt.first;
        LaneBitmask OutMask = OutIt.second;
        LaneBitmask MBBBeginMask;
        if (CrossLive.find(Reg) != CrossLive.end())
          MBBBeginMask = CrossLive[Reg];
        // Check mask which live in both BeginSlot and exp output when sink to
        // kill the output. Check mask which not live in BeginSlot  in
        // exp output when hoist to live the output.
        LaneBitmask ProfitMask = Exp.IsHoist ? (OutMask & (~MBBBeginMask))
                                             : (OutMask & MBBBeginMask);
        if (MBBBeginMask.any()) {
          unsigned Size = getRegSize(Reg, ProfitMask, MRI, SIRI);
          LLVM_DEBUG(std::string movStr =
                         Exp.IsHoist ? "output hoist:" : "output sink:";
                     dbgs()
                     << movStr << Register::virtReg2Index(Reg) << " " << Size);
          // Exp out live at block input.
          // It will descrease live for MBB when sink and increase when hoist.
          if (SIRI->isVGPR(MRI, Reg)) {
            LLVM_DEBUG(dbgs() << "v\n");
            if (Exp.IsHoist)
              VgprDiff += Size;
            else
              VgprDiff -= Size;
          } else {
            LLVM_DEBUG(dbgs() << "s\n");
            if (Exp.IsHoist)
              SgprDiff += Size;
            else
              SgprDiff -= Size;
          }
        }
      }

      for (auto InIt : Exp.InputLive) {
        unsigned Reg = InIt.first;
        LaneBitmask InMask = InIt.second;
        LaneBitmask MBBBeginMask;
        if (CrossLive.find(Reg) != CrossLive.end())
          MBBBeginMask = CrossLive[Reg];
        // Check mask which not live in BeginSlot  in exp input when
        // sink to live the input. Check mask which live in both BeginSlot and
        // exp output when hoist to kill the input.
        LaneBitmask ProfitMask =
            Exp.IsHoist ? (InMask & MBBBeginMask) : (InMask & (~MBBBeginMask));
        if (ProfitMask.any()) {
          // Update input live to avoid count same input more than once.
          NewInput[Reg] |= InMask;
          // Exp in not live at block input.
          // It will increase live for MBB.
          unsigned Size = getRegSize(Reg, ProfitMask, MRI, SIRI);

          LLVM_DEBUG(
              std::string movStr = Exp.IsHoist ? "input hoist:" : "input sink:";
              dbgs() << movStr << Register::virtReg2Index(Reg) << " " << Size);
          if (SIRI->isVGPR(MRI, Reg)) {
            LLVM_DEBUG(dbgs() << "v\n");
            if (Exp.IsHoist)
              VgprDiff -= Size;
            else
              VgprDiff += Size;
          } else {
            LLVM_DEBUG(dbgs() << "s\n");
            if (Exp.IsHoist)
              SgprDiff -= Size;
            else
              SgprDiff += Size;
          }
        }
      }
    } else {
      // When sink into loop, the input will live for every block inside loop.
      // The output will only lived between to blocks and the use blocks.
      // If MBB dominate any user of output live reg, It will still live in
      // MBB. So cannot count that output live reg as profit.
      // Hoist into loop is not supported now.
      for (auto OutIt : Exp.OutputLive) {
        unsigned Reg = OutIt.first;
        bool IsDomUser = false;
        for (MachineInstr &MI : MRI.use_nodbg_instructions(Reg)) {
          MachineBasicBlock *UserMBB = MI.getParent();
          if (DT->dominates(MBB, UserMBB)) {
            IsDomUser = true;
            break;
          }
        }
        if (IsDomUser)
          continue;

        LaneBitmask OutMask = OutIt.second;
        LaneBitmask MBBBeginMask;
        if (InputLive.find(Reg) != InputLive.end())
          MBBBeginMask = InputLive[Reg];
        LaneBitmask ProfitMask = OutMask & MBBBeginMask;
        if (MBBBeginMask.any()) {
          unsigned Size = getRegSize(Reg, ProfitMask, MRI, SIRI);
          LLVM_DEBUG(dbgs()
                     << "move:" << Register::virtReg2Index(Reg) << " " << Size);
          // Exp out live at block input.
          // It will descrease live for MBB.
          if (SIRI->isVGPR(MRI, Reg)) {
            LLVM_DEBUG(dbgs() << "v\n");
            VgprDiff -= Size;
          } else {
            LLVM_DEBUG(dbgs() << "s\n");
            SgprDiff -= Size;
          }
        }
      }

      for (auto InIt : Exp.InputLive) {
        unsigned Reg = InIt.first;
        LaneBitmask InMask = InIt.second;
        LaneBitmask MBBBeginMask;
        if (InputLive.find(Reg) != InputLive.end())
          MBBBeginMask = InputLive[Reg];
        // Check mask which not live in BeginSlot  in exp input.
        LaneBitmask ProfitMask = InMask & (~MBBBeginMask);
        if (ProfitMask.any()) {
          // Update input live to avoid count same input more than once.
          NewInput[Reg] |= InMask;
          // Exp in not live at block input.
          // It will increase live for MBB.
          unsigned Size = getRegSize(Reg, ProfitMask, MRI, SIRI);

          LLVM_DEBUG(dbgs()
                     << "add:" << Register::virtReg2Index(Reg) << " " << Size);
          if (SIRI->isVGPR(MRI, Reg)) {
            LLVM_DEBUG(dbgs() << "v\n");
            VgprDiff += Size;
          } else {
            LLVM_DEBUG(dbgs() << "s\n");
            SgprDiff += Size;
          }
        }
      }
    }

    if (IsVOutBound && VgprDiff > 0)
      continue;

    if (IsSOutBound && SgprDiff > 0)
      continue;
    llvm::mergeLiveRegSet(CrossLive, NewInput);
    Vgpr += VgprDiff;
    Sgpr += SgprDiff;
    if (IsClone)
      Exp.IsCloneOnly = true;
  }

  return std::make_pair(Vgpr, Sgpr);
}

void addExpCandidates(std::vector<SubExp> &SubExpCandidates,
                      std::vector<SubExp> &SubExps,
                      GCNRPTracker::LiveRegSet &UsedRegs) {
  SubExpCandidates.insert(SubExpCandidates.end(), SubExps.begin(),
                          SubExps.end());
  for (auto &Exp : SubExps) {
    if (Exp.IsHoist) {
      for (auto &Reg : Exp.TopRegs) {
        UsedRegs[Reg];
      }
    } else {
      for (auto &Reg : Exp.BottomRegs) {
        UsedRegs[Reg];
      }
    }
  }
}

bool tryToAddSubExps(
    Remat *Remat, HotBlock &HotBB, RematStatus &Status,
    std::vector<SubExp> &SubExpCandidates,
    std::vector<SubExp> &InBlockCloneSubExps,
    DenseMap<MachineBasicBlock *, MachineInstr *> &InBlockHotVInstMap,
    DenseMap<MachineBasicBlock *, MachineInstr *> &InBlockHotSInstMap,
    SmallVector<std::pair<MachineBasicBlock *, GCNRPTracker::LiveRegSet>>
        Candidates,
    int Vgpr, int Sgpr, const GCNRPTracker::LiveRegSet &SavingInputLive,
    const GCNRPTracker::LiveRegSet &SavingOutputLive,
    GCNRPTracker::LiveRegSet &PassThrus, GCNRPTracker::LiveRegSet &UsedRegs,
    MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
    const SIInstrInfo *SIII, const MachineLoopInfo *MLI, SlotIndexes *SI,
    LiveIntervals *LIS, MachineDominatorTree *DT, bool IsCanClone,
    bool IsVOutBound, bool IsSOutBound,
    GCNRPTracker::LiveRegSet &UnusedPassThrus, bool AllowPartialUseInSubExp) {
  std::vector<SubExp> PartialSubExps =
      buildSubExpCandidates(Remat, Candidates, PassThrus, MRI, SIRI, SIII, MLI,
                            SI, DT, IsCanClone, IsSOutBound, UnusedPassThrus,
                            Status.MemWriteMBBSet, AllowPartialUseInSubExp);

  GCNRPTracker::LiveRegSet TmpSavingInputLive = SavingInputLive;
  GCNRPTracker::LiveRegSet TmpSavingOutputLive = SavingOutputLive;
  std::pair<int, int> CurSaving = calculateSaving(
      HotBB, PartialSubExps, TmpSavingInputLive, TmpSavingOutputLive,
      IsVOutBound, IsSOutBound, IsCanClone, DT, MRI, SIRI);
  const int VLimit = Status.TargetVLimit;
  const int SLimit = Status.TargetSLimit;

  Vgpr += CurSaving.first;
  Sgpr += CurSaving.second;

  if (Vgpr <= VLimit && Sgpr <= SLimit) {
    // nrmSubExps can help reach target occupancy, add It to
    // subExpCandidates.
    addExpCandidates(SubExpCandidates, PartialSubExps, UsedRegs);
    return true;
  }

  if (EnableSubExpAggressive) {
    // Build candidates from passThrus  used in partialSubExps.
    GCNRPTracker::LiveRegSet SinkUsedRegs;
    for (auto &Exp : PartialSubExps) {
      for (auto &Reg : Exp.BottomRegs) {
        SinkUsedRegs[Reg];
      }
    }
    MapVector<MachineBasicBlock *, GCNRPTracker::LiveRegSet> HoistCandidates;
    for (auto &It : HotBB.InputLive) {
      unsigned Reg = It.first;
      // Skip reg which already used for sink exp.
      if (SinkUsedRegs.count(Reg))
        continue;
      if (UsedRegs.count(Reg))
        continue;
      // Skip unsafe reg.
      if (!isSafeCandidate(Remat, Reg, MRI, SIRI, SIII, /*IsSink*/ false)) {
        LLVM_DEBUG(dbgs() << " is not safe to hoist\n");
        continue;
      }
      // DefMI is already checked in isSafeCandidate.
      MachineInstr *DefMI = MRI.getUniqueVRegDef(Reg);
      MachineBasicBlock *DefMBB = DefMI->getParent();
      DenseSet<MachineBasicBlock *> UseMBBSet;
      // Make sure all uses not in Def block are in same block.
      for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
        MachineBasicBlock *UseMBB = UseMI.getParent();
        if (UseMBB == DefMBB)
          continue;
        UseMBBSet.insert(UseMBB);
      }

      if (UseMBBSet.size() != 1)
        continue;
      MachineBasicBlock *UseMBB = *UseMBBSet.begin();
      GCNRPTracker::LiveRegSet &UseInMBB = HoistCandidates[UseMBB];
      UseInMBB[Reg] = getRegMask(DefMI->getOperand(0), MRI);
    }

    // Build exp dag on define blocks.
    std::vector<SubExp> HoistSubExpCandidates;
    // Save profit candidates into list.
    for (auto It : HoistCandidates) {
      MachineBasicBlock *UseMBB = It.first;
      // Try to remove out reg def sub exp from DefMBB.
      GCNRPTracker::LiveRegSet &UseInMBB = It.second;
      // Go up on the dag until reach share node.
      auto SubExps = buildSubExpFromCandidatesTopBottom(Remat, UseInMBB, UseMBB,
                                                        SIRI, SIII, MRI);
      for (SubExp &SubExp : SubExps) {
        if (!canHelpPressureWhenHoist(SubExp, MRI, MLI, IsSOutBound))
          continue;
        SubExp.IsHoist = true;
        HoistSubExpCandidates.emplace_back(SubExp);
      }
    }

    std::pair<int, int> HoistSaving = calculateSaving(
        HotBB, HoistSubExpCandidates, TmpSavingInputLive, TmpSavingOutputLive,
        IsVOutBound, IsSOutBound, IsCanClone, DT, MRI, SIRI);

    int HoistVgpr = Vgpr + HoistSaving.first;
    int HoistSgpr = Sgpr + HoistSaving.second;

    if ((HoistVgpr <= VLimit && HoistSgpr <= SLimit) ||
        // If status not balance, do the remat even cannot reach target.
        // TODO: check the result not help even one occupancy.
        (!HoistSubExpCandidates.empty() && !Status.NotBalance &&
         TargetOccupancy != 0)) {
      // nrmSubExps can help reach target occupancy, add It to
      // subExpCandidates.
      addExpCandidates(SubExpCandidates, PartialSubExps, UsedRegs);
      addExpCandidates(SubExpCandidates, HoistSubExpCandidates, UsedRegs);

      return true;
    }
  }

  if (EnableVmemDegree &&
      // Only expect vmem when last tryToAddSubExps.
      // If not, AllowPartialUseInSubExp will no chance to be true.
      (AllowPartialUseInSubExp || !EnableSubExpAggressive)) {
    // Assume vmemLdSize could be optimized by not parallel.
    if (((Vgpr - HotBB.VmemLdInputSize) <= VLimit ||
         (Vgpr - HotBB.VmemLdOutputSize) <= VLimit) &&
        Sgpr <= SLimit) {
      // nrmSubExps can help reach target occupancy, add It to
      // subExpCandidates.
      addExpCandidates(SubExpCandidates, PartialSubExps, UsedRegs);
      return true;
    }
  }

  int VDistance = Vgpr - (int)VLimit;
  int SDistance = Status.TargetOcc > 4 ? (Sgpr - (int)SLimit) : 0;
  int VSaved = HotBB.MaxPressures.first - Vgpr;
  int SSaved = HotBB.MaxPressures.second - Sgpr;
  // Try to add inBlockCloneSubExps.
  if (!tryRematInHotSpot(*HotBB.MBB, Status, VDistance, SDistance, VSaved,
                         SSaved, InBlockCloneSubExps, InBlockHotVInstMap,
                         InBlockHotSInstMap, LIS, MRI, SIRI, SIII)) {
    // return false always when not allow partialUseInSubExp, It will try again
    // with partialUseInSubExp enabled.
    if (!AllowPartialUseInSubExp)
      return false;
    // If status not balance, do the remat even cannot reach target.
    // TODO: check the result not help even one occupancy.
    if (!Status.NotBalance && TargetOccupancy == 0)
      return false;
  }
  // nrmSubExps can help reach target occupancy, add It to
  // subExpCandidates.
  addExpCandidates(SubExpCandidates, PartialSubExps, UsedRegs);
  return true;
}

// Remat passthru regs per hot block.
// Reason to do It per block is to make sure passthru reuse is precise.
// If try remat on all hot blocks together, the passthru might be on one block,
//  reuse in on another block which the reg is not passthru there.
bool perBlockPassthruRemat(Remat *Remat, std::vector<HotBlock> &HotBlocks,
                           RematStatus &Status,
                           GCNRPTracker::LiveRegSet &LiveRegCandidates,
                           const GCNSubtarget *ST, LiveIntervals *LIS,
                           const MachineLoopInfo *MLI, MachineDominatorTree *DT,
                           MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                           const SIInstrInfo *SIII) {
  bool IsUpdated = false;
  bool IsCanClone = EnableSubExpClone || EnableSubExpAggressive;

  SlotIndexes *SlotIndexes = LIS->getSlotIndexes();
  // Sort hot blocks by pressure first.
  // The hot block with higher pressure is easier to fail.
  // If fail, fail fast. It It works, save the subExpCandidates. The
  // subExpCandidates may help other hotblocks.
  std::sort(HotBlocks.begin(), HotBlocks.end(),
            [&ST](const HotBlock &A, const HotBlock &B) {
              return pressureHigher(A.MaxPressures.first, A.MaxPressures.second,
                                    B.MaxPressures.first, B.MaxPressures.second,
                                    ST);
            });

  std::vector<SubExp> SubExpCandidates;
  // For inBlock remat clone.
  std::vector<SubExp> InBlockCloneSubExps;
  DenseMap<MachineBasicBlock *, MachineInstr *> InBlockHotVInstMap;
  DenseMap<MachineBasicBlock *, MachineInstr *> InBlockHotSInstMap;

  // Save used passThrus to avoid use same reg on different MBB.
  GCNRPTracker::LiveRegSet UsedPassThrus;
  // Save moved regs to avoid use same reg hoist and sink.
  GCNRPTracker::LiveRegSet UsedRegs;

  const int VLimit = Status.TargetVLimit;
  const int SLimit = Status.TargetSLimit;
  // Collect passthru for hot block.
  // Try remat on It.
  for (auto &It : HotBlocks) {
    MachineBasicBlock *MBB = It.MBB;

    const GCNRPTracker::LiveRegSet InputLive = Status.MBBInputLiveMap[MBB];
    const GCNRPTracker::LiveRegSet OutputLive = Status.MBBOutputLiveMap[MBB];

    It.InputLive = InputLive;

    // Add pressure by 1 to consider spill to vgpr.
    const int PressureDelta = -1;
    int Vgpr = It.MaxPressures.first - PressureDelta;
    int Sgpr = It.MaxPressures.second;
    bool IsVOutBound = Vgpr > VLimit;
    bool IsSOutBound = Sgpr > SLimit;
    // savingInputLive is used to calculate saving which will be modified to
    // avoid count same input multiple times.
    GCNRPTracker::LiveRegSet SavingInputLive = InputLive;
    GCNRPTracker::LiveRegSet SavingOutputLive = OutputLive;
    std::pair<int, int> CurSaving =
        calculateSaving(It, SubExpCandidates, SavingInputLive, SavingOutputLive,
                        IsVOutBound, IsSOutBound, IsCanClone, DT, MRI, SIRI);

    Vgpr += CurSaving.first;
    Sgpr += CurSaving.second;

    if (Vgpr <= VLimit && Sgpr <= SLimit)
      continue;

    // Collect pass thru regs.
    GCNRPTracker::LiveRegSet PassThrus =
        collectPassThrus(MBB, InputLive, OutputLive,
                         LiveRegCandidates, MRI, IsCanClone);

    // Group pass thru regs by def MBB.
    SmallVector<std::pair<MachineBasicBlock *, GCNRPTracker::LiveRegSet>>
        Candidates = groupPassThruByDefBlock(Remat, PassThrus, UsedPassThrus,
                                             MRI, SIRI, SIII);
    // unUsedPassThrus used to collect passThru which is skipped when build
    // subExp.
    GCNRPTracker::LiveRegSet UnusedPassThrus;
    // Build exp dag on define blocks.
    bool AllowPartialUseInSubExp = false;
    if (tryToAddSubExps(
            Remat, It, Status, SubExpCandidates, InBlockCloneSubExps,
            InBlockHotVInstMap, InBlockHotSInstMap, Candidates, Vgpr, Sgpr,
            SavingInputLive, SavingOutputLive, PassThrus, UsedRegs, MRI, SIRI,
            SIII, MLI, SlotIndexes, LIS, DT, IsCanClone, IsVOutBound,
            IsSOutBound, UnusedPassThrus, AllowPartialUseInSubExp)) {
      // Remove unusedPassThrus from passThrus first.
      llvm::andNotLiveRegSet(PassThrus, UnusedPassThrus);
      llvm::mergeLiveRegSet(UsedPassThrus, PassThrus);
      continue;
    }
    // If cannot clone, don't need to try partialUseInSubExp which must clone.
    if (!IsCanClone)
      return false;

    // Partial use subExp may result  count caused by clone.
    // Only try It when enable aggressive remat.
    if (!EnableSubExpAggressive)
      return false;

    AllowPartialUseInSubExp = true;
    if (!tryToAddSubExps(
            Remat, It, Status, SubExpCandidates, InBlockCloneSubExps,
            InBlockHotVInstMap, InBlockHotSInstMap, Candidates, Vgpr, Sgpr,
            SavingInputLive, SavingOutputLive, PassThrus, UsedRegs, MRI, SIRI,
            SIII, MLI, SlotIndexes, LIS, DT, IsCanClone, IsVOutBound,
            IsSOutBound, UnusedPassThrus, AllowPartialUseInSubExp)) {
      return false;
    }
    // Just merge all passThrus after tryToAddSubExps allow partialUseInSubExp.
    llvm::mergeLiveRegSet(UsedPassThrus, PassThrus);
  }

  // Apply changes.
  {
    // sort subExpCandidates to make sure input use apply before output use if a
    // reg is input and output of subExps.
    LLVM_DEBUG(for (SubExp &Exp : SubExpCandidates) { Exp.dump(MRI, SIRI); });
    sortSubExpCandidates(SubExpCandidates);

    for (SubExp &Exp : SubExpCandidates) {
      // Skip exp which is cleared in sort for hoist sink conflict.
      if (Exp.SUnits.empty())
        continue;
      LLVM_DEBUG(Exp.dump(MRI, SIRI));
      if (Exp.IsHoist) {
        applySubExpMoveNearDefine(Exp, MRI, SlotIndexes, SIII, SIRI);
      } else {
        if (Exp.IsCloneOnly)
          applySubExpCloneNearUser(Exp, HotBlocks, DT, MRI, SlotIndexes, SIII,
                                   SIRI);
        else
          applySubExpMoveNearUser(Exp, MRI, DT, SlotIndexes);
      }
    }

    for (SubExp &Exp : InBlockCloneSubExps) {
      applySubExpCloneNearUserInBlock(
          Exp, InBlockHotVInstMap, InBlockHotSInstMap, MRI, SlotIndexes, SIRI);
    }
    // Try to see possible occupancy could reach, then dicide a target.
    // Apply remat.
    IsUpdated = SubExpCandidates.size();
  }

  return IsUpdated;
}

int getVMemLdSize(MachineBasicBlock &MBB, const SIInstrInfo *SIII,
                  const SIRegisterInfo *SIRI, const MachineRegisterInfo &MRI) {
  int VmemLdSize = 0;
  // Collect vmemLd when enable split.
  for (MachineInstr &MI : MBB) {
    bool IsHighLatency = SIII->isHighLatencyInstruction(MI);
    if (!IsHighLatency)
      continue;
    if (!(MI.mayLoad() &&
          // Skip case like atomic which not return value.
          MI.getNumDefs() > 0))
      continue;
    // a vmem ld.
    MachineOperand &Dst = MI.getOperand(0);
    LaneBitmask Mask = llvm::getRegMask(Dst, MRI);
    unsigned Size = llvm::getRegSize(Dst.getReg(), Mask, MRI, SIRI);
    VmemLdSize += Size;
  }
  return VmemLdSize;
}

} // namespace

bool groupRemat(Remat *Remat, MachineFunction &MF, MachineLoopInfo *MLI,
                LiveIntervals *LIS, MachineDominatorTree *DT,
                MachinePostDominatorTree *PDT, AliasAnalysis *AA) {
  if (MF.size() < 2)
    return false;
  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();

  const SIInstrInfo *SIII = ST->getInstrInfo();
  const SIRegisterInfo *SIRI = ST->getRegisterInfo();

  auto &MRI = MF.getRegInfo();

  RematStatus Status = getRematStatus(MF, MLI, LIS, MRI, ST);

  const unsigned MaxOcc = ST->getWavesPerEU(MF.getFunction()).second;
  if (Status.TargetOcc >= MaxOcc)
    return false;

  unsigned VLimit = Status.TargetVLimit;
  unsigned SLimit = Status.TargetSLimit;

  int RematVCnt = Status.MaxVPressure - VLimit;
  int RematSCnt = Status.MaxSPressure - SLimit;

  bool IsSGPRSpill = false;
  if (RematSCnt > 0) {
    IsSGPRSpill = nearSgprSpill(Status.MaxSPressure, ST, MF);
  }

  // If bound by lds, skip.
  if ((Status.TargetOcc + 1) > ST->getOccupancyWithWorkGroupSizes(MF).second &&
      !IsSGPRSpill)
    return false;

  bool IsBothOutLimit = RematVCnt > 0 && RematSCnt > 0;
  // TODO: use check wqm and support vreg remat.
  bool IsCheckWQM = MF.getFunction().getCallingConv() == CallingConv::AMDGPU_PS;
  RematVCnt = IsCheckWQM & false;

  // Remat on every hot block.

  // Collect all hot blocks.
  std::vector<HotBlock> HotBlocks;
  for (MachineBasicBlock &MBB : MF) {
    // Collect reg pressure.
    auto &RP = Status.MBBPressureMap[&MBB];
    unsigned MaxLocalVPressure = RP.getVGPRNum(ST->hasGFX90AInsts());
    unsigned MaxLocalSPressure = RP.getMaxSGPR();

    MaxLocalSPressure += RegForVCC;

    if (!EnableInBlockRemat) {
      if (MaxLocalVPressure <= VLimit && MaxLocalSPressure <= SLimit)
        continue;
    }

    // Move inst which input is imm/pass thru reg/out reg to help pressure.
    if (tryHoldPacifist(MBB, LIS, MRI, SIRI, AA, Status)) {
      MaxLocalVPressure = 0;
      MaxLocalSPressure = 0;
      collectMBBPressure(MBB, LIS, ST, MaxLocalVPressure, MaxLocalSPressure,
                         Status);

      MaxLocalSPressure += RegForVCC;
    }
    if (MaxLocalVPressure <= VLimit && MaxLocalSPressure <= SLimit)
      continue;

    // When both vgpr sgpr out limit, only help vgpr.
    if (IsBothOutLimit && MaxLocalVPressure <= VLimit)
      continue;
    GCNRPTracker::LiveRegSet LiveSet;
    HotBlocks.push_back({&MBB, LiveSet,
                         std::make_pair(MaxLocalVPressure, MaxLocalSPressure),
                         0, 0});
  }
  // Collect vmemLdInput/OutputSize.
  if (EnableVmemDegree) {
    DenseMap<MachineBasicBlock *, unsigned> OutputVMemLdSizeMap;
    for (auto It : HotBlocks) {
      MachineBasicBlock *MBB = It.MBB;
      // Collect vmemLd when enable split.
      int VmemLdSize = getVMemLdSize(*MBB, SIII, SIRI, MRI);
      if (VmemLdSize) {
        OutputVMemLdSizeMap[MBB] = VmemLdSize;
      }
    }
    for (auto &It : HotBlocks) {
      MachineBasicBlock *MBB = It.MBB;

      auto OIt = OutputVMemLdSizeMap.find(MBB);
      if (OIt != OutputVMemLdSizeMap.end())
        It.VmemLdOutputSize = OIt->second;

      if (MBB->pred_size() != 1)
        continue;

      MachineBasicBlock *Pred = *MBB->pred_begin();
      OIt = OutputVMemLdSizeMap.find(Pred);
      if (OIt != OutputVMemLdSizeMap.end()) {
        It.VmemLdInputSize = OIt->second;
      } else {
        if (Pred->getFirstTerminator() != Pred->end())
          continue;
        if (Pred->empty())
          continue;
        bool IsHighLatency = SIII->isHighLatencyInstruction(Pred->back());
        if (!IsHighLatency)
          continue;
        int VmemLdSize = getVMemLdSize(*Pred, SIII, SIRI, MRI);
        It.VmemLdInputSize = VmemLdSize;
      }
    }
  }

  if (EnableUniformVectorToScalar) {
    if (rematUniformVgprToSgpr(Remat, MF, Status, HotBlocks, LIS, MRI, SIRI,
                               SIII, MLI)) {
      // Rebuild LIS.
      LIS->reanalyze(MF);
      Status = getRematStatus(MF, MLI, LIS, MRI, ST);
      bool IsSgprSpilled = nearSgprSpill(Status.MaxSPressure, ST, MF);
      if (IsSgprSpilled) {
        bool IsNearTarget = false;
        hotBlockRemat(Remat, MF, MLI, LIS, DT, PDT, IsNearTarget);
        // Rebuild LIS.
        LIS->reanalyze(MF);
        Status = getRematStatus(MF, MLI, LIS, MRI, ST);
      }

      for (auto &It : HotBlocks) {
        MachineBasicBlock *MBB = It.MBB;

        // Update pressure.
        auto &RP = Status.MBBPressureMap[MBB];
        unsigned MaxLocalVPressure = RP.getVGPRNum(ST->hasGFX90AInsts());
        unsigned MaxLocalSPressure = RP.getMaxSGPR();

        MaxLocalSPressure += RegForVCC;
        It.MaxPressures.first = MaxLocalVPressure;
        It.MaxPressures.second = MaxLocalSPressure;
      }
    }
  }

  // Collect all live reg which cross hot blocks.
  GCNRPTracker::LiveRegSet LiveRegCandidates;
  for (auto It : HotBlocks) {
    MachineBasicBlock *MBB = It.MBB;

    const GCNRPTracker::LiveRegSet InputLive = Status.MBBInputLiveMap[MBB];

    const GCNRPTracker::LiveRegSet OutputLive = Status.MBBOutputLiveMap[MBB];

    llvm::mergeLiveRegSet(LiveRegCandidates, InputLive);
    llvm::mergeLiveRegSet(LiveRegCandidates, OutputLive);
  }

  // Check min VGPR bound.
  BlockSet PressureUnderLimitSet;
  if (EnableSubExpMinReg) {
    for (auto &It : HotBlocks) {
      MachineBasicBlock *MBB = It.MBB;
      unsigned MaxLocalVGPR = 0;
      unsigned MaxLocalSGPR = 0;
      llvm::getRegBound(MBB, MRI, SIRI, SIII, LIS, MaxLocalVGPR, MaxLocalSGPR);

      if (MaxLocalVGPR < VLimit && MaxLocalSGPR < SLimit) {
        PressureUnderLimitSet.insert(MBB);
      } else {
        if (MaxLocalVGPR < It.MaxPressures.first)
          It.MaxPressures =
              std::make_pair(MaxLocalVGPR, It.MaxPressures.second);
        if (MaxLocalSGPR < It.MaxPressures.second)
          It.MaxPressures = std::make_pair(It.MaxPressures.first, MaxLocalSGPR);
      }
    }
  }

  bool IsUpdated =
      perBlockPassthruRemat(Remat, HotBlocks, Status, LiveRegCandidates, ST,
                            LIS, MLI, DT, MRI, SIRI, SIII);

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
  AliasAnalysis *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();

  {
    MachineCycleInfo CI;
    CI.compute(MF);
    auto TTI = MF.getTarget().getTargetTransformInfo(MF.getFunction());
    MachineUniformityInfo MachineUniformity =
        llvm::computeMachineUniformityInfo(MF, CI, *DT,
                                           /*HasBranchDivergence*/ true);
    TotalUniformInsts.clear();
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (MachineUniformity.isUniform(&MI)) {
          TotalUniformInsts.insert(&MI);
        }
      }
    }
  }

  // LLVM_DEBUG(pressure::write_pressure(MF, LIS, R"(D:\Temp\d.json)"));
  //  For non-cs/ps, set target occ as 4.
  bool IsNearTarget = false;
  bool IsFinalUpdated = false;
  bool IsUpdated = hotBlockRemat(this, MF, MLI, LIS, DT, PDT, IsNearTarget);
  IsFinalUpdated |= IsUpdated;
  if (EnableSubExp) {
    if (IsUpdated) {
      // Rebuild LIS.
      LIS->reanalyze(MF);
    }

    IsUpdated = groupRemat(this, MF, MLI, LIS, DT, PDT, AA);

    IsFinalUpdated |= IsUpdated;
  }
  return IsFinalUpdated;
}

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
