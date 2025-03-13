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

MachineBasicBlock *NearestCommonDominator(MachineDominatorTree *DT,
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
    MachineBasicBlock *BDom = NearestCommonDominator(DT, BBSet);
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
  // hotBlockRemat will fail it when process BB.

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
  for (unsigned i = 0; i < OpNum; i++) {
    MachineOperand &Op = DefMI->getOperand(i);
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
  // More occupancy can help more than latency cost to reach it.
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
  // Skip processing current block if it has only debug instructions
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
      dbgs() << "output live"; for (auto &it
                                    : Status.MBBOutputLiveMap) {
        unsigned Idx = it.first->getNumber();
        auto LiveReg = it.second;
        dbgs() << "MBB" << Idx << ":";
        llvm::dumpLiveSet(LiveReg, SIRI);
      } dbgs() << "input live";
      for (auto &it
           : Status.MBBInputLiveMap) {
        unsigned Idx = it.first->getNumber();
        auto LiveReg = it.second;
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
      // still before LiveInfo.BB, it is still live.
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
          for (int i = 0; i < MOSize; i++) {
            if (SharedMask & (1 << i)) {
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
      // moved after LiveInfo.BB, it is not live anymore.
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
      // The reg might share with other candidates,  check it here.
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
      // If input not live in hotspot, move it cross hotspot should have
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
            [](RematNode &i, RematNode &j) { return i.Size > j.Size; });
}

// For case like
//   %477:sreg_32_xm0 = S_AND_B32 %472.sub0:sreg_64_xexec, %304:sreg_32_xm0,
//   implicit-def dead $scc; xb.uniform
//  S_CMP_EQ_U32 %302:sreg_32_xm0, %475:sreg_32_xm0, implicit-def $scc;
//  xb.uniform %2489:sreg_32_xm0 = S_CSELECT_B32 %477:sreg_32_xm0, 16, implicit
//  killed $scc; xb.uniform
// Sink S_AND right before S_CSELECT will overwrite SCC.
// To avoid it, skip case when DefMI and UseMI has implicit define use.
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

  for (size_t i = 0; i < CloneList.size(); i++) {
    auto *Node = CloneList[i];
    unsigned Reg = Node->Reg;
    MachineInstr *DefMI = Node->DefMI;
    // Group user in same blocks.
    BlockSet &UserSet = UserSetList[i];

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
                 SmallVector<MachineInstr *, 2> &userMIs) {
  for (MachineInstr *UseMI : userMIs) {
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
    unsigned Reg, BlockMap<SmallVector<MachineInstr *, 2>> &userBlocks,
    DenseSet<MachineBasicBlock *> &UserMBBSet,
    std::vector<BlockLiveInfo> &hotBlocks, MachineDominatorTree *DT) {
  // Collect hot blocks which Exp is live in.
  DenseSet<MachineBasicBlock *> hotBlockSet;
  for (BlockLiveInfo &hotBlock : hotBlocks) {
    if (hotBlock.InputLive.count(Reg)) {
      hotBlockSet.insert(hotBlock.BB);
    }
  }

  // For userBlocks which dominate all hotBlocks, don't need to clone because
  // the value not cross hotBlocks when later blocks are cloned.
  // For userBlocks which dominated by all hotBlocks, they could share clones
  // because once after hot block, the pressure is OK.
  DenseSet<MachineBasicBlock *> afterHotRangeMBBs;
  for (MachineBasicBlock *MBB : UserMBBSet) {
    // Always clone in hot block.
    if (hotBlockSet.count(MBB))
      continue;

    bool IsDomAllHotBlocks = true;
    bool IsDomedByAllHotBlocks = true;
    for (MachineBasicBlock *hotMBB : hotBlockSet) {
      if (!DT->dominates(MBB, hotMBB)) {
        IsDomAllHotBlocks = false;
      }
      if (!DT->dominates(hotMBB, MBB)) {
        IsDomedByAllHotBlocks = false;
      }
      if (!IsDomAllHotBlocks && !IsDomedByAllHotBlocks) {
        break;
      }
    }
    if (IsDomAllHotBlocks) {
      userBlocks.erase(MBB);
    } else if (IsDomedByAllHotBlocks) {
      afterHotRangeMBBs.insert(MBB);
    }
  }

  // Split after hotRange block set by domtree.
  DenseMap<MachineBasicBlock *, BlockSet> DomMap;
  if (!afterHotRangeMBBs.empty()) {
    for (auto it : afterHotRangeMBBs) {
      MachineBasicBlock *MBB = it;
      for (auto it2 : afterHotRangeMBBs) {
        MachineBasicBlock *MBB2 = it2;
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
    for (auto it : afterHotRangeMBBs) {
      MachineBasicBlock *MBB = it;
      auto &Dom = DomMap[MBB];
      for (MachineBasicBlock *domedMBB : Dom) {
        // Remove domedMBB.
        DomMap.erase(domedMBB);
        UserMBBSet.erase(domedMBB);
      }
    }
  }

  return DomMap;
}

// Look for an earlier insert point if the InstructionToMove
// writes to scc and scc is live at the CurrentInsertPoint.
static MachineBasicBlock::iterator AdjustInsertPointToAvoidSccSmash(
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
static MachineBasicBlock::iterator AdjustInsertPointForSubExpToAvoidSccSmash(
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
static bool WillSmashSccAtLocation(MachineInstr *MI, MachineBasicBlock *MBB,
                                   MachineBasicBlock::iterator Location) {
  // It is ok to pass nullptr to `modifiesRegister` for TRI here since
  // SCC has no subreg/suprereg relationships.
  return MI->modifiesRegister(AMDGPU::SCC, nullptr) &&
         llvm::IsSccLiveAt(MBB, Location);
}

void ApplyCloneRemat(Remat *Remat, RematNode &Node,
                     std::vector<BlockLiveInfo> &hotBlocks,
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
  for (auto useIt = MRI.use_instr_nodbg_begin(Reg);
       useIt != MRI.use_instr_nodbg_end();) {
    MachineInstr &UseMI = *(useIt++);
    UserMap[UseMI.getParent()].emplace_back(&UseMI);
    UserMBBSet.insert(UseMI.getParent());
  }

  DenseMap<MachineBasicBlock *, BlockSet> DomMap =
      reduceClonedMBBs(Reg, UserMap, UserMBBSet, hotBlocks, DT);

  for (auto useIt : UserMap) {
    MachineBasicBlock *MBB = useIt.first;
    // Skip same block uses.
    if (MBB == DefMI->getParent()) {
      continue;
    }
    // Skip MBB which share clone from other MBBs.
    if (UserMBBSet.count(MBB) == 0)
      continue;

    unsigned NewReg = MRI.createVirtualRegister(RC);
    auto NewDef = BuildMI(MF, DL, Desc).addDef(NewReg);
    for (unsigned i = 1; i < OpNum; i++) {
      NewDef = NewDef.add(DefMI->getOperand(i));
    }

    MachineInstr *InsertPointMI = useIt.second.front();
    SlotIndex lastSlot = SlotIndexes->getInstructionIndex(*InsertPointMI);

    for (MachineInstr *UseMI : useIt.second) {
      SlotIndex slot = SlotIndexes->getInstructionIndex(*UseMI);
      if (lastSlot > slot) {
        lastSlot = slot;
        InsertPointMI = UseMI;
      }
    }

    MachineBasicBlock::iterator InsertPoint = AdjustInsertPointToAvoidSccSmash(
        DefMI, InsertPointMI->getParent(), InsertPointMI, MRI, SIRI, SIII);

    for (MachineMemOperand *MO : DefMI->memoperands()) {
      NewDef->addMemOperand(MF, MO);
    }

    MBB->insert(InsertPoint, NewDef);

    SlotIndexes->insertMachineInstrInMaps(*NewDef);

    SmallVector<MachineInstr *, 2> &userMIs = useIt.second;
    updateUsers(Reg, NewReg, IsSubRegDef, userMIs);

    // update users in dom MBBs.
    auto domMapIt = DomMap.find(MBB);
    if (domMapIt != DomMap.end()) {
      for (MachineBasicBlock *UpdateMBB : domMapIt->second) {
        SmallVector<MachineInstr *, 2> &userMIs = UserMap[UpdateMBB];
        updateUsers(Reg, NewReg, IsSubRegDef, userMIs);
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

void ApplyOneDefOneUseRemat(RematNode &Node, MachineRegisterInfo &MRI,
                            SlotIndexes *slotIndexes,
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

  InsertPoint = AdjustInsertPointToAvoidSccSmash(DefMI, MBB, InsertPoint, MRI,
                                                 SIRI, SIII);

  // Move instruction to new location.
  DefMI->removeFromParent();
  InsertPoint->getParent()->insert(InsertPoint, DefMI);

  // Update slot index.
  slotIndexes->removeSingleMachineInstrFromMaps(*DefMI);
  slotIndexes->insertMachineInstrInMaps(*DefMI);
}

void ApplyRemat(Remat *Remat, MapVector<Register, RematNode> &RematMap,
                std::vector<BlockLiveInfo> &hotBlocks,
                MachineDominatorTree *DT, SlotIndexes *slotIndexes,
                MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                const SIInstrInfo *SIII, MachineFunction &MF) {
  std::vector<RematNode> UpdateList;
  for (auto &it : RematMap) {
    UpdateList.emplace_back(it.second);
  }
  // Sort update list with slotIndex to make sure def moved before use.
  // If use moved before def, it might not be the first use anymore.
  std::sort(UpdateList.begin(), UpdateList.end(),
            [&slotIndexes](RematNode &i, RematNode &j) {
              SlotIndex a = slotIndexes->getInstructionIndex(*i.DefMI);
              SlotIndex b = slotIndexes->getInstructionIndex(*j.DefMI);
              return a < b;
            });

  for (RematNode &Node : UpdateList) {
    if (Node.Kind == RematNode::RematKind::OneDefOneUse) {
      ApplyOneDefOneUseRemat(Node, MRI, slotIndexes, SIRI, SIII);
    } else if (Node.Kind == RematNode::RematKind::Clone) {
      ApplyCloneRemat(Remat, Node, hotBlocks, DT, MRI, slotIndexes, SIRI, SIII,
                      MF);
    }
  }
}

void dumpRematMap(MapVector<Register, RematNode> &RematMap,
                  const SIRegisterInfo *SIRI) {
  dbgs() << "\n rematMap: \n";
  for (auto it : RematMap) {
    int Reg = it.first;
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
  RematStatus status = getRematStatus(MF, MLI, LIS, MRI, ST);

  const unsigned MaxOcc = ST->getWavesPerEU(MF.getFunction()).second;
  if (status.TargetOcc >= MaxOcc)
    return false;

  unsigned VLimit = status.TargetVLimit;
  unsigned SLimit = status.TargetSLimit;

  int rematSCnt = status.MaxSPressure - SLimit;
  // when agressive sgpr remat, reserve some for allocation lost.
  if (EnableAggressive)
    rematSCnt += NearTargetRegLimit;

  bool IsSGPRSpill = false;
  if (rematSCnt > 0) {
    IsSGPRSpill = nearSgprSpill(status.MaxSPressure, ST, MF);
  }

  bool IsForceRematSgpr = IsSGPRSpill | status.NotBalance;

  // If bound by lds, skip.
  if (status.TargetOcc > ST->getOccupancyWithWorkGroupSizes(MF).second &&
      !IsForceRematSgpr)
    return false;

  MachineBasicBlock *EntryMBB = &MF.front();

  auto *SlotIndexes = LIS->getSlotIndexes();

  // Reg which already marked remat.
  MapVector<Register, RematNode> VRematMap;
  MapVector<Register, RematNode> SRematMap;
  // Reg which cannot move around to remat.
  DenseSet<unsigned> PinnedRegSet;
  std::vector<BlockLiveInfo> hotBlocks;
  for (auto it = po_begin(EntryMBB); it != po_end(EntryMBB); it++) {
    MachineBasicBlock *MBB = *it;
    auto &RP = status.MBBPressureMap[MBB];
    // ignore block not hot.
    if (RP.getVGPRNum(ST->hasGFX90AInsts()) < status.TargetVLimit &&
        (RP.getMaxSGPR() + RegForVCC + status.InputPhysicalSPressure) <
            status.TargetSLimit)
      continue;
    // Collect reg pressure.
    unsigned maxVPressure = 0;
    unsigned maxSPressure = 0;
    const GCNRPTracker::LiveRegSet inputLive = status.MBBInputLiveMap[MBB];

    const GCNRPTracker::LiveRegSet outputLive = status.MBBOutputLiveMap[MBB];
    LLVM_DEBUG(
        dumpHotBlock(inputLive, VRematMap, SRematMap, MBB->getNumber(), SIRI));

    GCNDownwardRPTracker Tracker(*LIS);

    Tracker.reset(*MBB->begin(), &inputLive);

    for (MachineInstr &MI : *MBB) {
      if (MI.isDebugInstr())
        continue;
      Tracker.advance();
      auto LISLR = Tracker.getLiveRegs();
      // Update live set for things already remated.
      updateLiveInfo(VRematMap, LISLR, inputLive, MBB, RPOTIndexMap);
      updateLiveInfo(SRematMap, LISLR, inputLive, MBB, RPOTIndexMap);

      const GCNRPTracker::LiveRegSet &liveSet = LISLR;
      unsigned VPressure = 0;
      unsigned SPressure = 0;
      CollectLiveSetPressure(liveSet, MRI, SIRI, VPressure, SPressure);
      if (maxVPressure < VPressure)
        maxVPressure = VPressure;
      if (maxSPressure < SPressure)
        maxSPressure = SPressure;
    }
    maxSPressure += RegForVCC + status.InputPhysicalSPressure;
    if (maxVPressure <= VLimit && maxSPressure <= SLimit)
      continue;

    // Build block live info.
    // Use outputLive for EntryMBB.
    BlockLiveInfo LiveInfo = {MBB, maxSPressure, maxVPressure,
                              MBB != EntryMBB ? inputLive : outputLive};
    // Skip entry block when save hotBlock to reduce clone because not clone in
    // entry block.
    if (MBB != EntryMBB)
      hotBlocks.emplace_back(LiveInfo);
    GCNRPTracker::LiveRegSet CandidateRegs = LiveInfo.InputLive;

    // Update reg pressure based on remat list.
    InstSet VReducedInsts;
    InstSet SReducedInsts;
    int VReduced = getReducedSize(VRematMap, CandidateRegs, VReducedInsts, MRI,
                                  LiveInfo, RPOTIndexMap);
    int SReduced = getReducedSize(SRematMap, CandidateRegs, SReducedInsts, MRI,
                                  LiveInfo, RPOTIndexMap);

    // Calculate size need to be remat.
    int rematVCnt = maxVPressure - VReduced - VLimit;
    int rematSCnt = maxSPressure - SReduced - SLimit;

    bool IsSGPRSpill = false;
    if (rematSCnt > 0) {
      IsSGPRSpill = nearSgprSpill(maxSPressure, ST, MF);
    }
    bool IsForceRematSgpr = IsSGPRSpill || status.NotBalance;
    // Try to add candidates into remat list.

    int newRematSCnt = 0;
    if (rematSCnt > 0) {
      // Build candidate nodes.
      std::vector<RematNode> SRematCandidates;
      buildRematCandiates(SRematCandidates, CandidateRegs, PinnedRegSet, MRI,
                          SIII, SIRI, /*IsVGPR*/ false);

      LLVM_DEBUG(dumpCandidates(SRematCandidates, MBB->getNumber(), SIRI));
      std::vector<RematNode> SRematList;
      // Filter candidates.
      newRematSCnt = filterRematCandiates(SRematCandidates, SRematList,
                                          PinnedRegSet, DT, PDT, MLI, MRI,
                                          /*IsVGPR*/ false, status.MemBound);
      if (newRematSCnt > rematSCnt) {
        // Has enough remat node to cover rematCnt.
        int rematCnt = 0;
        for (RematNode &Node : SRematList) {
          SRematMap[Node.Reg] = Node;
          rematCnt += Node.Size;
          if (rematCnt > rematSCnt && !EnableAggressive)
            break;
        }
        newRematSCnt = 0;
      } else {

        for (RematNode &Node : SRematList) {
          SReducedInsts.insert(Node.DefMI);
        }
        // Check shared size.
        int SharedReducedSize =
            getSharedReducedSize(SReducedInsts, /*IsVGPR*/ false, MRI, SIRI);
        if (((newRematSCnt + SharedReducedSize) + (int)NearTargetRegLimit) >=
            rematSCnt) {
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
            int gain = rematGain(&MI, Reg, MRI, SIRI,
                                 /*IsVGPR*/ false);
            if (gain > 0) {
              // Skip case when DefMI has implicit define which used by UseMI.
              if (isImplicitDefUse(&MI, &UseMI)) {
                continue;
              }
              RematNode Node = {Reg, &MI, (unsigned)gain >> 5};
              Node.InsertPointMI = &UseMI;
              Node.Kind = RematNode::RematKind::OneDefOneUse;
              SRematMap[Reg] = Node;
              SharedReducedSize += Node.Size;
            }
          }
        }
        newRematSCnt = rematSCnt - newRematSCnt - SharedReducedSize;
      }
    }
    // If works, continue.

    // Collect live range from hot inst.
    // find common live range in hot insts.
    // Remat these common live range.
    // Apply the remat.

    int NewRematVCnt = 0;
    if (rematVCnt > 0) {
      // TODO: V remat.
    }

    bool NeedSRemat = rematSCnt > 0;
    bool NeedVRemat = rematVCnt > 0;
    // If sgpr spill, always do remat.
    bool IsSRematOK =
        (newRematSCnt <= 0 && !SRematMap.empty()) || IsForceRematSgpr;
    bool IsVRematOK =
        (status.NotBalance || NewRematVCnt <= 0) && !VRematMap.empty();
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
    if (newRematSCnt > 0) {
      if ((unsigned)newRematSCnt <= NearTargetRegLimit) {
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
    ApplyRemat(Remat, SRematMap, hotBlocks, DT, SlotIndexes, MRI, SIRI, SIII,
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

static bool IsImplicitUseOfReg(const MachineOperand &MO, unsigned Reg) {
  if (!MO.isImplicit() || !MO.isUse() || !MO.isReg()) {
    return false;
  }

  return MO.getReg() == Reg;
}

static bool IsImplicitDefOfReg(const MachineOperand &MO, unsigned Reg) {
  if (!MO.isImplicit() || !MO.isDef() || !MO.isReg()) {
    return false;
  }

  return MO.getReg() == Reg;
}

static bool IsSafeRematCandidateUser(const MachineInstr *UseMI,
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
  for (unsigned i = 0; i < OpNum; i++) {
    MachineOperand &Op = DefMI->getOperand(i);
    if (!Op.isReg())
      continue;
    Register OpReg = Op.getReg();
    if (IsImplicitUseOfReg(Op, AMDGPU::EXEC) ||
        IsImplicitUseOfReg(Op, AMDGPU::EXEC_LO))
      continue;
    if (IsImplicitUseOfReg(Op, AMDGPU::MODE))
      continue;
    if (IsImplicitUseOfReg(Op, AMDGPU::M0) && isPhyRegUniqueDef(OpReg, MRI))
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
      if (!IsSafeRematCandidateUser(&UseMI, SIII))
        return false;
    }
  }

  return true;
}

std::vector<SubExp> buildSubExpFromCandidates(
    Remat *Remat, GCNRPTracker::LiveRegSet &Candidates, MachineBasicBlock *MBB,
    const SIRegisterInfo *SIRI, const SIInstrInfo *SIII,
    const MachineRegisterInfo &MRI, SlotIndexes *slotIndexes,
    GCNRPTracker::LiveRegSet &unUsedPassThrus, bool AllowPartialUseInSubExp) {
  InstSet CandidateDefs;
  DenseSet<unsigned> RemovedCandidates;
  std::vector<unsigned> CandidateRegs;
  CandidateRegs.reserve(Candidates.size());
  for (auto it : Candidates) {
    unsigned Reg = it.first;
    CandidateRegs.emplace_back(Reg);
  }
  // Sort candidate by defMI order to make sure defMI has dependent check after
  // all its dependent node.
  std::sort(CandidateRegs.begin(), CandidateRegs.end(),
            [&MRI, &slotIndexes](const unsigned a, unsigned b) {
              MachineInstr *MIa = MRI.getUniqueVRegDef(a);

              MachineInstr *MIb = MRI.getUniqueVRegDef(b);
              // Later instr first.
              return !SlotIndex::isEarlierInstr(
                  slotIndexes->getInstructionIndex(*MIa),
                  slotIndexes->getInstructionIndex(*MIb));
            });

  // If Candidate def has user in MBB, add it when allow partial candidates.
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
          unsigned UserDefReg = UseMI.getOperand(0).getReg();
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
    unUsedPassThrus[Reg] = Candidates[Reg];
    Candidates.erase(Reg);
  }

  // iterate MBB backward.
  // add inst which only used for candidate defines.
  for (auto it = MBB->rbegin(); it != MBB->rend(); it++) {
    MachineInstr &MI = *it;
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
  std::vector<MachineInstr *> defs;
  defs.reserve(CandidateDefs.size());
  for (MachineInstr &MI : *MBB) {
    if (CandidateDefs.count(&MI) == 0)
      continue;
    defs.emplace_back(&MI);
  }

  LLVM_DEBUG(dbgs() << "\nFinished Candidate Defs:\n"; for (MachineInstr *MI
                                                            : defs) {
    MI->dump();
  } dbgs() << "\nFinished Candidate Defs End\n";);

  // Build SubExp with CandidateDefs as Nodes, CandidateInput as input
  // Candidates as output.
  ExpDag dag(MRI, SIRI, SIII, /*IsJoinInput*/ true);
  dag.build(CandidateInput, Candidates, defs);
  if (AllowPartialUseInSubExp) {
    for (auto &subExp : dag.SubExps) {
      for (auto *MI : subExp.SUnits) {
        if (PartialCandidates.count(MI)) {
          subExp.IsCloneOnly = true;
          break;
        }
      }
    }
  }
  return dag.SubExps;
}

std::vector<SubExp> buildSubExpFromCandidatesTopBottom(
    Remat *Remat, GCNRPTracker::LiveRegSet &Candidates, MachineBasicBlock *MBB,
    const SIRegisterInfo *SIRI, const SIInstrInfo *SIII,
    const MachineRegisterInfo &MRI, SlotIndexes *slotIndexes) {
  InstSet CandidateDefs;

  LLVM_DEBUG(dbgs() << "\nCandidate Defs:\n";);
  for (auto it : Candidates) {
    unsigned Reg = it.first;
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
  for (auto it = MBB->begin(); it != MBB->end(); it++) {
    MachineInstr &MI = *it;
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
  std::vector<MachineInstr *> defs;
  defs.reserve(CandidateDefs.size());
  for (MachineInstr &MI : *MBB) {
    if (CandidateDefs.count(&MI) == 0)
      continue;
    defs.emplace_back(&MI);
  }

  LLVM_DEBUG(dbgs() << "\nFinished Candidate Defs:\n"; for (MachineInstr *MI
                                                            : defs) {
    MI->dump();
  } dbgs() << "\nFinished Candidate Defs End\n";);

  LLVM_DEBUG(dbgs() << "\nLocalCandidates:\n"; for (auto it
                                                    : LocalCandidates) {
    pressure::print_reg(it.first, MRI, SIRI, llvm::dbgs());
  } dbgs() << "\nLocalCandidates End\n";);
  // Make sure all input reg are uniqueDef.
  // Input is Candidates, output is?
  // Build SubExp with CandidateDefs as Nodes, CandidateInput as input
  // Candidates as output.
  ExpDag dag(MRI, SIRI, SIII, /*IsJoinInput*/ true);
  dag.build(Candidates, LocalCandidates, defs);
  return dag.SubExps;
}

void print_vreg(Register Reg, const MachineRegisterInfo &MRI) {
  if (Reg.isVirtual()) {
    StringRef Name = MRI.getVRegName(Reg);
    if (Name != "") {
      dbgs() << '%' << Name;
    } else {
      dbgs() << '%' << Register::virtReg2Index(Reg);
    }
  }
}

MachineBasicBlock *FindTargetBlock(unsigned Reg, MachineBasicBlock *FromBB,
                                   const MachineRegisterInfo &MRI,
                                   MachineDominatorTree *DT) {
  BlockSet userBlocks;
  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
    MachineBasicBlock *UserBB = UseMI.getParent();
    // Skip current BB.
    if (UserBB != FromBB)
      userBlocks.insert(UserBB);
    else
      // When has user in FromBB, userBlock will be FromBB.
      return nullptr;
  }
  if (userBlocks.empty())
    return nullptr;
  MachineBasicBlock *userBlock = NearestCommonDominator(DT, userBlocks);
  if (!DT->dominates(FromBB, userBlock)) {
    return nullptr;
  }
  if (userBlock == FromBB)
    return nullptr;
  return userBlock;
}

void ApplySubExpMoveNearUser(SubExp &Exp, const MachineRegisterInfo &MRI,
                             MachineDominatorTree *DT,
                             SlotIndexes *slotIndexes, const SIInstrInfo *SIII,
                             const SIRegisterInfo *SIRI) {
  // Move from bottom.
  MachineBasicBlock *FromBB = Exp.FromBB;
  for (auto it = Exp.SUnits.rbegin(); it != Exp.SUnits.rend(); it++) {
    MachineInstr *DefMI = *it;
    if (DefMI->getNumExplicitDefs() != 1)
      continue;

    unsigned Reg = DefMI->getOperand(0).getReg();
    MachineBasicBlock *ToBB = FindTargetBlock(Reg, FromBB, MRI, DT);
    if (!ToBB)
      continue;

    // Do not overwrite a live scc.
    MachineBasicBlock::iterator InsertPoint =
        ToBB->SkipPHIsAndLabels(ToBB->begin());
    if (WillSmashSccAtLocation(DefMI, ToBB, InsertPoint))
      continue;

    DefMI->removeFromParent();
    assert(!llvm::isExecUpdateForControlFlow(*InsertPoint) &&
           "invalid insert point");
    ToBB->insert(InsertPoint, DefMI);
    // Debug insts don't need slot index.
    if (DefMI->isDebugInstr())
      continue;
    // Update slot index.
    slotIndexes->removeSingleMachineInstrFromMaps(*DefMI);
    slotIndexes->insertMachineInstrInMaps(*DefMI);
  }
}

void ApplySubExpMoveNearDefine(SubExp &Exp, MachineRegisterInfo &MRI,
                               MachineDominatorTree *DT,
                               SlotIndexes *slotIndexes,
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

  Terminator = AdjustInsertPointForSubExpToAvoidSccSmash(Exp, ToBB, Terminator,
                                                         MRI, SIRI, SIII);

  for (auto it = Exp.SUnits.begin(); it != Exp.SUnits.end(); it++) {
    MachineInstr *DefMI = *it;
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
    slotIndexes->removeSingleMachineInstrFromMaps(*DefMI);
    slotIndexes->insertMachineInstrInMaps(*DefMI);
  }
}

DenseSet<MachineInstr *> buildCloneSet(ExpDag &dag,
                                       DenseSet<SUnit *> &dagBottoms,
                                       GCNRPTracker::LiveRegSet &usedOutput) {
  DenseSet<MachineInstr *> copySet;
  for (auto it = dag.SUnits.rbegin(); it != dag.SUnits.rend(); it++) {
    SUnit &SU = *it;
    // Skip non-inst node.
    if (!SU.isInstr())
      continue;
    MachineInstr *MI = SU.getInstr();
    if (dagBottoms.find(&SU) != dagBottoms.end()) {
      bool IsUsed = false;
      // For bottom SU, if in usedOutput, add to copySet;
      for (MachineOperand &DefMO : MI->defs()) {
        if (!DefMO.isReg())
          continue;
        unsigned Reg = DefMO.getReg();
        if (usedOutput.count(Reg) > 0) {
          IsUsed = true;
          break;
        }
      }
      if (IsUsed) {
        copySet.insert(MI);
        continue;
      }
      // bottom SU may still have succNode when it used both inExp and outExp.
      // So continue check succNode.
    }

    // If any SuccNode is in copySet, add to copySet.
    bool IsSuccCopied = false;
    for (SDep &SucDep : SU.Succs) {
      SUnit *SucSU = SucDep.getSUnit();
      MachineInstr *SuccMI = SucSU->getInstr();
      if (copySet.count(SuccMI) > 0) {
        IsSuccCopied = true;
        break;
      }
    }
    if (IsSuccCopied)
      copySet.insert(MI);
  }
  return copySet;
}

void updateUsers(SmallVector<MachineInstr *, 2> &userMIs,
                 DenseMap<unsigned, unsigned> &RegMap) {

  for (MachineInstr *UserMI : userMIs) {
    for (MachineOperand &MO : UserMI->uses()) {
      if (!MO.isReg())
        continue;
      unsigned Reg = MO.getReg();
      auto it = RegMap.find(Reg);
      if (it == RegMap.end())
        continue;
      unsigned NewReg = it->second;
      MO.setReg(NewReg);
    }
  }
}

struct HotBlock {
  MachineBasicBlock *MBB = nullptr;
  GCNRPTracker::LiveRegSet inputLive;
  std::pair<unsigned, unsigned> maxPressures;
  // Info about vmemLd.
  int vmemLdInputSize;
  int vmemLdOutputSize;
};

DenseMap<MachineBasicBlock *, BlockSet> reduceClonedMBBs(
    SubExp &Exp,
    MapVector<MachineBasicBlock *, SmallVector<MachineInstr *, 2>> &userBlocks,
    DenseMap<MachineBasicBlock *, GCNRPTracker::LiveRegSet> &userBlocksLiveRegs,
    std::vector<HotBlock> &hotBlocks, MachineDominatorTree *DT) {
  // Collect hot blocks which Exp is live in.
  DenseSet<MachineBasicBlock *> hotBlockSet;
  for (HotBlock &hotBlock : hotBlocks) {
    for (unsigned Reg : Exp.BottomRegs) {
      if (hotBlock.inputLive.count(Reg)) {
        hotBlockSet.insert(hotBlock.MBB);
        break;
      }
    }
  }

  // For userBlocks which dominate all hotBlocks, don't need to clone because
  // the value not cross hotBlocks when later blocks are cloned.
  // For userBlocks which dominated by all hotBlocks, they could share clones
  // because once after hot block, the pressure is OK.
  DenseSet<MachineBasicBlock *> afterHotRangeMBBs;
  for (auto it : userBlocksLiveRegs) {
    MachineBasicBlock *MBB = it.first;
    // Always clone in hot block.
    if (hotBlockSet.count(MBB))
      continue;

    bool IsDomAllHotBlocks = true;
    bool IsDomedByAllHotBlocks = true;
    for (MachineBasicBlock *hotMBB : hotBlockSet) {
      if (!DT->dominates(MBB, hotMBB)) {
        IsDomAllHotBlocks = false;
      }
      if (!DT->dominates(hotMBB, MBB)) {
        IsDomedByAllHotBlocks = false;
      }
      if (!IsDomAllHotBlocks && !IsDomedByAllHotBlocks) {
        break;
      }
    }
    if (IsDomAllHotBlocks) {
      userBlocks.erase(MBB);
    } else if (IsDomedByAllHotBlocks) {
      afterHotRangeMBBs.insert(MBB);
    }
  }

  // Split after hotRange block set by domtree.
  DenseMap<MachineBasicBlock *, BlockSet> DomMap;
  if (!afterHotRangeMBBs.empty()) {
    for (auto it : afterHotRangeMBBs) {
      MachineBasicBlock *MBB = it;
      for (auto it2 : afterHotRangeMBBs) {
        MachineBasicBlock *MBB2 = it2;
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
    for (auto it : afterHotRangeMBBs) {
      MachineBasicBlock *MBB = it;
      auto &usedOutput = userBlocksLiveRegs[MBB];
      auto &Dom = DomMap[MBB];
      for (MachineBasicBlock *domedMBB : Dom) {
        // Merge domed use to MBB use.
        mergeLiveRegSet(usedOutput, userBlocksLiveRegs[domedMBB]);
        // Remove domedMBB.
        DomMap.erase(domedMBB);
        userBlocksLiveRegs.erase(domedMBB);
      }
    }
  }

  return DomMap;
}

void ApplySubExpCloneNearUser(SubExp &Exp, std::vector<HotBlock> &hotBlocks,
                              MachineDominatorTree *DT,
                              MachineRegisterInfo &MRI,
                              SlotIndexes *slotIndexes, const SIInstrInfo *SIII,
                              const SIRegisterInfo *SIRI) {
  MapVector<MachineBasicBlock *, SmallVector<MachineInstr *, 2>> userBlocks;
  DenseMap<MachineBasicBlock *, GCNRPTracker::LiveRegSet> userBlocksLiveRegs;
  for (unsigned Reg : Exp.BottomRegs) {
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
      MachineBasicBlock *UserBB = UseMI.getParent();
      // Skip current BB.
      if (UserBB == Exp.FromBB)
        continue;

      userBlocks[UserBB].emplace_back(&UseMI);
      auto &userLives = userBlocksLiveRegs[UserBB];
      for (MachineOperand &MO : UseMI.uses()) {
        if (!MO.isReg())
          continue;
        unsigned UseReg = MO.getReg();
        if (Reg != UseReg)
          continue;
        userLives[Reg] |= getRegMask(MO, MRI);
      }
    }
  }
  // Build dag for SubExp to help remove unused inst when clone.
  ExpDag dag(MRI, SIRI, SIII, /*IsJoinInput*/ true);
  dag.build(Exp.inputLive, Exp.outputLive, Exp.SUnits);
  DenseSet<SUnit *> dagBottoms;
  for (SUnit &SU : dag.SUnits) {
    if (!SU.isInstr())
      continue;
    if (SU.NumSuccs == 0) {
      dagBottoms.insert(&SU);
    } else {
      MachineInstr *MI = SU.getInstr();
      // Add SU which def value in Exp.outputLive.
      for (MachineOperand &DefMO : MI->defs()) {
        if (!DefMO.isReg())
          continue;
        unsigned Reg = DefMO.getReg();
        if (Exp.BottomRegs.count(Reg) > 0) {
          dagBottoms.insert(&SU);
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
      reduceClonedMBBs(Exp, userBlocks, userBlocksLiveRegs, hotBlocks, DT);

  // Sort to make stable order.
  std::sort(
      userBlocks.begin(), userBlocks.end(),
      [](std::pair<MachineBasicBlock *, SmallVector<MachineInstr *, 2>> &it0,
         std::pair<MachineBasicBlock *, SmallVector<MachineInstr *, 2>> &it1) {
        return it0.first->getNumber() < it1.first->getNumber();
      });

  const bool IsModifiesScc = Exp.modifiesRegister(AMDGPU::SCC, SIRI);

  // Clone for each userBlocks. Not share clone thru dom tree which cannot help
  // reg pressure.
  for (auto it : userBlocks) {
    MachineBasicBlock *MBB = it.first;
    // Skip MBB which share clone from other MBBs.
    if (userBlocksLiveRegs.count(MBB) == 0)
      continue;
    auto &usedOutput = userBlocksLiveRegs[MBB];
    auto copySet = buildCloneSet(dag, dagBottoms, usedOutput);
    // Clone to MBB.
    // Create new regs first.
    DenseMap<unsigned, unsigned> RegMap;
    auto insertPtr = MBB->getFirstNonPHI();
    // If Exp has scc read/write, make sure MBB not have scc in liveins.
    if (IsModifiesScc && llvm::IsSccLiveAt(MBB, insertPtr))
      continue;
    MachineFunction *MF = MBB->getParent();
    for (auto it = Exp.SUnits.begin(); it != Exp.SUnits.end(); it++) {
      MachineInstr *DefMI = *it;
      // Not clone if already in MBB.
      if (DefMI->getParent() == MBB)
        continue;
      // Not clone if not used for MBB.
      if (copySet.count(DefMI) == 0)
        continue;

      auto ClonedMI =
          BuildMI(*MBB, insertPtr, DefMI->getDebugLoc(), DefMI->getDesc());

      for (MachineOperand &Def : DefMI->defs()) {
        Register Reg = Def.getReg();
        if (Reg.isPhysical()) {
          if (Def.isImplicit())
            continue;
          ClonedMI.addDef(Reg, 0, Def.getSubReg());
        } else {
          unsigned NewReg = MRI.createVirtualRegister(MRI.getRegClass(Reg));
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
            auto it = RegMap.find(Reg);
            if (it == RegMap.end()) {
              ClonedMI.addReg(Reg, 0, MO.getSubReg());
            } else {
              ClonedMI.addReg(it->second, 0, MO.getSubReg());
            }
          }
        } else {
          ClonedMI.add(MO);
        }
      }

      MachineInstr *NewDef = ClonedMI.getInstr();
      slotIndexes->insertMachineInstrInMaps(*NewDef);
      // Set mem operand
      for (MachineMemOperand *MO : DefMI->memoperands()) {
        NewDef->addMemOperand(*MF, MO);
      }
    }

    // update users in MBB.
    SmallVector<MachineInstr *, 2> &userMIs = it.second;
    updateUsers(userMIs, RegMap);

    // update users in dom MBBs.
    auto domMapIt = DomMap.find(MBB);
    if (domMapIt != DomMap.end()) {
      for (MachineBasicBlock *UpdateMBB : domMapIt->second) {
        SmallVector<MachineInstr *, 2> &userMIs = userBlocks[UpdateMBB];
        updateUsers(userMIs, RegMap);
      }
    }
  }
}

void ApplySubExpCloneNearUserInBlock(
    SubExp &Exp,
    DenseMap<MachineBasicBlock *, MachineInstr *> &inBlockHotVInstMap,
    DenseMap<MachineBasicBlock *, MachineInstr *> &inBlockHotSInstMap,
    MachineRegisterInfo &MRI, SlotIndexes *slotIndexes, const SIInstrInfo *SIII,
    const SIRegisterInfo *SIRI) {
  MachineBasicBlock *MBB = Exp.FromBB;
  MachineFunction *MF = MBB->getParent();
  MachineInstr *hotVMI = inBlockHotVInstMap[MBB];
  MachineInstr *hotSMI = inBlockHotSInstMap[MBB];
  // Exp is build with hotVMI or hotSMI, cannot mix.
  assert(!(hotVMI && hotSMI) && "cannot mix hot MI");
  MachineInstr *hotMI = hotVMI;
  if (!hotMI) {
    hotMI = hotSMI;
  }

  SlotIndex hotSlot = slotIndexes->getInstructionIndex(*hotMI).getBaseIndex();
  const bool IsModifiesScc = Exp.modifiesRegister(AMDGPU::SCC, SIRI);

  for (unsigned Reg : Exp.BottomRegs) {

    SmallVector<MachineInstr *, 2> useMIs;
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
      MachineBasicBlock *UserBB = UseMI.getParent();
      // Skip current BB.
      if (UserBB != Exp.FromBB)
        continue;
      // Skip inst in Exp.
      if (Exp.BottomRoots.find(&UseMI) != Exp.BottomRoots.end())
        continue;
      SlotIndex useSlot =
          slotIndexes->getInstructionIndex(UseMI).getBaseIndex();
      // Only clone for use after hot slot.
      if (useSlot < hotSlot)
        continue;

      // Do not overwrite a live scc.
      if (IsModifiesScc && llvm::IsSccLiveAt(UserBB, &UseMI))
        continue;

      useMIs.emplace_back(&UseMI);
    }
    if (useMIs.empty())
      continue;
    DenseMap<unsigned, unsigned> RegMap;

    std::sort(useMIs.begin(), useMIs.end(),
              [&slotIndexes](const MachineInstr *MIa, const MachineInstr *MIb) {
                return slotIndexes->getInstructionIndex(*MIa).getBaseIndex() <
                       slotIndexes->getInstructionIndex(*MIb).getBaseIndex();
              });
    auto insertPtr = useMIs.front()->getIterator();

    for (auto it = Exp.SUnits.begin(); it != Exp.SUnits.end(); it++) {
      MachineInstr *DefMI = *it;
      auto ClonedMI =
          BuildMI(*MBB, insertPtr, DefMI->getDebugLoc(), DefMI->getDesc());

      for (MachineOperand &Def : DefMI->defs()) {
        Register Reg = Def.getReg();
        if (Reg.isPhysical()) {
          ClonedMI.addDef(Reg, 0, Def.getSubReg());
        } else {
          unsigned NewReg = MRI.createVirtualRegister(MRI.getRegClass(Reg));
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
            auto it = RegMap.find(Reg);
            if (it == RegMap.end()) {
              ClonedMI.addReg(Reg, 0, MO.getSubReg());
            } else {
              ClonedMI.addReg(it->second, 0, MO.getSubReg());
            }
          }
        } else {
          ClonedMI.add(MO);
        }
      }

      MachineInstr *NewDef = ClonedMI.getInstr();
      slotIndexes->insertMachineInstrInMaps(*NewDef);
      // Set mem operand
      for (MachineMemOperand *MO : DefMI->memoperands()) {
        NewDef->addMemOperand(*MF, MO);
      }
    }
    // TODO: only clone to cross hot range.
    for (MachineInstr *UseMI : useMIs) {
      for (MachineOperand &MO : UseMI->uses()) {
        if (!MO.isReg())
          continue;
        unsigned Reg = MO.getReg();
        auto it = RegMap.find(Reg);
        if (it == RegMap.end())
          continue;
        unsigned NewReg = it->second;
        MO.setReg(NewReg);
      }
    }
  }
}

bool isInLiveSet(unsigned Reg, LaneBitmask mask,
                 const GCNRPTracker::LiveRegSet &live) {
  auto it = live.find(Reg);
  if (it == live.end())
    return false;

  LaneBitmask liveMask = it->second;
  return (liveMask | mask) == liveMask;
}

unsigned getPacifistLevel(unsigned Reg,
                          DenseMap<MachineInstr *, unsigned> &pacifistLevels,
                          const MachineRegisterInfo &MRI) {
  unsigned level = 0;
  for (MachineInstr &MI : MRI.def_instructions(Reg)) {
    auto it = pacifistLevels.find(&MI);
    if (it == pacifistLevels.end())
      continue;
    level = it->second;
  }
  return level;
}

bool hasInBlockDef(unsigned Reg, MachineBasicBlock *MBB,
                   const MachineRegisterInfo &MRI) {
  for (MachineInstr &def : MRI.def_instructions(Reg)) {
    if (def.getParent() != MBB)
      continue;
    return true;
  }
  return false;
}

MachineInstr *getInBlockUniqueDef(unsigned Reg, MachineBasicBlock *MBB,
                                  const GCNRPTracker::LiveRegSet &inputLive,
                                  const GCNRPTracker::LiveRegSet &outputLive,
                                  const MachineRegisterInfo &MRI) {
  MachineInstr *DefMI = nullptr;
  // If live as input for MBB, cannot be unique def.
  if (inputLive.count(Reg))
    return DefMI;
  for (MachineInstr &def : MRI.def_instructions(Reg)) {
    if (def.getParent() != MBB)
      continue;
    if (DefMI) {
      // Not unique.
      DefMI = nullptr;
      break;
    }
    DefMI = &def;
  }
  return DefMI;
}

bool isPassThru(unsigned Reg, const GCNRPTracker::LiveRegSet &inputLive,
                const GCNRPTracker::LiveRegSet &outputLive) {
  return inputLive.count(Reg) && outputLive.count(Reg);
}

// Instructions which only use imm/passThru reg/output only reg will not kill
// any live reg, so name them pacifist here.
bool collectPacifist(MachineInstr &MI,
                     const GCNRPTracker::LiveRegSet &inputLive,
                     const GCNRPTracker::LiveRegSet &outputLive,
                     const MachineRegisterInfo &MRI,
                     const SIRegisterInfo *SIRI) {
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
    // def in block. If not, it is not safe to move.
    if (!(nullptr != getInBlockUniqueDef(Reg, MI.getParent(), inputLive,
                                         outputLive, MRI) ||
          (isPassThru(Reg, inputLive, outputLive) &&
           !hasInBlockDef(Reg, MI.getParent(), MRI))))
      return false;

    LaneBitmask mask = llvm::getRegMask(MO, MRI);

    if (isInLiveSet(Reg, mask, outputLive))
      continue;

    return false;
  }
  bool IsHasDef = false;
  for (MachineOperand &MO : MI.defs()) {
    Register Reg = MO.getReg();

    if (Reg.isPhysical())
      return false;

    if (nullptr ==
        getInBlockUniqueDef(Reg, MI.getParent(), inputLive, outputLive, MRI))
      return false;

    IsHasDef = true;
  }
  // If no def, it will not increase pressure, don't mark it.
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
                                             SlotIndexes *slotIndexes) {

  SmallVector<MachineInstr *, 2> users;

  // We cannot move the pacifist instruction past any memory
  // op with which it aliases. Find the first instruction
  // that aliases the pacifist MI (if any) and add it to the list
  // of users. The sort() below will select the earliest user instruction.
  if (MachineInstr *AliasMI = findFirstAliasingLoadOrStoreInMBB(MI, MBB, AA)) {
    users.push_back(AliasMI);
  }

  for (MachineOperand &MO : MI.defs()) {
    unsigned Reg = MO.getReg();
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
      if (&MBB != UseMI.getParent())
        continue;
      users.emplace_back(&UseMI);
    }
  }
  if (users.empty())
    return nullptr;

  std::sort(users.begin(), users.end(),
            [&slotIndexes](const MachineInstr *MIa, MachineInstr *MIb) {
              // Early instr first.
              return SlotIndex::isEarlierInstr(
                  slotIndexes->getInstructionIndex(*MIa),
                  slotIndexes->getInstructionIndex(*MIb));
            });
  return users.front();
}

// Pacifist inst will only add pressure since they don't kill.
// Try to hold them as late as possible in a MBB to help pressure.
bool tryHoldPacifist(MachineBasicBlock &MBB, LiveIntervals *LIS,
                     MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                     const SIInstrInfo *SIII, AliasAnalysis *AA,
                     RematStatus &status) {
  const GCNRPTracker::LiveRegSet inputLive = status.MBBInputLiveMap[&MBB];
  const GCNRPTracker::LiveRegSet outputLive = status.MBBOutputLiveMap[&MBB];

  SmallVector<MachineInstr *, 32> pacifistList;
  LLVM_DEBUG(dbgs() << "pacifist begin\n");
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    if (collectPacifist(MI, inputLive, outputLive, MRI, SIRI)) {
      pacifistList.emplace_back(&MI);
      LLVM_DEBUG(MI.dump());
    }
  }
  LLVM_DEBUG(dbgs() << "pacifist end\n");

  SlotIndexes *slotIndexes = LIS->getSlotIndexes();
  bool IsUpdated = false;

  // Move pacifist to its first user.
  // for (MachineInstr *MI : pacifistList) {
  for (auto it = pacifistList.rbegin(); it != pacifistList.rend(); it++) {
    MachineInstr *MI = *it;
    MachineInstr *firstUser =
        findPacifistInsertPoint(*MI, MBB, MRI, AA, slotIndexes);
    if (firstUser == MI)
      continue;
    if (firstUser == MI->getNextNode())
      continue;

    auto insertPoint = MBB.getFirstInstrTerminator();
    if (firstUser) {
      insertPoint = firstUser->getIterator();
    } else {
      // When there's no terminator.
      if (insertPoint == MBB.end())
        insertPoint--;
      else
        // BRANCH may have exec update before it.
        insertPoint--;

      insertPoint =
          llvm::skipDebugInstructionsBackward(insertPoint, MBB.instr_begin());

      while ((insertPoint->definesRegister(AMDGPU::EXEC, SIRI) ||
              insertPoint->definesRegister(AMDGPU::EXEC_LO, SIRI)) &&
             insertPoint != MI->getIterator()) {
        insertPoint--;
        insertPoint =
            llvm::skipDebugInstructionsBackward(insertPoint, MBB.instr_begin());
      }
      if (insertPoint == MI->getIterator())
        continue;
    }
    // Do not overwrite a live scc.
    if (WillSmashSccAtLocation(MI, &MBB, insertPoint))
      continue;
    MI->removeFromParent();
    MBB.insert(insertPoint, MI);

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
      unsigned dstIdx =
          AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::vdst);
      if (dstIdx == (unsigned)-1)
        continue;
      MachineOperand &DstMO = MI.getOperand(dstIdx);
      if (DstMO.getSubReg() != 0)
        continue;
      if (DstMO.isTied())
        continue;
      unsigned Reg = DstMO.getReg();
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

// Try insert readfirstlane on uniform vgpr to turn it in sgpr and save vgpr
// pressure.
bool collectVToSCrossHotSpot(
    MachineBasicBlock &MBB, RematStatus &status,
    DenseMap<unsigned, MachineInstr *> &UniformMap,
    SmallMapVector<unsigned, MachineInstr *, 4> &VToSMap, LiveIntervals *LIS)
{
  unsigned VLimit = status.TargetVLimit;
  unsigned SLimit = status.TargetSLimit;
  auto &ST = MBB.getParent()->getSubtarget<GCNSubtarget>();

  GCNDownwardRPTracker Tracker(*LIS);

  bool IsUpdated = false;
  const auto inputLive = status.MBBInputLiveMap[&MBB];
  Tracker.reset(*MBB.begin(), &inputLive);
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
    for (auto it : CurLives) {
      unsigned Reg = it.first;
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
static bool IsCrossLoopUse(MachineInstr *Def, MachineInstr *User,
                           MachineLoopInfo *MLI) {
  MachineLoop *L = MLI->getLoopFor(Def->getParent());
  return L && !L->contains(User->getParent());
}

bool rematUniformVgprToSgpr(
    Remat *Remat, MachineFunction &MF, RematStatus &status,
    DenseMap<MachineBasicBlock *, GCNRegPressure> &MBBPressureMap,
    std::vector<HotBlock> &hotBlocks, LiveIntervals *LIS,
    MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
    const SIInstrInfo *SIII, MachineLoopInfo *MLI) {
  DenseMap<unsigned, MachineInstr *> UniformVgprMap =
      collectUniformVgprs(Remat, MF, MRI, SIRI);

  SmallMapVector<unsigned, MachineInstr *, 4> VToSMap;

  for (auto &hotBlock : hotBlocks) {
    MachineBasicBlock &MBB = *hotBlock.MBB;
    collectVToSCrossHotSpot(MBB, status, UniformVgprMap, VToSMap, LIS);
  }

  if (VToSMap.empty())
    return false;
  SlotIndexes *slotIndexes = LIS->getSlotIndexes();
  const MCInstrDesc &ReadFirstLaneDesc = SIII->get(AMDGPU::V_READFIRSTLANE_B32);
  for (auto it : VToSMap) {
    unsigned Reg = it.first;
    MachineInstr *MI = it.second;

    auto *VRC = SIRI->getRegClassForReg(MRI, Reg);
    // TODO: support bigger vgpr to sgpr.
    if (VRC != &AMDGPU::VGPR_32RegClass)
      continue;
    auto *NewRC = SIRI->getEquivalentSGPRClass(VRC);
    unsigned newDst = MRI.createVirtualRegister(NewRC);

    auto ReadFirstLane =
        BuildMI(MF, MI->getDebugLoc(), ReadFirstLaneDesc, newDst);
    SmallVector<MachineInstr *, 2> userMIs;
    for (MachineInstr &userMI : MRI.use_nodbg_instructions(Reg)) {
      // Do not replace v->s across loops. Even if the value is uniform
      // branch divergence can cause a uniform value in a loop to be
      // non-uniform when used outside a loop.
      if (IsSafeRematCandidateUser(&userMI, SIII) &&
          !IsCrossLoopUse(MI, &userMI, MLI))
        userMIs.emplace_back(&userMI);
    }

    // Finish readfirstlane
    ReadFirstLane.addReg(Reg);
    MachineInstr *VToSMI = ReadFirstLane.getInstr();
    Remat->TotalUniformInsts.insert(VToSMI);
    Remat->SafeToRemoveInsts.insert(VToSMI);
    MachineBasicBlock *MBB = MI->getParent();
    MBB->insertAfter(MI->getIterator(), VToSMI);
    slotIndexes->insertMachineInstrInMaps(*VToSMI);

    for (MachineInstr *userMI : userMIs) {
      const auto &Desc = userMI->getDesc();
      bool IsIllegal = false;
      for (unsigned i = 0; i < userMI->getNumOperands(); i++) {
        MachineOperand &MO = userMI->getOperand(i);
        if (!MO.isReg())
          continue;
        if (MO.isDef())
          continue;
        if (MO.getReg() != Reg)
          continue;
        if (i >= Desc.getNumOperands()) {
          IsIllegal = true;
          break;
        }

        MO.setReg(newDst);
        if (userMI->getDesc().operands()[i].RegClass != -1) {
          if (!SIII->isOperandLegal(*userMI, i, &MO)) {
            SIII->legalizeOperands(*userMI);
            // In case legalizeOperands not help, just legalize with mov.
            if (userMI->getDesc().operands()[i].RegClass != -1 &&
                !SIII->isOperandLegal(*userMI, i)) {
              SIII->legalizeOpWithMove(*userMI, i);
            }
          }
        } else {
          // consider not have limit on reg class.
        }
      }
      if (IsIllegal)
        continue;

      auto rit = userMI->getReverseIterator();
      rit++;
      auto endIt = userMI->getParent()->rend();
      while (rit != endIt && !rit->isDebugInstr() &&
             !slotIndexes->hasIndex(*rit))
        slotIndexes->insertMachineInstrInMaps(*(rit++));
    }
  }

  return true;
}

bool collectRematableHotReg(
    MachineInstr &MI, const GCNRPTracker::LiveRegSet &hotLive,
    GCNRPTracker::LiveRegSet &pureHotRematSet,
    DenseMap<MachineInstr *, unsigned> &pureHotRematLevels, unsigned &DefReg,
    const GCNRPTracker::LiveRegSet &inputLive,
    const GCNRPTracker::LiveRegSet &outputLive, const MachineRegisterInfo &MRI,
    const SIRegisterInfo *SIRI) {
  // Ignore inst not have def or more than 1 def.
  if (MI.getDesc().getNumDefs() != 1)
    return false;

  DefReg = MI.defs().begin()->getReg();

  unsigned level = 0;
  for (MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    if (MO.isDef())
      continue;

    Register Reg = MO.getReg();

    // If user is in same MI like
    //  %4:vgpr_32 = V_MAD_LEGACY_F32 %2:vgpr_32, %3:vgpr_32, %4:vgpr_32
    // remat it will not help.
    if (Reg == DefReg) {
      return false;
    }

    if (MO.isImplicit() && (Reg == AMDGPU::EXEC || Reg == AMDGPU::EXEC_LO))
      continue;
    if (Reg.isPhysical())
      return false;

    if (nullptr ==
        getInBlockUniqueDef(Reg, MI.getParent(), inputLive, outputLive, MRI))
      return false;

    LaneBitmask mask = llvm::getRegMask(MO, MRI);

    if (isInLiveSet(Reg, mask, hotLive))
      continue;

    if (isInLiveSet(Reg, mask, pureHotRematSet)) {
      unsigned regLevel = getPacifistLevel(Reg, pureHotRematLevels, MRI);
      level = std::max(level, regLevel);
      continue;
    }

    return false;
  }

  for (MachineOperand &MO : MI.defs()) {
    Register Reg = MO.getReg();

    if (Reg.isPhysical())
      return false;

    if (nullptr ==
        getInBlockUniqueDef(Reg, MI.getParent(), inputLive, outputLive, MRI))
      return false;

    LaneBitmask mask = llvm::getRegMask(MO, MRI);
    pureHotRematSet[Reg] |= mask;
  }

  pureHotRematLevels[&MI] = level + 1;
  // If no def, it will not increase pressure, don't mark it.
  return true;
}

bool tryRemat(MachineBasicBlock &MBB, MachineInstr *hotMI,
              std::vector<SubExp> &inBlockCloneSubExps, bool IsVGPR,
              const GCNRPTracker::LiveRegSet &inputLive,
              const GCNRPTracker::LiveRegSet &outputLive,
              DenseSet<MachineInstr *> &hotSet, int vDistance, int sDistance,
              unsigned VLimit, unsigned SLimit,
              const DenseSet<MachineBasicBlock *> &MemWriteMBBSet,
              LiveIntervals *LIS, const MachineRegisterInfo &MRI,
              const SIRegisterInfo *SIRI, const SIInstrInfo *SIII) {
  auto &ST = MBB.getParent()->getSubtarget<GCNSubtarget>();
  const auto &SI = LIS->getInstructionIndex(*hotMI).getBaseIndex();
  const auto LISLR = llvm::getLiveRegs(SI, *LIS, MRI);

  GCNRPTracker::LiveRegSet hotLive = LISLR;

  GCNRPTracker::LiveRegSet pureHotRematSet;
  std::vector<MachineInstr *> pureHotRematList;
  DenseMap<MachineInstr *, unsigned> pureHotRematLevels;

  GCNRPTracker::LiveRegSet outputSet;
  LLVM_DEBUG(dbgs() << "pure hot remat begin\n");
  // Find reg which could remat from other reg in liveSet.
  const unsigned kMaxRematLevel = 6;
  GCNDownwardRPTracker Tracker(*LIS);
  Tracker.reset(*MBB.begin(), &inputLive);
  for (auto it = MBB.begin(); it != MBB.end(); it++) {
    MachineInstr &MI = *it;
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
    if (&MI == hotMI)
      break;

    Tracker.advance();

    unsigned DefReg = 0;
    if (collectRematableHotReg(MI, hotLive, pureHotRematSet, pureHotRematLevels,
                               DefReg, inputLive, outputLive, MRI, SIRI)) {
      unsigned level = pureHotRematLevels[&MI];
      if (level >= kMaxRematLevel)
        continue;

      // If the def reg is in hot reg.
      // Add to output.
      if (hotLive.find(DefReg) != hotLive.end()) {
        bool IsUserIsHot = false;
        for (MachineInstr &UseMI : MRI.use_nodbg_instructions(DefReg)) {
          if (UseMI.getParent() != &MBB)
            continue;
          if (0 == hotSet.count(&UseMI))
            continue;

          const auto &useSI = LIS->getInstructionIndex(UseMI).getBaseIndex();
          // When has a hot user after hotMI, remat it may not help.
          if (useSI > SI) {
            IsUserIsHot = true;
            break;
          }
        }

        if (IsUserIsHot)
          continue;
        outputSet[DefReg];
        LLVM_DEBUG(dbgs() << "hotRemat:");
        LLVM_DEBUG(MI.getOperand(0).dump());
        // remove it from hotLive to avoid it as input when build dag.
        hotLive.erase(DefReg);
      }
      pureHotRematList.emplace_back(&MI);
      LLVM_DEBUG(dbgs() << "level:" << level);
      LLVM_DEBUG(MI.dump());
    }
  }

  LLVM_DEBUG(dbgs() << "pure hot remat end\n");

  // Create input/output for pure hot remat.
  // Input is things hot reg in level 1 and output is things level > 1.
  // Build SubExp with pureHotRematList as Nodes, hotLive as input
  // rematHot as output.
  // Not join input when build ExpDag to get small subExps.
  ExpDag dag(MRI, SIRI, SIII, /*IsJoinInput*/ false);
  dag.build(hotLive, outputSet, pureHotRematList);
  // Find best subExp add to inBlockCloneSubExps.
  // Sort by size of subExp.
  std::sort(dag.SubExps.begin(), dag.SubExps.end(),
            [](const SubExp &A, const SubExp &B) {
              return A.SUnits.size() < B.SUnits.size();
            });
  std::vector<SubExp> cloneSubExps;
  int distance = IsVGPR ? vDistance : sDistance;
  for (SubExp &subExp : dag.SubExps) {
    if (subExp.IsNotSafeToCopy)
      continue;
    if (IsVGPR) {
      if (subExp.vOutputSize == 0)
        continue;
    } else {
      if (subExp.sOutputSize == 0)
        continue;
    }
    if (!subExp.isSafeToMove(MRI, /*IsMoveUp*/ false))
      continue;
    // Not clone .
    if (subExp.SUnits.size() > 10)
      continue;
    // Do not allow remat in the block when the expression has a memory op and
    // the block has a write. We could allow this in some cases with better
    // analysis.
    if (subExp.IsHasMemInst && MemWriteMBBSet.count(&MBB))
      continue;
    if (IsVGPR) {
      distance -= subExp.vOutputSize;
    } else {
      distance -= subExp.sOutputSize;
    }
    cloneSubExps.emplace_back(subExp);
    if (distance <= 0)
      break;
  }
  if (distance <= 0) {
    inBlockCloneSubExps.insert(inBlockCloneSubExps.end(), cloneSubExps.begin(),
                               cloneSubExps.end());
  }
  return distance <= 0;
}

// Try to remat live reg in hot spot from other live reg in hot spot.
//
bool tryRematInHotSpot(
    MachineBasicBlock &MBB, RematStatus &status, int vDistance, int sDistance,
    int vSaved, int sSaved, std::vector<SubExp> &inBlockCloneSubExps,
    DenseMap<MachineBasicBlock *, MachineInstr *> &inBlockHotVInstMap,
    DenseMap<MachineBasicBlock *, MachineInstr *> &inBlockHotSInstMap,
    LiveIntervals *LIS, const MachineRegisterInfo &MRI,
    const SIRegisterInfo *SIRI, const SIInstrInfo *SIII) {
  unsigned VLimit = status.TargetVLimit;
  unsigned SLimit = status.TargetSLimit;

  auto &ST = MBB.getParent()->getSubtarget<GCNSubtarget>();
  const GCNRPTracker::LiveRegSet inputLive = status.MBBInputLiveMap[&MBB];

  const GCNRPTracker::LiveRegSet outputLive = status.MBBOutputLiveMap[&MBB];

  // Collect reg pressure.
  unsigned maxLocalVPressure = 0;
  unsigned maxLocalSPressure = 0;
  // Build a DAG or only on demand?
  MachineInstr *hotVMI = nullptr;
  MachineInstr *hotSMI = nullptr;
  DenseSet<MachineInstr *> hotSet;

  GCNDownwardRPTracker Tracker(*LIS);

  Tracker.reset(*MBB.begin(), &inputLive);
  for (auto it = MBB.begin(); it != MBB.end(); it++) {
    MachineInstr &MI = *it;
    if (MI.isDebugInstr()) {
      continue;
    }

    unsigned VPressure = Tracker.getPressure().getVGPRNum(ST.hasGFX90AInsts());
    unsigned SPressure = Tracker.getPressure().getMaxSGPR();

    SPressure += RegForVCC;

    VPressure -= vSaved;
    SPressure -= sSaved;
    Tracker.advance();

    if (VPressure <= VLimit && SPressure <= SLimit) {
      continue;
    }
    hotSet.insert(&MI);
    if (maxLocalVPressure < VPressure) {
      maxLocalVPressure = VPressure;
      hotVMI = &MI;
    }
    if (maxLocalSPressure < SPressure) {
      maxLocalSPressure = SPressure;
      hotSMI = &MI;
    }
  }

  inBlockHotVInstMap[&MBB] = hotVMI;
  inBlockHotSInstMap[&MBB] = hotSMI;
  if (vDistance > 0 && hotVMI) {
    // Use hotVMI when apply.
    inBlockHotSInstMap[&MBB] = nullptr;
    if (tryRemat(MBB, hotVMI, inBlockCloneSubExps, /*IsVGPR*/ true, inputLive,
                 outputLive, hotSet, vDistance, sDistance, VLimit, SLimit,
                 status.MemWriteMBBSet, LIS, MRI, SIRI, SIII))
      return true;
  }

  if (sDistance > 0 && hotSMI) {
    // Use hotSMI when apply.
    inBlockHotSInstMap[&MBB] = hotSMI;
    inBlockHotVInstMap[&MBB] = nullptr;
    return tryRemat(MBB, hotSMI, inBlockCloneSubExps, /*IsVGPR*/ false,
                    inputLive, outputLive, hotSet, vDistance, sDistance, VLimit,
                    SLimit, status.MemWriteMBBSet, LIS, MRI, SIRI, SIII);
  }
  return false;
}
// Sort subExpCandidates to make sure deeper subExp apply first.
// If subExp0 use result of subExp1, subExp0 is deeper than subExp1.
// When apply subExp1 before subExp0, new clone of subExp0 which use result of
// subExp1 will have old reg of subExp1. And reg pressure will not be reduced.
void sortSubExpCandidates(std::vector<SubExp> &subExpCandidates) {
  MapVector<unsigned, SetVector<SubExp *>> inputMap;
  MapVector<unsigned, SetVector<SubExp *>> outputMap;
  struct SortNode {
    SubExp Exp;
    unsigned Depth;
    bool IsDepthDirty;
    SmallDenseSet<SubExp *, 2> Preds;
    SmallDenseSet<SubExp *, 2> Succs;
  };

  {
    SmallVector<unsigned, 10> RegSortStorage;
    for (SubExp &Exp : subExpCandidates) {
      RegSortStorage.assign(Exp.TopRegs.begin(), Exp.TopRegs.end());
      std::sort(RegSortStorage.begin(), RegSortStorage.end());
      for (auto it : RegSortStorage) {
        unsigned Reg = it;
        inputMap[Reg].insert(&Exp);
      }

      RegSortStorage.assign(Exp.BottomRegs.begin(), Exp.BottomRegs.end());
      std::sort(RegSortStorage.begin(), RegSortStorage.end());
      for (auto it : RegSortStorage) {
        unsigned Reg = it;
        outputMap[Reg].insert(&Exp);
      }
    }
  }

  MapVector<SubExp *, SortNode> sortMap;
  for (auto it : inputMap) {
    unsigned Reg = it.first;
    auto outIt = outputMap.find(Reg);
    if (outIt == outputMap.end())
      continue;
    auto &inExps = it.second;
    auto &outExps = outIt->second;
    for (SubExp *inExp : inExps) {
      for (SubExp *outExp : outExps) {
        if (inExp->IsHoist != outExp->IsHoist) {
          // Different direction.
          // If output (def) move up, input(use) move down, nothing happens.
          if (outExp->IsHoist)
            continue;
          // Canot input(use) move up, output(def) move down.
          // Choose the exp which save more.
          int inExpGain = inExp->vOutputSize - inExp->vInputSize;
          int outExpGain = outExp->vInputSize - inExp->vOutputSize;
          if (inExpGain >= outExpGain) {
            outExp->SUnits.clear();
          } else {
            inExp->SUnits.clear();
          }
          continue;
        }
        // Link outExp to inExp.
        if (inExp->IsHoist) {
          sortMap[outExp].Preds.insert(inExp);
          sortMap[inExp].Succs.insert(outExp);
        } else {
          sortMap[inExp].Preds.insert(outExp);
          sortMap[outExp].Succs.insert(inExp);
        }
      }
    }
  }

  if (sortMap.empty())
    return;

  SmallVector<SubExp *, 8> WorkList;
  for (SubExp &Exp : subExpCandidates) {
    SortNode &Node = sortMap[&Exp];
    Node.Depth = 0;
    Node.Exp = Exp;
    Node.IsDepthDirty = !Node.Preds.empty();
    if (!Node.IsDepthDirty)
      WorkList.emplace_back(&Exp);
  }
  // Calc depth.
  while (!WorkList.empty()) {
    SubExp *Exp = WorkList.pop_back_val();
    SortNode &Node = sortMap[Exp];
    for (SubExp *Succ : Node.Succs) {
      SortNode &SuccNode = sortMap[Succ];
      SuccNode.Depth = std::max(SuccNode.Depth, Node.Depth + 1);
      bool IsAllPrevClean = true;
      for (SubExp *Prev : SuccNode.Preds) {
        SortNode &PrevNode = sortMap[Prev];
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

  std::vector<SortNode *> nodes;
  for (auto &it : sortMap) {
    SortNode &node = it.second;
    nodes.emplace_back(&node);
  }

  struct sorter {
    bool operator()(const SortNode *a, const SortNode *b) {
      return a->Depth > b->Depth;
    }
  };

  // subExp deeper should be apply first.
  std::sort(nodes.begin(), nodes.end(), sorter());

  subExpCandidates.clear();
  for (auto &node : nodes) {
    subExpCandidates.emplace_back(node->Exp);
  }
}

// Compare pressure, return ture if maxV0/maxS0 pressure is higher than
// maxV1/maxS1.
bool pressureHigher(unsigned maxV0, unsigned maxS0, unsigned maxV1,
                    unsigned maxS1, const GCNSubtarget *ST) {
  unsigned VTgtOcc0 = ST->getOccupancyWithNumVGPRs(maxV0);
  unsigned VTgtOcc1 = ST->getOccupancyWithNumVGPRs(maxV1);
  unsigned STgtOcc0 = ST->getOccupancyWithNumSGPRs(maxS0);
  unsigned STgtOcc1 = ST->getOccupancyWithNumSGPRs(maxS1);
  unsigned Occ0 = std::min(VTgtOcc0, STgtOcc0);
  unsigned Occ1 = std::min(VTgtOcc1, STgtOcc1);
  //  is low pressure.
  if (Occ0 > Occ1)
    return false;
  if (Occ0 < Occ1)
    return true;
  // When sgpr bound,  is high pressure.
  if (VTgtOcc0 > STgtOcc0 && VTgtOcc1 > STgtOcc1) {
    return maxS0 > maxS1;
  }
  // When vgpr bound or mix, vgpr higher is higher pressure.
  return maxV0 > maxV1;
}

// Return true if the subExp can help pressure for passThrus.
bool canHelpPressureWhenSink(
    SubExp &subExp, const GCNRPTracker::LiveRegSet &passThrus,
    const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
    const SIInstrInfo *SIII, const MachineLoopInfo *MLI,
    MachineDominatorTree *DT, bool IsCanClone, bool IsSgprBound) {
  LLVM_DEBUG(subExp.dump(MRI, SIRI));
  if (!subExp.isSafeToMove(MRI, /*IsMoveUp*/ false))
    return false;

  // Update input size to ignore lives in which already in
  // passThrus.
  for (auto it : subExp.inputLive) {
    unsigned Reg = it.first;
    if (passThrus.count(Reg) == 0)
      continue;
    unsigned Size = getRegSize(Reg, it.second, MRI, SIRI);
    if (SIRI->isVGPR(MRI, Reg)) {
      subExp.vInputSize -= Size;
    } else {
      subExp.sInputSize -= Size;
    }
  }

  if (subExp.vInputSize > subExp.vOutputSize)
    return false;

  if (subExp.sInputSize > subExp.sOutputSize && IsSgprBound)
    return false;

  if (subExp.sInputSize >= subExp.sOutputSize &&
      subExp.vInputSize == subExp.vOutputSize)
    return false;

  // Try to find a Insert Block.
  // Skip multi def output sub exp.
  // Collect user blocks, find common dom.
  BlockSet userBlocks;
  for (unsigned Reg : subExp.BottomRegs) {
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
      MachineBasicBlock *UserBB = UseMI.getParent();
      // Skip current BB.
      if (UserBB != subExp.FromBB)
        userBlocks.insert(UserBB);
    }
  }
  if (userBlocks.empty())
    return false;
  MachineBasicBlock *userBlock = NearestCommonDominator(DT, userBlocks);
  if (!DT->dominates(subExp.FromBB, userBlock)) {
    return false;
  }
  if (userBlock == subExp.FromBB &&
      // When allow clone, could go clone path if cannot move subExp.
      !IsCanClone)
    return false;

  subExp.ToBB = userBlock;
  if (auto *toLoop = MLI->getLoopFor(userBlock)) {
    auto *fromLoop = MLI->getLoopFor(subExp.FromBB);
    if (!fromLoop || fromLoop->getLoopDepth() < toLoop->getLoopDepth())
      subExp.IsMoveIntoLoop = true;
  } else if (auto *fromLoop = MLI->getLoopFor(subExp.FromBB)) {
    auto *toLoop = MLI->getLoopFor(userBlock);
    // not safe to move out of loop.
    if (!toLoop || fromLoop->getLoopDepth() > toLoop->getLoopDepth() ||
        toLoop != fromLoop)
      return false;
  }
  return true;
}

bool canHelpPressureWhenHoist(SubExp &subExp, const MachineRegisterInfo &MRI,
                              const SIRegisterInfo *SIRI,
                              const SIInstrInfo *SIII,
                              const MachineLoopInfo *MLI, bool IsSgprBound) {
  if (!subExp.isSafeToMove(MRI, /*IsMoveUp*/ true))
    return false;
  if (subExp.vInputSize < subExp.vOutputSize)
    return false;
  if (subExp.sInputSize < subExp.sOutputSize && IsSgprBound)
    return false;

  if (subExp.sInputSize <= subExp.sOutputSize &&
      subExp.vInputSize == subExp.vOutputSize)
    return false;

  // Try to find a Insert Block.
  // Skip multi def output sub exp.
  // Collect user blocks, find common dom.
  BlockSet defBlocks;
  for (unsigned Reg : subExp.TopRegs) {
    MachineInstr *DefMI = MRI.getUniqueVRegDef(Reg);
    if (!DefMI)
      continue;
    defBlocks.insert(DefMI->getParent());
  }
  if (defBlocks.size() != 1)
    return false;
  MachineBasicBlock *defBlock = *defBlocks.begin();
  subExp.ToBB = defBlock;
  // Not do same block hoist.
  if (subExp.ToBB == subExp.FromBB)
    return false;

  if (auto *toLoop = MLI->getLoopFor(defBlock)) {
    auto *fromLoop = MLI->getLoopFor(subExp.FromBB);
    // TODO: enable move into loop when hoist.
    if (!fromLoop || fromLoop->getLoopDepth() < toLoop->getLoopDepth())
      return false;
  } else if (auto *fromLoop = MLI->getLoopFor(subExp.FromBB)) {
    auto *toLoop = MLI->getLoopFor(defBlock);
    // not safe to move out of loop.
    if (!toLoop || fromLoop->getLoopDepth() > toLoop->getLoopDepth() ||
        toLoop != fromLoop)
      return false;
  }
  return true;
}

SmallVector<std::pair<MachineBasicBlock *, GCNRPTracker::LiveRegSet>>
groupPassThruByDefBlock(Remat *Remat, const GCNRPTracker::LiveRegSet &passThrus,
                        GCNRPTracker::LiveRegSet &usedPassThrus,
                        MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                        const SIInstrInfo *SIII) {
  MapVector<MachineBasicBlock *, GCNRPTracker::LiveRegSet> Candidates;

  // Group safe candidates by define block.
  for (auto it : passThrus) {
    unsigned Reg = it.first;
    // Skip used pass thru reg to avoid count it twice for different hot block.
    if (usedPassThrus.count(Reg))
      continue;
    LLVM_DEBUG(print_vreg(Reg, MRI));
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
    DefInMBB[Reg] = it.second;
  }

  llvm::SmallVector<std::pair<MachineBasicBlock *, GCNRPTracker::LiveRegSet>>
      result = Candidates.takeVector();

  LLVM_DEBUG(llvm::dbgs() << "Before sort candidates\n"; for (auto it
                                                              : result) {
    MachineBasicBlock *MBB = it.first;
    auto &defInMBB = it.second;
    MBB->dump();
    llvm::dumpLiveSet(defInMBB, SIRI);
  } llvm::dbgs() << "end of candidates\n";);

  std::sort(result.begin(), result.end(),
            [](std::pair<MachineBasicBlock *, GCNRPTracker::LiveRegSet> &it0,
               std::pair<MachineBasicBlock *, GCNRPTracker::LiveRegSet> &it1) {
              return it0.first->getNumber() < it1.first->getNumber();
            });

  LLVM_DEBUG(llvm::dbgs() << "After sort candidates\n"; for (auto it
                                                             : result) {
    MachineBasicBlock *MBB = it.first;
    auto &defInMBB = it.second;
    MBB->dump();
    llvm::dumpLiveSet(defInMBB, SIRI);
  } llvm::dbgs() << "end of candidates\n";);

  return result;
}

// collect pass thru regs of MBB.
GCNRPTracker::LiveRegSet
collectPassThrus(MachineBasicBlock *MBB,
                 const GCNRPTracker::LiveRegSet &inputLive,
                 const GCNRPTracker::LiveRegSet &outputLive,
                 const GCNRPTracker::LiveRegSet &usedPassThrus,
                 const GCNRPTracker::LiveRegSet &liveRegCandidates,
                 MachineRegisterInfo &MRI, bool IsCanClone) {
  GCNRPTracker::LiveRegSet passThrus;
  llvm::mergeLiveRegSet(passThrus, inputLive);
  llvm::andLiveRegSet(passThrus, outputLive);

  // Remove reg which not in liveRegCandidates.
  GCNRPTracker::LiveRegSet tmpPassThrus = passThrus;
  for (auto it : tmpPassThrus) {
    unsigned Reg = it.first;
    if (!liveRegCandidates.count(Reg)) {
      passThrus.erase(Reg);
    }
  }
  tmpPassThrus = passThrus;
  // Remove reg which has read/write in MBB.
  for (auto it : tmpPassThrus) {
    unsigned Reg = it.first;
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
      passThrus.erase(Reg);
  }
  return passThrus;
}
// Try to build a free subExp which all input is passThrus.
SubExp buildFreeSubExp(Remat *Remat, SubExp &subExp,
                       GCNRPTracker::LiveRegSet &passThrus,
                       MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI) {
  SubExp freeExp;
  // Try to split the subExp to find a help case.
  // Scan all inst in subExp, propagate free inst which input is from
  // passThrus.
  SmallDenseSet<unsigned, 4> freeRegs;
  SmallDenseSet<unsigned, 8> freeInstUseRegs;
  SmallVector<MachineInstr *, 4> freeInsts;
  for (MachineInstr *MI : subExp.SUnits) {
    bool IsFree = true;
    // Check all use regs are free.
    for (MachineOperand &MO : MI->uses()) {
      if (!MO.isReg())
        continue;
      unsigned Reg = MO.getReg();
      if (MO.isImplicit() && Reg == AMDGPU::EXEC)
        continue;
      if (MRI.getUniqueVRegDef(Reg) == nullptr) {
        IsFree = false;
        break;
      }
      // Skip local pass thrus unless it is free.
      if (passThrus.count(Reg) && subExp.TopRegs.count(Reg))
        continue;
      if (freeRegs.count(Reg))
        continue;
      IsFree = false;
      break;
    }
    // Check def is unique.
    for (MachineOperand &MO : MI->defs()) {
      unsigned Reg = MO.getReg();
      if (MRI.getUniqueVRegDef(Reg) == nullptr) {
        IsFree = false;
        break;
      }
    }
    if (!IsFree)
      continue;
    // Save inst as free inst.
    freeInsts.emplace_back(MI);
    // Save def as free reg.
    for (MachineOperand &MO : MI->defs()) {
      unsigned Reg = MO.getReg();
      freeRegs.insert(Reg);
    }
    // Save use regs as free use reg.
    for (MachineOperand &MO : MI->uses()) {
      if (!MO.isReg())
        continue;
      unsigned Reg = MO.getReg();

      freeInstUseRegs.insert(Reg);
    }
  }
  // Then remove local inst has no output use.
  for (MachineInstr *MI : freeInsts) {
    bool IsFreeUsed = false;
    for (MachineOperand &MO : MI->defs()) {
      unsigned Reg = MO.getReg();
      // Used as freeInst or output.
      IsFreeUsed |=
          freeInstUseRegs.count(Reg) > 0 || subExp.BottomRegs.count(Reg);
    }
    if (!IsFreeUsed)
      continue;
    freeExp.SUnits.emplace_back(MI);
  }
  if (freeExp.SUnits.empty()) {
    // mark has terminator to make it unsafe.
    freeExp.IsHasTerminatorInst = true;
    return freeExp;
  }
  // Build BottomRegs and TopRegs for freeExp.
  // BottomRegs is freeRegs in subExp.BottomRegs.
  for (unsigned freeReg : freeRegs) {
    if (subExp.BottomRegs.count(freeReg))
      freeExp.BottomRegs.insert(freeReg);
  }
  // TopRegs is freeInstUseRegs in subExp.TopRegs.
  for (unsigned freeInstUseReg : freeInstUseRegs) {
    if (subExp.TopRegs.count(freeInstUseReg))
      freeExp.TopRegs.insert(freeInstUseReg);
  }
  freeExp.FromBB = subExp.FromBB;
  freeExp.ToBB = subExp.ToBB;
  // must be clone since is partial of subExp.
  freeExp.IsCloneOnly = true;

  // Calc reg for freeExp.
  for (unsigned Reg : freeExp.TopRegs) {
    freeExp.inputLive[Reg];
  }

  for (unsigned Reg : freeExp.BottomRegs) {
    freeExp.outputLive[Reg];
  }

  CollectLiveSetPressure(freeExp.inputLive, MRI, SIRI, freeExp.vInputSize,
                         freeExp.sInputSize);
  CollectLiveSetPressure(freeExp.outputLive, MRI, SIRI, freeExp.vOutputSize,
                         freeExp.sOutputSize);
  return freeExp;
}

std::vector<SubExp> buildSubExpCandidates(
    Remat *Remat,
    SmallVector<std::pair<MachineBasicBlock *, GCNRPTracker::LiveRegSet>>
        &Candidates,
    GCNRPTracker::LiveRegSet &passThrus, MachineRegisterInfo &MRI,
    const SIRegisterInfo *SIRI, const SIInstrInfo *SIII,
    const MachineLoopInfo *MLI, SlotIndexes *slotIndexes,
    MachineDominatorTree *DT, bool IsCanClone, bool IsSgprBound,
    GCNRPTracker::LiveRegSet &unUsedPassThrus,
    DenseSet<MachineBasicBlock *> &MemWriteMBBSet,
    bool AllowPartialUseInSubExp) {
  std::vector<SubExp> subExpCandidates;
  // Build exp dag on define blocks.
  // Save profit candidates into list.
  for (auto &it : Candidates) {
    MachineBasicBlock *DefMBB = it.first;
    // Try to remove out reg def sub exp from DefMBB.
    GCNRPTracker::LiveRegSet &DefInMBB = it.second;
    // Go up on the dag until reach share node.
    auto subExps = buildSubExpFromCandidates(
        Remat, DefInMBB, DefMBB, SIRI, SIII, MRI, slotIndexes, unUsedPassThrus,
        AllowPartialUseInSubExp);
    for (SubExp &subExp : subExps) {
      if (subExp.IsHasMemInst) {
        // Skip when memory ld/st inst need to cross MBB which write memory.
        // TODO: check all MBBs in between FromBB and ToBB not write memory.
        // Currently just skip when any memory write exist.
        if (!MemWriteMBBSet.empty()) {
          MachineBasicBlock *FromBB = subExp.FromBB;
          MachineBasicBlock *ToBB = subExp.ToBB;
          if (subExp.IsHoist) {
            FromBB = subExp.ToBB;
            ToBB = subExp.FromBB;
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
      if (!canHelpPressureWhenSink(subExp, passThrus, MRI, SIRI, SIII, MLI, DT,
                                   IsCanClone, IsSgprBound)) {
        if (AllowPartialUseInSubExp &&
            subExp.isSafeToMove(MRI, /*IsMoveUp*/ false)) {
          SubExp freeSubExp =
              buildFreeSubExp(Remat, subExp, passThrus, MRI, SIRI);
          if (canHelpPressureWhenSink(freeSubExp, passThrus, MRI, SIRI, SIII,
                                      MLI, DT, IsCanClone, IsSgprBound)) {
            subExpCandidates.emplace_back(freeSubExp);
          }
        }
        continue;
      }

      subExpCandidates.emplace_back(subExp);
    }
  }
  return subExpCandidates;
}

std::pair<int, int>
calculateSaving(HotBlock &hotBB, std::vector<SubExp> &subExpCandidates,
                GCNRPTracker::LiveRegSet &inputLive,
                GCNRPTracker::LiveRegSet &outputLive, bool IsVOutBound,
                bool IsSOutBound, bool IsCanClone, MachineDominatorTree *DT,
                const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI) {
  int vgpr = 0;
  int sgpr = 0;
  MachineBasicBlock *MBB = hotBB.MBB;
  // Sink saving.
  for (SubExp &Exp : subExpCandidates) {
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
        if (IsVOutBound && Exp.vOutputSize < Exp.vInputSize)
          continue;
        if (IsSOutBound && Exp.sOutputSize < Exp.sInputSize)
          continue;
        vgpr += Exp.vInputSize;
        vgpr -= Exp.vOutputSize;
        sgpr += Exp.sInputSize;
        sgpr -= Exp.sOutputSize;
        continue;
      }
    }
    int vgprDiff = 0;
    int sgprDiff = 0;
    MachineBasicBlock *ToMBB = Exp.ToBB;
    // If subExp is to hotBB, it is crossing output instead of input.
    GCNRPTracker::LiveRegSet &crossLive = MBB == ToMBB ? outputLive : inputLive;

    bool IsClone = false;
    GCNRPTracker::LiveRegSet newInput;
    if (!Exp.IsMoveIntoLoop) {
      if (Exp.IsHoist) {
        // If FromBB dom hot block, it will not change live for MBB.
        if (Exp.FromBB != MBB && DT->dominates(Exp.FromBB, MBB))
          continue;
      } else {
        // If ToBB dom hot block, it will not change live for MBB.
        if (ToMBB != MBB && DT->dominates(ToMBB, MBB)) {
          if (IsCanClone && !Exp.IsNotSafeToCopy) {
            IsClone = true;
          } else {
            continue;
          }
        }
      }

      for (auto outIt : Exp.outputLive) {
        unsigned Reg = outIt.first;
        LaneBitmask outMask = outIt.second;
        LaneBitmask MBBBeginMask;
        if (crossLive.find(Reg) != crossLive.end())
          MBBBeginMask = crossLive[Reg];
        // Check mask which live in both BeginSlot and exp output when sink to
        // kill the output. Check mask which not live in BeginSlot  in
        // exp output when hoist to live the output.
        LaneBitmask profitMask = Exp.IsHoist ? (outMask & (~MBBBeginMask))
                                             : (outMask & MBBBeginMask);
        if (MBBBeginMask.any()) {
          unsigned Size = getRegSize(Reg, profitMask, MRI, SIRI);
          LLVM_DEBUG(std::string movStr =
                         Exp.IsHoist ? "output hoist:" : "output sink:";
                     dbgs()
                     << movStr << Register::virtReg2Index(Reg) << " " << Size);
          // Exp out live at block input.
          // It will descrease live for MBB when sink and increase when hoist.
          if (SIRI->isVGPR(MRI, Reg)) {
            LLVM_DEBUG(dbgs() << "v\n");
            if (Exp.IsHoist)
              vgprDiff += Size;
            else
              vgprDiff -= Size;
          } else {
            LLVM_DEBUG(dbgs() << "s\n");
            if (Exp.IsHoist)
              sgprDiff += Size;
            else
              sgprDiff -= Size;
          }
        }
      }

      for (auto inIt : Exp.inputLive) {
        unsigned Reg = inIt.first;
        LaneBitmask inMask = inIt.second;
        LaneBitmask MBBBeginMask;
        if (crossLive.find(Reg) != crossLive.end())
          MBBBeginMask = crossLive[Reg];
        // Check mask which not live in BeginSlot  in exp input when
        // sink to live the input. Check mask which live in both BeginSlot and
        // exp output when hoist to kill the input.
        LaneBitmask profitMask =
            Exp.IsHoist ? (inMask & MBBBeginMask) : (inMask & (~MBBBeginMask));
        if (profitMask.any()) {
          // Update input live to avoid count same input more than once.
          newInput[Reg] |= inMask;
          // Exp in not live at block input.
          // It will increase live for MBB.
          unsigned Size = getRegSize(Reg, profitMask, MRI, SIRI);

          LLVM_DEBUG(
              std::string movStr = Exp.IsHoist ? "input hoist:" : "input sink:";
              dbgs() << movStr << Register::virtReg2Index(Reg) << " " << Size);
          if (SIRI->isVGPR(MRI, Reg)) {
            LLVM_DEBUG(dbgs() << "v\n");
            if (Exp.IsHoist)
              vgprDiff -= Size;
            else
              vgprDiff += Size;
          } else {
            LLVM_DEBUG(dbgs() << "s\n");
            if (Exp.IsHoist)
              sgprDiff -= Size;
            else
              sgprDiff += Size;
          }
        }
      }
    } else {
      // When sink into loop, the input will live for every block inside loop.
      // The output will only lived between to blocks and the use blocks.
      // If MBB dominate any user of output live reg, it will still live in
      // MBB. So cannot count that output live reg as profit.
      // Hoist into loop is not supported now.
      for (auto outIt : Exp.outputLive) {
        unsigned Reg = outIt.first;
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

        LaneBitmask outMask = outIt.second;
        LaneBitmask MBBBeginMask;
        if (inputLive.find(Reg) != inputLive.end())
          MBBBeginMask = inputLive[Reg];
        LaneBitmask profitMask = outMask & MBBBeginMask;
        if (MBBBeginMask.any()) {
          unsigned Size = getRegSize(Reg, profitMask, MRI, SIRI);
          LLVM_DEBUG(dbgs()
                     << "move:" << Register::virtReg2Index(Reg) << " " << Size);
          // Exp out live at block input.
          // It will descrease live for MBB.
          if (SIRI->isVGPR(MRI, Reg)) {
            LLVM_DEBUG(dbgs() << "v\n");
            vgprDiff -= Size;
          } else {
            LLVM_DEBUG(dbgs() << "s\n");
            sgprDiff -= Size;
          }
        }
      }

      for (auto inIt : Exp.inputLive) {
        unsigned Reg = inIt.first;
        LaneBitmask inMask = inIt.second;
        LaneBitmask MBBBeginMask;
        if (inputLive.find(Reg) != inputLive.end())
          MBBBeginMask = inputLive[Reg];
        // Check mask which not live in BeginSlot  in exp input.
        LaneBitmask profitMask = inMask & (~MBBBeginMask);
        if (profitMask.any()) {
          // Update input live to avoid count same input more than once.
          newInput[Reg] |= inMask;
          // Exp in not live at block input.
          // It will increase live for MBB.
          unsigned Size = getRegSize(Reg, profitMask, MRI, SIRI);

          LLVM_DEBUG(dbgs()
                     << "add:" << Register::virtReg2Index(Reg) << " " << Size);
          if (SIRI->isVGPR(MRI, Reg)) {
            LLVM_DEBUG(dbgs() << "v\n");
            vgprDiff += Size;
          } else {
            LLVM_DEBUG(dbgs() << "s\n");
            sgprDiff += Size;
          }
        }
      }
    }

    if (IsVOutBound && vgprDiff > 0)
      continue;

    if (IsSOutBound && sgprDiff > 0)
      continue;
    llvm::mergeLiveRegSet(crossLive, newInput);
    vgpr += vgprDiff;
    sgpr += sgprDiff;
    if (IsClone)
      Exp.IsCloneOnly = true;
  }

  return std::make_pair(vgpr, sgpr);
}

void addExpCandidates(std::vector<SubExp> &subExpCandidates,
                      std::vector<SubExp> &subExps,
                      GCNRPTracker::LiveRegSet &usedRegs) {
  subExpCandidates.insert(subExpCandidates.end(), subExps.begin(),
                          subExps.end());
  for (auto &Exp : subExps) {
    if (Exp.IsHoist) {
      for (auto &Reg : Exp.TopRegs) {
        usedRegs[Reg];
      }
    } else {
      for (auto &Reg : Exp.BottomRegs) {
        usedRegs[Reg];
      }
    }
  }
}

bool tryToAddSubExps(
    Remat *Remat, HotBlock &hotBB, RematStatus &status,
    std::vector<SubExp> &subExpCandidates,
    std::vector<SubExp> &inBlockCloneSubExps,
    DenseMap<MachineBasicBlock *, MachineInstr *> &inBlockHotVInstMap,
    DenseMap<MachineBasicBlock *, MachineInstr *> &inBlockHotSInstMap,
    SmallVector<std::pair<MachineBasicBlock *, GCNRPTracker::LiveRegSet>>
        Candidates,
    int vgpr, int sgpr, const GCNRPTracker::LiveRegSet &savingInputLive,
    const GCNRPTracker::LiveRegSet &savingOutputLive,
    GCNRPTracker::LiveRegSet &passThrus, GCNRPTracker::LiveRegSet &usedRegs,
    MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
    const SIInstrInfo *SIII, const MachineLoopInfo *MLI,
    SlotIndexes *slotIndexes, LiveIntervals *LIS, MachineDominatorTree *DT,
    bool IsCanClone, bool IsVOutBound, bool IsSOutBound,
    GCNRPTracker::LiveRegSet &unUsedPassThrus, bool AllowPartialUseInSubExp) {
  std::vector<SubExp> partialSubExps = buildSubExpCandidates(
      Remat, Candidates, passThrus, MRI, SIRI, SIII, MLI, slotIndexes, DT,
      IsCanClone, IsSOutBound, unUsedPassThrus, status.MemWriteMBBSet,
      AllowPartialUseInSubExp);

  GCNRPTracker::LiveRegSet tmpSavingInputLive = savingInputLive;
  GCNRPTracker::LiveRegSet tmpSavingOutputLive = savingOutputLive;
  std::pair<int, int> curSaving = calculateSaving(
      hotBB, partialSubExps, tmpSavingInputLive, tmpSavingOutputLive,
      IsVOutBound, IsSOutBound, IsCanClone, DT, MRI, SIRI);
  const int VLimit = status.TargetVLimit;
  const int SLimit = status.TargetSLimit;

  vgpr += curSaving.first;
  sgpr += curSaving.second;

  if (vgpr <= VLimit && sgpr <= SLimit) {
    // nrmSubExps can help reach target occupancy, add it to
    // subExpCandidates.
    addExpCandidates(subExpCandidates, partialSubExps, usedRegs);
    return true;
  }

  if (EnableSubExpAggressive) {
    // Build candidates from passThrus  used in partialSubExps.
    GCNRPTracker::LiveRegSet sinkUsedRegs;
    for (auto &Exp : partialSubExps) {
      for (auto &Reg : Exp.BottomRegs) {
        sinkUsedRegs[Reg];
      }
    }
    MapVector<MachineBasicBlock *, GCNRPTracker::LiveRegSet> HoistCandidates;
    for (auto &it : hotBB.inputLive) {
      unsigned Reg = it.first;
      // Skip reg which already used for sink exp.
      if (sinkUsedRegs.count(Reg))
        continue;
      if (usedRegs.count(Reg))
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

    SlotIndexes *slotIndexes = LIS->getSlotIndexes();
    // Build exp dag on define blocks.
    std::vector<SubExp> hoistSubExpCandidates;
    // Save profit candidates into list.
    for (auto it : HoistCandidates) {
      MachineBasicBlock *UseMBB = it.first;
      // Try to remove out reg def sub exp from DefMBB.
      GCNRPTracker::LiveRegSet &UseInMBB = it.second;
      // Go up on the dag until reach share node.
      auto subExps = buildSubExpFromCandidatesTopBottom(
          Remat, UseInMBB, UseMBB, SIRI, SIII, MRI, slotIndexes);
      for (SubExp &subExp : subExps) {
        if (!canHelpPressureWhenHoist(subExp, MRI, SIRI, SIII, MLI,
                                      IsSOutBound))
          continue;
        subExp.IsHoist = true;
        hoistSubExpCandidates.emplace_back(subExp);
      }
    }

    std::pair<int, int> hoistSaving = calculateSaving(
        hotBB, hoistSubExpCandidates, tmpSavingInputLive, tmpSavingOutputLive,
        IsVOutBound, IsSOutBound, IsCanClone, DT, MRI, SIRI);

    int hoistVgpr = vgpr + hoistSaving.first;
    int hoistSgpr = sgpr + hoistSaving.second;

    if ((hoistVgpr <= VLimit && hoistSgpr <= SLimit) ||
        // If status not balance, do the remat even cannot reach target.
        // TODO: check the result not help even one occupancy.
        (!hoistSubExpCandidates.empty() && !status.NotBalance &&
         TargetOccupancy != 0)) {
      // nrmSubExps can help reach target occupancy, add it to
      // subExpCandidates.
      addExpCandidates(subExpCandidates, partialSubExps, usedRegs);
      addExpCandidates(subExpCandidates, hoistSubExpCandidates, usedRegs);

      return true;
    }
  }

  if (EnableVmemDegree &&
      // Only expect vmem when last tryToAddSubExps.
      // If not, AllowPartialUseInSubExp will no chance to be true.
      (AllowPartialUseInSubExp || !EnableSubExpAggressive)) {
    // Assume vmemLdSize could be optimized by not parallel.
    if (((vgpr - hotBB.vmemLdInputSize) <= VLimit ||
         (vgpr - hotBB.vmemLdOutputSize) <= VLimit) &&
        sgpr <= SLimit) {
      // nrmSubExps can help reach target occupancy, add it to
      // subExpCandidates.
      addExpCandidates(subExpCandidates, partialSubExps, usedRegs);
      return true;
    }
  }

  int vDistance = vgpr - (int)VLimit;
  int sDistance = status.TargetOcc > 4 ? (sgpr - (int)SLimit) : 0;
  int vSaved = hotBB.maxPressures.first - vgpr;
  int sSaved = hotBB.maxPressures.second - sgpr;
  // Try to add inBlockCloneSubExps.
  if (!tryRematInHotSpot(*hotBB.MBB, status, vDistance, sDistance, vSaved,
                         sSaved, inBlockCloneSubExps, inBlockHotVInstMap,
                         inBlockHotSInstMap, LIS, MRI, SIRI, SIII)) {
    // return false always when not allow partialUseInSubExp, it will try again
    // with partialUseInSubExp enabled.
    if (!AllowPartialUseInSubExp)
      return false;
    // If status not balance, do the remat even cannot reach target.
    // TODO: check the result not help even one occupancy.
    if (!status.NotBalance && TargetOccupancy == 0)
      return false;
  }
  // nrmSubExps can help reach target occupancy, add it to
  // subExpCandidates.
  addExpCandidates(subExpCandidates, partialSubExps, usedRegs);
  return true;
}

// Remat passthru regs per hot block.
// Reason to do it per block is to make sure passthru reuse is precise.
// If try remat on all hot blocks together, the passthru might be on one block,
//  reuse in on another block which the reg is not passthru there.
bool perBlockPassthruRemat(Remat *Remat, std::vector<HotBlock> &hotBlocks,
                           RematStatus &status,
                           GCNRPTracker::LiveRegSet &liveRegCandidates,
                           const GCNSubtarget *ST, LiveIntervals *LIS,
                           const MachineLoopInfo *MLI,
                           MachineDominatorTree *DT, MachineRegisterInfo &MRI,
                           const SIRegisterInfo *SIRI,
                           const SIInstrInfo *SIII) {
  bool IsUpdated = false;
  bool IsCanClone = EnableSubExpClone || EnableSubExpAggressive;

  SlotIndexes *slotIndexes = LIS->getSlotIndexes();
  // Sort hot blocks by pressure first.
  // The hot block with higher pressure is easier to fail.
  // If fail, fail fast. It it works, save the subExpCandidates. The
  // subExpCandidates may help other hotblocks.
  std::sort(hotBlocks.begin(), hotBlocks.end(),
            [&ST](const HotBlock &a, const HotBlock &b) {
              return pressureHigher(a.maxPressures.first, a.maxPressures.second,
                                    b.maxPressures.first, b.maxPressures.second,
                                    ST);
            });

  std::vector<SubExp> subExpCandidates;
  // For inBlock remat clone.
  std::vector<SubExp> inBlockCloneSubExps;
  DenseMap<MachineBasicBlock *, MachineInstr *> inBlockHotVInstMap;
  DenseMap<MachineBasicBlock *, MachineInstr *> inBlockHotSInstMap;

  // Save used passThrus to avoid use same reg on different MBB.
  GCNRPTracker::LiveRegSet usedPassThrus;
  // Save moved regs to avoid use same reg hoist and sink.
  GCNRPTracker::LiveRegSet usedRegs;

  const int VLimit = status.TargetVLimit;
  const int SLimit = status.TargetSLimit;
  // Collect passthru for hot block.
  // Try remat on it.
  for (auto &it : hotBlocks) {
    MachineBasicBlock *MBB = it.MBB;

    const GCNRPTracker::LiveRegSet inputLive = status.MBBInputLiveMap[MBB];
    const GCNRPTracker::LiveRegSet outputLive = status.MBBOutputLiveMap[MBB];

    it.inputLive = inputLive;

    // Add pressure by 1 to consider spill to vgpr.
    const int PressureDelta = -1;
    int vgpr = it.maxPressures.first - PressureDelta;
    int sgpr = it.maxPressures.second;
    bool IsVOutBound = vgpr > VLimit;
    bool IsSOutBound = sgpr > SLimit;
    // savingInputLive is used to calculate saving which will be modified to
    // avoid count same input multiple times.
    GCNRPTracker::LiveRegSet savingInputLive = inputLive;
    GCNRPTracker::LiveRegSet savingOutputLive = outputLive;
    std::pair<int, int> curSaving =
        calculateSaving(it, subExpCandidates, savingInputLive, savingOutputLive,
                        IsVOutBound, IsSOutBound, IsCanClone, DT, MRI, SIRI);

    vgpr += curSaving.first;
    sgpr += curSaving.second;

    if (vgpr <= VLimit && sgpr <= SLimit)
      continue;

    // Collect pass thru regs.
    GCNRPTracker::LiveRegSet passThrus =
        collectPassThrus(MBB, inputLive, outputLive, usedPassThrus,
                         liveRegCandidates, MRI, IsCanClone);

    // Group pass thru regs by def MBB.
    SmallVector<std::pair<MachineBasicBlock *, GCNRPTracker::LiveRegSet>>
        Candidates = groupPassThruByDefBlock(Remat, passThrus, usedPassThrus,
                                             MRI, SIRI, SIII);
    // unUsedPassThrus used to collect passThru which is skipped when build
    // subExp.
    GCNRPTracker::LiveRegSet unusedPassThrus;
    // Build exp dag on define blocks.
    bool AllowPartialUseInSubExp = false;
    if (tryToAddSubExps(
            Remat, it, status, subExpCandidates, inBlockCloneSubExps,
            inBlockHotVInstMap, inBlockHotSInstMap, Candidates, vgpr, sgpr,
            savingInputLive, savingOutputLive, passThrus, usedRegs, MRI, SIRI,
            SIII, MLI, slotIndexes, LIS, DT, IsCanClone, IsVOutBound,
            IsSOutBound, unusedPassThrus, AllowPartialUseInSubExp)) {
      // Remove unusedPassThrus from passThrus first.
      llvm::andNotLiveRegSet(passThrus, unusedPassThrus);
      llvm::mergeLiveRegSet(usedPassThrus, passThrus);
      continue;
    }
    // If cannot clone, don't need to try partialUseInSubExp which must clone.
    if (!IsCanClone)
      return false;

    // Partial use subExp may result  count caused by clone.
    // Only try it when enable aggressive remat.
    if (!EnableSubExpAggressive)
      return false;

    AllowPartialUseInSubExp = true;
    if (!tryToAddSubExps(
            Remat, it, status, subExpCandidates, inBlockCloneSubExps,
            inBlockHotVInstMap, inBlockHotSInstMap, Candidates, vgpr, sgpr,
            savingInputLive, savingOutputLive, passThrus, usedRegs, MRI, SIRI,
            SIII, MLI, slotIndexes, LIS, DT, IsCanClone, IsVOutBound,
            IsSOutBound, unusedPassThrus, AllowPartialUseInSubExp)) {
      return false;
    }
    // Just merge all passThrus after tryToAddSubExps allow partialUseInSubExp.
    llvm::mergeLiveRegSet(usedPassThrus, passThrus);
  }

  // Apply changes.
  {
    // sort subExpCandidates to make sure input use apply before output use if a
    // reg is input and output of subExps.
    LLVM_DEBUG(for (SubExp &Exp : subExpCandidates) { Exp.dump(MRI, SIRI); });
    sortSubExpCandidates(subExpCandidates);

    for (SubExp &Exp : subExpCandidates) {
      // Skip exp which is cleared in sort for hoist sink conflict.
      if (Exp.SUnits.empty())
        continue;
      LLVM_DEBUG(Exp.dump(MRI, SIRI));
      if (Exp.IsHoist) {
        ApplySubExpMoveNearDefine(Exp, MRI, DT, slotIndexes, SIII, SIRI);
      } else {
        if (Exp.IsCloneOnly)
          ApplySubExpCloneNearUser(Exp, hotBlocks, DT, MRI, slotIndexes, SIII,
                                   SIRI);
        else
          ApplySubExpMoveNearUser(Exp, MRI, DT, slotIndexes, SIII, SIRI);
      }
    }

    for (SubExp &Exp : inBlockCloneSubExps) {
      ApplySubExpCloneNearUserInBlock(Exp, inBlockHotVInstMap,
                                      inBlockHotSInstMap, MRI, slotIndexes,
                                      SIII, SIRI);
    }
    // Try to see possible occupancy could reach, then dicide a target.
    // Apply remat.
    IsUpdated = subExpCandidates.size();
  }

  return IsUpdated;
}

int getVMemLdSize(MachineBasicBlock &MBB, const SIInstrInfo *SIII,
                  const SIRegisterInfo *SIRI, const MachineRegisterInfo &MRI) {
  int vmemLdSize = 0;
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
    LaneBitmask mask = llvm::getRegMask(Dst, MRI);
    unsigned size = llvm::getRegSize(Dst.getReg(), mask, MRI, SIRI);
    vmemLdSize += size;
  }
  return vmemLdSize;
}

} // namespace

bool GroupRemat(Remat *Remat, MachineFunction &MF, MachineLoopInfo *MLI,
                LiveIntervals *LIS, MachineDominatorTree *DT,
                MachinePostDominatorTree *PDT, AliasAnalysis *AA) {
  if (MF.size() < 2)
    return false;
  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();

  const SIInstrInfo *SIII = ST->getInstrInfo();
  const SIRegisterInfo *SIRI = ST->getRegisterInfo();

  auto &MRI = MF.getRegInfo();

  RematStatus status = getRematStatus(MF, MLI, LIS, MRI, ST);

  const unsigned MaxOcc = ST->getWavesPerEU(MF.getFunction()).second;
  if (status.TargetOcc >= MaxOcc)
    return false;

  unsigned VLimit = status.TargetVLimit;
  unsigned SLimit = status.TargetSLimit;

  int rematVCnt = status.MaxVPressure - VLimit;
  int rematSCnt = status.MaxSPressure - SLimit;

  bool IsSGPRSpill = false;
  if (rematSCnt > 0) {
    IsSGPRSpill = nearSgprSpill(status.MaxSPressure, ST, MF);
  }

  // If bound by lds, skip.
  if ((status.TargetOcc + 1) > ST->getOccupancyWithWorkGroupSizes(MF).second &&
      !IsSGPRSpill)
    return false;

  bool IsBothOutLimit = rematVCnt > 0 && rematSCnt > 0;
  // TODO: use check wqm and support vreg remat.
  bool IsCheckWQM = MF.getFunction().getCallingConv() == CallingConv::AMDGPU_PS;
  rematVCnt = IsCheckWQM & false;

  // Remat on every hot block.

  // Collect all hot blocks.
  std::vector<HotBlock> hotBlocks;
  for (MachineBasicBlock &MBB : MF) {
    // Collect reg pressure.
    auto &RP = status.MBBPressureMap[&MBB];
    unsigned maxLocalVPressure = RP.getVGPRNum(ST->hasGFX90AInsts());
    unsigned maxLocalSPressure = RP.getMaxSGPR();

    maxLocalSPressure += RegForVCC;

    if (!EnableInBlockRemat) {
      if (maxLocalVPressure <= VLimit && maxLocalSPressure <= SLimit)
        continue;
    }

    // Move inst which input is imm/pass thru reg/out reg to help pressure.
    if (tryHoldPacifist(MBB, LIS, MRI, SIRI, SIII, AA, status)) {
      maxLocalVPressure = 0;
      maxLocalSPressure = 0;
      collectMBBPressure(MBB, LIS, ST, maxLocalVPressure, maxLocalSPressure,
                         status);

      maxLocalSPressure += RegForVCC;
    }
    if (maxLocalVPressure <= VLimit && maxLocalSPressure <= SLimit)
      continue;

    // When both vgpr sgpr out limit, only help vgpr.
    if (IsBothOutLimit && maxLocalVPressure <= VLimit)
      continue;
    GCNRPTracker::LiveRegSet liveSet;
    hotBlocks.push_back({&MBB, liveSet,
                         std::make_pair(maxLocalVPressure, maxLocalSPressure),
                         0, 0});
  }
  // Collect vmemLdInput/OutputSize.
  if (EnableVmemDegree) {
    DenseMap<MachineBasicBlock *, unsigned> outputVMemLdSizeMap;
    for (auto it : hotBlocks) {
      MachineBasicBlock *MBB = it.MBB;
      // Collect vmemLd when enable split.
      int vmemLdSize = getVMemLdSize(*MBB, SIII, SIRI, MRI);
      if (vmemLdSize) {
        outputVMemLdSizeMap[MBB] = vmemLdSize;
      }
    }
    for (auto &it : hotBlocks) {
      MachineBasicBlock *MBB = it.MBB;

      auto oit = outputVMemLdSizeMap.find(MBB);
      if (oit != outputVMemLdSizeMap.end())
        it.vmemLdOutputSize = oit->second;

      if (MBB->pred_size() != 1)
        continue;

      MachineBasicBlock *Pred = *MBB->pred_begin();
      oit = outputVMemLdSizeMap.find(Pred);
      if (oit != outputVMemLdSizeMap.end()) {
        it.vmemLdInputSize = oit->second;
      } else {
        if (Pred->getFirstTerminator() != Pred->end())
          continue;
        if (Pred->empty())
          continue;
        bool IsHighLatency = SIII->isHighLatencyInstruction(Pred->back());
        if (!IsHighLatency)
          continue;
        int vmemLdSize = getVMemLdSize(*Pred, SIII, SIRI, MRI);
        it.vmemLdInputSize = vmemLdSize;
      }
    }
  }

  if (EnableUniformVectorToScalar) {
    if (rematUniformVgprToSgpr(Remat, MF, status, status.MBBPressureMap,
                               hotBlocks, LIS, MRI, SIRI, SIII, MLI)) {
      // Rebuild LIS.
      LIS->reanalyze(MF);
      status = getRematStatus(MF, MLI, LIS, MRI, ST);
      bool IsSgprSpilled = nearSgprSpill(status.MaxSPressure, ST, MF);
      if (IsSgprSpilled) {
        bool IsNearTarget = false;
        hotBlockRemat(Remat, MF, MLI, LIS, DT, PDT, IsNearTarget);
        // Rebuild LIS.
        LIS->reanalyze(MF);
        status = getRematStatus(MF, MLI, LIS, MRI, ST);
      }

      for (auto &it : hotBlocks) {
        MachineBasicBlock *MBB = it.MBB;

        // Update pressure.
        auto &RP = status.MBBPressureMap[MBB];
        unsigned maxLocalVPressure = RP.getVGPRNum(ST->hasGFX90AInsts());
        unsigned maxLocalSPressure = RP.getMaxSGPR();

        maxLocalSPressure += RegForVCC;
        it.maxPressures.first = maxLocalVPressure;
        it.maxPressures.second = maxLocalSPressure;
      }
    }
  }

  // Collect all live reg which cross hot blocks.
  GCNRPTracker::LiveRegSet liveRegCandidates;
  for (auto it : hotBlocks) {
    MachineBasicBlock *MBB = it.MBB;

    const GCNRPTracker::LiveRegSet inputLive = status.MBBInputLiveMap[MBB];

    const GCNRPTracker::LiveRegSet outputLive = status.MBBOutputLiveMap[MBB];

    llvm::mergeLiveRegSet(liveRegCandidates, inputLive);
    llvm::mergeLiveRegSet(liveRegCandidates, outputLive);
  }

  // Check min VGPR bound.
  BlockSet PressureUnderLimitSet;
  if (EnableSubExpMinReg) {
    for (auto &it : hotBlocks) {
      MachineBasicBlock *MBB = it.MBB;
      unsigned MaxLocalVGPR = 0;
      unsigned MaxLocalSGPR = 0;
      llvm::getRegBound(MBB, MRI, SIRI, SIII, LIS, MaxLocalVGPR, MaxLocalSGPR);

      if (MaxLocalVGPR < VLimit && MaxLocalSGPR < SLimit) {
        PressureUnderLimitSet.insert(MBB);
      } else {
        if (MaxLocalVGPR < it.maxPressures.first)
          it.maxPressures =
              std::make_pair(MaxLocalVGPR, it.maxPressures.second);
        if (MaxLocalSGPR < it.maxPressures.second)
          it.maxPressures = std::make_pair(it.maxPressures.first, MaxLocalSGPR);
      }
    }
  }

  bool IsUpdated =
      perBlockPassthruRemat(Remat, hotBlocks, status, liveRegCandidates, ST,
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

    IsUpdated = GroupRemat(this, MF, MLI, LIS, DT, PDT, AA);

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
