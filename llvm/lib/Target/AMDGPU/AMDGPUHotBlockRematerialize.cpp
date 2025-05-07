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
#include "GCNRegPressure.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/SlotIndexes.h"

#define DEBUG_TYPE "amdgpu-hot-block-remat"

using namespace llvm;

static cl::opt<bool>
    EnableAggressiveSgpr("amdgpu-remat-enable-hot-block-remat-aggressive-sgpr");
static cl::opt<unsigned> TargetOccupancy("amdgpu-remat-target-occupancy");

namespace {

typedef DenseSet<MachineInstr *> InstSet;
typedef DenseSet<MachineBasicBlock *> BlockSet;
template <typename T> using BlockMap = MapVector<MachineBasicBlock *, T>;

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
  Register Reg;
  MachineInstr *DefMI;
  MachineBasicBlock *InsertBlock;
  union {
    MachineInstr *InsertPointMI;
    unsigned UserCount;
  };
  RematKind Kind;
  unsigned Size; // This is actually the Gain of the candidate.
};

struct BlockLiveInfo {
  MachineBasicBlock *BB;
  unsigned MaxSReg;
  unsigned MaxVReg;
  // Input live is the live reg which cross block.
  const GCNRPTracker::LiveRegSet InputLive;
};

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

  void applyCloneRemat(RematNode &Node, std::vector<BlockLiveInfo> &HotBlocks,
                       MachineDominatorTree *DT, MachineRegisterInfo &MRI,
                       SlotIndexes *SlotIndexes, const SIRegisterInfo *SIRI,
                       const SIInstrInfo *SIII, LiveIntervals *LIS,
                       MachineFunction &MF);
  void applyRemat(MapVector<Register, RematNode> &RematMap,
                  std::vector<BlockLiveInfo> &HotBlocks,
                  MachineDominatorTree *DT, llvm::SlotIndexes *SlotIndexes,
                  MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                  const SIInstrInfo *SIII, LiveIntervals *LIS,
                  MachineFunction &MF);
  bool hotBlockRemat(MachineFunction &MF, MachineLoopInfo *MLI,
                     LiveIntervals *LIS, MachineDominatorTree *DT,
                     MachinePostDominatorTree *PDT, bool &IsNearTarget);

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
};

MachineBasicBlock::iterator adjustInsertPointToAvoidSccSmash(
    MachineInstr *InstructionToMove, MachineBasicBlock *MBB,
    MachineBasicBlock::iterator CurrentInsertPoint, MachineRegisterInfo &MRI,
    const SIRegisterInfo *SIRI, const SIInstrInfo *SIII, LiveIntervals *LIS) {
  const bool WillSmashScc =
      InstructionToMove->modifiesRegister(AMDGPU::SCC, SIRI);
  if (WillSmashScc) {
    CurrentInsertPoint = llvm::findOrCreateInsertionPointForSccDef(
        MBB, CurrentInsertPoint, SIRI, SIII, &MRI, LIS);
  }

  return CurrentInsertPoint;
}

DenseMap<MachineBasicBlock *, BlockSet> reduceClonedMBBs(
    Register Reg, BlockMap<SmallVector<MachineInstr *, 2>> &UserBlocks,
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
      AfterHotRangeMBBs.insert(MBB);
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

void updateUsers(Register Reg, unsigned NewReg, bool IsSubRegDef,
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
  const unsigned OpNum = DefMI->getNumOperands();

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
    if (MBB == DefMI->getParent())
      continue;
    // Skip MBB which share clone from other MBBs.
    if (UserMBBSet.count(MBB) == 0)
      continue;

    Register NewReg = MRI.createVirtualRegister(RC);
    auto NewDef = BuildMI(MF, DL, Desc).addDef(NewReg);
    for (unsigned I = 1; I < OpNum; I++)
      NewDef = NewDef.add(DefMI->getOperand(I));

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
        DefMI, InsertPointMI->getParent(), InsertPointMI, MRI, SIRI, SIII, LIS);

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
  }
  if (MRI.use_empty(Reg)) {
    SlotIndexes->removeSingleMachineInstrFromMaps(*DefMI);
  }
}

void applyOneDefOneUseRemat(RematNode &Node, MachineRegisterInfo &MRI,
                            SlotIndexes *SlotIndexes,
                            const SIRegisterInfo *SIRI, const SIInstrInfo *SIII,
                            LiveIntervals *LIS) {
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
                                                 SIRI, SIII, LIS);

  // Move instruction to new location.
  DefMI->removeFromParent();
  InsertPoint->getParent()->insert(InsertPoint, DefMI);

  // Update slot index.
  SlotIndexes->removeSingleMachineInstrFromMaps(*DefMI);
  SlotIndexes->insertMachineInstrInMaps(*DefMI);
}

void AMDGPUHotBlockRematerialize::applyRemat(
    MapVector<Register, RematNode> &RematMap,
    std::vector<BlockLiveInfo> &HotBlocks, MachineDominatorTree *DT,
    llvm::SlotIndexes *SlotIndexes, MachineRegisterInfo &MRI,
    const SIRegisterInfo *SIRI, const SIInstrInfo *SIII, LiveIntervals *LIS,
    MachineFunction &MF) {
  std::vector<RematNode> UpdateList;
  for (auto &It : RematMap)
    UpdateList.emplace_back(It.second);

  // Sort update list with slotIndex to make sure def moved before use.
  // If use moved before def, It might not be the first use anymore.
  std::sort(UpdateList.begin(), UpdateList.end(),
            [&SlotIndexes](RematNode &I, RematNode &J) {
              SlotIndex A = SlotIndexes->getInstructionIndex(*I.DefMI);
              SlotIndex B = SlotIndexes->getInstructionIndex(*J.DefMI);
              return A < B;
            });

  for (RematNode &Node : UpdateList) {
    if (Node.Kind == RematNode::RematKind::OneDefOneUse)
      applyOneDefOneUseRemat(Node, MRI, SlotIndexes, SIRI, SIII, LIS);
    else if (Node.Kind == RematNode::RematKind::Clone)
      applyCloneRemat(Node, HotBlocks, DT, MRI, SlotIndexes, SIRI, SIII, LIS,
                      MF);
  }
}

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
  if (!llvm::getNonDebugMBBEnd(BBEnd, MBB))
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
  if (SPressure > MaxSPressure)
    MaxSPressure = SPressure;
  if (RP.getVGPRNum(ST->hasGFX90AInsts()) > MaxVPressure)
    MaxVPressure = RP.getVGPRNum(ST->hasGFX90AInsts());
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
      if (llvm::getNonDebugMBBEnd(BBEnd, MBB)) {
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
      if (LIS->intervalIsInOneMBB(LI))
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
      dbgs() << "output live"; for (auto &It : Status.MBBOutputLiveMap) {
        unsigned Idx = It.first->getNumber();
        auto LiveReg = It.second;
        dbgs() << "MBB" << Idx << ":";
        llvm::dumpLiveSet(LiveReg, SIRI);
      } dbgs() << "input live";
      for (auto &It : Status.MBBInputLiveMap) {
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

  llvm::SchedScore TotalScore = llvm::collectLatency(MF, *ST, MLI);
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
    Register RegSize = RC->getLaneMask().getNumLanes();
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
    if (Mask & SInputMask)
      SInputPressure += 4;
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

// Skip live reg remated to other block.
void updateLiveInfo(
    const MapVector<Register, RematNode> &RematMap,
    GCNRPTracker::LiveRegSet &LiveSet,
    const GCNRPTracker::LiveRegSet &InputLive, const MachineBasicBlock *CurBB,
    DenseMap<const MachineBasicBlock *, unsigned> &RPOTIndexMap) {
  for (auto &It : RematMap) {
    Register Reg = It.first;
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
      if (LiveBBIndex > InsertBBIndex)
        continue;
    }
    // Already in remat map, don't need to check again, remove from
    // candidate.
    LiveSet.erase(Reg);
  }
}

// Returns the actual register saving that would be achieved by moving or
// cloning this instruction. It's essentially:
//
//     size(defs) - size(uses)
//
// Note if it is not safe to move/clone this instruction, this function returns
// 0.
//
int rematGainInBits(MachineInstr *DefMI, Register Reg,
                    const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                    bool IsVGPR) {
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
      IsSingleDef = llvm::isSub0Sub1SingleDef(Reg, MRI);
    }

    if (IsSingleDef) {
      // The reg might share with other candidates,  check It here.
      // Count share reg in getReducedSize.
      if (EnableAggressiveSgpr) {
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

MachineBasicBlock *
findInsertBlock(MachineInstr &DefMI, Register Reg, MachineDominatorTree *DT,
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

bool isSafeToMoveOrClone(MachineInstr *DefMI, MachineRegisterInfo &MRI) {
  // Do not move PHI nodes
  if (isUsedByPhi(DefMI, MRI))
    return false;

  unsigned OpNum = DefMI->getNumOperands();
  // Only move DefMI which all operand is unique def.
  for (unsigned I = 0; I < OpNum; I++) {
    MachineOperand &Op = DefMI->getOperand(I);
    if (!Op.isReg())
      continue;
    if (!Op.getReg().isPhysical() && !MRI.getUniqueVRegDef(Op.getReg()) &&
        !llvm::isSub0Sub1SingleDef(Op.getReg(), MRI)) {
      return false;
    }
  }
  return true;
}

void addOneDefOneUseCandidate(std::vector<RematNode> *OutRematList,
                              int *OutRematCnt, const RematNode &Node,
                              MachineRegisterInfo &MRI,
                              MachineDominatorTree *DT,
                              MachinePostDominatorTree *PDT,
                              MachineLoopInfo *MLI, bool IsVGPR,
                              bool MemBound) {
  Register Reg = Node.Reg;
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

  RematNode FilteredNode = Node;
  FilteredNode.InsertBlock = InsertBB;
  FilteredNode.InsertPointMI = UseMI;
  FilteredNode.Kind = RematNode::RematKind::OneDefOneUse;
  OutRematList->emplace_back(FilteredNode);
  *OutRematCnt += Size;
}

// Build remat candidates from the registers in `CandidateRegSet`.
void buildRematCandiates(std::vector<RematNode> *OutCandidates,
                         DenseSet<Register> *PinnedRegSet,
                         GCNRPTracker::LiveRegSet &CandidateRegSet,
                         const MachineRegisterInfo &MRI,
                         const SIInstrInfo *SIII, const SIRegisterInfo *SIRI,
                         bool IsVGPR) {

  for (const auto &LiveRegIt : CandidateRegSet) {
    Register Reg = LiveRegIt.first;
    // Skip unsafe reg.
    if (PinnedRegSet->count(Reg))
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
      int Gain = rematGainInBits(MI, Reg, MRI, SIRI, IsVGPR);
      if (Gain > 0)
        OutCandidates->emplace_back(RematNode(Reg, MI, Gain >> 5));
      else
        IsSafeCandidate = false;
    }
    // Save unsafe reg.
    if (!IsSafeCandidate)
      PinnedRegSet->insert(Reg);
  }

  // Sort by gain.
  std::sort(OutCandidates->begin(), OutCandidates->end(),
            [](RematNode &I, RematNode &J) { return I.Size > J.Size; });
}

void addCloneCandidate(std::vector<RematNode> *OutRematList, int *OutRematCnt,
                       DenseSet<Register> *OutPinnedRegSet,
                       std::vector<RematNode *> &&CloneList,
                       const MachineRegisterInfo &MRI) {
  // Group user in same blocks.
  std::vector<BlockSet> UserSetList(CloneList.size());

  for (size_t I = 0; I < CloneList.size(); I++) {
    auto *Node = CloneList[I];
    Register Reg = Node->Reg;
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
        OutPinnedRegSet->insert(Reg);
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
    OutRematList->emplace_back(*Node);
    *OutRematCnt += Node->Size;
  }
}

// Filter `Candidates` into `OutRematList` based on whether
// safe to move, and decides on the actual type of Candidate (move vs cline).
//
// Updates `OutPinnedRegSet` with registers that cannot/should not be moved.
//
// Returns the accumulated size of all filtered candidates.
//
int filterRematCandiates(std::vector<RematNode> *OutRematList,
                         DenseSet<Register> *OutPinnedRegSet,
                         std::vector<RematNode> &&Candidates,
                         MachineDominatorTree *DT,
                         MachinePostDominatorTree *PDT, MachineLoopInfo *MLI,
                         MachineRegisterInfo &MRI, bool IsVGPR, bool MemBound) {
  int RematCnt = 0;
  // Work one def one use first.
  for (auto &Node : Candidates) {
    Register Reg = Node.Reg;
    if (!MRI.hasOneNonDBGUse(Reg))
      continue;

    MachineInstr *DefMI = Node.DefMI;
    if (!isSafeToMoveOrClone(DefMI, MRI)) {
      OutPinnedRegSet->insert(Reg);
      continue;
    }

    addOneDefOneUseCandidate(OutRematList, &RematCnt, Node, MRI, DT, PDT, MLI,
                             IsVGPR, MemBound);
  }

  if (!IsVGPR) {
    std::vector<RematNode *> CloneList;
    // Try multi use case.
    for (auto &Node : Candidates) {
      Register Reg = Node.Reg;
      if (MRI.hasOneNonDBGUse(Reg))
        continue;

      MachineInstr *DefMI = Node.DefMI;
      if (!isSafeToMoveOrClone(DefMI, MRI)) {
        OutPinnedRegSet->insert(Reg);
        continue;
      }

      // Clone for each user.
      CloneList.emplace_back(&Node);
    }

    addCloneCandidate(OutRematList, &RematCnt, OutPinnedRegSet,
                      std::move(CloneList), MRI);
  }

  return RematCnt;
}

// Calculate the reduced register pressure of RematMap w.r.t. the BB associated
// with LiveInfo.
// Returns the number of registers reduced, and the instructions associated with
// the reduction nodes into `OutReducedInsts`.
int getReducedSize(const MapVector<Register, RematNode> &RematMap,
                   GCNRPTracker::LiveRegSet &CanidateSet,
                   const MachineRegisterInfo &MRI,
                   const BlockLiveInfo &LiveInfo,
                   DenseMap<const MachineBasicBlock *, unsigned> &RPOTIndexMap,
                   InstSet *OutReducedInsts) {
  int ReducedSize = 0;
  for (const auto &It : RematMap) {
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
      OutReducedInsts->insert(Node.DefMI);
    }

    // Already in remat map, don't need to check again, remove from candidate.
    CanidateSet.erase(Reg);
  }

  return ReducedSize;
}

static unsigned getNumLanesIn32BitReg(bool IsVgpr) {
  const TargetRegisterClass *RC =
      IsVgpr ? &AMDGPU::VGPR_32RegClass : &AMDGPU::SGPR_32RegClass;
  return RC->LaneMask.getNumLanes();
}

// Calculate the amount of OVERLAPPING register pressure among all
// the instructions in `ReducedInsts`. E.g for:
//    x = COPY a:sgpr_32
//    y = COPY a:sgpr_32
// This function would return 1.
int getSharedReducedSize(const InstSet &ReducedInsts, bool IsVGPR,
                         const MachineRegisterInfo &MRI,
                         const SIRegisterInfo *SIRI) {
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

      if (IsVGPR != SIRI->isVGPR(MRI, MO.getReg()))
        // Not support mix of v and s when remat now.
        continue;

      const TargetRegisterClass *OpRC = MRI.getRegClass(Reg);
      const int MOSize = SIRI->getRegSizeInBits(*OpRC) >> 5;

      unsigned Mask = 0;
      // FIXME: Lane mask is now in the granularity of 16-bit lanes.
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

  const unsigned NumLanesPerReg = getNumLanesIn32BitReg(IsVGPR);
  return SharedSize / NumLanesPerReg;
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

// A heuristic number for keeping the target SGPR number away from the limit.
constexpr unsigned SgprLimitBias = 10;

bool AMDGPUHotBlockRematerialize::hotBlockRemat(MachineFunction &MF,
                                                MachineLoopInfo *MLI,
                                                LiveIntervals *LIS,
                                                MachineDominatorTree *DT,
                                                MachinePostDominatorTree *PDT,
                                                bool &IsNearTarget) {
  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();

  const SIInstrInfo *SIII = ST->getInstrInfo();
  const SIRegisterInfo *SIRI = ST->getRegisterInfo();

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  DenseMap<const MachineBasicBlock *, unsigned> RPOTIndexMap;
  for (const MachineBasicBlock *MBB : RPOT)
    RPOTIndexMap[MBB] = RPOTIndexMap.size();

  auto &MRI = MF.getRegInfo();

  bool IsUpdated = false;
  RematStatus Status = getRematStatus(MF, MLI, LIS, MRI, ST);

  const unsigned MaxOcc = ST->getWavesPerEU(MF.getFunction()).second;
  if (Status.TargetOcc >= MaxOcc)
    return false;

  // Early checks
  {
    int InitialRematSCnt = Status.MaxSPressure - Status.TargetSLimit;
    // when agressive sgpr remat, reserve some for allocation lost.
    if (EnableAggressiveSgpr)
      InitialRematSCnt += SgprLimitBias;

    bool InitialIsSGPRSpill = false;
    if (InitialRematSCnt > 0)
      InitialIsSGPRSpill = nearSgprSpill(Status.MaxSPressure, ST, MF);

    const bool InitialIsForceRematSgpr =
        InitialIsSGPRSpill || Status.NotBalance;

    // If bound by lds, skip.
    if (Status.TargetOcc > ST->getOccupancyWithWorkGroupSizes(MF).second &&
        !InitialIsForceRematSgpr)
      return false;
  }

  MachineBasicBlock *EntryMBB = &MF.front();

  auto *SlotIndexes = LIS->getSlotIndexes();

  // Reg which already marked remat.
  MapVector<Register, RematNode> VRematMap;
  MapVector<Register, RematNode> SRematMap;
  // Reg which cannot move around to remat.
  DenseSet<Register> PinnedRegSet;
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
      collectLiveSetPressure(LiveSet, MRI, SIRI, VPressure, SPressure);
      if (MaxVPressure < VPressure)
        MaxVPressure = VPressure;
      if (MaxSPressure < SPressure)
        MaxSPressure = SPressure;
    }
    MaxSPressure += RegForVCC + Status.InputPhysicalSPressure;
    if (MaxVPressure <= Status.TargetVLimit &&
        MaxSPressure <= Status.TargetSLimit)
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
    int VReduced = getReducedSize(VRematMap, CandidateRegs, MRI, LiveInfo,
                                  RPOTIndexMap, &VReducedInsts);
    int SReduced = getReducedSize(SRematMap, CandidateRegs, MRI, LiveInfo,
                                  RPOTIndexMap, &SReducedInsts);

    // Calculate size need to be remat for this BB.
    const int RematVCnt = MaxVPressure - VReduced - Status.TargetVLimit;
    const int RematSCnt = MaxSPressure - SReduced - Status.TargetSLimit;

    bool IsSGPRSpill = false;
    if (RematSCnt > 0)
      IsSGPRSpill = nearSgprSpill(MaxSPressure, ST, MF);

    bool IsForceRematSgpr = IsSGPRSpill || Status.NotBalance;
    // Try to add candidates into remat list.

    int NewRematSCnt = 0;
    if (RematSCnt > 0) {
      // Build candidate nodes.
      std::vector<RematNode> SRematCandidates;
      buildRematCandiates(&SRematCandidates, &PinnedRegSet, CandidateRegs, MRI,
                          SIII, SIRI, /*IsVGPR*/ false);

      LLVM_DEBUG(dumpCandidates(SRematCandidates, MBB->getNumber(), SIRI));
      std::vector<RematNode> SRematList;
      // Filter candidates.
      NewRematSCnt =
          filterRematCandiates(&SRematList, &PinnedRegSet,
                               std::move(SRematCandidates), DT, PDT, MLI, MRI,
                               /*IsVGPR*/ false, Status.MemBound);
      if (NewRematSCnt > RematSCnt) {
        // Has enough remat node to cover rematCnt.
        int RematCnt = 0;
        for (RematNode &Node : SRematList) {
          SRematMap[Node.Reg] = Node;
          RematCnt += Node.Size;
          // Stop if the size had reached the required amount, unless
          // aggressive is set.
          if (RematCnt > RematSCnt && !EnableAggressiveSgpr)
            break;
        }
        NewRematSCnt = 0;
      } else {
        for (RematNode &Node : SRematList) {
          SReducedInsts.insert(Node.DefMI);
        }
        // Check shared size. These are reg uses that are shared among all the
        // instructions. The overlap will not actually contribute to the
        // pressure increase when an instruction is moved/cloned, so it can be
        // treated as a gain.
        int SharedReducedSize =
            getSharedReducedSize(SReducedInsts, /*IsVGPR*/ false, MRI, SIRI);

        int LocalGains = 0;
        if (((NewRematSCnt + SharedReducedSize) + (int)SgprLimitBias) >=
            RematSCnt) {
          for (RematNode &Node : SRematList)
            SRematMap[Node.Reg] = Node;
        } else {
          if (!IsForceRematSgpr)
            return false;
          for (RematNode &Node : SRematList)
            SRematMap[Node.Reg] = Node;
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
            int Gain = rematGainInBits(&MI, Reg, MRI, SIRI,
                                       /*IsVGPR*/ false);
            if (Gain > 0) {
              // Skip case when DefMI has implicit define which used by UseMI.
              if (isImplicitDefUse(&MI, &UseMI))
                continue;
              RematNode Node = {Reg, &MI, (unsigned)Gain >> 5};
              Node.InsertPointMI = &UseMI;
              Node.Kind = RematNode::RematKind::OneDefOneUse;
              SRematMap[Reg] = Node;
              LocalGains += Node.Size;
            }
          }
        }
        NewRematSCnt =
            RematSCnt - NewRematSCnt - SharedReducedSize - LocalGains;
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
      if (IsVRematOK && IsSRematOK)
        IsUpdated = true;
      else if (IsSGPRSpill)
        IsUpdated = true;
    } else if (NeedSRemat) {
      if (IsSRematOK)
        IsUpdated = true;
    } else if (NeedVRemat) {
      if (IsVRematOK)
        IsUpdated = true;
    }
    // TODO: what to do when cannot reach target?
    if (NewRematSCnt > 0) {
      if ((unsigned)NewRematSCnt <= ST->getSGPRAllocGranule()) {
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
    applyRemat(SRematMap, HotBlocks, DT, SlotIndexes, MRI, SIRI, SIII, LIS, MF);
    LLVM_DEBUG(llvm::dbgs() << "after hotremat"; MF.print(dbgs()););
  }

  // Balance between vector and scalar if possible.
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
