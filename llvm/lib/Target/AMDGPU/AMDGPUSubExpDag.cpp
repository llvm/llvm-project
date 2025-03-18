//===----------- AMDGPUSubExpDag.cpp - AMDGPU Sub Expression DAG ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief AMDGPU Sub Expression DAG. Helper for building a dag based on sub
/// expressions.
//
//===----------------------------------------------------------------------===//

#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/SlotIndexes.h"

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/GraphWriter.h"

#include "llvm/Support/Debug.h"

#include "AMDGPUMIRUtils.h"
#include "AMDGPUSubExpDag.h"
#include "GCNRegPressure.h"
#include <unordered_set>

#define DEBUG_TYPE "xb-sub-exp-dag"
using namespace llvm;

namespace llvm {

// Expression Dag.

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void SubExp::dump(const MachineRegisterInfo &MRI,
                  const SIRegisterInfo *SIRI) const {
  dbgs() << "\nSubExp:\n";
  dbgs() << "input regs:\n";
  for (auto &Input : InputLive) {
    pressure::print_reg(Input.first, MRI, SIRI, llvm::dbgs());
    dbgs() << "\n";
  }
  dbgs() << "output regs:\n";
  for (auto &Output : OutputLive) {
    pressure::print_reg(Output.first, MRI, SIRI, llvm::dbgs());
    dbgs() << "\n";
  }

  for (MachineInstr *MI : SUnits) {
    MI->dump();
  }
  dbgs() << "End of SubExp\n";
}
#endif

bool SubExp::modifiesRegister(unsigned Reg, const SIRegisterInfo *SIRI) const {
  for (const MachineInstr *MI : SUnits) {
    if (MI->modifiesRegister(Reg, SIRI)) {
      return true;
    }
  }

  return false;
}

void SubExp::calcMaxPressure(const MachineRegisterInfo &MRI) {
  SMaxSize = std::max(SInputSize, SOutputSize);
  VMaxSize = std::max(VInputSize, VOutputSize);

  DenseMap<unsigned, LaneBitmask> LiveRegs;
  GCNRegPressure CurPressure;

  // Add output to pressure.
  for (MachineInstr *MI : BottomRoots) {
    for (MachineOperand &MO : MI->operands()) {
      if (!MO.isReg())
        continue;
      if (!MO.isDef())
        continue;
      Register Reg = MO.getReg();
      if (!Reg.isVirtual())
        continue;
      LaneBitmask Mask = getRegMask(MO, MRI);
      auto It = LiveRegs.find(Reg);
      if (It != LiveRegs.end()) {
        LiveRegs[Reg] = Mask | It->second;
      } else {
        LiveRegs[Reg] = Mask;
      }
    }
  }

  for (auto It : LiveRegs) {
    LaneBitmask EmptyMask;
    CurPressure.inc(It.first, EmptyMask, It.second, MRI);
  }

  for (auto It = SUnits.rbegin(); It != SUnits.rend(); It++) {
    MachineInstr *MI = *It;
    auto *ST =
        &MI->getMF()
             ->getSubtarget<GCNSubtarget>(); // TODO: Better way to get this.
    for (MachineOperand &MO : MI->operands()) {
      if (!MO.isReg())
        continue;
      Register Reg = MO.getReg();
      if (!Reg.isVirtual()) {
        if (Reg == AMDGPU::SCC)
          IsTouchSCC = true;
        continue;
      }

      LaneBitmask LiveMask = getRegMask(MO, MRI);
      LaneBitmask PrevMask;
      auto LiveIt = LiveRegs.find(Reg);
      if (LiveIt != LiveRegs.end()) {
        PrevMask = LiveIt->second;
      }

      if (MO.isDef()) {
        LiveMask = PrevMask & (~(LiveMask));
      } else {
        LiveMask = PrevMask | LiveMask;
      }

      CurPressure.inc(Reg, PrevMask, LiveMask, MRI);
      LiveRegs[Reg] = LiveMask;
    }

    unsigned SSize = CurPressure.getSGPRNum();
    unsigned VSize = CurPressure.getVGPRNum(ST->hasGFX90AInsts());
    if (SSize > SMaxSize)
      SMaxSize = SSize;
    if (VSize > VMaxSize)
      VMaxSize = VSize;
  }
}

bool SubExp::isSafeToMove(const MachineRegisterInfo &MRI) const {
  if (IsMultiDefOutput)
    return false;
  if (IsHasTerminatorInst)
    return false;
  if (IsUseIncomingReg)
    return false;

  // Input should be single def.
  for (unsigned Reg : TopRegs) {
    if (!MRI.hasOneDef(Reg) && !llvm::isSub0Sub1SingleDef(Reg, MRI))
      return false;
  }
  return true;
}

ExpDag::ExpDag(const llvm::MachineRegisterInfo &MRI,
               const llvm::SIRegisterInfo *SIRI, const SIInstrInfo *SIII,
               const bool IsJoinInput)
    : MRI(MRI), SIRI(SIRI), SIII(SIII), IsJoinInputToSubExp(IsJoinInput) {}

template <typename T>
void ExpDag::initNodes(const LiveSet &InputLiveReg, T &Insts) {
  unsigned NodeSize = InputLiveReg.size() + Insts.size();
  SUnits.reserve(NodeSize);

  for (MachineInstr *MI : Insts) {
    if (MI->isDebugInstr())
      continue;
    SUnits.emplace_back(MI, SUnits.size());
    SUnit *SU = &SUnits.back();
    SUnitMIMap[SU] = MI;
    MISUnitMap[MI] = SU;
  }

  for (auto It : InputLiveReg) {
    unsigned Reg = It.first;
    SUnits.emplace_back();
    SUnit *SU = &SUnits.back();
    SU->NodeNum = SUnits.size() - 1;
    SUnitInputMap[SU] = Reg;
    InputSUnitMap[Reg] = SU;
  }
}

template void ExpDag::initNodes<DenseSet<MachineInstr *>>(
    const LiveSet &InputLiveReg, DenseSet<MachineInstr *> &instRange);

template void ExpDag::initNodes<std::vector<MachineInstr *>>(
    const LiveSet &InputLiveReg, std::vector<MachineInstr *> &instRange);

template <typename T>
void ExpDag::build(const LiveSet &InputLiveReg, const LiveSet &OutputLiveReg,
                   T &Insts) {
  initNodes(InputLiveReg, Insts);
  addDataDep();
  addCtrlDep();
  buildSubExp(InputLiveReg, OutputLiveReg, SIRI, SIII);
}

template void
ExpDag::build<DenseSet<MachineInstr *>>(const LiveSet &InputLiveReg,
                                        const LiveSet &OutputLiveReg,
                                        DenseSet<MachineInstr *> &instRange);
template void ExpDag::build<std::vector<MachineInstr *>>(
    const LiveSet &InputLiveReg, const LiveSet &OutputLiveReg,
    std::vector<MachineInstr *> &instRange);

void ExpDag::buildSubExp(const LiveSet &StartLiveReg, const LiveSet &EndLiveReg,
                         const SIRegisterInfo *SIRI, const SIInstrInfo *SIII) {
  IntEqClasses SubtreeClasses(SUnits.size());
  std::vector<unsigned> PassThruInputs;
  for (SUnit &SU : SUnits) {
    if (SU.NumPredsLeft == 0 && SU.NumSuccsLeft == 0) {
      PassThruInputs.emplace_back(SU.NodeNum);
      continue;
    }
    if (!IsJoinInputToSubExp && !SU.isInstr())
      continue;
    // Join prev.
    for (SDep &PreDep : SU.Preds) {
      SUnit *PreSU = PreDep.getSUnit();
      if (!IsJoinInputToSubExp && !PreSU->isInstr())
        continue;
      SubtreeClasses.join(SU.NodeNum, PreSU->NodeNum);
    }
    // Join succ.
    for (SDep &SucDep : SU.Succs) {
      SUnit *SucSU = SucDep.getSUnit();
      SubtreeClasses.join(SU.NodeNum, SucSU->NodeNum);
    }
  }
  SubtreeClasses.compress();

  unsigned NumSubExps = SubtreeClasses.getNumClasses();
  // Not count PassThruInputs for subExps since they're exp with only 1 SU.
  // SubExpIndexMap is used to pack SubIdx within updated NumSubExps.
  NumSubExps -= PassThruInputs.size();
  SubExps.resize(NumSubExps);
  DenseMap<unsigned, unsigned> SubExpIndexMap;

  // Add SU to sub exp.
  for (SUnit &SU : SUnits) {
    if (SU.NumPredsLeft == 0 && SU.NumSuccsLeft == 0) {
      continue;
    }
    unsigned SubIdx = SubtreeClasses[SU.NodeNum];
    unsigned OriginSubIdx = SubIdx;
    // Pack subidx.
    if (SubExpIndexMap.count(SubIdx) == 0) {
      unsigned Count = SubExpIndexMap.size();
      SubExpIndexMap.insert(std::make_pair(SubIdx, Count));
    }
    SubIdx = SubExpIndexMap[SubIdx];
    // Use NodeQueueId as SubIdx. We don't do schedule on ExpDag.
    SU.NodeQueueId = SubIdx;

    SubExp &Exp = SubExps[SubIdx];
    auto It = SUnitInputMap.find(&SU);
    if (It != SUnitInputMap.end()) {
      // Input.
      Register Reg = It->second;
      Exp.TopRegs.insert(Reg);
    } else {
      MachineInstr *MI = SU.getInstr();
      MachineBasicBlock *MBB = MI->getParent();
      Exp.FromBB = MBB;
      for (MachineOperand &MO : MI->operands()) {
        if (!MO.isReg())
          continue;
        if (!MO.isUse())
          continue;
        Register Reg = MO.getReg();
        if (MRI.getLiveInPhysReg(Reg) || MRI.getLiveInVirtReg(Reg)) {
          Exp.IsUseIncomingReg = true;
        }
      }

      Exp.SUnits.emplace_back(MI);
      if (SU.NumSuccsLeft == 0) {
        Exp.BottomRoots.insert(MI);
        if (MI->isTerminator())
          Exp.IsHasTerminatorInst = true;
      }
      if (MI->isNotDuplicable())
        Exp.IsNotSafeToCopy = true;
      // Skip Scalar mem access since no scalar store.
      if (MI->mayLoadOrStore() && !SIII->isSMRD(*MI)) {
        Exp.IsHasMemInst = true;
      }
      // Add bottom regs.
      for (MachineOperand &MO : MI->operands()) {
        if (!MO.isReg())
          continue;
        if (!MO.isDef())
          continue;
        Register Reg = MO.getReg();
        // physical reg is not in live reg.
        if (!Reg.isVirtual())
          continue;
        if (SU.NumSuccsLeft) {
          // For SU which has used in current blk.
          // Check if used in other blks or subExps.
          bool IsUsedInOtherBlk = false;
          for (auto &UserMI : MRI.use_nodbg_instructions(Reg)) {
            if (UserMI.getParent() != MBB) {
              IsUsedInOtherBlk = true;
              break;
            }
            auto SuIt = MISUnitMap.find(&UserMI);
            // When UserMI is not in dag, treat it as other block.
            if (SuIt == MISUnitMap.end()) {
              IsUsedInOtherBlk = true;
              break;
            }
            SUnit *UseSU = SuIt->second;
            // UserMI should always be in same subExp.
            unsigned UseSubIdx = SubtreeClasses[UseSU->NodeNum];
            if (UseSubIdx != OriginSubIdx) {
              // When reg has multiple def, it is possible for user def in
              // different subExp.
              if (MRI.getUniqueVRegDef(Reg))
                llvm::report_fatal_error("user and def in different subExp");
              break;
            }
          }
          if (!IsUsedInOtherBlk)
            continue;
        }
        Exp.BottomRegs.insert(Reg);
        if (!MRI.getUniqueVRegDef(Reg)) {
          Exp.IsMultiDefOutput = true;
        }
      }
    }
  }
  // Calc reg for SubExp.
  // Get block live in and live out.
  // Only reg will miss live mask.
  for (SubExp &Exp : SubExps) {
    for (unsigned Reg : Exp.TopRegs) {
      auto It = StartLiveReg.find(Reg);
      assert(It != StartLiveReg.end() &&
             "cannot find input reg in block start live");
      Exp.InputLive[Reg] |= It->second;
    }

    for (unsigned Reg : Exp.BottomRegs) {
      auto It = EndLiveReg.find(Reg);
      if (It == EndLiveReg.end()) {
        //"cannot find output reg in block end live");
        // Bottom reg is killed inside current block, did not get out of the
        // block.
        // Or the bottom reg is not treat as output in this dag, not save to
        // OutputLive which will affect profit count.
        continue;
      }
      Exp.OutputLive[Reg] |= It->second;
    }

    collectLiveSetPressure(Exp.InputLive, MRI, SIRI, Exp.VInputSize,
                           Exp.SInputSize);
    collectLiveSetPressure(Exp.OutputLive, MRI, SIRI, Exp.VOutputSize,
                           Exp.SOutputSize);
  }
}

void ExpDag::addDataDep() {
  DenseMap<unsigned, MachineInstr *> CurDefMI;

  for (SUnit &SU : SUnits) {
    if (!SU.isInstr())
      continue;
    MachineInstr *MI = SU.getInstr();

    // Link use to the def.
    for (MachineOperand &MO : MI->operands()) {
      if (!MO.isReg())
        continue;
      if (MO.isDef())
        continue;

      Register Reg = MO.getReg();
      SUnit *DefSU = nullptr;

      auto CurDefIt = CurDefMI.find(Reg);
      // Check def inst first.
      if (CurDefIt != CurDefMI.end()) {
        MachineInstr *CurDef = CurDefIt->second;
        DefSU = MISUnitMap[CurDef];
      } else {
        // physical reg is not in live reg.
        if (!Reg.isVirtual())
          continue;
        if (MO.isUndef())
          continue;
        // Is it OK for degbug instr MO cannot find def?
        if (MI->isDebugInstr())
          continue;
        // Should be an input.
        assert(InputSUnitMap.count(Reg) > 0 && "cannot find def");
        DefSU = InputSUnitMap[Reg];
      }
      SU.addPred(SDep(DefSU, SDep::Data, Reg));
    }

    // Add def to curDefMI;
    for (MachineOperand &MO : MI->operands()) {
      if (!MO.isReg())
        continue;
      if (!MO.isDef())
        continue;
      Register Reg = MO.getReg();

      // For case like:
      // undef %808.sub0:sgpr_64 = COPY killed %795:sgpr_32
      // %808.sub1:sgpr_64 = S_MOV_B32 0
      // When partially write, link MI to previous def.
      if (MO.getSubReg() != 0) {
        SUnit *DefSU = nullptr;
        auto CurDefIt = CurDefMI.find(Reg);
        // Check def inst first.
        if (CurDefIt != CurDefMI.end()) {
          MachineInstr *CurDef = CurDefIt->second;
          DefSU = MISUnitMap[CurDef];
          // Add link between different defs.
          SU.addPred(SDep(DefSU, SDep::Data, Reg));
        }
      }

      CurDefMI[Reg] = MI;
    }
  }
}

void ExpDag::addCtrlDep() {
  // TODO: add depend for memory, barrier.
}

BlockExpDag::BlockExpDag(llvm::MachineBasicBlock *B, llvm::LiveIntervals *LIS,
                         const llvm::MachineRegisterInfo &MRI,
                         const llvm::SIRegisterInfo *SIRI,
                         const llvm::SIInstrInfo *SIII)
    : ExpDag(MRI, SIRI, SIII, /*IsJoinInput*/ true), LIS(LIS), MBB(B) {}

void BlockExpDag::build() {
  auto *SlotIndexes = LIS->getSlotIndexes();
  const auto StartIdx = SlotIndexes->getMBBStartIdx(MBB);
  const auto StartLiveReg = llvm::getLiveRegs(StartIdx, *LIS, MRI);

  const auto EndIdx = SlotIndexes->getMBBEndIdx(MBB);
  const auto EndLiveReg = llvm::getLiveRegs(EndIdx, *LIS, MRI);

  std::vector<MachineInstr *> Insts;
  for (MachineInstr &MI : *MBB) {
    Insts.emplace_back(&MI);
  }

  ExpDag::build(StartLiveReg, EndLiveReg, Insts);
}

void BlockExpDag::buildWithPressure() {
  auto *SlotIndexes = LIS->getSlotIndexes();
  const auto StartIdx = SlotIndexes->getMBBStartIdx(MBB);
  const auto StartLiveReg = llvm::getLiveRegs(StartIdx, *LIS, MRI);

  const auto EndIdx = SlotIndexes->getMBBEndIdx(MBB);
  const auto EndLiveReg = llvm::getLiveRegs(EndIdx, *LIS, MRI);

  std::vector<MachineInstr *> Insts;
  for (MachineInstr &MI : *MBB) {
    Insts.emplace_back(&MI);
  }

  ExpDag::build(StartLiveReg, EndLiveReg, Insts);
  // Build pressure.
  buildPressure(StartLiveReg, EndLiveReg);
}

void BlockExpDag::buildAvail(const LiveSet &PassThruSet,
                             DenseMap<SUnit *, LiveSet> &DagAvailRegMap) {
  DenseSet<SUnit *> Processed;

  DenseSet<SUnit *> WorkList;
  MachineInstr &BeginMI = MBB->instr_front();

  // Calc avaialbe for each node, live is avail & sum(input of success).
  // If a reg is avaiable from the node, then success node can use it from this
  // node. For dag live, pred output don't need to have all input a node needs.
  // As long as all pred outputs can cover inputs, it is OK.
  for (SUnit &SU : SUnits) {
    if (SU.NumPredsLeft == 0) {
      GCNDownwardRPTracker RP(*LIS);
      RP.reset(BeginMI, &PassThruSet);
      MachineInstr *MI = SU.getInstr();
      if (MI) {
        RP.reset(*MI, &PassThruSet);
        RP.advance();
      }
      DagAvailRegMap[&SU] = RP.getLiveRegs();

      // Add succ to work list.
      for (auto &Succ : SU.Succs) {
        SUnit *SuccSU = Succ.getSUnit();
        if (SuccSU->NumPredsLeft > 0)
          SuccSU->NumPredsLeft--;
        WorkList.insert(SuccSU);
      }
    }
  }
  while (!WorkList.empty()) {
    SmallVector<SUnit *, 4> ReadyNodes;
    for (SUnit *SU : WorkList) {
      if (SU->NumPredsLeft > 0)
        continue;
      ReadyNodes.emplace_back(SU);
      // Ready, move it to Processed.
      Processed.insert(SU);
      // Only update 1 node once.
      // Order of schedle here should not affect pressure.
      break;
    }

    for (SUnit *SU : ReadyNodes) {
      // Remove SU from worklist.
      WorkList.erase(SU);

      MachineInstr *MI = SU->getInstr();
      // Calc pressure based on pred nodes.
      GCNRPTracker::LiveRegSet DagLive;
      for (auto &Pred : SU->Preds) {
        SUnit *PredSU = Pred.getSUnit();
        GCNRPTracker::LiveRegSet PredLive = DagAvailRegMap[PredSU];

        GCNDownwardRPTracker RP(*LIS);
        RP.reset(BeginMI, &PredLive);
        if (MI) {
          RP.reset(*MI, &PredLive);
          // Update PredLive based on MI.
          RP.advance();
        }
        llvm::mergeLiveRegSet(DagLive, RP.getLiveRegs());
      }
      DagAvailRegMap[SU] = DagLive;

      // Add succ to work list.
      for (auto &Succ : SU->Succs) {
        SUnit *SuccSU = Succ.getSUnit();
        if (SuccSU->NumPredsLeft > 0)
          SuccSU->NumPredsLeft--;
        WorkList.insert(SuccSU);
      }
    }

    // Skip dead loop
    if (ReadyNodes.empty()) {
      printf("dead loop when build dag pressure");
      break;
    }
  }

  assert(WorkList.empty() && "schedule failed for available reg");
}

void BlockExpDag::buildPressure(const LiveSet &StartLiveReg,
                                const LiveSet &EndLiveReg) {
  if (MBB->empty())
    return;
  DenseMap<SUnit *, GCNRPTracker::LiveRegSet> DagAvailRegMap;
  GCNRPTracker::LiveRegSet PassThruSet;
  for (auto It : StartLiveReg) {
    Register Reg = It.first;
    auto EndReg = EndLiveReg.find(Reg);
    if (EndReg == EndLiveReg.end())
      continue;

    LaneBitmask Mask = It.second;
    LaneBitmask EndMask = EndReg->second;
    Mask &= EndMask;
    if (Mask.getAsInteger() == 0)
      continue;
    PassThruSet[Reg] = Mask;
  }

  // Build avial for each nodes.
  buildAvail(PassThruSet, DagAvailRegMap);

  // Calc avaialbe for each node, live is avail & sum(input of success).
  // If a reg is avaiable from the node, then success node can use it from this
  // node. For dag live, pred output don't need to have all input a node needs.
  // As long as all pred outputs can cover inputs, it is OK.
  DenseSet<SUnit *> Processed;

  DenseSet<SUnit *> WorkList;
  MachineInstr &BeginMI = MBB->instr_front();

  for (SUnit &SU : SUnits) {
    if (SU.NumSuccsLeft == 0) {
      // Calc pressure based on pass thru.
      // Using pass thru as base because output of current SU should not
      // affect other output SUs.
      GCNUpwardRPTracker RP(*LIS);
      RP.reset(BeginMI, &PassThruSet, /*After*/ true);
      MachineInstr *MI = SU.getInstr();
      if (MI) {
        RP.reset(*MI, &PassThruSet, /*After*/ true);
        RP.recede(*MI);
      }
      DagPressureMap[&SU] = RP.getLiveRegs();
      // Add pred to work list.
      for (auto &Pred : SU.Preds) {
        SUnit *PredSU = Pred.getSUnit();
        PredSU->NumSuccsLeft--;
        WorkList.insert(PredSU);
      }
    }
  }

  while (!WorkList.empty()) {
    SmallVector<SUnit *, 4> ReadyNodes;
    for (SUnit *SU : WorkList) {
      if (SU->NumSuccsLeft > 0)
        continue;
      ReadyNodes.emplace_back(SU);
      // Ready, move it to Processed.
      Processed.insert(SU);
      // Only update 1 node once.
      // Order of schedle here should not affect pressure.
      break;
    }

    for (SUnit *SU : ReadyNodes) {
      // Remove SU from worklist.
      WorkList.erase(SU);

      MachineInstr *MI = SU->getInstr();
      // Calc pressure based on succ nodes.
      GCNRPTracker::LiveRegSet DagLive;
      for (auto &Succ : SU->Succs) {
        SUnit *SuccSU = Succ.getSUnit();
        GCNRPTracker::LiveRegSet SuccLive = DagPressureMap[SuccSU];

        GCNUpwardRPTracker RP(*LIS);
        RP.reset(BeginMI, &SuccLive, /*After*/ true);
        if (MI) {
          RP.reset(*MI, &SuccLive, /*After*/ true);
          // Update SuccLive based on MI.
          RP.recede(*MI);
        }
        llvm::mergeLiveRegSet(DagLive, RP.getLiveRegs());
      }
      // Remove live which not avail in SU.
      GCNRPTracker::LiveRegSet AvailLive = DagAvailRegMap[SU];
      llvm::andLiveRegSet(DagLive, AvailLive);
      DagPressureMap[SU] = DagLive;

      // Add pred to work list.
      for (auto &Pred : SU->Preds) {
        SUnit *PredSU = Pred.getSUnit();
        PredSU->NumSuccsLeft--;
        WorkList.insert(PredSU);
      }
    }

    // Skip dead loop
    if (ReadyNodes.empty()) {
      printf("dead loop when build dag pressure");
      break;
    }
  }
}

// dump functions.

std::string ExpDag::getGraphNodeLabel(const SUnit *SU) const {
  std::string S;
  raw_string_ostream OSS(S);
  auto It = SUnitInputMap.find(SU);
  if (It != SUnitInputMap.end()) {
    OSS << "<input:" << llvm::printReg(It->second) << ">";
  } else {
    SU->getInstr()->print(OSS, /*SkipOpers=*/true);
  }

  return OSS.str();
}

/// Return the label.
std::string ExpDag::getDAGName() const { return "dag.exp"; }

/// viewGraph - Pop up a ghostview window with the reachable parts of the DAG
/// rendered using 'dot'.
///
void ExpDag::viewGraph(const Twine &Name, const Twine &Title) const {
  // This code is only for debugging!
#ifndef NDEBUG
  ViewGraph(const_cast<ExpDag *>(this), Name, false, Title);
#else
  errs() << "BlockExpDag::viewGraph is only available in debug builds on "
         << "systems with Graphviz or gv!\n";
#endif // NDEBUG
}

void ExpDag::dump() {
  viewGraph(getDAGName(), "Exp Dag Graph for " + getDAGName());
}

} // namespace llvm

// Expression Dag dump.
namespace llvm {

static DenseSet<const SUnit *> ViewNodes;

template <>
struct DOTGraphTraits<llvm::ExpDag *> : public DefaultDOTGraphTraits {

  DOTGraphTraits(bool IsSimple = false) : DefaultDOTGraphTraits(IsSimple) {}

  static std::string getGraphName(const llvm::ExpDag *) {
    return "ExpDag graph";
  }

  static bool renderGraphFromBottomUp() { return true; }

  static bool isNodeHidden(const SUnit *Node, const llvm::ExpDag *) {
    if (ViewNodes.empty())
      return false;

    return ViewNodes.count(Node) == 0;
  }

  static std::string getNodeIdentifierLabel(const SUnit *Node,
                                            const llvm::ExpDag *) {
    std::string R;
    raw_string_ostream OS(R);
    OS << static_cast<const void *>(Node);
    return R;
  }

  /// If you want to override the dot attributes printed for a particular
  /// edge, override this method.
  static std::string getEdgeAttributes(const SUnit *, SUnitIterator EI,
                                       const llvm::ExpDag *) {
    if (EI.isArtificialDep())
      return "color=cyan,style=dashed";
    if (EI.isCtrlDep())
      return "color=blue,style=dashed";
    return "";
  }

  static std::string getNodeLabel(const SUnit *SU, const llvm::ExpDag *) {
    std::string Str;
    raw_string_ostream SS(Str);
    SS << "SU:" << SU->NodeNum;
    return SS.str();
  }
  static std::string getNodeDescription(const SUnit *SU,
                                        const llvm::ExpDag *G) {
    return G->getGraphNodeLabel(SU);
  }
  static std::string getNodeAttributes(const SUnit *N, const llvm::ExpDag *) {
    std::string Str("shape=Mrecord");

    Str += ",style=filled,fillcolor=\"#";
    // Use NodeQueueId as SubIdx for ExpDag.
    Str += DOT::getColorString(N->NodeQueueId);
    Str += '"';

    return Str;
  }

  static void addCustomGraphFeatures(llvm::ExpDag *G,
                                     GraphWriter<llvm::ExpDag *> &GW) {
    return G->addCustomGraphFeatures(GW);
  }
};

template <> struct GraphTraits<llvm::ExpDag *> : public GraphTraits<SUnit *> {
  using nodes_iterator = pointer_iterator<std::vector<SUnit>::iterator>;
  static nodes_iterator nodes_begin(llvm::ExpDag *G) {
    return nodes_iterator(G->SUnits.begin());
  }
  static nodes_iterator nodes_end(llvm::ExpDag *G) {
    return nodes_iterator(G->SUnits.end());
  }
};

} // namespace llvm

namespace llvm {
void getRegBound(llvm::MachineBasicBlock *MBB,
                 const llvm::MachineRegisterInfo &MRI,
                 const llvm::SIRegisterInfo *SIRI, const SIInstrInfo *SIII,
                 llvm::LiveIntervals *LIS, unsigned &MaxVGPR,
                 unsigned &MaxSGPR) {
  // TODO: calc real reg bound.
  MaxVGPR = AMDGPU::VGPR255 - AMDGPU::VGPR0;
  MaxSGPR = AMDGPU::SGPR104 - AMDGPU::SGPR0;

  const auto &EndSlot = LIS->getMBBEndIdx(MBB);
  const GCNRPTracker::LiveRegSet OutputLive =
      llvm::getLiveRegs(EndSlot, *LIS, MRI);

  auto *ST =
      &MBB->getParent()
           ->getSubtarget<GCNSubtarget>(); // TODO: Better way to get this.
  if (MBB->empty()) {
    GCNRegPressure MaxPressure = getRegPressure(MRI, OutputLive);
    MaxSGPR = MaxPressure.getSGPRNum();
    MaxVGPR = MaxPressure.getVGPRNum(ST->hasGFX90AInsts());
    return;
  }

  BlockExpDag Dag(MBB, LIS, MRI, SIRI, SIII);
  Dag.build();

  std::vector<SUnit> &SUnits = Dag.SUnits;
  // Remove input nodes.
  for (SUnit &SU : SUnits) {
    if (!SU.isInstr())
      continue;
    std::vector<SDep> InputDeps;
    for (SDep &Dep : SU.Preds) {
      SUnit *Pred = Dep.getSUnit();
      if (Pred->isInstr())
        continue;
      InputDeps.emplace_back(Dep);
    }
    for (SDep &Dep : InputDeps) {
      SU.removePred(Dep);
    }
  }

  const unsigned InputSize = Dag.InputSUnitMap.size();
  const unsigned InstNodeSize = SUnits.size() - InputSize;
  SUnits.erase(SUnits.begin() + InstNodeSize, SUnits.end());

  std::vector<llvm::SUnit *> BotRoots;
  for (SUnit &SU : SUnits) {
    if (SU.NumSuccsLeft == 0)
      BotRoots.emplace_back(&SU);
  }

  auto SchedResult = hrbSched(SUnits, BotRoots, MRI, SIRI);

  GCNUpwardRPTracker RPTracker(*LIS);
  RPTracker.reset(MBB->front(), &OutputLive, /*After*/ true);
  for (auto It = SchedResult.rbegin(); It != SchedResult.rend(); It++) {
    const SUnit *SU = *It;
    if (!SU->isInstr())
      continue;
    MachineInstr *MI = SU->getInstr();
    RPTracker.recede(*MI);
  }

  GCNRegPressure MaxPressure = RPTracker.getMaxPressureAndReset();
  MaxSGPR = MaxPressure.getSGPRNum();
  MaxVGPR = MaxPressure.getVGPRNum(ST->hasGFX90AInsts());
}
} // namespace llvm

// HRB
namespace {

std::vector<SUnit *> buildWorkList(std::vector<llvm::SUnit> &SUnits) {
  std::vector<SUnit *> ResultList;
  ResultList.reserve(SUnits.size());
  for (SUnit &SU : SUnits) {
    ResultList.emplace_back(&SU);
  }
  return ResultList;
}

void sortByHeight(std::vector<SUnit *> &WorkList) {
  std::sort(WorkList.begin(), WorkList.end(),
            [](const SUnit *A, const SUnit *B) {
              // Lowest height first.
              if (A->getHeight() < B->getHeight())
                return true;
              // If height the same, NodeNum big first.
              if (A->getHeight() == B->getHeight())
                return A->NodeNum > B->NodeNum;
              return false;
            });
}

void sortByInChain(std::vector<SUnit *> &WorkList, DenseSet<SUnit *> &Chained) {
  // In chain nodes at end.
  std::sort(WorkList.begin(), WorkList.end(),
            [&Chained](const SUnit *A, const SUnit *B) {
              return Chained.count(A) < Chained.count(B);
            });
}

const TargetRegisterClass *getRegClass(SUnit *SU,
                                       const MachineRegisterInfo &MRI,
                                       const SIRegisterInfo *SIRI) {
  if (!SU->isInstr())
    return nullptr;
  MachineInstr *MI = SU->getInstr();
  if (MI->getNumDefs() == 0)
    return nullptr;

  // For MI has more than one dst, always use first dst.
  MachineOperand *MO = MI->defs().begin();
  if (!MO->isReg())
    return nullptr;
  Register Reg = MO->getReg();
  return SIRI->getRegClassForReg(MRI, Reg);
}

unsigned getVGPRSize(const TargetRegisterClass *RC,
                     const SIRegisterInfo *SIRI) {
  if (!RC)
    return 0;
  if (SIRI->isSGPRClass(RC))
    return 0;
  return RC->getLaneMask().getNumLanes();
}
unsigned getSGPRSize(const TargetRegisterClass *RC,
                     const SIRegisterInfo *SIRI) {
  if (!RC)
    return 0;
  if (!SIRI->isSGPRClass(RC))
    return 0;
  return RC->getLaneMask().getNumLanes();
}

} // namespace

namespace llvm {

void HRB::Lineage::addNode(llvm::SUnit *SU) { Nodes.emplace_back(SU); }
unsigned HRB::Lineage::getSize() const {
  return RC ? RC->getLaneMask().getNumLanes() : 0;
}
unsigned HRB::Lineage::length() const { return Nodes.size(); }

SUnit *HRB::Lineage::getHead() const { return Nodes.front(); }
SUnit *HRB::Lineage::getTail() const { return Nodes.back(); }

void HRB::buildLinear(std::vector<llvm::SUnit> &SUnits) {
  // Working list from TopRoots.
  std::vector<SUnit *> WorkList = buildWorkList(SUnits);
  IntEqClasses EqClasses(SUnits.size());

  while (!WorkList.empty()) {
    sortByHeight(WorkList);
    // Highest SU.
    SUnit *SU = WorkList.back();
    WorkList.pop_back();
    if (!SU->isInstr())
      continue;
    if (ChainedNodes.count(SU) > 0)
      continue;
    IsRecomputeHeight = false;
    Lineage Lineage = buildChain(SU, SUnits);

    // Remove chained nodes from worklist.
    sortByInChain(WorkList, ChainedNodes);
    while (!WorkList.empty()) {
      SUnit *Back = WorkList.back();
      if (ChainedNodes.count(Back))
        WorkList.pop_back();
      else
        break;
    }

    Lineages.emplace_back(Lineage);

    if (IsRecomputeHeight) {
      // Update height from tail.
      SUnit *Tail = Lineage.Nodes.back();
      Tail->setDepthDirty();
      Tail->getHeight();
    }
  }

  DenseSet<SUnit *> TailSet;
  for (Lineage &L : Lineages) {
    if (L.Nodes.size() < 2)
      continue;
    auto It = L.Nodes.rbegin();
    It++;
    SUnit *Tail = L.Nodes.back();
    // If already as tail for other Lineage, start from next.
    if (TailSet.count(Tail) > 0) {
      Tail = *It;
      It++;
    } else {
      TailSet.insert(Tail);
    }
    for (; It != L.Nodes.rend(); It++) {
      SUnit *SU = *It;
      if (Tail->NodeNum == (unsigned)-1)
        continue;
      EqClasses.join(SU->NodeNum, Tail->NodeNum);
    }
  }

  EqClasses.compress();
  // TODO: assign sub class to node.
  for (Lineage &L : Lineages) {
    for (SUnit *SU : L.Nodes) {
      if (SU->NodeNum == (unsigned)-1)
        continue;
      unsigned SubIdx = EqClasses[SU->NodeNum];
      //// Pack subidx.
      // if (EqClasses.count(SubIdx) == 0)
      //  EqClasses[SubIdx] = EqClasses.size();
      SubIdx = EqClasses[SubIdx];
      // Use NodeQueueId as SubIdx. We don't do schedule on ExpDag.
      SU->NodeQueueId = SubIdx;
    }
  }

  LLVM_DEBUG(
      dbgs() << "Chained Nodes:"; for (SUnit *SU
                                       : ChainedNodes) {
        dbgs() << " " << SU->NodeNum << "\n";
      } for (unsigned i = 0; i < Lineages.size(); i++) {
        dbgs() << "Lineage" << i << ":";
        Lineage &L = Lineages[i];
        for (SUnit *SU : L.Nodes) {
          dbgs() << " " << SU->NodeNum;
        }
        dbgs() << "\n";
      });
}

SUnit *HRB::findHeir(SUnit *SU, std::vector<llvm::SUnit> &SUnits) {
  std::vector<SUnit *> Candidates;
  for (SDep &Dep : SU->Succs) {
    // Only check data dep.
    if (Dep.getKind() != SDep::Data)
      continue;

    SUnit *Succ = Dep.getSUnit();
    Candidates.emplace_back(Succ);
  }

  if (Candidates.empty())
    return nullptr;

  if (Candidates.size() == 1)
    return Candidates.front();

  sortByHeight(Candidates);
  // Lowest height.
  SUnit *Heir = Candidates.front();
  SmallVector<SUnit *, 2> SameHeightCandidate;
  for (SUnit *SU : Candidates) {
    if (Heir->getHeight() != SU->getHeight())
      break;
    SameHeightCandidate.emplace_back(SU);
  }
  // Make sure choose lowest dependence between SameHeightCandidate.
  if (SameHeightCandidate.size() > 1) {
    for (size_t i = 1; i < SameHeightCandidate.size(); i++) {
      SUnit *SU = SameHeightCandidate[i];
      // If Heir is pred of SU, use SU.
      if (canReach(SU, Heir))
        Heir = SU;
    }
  }

  unsigned HeriHeight = Heir->getHeight();

  // if lowest node is in ChainedNodes, try to find same height nodes?

  for (SDep &Dep : SU->Succs) {
    // Only check data dep.
    if (Dep.getKind() != SDep::Data)
      continue;
    SUnit *Succ = Dep.getSUnit();
    if (Succ == Heir)
      continue;
    // Avoid cycle in DAG.
    if (canReach(Heir, Succ))
      return nullptr;
    // Make sure Succ is before Heir.
    Heir->addPred(SDep(Succ, SDep::Artificial));
    updateReachForEdge(Succ, Heir, SUnits);
    LLVM_DEBUG(dbgs() << "add edge from " << Succ->NodeNum << "("
                      << Succ->getHeight() << ") to " << Heir->NodeNum << "("
                      << HeriHeight << ")\n");
    // Update height if need.
    unsigned Height = Succ->getHeight();
    if (Height <= HeriHeight) {
      IsRecomputeHeight = true;
    }
  }
  return Heir;
}

HRB::Lineage HRB::buildChain(SUnit *Node, std::vector<llvm::SUnit> &SUnits) {
  HRB::Lineage Chain;
  Chain.addNode(Node);
  ChainedNodes.insert(Node);
  LLVM_DEBUG(dbgs() << "start chain " << Node->NodeNum << "("
                    << Node->getHeight() << ")\n");
  while (Node->NumSuccsLeft > 0) {
    SUnit *Heir = findHeir(Node, SUnits);
    if (!Heir)
      break;
    Chain.addNode(Heir);

    LLVM_DEBUG(dbgs() << "add node to chain " << Heir->NodeNum << "\n");
    if (ChainedNodes.count(Heir) > 0)
      break;
    ChainedNodes.insert(Heir);

    Node = Heir;
  }
  // Find biggest vgpr RC for the chain.
  // TODO: Build conflict and allocate on each edge of the chain.
  const TargetRegisterClass *RC = nullptr;
  unsigned MaxRCSize = 0;
  for (SUnit *SU : Chain.Nodes) {
    const TargetRegisterClass *SuRC = getRegClass(SU, MRI, SIRI);
    unsigned RCSize = getVGPRSize(SuRC, SIRI);
    if (RCSize > MaxRCSize) {
      MaxRCSize = RCSize;
      RC = SuRC;
    }
  }
  if (!RC) {
    // TODO: Find biggest sgpr RC.
    unsigned MaxRCSize = 0;
    for (SUnit *SU : Chain.Nodes) {
      const TargetRegisterClass *SuRC = getRegClass(SU, MRI, SIRI);
      unsigned RCSize = getSGPRSize(SuRC, SIRI);
      if (RCSize > MaxRCSize) {
        MaxRCSize = RCSize;
        RC = SuRC;
      }
    }
  }
  Chain.RC = RC;
  return Chain;
}

void HRB::buildConflict() {

  for (unsigned i = 0; i < Lineages.size(); i++) {
    Lineage &A = Lineages[i];
    for (unsigned j = i + 1; j < Lineages.size(); j++) {
      Lineage &B = Lineages[j];
      if (isConflict(A, B)) {
        Color.Conflicts[i].insert(j);
        Color.Conflicts[j].insert(i);
        LLVM_DEBUG(dbgs() << i << " conflict" << j << "\n");
      }
    }
    // SelfConflict.
    Color.Conflicts[i].insert(i);
  }
}

bool HRB::canReach(llvm::SUnit *A, llvm::SUnit *B) {
  auto It = ReachMap.find(A);
  // If no reach info, treat as reach.
  if (It == ReachMap.end())
    return true;
  DenseSet<SUnit *> &CurReach = It->second;
  return CurReach.find(B) != CurReach.end();
}

void HRB::updateReachForEdge(llvm::SUnit *A, llvm::SUnit *B,
                             std::vector<llvm::SUnit> &SUnits) {
  DenseSet<SUnit *> &ReachA = ReachMap[A];
  ReachA.insert(B);
  DenseSet<SUnit *> &ReachB = ReachMap[B];
  ReachA.insert(ReachB.begin(), ReachB.end());

  for (SUnit &SU : SUnits) {
    if (!canReach(&SU, A))
      continue;

    DenseSet<SUnit *> &CurReach = ReachMap[&SU];
    CurReach.insert(ReachA.begin(), ReachA.end());
  }
}

void HRB::buildReachRelation(ArrayRef<SUnit *> BotRoots) {
  // Add fake entry to do PostOrder traversal.
  // SUnit using Pred to traversal, so need to Revrese post order.
  SUnit FakeEntry;
  SmallVector<SDep, 4> FakeDeps;
  for (SUnit *Root : BotRoots) {
    SDep Dep = SDep(Root, SDep::Artificial);
    FakeEntry.addPred(Dep);
    FakeDeps.emplace_back(Dep);
  }

  ReversePostOrderTraversal<SUnit *> RPOT(&FakeEntry);
  for (SUnit *SU : RPOT) {
    // Create Reach Set first.
    ReachMap[SU].clear();
  }
  for (SUnit *SU : RPOT) {
    DenseSet<SUnit *> &CurReach = ReachMap[SU];
    // All Preds can reach SU and SU's reach.
    for (SDep &Dep : SU->Preds) {
      // Igonre week dep.
      if (Dep.isWeak())
        continue;
      DenseSet<SUnit *> &PrevReach = ReachMap[Dep.getSUnit()];
      PrevReach.insert(SU);
      PrevReach.insert(CurReach.begin(), CurReach.end());
    }
    assert(CurReach.count(SU) == 0 && "dead loop");
  }
  // Remove fake entry.
  for (SDep &Dep : FakeDeps) {
    FakeEntry.removePred(Dep);
  }
  ReachMap.erase(&FakeEntry);

  LLVM_DEBUG(for (Lineage &L
                  : Lineages) {
    for (SUnit *SU : L.Nodes) {
      DenseSet<SUnit *> &CurReach = ReachMap[SU];
      dbgs() << SU->NodeNum << " reach: ";
      for (SUnit *R : CurReach) {
        dbgs() << R->NodeNum << " ";
      }
      dbgs() << "\n";
    }
  });
}

bool HRB::isConflict(const Lineage &A, const Lineage &B) {
  // Make conflict between sgpr and vgpr to help group lineages when share
  // colors. Keep the conflict will group lineages in avoid mix use color in
  // different sub exp.
  SUnit *Head0 = A.getHead();
  SUnit *Tail0 = A.getTail();
  SUnit *Head1 = B.getHead();
  SUnit *Tail1 = B.getTail();
  DenseSet<SUnit *> &Reach0 = ReachMap[Head0];
  DenseSet<SUnit *> &Reach1 = ReachMap[Head1];
  bool R01 = Reach0.count(Tail1) != 0;
  bool R10 = Reach1.count(Tail0) != 0;
  return R01 && R10;
}
bool HRB::canFuse(const Lineage &A, const Lineage &B) {
  if (A.RC != B.RC) {
    // no RC will not conflict with other nodes.
    if (!A.RC)
      return false;
    if (!B.RC)
      return false;
    // SGRP and VGPR not conflict.
    if (SIRI->isSGPRClass(A.RC) != SIRI->isSGPRClass(B.RC))
      return false;
  }
  // Can Fuse if a.head reach b.tail but b.head not reach a.tail and vice versa.
  SUnit *Head0 = A.getHead();
  SUnit *Tail0 = A.getTail();
  SUnit *Head1 = B.getHead();
  SUnit *Tail1 = B.getTail();
  DenseSet<SUnit *> &Reach0 = ReachMap[Head0];
  DenseSet<SUnit *> &Reach1 = ReachMap[Head1];
  bool R01 = Reach0.count(Tail1) != 0;
  bool R10 = Reach1.count(Tail0) != 0;
  return R01 != R10;
}

bool HRB::tryFuse(Lineage &A, Lineage &B, std::vector<llvm::SUnit> &SUnits) {

  // Can Fuse if a.head reach b.tail but b.head not reach a.tail and vice versa.
  SUnit *Head0 = A.getHead();
  SUnit *Tail0 = A.getTail();
  SUnit *Head1 = B.getHead();
  SUnit *Tail1 = B.getTail();
  DenseSet<SUnit *> &Reach0 = ReachMap[Head0];
  DenseSet<SUnit *> &Reach1 = ReachMap[Head1];
  bool R01 = Reach0.count(Tail1) != 0;
  bool R10 = Reach1.count(Tail0) != 0;
  if (R01 == R10)
    return false;
  Lineage *NewHead = &A;
  Lineage *NewTail = &B;
  if (R01) {
    // a reach b, b cannot reach a.
    // link a.tail->b.head.
    NewHead = &A;
    NewTail = &B;
  } else {
    // b reach a, a cannot reach b.
    // link b.tail->a.head.
    NewHead = &B;
    NewTail = &A;
  }

  // Merge reg class.
  const TargetRegisterClass *RC0 = NewHead->RC;
  const TargetRegisterClass *RC1 = NewTail->RC;
  unsigned RC0Size = getVGPRSize(RC0, SIRI);
  unsigned RC1Size = getVGPRSize(RC1, SIRI);
  if (RC1Size > RC0Size)
    NewHead->RC = RC1;
  // Merge chain.
  SUnit *FuseTail = NewHead->getTail();
  SUnit *FuseHead = NewTail->getHead();
  assert(ReachMap[FuseHead].count(FuseTail) == 0 && "");
  FuseHead->addPred(SDep(FuseTail, SDep::Artificial));
  LLVM_DEBUG(dbgs() << "fuse " << FuseTail->NodeNum << "->" << FuseHead->NodeNum
                    << "\n");
  // Update reach map.
  updateReachForEdge(FuseTail, FuseHead, SUnits);
  // Merge Nodes.
  NewHead->Nodes.append(NewTail->Nodes.begin(), NewTail->Nodes.end());
  // Clear newTail.
  NewTail->Nodes.clear();
  NewTail->RC = nullptr;
  return true;
}

void HRB::fusionLineages(std::vector<llvm::SUnit> &SUnits) {
  if (Lineages.empty())
    return;
  bool IsUpdated = true;
  while (IsUpdated) {
    IsUpdated = false;
    int Size = Lineages.size();
    for (int i = 0; i < Size; i++) {
      Lineage &A = Lineages[i];
      if (A.length() == 0)
        continue;

      for (int j = i + 1; j < Size; j++) {
        Lineage &B = Lineages[j];
        if (B.length() == 0)
          continue;
        if (tryFuse(A, B, SUnits)) {
          IsUpdated = true;
          if (A.length() == 0)
            break;
        }
      }
    }
    // Remove empty lineages.
    std::sort(Lineages.begin(), Lineages.end(),
              [](const Lineage &A, const Lineage &B) {
                return A.length() > B.length();
              });
    while (Lineages.back().length() == 0) {
      Lineages.pop_back();
    }
  }
  // Set ID after fusion.
  unsigned ID = 0;
  for (Lineage &L : Lineages) {
    L.ID = ID++;
  }
}

unsigned HRB::colorLineages(std::vector<Lineage *> &InLineages,
                            DenseMap<Lineage *, unsigned> &AllocMap,
                            const unsigned Limit) {
  // allocate long Lineage first. How about size of RC?
  std::sort(InLineages.begin(), InLineages.end(),
            [](const Lineage *a, const Lineage *b) {
              // Make sure root allocate first.
              return a->length() > b->length();
            });

  unsigned MaxColor = 0;
  const unsigned VGPR_LIMIT = 256 * 4;

  for (Lineage *L : InLineages) {
    unsigned ID = L->ID;
    auto &Conflict = Color.Conflicts[ID];
    std::bitset<VGPR_LIMIT> Colors;
    for (unsigned j : Conflict) {
      Lineage *LineageC = &Lineages[j];
      if (AllocMap.count(LineageC) == 0)
        continue;
      unsigned C = AllocMap[LineageC];
      unsigned S = LineageC->getSize();
      for (unsigned i = 0; i < S; i++) {
        unsigned Pos = C + i;
        Colors.set(Pos);
      }
    }

    unsigned Color = Limit;
    unsigned Size = L->getSize();
    for (unsigned i = 0; i < Limit - Size;) {
      unsigned OldI = i;
      for (unsigned j = 0; j < Size; j++) {
        unsigned Pos = i + Size - 1 - j;
        if (Colors.test(Pos)) {
          i = Pos + 1;
          break;
        }
      }

      if (i != OldI)
        continue;
      Color = i;
      break;
    }

    AllocMap[L] = Color;
    Color += Size;
    if (Color > MaxColor)
      MaxColor = Color;
  }
  return MaxColor;
}

void HRB::ColorResult::colorSU(SUnit *SU, unsigned Color) {
  ColorMap[SU] = Color;
}

unsigned HRB::ColorResult::getLineage(SUnit *SU) const {
  return LineageMap.find(SU)->second;
}

bool HRB::ColorResult::isConflict(const SUnit *SU0, unsigned Lineage) const {
  const unsigned L = LineageMap.find(SU0)->second;
  const auto &Conflict = Conflicts.find(L)->second;
  return Conflict.count(Lineage) > 0;
}

bool HRB::ColorResult::isHead(SUnit *SU) const { return HeadSet.count(SU); }
bool HRB::ColorResult::isTail(SUnit *SU) const { return TailSet.count(SU); }

const SUnit *HRB::ColorResult::getTail(SUnit *SU) const {
  if (!isHead(SU))
    return nullptr;
  auto It = HeadTailMap.find(SU);
  return It->second;
}

unsigned HRB::ColorResult::getColor(const llvm::SUnit *SU) const {
  auto It = ColorMap.find(SU);
  return It->second;
}

unsigned HRB::ColorResult::getSize(const llvm::SUnit *SU) const {
  auto It = SizeMap.find(SU);
  return It->second;
}

HRB::ColorResult &HRB::coloring() {
  // Collect VGPR lineages.
  std::vector<Lineage *> VgprLineages;
  for (Lineage &L : Lineages) {
    const auto *RC = L.RC;
    if (!RC)
      continue;
    if (SIRI->isSGPRClass(RC))
      continue;
    VgprLineages.emplace_back(&L);
  }

  const unsigned VGPR_LIMIT = 256 * 4;
  DenseMap<Lineage *, unsigned> VAllocMap;
  const unsigned MaxVGPR = colorLineages(VgprLineages, VAllocMap, VGPR_LIMIT);

  // Collect SGPR lineages.
  std::vector<Lineage *> SgprLineages;
  for (Lineage &L : Lineages) {
    const auto *RC = L.RC;
    if (!RC)
      continue;
    if (!SIRI->isSGPRClass(RC))
      continue;
    SgprLineages.emplace_back(&L);
  }

  const unsigned SGPR_LIMIT = 104;
  DenseMap<Lineage *, unsigned> SAllocMap;
  const unsigned MaxSGPR = colorLineages(SgprLineages, SAllocMap, SGPR_LIMIT);
  // +1 for each type of lineages(SGPR, VGPR, no reg).
  const unsigned MaxReg = MaxSGPR + 1 + MaxVGPR + 1 + 1;
  const unsigned SgprBase = MaxVGPR + 1;

  for (Lineage &L : Lineages) {
    // Collect HeadSet.
    Color.HeadSet.insert(L.getHead());
    Color.TailSet.insert(L.getTail());
    Color.HeadTailMap[L.getHead()] = L.getTail();
    // Save color.
    const auto *RC = L.RC;
    // All no reg lineage goes to maxReg.
    unsigned RegColor = MaxReg;
    if (!RC) {
    } else if (SIRI->isSGPRClass(RC)) {
      RegColor = SAllocMap[&L] + SgprBase;
    } else {
      RegColor = VAllocMap[&L];
    }
    unsigned Size = L.getSize();
    for (SUnit *SU : L.Nodes) {
      Color.colorSU(SU, RegColor);
      Color.SizeMap[SU] = Size;
      Color.LineageMap[SU] = L.ID;
    }
  }
  Color.MaxReg = MaxReg;
  Color.MaxSGPR = MaxSGPR;
  Color.MaxVGPR = MaxVGPR;

  for (unsigned i = 0; i < Lineages.size(); i++) {
    Lineage &A = Lineages[i];
    SUnit *HeadA = A.getHead();
    unsigned ColorA = Color.getColor(HeadA);
    unsigned SizeA = Color.getSize(HeadA);
    for (unsigned j = i + 1; j < Lineages.size(); j++) {
      Lineage &B = Lineages[j];

      SUnit *HeadB = B.getHead();
      unsigned ColorB = Color.getColor(HeadB);
      unsigned SizeB = Color.getSize(HeadB);

      if (ColorB >= (ColorA + SizeA))
        continue;
      if (ColorA >= (ColorB + SizeB))
        continue;
      Color.ShareColorLineages.insert(i);
      Color.ShareColorLineages.insert(j);
    }
  }

  return Color;
}

void HRB::dump() {
  for (unsigned i = 0; i < Lineages.size(); i++) {
    dbgs() << "Lineage" << i << ":";
    Lineage &L = Lineages[i];
    for (SUnit *SU : L.Nodes) {
      dbgs() << " " << SU->NodeNum;
    }
    dbgs() << "\n";
    if (!Color.ColorMap.empty()) {
      dbgs() << "color:" << Color.getColor(L.getHead())
             << " size: " << Color.getSize(L.getHead()) << "\n";
    }
    if (!ReachMap.empty()) {
      dbgs() << "conflict:";
      for (unsigned j = 0; j < Lineages.size(); j++) {
        if (i == j)
          continue;
        if (isConflict(L, Lineages[j])) {
          dbgs() << " " << j;
        }
      }
      dbgs() << "\n";
    }
  }
}

void HRB::dumpReachMap() {
  if (!ReachMap.empty()) {
    dbgs() << "reachMap:";
    for (auto It : ReachMap) {
      SUnit *SU = It.first;
      auto &Reach = It.second;
      if (SU->isInstr()) {
        MachineInstr *MI = SU->getInstr();
        MI->print(dbgs());
      }
      dbgs() << SU->NodeNum << "can reach :\n";
      for (SUnit *R : Reach) {
        dbgs() << R->NodeNum << " ";
      }
      dbgs() << "\n";
    }
    dbgs() << "\n";
  }
}

// schedule base on HRB lineages and color result.

std::vector<const SUnit *> hrbSched(std::vector<SUnit> &SUnits,
                                    std::vector<SUnit *> &BRoots,
                                    const llvm::MachineRegisterInfo &MRI,
                                    const llvm::SIRegisterInfo *SIRI) {
  HRB Hrb(MRI, SIRI);
  // build reach info to avoid dead loop when build linear.
  Hrb.buildReachRelation(BRoots);
  Hrb.buildLinear(SUnits);

  std::sort(BRoots.begin(), BRoots.end(), [](const SUnit *A, const SUnit *B) {
    return A->NumSuccsLeft < B->NumSuccsLeft;
  });
  while (!BRoots.empty() && BRoots.back()->NumSuccsLeft > 0) {
    BRoots.pop_back();
  }

  Hrb.buildReachRelation(BRoots);
  Hrb.fusionLineages(SUnits);
  Hrb.buildConflict();
  const HRB::ColorResult &ColorRes = Hrb.coloring();

  LLVM_DEBUG(Hrb.dump());

  // All lineage head which don't has Pred is TopRoots.
  // Put top roots in worklist.
  // while worklist not empty.
  //    if not head or color avail
  //        is candidate.
  //    choose best candidate by height.
  //    update worklist.
  std::vector<SUnit *> ReadyList;
  for (SUnit &SU : SUnits) {
    if (SU.NumPredsLeft == 0)
      ReadyList.emplace_back(&SU); //.insert(&SU);
  }
  // When there're more than one sub exp in the DAG, make sure not mix different
  // sub exp or it will dead loop for color goes different subexp.

  std::bitset<512 * 2> Colors;
  auto IsColorAvail = [&Colors](unsigned Color, unsigned Size) -> bool {
    for (unsigned i = 0; i < Size; i++) {
      unsigned Pos = Color + i;
      if (Colors.test(Pos))
        return false;
    }
    return true;
  };
  auto AllocColor = [&Colors](unsigned Color, unsigned Size) {
    for (unsigned i = 0; i < Size; i++) {
      unsigned Pos = Color + i;
      assert(!Colors.test(Pos) && "color already allocated");
      LLVM_DEBUG(dbgs() << Pos << "is allocated\n");
      Colors.set(Pos);
    }
  };

  auto FreeColor = [&Colors](unsigned Color, unsigned Size) {
    for (unsigned i = 0; i < Size; i++) {
      unsigned Pos = Color + i;
      assert(Colors.test(Pos) && "color has not been allocated");
      LLVM_DEBUG(dbgs() << Pos << "is free\n");
      Colors.reset(Pos);
    }
  };

  // Save color and size for tail to support case two lineage share tail.
  // When finish a tail, free color for working lineage which end with tail.
  DenseMap<const SUnit *,
           SmallVector<std::tuple<unsigned, unsigned, unsigned>, 2>>
      TailMap;

  // For lineages share same color, need to choose correct order.
  // If l0 has color 0, l1 has color 1, l2 has color 0, l3 has color 1.
  // l0 and l3 conflict, l1 and l2 conflict.
  // l0 and l3 must sched together.
  // If sched l0 and l1, it may dead lock for l0 wait something in l3 and l1
  // wait something in l2.
  // ShareColorLineages will mark lineages which share color with other
  // lineages. When sched, choose new lineages which has more conflict with
  // ShareColorLineages.
  const DenseSet<unsigned> &ShareColorLineages = ColorRes.ShareColorLineages;

  std::vector<const SUnit *> Schedule;
  DenseSet<unsigned> UnfinishedLineages;
  while (!ReadyList.empty()) {
    // Make sure node conflict with predLineage first.
    std::sort(ReadyList.begin(), ReadyList.end(),
              [&UnfinishedLineages, &ColorRes](const SUnit *A, const SUnit *B) {
                unsigned ConfA = 0;
                for (unsigned L : UnfinishedLineages) {
                  if (ColorRes.isConflict(A, L))
                    ConfA++;
                }
                unsigned ConfB = 0;
                for (unsigned L : UnfinishedLineages) {
                  if (ColorRes.isConflict(B, L))
                    ConfB++;
                }
                return ConfA > ConfB;
              });

    LLVM_DEBUG(dbgs() << "ReadyList:\n"; for (SUnit *SU
                                              : ReadyList) {
      dbgs() << " " << SU->NodeNum;
    } dbgs() << "\n";);
    SUnit *Candidate = nullptr;
    for (auto It = ReadyList.begin(); It != ReadyList.end(); It++) {
      SUnit *SU = *It;
      unsigned Color = ColorRes.getColor(SU);
      unsigned Size = ColorRes.getSize(SU);
      // If SU is not head or color is available, SU is the candidate.
      if (ColorRes.isHead(SU)) {
        if (!IsColorAvail(Color, Size))
          continue;
        // alloc color.
        AllocColor(Color, Size);
        // save tail color.
        const SUnit *Tail = ColorRes.getTail(SU);
        unsigned ID = ColorRes.getLineage(SU);
        SmallVector<std::tuple<unsigned, unsigned, unsigned>, 2> &TailColors =
            TailMap[Tail];
        TailColors.emplace_back(std::make_tuple(Color, Size, ID));
        if (ShareColorLineages.count(ID))
          UnfinishedLineages.insert(ID);
      }

      // free color for working lineage which end with SU.
      if (ColorRes.isTail(SU)) {
        auto &TailColors = TailMap[SU];
        for (auto &TailTuple : TailColors) {
          unsigned LineageColor, LineageSize, ID;
          std::tie(LineageColor, LineageSize, ID) = TailTuple;
          FreeColor(LineageColor, LineageSize);
          if (ShareColorLineages.count(ID))
            UnfinishedLineages.insert(ID);
        }
        // Clear the tail.
        TailMap.erase(SU);
      }

      Candidate = SU;
      // Remove Candidate from ReadyList.
      ReadyList.erase(It);
      break;
    }

    if (!Candidate) {
      // In case failed to find candidate, start a lineage if there is one.
      for (auto It = ReadyList.begin(); It != ReadyList.end(); It++) {
        SUnit *SU = *It;

        if (!ColorRes.isHead(SU)) {
          continue;
        }
        Candidate = SU;
        // Remove Candidate from ReadyList.
        ReadyList.erase(It);
        break;
      }
    }
    assert(Candidate && "fail to find a Candidate");
    LLVM_DEBUG(dbgs() << "Sched " << Candidate->NodeNum << "\n");

    // Add all Candidate succ which is Ready.
    for (SDep &Dep : Candidate->Succs) {
      if (Dep.isWeak())
        continue;
      SUnit *Succ = Dep.getSUnit();

      if (Succ->NumPredsLeft > 0)
        Succ->NumPredsLeft--;
      LLVM_DEBUG(dbgs() << "Succ " << Succ->NodeNum << " has "
                        << Succ->NumPredsLeft << " preds\n");
      if (Succ->NumPredsLeft == 0)
        ReadyList.emplace_back(Succ);
    }

    // Sched Candidate.
    assert(Candidate->isInstr() && "Candidate must be instr Node");
    Schedule.emplace_back(Candidate);
  }
  assert(Schedule.size() == SUnits.size() && "SUnit size should match");
  return Schedule;
}

} // namespace llvm
