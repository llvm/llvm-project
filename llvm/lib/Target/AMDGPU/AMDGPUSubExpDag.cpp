#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/SlotIndexes.h"

// #include "dxc/DXIL/DxilMetadataHelper.h"
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
  for (auto &input : inputLive) {
    pressure::print_reg(input.first, MRI, SIRI, llvm::dbgs());
    dbgs() << "\n";
  }
  dbgs() << "output regs:\n";
  for (auto &output : outputLive) {
    pressure::print_reg(output.first, MRI, SIRI, llvm::dbgs());
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

void SubExp::calcMaxPressure(const MachineRegisterInfo &MRI,
                             const SIRegisterInfo *SIRI) {
  sMaxSize = std::max(sInputSize, sOutputSize);
  vMaxSize = std::max(vInputSize, vOutputSize);

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
      LaneBitmask mask = getRegMask(MO, MRI);
      auto it = LiveRegs.find(Reg);
      if (it != LiveRegs.end()) {
        LiveRegs[Reg] = mask | it->second;
      } else {
        LiveRegs[Reg] = mask;
      }
    }
  }

  for (auto it : LiveRegs) {
    LaneBitmask emptyMask;
    CurPressure.inc(it.first, emptyMask, it.second, MRI);
  }

  for (auto it = SUnits.rbegin(); it != SUnits.rend(); it++) {
    MachineInstr *MI = *it;
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
      auto liveIt = LiveRegs.find(Reg);
      if (liveIt != LiveRegs.end()) {
        PrevMask = liveIt->second;
      }

      if (MO.isDef()) {
        LiveMask = PrevMask & (~(LiveMask));
      } else {
        LiveMask = PrevMask | LiveMask;
      }

      CurPressure.inc(Reg, PrevMask, LiveMask, MRI);
      LiveRegs[Reg] = LiveMask;
    }

    unsigned sSize = CurPressure.getSGPRNum();
    unsigned vSize = CurPressure.getVGPRNum(ST->hasGFX90AInsts());
    if (sSize > sMaxSize)
      sMaxSize = sSize;
    if (vSize > vMaxSize)
      vMaxSize = vSize;
  }
}

bool SubExp::isSafeToMove(const MachineRegisterInfo &MRI, bool IsMoveUp) const {
  if (IsMultiDefOutput)
    return false;
  if (IsHasTerminatorInst)
    return false;
  if (IsUseIncomingReg)
    return false;

  // Input should be single def.
  for (unsigned Reg : TopRegs) {
    if (!MRI.hasOneDef(Reg) && !llvm::IsSub0Sub1SingleDef(Reg, MRI))
      return false;
  }
  return true;
}

ExpDag::ExpDag(const llvm::MachineRegisterInfo &MRI,
               const llvm::SIRegisterInfo *SIRI, const SIInstrInfo *SIII,
               const bool IsJoinInput)
    : MRI(MRI), SIRI(SIRI), SIII(SIII), IsJoinInputToSubExp(IsJoinInput) {}

template <typename T>
void ExpDag::initNodes(const LiveSet &InputLiveReg, T &insts) {
  unsigned NodeSize = InputLiveReg.size() + insts.size();
  SUnits.reserve(NodeSize);

  for (MachineInstr *MI : insts) {
    if (MI->isDebugInstr())
      continue;
    SUnits.emplace_back(MI, SUnits.size());
    SUnit *SU = &SUnits.back();
    SUnitMIMap[SU] = MI;
    MISUnitMap[MI] = SU;
  }

  for (auto it : InputLiveReg) {
    unsigned Reg = it.first;
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
                   T &insts) {
  initNodes(InputLiveReg, insts);
  addDataDep(SIRI);
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
  std::vector<unsigned> passThruInputs;
  for (SUnit &SU : SUnits) {
    if (SU.NumPredsLeft == 0 && SU.NumSuccsLeft == 0) {
      passThruInputs.emplace_back(SU.NodeNum);
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
  // Not count passThruInputs for subExps since they're exp with only 1 SU.
  // SubExpIndexMap is used to pack SubIdx within updated NumSubExps.
  NumSubExps -= passThruInputs.size();
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
      unsigned count = SubExpIndexMap.size();
      SubExpIndexMap.insert(std::make_pair(SubIdx, count));
    }
    SubIdx = SubExpIndexMap[SubIdx];
    // Use NodeQueueId as SubIdx. We don't do schedule on ExpDag.
    SU.NodeQueueId = SubIdx;

    SubExp &Exp = SubExps[SubIdx];
    auto it = SUnitInputMap.find(&SU);
    if (it != SUnitInputMap.end()) {
      // Input.
      unsigned Reg = it->second;
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
        unsigned Reg = MO.getReg();
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
            auto suIt = MISUnitMap.find(&UserMI);
            // When UserMI is not in dag, treat it as other block.
            if (suIt == MISUnitMap.end()) {
              IsUsedInOtherBlk = true;
              break;
            }
            SUnit *UseSU = suIt->second;
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
      auto it = StartLiveReg.find(Reg);
      assert(it != StartLiveReg.end() &&
             "cannot find input reg in block start live");
      Exp.inputLive[Reg] |= it->second;
    }

    for (unsigned Reg : Exp.BottomRegs) {
      auto it = EndLiveReg.find(Reg);
      if (it == EndLiveReg.end()) {
        //"cannot find output reg in block end live");
        // Bottom reg is killed inside current block, did not get out of the
        // block.
        // Or the bottom reg is not treat as output in this dag, not save to
        // outputLive which will affect profit count.
        continue;
      }
      Exp.outputLive[Reg] |= it->second;
    }

    CollectLiveSetPressure(Exp.inputLive, MRI, SIRI, Exp.vInputSize,
                           Exp.sInputSize);
    CollectLiveSetPressure(Exp.outputLive, MRI, SIRI, Exp.vOutputSize,
                           Exp.sOutputSize);
  }
}

void ExpDag::addDataDep(const SIRegisterInfo *SIRI) {
  DenseMap<unsigned, MachineInstr *> curDefMI;

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

      auto curDefIt = curDefMI.find(Reg);
      // Check def inst first.
      if (curDefIt != curDefMI.end()) {
        MachineInstr *curDef = curDefIt->second;
        DefSU = MISUnitMap[curDef];
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
      unsigned Reg = MO.getReg();

      // For case like:
      // undef %808.sub0:sgpr_64 = COPY killed %795:sgpr_32
      // %808.sub1:sgpr_64 = S_MOV_B32 0
      // When partially write, link MI to previous def.
      if (MO.getSubReg() != 0) {
        SUnit *DefSU = nullptr;
        auto curDefIt = curDefMI.find(Reg);
        // Check def inst first.
        if (curDefIt != curDefMI.end()) {
          MachineInstr *curDef = curDefIt->second;
          DefSU = MISUnitMap[curDef];
          // Add link between different defs.
          SU.addPred(SDep(DefSU, SDep::Data, Reg));
        }
      }

      curDefMI[Reg] = MI;
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

  std::vector<MachineInstr *> insts;
  for (MachineInstr &MI : *MBB) {
    insts.emplace_back(&MI);
  }

  ExpDag::build(StartLiveReg, EndLiveReg, insts);
}

void BlockExpDag::buildWithPressure() {
  auto *SlotIndexes = LIS->getSlotIndexes();
  const auto StartIdx = SlotIndexes->getMBBStartIdx(MBB);
  const auto StartLiveReg = llvm::getLiveRegs(StartIdx, *LIS, MRI);

  const auto EndIdx = SlotIndexes->getMBBEndIdx(MBB);
  const auto EndLiveReg = llvm::getLiveRegs(EndIdx, *LIS, MRI);

  std::vector<MachineInstr *> insts;
  for (MachineInstr &MI : *MBB) {
    insts.emplace_back(&MI);
  }

  ExpDag::build(StartLiveReg, EndLiveReg, insts);
  // Build pressure.
  buildPressure(StartLiveReg, EndLiveReg);
}

void BlockExpDag::buildAvail(const LiveSet &passThruSet,
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
      RP.reset(BeginMI, &passThruSet);
      MachineInstr *MI = SU.getInstr();
      if (MI) {
        RP.reset(*MI, &passThruSet);
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
    bool IsUpdated = false;
    SmallVector<SUnit *, 4> ReadyNodes;
    for (SUnit *SU : WorkList) {
      if (SU->NumPredsLeft > 0)
        continue;
      ReadyNodes.emplace_back(SU);
      // Ready, move it to Processed.
      Processed.insert(SU);
      IsUpdated = true;
      // Only update 1 node once.
      // Order of schedle here should not affect pressure.
      break;
    }

    for (SUnit *SU : ReadyNodes) {
      // Remove SU from worklist.
      WorkList.erase(SU);

      MachineInstr *MI = SU->getInstr();
      // Calc pressure based on pred nodes.
      GCNRPTracker::LiveRegSet dagLive;
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
        llvm::mergeLiveRegSet(dagLive, RP.getLiveRegs());
      }
      DagAvailRegMap[SU] = dagLive;

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
  GCNRPTracker::LiveRegSet passThruSet;
  for (auto Reg : StartLiveReg) {
    unsigned reg = Reg.first;
    auto EndReg = EndLiveReg.find(reg);
    if (EndReg == EndLiveReg.end())
      continue;

    LaneBitmask mask = Reg.second;
    LaneBitmask endMask = EndReg->second;
    mask &= endMask;
    if (mask.getAsInteger() == 0)
      continue;
    passThruSet[reg] = mask;
  }

  // Build avial for each nodes.
  buildAvail(passThruSet, DagAvailRegMap);

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
      RP.reset(BeginMI, &passThruSet, /*After*/ true);
      MachineInstr *MI = SU.getInstr();
      if (MI) {
        RP.reset(*MI, &passThruSet, /*After*/ true);
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
    bool IsUpdated = false;
    SmallVector<SUnit *, 4> ReadyNodes;
    for (SUnit *SU : WorkList) {
      if (SU->NumSuccsLeft > 0)
        continue;
      ReadyNodes.emplace_back(SU);
      // Ready, move it to Processed.
      Processed.insert(SU);
      IsUpdated = true;
      // Only update 1 node once.
      // Order of schedle here should not affect pressure.
      break;
    }

    for (SUnit *SU : ReadyNodes) {
      // Remove SU from worklist.
      WorkList.erase(SU);

      MachineInstr *MI = SU->getInstr();
      // Calc pressure based on succ nodes.
      GCNRPTracker::LiveRegSet dagLive;
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
        llvm::mergeLiveRegSet(dagLive, RP.getLiveRegs());
      }
      // Remove live which not avail in SU.
      GCNRPTracker::LiveRegSet availLive = DagAvailRegMap[SU];
      llvm::andLiveRegSet(dagLive, availLive);
      DagPressureMap[SU] = dagLive;

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
  std::string s;
  raw_string_ostream oss(s);
  auto it = SUnitInputMap.find(SU);
  if (it != SUnitInputMap.end()) {
    oss << "<input:" << llvm::printReg(it->second) << ">";
  } else {
    SU->getInstr()->print(oss, /*SkipOpers=*/true);
  }

  return oss.str();
}

/// Return the label.
std::string ExpDag::getDAGName() const { return "dag.exp"; }

/// viewGraph - Pop up a ghostview window with the reachable parts of the DAG
/// rendered using 'dot'.
///
void ExpDag::viewGraph(const Twine &Name, const Twine &Title) const {
#if 0 // TODO: Re-enable this
  // This code is only for debugging!
#ifndef NDEBUG
  ViewGraph(const_cast<ExpDag *>(this), Name, false, Title);
#else
  errs() << "BlockExpDag::viewGraph is only available in debug builds on "
         << "systems with Graphviz or gv!\n";
#endif // NDEBUG
#endif
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

  DOTGraphTraits(bool isSimple = false) : DefaultDOTGraphTraits(isSimple) {}

  static std::string getGraphName(const llvm::ExpDag *G) {
    return "ExpDag graph";
  }

  static bool renderGraphFromBottomUp() { return true; }

  static bool isNodeHidden(const SUnit *Node) {
    if (ViewNodes.empty())
      return false;

    return ViewNodes.count(Node) == 0;
  }

  static std::string getNodeIdentifierLabel(const SUnit *Node,
                                            const llvm::ExpDag *Graph) {
    std::string R;
    raw_string_ostream OS(R);
    OS << static_cast<const void *>(Node);
    return R;
  }

  /// If you want to override the dot attributes printed for a particular
  /// edge, override this method.
  static std::string getEdgeAttributes(const SUnit *Node, SUnitIterator EI,
                                       const llvm::ExpDag *Graph) {
    if (EI.isArtificialDep())
      return "color=cyan,style=dashed";
    if (EI.isCtrlDep())
      return "color=blue,style=dashed";
    return "";
  }

  static std::string getNodeLabel(const SUnit *SU, const llvm::ExpDag *Graph) {
    std::string Str;
    raw_string_ostream SS(Str);
    SS << "SU:" << SU->NodeNum;
    return SS.str();
  }
  static std::string getNodeDescription(const SUnit *SU,
                                        const llvm::ExpDag *G) {
    return G->getGraphNodeLabel(SU);
  }
  static std::string getNodeAttributes(const SUnit *N,
                                       const llvm::ExpDag *Graph) {
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
  const GCNRPTracker::LiveRegSet outputLive =
      llvm::getLiveRegs(EndSlot, *LIS, MRI);

  auto *ST =
      &MBB->getParent()
           ->getSubtarget<GCNSubtarget>(); // TODO: Better way to get this.
  if (MBB->empty()) {
    GCNRegPressure MaxPressure = getRegPressure(MRI, outputLive);
    MaxSGPR = MaxPressure.getSGPRNum();
    MaxVGPR = MaxPressure.getVGPRNum(ST->hasGFX90AInsts());
    return;
  }

  BlockExpDag dag(MBB, LIS, MRI, SIRI, SIII);
  dag.build();

  std::vector<SUnit> &SUnits = dag.SUnits;
  // Remove input nodes.
  for (SUnit &SU : SUnits) {
    if (!SU.isInstr())
      continue;
    std::vector<SDep> inputDeps;
    for (SDep &Dep : SU.Preds) {
      SUnit *Pred = Dep.getSUnit();
      if (Pred->isInstr())
        continue;
      inputDeps.emplace_back(Dep);
    }
    for (SDep &Dep : inputDeps) {
      SU.removePred(Dep);
    }
  }

  unsigned inputSize = dag.InputSUnitMap.size();
  unsigned instNodeSize = SUnits.size() - inputSize;
  SUnits.erase(SUnits.begin() + instNodeSize, SUnits.end());

  std::vector<llvm::SUnit *> BotRoots;
  for (SUnit &SU : SUnits) {
    if (SU.NumSuccsLeft == 0)
      BotRoots.emplace_back(&SU);
  }

  auto SchedResult = hrbSched(SUnits, BotRoots, MRI, SIRI);

  GCNUpwardRPTracker RPTracker(*LIS);
  RPTracker.reset(MBB->front(), &outputLive, /*After*/ true);
  for (auto it = SchedResult.rbegin(); it != SchedResult.rend(); it++) {
    const SUnit *SU = *it;
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
  std::vector<SUnit *> resultList;
  resultList.reserve(SUnits.size());
  for (SUnit &SU : SUnits) {
    resultList.emplace_back(&SU);
  }
  return resultList;
}

void sortByHeight(std::vector<SUnit *> &workList) {
  std::sort(workList.begin(), workList.end(),
            [](const SUnit *a, const SUnit *b) {
              // Lowest height first.
              if (a->getHeight() < b->getHeight())
                return true;
              // If height the same, NodeNum big first.
              if (a->getHeight() == b->getHeight())
                return a->NodeNum > b->NodeNum;
              return false;
            });
}

void sortByInChain(std::vector<SUnit *> &workList, DenseSet<SUnit *> &Chained) {
  // In chain nodes at end.
  std::sort(workList.begin(), workList.end(),
            [&Chained](const SUnit *a, const SUnit *b) {
              return Chained.count(a) < Chained.count(b);
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
  unsigned Reg = MO->getReg();
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

void collectSameHeightBackNodes(SUnit *SU, SmallDenseSet<SUnit *, 2> &backNodes,
                                unsigned NodeNum,
                                SmallDenseSet<SUnit *, 4> &visitedNodes) {
  if (visitedNodes.count(SU))
    return;
  visitedNodes.insert(SU);

  for (SDep &Dep : SU->Succs) {
    if (Dep.isWeak())
      continue;
    if (Dep.getLatency() > 0)
      continue;

    SUnit *Succ = Dep.getSUnit(); /*
     if (Succ->NodeNum >= NodeNum)
       continue;*/

    backNodes.insert(Succ);
    collectSameHeightBackNodes(Succ, backNodes, NodeNum, visitedNodes);
  }
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
  std::vector<SUnit *> workList = buildWorkList(SUnits);
  IntEqClasses EqClasses(SUnits.size());

  while (!workList.empty()) {
    sortByHeight(workList);
    // Highest SU.
    SUnit *SU = workList.back();
    workList.pop_back();
    if (!SU->isInstr())
      continue;
    if (ChainedNodes.count(SU) > 0)
      continue;
    IsRecomputeHeight = false;
    Lineage lineage = buildChain(SU, SUnits);

    // Remove chained nodes from worklist.
    sortByInChain(workList, ChainedNodes);
    while (!workList.empty()) {
      SUnit *back = workList.back();
      if (ChainedNodes.count(back))
        workList.pop_back();
      else
        break;
    }

    Lineages.emplace_back(lineage);

    if (IsRecomputeHeight) {
      // Update height from tail.
      SUnit *tail = lineage.Nodes.back();
      tail->setDepthDirty();
      tail->getHeight();
    }
  }

  DenseSet<SUnit *> tailSet;
  for (Lineage &L : Lineages) {
    if (L.Nodes.size() < 2)
      continue;
    auto it = L.Nodes.rbegin();
    it++;
    SUnit *tail = L.Nodes.back();
    // If already as tail for other lineage, start from next.
    if (tailSet.count(tail) > 0) {
      tail = *it;
      it++;
    } else {
      tailSet.insert(tail);
    }
    for (; it != L.Nodes.rend(); it++) {
      SUnit *SU = *it;
      if (tail->NodeNum == -1)
        continue;
      EqClasses.join(SU->NodeNum, tail->NodeNum);
    }
  }

  EqClasses.compress();
  // TODO: assign sub class to node.
  for (Lineage &L : Lineages) {
    for (SUnit *SU : L.Nodes) {
      if (SU->NodeNum == -1)
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
      } for (int i = 0; i < Lineages.size(); i++) {
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
    for (int i = 1; i < SameHeightCandidate.size(); i++) {
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
  HRB::Lineage chain;
  chain.addNode(Node);
  ChainedNodes.insert(Node);
  LLVM_DEBUG(dbgs() << "start chain " << Node->NodeNum << "("
                    << Node->getHeight() << ")\n");
  while (Node->NumSuccsLeft > 0) {
    SUnit *Heir = findHeir(Node, SUnits);
    if (!Heir)
      break;
    chain.addNode(Heir);

    LLVM_DEBUG(dbgs() << "add node to chain " << Heir->NodeNum << "\n");
    if (ChainedNodes.count(Heir) > 0)
      break;
    ChainedNodes.insert(Heir);

    Node = Heir;
  }
  // Find biggest vgpr RC for the chain.
  // TODO: Build conflict and allocate on each edge of the chain.
  const TargetRegisterClass *RC = nullptr;
  unsigned maxRCSize = 0;
  for (SUnit *SU : chain.Nodes) {
    const TargetRegisterClass *SuRC = getRegClass(SU, MRI, SIRI);
    unsigned RCSize = getVGPRSize(SuRC, SIRI);
    if (RCSize > maxRCSize) {
      maxRCSize = RCSize;
      RC = SuRC;
    }
  }
  if (!RC) {
    // TODO: Find biggest sgpr RC.
    unsigned maxRCSize = 0;
    for (SUnit *SU : chain.Nodes) {
      const TargetRegisterClass *SuRC = getRegClass(SU, MRI, SIRI);
      unsigned RCSize = getSGPRSize(SuRC, SIRI);
      if (RCSize > maxRCSize) {
        maxRCSize = RCSize;
        RC = SuRC;
      }
    }
  }
  chain.RC = RC;
  return chain;
}

void HRB::buildConflict() {

  for (unsigned i = 0; i < Lineages.size(); i++) {
    Lineage &a = Lineages[i];
    for (unsigned j = i + 1; j < Lineages.size(); j++) {
      Lineage &b = Lineages[j];
      if (isConflict(a, b)) {
        Color.Conflicts[i].insert(j);
        Color.Conflicts[j].insert(i);
        LLVM_DEBUG(dbgs() << i << " conflict" << j << "\n");
      }
    }
    // SelfConflict.
    Color.Conflicts[i].insert(i);
  }
}

bool HRB::canReach(llvm::SUnit *a, llvm::SUnit *b) {
  auto it = ReachMap.find(a);
  // If no reach info, treat as reach.
  if (it == ReachMap.end())
    return true;
  DenseSet<SUnit *> &CurReach = it->second;
  return CurReach.find(b) != CurReach.end();
}

void HRB::updateReachForEdge(llvm::SUnit *a, llvm::SUnit *b,
                             std::vector<llvm::SUnit> &SUnits) {
  DenseSet<SUnit *> &ReachA = ReachMap[a];
  ReachA.insert(b);
  DenseSet<SUnit *> &ReachB = ReachMap[b];
  ReachA.insert(ReachB.begin(), ReachB.end());

  for (SUnit &SU : SUnits) {
    if (!canReach(&SU, a))
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

bool HRB::isConflict(const Lineage &a, const Lineage &b) {
  // Make conflict between sgpr and vgpr to help group lineages when share
  // colors. Keep the conflict will group lineages in avoid mix use color in
  // different sub exp.
  SUnit *head0 = a.getHead();
  SUnit *tail0 = a.getTail();
  SUnit *head1 = b.getHead();
  SUnit *tail1 = b.getTail();
  DenseSet<SUnit *> &Reach0 = ReachMap[head0];
  DenseSet<SUnit *> &Reach1 = ReachMap[head1];
  bool r01 = Reach0.count(tail1) != 0;
  bool r10 = Reach1.count(tail0) != 0;
  return r01 && r10;
}
bool HRB::canFuse(const Lineage &a, const Lineage &b) {
  if (a.RC != b.RC) {
    // no RC will not conflict with other nodes.
    if (!a.RC)
      return false;
    if (!b.RC)
      return false;
    // SGRP and VGPR not conflict.
    if (SIRI->isSGPRClass(a.RC) != SIRI->isSGPRClass(b.RC))
      return false;
  }
  // Can Fuse if a.head reach b.tail but b.head not reach a.tail and vice versa.
  SUnit *head0 = a.getHead();
  SUnit *tail0 = a.getTail();
  SUnit *head1 = b.getHead();
  SUnit *tail1 = b.getTail();
  DenseSet<SUnit *> &Reach0 = ReachMap[head0];
  DenseSet<SUnit *> &Reach1 = ReachMap[head1];
  bool r01 = Reach0.count(tail1) != 0;
  bool r10 = Reach1.count(tail0) != 0;
  return r01 != r10;
}

bool HRB::tryFuse(Lineage &a, Lineage &b, std::vector<llvm::SUnit> &SUnits) {

  // Can Fuse if a.head reach b.tail but b.head not reach a.tail and vice versa.
  SUnit *head0 = a.getHead();
  SUnit *tail0 = a.getTail();
  SUnit *head1 = b.getHead();
  SUnit *tail1 = b.getTail();
  DenseSet<SUnit *> &Reach0 = ReachMap[head0];
  DenseSet<SUnit *> &Reach1 = ReachMap[head1];
  bool r01 = Reach0.count(tail1) != 0;
  bool r10 = Reach1.count(tail0) != 0;
  if (r01 == r10)
    return false;
  Lineage *newHead = &a;
  Lineage *newTail = &b;
  if (r01) {
    // a reach b, b cannot reach a.
    // link a.tail->b.head.
    newHead = &a;
    newTail = &b;
  } else {
    // b reach a, a cannot reach b.
    // link b.tail->a.head.
    newHead = &b;
    newTail = &a;
  }

  // Merge reg class.
  const TargetRegisterClass *RC0 = newHead->RC;
  const TargetRegisterClass *RC1 = newTail->RC;
  unsigned RC0Size = getVGPRSize(RC0, SIRI);
  unsigned RC1Size = getVGPRSize(RC1, SIRI);
  if (RC1Size > RC0Size)
    newHead->RC = RC1;
  // Merge chain.
  SUnit *fuseTail = newHead->getTail();
  SUnit *fuseHead = newTail->getHead();
  assert(ReachMap[fuseHead].count(fuseTail) == 0 && "");
  fuseHead->addPred(SDep(fuseTail, SDep::Artificial));
  LLVM_DEBUG(dbgs() << "fuse " << fuseTail->NodeNum << "->" << fuseHead->NodeNum
                    << "\n");
  // Update reach map.
  updateReachForEdge(fuseTail, fuseHead, SUnits);
  // Merge Nodes.
  newHead->Nodes.append(newTail->Nodes.begin(), newTail->Nodes.end());
  // Clear newTail.
  newTail->Nodes.clear();
  newTail->RC = nullptr;
  return true;
}

void HRB::fusionLineages(std::vector<llvm::SUnit> &SUnits) {
  if (Lineages.empty())
    return;
  bool IsUpdated = true;
  while (IsUpdated) {
    IsUpdated = false;
    int size = Lineages.size();
    for (int i = 0; i < size; i++) {
      Lineage &a = Lineages[i];
      if (a.length() == 0)
        continue;

      for (int j = i + 1; j < size; j++) {
        Lineage &b = Lineages[j];
        if (b.length() == 0)
          continue;
        if (tryFuse(a, b, SUnits)) {
          IsUpdated = true;
          if (a.length() == 0)
            break;
        }
      }
    }
    // Remove empty lineages.
    std::sort(Lineages.begin(), Lineages.end(),
              [](const Lineage &a, const Lineage &b) {
                return a.length() > b.length();
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

unsigned HRB::colorLineages(std::vector<Lineage *> &lineages,
                            DenseMap<Lineage *, unsigned> &AllocMap,
                            const unsigned Limit) {
  // allocate long Lineage first. How about size of RC?
  std::sort(lineages.begin(), lineages.end(),
            [](const Lineage *a, const Lineage *b) {
              // Make sure root allocate first.
              return a->length() > b->length();
            });

  unsigned maxColor = 0;
  const unsigned VGPR_LIMIT = 256 * 4;

  for (Lineage *L : lineages) {
    unsigned ID = L->ID;
    auto &Conflict = Color.Conflicts[ID];
    std::bitset<VGPR_LIMIT> colors;
    for (unsigned j : Conflict) {
      Lineage *C = &Lineages[j];
      if (AllocMap.count(C) == 0)
        continue;
      unsigned c = AllocMap[C];
      unsigned s = C->getSize();
      for (unsigned i = 0; i < s; i++) {
        unsigned pos = c + i;
        colors.set(pos);
      }
    }

    unsigned color = Limit;
    unsigned size = L->getSize();
    for (unsigned i = 0; i < Limit - size;) {
      unsigned oldI = i;
      for (unsigned j = 0; j < size; j++) {
        unsigned pos = i + size - 1 - j;
        if (colors.test(pos)) {
          i = pos + 1;
          break;
        }
      }

      if (i != oldI)
        continue;
      color = i;
      break;
    }

    AllocMap[L] = color;
    color += size;
    if (color > maxColor)
      maxColor = color;
  }
  return maxColor;
}

void HRB::ColorResult::colorSU(SUnit *SU, unsigned color) {
  ColorMap[SU] = color;
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
  auto it = HeadTailMap.find(SU);
  return it->second;
}

unsigned HRB::ColorResult::getColor(const llvm::SUnit *SU) const {
  auto it = ColorMap.find(SU);
  return it->second;
}

unsigned HRB::ColorResult::getSize(const llvm::SUnit *SU) const {
  auto it = SizeMap.find(SU);
  return it->second;
}

HRB::ColorResult &HRB::coloring() {
  // Collect VGPR lineages.
  std::vector<Lineage *> vgprLineages;
  for (Lineage &L : Lineages) {
    auto RC = L.RC;
    if (!RC)
      continue;
    if (SIRI->isSGPRClass(RC))
      continue;
    vgprLineages.emplace_back(&L);
  }

  const unsigned VGPR_LIMIT = 256 * 4;
  DenseMap<Lineage *, unsigned> VAllocMap;
  const unsigned maxVGPR = colorLineages(vgprLineages, VAllocMap, VGPR_LIMIT);

  // Collect SGPR lineages.
  std::vector<Lineage *> sgprLineages;
  for (Lineage &L : Lineages) {
    auto RC = L.RC;
    if (!RC)
      continue;
    if (!SIRI->isSGPRClass(RC))
      continue;
    sgprLineages.emplace_back(&L);
  }

  const unsigned SGPR_LIMIT = 104;
  DenseMap<Lineage *, unsigned> SAllocMap;
  const unsigned maxSGPR = colorLineages(sgprLineages, SAllocMap, SGPR_LIMIT);
  // +1 for each type of lineages(SGPR, VGPR, no reg).
  const unsigned maxReg = maxSGPR + 1 + maxVGPR + 1 + 1;
  const unsigned sgprBase = maxVGPR + 1;

  for (Lineage &L : Lineages) {
    // Collect HeadSet.
    Color.HeadSet.insert(L.getHead());
    Color.TailSet.insert(L.getTail());
    Color.HeadTailMap[L.getHead()] = L.getTail();
    // Save color.
    auto RC = L.RC;
    // All no reg lineage goes to maxReg.
    unsigned color = maxReg;
    if (!RC) {
    } else if (SIRI->isSGPRClass(RC)) {
      color = SAllocMap[&L] + sgprBase;
    } else {
      color = VAllocMap[&L];
    }
    unsigned size = L.getSize();
    for (SUnit *SU : L.Nodes) {
      Color.colorSU(SU, color);
      Color.SizeMap[SU] = size;
      Color.LineageMap[SU] = L.ID;
    }
  }
  Color.maxReg = maxReg;
  Color.maxSGPR = maxSGPR;
  Color.maxVGPR = maxVGPR;

  for (unsigned i = 0; i < Lineages.size(); i++) {
    Lineage &a = Lineages[i];
    SUnit *headA = a.getHead();
    unsigned colorA = Color.getColor(headA);
    unsigned sizeA = Color.getSize(headA);
    for (unsigned j = i + 1; j < Lineages.size(); j++) {
      Lineage &b = Lineages[j];

      SUnit *headB = b.getHead();
      unsigned colorB = Color.getColor(headB);
      unsigned sizeB = Color.getSize(headB);

      if (colorB >= (colorA + sizeA))
        continue;
      if (colorA >= (colorB + sizeB))
        continue;
      Color.ShareColorLineages.insert(i);
      Color.ShareColorLineages.insert(j);
    }
  }

  return Color;
}

void HRB::dump() {
  for (int i = 0; i < Lineages.size(); i++) {
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
      for (int j = 0; j < Lineages.size(); j++) {
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
    for (auto it : ReachMap) {
      SUnit *SU = it.first;
      auto &Reach = it.second;
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
  HRB hrb(MRI, SIRI);
  // build reach info to avoid dead loop when build linear.
  hrb.buildReachRelation(BRoots);
  hrb.buildLinear(SUnits);

  std::sort(BRoots.begin(), BRoots.end(), [](const SUnit *a, const SUnit *b) {
    return a->NumSuccsLeft < b->NumSuccsLeft;
  });
  while (!BRoots.empty() && BRoots.back()->NumSuccsLeft > 0) {
    BRoots.pop_back();
  }

  hrb.buildReachRelation(BRoots);
  hrb.fusionLineages(SUnits);
  hrb.buildConflict();
  const HRB::ColorResult &Color = hrb.coloring();

  LLVM_DEBUG(hrb.dump());

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

  std::bitset<512 * 2> colors;
  auto isColorAvail = [&colors](unsigned color, unsigned size) -> bool {
    for (unsigned i = 0; i < size; i++) {
      unsigned pos = color + i;
      if (colors.test(pos))
        return false;
    }
    return true;
  };
  auto allocColor = [&colors](unsigned color, unsigned size) {
    for (unsigned i = 0; i < size; i++) {
      unsigned pos = color + i;
      assert(!colors.test(pos) && "color already allocated");
      LLVM_DEBUG(dbgs() << pos << "is allocated\n");
      colors.set(pos);
    }
  };

  auto freeColor = [&colors](unsigned color, unsigned size) {
    for (unsigned i = 0; i < size; i++) {
      unsigned pos = color + i;
      assert(colors.test(pos) && "color has not been allocated");
      LLVM_DEBUG(dbgs() << pos << "is free\n");
      colors.reset(pos);
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
  const DenseSet<unsigned> &ShareColorLineages = Color.ShareColorLineages;

  std::vector<const SUnit *> Schedule;
  DenseSet<unsigned> UnfinishedLineages;
  while (!ReadyList.empty()) {
    // Make sure node conflict with predLineage first.
    std::sort(ReadyList.begin(), ReadyList.end(),
              [&UnfinishedLineages, &Color](const SUnit *a, const SUnit *b) {
                unsigned confA = 0;
                for (unsigned L : UnfinishedLineages) {
                  if (Color.isConflict(a, L))
                    confA++;
                }
                unsigned confB = 0;
                for (unsigned L : UnfinishedLineages) {
                  if (Color.isConflict(b, L))
                    confB++;
                }
                return confA > confB;
              });

    LLVM_DEBUG(dbgs() << "ReadyList:\n"; for (SUnit *SU
                                              : ReadyList) {
      dbgs() << " " << SU->NodeNum;
    } dbgs() << "\n";);
    SUnit *Candidate = nullptr;
    for (auto it = ReadyList.begin(); it != ReadyList.end(); it++) {
      SUnit *SU = *it;
      unsigned color = Color.getColor(SU);
      unsigned size = Color.getSize(SU);
      // If SU is not head or color is available, SU is the candidate.
      if (Color.isHead(SU)) {
        if (!isColorAvail(color, size))
          continue;
        // alloc color.
        allocColor(color, size);
        // save tail color.
        const SUnit *Tail = Color.getTail(SU);
        unsigned ID = Color.getLineage(SU);
        SmallVector<std::tuple<unsigned, unsigned, unsigned>, 2> &tailColors =
            TailMap[Tail];
        tailColors.emplace_back(std::make_tuple(color, size, ID));
        if (ShareColorLineages.count(ID))
          UnfinishedLineages.insert(ID);
      }

      // free color for working lineage which end with SU.
      if (Color.isTail(SU)) {
        auto &tailColors = TailMap[SU];
        for (auto &tailTuple : tailColors) {
          unsigned lineageColor, lineageSize, ID;
          std::tie(lineageColor, lineageSize, ID) = tailTuple;
          freeColor(lineageColor, lineageSize);
          if (ShareColorLineages.count(ID))
            UnfinishedLineages.insert(ID);
        }
        // Clear the tail.
        TailMap.erase(SU);
      }

      Candidate = SU;
      // Remove Candidate from ReadyList.
      ReadyList.erase(it);
      break;
    }

    if (!Candidate) {
      // In case failed to find candidate, start a lineage if there is one.
      for (auto it = ReadyList.begin(); it != ReadyList.end(); it++) {
        SUnit *SU = *it;

        if (!Color.isHead(SU)) {
          continue;
        }
        Candidate = SU;
        // Remove Candidate from ReadyList.
        ReadyList.erase(it);
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
