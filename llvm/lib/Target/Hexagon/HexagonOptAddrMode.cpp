//===- HexagonOptAddrMode.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements a Hexagon-specific pass to optimize addressing mode for
// load/store instructions.
//===----------------------------------------------------------------------===//

#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominanceFrontier.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RDFGraph.h"
#include "llvm/CodeGen/RDFLiveness.h"
#include "llvm/CodeGen/RDFRegisters.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>

#define DEBUG_TYPE "opt-addr-mode"

using namespace llvm;
using namespace rdf;

static cl::opt<int> CodeGrowthLimit("hexagon-amode-growth-limit",
  cl::Hidden, cl::init(0), cl::desc("Code growth limit for address mode "
  "optimization"));

extern cl::opt<unsigned> RDFFuncBlockLimit;

namespace llvm {

  FunctionPass *createHexagonOptAddrMode();
  void initializeHexagonOptAddrModePass(PassRegistry&);

} // end namespace llvm

namespace {

class HexagonOptAddrMode : public MachineFunctionPass {
public:
  static char ID;

  HexagonOptAddrMode() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "Optimize addressing mode of load/store";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachineDominanceFrontier>();
    AU.setPreservesAll();
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  using MISetType = DenseSet<MachineInstr *>;
  using InstrEvalMap = DenseMap<MachineInstr *, bool>;
  DenseSet<MachineInstr *> ProcessedAddiInsts;

  MachineRegisterInfo *MRI = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const HexagonInstrInfo *HII = nullptr;
  const HexagonRegisterInfo *HRI = nullptr;
  MachineDominatorTree *MDT = nullptr;
  DataFlowGraph *DFG = nullptr;
  DataFlowGraph::DefStackMap DefM;
  Liveness *LV = nullptr;
  MISetType Deleted;

  bool processBlock(NodeAddr<BlockNode *> BA);
  bool xformUseMI(MachineInstr *TfrMI, MachineInstr *UseMI,
                  NodeAddr<UseNode *> UseN, unsigned UseMOnum);
  bool processAddBases(NodeAddr<StmtNode *> AddSN, MachineInstr *AddMI);
  bool usedInLoadStore(NodeAddr<StmtNode *> CurrentInstSN, int64_t NewOffset);
  bool findFirstReachedInst(
      MachineInstr *AddMI,
      std::vector<std::pair<NodeAddr<StmtNode *>, NodeAddr<UseNode *>>>
          &AddiList,
      NodeAddr<StmtNode *> &UseSN);
  bool updateAddBases(MachineInstr *CurrentMI, MachineInstr *FirstReachedMI,
                      int64_t NewOffset);
  bool processAddUses(NodeAddr<StmtNode *> AddSN, MachineInstr *AddMI,
                      const NodeList &UNodeList);
  bool updateAddUses(MachineInstr *AddMI, MachineInstr *UseMI);
  bool analyzeUses(unsigned DefR, const NodeList &UNodeList,
                   InstrEvalMap &InstrEvalResult, short &SizeInc);
  bool hasRepForm(MachineInstr &MI, unsigned TfrDefR);
  bool canRemoveAddasl(NodeAddr<StmtNode *> AddAslSN, MachineInstr &MI,
                       const NodeList &UNodeList);
  bool isSafeToExtLR(NodeAddr<StmtNode *> SN, MachineInstr *MI,
                     unsigned LRExtReg, const NodeList &UNodeList);
  void getAllRealUses(NodeAddr<StmtNode *> SN, NodeList &UNodeList);
  bool allValidCandidates(NodeAddr<StmtNode *> SA, NodeList &UNodeList);
  short getBaseWithLongOffset(const MachineInstr &MI) const;
  bool changeStore(MachineInstr *OldMI, MachineOperand ImmOp,
                   unsigned ImmOpNum);
  bool changeLoad(MachineInstr *OldMI, MachineOperand ImmOp, unsigned ImmOpNum);
  bool changeAddAsl(NodeAddr<UseNode *> AddAslUN, MachineInstr *AddAslMI,
                    const MachineOperand &ImmOp, unsigned ImmOpNum);
  bool isValidOffset(MachineInstr *MI, int Offset);
  unsigned getBaseOpPosition(MachineInstr *MI);
  unsigned getOffsetOpPosition(MachineInstr *MI);
};

} // end anonymous namespace

char HexagonOptAddrMode::ID = 0;

INITIALIZE_PASS_BEGIN(HexagonOptAddrMode, "amode-opt",
                      "Optimize addressing mode", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominanceFrontier)
INITIALIZE_PASS_END(HexagonOptAddrMode, "amode-opt", "Optimize addressing mode",
                    false, false)

bool HexagonOptAddrMode::hasRepForm(MachineInstr &MI, unsigned TfrDefR) {
  const MCInstrDesc &MID = MI.getDesc();

  if ((!MID.mayStore() && !MID.mayLoad()) || HII->isPredicated(MI))
    return false;

  if (MID.mayStore()) {
    MachineOperand StOp = MI.getOperand(MI.getNumOperands() - 1);
    if (StOp.isReg() && StOp.getReg() == TfrDefR)
      return false;
  }

  if (HII->getAddrMode(MI) == HexagonII::BaseRegOffset)
    // Tranform to Absolute plus register offset.
    return (HII->changeAddrMode_rr_ur(MI) >= 0);
  else if (HII->getAddrMode(MI) == HexagonII::BaseImmOffset)
    // Tranform to absolute addressing mode.
    return (HII->changeAddrMode_io_abs(MI) >= 0);

  return false;
}

// Check if addasl instruction can be removed. This is possible only
// if it's feeding to only load/store instructions with base + register
// offset as these instruction can be tranformed to use 'absolute plus
// shifted register offset'.
// ex:
// Rs = ##foo
// Rx = addasl(Rs, Rt, #2)
// Rd = memw(Rx + #28)
// Above three instructions can be replaced with Rd = memw(Rt<<#2 + ##foo+28)

bool HexagonOptAddrMode::canRemoveAddasl(NodeAddr<StmtNode *> AddAslSN,
                                         MachineInstr &MI,
                                         const NodeList &UNodeList) {
  // check offset size in addasl. if 'offset > 3' return false
  const MachineOperand &OffsetOp = MI.getOperand(3);
  if (!OffsetOp.isImm() || OffsetOp.getImm() > 3)
    return false;

  Register OffsetReg = MI.getOperand(2).getReg();
  RegisterRef OffsetRR;
  NodeId OffsetRegRD = 0;
  for (NodeAddr<UseNode *> UA : AddAslSN.Addr->members_if(DFG->IsUse, *DFG)) {
    RegisterRef RR = UA.Addr->getRegRef(*DFG);
    if (OffsetReg == RR.Reg) {
      OffsetRR = RR;
      OffsetRegRD = UA.Addr->getReachingDef();
    }
  }

  for (auto I = UNodeList.rbegin(), E = UNodeList.rend(); I != E; ++I) {
    NodeAddr<UseNode *> UA = *I;
    NodeAddr<InstrNode *> IA = UA.Addr->getOwner(*DFG);
    if (UA.Addr->getFlags() & NodeAttrs::PhiRef)
      return false;
    NodeAddr<RefNode*> AA = LV->getNearestAliasedRef(OffsetRR, IA);
    if ((DFG->IsDef(AA) && AA.Id != OffsetRegRD) ||
         AA.Addr->getReachingDef() != OffsetRegRD)
      return false;

    MachineInstr &UseMI = *NodeAddr<StmtNode *>(IA).Addr->getCode();
    NodeAddr<DefNode *> OffsetRegDN = DFG->addr<DefNode *>(OffsetRegRD);
    // Reaching Def to an offset register can't be a phi.
    if ((OffsetRegDN.Addr->getFlags() & NodeAttrs::PhiRef) &&
        MI.getParent() != UseMI.getParent())
    return false;

    const MCInstrDesc &UseMID = UseMI.getDesc();
    if ((!UseMID.mayLoad() && !UseMID.mayStore()) ||
        HII->getAddrMode(UseMI) != HexagonII::BaseImmOffset ||
        getBaseWithLongOffset(UseMI) < 0)
      return false;

    // Addasl output can't be a store value.
    if (UseMID.mayStore() && UseMI.getOperand(2).isReg() &&
        UseMI.getOperand(2).getReg() == MI.getOperand(0).getReg())
      return false;

    for (auto &Mo : UseMI.operands())
      // Is it a frame index?
      if (Mo.isFI())
        return false;
    // Is the OffsetReg definition actually reaches UseMI?
    if (!UseMI.getParent()->isLiveIn(OffsetReg) &&
        MI.getParent() != UseMI.getParent()) {
      LLVM_DEBUG(dbgs() << "  The offset reg " << printReg(OffsetReg, TRI)
                        << " is NOT live in to MBB "
                        << UseMI.getParent()->getName() << "\n");
      return false;
    }
  }
  return true;
}

bool HexagonOptAddrMode::allValidCandidates(NodeAddr<StmtNode *> SA,
                                            NodeList &UNodeList) {
  for (auto I = UNodeList.rbegin(), E = UNodeList.rend(); I != E; ++I) {
    NodeAddr<UseNode *> UN = *I;
    RegisterRef UR = UN.Addr->getRegRef(*DFG);
    NodeSet Visited, Defs;
    const auto &P = LV->getAllReachingDefsRec(UR, UN, Visited, Defs);
    if (!P.second) {
      LLVM_DEBUG({
        dbgs() << "*** Unable to collect all reaching defs for use ***\n"
               << PrintNode<UseNode*>(UN, *DFG) << '\n'
               << "The program's complexity may exceed the limits.\n";
      });
      return false;
    }
    const auto &ReachingDefs = P.first;
    if (ReachingDefs.size() > 1) {
      LLVM_DEBUG({
        dbgs() << "*** Multiple Reaching Defs found!!! ***\n";
        for (auto DI : ReachingDefs) {
          NodeAddr<UseNode *> DA = DFG->addr<UseNode *>(DI);
          NodeAddr<StmtNode *> TempIA = DA.Addr->getOwner(*DFG);
          dbgs() << "\t\t[Reaching Def]: "
                 << Print<NodeAddr<InstrNode *>>(TempIA, *DFG) << "\n";
        }
      });
      return false;
    }
  }
  return true;
}

void HexagonOptAddrMode::getAllRealUses(NodeAddr<StmtNode *> SA,
                                        NodeList &UNodeList) {
  for (NodeAddr<DefNode *> DA : SA.Addr->members_if(DFG->IsDef, *DFG)) {
    LLVM_DEBUG(dbgs() << "\t\t[DefNode]: "
                      << Print<NodeAddr<DefNode *>>(DA, *DFG) << "\n");
    RegisterRef DR = DA.Addr->getRegRef(*DFG);

    auto UseSet = LV->getAllReachedUses(DR, DA);

    for (auto UI : UseSet) {
      NodeAddr<UseNode *> UA = DFG->addr<UseNode *>(UI);
      LLVM_DEBUG({
        NodeAddr<StmtNode *> TempIA = UA.Addr->getOwner(*DFG);
        dbgs() << "\t\t\t[Reached Use]: "
               << Print<NodeAddr<InstrNode *>>(TempIA, *DFG) << "\n";
      });

      if (UA.Addr->getFlags() & NodeAttrs::PhiRef) {
        NodeAddr<PhiNode *> PA = UA.Addr->getOwner(*DFG);
        NodeId id = PA.Id;
        const Liveness::RefMap &phiUse = LV->getRealUses(id);
        LLVM_DEBUG(dbgs() << "\t\t\t\tphi real Uses"
                          << Print<Liveness::RefMap>(phiUse, *DFG) << "\n");
        if (!phiUse.empty()) {
          for (auto I : phiUse) {
            if (!DFG->getPRI().alias(RegisterRef(I.first), DR))
              continue;
            auto phiUseSet = I.second;
            for (auto phiUI : phiUseSet) {
              NodeAddr<UseNode *> phiUA = DFG->addr<UseNode *>(phiUI.first);
              UNodeList.push_back(phiUA);
            }
          }
        }
      } else
        UNodeList.push_back(UA);
    }
  }
}

bool HexagonOptAddrMode::isSafeToExtLR(NodeAddr<StmtNode *> SN,
                                       MachineInstr *MI, unsigned LRExtReg,
                                       const NodeList &UNodeList) {
  RegisterRef LRExtRR;
  NodeId LRExtRegRD = 0;
  // Iterate through all the UseNodes in SN and find the reaching def
  // for the LRExtReg.
  for (NodeAddr<UseNode *> UA : SN.Addr->members_if(DFG->IsUse, *DFG)) {
    RegisterRef RR = UA.Addr->getRegRef(*DFG);
    if (LRExtReg == RR.Reg) {
      LRExtRR = RR;
      LRExtRegRD = UA.Addr->getReachingDef();
    }
  }

  for (auto I = UNodeList.rbegin(), E = UNodeList.rend(); I != E; ++I) {
    NodeAddr<UseNode *> UA = *I;
    NodeAddr<InstrNode *> IA = UA.Addr->getOwner(*DFG);
    // The reaching def of LRExtRR at load/store node should be same as the
    // one reaching at the SN.
    if (UA.Addr->getFlags() & NodeAttrs::PhiRef)
      return false;
    NodeAddr<RefNode*> AA = LV->getNearestAliasedRef(LRExtRR, IA);
    if ((DFG->IsDef(AA) && AA.Id != LRExtRegRD) ||
        AA.Addr->getReachingDef() != LRExtRegRD) {
      LLVM_DEBUG(
          dbgs() << "isSafeToExtLR: Returning false; another reaching def\n");
      return false;
    }

    // If the register is undefined (for example if it's a reserved register),
    // it may still be possible to extend the range, but it's safer to be
    // conservative and just punt.
    if (LRExtRegRD == 0)
      return false;

    MachineInstr *UseMI = NodeAddr<StmtNode *>(IA).Addr->getCode();
    NodeAddr<DefNode *> LRExtRegDN = DFG->addr<DefNode *>(LRExtRegRD);
    // Reaching Def to LRExtReg can't be a phi.
    if ((LRExtRegDN.Addr->getFlags() & NodeAttrs::PhiRef) &&
        MI->getParent() != UseMI->getParent())
      return false;
    // Is the OffsetReg definition actually reaches UseMI?
    if (!UseMI->getParent()->isLiveIn(LRExtReg) &&
        MI->getParent() != UseMI->getParent()) {
      LLVM_DEBUG(dbgs() << "  The LRExtReg reg " << printReg(LRExtReg, TRI)
                        << " is NOT live in to MBB "
                        << UseMI->getParent()->getName() << "\n");
      return false;
    }
  }
  return true;
}

bool HexagonOptAddrMode::isValidOffset(MachineInstr *MI, int Offset) {
  if (HII->isHVXVec(*MI)) {
    // only HVX vgather instructions handled
    // TODO: extend the pass to other vector load/store operations
    switch (MI->getOpcode()) {
    case Hexagon::V6_vgathermh_pseudo:
    case Hexagon::V6_vgathermw_pseudo:
    case Hexagon::V6_vgathermhw_pseudo:
    case Hexagon::V6_vgathermhq_pseudo:
    case Hexagon::V6_vgathermwq_pseudo:
    case Hexagon::V6_vgathermhwq_pseudo:
      return HII->isValidOffset(MI->getOpcode(), Offset, HRI, false);
    default:
      if (HII->getAddrMode(*MI) == HexagonII::BaseImmOffset) {
        // The immediates are mentioned in multiples of vector counts
        unsigned AlignMask = HII->getMemAccessSize(*MI) - 1;
        if ((AlignMask & Offset) == 0)
          return HII->isValidOffset(MI->getOpcode(), Offset, HRI, false);
      }
      return false;
    }
  }

  if (HII->getAddrMode(*MI) != HexagonII::BaseImmOffset)
    return false;

  unsigned AlignMask = 0;
  switch (HII->getMemAccessSize(*MI)) {
  case HexagonII::MemAccessSize::DoubleWordAccess:
    AlignMask = 0x7;
    break;
  case HexagonII::MemAccessSize::WordAccess:
    AlignMask = 0x3;
    break;
  case HexagonII::MemAccessSize::HalfWordAccess:
    AlignMask = 0x1;
    break;
  case HexagonII::MemAccessSize::ByteAccess:
    AlignMask = 0x0;
    break;
  default:
    return false;
  }

  if ((AlignMask & Offset) != 0)
    return false;
  return HII->isValidOffset(MI->getOpcode(), Offset, HRI, false);
}

unsigned HexagonOptAddrMode::getBaseOpPosition(MachineInstr *MI) {
  const MCInstrDesc &MID = MI->getDesc();
  switch (MI->getOpcode()) {
  // vgather pseudos are mayLoad and mayStore
  // hence need to explicitly specify Base and
  // Offset operand positions
  case Hexagon::V6_vgathermh_pseudo:
  case Hexagon::V6_vgathermw_pseudo:
  case Hexagon::V6_vgathermhw_pseudo:
  case Hexagon::V6_vgathermhq_pseudo:
  case Hexagon::V6_vgathermwq_pseudo:
  case Hexagon::V6_vgathermhwq_pseudo:
    return 0;
  default:
    return MID.mayLoad() ? 1 : 0;
  }
}

unsigned HexagonOptAddrMode::getOffsetOpPosition(MachineInstr *MI) {
  assert(
      (HII->getAddrMode(*MI) == HexagonII::BaseImmOffset) &&
      "Looking for an offset in non-BaseImmOffset addressing mode instruction");

  const MCInstrDesc &MID = MI->getDesc();
  switch (MI->getOpcode()) {
  // vgather pseudos are mayLoad and mayStore
  // hence need to explicitly specify Base and
  // Offset operand positions
  case Hexagon::V6_vgathermh_pseudo:
  case Hexagon::V6_vgathermw_pseudo:
  case Hexagon::V6_vgathermhw_pseudo:
  case Hexagon::V6_vgathermhq_pseudo:
  case Hexagon::V6_vgathermwq_pseudo:
  case Hexagon::V6_vgathermhwq_pseudo:
    return 1;
  default:
    return MID.mayLoad() ? 2 : 1;
  }
}

bool HexagonOptAddrMode::usedInLoadStore(NodeAddr<StmtNode *> CurrentInstSN,
                                         int64_t NewOffset) {
  NodeList LoadStoreUseList;

  getAllRealUses(CurrentInstSN, LoadStoreUseList);
  bool FoundLoadStoreUse = false;
  for (auto I = LoadStoreUseList.begin(), E = LoadStoreUseList.end(); I != E;
       ++I) {
    NodeAddr<UseNode *> UN = *I;
    NodeAddr<StmtNode *> SN = UN.Addr->getOwner(*DFG);
    MachineInstr *LoadStoreMI = SN.Addr->getCode();
    const MCInstrDesc &MID = LoadStoreMI->getDesc();
    if ((MID.mayLoad() || MID.mayStore()) &&
        isValidOffset(LoadStoreMI, NewOffset)) {
      FoundLoadStoreUse = true;
      break;
    }
  }
  return FoundLoadStoreUse;
}

bool HexagonOptAddrMode::findFirstReachedInst(
    MachineInstr *AddMI,
    std::vector<std::pair<NodeAddr<StmtNode *>, NodeAddr<UseNode *>>> &AddiList,
    NodeAddr<StmtNode *> &UseSN) {
  // Find the very first Addi instruction in the current basic block among the
  // AddiList This is the Addi that should be preserved so that we do not need
  // to handle the complexity of moving instructions
  //
  // TODO: find Addi instructions across basic blocks
  //
  // TODO: Try to remove this and add a solution that optimizes the number of
  // Addi instructions that can be modified.
  // This change requires choosing the Addi with the median offset value, but
  // would also require moving that instruction above the others. Since this
  // pass runs after register allocation, there might be multiple cases that
  // need to be handled if we move instructions around
  MachineBasicBlock *CurrentMBB = AddMI->getParent();
  for (auto &InstIter : *CurrentMBB) {
    // If the instruction is an Addi and is in the AddiList
    if (InstIter.getOpcode() == Hexagon::A2_addi) {
      auto Iter = std::find_if(
          AddiList.begin(), AddiList.end(), [&InstIter](const auto &SUPair) {
            return SUPair.first.Addr->getCode() == &InstIter;
          });
      if (Iter != AddiList.end()) {
        UseSN = Iter->first;
        return true;
      }
    }
  }
  return false;
}

// This function tries to modify the immediate value in Hexagon::Addi
// instructions, so that the immediates could then be moved into a load/store
// instruction with offset and the add removed completely when we call
// processAddUses
//
// For Example, If we have the below sequence of instructions:
//
//         r1 = add(r2,#1024)
//               ...
//         r3 = add(r2,#1152)
//               ...
//         r4 = add(r2,#1280)
//
// Where the register r2 has the same reaching definition, They get modified to
// the below sequence:
//
//         r1 = add(r2,#1024)
//              ...
//         r3 = add(r1,#128)
//              ...
//         r4 = add(r1,#256)
//
//  The below change helps the processAddUses method to later move the
//  immediates #128 and #256 into a load/store instruction that can take an
//  offset, like the Vd = mem(Rt+#s4)
bool HexagonOptAddrMode::processAddBases(NodeAddr<StmtNode *> AddSN,
                                         MachineInstr *AddMI) {

  bool Changed = false;

  LLVM_DEBUG(dbgs() << "\n\t\t[Processing Addi]: " << *AddMI << "\n");

  auto Processed =
      [](const MachineInstr *MI,
         const DenseSet<MachineInstr *> &ProcessedAddiInsts) -> bool {
    // If we've already processed this Addi, just return
    if (ProcessedAddiInsts.find(MI) != ProcessedAddiInsts.end()) {
      LLVM_DEBUG(dbgs() << "\t\t\tAddi already found in ProcessedAddiInsts: "
                        << *MI << "\n\t\t\tSkipping...");
      return true;
    }
    return false;
  };

  if (Processed(AddMI, ProcessedAddiInsts))
    return Changed;
  ProcessedAddiInsts.insert(AddMI);

  // Get the base register that would be shared by other Addi Intructions
  Register BaseReg = AddMI->getOperand(1).getReg();

  // Store a list of all Addi instructions that share the above common base
  // register
  std::vector<std::pair<NodeAddr<StmtNode *>, NodeAddr<UseNode *>>> AddiList;

  NodeId UAReachingDefID;
  // Find the UseNode that contains the base register and it's reachingDef
  for (NodeAddr<UseNode *> UA : AddSN.Addr->members_if(DFG->IsUse, *DFG)) {
    RegisterRef URR = UA.Addr->getRegRef(*DFG);
    if (BaseReg != URR.Reg)
      continue;

    UAReachingDefID = UA.Addr->getReachingDef();
    NodeAddr<DefNode *> UADef = DFG->addr<DefNode *>(UAReachingDefID);
    if (!UAReachingDefID || UADef.Addr->getFlags() & NodeAttrs::PhiRef) {
      LLVM_DEBUG(dbgs() << "\t\t\t Could not find reachingDef. Skipping...\n");
      return false;
    }
  }

  NodeAddr<DefNode *> UAReachingDef = DFG->addr<DefNode *>(UAReachingDefID);
  NodeAddr<StmtNode *> ReachingDefStmt = UAReachingDef.Addr->getOwner(*DFG);

  // If the reaching definition is a predicated instruction, this might not be
  // the only definition of our base register, so return immediately.
  MachineInstr *ReachingDefInstr = ReachingDefStmt.Addr->getCode();
  if (HII->isPredicated(*ReachingDefInstr))
    return false;

  NodeList AddiUseList;

  // Find all Addi instructions that share the same base register and add them
  // to the AddiList
  getAllRealUses(ReachingDefStmt, AddiUseList);
  for (auto I = AddiUseList.begin(), E = AddiUseList.end(); I != E; ++I) {
    NodeAddr<UseNode *> UN = *I;
    NodeAddr<StmtNode *> SN = UN.Addr->getOwner(*DFG);
    MachineInstr *MI = SN.Addr->getCode();

    // Only add instructions if it's an Addi and it's not already processed.
    if (MI->getOpcode() == Hexagon::A2_addi &&
        !(MI != AddMI && Processed(MI, ProcessedAddiInsts))) {
      AddiList.push_back({SN, UN});

      // This ensures that we process each instruction only once
      ProcessedAddiInsts.insert(MI);
    }
  }

  // If there's only one Addi instruction, nothing to do here
  if (AddiList.size() <= 1)
    return Changed;

  NodeAddr<StmtNode *> FirstReachedUseSN;
  // Find the first reached use of Addi instruction from the list
  if (!findFirstReachedInst(AddMI, AddiList, FirstReachedUseSN))
    return Changed;

  // If we reach this point we know that the StmtNode FirstReachedUseSN is for
  // an Addi instruction. So, we're guaranteed to have just one DefNode, and
  // hence we can access the front() directly without checks
  NodeAddr<DefNode *> FirstReachedUseDN =
      FirstReachedUseSN.Addr->members_if(DFG->IsDef, *DFG).front();

  MachineInstr *FirstReachedMI = FirstReachedUseSN.Addr->getCode();
  const MachineOperand FirstReachedMIImmOp = FirstReachedMI->getOperand(2);
  if (!FirstReachedMIImmOp.isImm())
    return false;

  for (auto &I : AddiList) {
    NodeAddr<StmtNode *> CurrentInstSN = I.first;
    NodeAddr<UseNode *> CurrentInstUN = I.second;

    MachineInstr *CurrentMI = CurrentInstSN.Addr->getCode();
    MachineOperand &CurrentMIImmOp = CurrentMI->getOperand(2);

    int64_t NewOffset;

    // Even though we know it's an Addi instruction, the second operand could be
    // a global value and not an immediate
    if (!CurrentMIImmOp.isImm())
      continue;

    NewOffset = CurrentMIImmOp.getImm() - FirstReachedMIImmOp.getImm();

    // This is the first occuring Addi, so skip modifying this
    if (CurrentMI == FirstReachedMI) {
      continue;
    }

    if (CurrentMI->getParent() != FirstReachedMI->getParent())
      continue;

    // Modify the Addi instruction only if it could be used to modify a
    // future load/store instruction and get removed
    //
    // This check is needed because, if we modify the current Addi instruction
    // we create RAW dependence between the FirstReached Addi and the current
    // one, which could result in extra packets. So we only do this change if
    // we know the current Addi would get removed later
    if (!usedInLoadStore(CurrentInstSN, NewOffset)) {
      return false;
    }

    // Verify whether the First Addi's definition register is still live when
    // we reach the current Addi
    RegisterRef FirstReachedDefRR = FirstReachedUseDN.Addr->getRegRef(*DFG);
    NodeAddr<InstrNode *> CurrentAddiIN = CurrentInstUN.Addr->getOwner(*DFG);
    NodeAddr<RefNode *> NearestAA =
        LV->getNearestAliasedRef(FirstReachedDefRR, CurrentAddiIN);
    if ((DFG->IsDef(NearestAA) && NearestAA.Id != FirstReachedUseDN.Id) ||
        (!DFG->IsDef(NearestAA) &&
         NearestAA.Addr->getReachingDef() != FirstReachedUseDN.Id)) {
      // Found another definition of FirstReachedDef
      LLVM_DEBUG(dbgs() << "\t\t\tCould not modify below Addi since the first "
                           "defined Addi register was redefined\n");
      continue;
    }

    MachineOperand CurrentMIBaseOp = CurrentMI->getOperand(1);
    if (CurrentMIBaseOp.getReg() != FirstReachedMI->getOperand(1).getReg()) {
      continue;
    }

    // If we reached this point, then we can modify MI to use the result of
    // FirstReachedMI
    Changed |= updateAddBases(CurrentMI, FirstReachedMI, NewOffset);

    // Update the reachingDef of the Current AddI use after change
    CurrentInstUN.Addr->linkToDef(CurrentInstUN.Id, FirstReachedUseDN);
  }

  return Changed;
}

bool HexagonOptAddrMode::updateAddBases(MachineInstr *CurrentMI,
                                        MachineInstr *FirstReachedMI,
                                        int64_t NewOffset) {
  LLVM_DEBUG(dbgs() << "[About to modify the Addi]: " << *CurrentMI << "\n");
  const MachineOperand FirstReachedDef = FirstReachedMI->getOperand(0);
  Register FirstDefRegister = FirstReachedDef.getReg();

  MachineOperand &CurrentMIBaseOp = CurrentMI->getOperand(1);
  MachineOperand &CurrentMIImmOp = CurrentMI->getOperand(2);

  CurrentMIBaseOp.setReg(FirstDefRegister);
  CurrentMIBaseOp.setIsUndef(FirstReachedDef.isUndef());
  CurrentMIBaseOp.setImplicit(FirstReachedDef.isImplicit());
  CurrentMIImmOp.setImm(NewOffset);
  ProcessedAddiInsts.insert(CurrentMI);
  MRI->clearKillFlags(FirstDefRegister);
  return true;
}

bool HexagonOptAddrMode::processAddUses(NodeAddr<StmtNode *> AddSN,
                                        MachineInstr *AddMI,
                                        const NodeList &UNodeList) {

  Register AddDefR = AddMI->getOperand(0).getReg();
  Register BaseReg = AddMI->getOperand(1).getReg();
  for (auto I = UNodeList.rbegin(), E = UNodeList.rend(); I != E; ++I) {
    NodeAddr<UseNode *> UN = *I;
    NodeAddr<StmtNode *> SN = UN.Addr->getOwner(*DFG);
    MachineInstr *MI = SN.Addr->getCode();
    const MCInstrDesc &MID = MI->getDesc();
    if ((!MID.mayLoad() && !MID.mayStore()) ||
        HII->getAddrMode(*MI) != HexagonII::BaseImmOffset)
      return false;

    MachineOperand BaseOp = MI->getOperand(getBaseOpPosition(MI));

    if (!BaseOp.isReg() || BaseOp.getReg() != AddDefR)
      return false;

    MachineOperand OffsetOp = MI->getOperand(getOffsetOpPosition(MI));
    if (!OffsetOp.isImm())
      return false;

    int64_t newOffset = OffsetOp.getImm() + AddMI->getOperand(2).getImm();
    if (!isValidOffset(MI, newOffset))
      return false;

    // Since we'll be extending the live range of Rt in the following example,
    // make sure that is safe. another definition of Rt doesn't exist between 'add'
    // and load/store instruction.
    //
    // Ex: Rx= add(Rt,#10)
    //     memw(Rx+#0) = Rs
    // will be replaced with =>  memw(Rt+#10) = Rs
    if (!isSafeToExtLR(AddSN, AddMI, BaseReg, UNodeList))
      return false;
  }

  NodeId LRExtRegRD = 0;
  // Iterate through all the UseNodes in SN and find the reaching def
  // for the LRExtReg.
  for (NodeAddr<UseNode *> UA : AddSN.Addr->members_if(DFG->IsUse, *DFG)) {
    RegisterRef RR = UA.Addr->getRegRef(*DFG);
    if (BaseReg == RR.Reg)
      LRExtRegRD = UA.Addr->getReachingDef();
  }

  // Update all the uses of 'add' with the appropriate base and offset
  // values.
  bool Changed = false;
  for (auto I = UNodeList.rbegin(), E = UNodeList.rend(); I != E; ++I) {
    NodeAddr<UseNode *> UseN = *I;
    assert(!(UseN.Addr->getFlags() & NodeAttrs::PhiRef) &&
           "Found a PhiRef node as a real reached use!!");

    NodeAddr<StmtNode *> OwnerN = UseN.Addr->getOwner(*DFG);
    MachineInstr *UseMI = OwnerN.Addr->getCode();
    LLVM_DEBUG(dbgs() << "\t\t[MI <BB#" << UseMI->getParent()->getNumber()
                      << ">]: " << *UseMI << "\n");
    Changed |= updateAddUses(AddMI, UseMI);

    // Set the reachingDef for UseNode under consideration
    // after updating the Add use. This local change is
    // to avoid rebuilding of the RDF graph after update.
    NodeAddr<DefNode *> LRExtRegDN = DFG->addr<DefNode *>(LRExtRegRD);
    UseN.Addr->linkToDef(UseN.Id, LRExtRegDN);
  }

  if (Changed)
    Deleted.insert(AddMI);

  return Changed;
}

bool HexagonOptAddrMode::updateAddUses(MachineInstr *AddMI,
                                       MachineInstr *UseMI) {
  const MachineOperand ImmOp = AddMI->getOperand(2);
  const MachineOperand AddRegOp = AddMI->getOperand(1);
  Register NewReg = AddRegOp.getReg();

  MachineOperand &BaseOp = UseMI->getOperand(getBaseOpPosition(UseMI));
  MachineOperand &OffsetOp = UseMI->getOperand(getOffsetOpPosition(UseMI));
  BaseOp.setReg(NewReg);
  BaseOp.setIsUndef(AddRegOp.isUndef());
  BaseOp.setImplicit(AddRegOp.isImplicit());
  OffsetOp.setImm(ImmOp.getImm() + OffsetOp.getImm());
  MRI->clearKillFlags(NewReg);

  return true;
}

bool HexagonOptAddrMode::analyzeUses(unsigned tfrDefR,
                                     const NodeList &UNodeList,
                                     InstrEvalMap &InstrEvalResult,
                                     short &SizeInc) {
  bool KeepTfr = false;
  bool HasRepInstr = false;
  InstrEvalResult.clear();

  for (auto I = UNodeList.rbegin(), E = UNodeList.rend(); I != E; ++I) {
    bool CanBeReplaced = false;
    NodeAddr<UseNode *> UN = *I;
    NodeAddr<StmtNode *> SN = UN.Addr->getOwner(*DFG);
    MachineInstr &MI = *SN.Addr->getCode();
    const MCInstrDesc &MID = MI.getDesc();
    if ((MID.mayLoad() || MID.mayStore())) {
      if (!hasRepForm(MI, tfrDefR)) {
        KeepTfr = true;
        continue;
      }
      SizeInc++;
      CanBeReplaced = true;
    } else if (MI.getOpcode() == Hexagon::S2_addasl_rrri) {
      NodeList AddaslUseList;

      LLVM_DEBUG(dbgs() << "\nGetting ReachedUses for === " << MI << "\n");
      getAllRealUses(SN, AddaslUseList);
      // Process phi nodes.
      if (allValidCandidates(SN, AddaslUseList) &&
          canRemoveAddasl(SN, MI, AddaslUseList)) {
        SizeInc += AddaslUseList.size();
        SizeInc -= 1; // Reduce size by 1 as addasl itself can be removed.
        CanBeReplaced = true;
      } else
        SizeInc++;
    } else
      // Currently, only load/store and addasl are handled.
      // Some other instructions to consider -
      // A2_add -> A2_addi
      // M4_mpyrr_addr -> M4_mpyrr_addi
      KeepTfr = true;

    InstrEvalResult[&MI] = CanBeReplaced;
    HasRepInstr |= CanBeReplaced;
  }

  // Reduce total size by 2 if original tfr can be deleted.
  if (!KeepTfr)
    SizeInc -= 2;

  return HasRepInstr;
}

bool HexagonOptAddrMode::changeLoad(MachineInstr *OldMI, MachineOperand ImmOp,
                                    unsigned ImmOpNum) {
  bool Changed = false;
  MachineBasicBlock *BB = OldMI->getParent();
  auto UsePos = MachineBasicBlock::iterator(OldMI);
  MachineBasicBlock::instr_iterator InsertPt = UsePos.getInstrIterator();
  ++InsertPt;
  unsigned OpStart;
  unsigned OpEnd = OldMI->getNumOperands();
  MachineInstrBuilder MIB;

  if (ImmOpNum == 1) {
    if (HII->getAddrMode(*OldMI) == HexagonII::BaseRegOffset) {
      short NewOpCode = HII->changeAddrMode_rr_ur(*OldMI);
      assert(NewOpCode >= 0 && "Invalid New opcode\n");
      MIB = BuildMI(*BB, InsertPt, OldMI->getDebugLoc(), HII->get(NewOpCode));
      MIB.add(OldMI->getOperand(0));
      MIB.add(OldMI->getOperand(2));
      MIB.add(OldMI->getOperand(3));
      MIB.add(ImmOp);
      OpStart = 4;
      Changed = true;
    } else if (HII->getAddrMode(*OldMI) == HexagonII::BaseImmOffset &&
               OldMI->getOperand(2).isImm()) {
      short NewOpCode = HII->changeAddrMode_io_abs(*OldMI);
      assert(NewOpCode >= 0 && "Invalid New opcode\n");
      MIB = BuildMI(*BB, InsertPt, OldMI->getDebugLoc(), HII->get(NewOpCode))
                .add(OldMI->getOperand(0));
      const GlobalValue *GV = ImmOp.getGlobal();
      int64_t Offset = ImmOp.getOffset() + OldMI->getOperand(2).getImm();

      MIB.addGlobalAddress(GV, Offset, ImmOp.getTargetFlags());
      OpStart = 3;
      Changed = true;
    } else
      Changed = false;

    LLVM_DEBUG(dbgs() << "[Changing]: " << *OldMI << "\n");
    LLVM_DEBUG(dbgs() << "[TO]: " << *MIB << "\n");
  } else if (ImmOpNum == 2) {
    if (OldMI->getOperand(3).isImm() && OldMI->getOperand(3).getImm() == 0) {
      short NewOpCode = HII->changeAddrMode_rr_io(*OldMI);
      assert(NewOpCode >= 0 && "Invalid New opcode\n");
      MIB = BuildMI(*BB, InsertPt, OldMI->getDebugLoc(), HII->get(NewOpCode));
      MIB.add(OldMI->getOperand(0));
      MIB.add(OldMI->getOperand(1));
      MIB.add(ImmOp);
      OpStart = 4;
      Changed = true;
      LLVM_DEBUG(dbgs() << "[Changing]: " << *OldMI << "\n");
      LLVM_DEBUG(dbgs() << "[TO]: " << *MIB << "\n");
    }
  }

  if (Changed)
    for (unsigned i = OpStart; i < OpEnd; ++i)
      MIB.add(OldMI->getOperand(i));

  return Changed;
}

bool HexagonOptAddrMode::changeStore(MachineInstr *OldMI, MachineOperand ImmOp,
                                     unsigned ImmOpNum) {
  bool Changed = false;
  unsigned OpStart = 0;
  unsigned OpEnd = OldMI->getNumOperands();
  MachineBasicBlock *BB = OldMI->getParent();
  auto UsePos = MachineBasicBlock::iterator(OldMI);
  MachineBasicBlock::instr_iterator InsertPt = UsePos.getInstrIterator();
  ++InsertPt;
  MachineInstrBuilder MIB;
  if (ImmOpNum == 0) {
    if (HII->getAddrMode(*OldMI) == HexagonII::BaseRegOffset) {
      short NewOpCode = HII->changeAddrMode_rr_ur(*OldMI);
      assert(NewOpCode >= 0 && "Invalid New opcode\n");
      MIB = BuildMI(*BB, InsertPt, OldMI->getDebugLoc(), HII->get(NewOpCode));
      MIB.add(OldMI->getOperand(1));
      MIB.add(OldMI->getOperand(2));
      MIB.add(ImmOp);
      MIB.add(OldMI->getOperand(3));
      OpStart = 4;
      Changed = true;
    } else if (HII->getAddrMode(*OldMI) == HexagonII::BaseImmOffset) {
      short NewOpCode = HII->changeAddrMode_io_abs(*OldMI);
      assert(NewOpCode >= 0 && "Invalid New opcode\n");
      MIB = BuildMI(*BB, InsertPt, OldMI->getDebugLoc(), HII->get(NewOpCode));
      const GlobalValue *GV = ImmOp.getGlobal();
      int64_t Offset = ImmOp.getOffset() + OldMI->getOperand(1).getImm();
      MIB.addGlobalAddress(GV, Offset, ImmOp.getTargetFlags());
      MIB.add(OldMI->getOperand(2));
      OpStart = 3;
      Changed = true;
    }
  } else if (ImmOpNum == 1 && OldMI->getOperand(2).getImm() == 0) {
    short NewOpCode = HII->changeAddrMode_rr_io(*OldMI);
    assert(NewOpCode >= 0 && "Invalid New opcode\n");
    MIB = BuildMI(*BB, InsertPt, OldMI->getDebugLoc(), HII->get(NewOpCode));
    MIB.add(OldMI->getOperand(0));
    MIB.add(ImmOp);
    OpStart = 3;
    Changed = true;
  }
  if (Changed) {
    LLVM_DEBUG(dbgs() << "[Changing]: " << *OldMI << "\n");
    LLVM_DEBUG(dbgs() << "[TO]: " << *MIB << "\n");

    for (unsigned i = OpStart; i < OpEnd; ++i)
      MIB.add(OldMI->getOperand(i));
  }

  return Changed;
}

short HexagonOptAddrMode::getBaseWithLongOffset(const MachineInstr &MI) const {
  if (HII->getAddrMode(MI) == HexagonII::BaseImmOffset) {
    short TempOpCode = HII->changeAddrMode_io_rr(MI);
    return HII->changeAddrMode_rr_ur(TempOpCode);
  }
  return HII->changeAddrMode_rr_ur(MI);
}

bool HexagonOptAddrMode::changeAddAsl(NodeAddr<UseNode *> AddAslUN,
                                      MachineInstr *AddAslMI,
                                      const MachineOperand &ImmOp,
                                      unsigned ImmOpNum) {
  NodeAddr<StmtNode *> SA = AddAslUN.Addr->getOwner(*DFG);

  LLVM_DEBUG(dbgs() << "Processing addasl :" << *AddAslMI << "\n");

  NodeList UNodeList;
  getAllRealUses(SA, UNodeList);

  for (auto I = UNodeList.rbegin(), E = UNodeList.rend(); I != E; ++I) {
    NodeAddr<UseNode *> UseUN = *I;
    assert(!(UseUN.Addr->getFlags() & NodeAttrs::PhiRef) &&
           "Can't transform this 'AddAsl' instruction!");

    NodeAddr<StmtNode *> UseIA = UseUN.Addr->getOwner(*DFG);
    LLVM_DEBUG(dbgs() << "[InstrNode]: "
                      << Print<NodeAddr<InstrNode *>>(UseIA, *DFG) << "\n");
    MachineInstr *UseMI = UseIA.Addr->getCode();
    LLVM_DEBUG(dbgs() << "[MI <" << printMBBReference(*UseMI->getParent())
                      << ">]: " << *UseMI << "\n");
    const MCInstrDesc &UseMID = UseMI->getDesc();
    assert(HII->getAddrMode(*UseMI) == HexagonII::BaseImmOffset);

    auto UsePos = MachineBasicBlock::iterator(UseMI);
    MachineBasicBlock::instr_iterator InsertPt = UsePos.getInstrIterator();
    short NewOpCode = getBaseWithLongOffset(*UseMI);
    assert(NewOpCode >= 0 && "Invalid New opcode\n");

    unsigned OpStart;
    unsigned OpEnd = UseMI->getNumOperands();

    MachineBasicBlock *BB = UseMI->getParent();
    MachineInstrBuilder MIB =
        BuildMI(*BB, InsertPt, UseMI->getDebugLoc(), HII->get(NewOpCode));
    // change mem(Rs + # ) -> mem(Rt << # + ##)
    if (UseMID.mayLoad()) {
      MIB.add(UseMI->getOperand(0));
      MIB.add(AddAslMI->getOperand(2));
      MIB.add(AddAslMI->getOperand(3));
      const GlobalValue *GV = ImmOp.getGlobal();
      MIB.addGlobalAddress(GV, UseMI->getOperand(2).getImm()+ImmOp.getOffset(),
                           ImmOp.getTargetFlags());
      OpStart = 3;
    } else if (UseMID.mayStore()) {
      MIB.add(AddAslMI->getOperand(2));
      MIB.add(AddAslMI->getOperand(3));
      const GlobalValue *GV = ImmOp.getGlobal();
      MIB.addGlobalAddress(GV, UseMI->getOperand(1).getImm()+ImmOp.getOffset(),
                           ImmOp.getTargetFlags());
      MIB.add(UseMI->getOperand(2));
      OpStart = 3;
    } else
      llvm_unreachable("Unhandled instruction");

    for (unsigned i = OpStart; i < OpEnd; ++i)
      MIB.add(UseMI->getOperand(i));
    Deleted.insert(UseMI);
  }

  return true;
}

bool HexagonOptAddrMode::xformUseMI(MachineInstr *TfrMI, MachineInstr *UseMI,
                                    NodeAddr<UseNode *> UseN,
                                    unsigned UseMOnum) {
  const MachineOperand ImmOp = TfrMI->getOperand(1);
  const MCInstrDesc &MID = UseMI->getDesc();
  unsigned Changed = false;
  if (MID.mayLoad())
    Changed = changeLoad(UseMI, ImmOp, UseMOnum);
  else if (MID.mayStore())
    Changed = changeStore(UseMI, ImmOp, UseMOnum);
  else if (UseMI->getOpcode() == Hexagon::S2_addasl_rrri)
    Changed = changeAddAsl(UseN, UseMI, ImmOp, UseMOnum);

  if (Changed)
    Deleted.insert(UseMI);

  return Changed;
}

bool HexagonOptAddrMode::processBlock(NodeAddr<BlockNode *> BA) {
  bool Changed = false;

  for (auto IA : BA.Addr->members(*DFG)) {
    if (!DFG->IsCode<NodeAttrs::Stmt>(IA))
      continue;

    NodeAddr<StmtNode *> SA = IA;
    MachineInstr *MI = SA.Addr->getCode();
    if ((MI->getOpcode() != Hexagon::A2_tfrsi ||
         !MI->getOperand(1).isGlobal()) &&
        (MI->getOpcode() != Hexagon::A2_addi ||
         !MI->getOperand(2).isImm() || HII->isConstExtended(*MI)))
    continue;

    LLVM_DEBUG(dbgs() << "[Analyzing " << HII->getName(MI->getOpcode())
                      << "]: " << *MI << "\n\t[InstrNode]: "
                      << Print<NodeAddr<InstrNode *>>(IA, *DFG) << '\n');

    if (MI->getOpcode() == Hexagon::A2_addi)
      Changed |= processAddBases(SA, MI);
    NodeList UNodeList;
    getAllRealUses(SA, UNodeList);

    if (!allValidCandidates(SA, UNodeList))
      continue;

    // Analyze all uses of 'add'. If the output of 'add' is used as an address
    // in the base+immediate addressing mode load/store instructions, see if
    // they can be updated to use the immediate value as an offet. Thus,
    // providing us the opportunity to eliminate 'add'.
    // Ex: Rx= add(Rt,#12)
    //     memw(Rx+#0) = Rs
    // This can be replaced with memw(Rt+#12) = Rs
    //
    // This transformation is only performed if all uses can be updated and
    // the offset isn't required to be constant extended.
    if (MI->getOpcode() == Hexagon::A2_addi) {
      Changed |= processAddUses(SA, MI, UNodeList);
      continue;
    }

    short SizeInc = 0;
    Register DefR = MI->getOperand(0).getReg();
    InstrEvalMap InstrEvalResult;

    // Analyze all uses and calculate increase in size. Perform the optimization
    // only if there is no increase in size.
    if (!analyzeUses(DefR, UNodeList, InstrEvalResult, SizeInc))
      continue;
    if (SizeInc > CodeGrowthLimit)
      continue;

    bool KeepTfr = false;

    LLVM_DEBUG(dbgs() << "\t[Total reached uses] : " << UNodeList.size()
                      << "\n");
    LLVM_DEBUG(dbgs() << "\t[Processing Reached Uses] ===\n");
    for (auto I = UNodeList.rbegin(), E = UNodeList.rend(); I != E; ++I) {
      NodeAddr<UseNode *> UseN = *I;
      assert(!(UseN.Addr->getFlags() & NodeAttrs::PhiRef) &&
             "Found a PhiRef node as a real reached use!!");

      NodeAddr<StmtNode *> OwnerN = UseN.Addr->getOwner(*DFG);
      MachineInstr *UseMI = OwnerN.Addr->getCode();
      LLVM_DEBUG(dbgs() << "\t\t[MI <" << printMBBReference(*UseMI->getParent())
                        << ">]: " << *UseMI << "\n");

      int UseMOnum = -1;
      unsigned NumOperands = UseMI->getNumOperands();
      for (unsigned j = 0; j < NumOperands - 1; ++j) {
        const MachineOperand &op = UseMI->getOperand(j);
        if (op.isReg() && op.isUse() && DefR == op.getReg())
          UseMOnum = j;
      }
      // It is possible that the register will not be found in any operand.
      // This could happen, for example, when DefR = R4, but the used
      // register is D2.

      // Change UseMI if replacement is possible. If any replacement failed,
      // or wasn't attempted, make sure to keep the TFR.
      bool Xformed = false;
      if (UseMOnum >= 0 && InstrEvalResult[UseMI])
        Xformed = xformUseMI(MI, UseMI, UseN, UseMOnum);
      Changed |=  Xformed;
      KeepTfr |= !Xformed;
    }
    if (!KeepTfr)
      Deleted.insert(MI);
  }
  return Changed;
}

bool HexagonOptAddrMode::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  // Perform RDF optimizations only if number of basic blocks in the
  // function is less than the limit
  if (MF.size() > RDFFuncBlockLimit) {
    LLVM_DEBUG(dbgs() << "Skipping " << getPassName()
                      << ": too many basic blocks\n");
    return false;
  }

  bool Changed = false;
  auto &HST = MF.getSubtarget<HexagonSubtarget>();
  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  HII = HST.getInstrInfo();
  HRI = HST.getRegisterInfo();
  const auto &MDF = getAnalysis<MachineDominanceFrontier>();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  DataFlowGraph G(MF, *HII, *HRI, *MDT, MDF);
  // Need to keep dead phis because we can propagate uses of registers into
  // nodes dominated by those would-be phis.
  G.build(BuildOptions::KeepDeadPhis);
  DFG = &G;

  Liveness L(*MRI, *DFG);
  L.computePhiInfo();
  LV = &L;

  Deleted.clear();
  ProcessedAddiInsts.clear();
  NodeAddr<FuncNode *> FA = DFG->getFunc();
  LLVM_DEBUG(dbgs() << "==== [RefMap#]=====:\n "
                    << Print<NodeAddr<FuncNode *>>(FA, *DFG) << "\n");

  for (NodeAddr<BlockNode *> BA : FA.Addr->members(*DFG))
    Changed |= processBlock(BA);

  for (auto *MI : Deleted)
    MI->eraseFromParent();

  if (Changed) {
    G.build();
    L.computeLiveIns();
    L.resetLiveIns();
    L.resetKills();
  }

  return Changed;
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createHexagonOptAddrMode() {
  return new HexagonOptAddrMode();
}
