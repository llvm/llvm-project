//===--- HexagonOptAddrMode.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This implements a Hexagon-specific pass to optimize addressing mode for
// load/store instructions.
//===----------------------------------------------------------------------===//

#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "RDFGraph.h"
#include "RDFLiveness.h"
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
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>

#define DEBUG_TYPE "opt-addr-mode"

static cl::opt<int> CodeGrowthLimit("hexagon-amode-growth-limit",
  cl::Hidden, cl::init(0), cl::desc("Code growth limit for address mode "
  "optimization"));

using namespace llvm;
using namespace rdf;

namespace llvm {
  FunctionPass *createHexagonOptAddrMode();
  void initializeHexagonOptAddrModePass(PassRegistry&);
} // end namespace llvm

namespace {

class HexagonOptAddrMode : public MachineFunctionPass {
public:
  static char ID;

  HexagonOptAddrMode()
      : MachineFunctionPass(ID), HII(nullptr), MDT(nullptr), DFG(nullptr),
        LV(nullptr) {}

  StringRef getPassName() const override {
    return "Optimize addressing mode of load/store";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<MachineDominanceFrontier>();
    AU.setPreservesAll();
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  typedef DenseSet<MachineInstr *> MISetType;
  typedef DenseMap<MachineInstr *, bool> InstrEvalMap;
  const HexagonInstrInfo *HII;
  MachineDominatorTree *MDT;
  DataFlowGraph *DFG;
  DataFlowGraph::DefStackMap DefM;
  Liveness *LV;
  MISetType Deleted;

  bool processBlock(NodeAddr<BlockNode *> BA);
  bool xformUseMI(MachineInstr *TfrMI, MachineInstr *UseMI,
                  NodeAddr<UseNode *> UseN, unsigned UseMOnum);
  bool analyzeUses(unsigned DefR, const NodeList &UNodeList,
                   InstrEvalMap &InstrEvalResult, short &SizeInc);
  bool hasRepForm(MachineInstr &MI, unsigned TfrDefR);
  bool canRemoveAddasl(NodeAddr<StmtNode *> AddAslSN, MachineInstr &MI,
                       const NodeList &UNodeList);
  void getAllRealUses(NodeAddr<StmtNode *> SN, NodeList &UNodeList);
  bool allValidCandidates(NodeAddr<StmtNode *> SA, NodeList &UNodeList);
  short getBaseWithLongOffset(const MachineInstr &MI) const;
  bool changeStore(MachineInstr *OldMI, MachineOperand ImmOp,
                   unsigned ImmOpNum);
  bool changeLoad(MachineInstr *OldMI, MachineOperand ImmOp, unsigned ImmOpNum);
  bool changeAddAsl(NodeAddr<UseNode *> AddAslUN, MachineInstr *AddAslMI,
                    const MachineOperand &ImmOp, unsigned ImmOpNum);
};

} // end anonymous namespace

char HexagonOptAddrMode::ID = 0;

INITIALIZE_PASS_BEGIN(HexagonOptAddrMode, "amode-opt",
                      "Optimize addressing mode", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
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
    return (HII->getBaseWithLongOffset(MI) >= 0);
  else if (HII->getAddrMode(MI) == HexagonII::BaseImmOffset)
    // Tranform to absolute addressing mode.
    return (HII->getAbsoluteForm(MI) >= 0);

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

  unsigned OffsetReg = MI.getOperand(2).getReg();
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
      if (Mo.isFI())
        return false;
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
      DEBUG({
        dbgs() << "*** Unable to collect all reaching defs for use ***\n"
               << PrintNode<UseNode*>(UN, *DFG) << '\n'
               << "The program's complexity may exceed the limits.\n";
      });
      return false;
    }
    const auto &ReachingDefs = P.first;
    if (ReachingDefs.size() > 1) {
      DEBUG({
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
    DEBUG(dbgs() << "\t\t[DefNode]: " << Print<NodeAddr<DefNode *>>(DA, *DFG)
                 << "\n");
    RegisterRef DR = DFG->getPRI().normalize(DA.Addr->getRegRef(*DFG));

    auto UseSet = LV->getAllReachedUses(DR, DA);

    for (auto UI : UseSet) {
      NodeAddr<UseNode *> UA = DFG->addr<UseNode *>(UI);
      DEBUG({
        NodeAddr<StmtNode *> TempIA = UA.Addr->getOwner(*DFG);
        dbgs() << "\t\t\t[Reached Use]: "
               << Print<NodeAddr<InstrNode *>>(TempIA, *DFG) << "\n";
      });

      if (UA.Addr->getFlags() & NodeAttrs::PhiRef) {
        NodeAddr<PhiNode *> PA = UA.Addr->getOwner(*DFG);
        NodeId id = PA.Id;
        const Liveness::RefMap &phiUse = LV->getRealUses(id);
        DEBUG(dbgs() << "\t\t\t\tphi real Uses"
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

      DEBUG(dbgs() << "\nGetting ReachedUses for === " << MI << "\n");
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
      short NewOpCode = HII->getBaseWithLongOffset(*OldMI);
      assert(NewOpCode >= 0 && "Invalid New opcode\n");
      MIB = BuildMI(*BB, InsertPt, OldMI->getDebugLoc(), HII->get(NewOpCode));
      MIB.add(OldMI->getOperand(0));
      MIB.add(OldMI->getOperand(2));
      MIB.add(OldMI->getOperand(3));
      MIB.add(ImmOp);
      OpStart = 4;
      Changed = true;
    } else if (HII->getAddrMode(*OldMI) == HexagonII::BaseImmOffset) {
      short NewOpCode = HII->getAbsoluteForm(*OldMI);
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

    DEBUG(dbgs() << "[Changing]: " << *OldMI << "\n");
    DEBUG(dbgs() << "[TO]: " << MIB << "\n");
  } else if (ImmOpNum == 2 && OldMI->getOperand(3).getImm() == 0) {
    short NewOpCode = HII->xformRegToImmOffset(*OldMI);
    assert(NewOpCode >= 0 && "Invalid New opcode\n");
    MIB = BuildMI(*BB, InsertPt, OldMI->getDebugLoc(), HII->get(NewOpCode));
    MIB.add(OldMI->getOperand(0));
    MIB.add(OldMI->getOperand(1));
    MIB.add(ImmOp);
    OpStart = 4;
    Changed = true;
    DEBUG(dbgs() << "[Changing]: " << *OldMI << "\n");
    DEBUG(dbgs() << "[TO]: " << MIB << "\n");
  }

  if (Changed)
    for (unsigned i = OpStart; i < OpEnd; ++i)
      MIB.add(OldMI->getOperand(i));

  return Changed;
}

bool HexagonOptAddrMode::changeStore(MachineInstr *OldMI, MachineOperand ImmOp,
                                     unsigned ImmOpNum) {
  bool Changed = false;
  unsigned OpStart;
  unsigned OpEnd = OldMI->getNumOperands();
  MachineBasicBlock *BB = OldMI->getParent();
  auto UsePos = MachineBasicBlock::iterator(OldMI);
  MachineBasicBlock::instr_iterator InsertPt = UsePos.getInstrIterator();
  ++InsertPt;
  MachineInstrBuilder MIB;
  if (ImmOpNum == 0) {
    if (HII->getAddrMode(*OldMI) == HexagonII::BaseRegOffset) {
      short NewOpCode = HII->getBaseWithLongOffset(*OldMI);
      assert(NewOpCode >= 0 && "Invalid New opcode\n");
      MIB = BuildMI(*BB, InsertPt, OldMI->getDebugLoc(), HII->get(NewOpCode));
      MIB.add(OldMI->getOperand(1));
      MIB.add(OldMI->getOperand(2));
      MIB.add(ImmOp);
      MIB.add(OldMI->getOperand(3));
      OpStart = 4;
    } else if (HII->getAddrMode(*OldMI) == HexagonII::BaseImmOffset) {
      short NewOpCode = HII->getAbsoluteForm(*OldMI);
      assert(NewOpCode >= 0 && "Invalid New opcode\n");
      MIB = BuildMI(*BB, InsertPt, OldMI->getDebugLoc(), HII->get(NewOpCode));
      const GlobalValue *GV = ImmOp.getGlobal();
      int64_t Offset = ImmOp.getOffset() + OldMI->getOperand(1).getImm();
      MIB.addGlobalAddress(GV, Offset, ImmOp.getTargetFlags());
      MIB.add(OldMI->getOperand(2));
      OpStart = 3;
    }
    Changed = true;
    DEBUG(dbgs() << "[Changing]: " << *OldMI << "\n");
    DEBUG(dbgs() << "[TO]: " << MIB << "\n");
  } else if (ImmOpNum == 1 && OldMI->getOperand(2).getImm() == 0) {
    short NewOpCode = HII->xformRegToImmOffset(*OldMI);
    assert(NewOpCode >= 0 && "Invalid New opcode\n");
    MIB = BuildMI(*BB, InsertPt, OldMI->getDebugLoc(), HII->get(NewOpCode));
    MIB.add(OldMI->getOperand(0));
    MIB.add(ImmOp);
    MIB.add(OldMI->getOperand(1));
    OpStart = 2;
    Changed = true;
    DEBUG(dbgs() << "[Changing]: " << *OldMI << "\n");
    DEBUG(dbgs() << "[TO]: " << MIB << "\n");
  }
  if (Changed)
    for (unsigned i = OpStart; i < OpEnd; ++i)
      MIB.add(OldMI->getOperand(i));

  return Changed;
}

short HexagonOptAddrMode::getBaseWithLongOffset(const MachineInstr &MI) const {
  if (HII->getAddrMode(MI) == HexagonII::BaseImmOffset) {
    short TempOpCode = HII->getBaseWithRegOffset(MI);
    return HII->getBaseWithLongOffset(TempOpCode);
  } else
    return HII->getBaseWithLongOffset(MI);
}

bool HexagonOptAddrMode::changeAddAsl(NodeAddr<UseNode *> AddAslUN,
                                      MachineInstr *AddAslMI,
                                      const MachineOperand &ImmOp,
                                      unsigned ImmOpNum) {
  NodeAddr<StmtNode *> SA = AddAslUN.Addr->getOwner(*DFG);

  DEBUG(dbgs() << "Processing addasl :" << *AddAslMI << "\n");

  NodeList UNodeList;
  getAllRealUses(SA, UNodeList);

  for (auto I = UNodeList.rbegin(), E = UNodeList.rend(); I != E; ++I) {
    NodeAddr<UseNode *> UseUN = *I;
    assert(!(UseUN.Addr->getFlags() & NodeAttrs::PhiRef) &&
           "Can't transform this 'AddAsl' instruction!");

    NodeAddr<StmtNode *> UseIA = UseUN.Addr->getOwner(*DFG);
    DEBUG(dbgs() << "[InstrNode]: " << Print<NodeAddr<InstrNode *>>(UseIA, *DFG)
                 << "\n");
    MachineInstr *UseMI = UseIA.Addr->getCode();
    DEBUG(dbgs() << "[MI <BB#" << UseMI->getParent()->getNumber()
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
    if (MI->getOpcode() != Hexagon::A2_tfrsi ||
        !MI->getOperand(1).isGlobal())
      continue;

    DEBUG(dbgs() << "[Analyzing " << HII->getName(MI->getOpcode()) << "]: "
                 << *MI << "\n\t[InstrNode]: "
                 << Print<NodeAddr<InstrNode *>>(IA, *DFG) << '\n');

    NodeList UNodeList;
    getAllRealUses(SA, UNodeList);

    if (!allValidCandidates(SA, UNodeList))
      continue;

    short SizeInc = 0;
    unsigned DefR = MI->getOperand(0).getReg();
    InstrEvalMap InstrEvalResult;

    // Analyze all uses and calculate increase in size. Perform the optimization
    // only if there is no increase in size.
    if (!analyzeUses(DefR, UNodeList, InstrEvalResult, SizeInc))
      continue;
    if (SizeInc > CodeGrowthLimit)
      continue;

    bool KeepTfr = false;

    DEBUG(dbgs() << "\t[Total reached uses] : " << UNodeList.size() << "\n");
    DEBUG(dbgs() << "\t[Processing Reached Uses] ===\n");
    for (auto I = UNodeList.rbegin(), E = UNodeList.rend(); I != E; ++I) {
      NodeAddr<UseNode *> UseN = *I;
      assert(!(UseN.Addr->getFlags() & NodeAttrs::PhiRef) &&
             "Found a PhiRef node as a real reached use!!");

      NodeAddr<StmtNode *> OwnerN = UseN.Addr->getOwner(*DFG);
      MachineInstr *UseMI = OwnerN.Addr->getCode();
      DEBUG(dbgs() << "\t\t[MI <BB#" << UseMI->getParent()->getNumber()
                   << ">]: " << *UseMI << "\n");

      int UseMOnum = -1;
      unsigned NumOperands = UseMI->getNumOperands();
      for (unsigned j = 0; j < NumOperands - 1; ++j) {
        const MachineOperand &op = UseMI->getOperand(j);
        if (op.isReg() && op.isUse() && DefR == op.getReg())
          UseMOnum = j;
      }
      assert(UseMOnum >= 0 && "Invalid reached use!");

      if (InstrEvalResult[UseMI])
        // Change UseMI if replacement is possible.
        Changed |= xformUseMI(MI, UseMI, UseN, UseMOnum);
      else
        KeepTfr = true;
    }
    if (!KeepTfr)
      Deleted.insert(MI);
  }
  return Changed;
}

bool HexagonOptAddrMode::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;

  bool Changed = false;
  auto &HST = MF.getSubtarget<HexagonSubtarget>();
  auto &MRI = MF.getRegInfo();
  HII = HST.getInstrInfo();
  const auto &MDF = getAnalysis<MachineDominanceFrontier>();
  MDT = &getAnalysis<MachineDominatorTree>();
  const auto &TRI = *MF.getSubtarget().getRegisterInfo();
  const TargetOperandInfo TOI(*HII);

  DataFlowGraph G(MF, *HII, TRI, *MDT, MDF, TOI);
  // Need to keep dead phis because we can propagate uses of registers into
  // nodes dominated by those would-be phis.
  G.build(BuildOptions::KeepDeadPhis);
  DFG = &G;

  Liveness L(MRI, *DFG);
  L.computePhiInfo();
  LV = &L;

  Deleted.clear();
  NodeAddr<FuncNode *> FA = DFG->getFunc();
  DEBUG(dbgs() << "==== [RefMap#]=====:\n "
               << Print<NodeAddr<FuncNode *>>(FA, *DFG) << "\n");

  for (NodeAddr<BlockNode *> BA : FA.Addr->members(*DFG))
    Changed |= processBlock(BA);

  for (auto MI : Deleted)
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
