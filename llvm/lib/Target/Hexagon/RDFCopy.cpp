//===--- RDFCopy.cpp ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Simplistic RDF-based copy propagation.

#include "RDFCopy.h"
#include "RDFGraph.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/CommandLine.h"

#include <atomic>

#ifndef NDEBUG
static cl::opt<unsigned> CpLimit("rdf-cp-limit", cl::init(0), cl::Hidden);
static unsigned CpCount = 0;
#endif

using namespace llvm;
using namespace rdf;

void CopyPropagation::recordCopy(NodeAddr<StmtNode*> SA, MachineInstr *MI) {
  assert(MI->getOpcode() == TargetOpcode::COPY);
  const MachineOperand &Op0 = MI->getOperand(0), &Op1 = MI->getOperand(1);
  RegisterRef DstR = { Op0.getReg(), Op0.getSubReg() };
  RegisterRef SrcR = { Op1.getReg(), Op1.getSubReg() };
  auto FS = DefM.find(SrcR);
  if (FS == DefM.end() || FS->second.empty())
    return;
  Copies.push_back(SA.Id);
  RDefMap[SrcR][SA.Id] = FS->second.top()->Id;
  // Insert DstR into the map.
  RDefMap[DstR];
}


void CopyPropagation::updateMap(NodeAddr<InstrNode*> IA) {
  RegisterSet RRs;
  for (NodeAddr<RefNode*> RA : IA.Addr->members(DFG))
    RRs.insert(RA.Addr->getRegRef());
  bool Common = false;
  for (auto &R : RDefMap) {
    if (!RRs.count(R.first))
      continue;
    Common = true;
    break;
  }
  if (!Common)
    return;

  for (auto &R : RDefMap) {
    if (!RRs.count(R.first))
      continue;
    auto F = DefM.find(R.first);
    if (F == DefM.end() || F->second.empty())
      continue;
    R.second[IA.Id] = F->second.top()->Id;
  }
}


bool CopyPropagation::scanBlock(MachineBasicBlock *B) {
  bool Changed = false;
  auto BA = DFG.getFunc().Addr->findBlock(B, DFG);
  DFG.markBlock(BA.Id, DefM);

  for (NodeAddr<InstrNode*> IA : BA.Addr->members(DFG)) {
    if (DFG.IsCode<NodeAttrs::Stmt>(IA)) {
      NodeAddr<StmtNode*> SA = IA;
      MachineInstr *MI = SA.Addr->getCode();
      if (MI->isCopy())
        recordCopy(SA, MI);
    }

    updateMap(IA);
    DFG.pushDefs(IA, DefM);
  }

  MachineDomTreeNode *N = MDT.getNode(B);
  for (auto I : *N)
    Changed |= scanBlock(I->getBlock());

  DFG.releaseBlock(BA.Id, DefM);
  return Changed;
}


bool CopyPropagation::run() {
  scanBlock(&DFG.getMF().front());

  if (trace()) {
    dbgs() << "Copies:\n";
    for (auto I : Copies)
      dbgs() << *DFG.addr<StmtNode*>(I).Addr->getCode();
    dbgs() << "\nRDef map:\n";
    for (auto R : RDefMap) {
      dbgs() << Print<RegisterRef>(R.first, DFG) << " -> {";
      for (auto &M : R.second)
        dbgs() << ' ' << Print<NodeId>(M.first, DFG) << ':'
               << Print<NodeId>(M.second, DFG);
      dbgs() << " }\n";
    }
  }

  bool Changed = false;
  NodeSet Deleted;
#ifndef NDEBUG
  bool HasLimit = CpLimit.getNumOccurrences() > 0;
#endif

  for (auto I : Copies) {
#ifndef NDEBUG
    if (HasLimit && CpCount >= CpLimit)
      break;
#endif
    if (Deleted.count(I))
      continue;
    auto SA = DFG.addr<InstrNode*>(I);
    NodeList Ds = SA.Addr->members_if(DFG.IsDef, DFG);
    if (Ds.size() != 1)
      continue;
    NodeAddr<DefNode*> DA = Ds[0];
    RegisterRef DR0 = DA.Addr->getRegRef();
    NodeList Us = SA.Addr->members_if(DFG.IsUse, DFG);
    if (Us.size() != 1)
      continue;
    NodeAddr<UseNode*> UA0 = Us[0];
    RegisterRef UR0 = UA0.Addr->getRegRef();
    NodeId RD0 = UA0.Addr->getReachingDef();

    for (NodeId N = DA.Addr->getReachedUse(), NextN; N; N = NextN) {
      auto UA = DFG.addr<UseNode*>(N);
      NextN = UA.Addr->getSibling();
      uint16_t F = UA.Addr->getFlags();
      if ((F & NodeAttrs::PhiRef) || (F & NodeAttrs::Fixed))
        continue;
      if (UA.Addr->getRegRef() != DR0)
        continue;
      NodeAddr<InstrNode*> IA = UA.Addr->getOwner(DFG);
      assert(DFG.IsCode<NodeAttrs::Stmt>(IA));
      MachineInstr *MI = NodeAddr<StmtNode*>(IA).Addr->getCode();
      if (RDefMap[UR0][IA.Id] != RD0)
        continue;
      MachineOperand &Op = UA.Addr->getOp();
      if (Op.isTied())
        continue;
      if (trace()) {
        dbgs() << "can replace " << Print<RegisterRef>(DR0, DFG)
               << " with " << Print<RegisterRef>(UR0, DFG) << " in "
               << *NodeAddr<StmtNode*>(IA).Addr->getCode();
      }

      Op.setReg(UR0.Reg);
      Op.setSubReg(UR0.Sub);
      Changed = true;
#ifndef NDEBUG
      if (HasLimit && CpCount >= CpLimit)
        break;
      CpCount++;
#endif

      if (MI->isCopy()) {
        MachineOperand &Op0 = MI->getOperand(0), &Op1 = MI->getOperand(1);
        if (Op0.getReg() == Op1.getReg() && Op0.getSubReg() == Op1.getSubReg())
          MI->eraseFromParent();
        Deleted.insert(IA.Id);
      }
    }
  }

  return Changed;
}

