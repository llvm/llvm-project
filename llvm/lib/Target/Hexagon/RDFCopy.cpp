//===--- RDFCopy.cpp ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// RDF-based copy propagation.

#include "RDFCopy.h"
#include "RDFGraph.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"

#ifndef NDEBUG
static cl::opt<unsigned> CpLimit("rdf-cp-limit", cl::init(0), cl::Hidden);
static unsigned CpCount = 0;
#endif

using namespace llvm;
using namespace rdf;

bool CopyPropagation::interpretAsCopy(const MachineInstr *MI, EqualityMap &EM) {
  unsigned Opc = MI->getOpcode();
  switch (Opc) {
    case TargetOpcode::COPY: {
      const MachineOperand &Dst = MI->getOperand(0);
      const MachineOperand &Src = MI->getOperand(1);
      RegisterRef DstR = { Dst.getReg(), Dst.getSubReg() };
      RegisterRef SrcR = { Src.getReg(), Src.getSubReg() };
      if (TargetRegisterInfo::isVirtualRegister(DstR.Reg)) {
        if (!TargetRegisterInfo::isVirtualRegister(SrcR.Reg))
          return false;
        MachineRegisterInfo &MRI = DFG.getMF().getRegInfo();
        if (MRI.getRegClass(DstR.Reg) != MRI.getRegClass(SrcR.Reg))
          return false;
      } else if (TargetRegisterInfo::isPhysicalRegister(DstR.Reg)) {
        if (!TargetRegisterInfo::isPhysicalRegister(SrcR.Reg))
          return false;
        const TargetRegisterInfo &TRI = DFG.getTRI();
        if (TRI.getMinimalPhysRegClass(DstR.Reg) !=
            TRI.getMinimalPhysRegClass(SrcR.Reg))
          return false;
      } else {
        // Copy between some unknown objects.
        return false;
      }
      EM.insert(std::make_pair(DstR, SrcR));
      return true;
    }
    case TargetOpcode::REG_SEQUENCE: {
      const MachineOperand &Dst = MI->getOperand(0);
      RegisterRef DefR = { Dst.getReg(), Dst.getSubReg() };
      SmallVector<TargetInstrInfo::RegSubRegPairAndIdx,2> Inputs;
      const TargetInstrInfo &TII = DFG.getTII();
      if (!TII.getRegSequenceInputs(*MI, 0, Inputs))
        return false;
      for (auto I : Inputs) {
        unsigned S = DFG.getTRI().composeSubRegIndices(DefR.Sub, I.SubIdx);
        RegisterRef DR = { DefR.Reg, S };
        RegisterRef SR = { I.Reg, I.SubReg };
        EM.insert(std::make_pair(DR, SR));
      }
      return true;
    }
  }
  return false;
}


void CopyPropagation::recordCopy(NodeAddr<StmtNode*> SA, EqualityMap &EM) {
  CopyMap.insert(std::make_pair(SA.Id, EM));
  Copies.push_back(SA.Id);

  for (auto I : EM) {
    auto FS = DefM.find(I.second);
    if (FS == DefM.end() || FS->second.empty())
      continue; // Undefined source
    RDefMap[I.second][SA.Id] = FS->second.top()->Id;
    // Insert DstR into the map.
    RDefMap[I.first];
  }
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
      EqualityMap EM;
      if (interpretAsCopy(SA.Addr->getCode(), EM))
        recordCopy(SA, EM);
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
    for (auto I : Copies) {
      dbgs() << "Instr: " << *DFG.addr<StmtNode*>(I).Addr->getCode();
      dbgs() << "   eq: {";
      for (auto J : CopyMap[I])
        dbgs() << ' ' << Print<RegisterRef>(J.first, DFG) << '='
               << Print<RegisterRef>(J.second, DFG);
      dbgs() << " }\n";
    }
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
#ifndef NDEBUG
  bool HasLimit = CpLimit.getNumOccurrences() > 0;
#endif

  for (auto C : Copies) {
#ifndef NDEBUG
    if (HasLimit && CpCount >= CpLimit)
      break;
#endif
    auto SA = DFG.addr<InstrNode*>(C);
    auto FS = CopyMap.find(SA.Id);
    if (FS == CopyMap.end())
      continue;

    EqualityMap &EM = FS->second;
    for (NodeAddr<DefNode*> DA : SA.Addr->members_if(DFG.IsDef, DFG)) {
      RegisterRef DR = DA.Addr->getRegRef();
      auto FR = EM.find(DR);
      if (FR == EM.end())
        continue;
      RegisterRef SR = FR->second;
      if (DR == SR)
        continue;

      auto &RDefSR = RDefMap[SR];
      NodeId RDefSR_SA = RDefSR[SA.Id];

      for (NodeId N = DA.Addr->getReachedUse(), NextN; N; N = NextN) {
        auto UA = DFG.addr<UseNode*>(N);
        NextN = UA.Addr->getSibling();
        uint16_t F = UA.Addr->getFlags();
        if ((F & NodeAttrs::PhiRef) || (F & NodeAttrs::Fixed))
          continue;
        if (UA.Addr->getRegRef() != DR)
          continue;

        NodeAddr<InstrNode*> IA = UA.Addr->getOwner(DFG);
        assert(DFG.IsCode<NodeAttrs::Stmt>(IA));
        if (RDefSR[IA.Id] != RDefSR_SA)
          continue;

        MachineOperand &Op = UA.Addr->getOp();
        if (Op.isTied())
          continue;
        if (trace()) {
          dbgs() << "Can replace " << Print<RegisterRef>(DR, DFG)
                 << " with " << Print<RegisterRef>(SR, DFG) << " in "
                 << *NodeAddr<StmtNode*>(IA).Addr->getCode();
        }

        Op.setReg(SR.Reg);
        Op.setSubReg(SR.Sub);
        DFG.unlinkUse(UA, false);
        UA.Addr->linkToDef(UA.Id, DFG.addr<DefNode*>(RDefSR_SA));

        Changed = true;
  #ifndef NDEBUG
        if (HasLimit && CpCount >= CpLimit)
          break;
        CpCount++;
  #endif

        auto FC = CopyMap.find(IA.Id);
        if (FC != CopyMap.end()) {
          // Update the EM map in the copy's entry.
          auto &M = FC->second;
          for (auto &J : M) {
            if (J.second != DR)
              continue;
            J.second = SR;
            break;
          }
        }
      } // for (N in reached-uses)
    } // for (DA in defs)
  } // for (C in Copies)

  return Changed;
}

