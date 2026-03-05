//===--- AggressiveRDFCopy.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RDF-based aggressive copy propagation.
//
// This optimization extends the standard RDF copy propagation with support for
// super-register and sub-register copy propagation. It determines candidates
// for copy propagation by verifying that both the copy instruction and all
// reached uses have the same reaching definitions for the source register(s).
//
// Key differences:
// 1. Super-register handling: Can propagate copies involving super-registers
//    and their sub-registers (e.g., double-register pairs on Hexagon).
// 2. Sub-register coverage: Verifies that all sub-registers of the source
//    register have consistent reaching definitions before propagating.
// 3. Combine instruction support: Handles A2_combinew instructions that
//    combine two 32-bit registers into a 64-bit register pair.
//
// Algorithm:
// 1. Scan all basic blocks in dominator tree order, maintaining a stack of
//    reaching definitions for each register.
// 2. For each copy instruction:
//    a. Record the copy and its source/destination register mapping.
//    b. Find the reaching definition for the source register at the copy.
//    c. For each use reached by the copy's destination register:
//       - Check if all sub-registers of the source have the same reaching
//         definition at both the copy and the use.
//       - If yes, mark the use as replaceable.
// 3. Replace all marked uses with the source register of the copy.
//
// Example:
//   BB1:
//     R1 = ...                    // Def1
//     D0 = A2_combinew R1, R0     // Copy: D0 = {R1, R0}
//     ... = D0                    // Use of D0
//
// If R1 and R0 have the same reaching definitions at both the copy and the
// use, the use of D0 can be replaced with the original source registers.
// D0 is super-register corresponding to R1:0.
//
//===----------------------------------------------------------------------===//

#include "AggressiveRDFCopy.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RDFGraph.h"
#include "llvm/CodeGen/RDFLiveness.h"
#include "llvm/CodeGen/RDFRegisters.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <utility>

using namespace llvm;
using namespace rdf;

#ifndef NDEBUG
extern cl::opt<unsigned> RDFCpLimit;
static unsigned RDFCpCount = 0;
#endif

// Record destination and source registers in EqualityMap
// if this is a copy instruction
bool AggressiveCopyPropagation::interpretAsCopy(const MachineInstr *MI,
                                                EqualityMap &EM) {
  unsigned Opc = MI->getOpcode();
  switch (Opc) {
  case TargetOpcode::COPY: {
    const MachineOperand &Dst = MI->getOperand(0);
    const MachineOperand &Src = MI->getOperand(1);
    RegisterRef DstR = DFG.makeRegRef(Dst.getReg(), Dst.getSubReg());
    RegisterRef SrcR = DFG.makeRegRef(Src.getReg(), Src.getSubReg());
    assert(Register::isPhysicalRegister(DstR.Id));
    assert(Register::isPhysicalRegister(SrcR.Id));
    if (HRI.isFakeReg(DstR.Id) || HRI.isFakeReg(SrcR.Id))
      return false;
    if (TRI.getMinimalPhysRegClass(DstR.Id) !=
        TRI.getMinimalPhysRegClass(SrcR.Id))
      return false;
    if (!DFG.isTracked(SrcR) || !DFG.isTracked(DstR))
      return false;
    EM.insert(std::make_pair(DstR, SrcR));
    return true;
  }
  case TargetOpcode::REG_SEQUENCE:
    llvm_unreachable("Unexpected REG_SEQUENCE");
  }
  return false;
}

// Track instructions determined to be copies along with their uses.
// The register, sub-register copy pairs are given by EqualityMap
// Find the reaching def from DefM stack for source (LHS) registers in each
// copy. ReachedUseToCopyMap stores each reached use (UseNode) of a copy along
// with the copy DefNode and source register
void AggressiveCopyPropagation::recordCopy(NodeAddr<StmtNode *> SA,
                                           EqualityMap &EM) {
  if (trace())
    CopyMap.insert(std::make_pair(SA.Id, EM));

  // Find and store reaching def for each source register
  // EqualityMap should also contain subregs
  for (auto I : EM) {
    if (PRI.equal_to(I.first, I.second))
      continue;
    NodeId RDefId = 0;
    auto FS = DefM.find(I.second.Id);
    if (FS != DefM.end() && !FS->second.empty()) {
      auto Def = FS->second.top()->Addr->getRegRef(DFG);
      // Avoid adding subreg as reaching def for superreg
      if (Register::isPhysicalRegister(Def.Id) &&
          TRI.isSuperRegister(Def.Id, I.second.Id))
        continue;
      RDefId = FS->second.top()->Id;
    }
    RDefMap[I.second][SA.Id] = RDefId;
  }
  for (NodeAddr<DefNode *> DA : SA.Addr->members_if(DFG.IsDef, DFG)) {
    RegisterRef DR = DA.Addr->getRegRef(DFG);
    auto FR = EM.find(DR);
    if (FR == EM.end())
      continue;
    // Iterate over DR and its subregisters
    // if present in EqualityMap, find its reached uses
    for (MCPhysReg SubDReg : TRI.subregs_inclusive(DR.Id)) {
      auto SubDR = DFG.makeRegRef(SubDReg, 0);
      auto FR = EM.find(SubDR);
      if (FR == EM.end())
        continue;
      RegisterRef SR = FR->second;
      // Redundant copy
      if (PRI.equal_to(SubDR, SR))
        continue;
      for (NodeId N = DA.Addr->getReachedUse(), NextN; N; N = NextN) {
        auto UA = DFG.addr<UseNode *>(N);
        NextN = UA.Addr->getSibling();
        uint16_t F = UA.Addr->getFlags();
        // Skip phi node uses
        // Skip shadow uses. When shadow nodes are present, the register has
        // multiple reaching defs.
        if ((F & NodeAttrs::PhiRef) || (F & NodeAttrs::Fixed) ||
            (F & NodeAttrs::Shadow))
          continue;
        if (!PRI.equal_to(UA.Addr->getRegRef(DFG), SubDR))
          continue;
        MachineOperand &Op = UA.Addr->getOp();
        // Skip operand if def and use of a register happens in same instruction
        if (Op.isTied())
          continue;
        if (ReachedUseToCopyMap.find(UA.Id) != ReachedUseToCopyMap.end())
          llvm_unreachable("Multiple copy instructions reach this use");
        ReachedUseToCopyMap.insert(
            std::make_pair(UA.Id, std::make_pair(DA, SR)));
      }
    }
  }
}

// Now that we can obtain reaching def for uses from DefM,
// check that reaching defs for source register and subregisters at the use
// instruction, are the same as reaching defs for the copy. Uses that satisfy
// this check can be replaced with the source registers of the copy.
void AggressiveCopyPropagation::recordReplacableUses(NodeAddr<InstrNode *> IA) {
  for (NodeAddr<UseNode *> UA : IA.Addr->members_if(DFG.IsUse, DFG)) {
    // Check if any uses of the instruction are reached by a copy
    auto CopyUseIt = ReachedUseToCopyMap.find(UA.Id);
    if (CopyUseIt == ReachedUseToCopyMap.end())
      continue;
    [[maybe_unused]] auto UseReg = UA.Addr->getRegRef(DFG);
    auto DA = CopyUseIt->second.first;
    auto SR = CopyUseIt->second.second;
    [[maybe_unused]] auto DefReg = DA.Addr->getRegRef(DFG);
    assert(PRI.equal_to(DefReg, UseReg));
    NodeAddr<InstrNode *> DefI = DA.Addr->getOwner(DFG);
    // Aggr of subregs that have same reaching def (at IA) as DefI
    RegisterAggr RRs(PRI);
    // Registers that need to be added as use nodes in updated IA
    SmallVector<RegisterRef, 4> UseRefs;
    for (MCPhysReg S : TRI.subregs_inclusive(SR.Id)) {
      auto SRef = DFG.makeRegRef(S, 0);
      NodeId RDefId = 0;
      // If there is no reaching def for SRef at DefI,
      // do not check if SRef can be propagated
      auto RDefIt = RDefMap.find(SRef);
      if (RDefIt == RDefMap.end())
        continue;
      auto DefIIt = RDefIt->second.find(DefI.Id);
      if (DefIIt == RDefIt->second.end())
        continue;
      // If we already have reaching def for SRef at IA,
      // use it instead of searching DefM.
      auto IAIt = RDefIt->second.find(IA.Id);
      if (IAIt != RDefIt->second.end()) {
        RDefId = IAIt->second;
      } else {
        auto F = DefM.find(S);
        if (F != DefM.end() && !F->second.empty()) {
          auto Def = F->second.top()->Addr->getRegRef(DFG);
          // Avoid adding subreg as reaching def for superreg
          if (Register::isPhysicalRegister(Def.Id) &&
              TRI.isSuperRegister(Def.Id, S))
            continue;
          RDefId = F->second.top()->Id;
        }
      }
      // If reaching def for SRef at DefI is not same as IA,
      // SRef can not propagated to IA.
      if (DefIIt->second != RDefId)
        continue;
      RRs.insert(SRef);
      UseRefs.push_back(SRef);
      RDefIt->second[IA.Id] = RDefId;
      // If registers or sub-registers that can be propagated cover SR,
      // the use node is a candidate for copy propagation
      if (RRs.hasCoverOf(SR))
        break;
    }
    // Use node can be replaced with new use nodes created from UseRefs
    if (RRs.hasCoverOf(SR))
      ReplacableUses.push_back(std::make_pair(UA, UseRefs));
  }
}

// Recursively process all children in the dominator tree.
// Find copy instructions and reached uses that are candidates for propagation
void AggressiveCopyPropagation::scanBlock(MachineBasicBlock *B) {
  NodeAddr<BlockNode *> BA = DFG.findBlock(B);
  DFG.markBlock(BA.Id, DefM);

  for (NodeAddr<InstrNode *> IA : BA.Addr->members(DFG)) {
    if (DFG.IsCode<NodeAttrs::Stmt>(IA)) {
      NodeAddr<StmtNode *> SA = IA;
      EqualityMap EM(RegisterRefLess(DFG.getPRI()));
      if (interpretAsCopy(SA.Addr->getCode(), EM))
        recordCopy(SA, EM);
      recordReplacableUses(IA);
    }
    DFG.pushAllDefs(IA, DefM);
  }

  MachineDomTreeNode *N = MDT.getNode(B);
  for (auto *I : *N)
    scanBlock(I->getBlock());

  DFG.releaseBlock(BA.Id, DefM);
  return;
}

bool AggressiveCopyPropagation::run() {
  scanBlock(&DFG.getMF().front());

  if (trace()) {
    dbgs() << "Copies:\n";
    for (auto &C : CopyMap) {
      dbgs() << "Instr: " << *DFG.addr<StmtNode *>(C.first).Addr->getCode();
      dbgs() << "   eq: {";
      for (auto J : C.second)
        dbgs() << ' ' << Print<RegisterRef>(J.first, DFG) << '='
               << Print<RegisterRef>(J.second, DFG);
      dbgs() << " }\n";
    }
    dbgs() << "\nCopy def-use:\n";
    for (auto &U : ReachedUseToCopyMap) {
      auto DA = U.second.first;
      auto DefI = DA.Addr->getOwner(DFG);
      auto UseI = DFG.addr<UseNode *>(U.first).Addr->getOwner(DFG);
      dbgs() << "Copy def: " << *DFG.addr<StmtNode *>(DefI.Id).Addr->getCode();
      dbgs() << "Copy use: " << *DFG.addr<StmtNode *>(UseI.Id).Addr->getCode();
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
  bool HasLimit = RDFCpLimit.getNumOccurrences() > 0;
#endif

  auto MinPhysReg = [this](RegisterRef RR) -> unsigned {
    const TargetRegisterClass &RC = *TRI.getMinimalPhysRegClass(RR.Id);
    if ((RC.LaneMask & RR.Mask) == RC.LaneMask)
      return RR.Id;
    for (MCSubRegIndexIterator S(RR.Id, &TRI); S.isValid(); ++S)
      if (RR.Mask == TRI.getSubRegIndexLaneMask(S.getSubRegIndex()))
        return S.getSubReg();
    llvm_unreachable("Should have found a register");
    return 0;
  };

  // Iterate over all candidate uses found and replace with source register of
  // copy
  for (auto P : ReplacableUses) {
#ifndef NDEBUG
    if (HasLimit && RDFCpCount >= RDFCpLimit)
      break;
#endif
    NodeAddr<UseNode *> UA = P.first;
    SmallVector<RegisterRef, 4> UseRefs = P.second;

    // UseRefs should never be empty if RRs.hasCoverOf(SR) was true
    assert(!UseRefs.empty() &&
           "UseRefs should not be empty for replaceable use");

    auto IA = UA.Addr->getOwner(DFG);
    auto DR = UA.Addr->getRegRef(DFG);
    auto SR = ReachedUseToCopyMap[UA.Id].second;
    if (HRI.isFakeReg(SR.Id))
      continue;

    if (trace()) {
      dbgs() << "Can replace " << Print<RegisterRef>(DR, DFG) << " with "
             << Print<RegisterRef>(SR, DFG) << " in "
             << *NodeAddr<StmtNode *>(IA).Addr->getCode();
    }

    // Update existing use node to use the source register
    MachineOperand &Op = UA.Addr->getOp();
    unsigned NewReg = MinPhysReg(SR);
    Op.setReg(NewReg);
    Op.setSubReg(0);
    DFG.unlinkUse(UA, false);
    bool firstUseNode = true;

    for (auto UR : UseRefs) {
      // If we have more than one use (such as multiple subregs),
      // add a new shadow use node
      if (!firstUseNode) {
        UA.Addr->setFlags(UA.Addr->getFlags() | NodeAttrs::Shadow);
        UA = DFG.getNextShadow(IA, UA, true);
      }
      if (RDefMap[UR][IA.Id] != 0) {
        UA.Addr->linkToDef(UA.Id, DFG.addr<DefNode *>(RDefMap[UR][IA.Id]));
      } else {
        // No reaching def present
        UA.Addr->setReachingDef(0);
        UA.Addr->setSibling(0);
      }
      firstUseNode = false;
    }

    Changed = true;
#ifndef NDEBUG
    if (HasLimit && RDFCpCount >= RDFCpLimit)
      break;
    RDFCpCount++;
#endif

  } // for (UA in replacable uses)

  return Changed;
}
