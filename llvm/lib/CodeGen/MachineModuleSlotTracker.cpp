//===-- llvm/CodeGen/MachineModuleInfo.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineModuleSlotTracker.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;

void MachineModuleSlotTracker::collectMachineFunctionMetadata(
    SmallVectorImpl<const MDNode *> &Metadata, const MachineFunction &MF,
    SmallVectorImpl<const MDNode *> *DebugLocations) const {
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB.instrs()) {
      if (DebugLocations)
        if (DebugLoc DL = MI.getDebugLoc())
          DebugLocations->push_back(DL.getAsMDNode());

      if (MDNode *N = MI.getHeapAllocMarker())
        Metadata.push_back(N);
      if (MDNode *N = MI.getPCSections())
        Metadata.push_back(N);
      if (MDNode *N = MI.getMMRAMetadata())
        Metadata.push_back(N);

      for (const MachineOperand &MO : MI.operands())
        if (MO.isMetadata())
          Metadata.push_back(MO.getMetadata());

      for (const MachineMemOperand *MMO : MI.memoperands()) {
        AAMDNodes AAInfo = MMO->getAAInfo();
        if (AAInfo.TBAA)
          Metadata.push_back(AAInfo.TBAA);
        if (AAInfo.TBAAStruct)
          Metadata.push_back(AAInfo.TBAAStruct);
        if (AAInfo.Scope)
          Metadata.push_back(AAInfo.Scope);
        if (AAInfo.NoAlias)
          Metadata.push_back(AAInfo.NoAlias);
        if (AAInfo.NoAliasAddrSpace)
          Metadata.push_back(AAInfo.NoAliasAddrSpace);
        if (const MDNode *N = MMO->getRanges())
          Metadata.push_back(N);
        if (const MDNode *N = MMO->getMemCacheHint())
          Metadata.push_back(N);
      }
    }

  for (const MachineFunction::VariableDbgInfo &DebugVar :
       MF.getVariableDbgInfo()) {
    Metadata.push_back(DebugVar.Var);
  }
}

void MachineModuleSlotTracker::collectMachineMDNodes(
    MachineMDNodeListType &L) const {
  L.insert(L.end(), MachineMDNodes.begin(), MachineMDNodes.end());
}

void MachineModuleSlotTracker::renumberMetadataForAssembly() {
  if (!TheMF)
    return;

  SmallVector<const MDNode *, 16> Metadata;
  collectMachineFunctionMetadata(Metadata, *TheMF);
  MachineMDNodes.clear();
  ModuleSlotTracker::renumberMetadataForAssembly(Metadata, &MachineMDNodes);
}

MachineModuleSlotTracker::MachineModuleSlotTracker(MFGetterFnT Fn,
                                                   const MachineFunction *MF)
    : ModuleSlotTracker(MF->getFunction().getParent()),
      TheMF(Fn(MF->getFunction())) {
  if (!TheMF)
    return;

  SmallVector<const MDNode *, 16> Metadata;
  SmallVector<const MDNode *, 16> DebugLocations;
  collectMachineFunctionMetadata(Metadata, *TheMF, &DebugLocations);
  collectAdditionalMetadata(Metadata, MachineMDNodes);

  if (DebugLocations.empty())
    return;

  MachineMDNodeListType DebugMetadataNodes;
  collectAdditionalMetadata(DebugLocations, DebugMetadataNodes);
  SmallPtrSet<const MDNode *, 16> MachineMetadata;
  for (const auto &Entry : MachineMDNodes)
    MachineMetadata.insert(Entry.second);

  for (const auto &Entry : DebugMetadataNodes)
    if (isa<DILocation>(Entry.second) &&
        !MachineMetadata.contains(Entry.second))
      InlineDebugLocations.insert(cast<DILocation>(Entry.second));
}

MachineModuleSlotTracker::~MachineModuleSlotTracker() = default;
