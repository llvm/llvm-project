//===-- SuperHMachineFunctionInfo.h - SuperH private data -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the SuperH specific subclass of MachineFunctionInfo.
///
//===----------------------------------------------------------------------===//

#include "SuperHMachineFunctionInfo.h"
#include "SuperHConstantPoolValue.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include <type_traits>

using namespace llvm;

void SuperHMachineFunctionInfo::anchor() {}

MachineFunctionInfo *SuperHMachineFunctionInfo::clone(
    BumpPtrAllocator &Allocator, MachineFunction &DestMF,
    const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
    const {
  return DestMF.cloneInfo<SuperHMachineFunctionInfo>(*this);
}

SuperHConstantPoolConstant *SuperHMachineFunctionInfo::tryGetConstant(
        GlobalAddressSDNode *N, 
        SelectionDAG &DAG, 
        SHCP::SHCPModifier Modifier) {

  // Early exit for null node.
  if (!N)
    return nullptr;

  // Run though the constant pool that is tied to the DAG and search for 
  // the constant there.
  MachineConstantPool *MCP = DAG.getMachineFunction().getConstantPool();
  for (auto &MC : MCP->getConstants()) {
    if (MC.isMachineConstantPoolEntry()) {
      if (auto *CPV = (SuperHConstantPoolConstant*)MC.Val.MachineCPVal) {
        if (CPV->getGV() == N->getGlobal())
          return CPV;
      }
    }
  }

  // If not found, create a new one and add it.
  MachineFunction &MF = DAG.getMachineFunction();
  SuperHMachineFunctionInfo *SFI = MF.getInfo<SuperHMachineFunctionInfo>();
  unsigned LabelIndex = SFI->createConstIndex();
  return SuperHConstantPoolConstant::Create(
    N->getGlobal(), 
    LabelIndex,
    SHCP::SHCPKind::CPValue,
    Modifier
  );
}

SuperHConstantPoolConstant *SuperHMachineFunctionInfo::tryGetConstant(
        BlockAddressSDNode *N, 
        SelectionDAG &DAG, 
        SHCP::SHCPModifier Modifier) {

  // Early exit for null node.
  if (!N)
    return nullptr;

  // Run though the constant pool that is tied to the DAG and search for 
  // the constant there.
  MachineConstantPool *MCP = DAG.getMachineFunction().getConstantPool();
  for (auto &MC : MCP->getConstants()) {
    if (MC.isMachineConstantPoolEntry()) {
      if (auto *CPV = (SuperHConstantPoolConstant*)MC.Val.MachineCPVal) {
        if (CPV->getBlockAddress() == N->getBlockAddress())
          return CPV;
      }
    }
  }

  // If not found, create a new one and add it.
  MachineFunction &MF = DAG.getMachineFunction();
  SuperHMachineFunctionInfo *SFI = MF.getInfo<SuperHMachineFunctionInfo>();
  unsigned LabelIndex = SFI->createConstIndex();
  return SuperHConstantPoolConstant::Create(
    N->getBlockAddress(), 
    LabelIndex,
    SHCP::SHCPKind::CPBlockAddress,
    Modifier
  );
}

SuperHConstantPoolSymbol *SuperHMachineFunctionInfo::tryGetConstant(
        ExternalSymbolSDNode *N, 
        SelectionDAG &DAG, 
        SHCP::SHCPModifier Modifier) {

  // Early exit for null node.
  if (!N)
    return nullptr;

  // Run though the constant pool that is tied to the DAG and search for 
  // the constant there.
  MachineConstantPool *MCP = DAG.getMachineFunction().getConstantPool();
  for (auto &MC : MCP->getConstants()) {
    if (MC.isMachineConstantPoolEntry()) {
      if (auto *CPV = (SuperHConstantPoolSymbol*)MC.Val.MachineCPVal) {
        if (CPV->getSymbol() == N->getSymbol())
          return CPV;
      }
    }
  }
  
  // If not found, create a new one and add it.
  MachineFunction &MF = DAG.getMachineFunction();
  SuperHMachineFunctionInfo *SFI = MF.getInfo<SuperHMachineFunctionInfo>();
  unsigned LabelIndex = SFI->createConstIndex();
  return SuperHConstantPoolSymbol::Create(
    *DAG.getContext(), 
    N->getSymbol(), 
    LabelIndex
  );
}