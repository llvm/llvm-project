//=- RISCVMachineFunctionInfo.cpp - RISC-V machine function info --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares RISCV-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#include "RISCVMachineFunctionInfo.h"

using namespace llvm;

yaml::RISCVMachineFunctionInfo::RISCVMachineFunctionInfo(
    const llvm::RISCVMachineFunctionInfo &MFI)
    : VarArgsFrameIndex(MFI.getVarArgsFrameIndex()),
      VarArgsSaveSize(MFI.getVarArgsSaveSize()) {}

MachineFunctionInfo *RISCVMachineFunctionInfo::clone(
    BumpPtrAllocator &Allocator, MachineFunction &DestMF,
    const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
    const {
  return DestMF.cloneInfo<RISCVMachineFunctionInfo>(*this);
}

void yaml::RISCVMachineFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<RISCVMachineFunctionInfo>::mapping(YamlIO, *this);
}

void RISCVMachineFunctionInfo::initializeBaseYamlFields(
    const yaml::RISCVMachineFunctionInfo &YamlMFI) {
  VarArgsFrameIndex = YamlMFI.VarArgsFrameIndex;
  VarArgsSaveSize = YamlMFI.VarArgsSaveSize;
}

void RISCVMachineFunctionInfo::addSExt32Register(Register Reg) {
  SExt32Registers.push_back(Reg);
}

bool RISCVMachineFunctionInfo::isSExt32Register(Register Reg) const {
  return is_contained(SExt32Registers, Reg);
}

void RISCVMachineFunctionInfo::recordCFIInfo(MachineInstr *MI, int Reg,
                                             int FrameReg, int64_t Offset) {
  assert(Reg >= 0 && "Negative dwarf reg number!");
  CFIInfoMap[MI] = std::make_tuple(Reg, FrameReg, Offset);
}

bool RISCVMachineFunctionInfo::getCFIInfo(MachineInstr *MI, int &Reg,
                                          int &FrameReg, int64_t &Offset) {
  auto Found = CFIInfoMap.find(MI);
  if (Found == CFIInfoMap.end()) {
    return false;
  }
  Reg = get<0>(Found->second);
  FrameReg = get<1>(Found->second);
  assert(Reg >= 0 && "Negative dwarf reg number!");
  assert(FrameReg >= 0 && "Negative dwarf reg number!");
  Offset = get<2>(Found->second);
  return true;
}
