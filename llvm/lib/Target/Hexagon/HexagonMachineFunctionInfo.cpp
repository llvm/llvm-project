//= HexagonMachineFunctionInfo.cpp - Hexagon machine function info *- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HexagonMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// pin vtable to this file
void HexagonMachineFunctionInfo::anchor() {}

MachineFunctionInfo *HexagonMachineFunctionInfo::clone(
    BumpPtrAllocator &Allocator, MachineFunction &DestMF,
    const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
    const {
  return DestMF.cloneInfo<HexagonMachineFunctionInfo>(*this);
}

static yaml::StringValue regToString(Register Reg,
                                     const TargetRegisterInfo &TRI) {
  yaml::StringValue Dest;
  if (Reg.isValid()) {
    raw_string_ostream OS(Dest.Value);
    OS << printReg(Reg, &TRI);
  }
  return Dest;
}

yaml::HexagonFunctionInfo::HexagonFunctionInfo(
    const llvm::HexagonMachineFunctionInfo &MFI, const TargetRegisterInfo &TRI)
    : StackAlignBaseReg(regToString(MFI.getStackAlignBaseReg(), TRI)) {}

void yaml::HexagonFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<HexagonFunctionInfo>::mapping(YamlIO, *this);
}

void HexagonMachineFunctionInfo::initializeBaseYamlFields(
    const yaml::HexagonFunctionInfo &YamlMFI) {}
