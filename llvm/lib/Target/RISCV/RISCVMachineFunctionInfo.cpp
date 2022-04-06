//=- RISCVMachineFunctionInfo.cpp - RISCV machine function info ---*- C++ -*-=//
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

void yaml::RISCVMachineFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<RISCVMachineFunctionInfo>::mapping(YamlIO, *this);
}

void RISCVMachineFunctionInfo::initializeBaseYamlFields(
    const yaml::RISCVMachineFunctionInfo &YamlMFI) {
  VarArgsFrameIndex = YamlMFI.VarArgsFrameIndex;
  VarArgsSaveSize = YamlMFI.VarArgsSaveSize;
}
