//===-- SPIRVMachineFunctionInfo.cpp - SPIR-V machine function info -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVMachineFunctionInfo.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVSubtarget.h"

using namespace llvm;

yaml::SPIRVMachineFunctionInfo::SPIRVMachineFunctionInfo(
    const llvm::MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<SPIRVSubtarget>();
  const TargetRegisterInfo *TRI = ST.getRegisterInfo();
  const SPIRVGlobalRegistry *GR = ST.getSPIRVGlobalRegistry();
  for (auto [Reg, TypeReg] : GR->getVRegTypeIDs(MF)) {
    SPIRVVRegType &Entry = VRegTypes.emplace_back();
    raw_string_ostream(Entry.Reg.Value) << printReg(Reg, TRI);
    raw_string_ostream(Entry.Type.Value) << printReg(TypeReg, TRI);
  }
  if (Register Reg = GR->getLastInsertedTypeReg(MF))
    raw_string_ostream(LastInsertedType.Value) << printReg(Reg, TRI);
}

void yaml::SPIRVMachineFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<SPIRVMachineFunctionInfo>::mapping(YamlIO, *this);
}
