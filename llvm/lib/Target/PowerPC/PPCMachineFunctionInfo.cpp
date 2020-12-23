//===-- PPCMachineFunctionInfo.cpp - Private data used for PowerPC --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PPCMachineFunctionInfo.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
static cl::opt<bool> PPCDisableNonVolatileCR(
    "ppc-disable-non-volatile-cr",
    cl::desc("Disable the use of non-volatile CR register fields"),
    cl::init(false), cl::Hidden);

void PPCFunctionInfo::anchor() {}
PPCFunctionInfo::PPCFunctionInfo(const MachineFunction &MF)
    : DisableNonVolatileCR(PPCDisableNonVolatileCR) {}

MCSymbol *PPCFunctionInfo::getPICOffsetSymbol(MachineFunction &MF) const {
  const DataLayout &DL = MF.getDataLayout();
  return MF.getContext().getOrCreateSymbol(Twine(DL.getPrivateGlobalPrefix()) +
                                           Twine(MF.getFunctionNumber()) +
                                           "$poff");
}

MCSymbol *PPCFunctionInfo::getGlobalEPSymbol(MachineFunction &MF) const {
  const DataLayout &DL = MF.getDataLayout();
  return MF.getContext().getOrCreateSymbol(Twine(DL.getPrivateGlobalPrefix()) +
                                           "func_gep" +
                                           Twine(MF.getFunctionNumber()));
}

MCSymbol *PPCFunctionInfo::getLocalEPSymbol(MachineFunction &MF) const {
  const DataLayout &DL = MF.getDataLayout();
  return MF.getContext().getOrCreateSymbol(Twine(DL.getPrivateGlobalPrefix()) +
                                           "func_lep" +
                                           Twine(MF.getFunctionNumber()));
}

MCSymbol *PPCFunctionInfo::getTOCOffsetSymbol(MachineFunction &MF) const {
  const DataLayout &DL = MF.getDataLayout();
  return MF.getContext().getOrCreateSymbol(Twine(DL.getPrivateGlobalPrefix()) +
                                           "func_toc" +
                                           Twine(MF.getFunctionNumber()));
}

bool PPCFunctionInfo::isLiveInSExt(Register VReg) const {
  for (const std::pair<Register, ISD::ArgFlagsTy> &LiveIn : LiveInAttrs)
    if (LiveIn.first == VReg)
      return LiveIn.second.isSExt();
  return false;
}

bool PPCFunctionInfo::isLiveInZExt(Register VReg) const {
  for (const std::pair<Register, ISD::ArgFlagsTy> &LiveIn : LiveInAttrs)
    if (LiveIn.first == VReg)
      return LiveIn.second.isZExt();
  return false;
}

void PPCFunctionInfo::appendParameterType(ParamType Type) {
  uint32_t CopyParamType = ParameterType;
  int Bits = 0;

  // If it is fixed type, we only need to increase the FixedParamNum, for
  // the bit encode of fixed type is bit of zero, we do not need to change the
  // ParamType.
  if (Type == FixedType) {
    ++FixedParamNum;
    return;
  }

  ++FloatingPointParamNum;

  for (int I = 0;
       I < static_cast<int>(FloatingPointParamNum + FixedParamNum - 1); ++I) {
    if (CopyParamType & XCOFF::TracebackTable::ParmTypeIsFloatingBit) {
      // '10'b => floating point short parameter.
      // '11'b => floating point long parameter.
      CopyParamType <<= 2;
      Bits += 2;
    } else {
      // '0'b => fixed parameter.
      CopyParamType <<= 1;
      ++Bits;
    }
  }

  assert(Type != FixedType && "FixedType should already be handled.");
  if (Bits < 31)
    ParameterType |= Type << (30 - Bits);
}
