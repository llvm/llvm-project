//===-- LanaiMachineFuctionInfo.cpp - Lanai machine function info ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LanaiMachineFunctionInfo.h"

using namespace llvm;

void LanaiMachineFunctionInfo::anchor() {}

Register LanaiMachineFunctionInfo::getGlobalBaseReg() {
  // Return if it has already been initialized.
  if (GlobalBaseReg)
    return GlobalBaseReg;

  return GlobalBaseReg =
             MF.getRegInfo().createVirtualRegister(&Lanai::GPRRegClass);
}
