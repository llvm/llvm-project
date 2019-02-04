//===-- SystemZCallingConv.cpp - Calling conventions for SystemZ ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZCallingConv.h"
#include "SystemZRegisterInfo.h"

using namespace llvm;

const MCPhysReg SystemZ::ArgGPRs[SystemZ::NumArgGPRs] = {
  SystemZ::R2D, SystemZ::R3D, SystemZ::R4D, SystemZ::R5D, SystemZ::R6D
};

const MCPhysReg SystemZ::ArgFPRs[SystemZ::NumArgFPRs] = {
  SystemZ::F0D, SystemZ::F2D, SystemZ::F4D, SystemZ::F6D
};
