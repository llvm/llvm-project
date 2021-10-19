//===- PPCRegisterBankInfo.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the RegisterBankInfo class for
/// PowerPC.
//===----------------------------------------------------------------------===//

#include "PPCRegisterBankInfo.h"
#include "PPCRegisterInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ppc-reg-bank-info"

#define GET_TARGET_REGBANK_IMPL
#include "PPCGenRegisterBank.inc"

using namespace llvm;

PPCRegisterBankInfo::PPCRegisterBankInfo(const TargetRegisterInfo &TRI)
    : PPCGenRegisterBankInfo() {}
