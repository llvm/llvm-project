//===-- PPCRegisterBankInfo.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the targeting of the RegisterBankInfo class for PowerPC.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PPC_GISEL_PPCREGISTERBANKINFO_H
#define LLVM_LIB_TARGET_PPC_GISEL_PPCREGISTERBANKINFO_H

#include "llvm/CodeGen/GlobalISel/RegisterBank.h"
#include "llvm/CodeGen/GlobalISel/RegisterBankInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGBANK_DECLARATIONS
#include "PPCGenRegisterBank.inc"

namespace llvm {
class TargetRegisterInfo;

class PPCGenRegisterBankInfo : public RegisterBankInfo {
protected:
#define GET_TARGET_REGBANK_CLASS
#include "PPCGenRegisterBank.inc"
};

class PPCRegisterBankInfo final : public PPCGenRegisterBankInfo {
public:
  PPCRegisterBankInfo(const TargetRegisterInfo &TRI);
};
} // namespace llvm

#endif
