//===-- Nios2MCTargetDesc.cpp - Nios2 Target Descriptions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Nios2 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "Nios2MCTargetDesc.h"
#include "llvm/MC/MCInstrInfo.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#include "Nios2GenInstrInfo.inc"

#define GET_REGINFO_MC_DESC
#include "Nios2GenRegisterInfo.inc"

extern "C" void LLVMInitializeNios2TargetMC() {}
