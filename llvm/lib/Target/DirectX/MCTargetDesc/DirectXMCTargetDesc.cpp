//===- DirectXMCTargetDesc.cpp - DirectX Target Implementation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains DirectX target initializer.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

#define GET_SUBTARGETINFO_MC_DESC
#include "DirectXGenSubtargetInfo.inc"

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeDirectXTargetMC() {}
