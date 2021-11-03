//===-- CSKYInstrInfo.h - CSKY Instruction Information --------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the CSKY implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "CSKYInstrInfo.h"
#include "llvm/MC/MCContext.h"

#define DEBUG_TYPE "csky-instr-info"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "CSKYGenInstrInfo.inc"

CSKYInstrInfo::CSKYInstrInfo(CSKYSubtarget &STI)
    : CSKYGenInstrInfo(CSKY::ADJCALLSTACKDOWN, CSKY::ADJCALLSTACKUP), STI(STI) {
}
