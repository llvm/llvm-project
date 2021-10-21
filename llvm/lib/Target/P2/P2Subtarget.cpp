//===-- P2Subtarget.cpp - P2 Subtarget Information ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the P2 specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "P2Subtarget.h"
#include "P2.h"

using namespace llvm;

#define DEBUG_TYPE "p2-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "P2GenSubtargetInfo.inc"

void P2Subtarget::anchor() {}

P2Subtarget::P2Subtarget(const Triple &TT, const std::string &CPU, const std::string &FS, const P2TargetMachine &TM, bool cogex) :
        P2GenSubtargetInfo(TT, CPU, CPU, FS),
        FrameLowering(TM), InstrInfo(), TLInfo(TM), is_cogex(cogex) {
    ParseSubtargetFeatures(CPU, CPU, FS);
}
