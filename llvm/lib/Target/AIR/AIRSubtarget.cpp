//===-- AIRSubtarget.cpp - AIR Subtarget Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the AIR-specific subclass of TargetSubtargetInfo.
///
//===----------------------------------------------------------------------===//

#include "AIRSubtarget.h"
#include "AIRTargetLowering.h"

using namespace llvm;

#define DEBUG_TYPE "air-subtarget"

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_TARGET_DESC
#include "AIRGenSubtargetInfo.inc"

AIRSubtarget::AIRSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                               const AIRTargetMachine &TM)
    : AIRGenSubtargetInfo(TT, CPU, CPU, FS), InstrInfo(*this), FL(*this),
      TL(TM, *this) {}

void AIRSubtarget::anchor() {}
