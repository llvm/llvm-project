//===-- MetalSubtarget.cpp - Metal Subtarget Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the Metal-specific subclass of TargetSubtargetInfo.
///
//===----------------------------------------------------------------------===//

#include "MetalSubtarget.h"
#include "MetalTargetLowering.h"

using namespace llvm;

#define DEBUG_TYPE "metal-subtarget"

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_TARGET_DESC
#include "MetalGenSubtargetInfo.inc"

MetalSubtarget::MetalSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                               const MetalTargetMachine &TM)
    : MetalGenSubtargetInfo(TT, CPU, CPU, FS), InstrInfo(*this), FL(*this),
      TL(TM, *this) {}

void MetalSubtarget::anchor() {}
