//===-- M88kSubtarget.cpp - M88k Subtarget Information ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the M88k specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "M88kSubtarget.h"
#include "M88k.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "m88k-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "M88kGenSubtargetInfo.inc"

void M88kSubtarget::anchor() {}

M88kSubtarget::M88kSubtarget(const Triple &TT, const std::string &CPU,
                             const std::string &FS, const TargetMachine &TM)
    : M88kGenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS), TargetTriple(TT),
      InstrInfo(*this), TLInfo(TM, *this), FrameLowering() {}
