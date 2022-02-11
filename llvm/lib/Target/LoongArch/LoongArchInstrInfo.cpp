//=- LoongArchInstrInfo.cpp - LoongArch Instruction Information -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the LoongArch implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "LoongArchInstrInfo.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "LoongArchGenInstrInfo.inc"

LoongArchInstrInfo::LoongArchInstrInfo(LoongArchSubtarget &STI)
    // FIXME: add CFSetup and CFDestroy Inst when we implement function call.
    : LoongArchGenInstrInfo(),

      STI(STI) {}
