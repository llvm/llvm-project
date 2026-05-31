//===-- AIRInstrInfo.cpp - InstrInfo for AIR -*- C++ --------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AIR specific subclass of TargetInstrInfo.
//
//===----------------------------------------------------------------------===//

#include "AIRInstrInfo.h"
#include "AIRSubtarget.h"

#define GET_INSTRINFO_CTOR_DTOR
#include "AIRGenInstrInfo.inc"

using namespace llvm;

AIRInstrInfo::AIRInstrInfo(const AIRSubtarget &STI)
    : AIRGenInstrInfo(STI, RI) {}

AIRInstrInfo::~AIRInstrInfo() {}
