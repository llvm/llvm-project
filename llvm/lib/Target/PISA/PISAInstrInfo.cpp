//===-- PISAInstrInfo.cpp - PISA Instruction Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISAInstrInfo.h"
#include "PISA.h"
#include "PISASubtarget.h"

#define GET_INSTRINFO_CTOR_DTOR
#include "PISAGenInstrInfo.inc"

using namespace llvm;

PISAInstrInfo::PISAInstrInfo(const PISASubtarget &STI)
    : PISAGenInstrInfo(STI, RI), RI() {}
