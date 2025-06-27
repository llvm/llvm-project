//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWARFCFIChecker/DWARFCFIFunctionFrameReceiver.h"

using namespace llvm;

CFIFunctionFrameReceiver::~CFIFunctionFrameReceiver() = default;

void CFIFunctionFrameReceiver::startFunctionUnit(
    bool IsEH, ArrayRef<MCCFIInstruction> Prologue) {}

void CFIFunctionFrameReceiver::emitInstructionAndDirectives(
    const MCInst &Inst, ArrayRef<MCCFIInstruction> Directives) {}

void CFIFunctionFrameReceiver::finishFunctionUnit() {}
