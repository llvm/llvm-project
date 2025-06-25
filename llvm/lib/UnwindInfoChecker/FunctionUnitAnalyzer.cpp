//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/UnwindInfoChecker/FunctionUnitAnalyzer.h"

using namespace llvm;

FunctionUnitAnalyzer::~FunctionUnitAnalyzer() = default;

void FunctionUnitAnalyzer::startFunctionUnit(
    bool IsEH, ArrayRef<MCCFIInstruction> Prologue) {}

void FunctionUnitAnalyzer::emitInstructionAndDirectives(
    const MCInst &Inst, ArrayRef<MCCFIInstruction> Directives) {}

void FunctionUnitAnalyzer::finishFunctionUnit() {}
