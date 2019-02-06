//===- lib/MC/MCTargetOptions.cpp - MC Target Options ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCTargetOptions.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;

MCTargetOptions::MCTargetOptions()
    : SanitizeAddress(false), MCRelaxAll(false), MCNoExecStack(false),
      MCFatalWarnings(false), MCNoWarn(false), MCNoDeprecatedWarn(false),
      MCSaveTempLabels(false), MCUseDwarfDirectory(false),
      MCIncrementalLinkerCompatible(false), MCPIECopyRelocations(false),
      ShowMCEncoding(false), ShowMCInst(false), AsmVerbose(false),
      PreserveAsmComments(true) {}

StringRef MCTargetOptions::getABIName() const {
  return ABIName;
}
