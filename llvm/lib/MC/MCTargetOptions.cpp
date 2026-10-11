//===- lib/MC/MCTargetOptions.cpp - MC Target Options ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCTargetOptions.h"
#include "llvm/ADT/StringRef.h"
#include <climits>

using namespace llvm;

MCTargetOptions::MCTargetOptions()
    : MCRelaxAll(false), MCNoExecStack(false), MCFatalWarnings(false),
      MCNoWarn(false), MCNoDeprecatedWarn(false), MCNoTypeCheck(false),
      MCSaveTempLabels(false), MCIncrementalLinkerCompatible(false),
      FDPIC(false), ShowMCEncoding(false), ShowMCInst(false), AsmVerbose(false),
      PreserveAsmComments(true), Dwarf64(false),
      EmitDwarfUnwind(EmitDwarfUnwindType::Default),
      MCUseDwarfDirectory(DefaultDwarfDirectory),
      EmitCompactUnwindNonCanonical(false), EmitSFrameUnwind(false),
      PPCUseFullRegisterNames(false), LargeEHEncoding(false) {}

std::pair<int, int> MCTargetOptions::parseBinutilsVersion(StringRef Version) {
  if (Version == "none")
    return {INT_MAX, INT_MAX}; // Make binutilsIsAtLeast() return true.
  std::pair<int, int> Ret;
  if (!Version.consumeInteger(10, Ret.first) && Version.consume_front("."))
    Version.consumeInteger(10, Ret.second);
  return Ret;
}

StringRef MCTargetOptions::getABIName() const {
  return ABIName;
}

StringRef MCTargetOptions::getAssemblyLanguage() const {
  return AssemblyLanguage;
}
