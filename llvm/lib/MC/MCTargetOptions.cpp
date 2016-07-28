//===- lib/MC/MCTargetOptions.cpp - MC Target Options --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCTargetOptions.h"

namespace llvm {

MCTargetOptions::MCTargetOptions()
    : SanitizeAddress(false), MCRelaxAll(false), MCNoExecStack(false),
      MCFatalWarnings(false), MCNoWarn(false), MCSaveTempLabels(false),
      MCUseDwarfDirectory(false), MCIncrementalLinkerCompatible(false),
      ShowMCEncoding(false), ShowMCInst(false), AsmVerbose(false),
      PreserveAsmComments(true), DwarfVersion(0), ABIName() {}

StringRef MCTargetOptions::getABIName() const {
  return ABIName;
}

} // end namespace llvm
