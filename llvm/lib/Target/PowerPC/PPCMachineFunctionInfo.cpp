//===-- PPCMachineFunctionInfo.cpp - Private data used for PowerPC --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PPCMachineFunctionInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

void PPCFunctionInfo::anchor() { }

MCSymbol *PPCFunctionInfo::getPICOffsetSymbol() const {
  const DataLayout &DL = MF.getDataLayout();
  return MF.getContext().getOrCreateSymbol(Twine(DL.getPrivateGlobalPrefix()) +
                                           Twine(MF.getFunctionNumber()) +
                                           "$poff");
}
