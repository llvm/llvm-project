//===-- CodeGen/AsmPrinter/WinCFGuard.cpp - Control Flow Guard Impl ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing Win64 exception info into asm files.
//
//===----------------------------------------------------------------------===//

#include "WinCFGuard.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCStreamer.h"

#include <vector>

using namespace llvm;

WinCFGuard::WinCFGuard(AsmPrinter *A) : AsmPrinterHandler(), Asm(A) {}

WinCFGuard::~WinCFGuard() {}

void WinCFGuard::endModule() {
  const Module *M = Asm->MMI->getModule();
  std::vector<const Function *> Functions;
  for (const Function &F : *M)
    if (F.hasAddressTaken())
      Functions.push_back(&F);
  if (Functions.empty())
    return;
  auto &OS = *Asm->OutStreamer;
  OS.SwitchSection(Asm->OutContext.getObjectFileInfo()->getGFIDsSection());
  for (const Function *F : Functions)
    OS.EmitCOFFSymbolIndex(Asm->getSymbol(F));
}
