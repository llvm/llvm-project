//===- llvm/CodeGen/GlobalISel/RegBankSelect.cpp - RegBankSelect -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the RegBankSelect class.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"

#define DEBUG_TYPE "regbankselect"

using namespace llvm;

char RegBankSelect::ID = 0;
INITIALIZE_PASS(RegBankSelect, "regbankselect",
                "Assign register bank of generic virtual registers",
                false, false);

RegBankSelect::RegBankSelect() : MachineFunctionPass(ID), RBI(nullptr) {
  initializeRegBankSelectPass(*PassRegistry::getPassRegistry());
}

bool RegBankSelect::runOnMachineFunction(MachineFunction &MF) {
  // Avoid unused field member warning.
  (void)RBI;
  return false;
}
