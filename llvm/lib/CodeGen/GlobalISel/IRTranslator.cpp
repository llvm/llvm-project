//===-- llvm/CodeGen/GlobalISel/IRTranslator.cpp - IRTranslator --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the IRTranslator class.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/IRTranslator.h"

using namespace llvm;

char IRTranslator::ID = 0;

bool IRTranslator::translateADD(const Instruction &Inst) {
  return false;
}

bool IRTranslator::translate(const Instruction &) {
  return false;
}


void IRTranslator::finalize() {
}

IRTranslator::IRTranslator()
  : MachineFunctionPass(ID) {
}

bool IRTranslator::runOnMachineFunction(MachineFunction &MF) {
  return false;
}
