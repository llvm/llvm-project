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

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/Function.h"

#define DEBUG_TYPE "irtranslator"

using namespace llvm;

char IRTranslator::ID = 0;

bool IRTranslator::translateADD(const Instruction &Inst) {
  // Get or create a virtual register for each value.
  // Unless the value is a Constant => loadimm cst?
  // or inline constant each time?
  // Creation of a virtual register needs to have a size.
  return false;
}

bool IRTranslator::translate(const Instruction &Inst) {
  switch(Inst.getOpcode()) {
    case Instruction::Add: {
      return translateADD(Inst);
    default:
      llvm_unreachable("Opcode not supported");
    }
  }
}


void IRTranslator::finalize() {
  // Release the memory used by the different maps we
  // needed during the translation.
  ValToVRegs.clear();
  Constants.clear();
}

IRTranslator::IRTranslator()
  : MachineFunctionPass(ID) {
}

bool IRTranslator::runOnMachineFunction(MachineFunction &MF) {
  const Function &F = *MF.getFunction();
  for (const BasicBlock &BB: F) {
    for (const Instruction &Inst: BB) {
      bool Succeeded = translate(Inst);
      if (!Succeeded) {
        DEBUG(dbgs() << "Cannot translate: " << Inst << '\n');
        report_fatal_error("Unable to translate instruction");
      }
    }
  }
  return false;
}
