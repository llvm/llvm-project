//===-- AVRTargetInfo.cpp - AVR Target Implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"

namespace llvm {
Target TheAVRTarget;
}

extern "C" void LLVMInitializeAVRTargetInfo() {
  llvm::RegisterTarget<llvm::Triple::avr> X(
      llvm::TheAVRTarget, "avr", "Atmel AVR Microcontroller");
}

// FIXME: Temporary stub - this function must be defined for linking
// to succeed. Remove once this function is properly implemented.
extern "C" void LLVMInitializeAVRTargetMC() {
}
