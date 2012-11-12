//===-- MBlazeTargetInfo.cpp - MBlaze Target Implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MBlaze.h"
#include "llvm/Module.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target llvm::TheMBlazeTarget;

extern "C" void LLVMInitializeMBlazeTargetInfo() {
  RegisterTarget<Triple::mblaze> X(TheMBlazeTarget, "mblaze", "MBlaze");
}
