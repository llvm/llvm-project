//===-- ARMTargetInfo.cpp - ARM Target Implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "llvm/Module.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target llvm::TheARMTarget, llvm::TheThumbTarget;

extern "C" void LLVMInitializeARMTargetInfo() { 
  RegisterTarget<Triple::arm, /*HasJIT=*/true>
    X(TheARMTarget, "arm", "ARM");

  RegisterTarget<Triple::thumb, /*HasJIT=*/true>
    Y(TheThumbTarget, "thumb", "Thumb");
}
