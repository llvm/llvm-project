//===-- PowerPCTargetInfo.cpp - PowerPC Target Implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "llvm/Module.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target llvm::ThePPC32Target, llvm::ThePPC64Target;

extern "C" void LLVMInitializePowerPCTargetInfo() { 
  RegisterTarget<Triple::ppc, /*HasJIT=*/true>
    X(ThePPC32Target, "ppc32", "PowerPC 32");

  RegisterTarget<Triple::ppc64, /*HasJIT=*/true>
    Y(ThePPC64Target, "ppc64", "PowerPC 64");
}
