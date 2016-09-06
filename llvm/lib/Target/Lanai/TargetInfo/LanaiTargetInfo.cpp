//===-- LanaiTargetInfo.cpp - Lanai Target Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Lanai.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

Target llvm::TheLanaiTarget;

extern "C" void LLVMInitializeLanaiTargetInfo() {
  RegisterTarget<Triple::lanai> X(TheLanaiTarget, "lanai", "Lanai");
}
