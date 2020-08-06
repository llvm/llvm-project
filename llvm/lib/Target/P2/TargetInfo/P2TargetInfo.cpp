//===-- P2TargetInfo.cpp - P2 Target Implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "P2.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target llvm::TheP2Target;

extern "C" void LLVMInitializeP2TargetInfo() {
    RegisterTarget<Triple::p2, false> X(TheP2Target, "p2", "Propeller 2", "P2");
}