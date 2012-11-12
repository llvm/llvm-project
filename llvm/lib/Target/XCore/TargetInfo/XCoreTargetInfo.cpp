//===-- XCoreTargetInfo.cpp - XCore Target Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "XCore.h"
#include "llvm/Module.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target llvm::TheXCoreTarget;

extern "C" void LLVMInitializeXCoreTargetInfo() { 
  RegisterTarget<Triple::xcore> X(TheXCoreTarget, "xcore", "XCore");
}
