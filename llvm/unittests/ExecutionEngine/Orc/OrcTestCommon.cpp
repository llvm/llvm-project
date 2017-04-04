//===--------- OrcTestCommon.cpp - Utilities for Orc Unit Tests -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common utilities for Orc unit tests.
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"

using namespace llvm;

bool OrcNativeTarget::NativeTargetInitialized = false;

ModuleBuilder::ModuleBuilder(LLVMContext &Context, StringRef Triple,
                             StringRef Name)
  : M(new Module(Name, Context)) {
  if (Triple != "")
    M->setTargetTriple(Triple);
}
