//===--------- OrcTestCommon.cpp - Utilities for Orc Unit Tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common utilities for Orc unit tests.
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"

using namespace llvm;

const JITTargetAddress llvm::orc::CoreAPIsBasedStandardTest::FooAddr;
const JITTargetAddress llvm::orc::CoreAPIsBasedStandardTest::BarAddr;
const JITTargetAddress llvm::orc::CoreAPIsBasedStandardTest::BazAddr;
const JITTargetAddress llvm::orc::CoreAPIsBasedStandardTest::QuxAddr;

bool OrcNativeTarget::NativeTargetInitialized = false;

ModuleBuilder::ModuleBuilder(LLVMContext &Context, StringRef Triple,
                             StringRef Name)
  : M(new Module(Name, Context)) {
  if (Triple != "")
    M->setTargetTriple(Triple);
}
