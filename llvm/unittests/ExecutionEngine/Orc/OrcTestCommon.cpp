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

bool OrcNativeTarget::NativeTargetInitialized = false;

ModuleBuilder::ModuleBuilder(LLVMContext &Context, StringRef Triple,
                             StringRef Name)
  : M(new Module(Name, Context)) {
  if (Triple != "")
    M->setTargetTriple(Triple);
}

void llvm::orc::CoreAPIsBasedStandardTest::OverridableDispatcher::dispatch(
    std::unique_ptr<Task> T) {
  if (Parent.DispatchOverride)
    Parent.DispatchOverride(std::move(T));
  else
    InPlaceTaskDispatcher::dispatch(std::move(T));
}

std::unique_ptr<llvm::orc::ExecutorProcessControl>
llvm::orc::CoreAPIsBasedStandardTest::makeEPC(
    std::shared_ptr<SymbolStringPool> SSP) {
  return std::make_unique<UnsupportedExecutorProcessControl>(
      std::move(SSP), std::make_unique<OverridableDispatcher>(*this));
}
