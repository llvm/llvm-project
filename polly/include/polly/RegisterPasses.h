//===------ polly/RegisterPasses.h - Register the Polly passes *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions to register the Polly passes in a LLVM pass manager.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_REGISTER_PASSES_H
#define POLLY_REGISTER_PASSES_H

#include "llvm/Support/Compiler.h"

namespace llvm {
class PassRegistry;
class PassBuilder;
struct PassPluginLibraryInfo;
namespace legacy {
class PassManagerBase;
} // namespace legacy
} // namespace llvm

namespace polly {
void initializePollyPasses(llvm::PassRegistry &Registry);
void registerPollyPasses(llvm::PassBuilder &PB);
} // namespace polly

LLVM_ALWAYS_EXPORT llvm::PassPluginLibraryInfo getPollyPluginInfo();

#endif
