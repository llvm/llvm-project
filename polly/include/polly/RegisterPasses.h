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

#include "llvm/IR/LegacyPassManager.h"

namespace llvm {
namespace legacy {
class PassManagerBase;
} // namespace legacy
} // namespace llvm

namespace polly {
void initializePollyPasses(llvm::PassRegistry &Registry);
void registerPollyPasses(llvm::legacy::PassManagerBase &PM);
} // namespace polly
#endif
