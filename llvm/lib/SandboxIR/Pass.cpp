//===- Pass.cpp - Passes that operate on Sandbox IR -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Pass.h"
#include "llvm/Support/Debug.h"

using namespace llvm::sandboxir;

#ifndef NDEBUG
void Pass::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG
