//===- TransactionSave.cpp - Save the IR state ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/TransactionSave.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InstructionCost.h"

namespace llvm::sandboxir {

bool TransactionSave::runOnRegion(Region &Rgn, const Analyses &A) {
  Rgn.getContext().save();
  return false;
}

} // namespace llvm::sandboxir
