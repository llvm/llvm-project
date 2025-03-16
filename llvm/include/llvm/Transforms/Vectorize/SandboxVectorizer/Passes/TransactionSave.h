//===- TransactionSave.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a region pass that simply calls Context::save() to save the IR state.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_TRANSACTIONSAVE_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_TRANSACTIONSAVE_H

#include "llvm/SandboxIR/Pass.h"
#include "llvm/SandboxIR/Region.h"

namespace llvm::sandboxir {

class TransactionSave : public RegionPass {
public:
  TransactionSave() : RegionPass("tr-save") {}
  bool runOnRegion(Region &Rgn, const Analyses &A) final;
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_TRANSACTIONSAVE_H
