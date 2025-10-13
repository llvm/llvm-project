//===- SeedCollection.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The seed-collection pass of the bottom-up vectorizer.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_SEEDCOLLECTION_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_SEEDCOLLECTION_H

#include "llvm/SandboxIR/Pass.h"
#include "llvm/SandboxIR/PassManager.h"

namespace llvm::sandboxir {

/// This pass collects the instructions that can become vectorization "seeds",
/// like stores to consecutive memory addresses. It then goes over the collected
/// seeds, slicing them into appropriately sized chunks, creating a Region with
/// the seed slice as the Auxiliary vector and runs the region pass pipeline.
class SeedCollection final : public FunctionPass {

  /// The PM containing the pipeline of region passes.
  RegionPassManager RPM;

public:
  SeedCollection(StringRef Pipeline);
  bool runOnFunction(Function &F, const Analyses &A) final;
  void printPipeline(raw_ostream &OS) const final {
    OS << getName() << "\n";
    RPM.printPipeline(OS);
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_SEEDCOLLECTION_H
