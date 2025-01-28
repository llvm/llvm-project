//===- RegionsFromMetadata.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A SandboxIR function pass that builds regions from IR metadata and then runs
// a pipeline of region passes on them. This is useful to test region passes in
// isolation without relying on the output of the bottom-up vectorizer.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_REGIONSFROMMETADATA_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_REGIONSFROMMETADATA_H

#include "llvm/ADT/StringRef.h"
#include "llvm/SandboxIR/Pass.h"
#include "llvm/SandboxIR/PassManager.h"

namespace llvm::sandboxir {

class RegionsFromMetadata final : public FunctionPass {
  // The PM containing the pipeline of region passes.
  RegionPassManager RPM;

public:
  RegionsFromMetadata(StringRef Pipeline);
  bool runOnFunction(Function &F, const Analyses &A) final;
  void printPipeline(raw_ostream &OS) const final {
    OS << getName() << "\n";
    RPM.printPipeline(OS);
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_REGIONSFROMMETADATA_H
