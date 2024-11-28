//===- BottomUpVec.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A Bottom-Up Vectorizer pass.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_BOTTOMUPVEC_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_BOTTOMUPVEC_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/SandboxIR/Constant.h"
#include "llvm/SandboxIR/Pass.h"
#include "llvm/SandboxIR/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"

namespace llvm::sandboxir {

class BottomUpVec final : public FunctionPass {
  bool Change = false;
  std::unique_ptr<LegalityAnalysis> Legality;
  SmallVector<Instruction *> DeadInstrCandidates;

  /// Creates and returns a vector instruction that replaces the instructions in
  /// \p Bndl. \p Operands are the already vectorized operands.
  Value *createVectorInstr(ArrayRef<Value *> Bndl, ArrayRef<Value *> Operands);
  /// Erases all dead instructions from the dead instruction candidates
  /// collected during vectorization.
  void tryEraseDeadInstrs();
  /// Packs all elements of \p ToPack into a vector and returns that vector.
  Value *createPack(ArrayRef<Value *> ToPack);
  /// Recursively try to vectorize \p Bndl and its operands.
  Value *vectorizeRec(ArrayRef<Value *> Bndl, unsigned Depth);
  /// Entry point for vectorization starting from \p Seeds.
  bool tryVectorize(ArrayRef<Value *> Seeds);

  /// The PM containing the pipeline of region passes.
  RegionPassManager RPM;

public:
  BottomUpVec(StringRef Pipeline);
  bool runOnFunction(Function &F, const Analyses &A) final;
  void printPipeline(raw_ostream &OS) const final {
    OS << getName() << "\n";
    RPM.printPipeline(OS);
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_BOTTOMUPVEC_H
