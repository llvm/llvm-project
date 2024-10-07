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
#include "llvm/SandboxIR/Constant.h"
#include "llvm/SandboxIR/Pass.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"

namespace llvm::sandboxir {

class BottomUpVec final : public FunctionPass {
  bool Change = false;
  LegalityAnalysis Legality;
  void vectorizeRec(ArrayRef<Value *> Bndl);
  void tryVectorize(ArrayRef<Value *> Seeds);

public:
  BottomUpVec() : FunctionPass("bottom-up-vec") {}
  bool runOnFunction(Function &F) final;
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_BOTTOMUPVEC_H
