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

#include "llvm/SandboxIR/Pass.h"

namespace llvm::sandboxir {

class BottomUpVec final : public FunctionPass {

public:
  BottomUpVec() : FunctionPass("bottom-up-vec") {}
  bool runOnFunction(Function &F) final;
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_BOTTOMUPVEC_H
