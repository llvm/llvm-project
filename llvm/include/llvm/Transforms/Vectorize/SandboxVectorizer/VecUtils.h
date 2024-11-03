//===- VecUtils.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Collector for SandboxVectorizer related convenience functions that don't
// belong in other classes.

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_VECUTILS_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_VECUTILS_H

#include "llvm/SandboxIR/Type.h"

namespace llvm::sandboxir {

class VecUtils {
public:
  /// \Returns the number of elements in \p Ty. That is the number of lanes if a
  /// fixed vector or 1 if scalar. ScalableVectors have unknown size and
  /// therefore are unsupported.
  static int getNumElements(Type *Ty) {
    assert(!isa<ScalableVectorType>(Ty));
    return Ty->isVectorTy() ? cast<FixedVectorType>(Ty)->getNumElements() : 1;
  }
  /// Returns \p Ty if scalar or its element type if vector.
  static Type *getElementType(Type *Ty) {
    return Ty->isVectorTy() ? cast<FixedVectorType>(Ty)->getElementType() : Ty;
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_VECUTILS_H
