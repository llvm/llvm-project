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

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/SandboxIR/Type.h"
#include "llvm/SandboxIR/Utils.h"

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

  /// \Returns true if \p I1 and \p I2 are load/stores accessing consecutive
  /// memory addresses.
  template <typename LoadOrStoreT>
  static bool areConsecutive(LoadOrStoreT *I1, LoadOrStoreT *I2,
                             ScalarEvolution &SE, const DataLayout &DL) {
    static_assert(std::is_same<LoadOrStoreT, LoadInst>::value ||
                      std::is_same<LoadOrStoreT, StoreInst>::value,
                  "Expected Load or Store!");
    auto Diff = Utils::getPointerDiffInBytes(I1, I2, SE);
    if (!Diff)
      return false;
    int ElmBytes = Utils::getNumBits(I1) / 8;
    return *Diff == ElmBytes;
  }

  template <typename LoadOrStoreT>
  static bool areConsecutive(ArrayRef<Value *> &Bndl, ScalarEvolution &SE,
                             const DataLayout &DL) {
    static_assert(std::is_same<LoadOrStoreT, LoadInst>::value ||
                      std::is_same<LoadOrStoreT, StoreInst>::value,
                  "Expected Load or Store!");
    assert(isa<LoadOrStoreT>(Bndl[0]) && "Expected Load or Store!");
    auto *LastLS = cast<LoadOrStoreT>(Bndl[0]);
    for (Value *V : drop_begin(Bndl)) {
      assert(isa<LoadOrStoreT>(V) &&
             "Unimplemented: we only support StoreInst!");
      auto *LS = cast<LoadOrStoreT>(V);
      if (!VecUtils::areConsecutive(LastLS, LS, SE, DL))
        return false;
      LastLS = LS;
    }
    return true;
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_VECUTILS_H
