//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass that drops assumes that are unlikely to be useful.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_DROPUNNECESSARYASSUMES_H
#define LLVM_TRANSFORMS_SCALAR_DROPUNNECESSARYASSUMES_H

#include "llvm/IR/PassManager.h"

namespace llvm {

struct DropUnnecessaryAssumesPass
    : public PassInfoMixin<DropUnnecessaryAssumesPass> {
  DropUnnecessaryAssumesPass(bool DropDereferenceable = false,
                             bool DropArrayBounds = false)
      : DropDereferenceable(DropDereferenceable),
        DropArrayBounds(DropArrayBounds) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  bool DropDereferenceable;
  // When true, drop assumes with "llvm.array.bounds" metadata. These are
  // array bounds assumes from Fortran/C/C++ that should be dropped before
  // vectorization to prevent IR bloat and avoid negatively impacting cost
  // models in later passes like LSR.
  bool DropArrayBounds;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_DROPUNNECESSARYASSUMES_H
