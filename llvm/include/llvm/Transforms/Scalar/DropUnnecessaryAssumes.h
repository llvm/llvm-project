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
  DropUnnecessaryAssumesPass(bool DropDereferenceable = false)
      : DropDereferenceable(DropDereferenceable) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  bool DropDereferenceable;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_DROPUNNECESSARYASSUMES_H
