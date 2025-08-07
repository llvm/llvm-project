//===- DXILFlattenArrays.h - Perform flattening of DXIL Arrays -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_TARGET_DIRECTX_DXILFLATTENARRAYS_H
#define LLVM_TARGET_DIRECTX_DXILFLATTENARRAYS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// A pass that transforms multidimensional arrays into one-dimensional arrays.
class DXILFlattenArrays : public PassInfoMixin<DXILFlattenArrays> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};
} // namespace llvm

#endif // LLVM_TARGET_DIRECTX_DXILFLATTENARRAYS_H
