//===- DXILRefineTypes.h - Infer additional type information ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass infers increased type information fidelity of memory operations and
// replaces their uses with it. Primarily load/store.
//
// This is used to prevent the propogation of introduced problematic types. For
// instance: the InstCombine pass can promote aggregates of 16/32-bit types to
// be i32/64 loads and stores.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_DXILREFINETYPES_H
#define LLVM_TRANSFORMS_SCALAR_DXILREFINETYPES_H

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

class DXILRefineTypesPass : public PassInfoMixin<DXILRefineTypesPass> {
private:
  bool runImpl(Function &F);

public:
  DXILRefineTypesPass() = default;

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_DXILREFINETYPES_H
