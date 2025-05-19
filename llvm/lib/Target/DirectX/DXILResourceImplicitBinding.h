//===- DXILResourceImplicitBindings.h --_____________-----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file Assign register slots to resources without explicit binding.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DXILRESOURCEIMPLICITBINDING_H
#define LLVM_LIB_TARGET_DIRECTX_DXILRESOURCEIMPLICITBINDING_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class DXILResourceImplicitBinding
    : public PassInfoMixin<DXILResourceImplicitBinding> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_DXILRESOURCEIMPLICITBINDING_H
