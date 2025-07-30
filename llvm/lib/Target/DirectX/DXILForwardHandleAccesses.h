//===- DXILForwardHandleAccesses.h - Cleanup Handles ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file Eliminate redundant stores and loads from handle globals.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DXILFORWARDHANDLEACCESS_H
#define LLVM_LIB_TARGET_DIRECTX_DXILFORWARDHANDLEACCESS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class DXILForwardHandleAccesses
    : public PassInfoMixin<DXILForwardHandleAccesses> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_DXILFORWARDHANDLEACCESS_H
