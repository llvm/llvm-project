//===- DXILCBufferAccess.h - Translate CBuffer Loads ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file Pass for replacing loads from cbuffers in the cbuffer address space to
// cbuffer load intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DXILCBUFFERACCESS_H
#define LLVM_LIB_TARGET_DIRECTX_DXILCBUFFERACCESS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class DXILCBufferAccess : public PassInfoMixin<DXILCBufferAccess> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_DXILCBUFFERACCESS_H
