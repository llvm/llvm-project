//===- DXILOpLowering.h - Lowering to DXIL operations -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file Pass for lowering llvm intrinsics into DXIL operations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DXILOPLOWERING_H
#define LLVM_LIB_TARGET_DIRECTX_DXILOPLOWERING_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class DXILOpLowering : public PassInfoMixin<DXILOpLowering> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_DXILOPLOWERING_H
