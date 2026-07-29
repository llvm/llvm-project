//===- SPIRVFinalizeShaderLinkage.h - Finalize shader linkage --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Shader-only analogue of DXILFinalizeLinkage: internalizes non-entry,
/// non-exported HLSL helper functions and erases the resulting dead ones.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVFINALIZESHADERLINKAGE_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVFINALIZESHADERLINKAGE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SPIRVTargetMachine;

class SPIRVFinalizeShaderLinkage
    : public OptionalPassInfoMixin<SPIRVFinalizeShaderLinkage> {
  const SPIRVTargetMachine &TM;

public:
  SPIRVFinalizeShaderLinkage(const SPIRVTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVFINALIZESHADERLINKAGE_H
