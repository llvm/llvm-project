//===- SPIRVCBufferAccess.h - Translate CBuffer Loads ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVCBUFFERACCESS_H_
#define LLVM_LIB_TARGET_SPIRV_SPIRVCBUFFERACCESS_H_

#include "llvm/IR/PassManager.h"

namespace llvm {

class SPIRVCBufferAccess : public PassInfoMixin<SPIRVCBufferAccess> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVCBUFFERACCESS_H_
