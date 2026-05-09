//===- SPIRVLegalizeImplicitBinding.h - Legalize implicit bindings -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVLEGALIZEIMPLICITBINDING_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVLEGALIZEIMPLICITBINDING_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SPIRVLegalizeImplicitBinding
    : public OptionalPassInfoMixin<SPIRVLegalizeImplicitBinding> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVLEGALIZEIMPLICITBINDING_H
