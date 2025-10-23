//===- DXILValidateMetadata.h - Pass to validate DXIL metadata --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_DIRECTX_DXILVALIDATEMETADATA_H
#define LLVM_TARGET_DIRECTX_DXILVALIDATEMETADATA_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// A pass that validates metadata to be DXIL compatible
class DXILValidateMetadata : public PassInfoMixin<DXILValidateMetadata> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_TARGET_DIRECTX_DXILVALIDATEMETADATA_H
