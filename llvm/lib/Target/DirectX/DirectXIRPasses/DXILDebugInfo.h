//===----- DebugInfo.h - analysis and lowering for Debug info -*- C++ -*- -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file Analyze and downgrade debug info metadata to match DXIL (LLVM 3.7).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DXILDEBUGINFO_H
#define LLVM_LIB_TARGET_DIRECTX_DXILDEBUGINFO_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

namespace dxil {

/// A pass that downgrades debug information to forms supported by DXIL.
class DXILDebugInfo : public OptionalPassInfoMixin<DXILDebugInfo> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace dxil
} // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_DXILDEBUGINFO_H
