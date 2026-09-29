//===- LibcallLoweringInfo.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Legacy-pass-manager wrapper around the Analysis-layer libcall lowering info,
// which is aware of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIBCALLLOWERINGINFO_H
#define LLVM_CODEGEN_LIBCALLLOWERINGINFO_H

#include "llvm/Analysis/LibcallLoweringInfo.h"
#include "llvm/Pass.h"

namespace llvm {
class RuntimeLibraryInfoWrapper;
class TargetSubtargetInfo;

/// Resolve the LibcallLoweringInfo for \p Subtarget from the module-level \p
/// ModuleInfo, applying the subtarget's libcall overrides.
LLVM_ABI const LibcallLoweringInfo &
getLibcallLowering(const ModuleLibcallLoweringInfo &ModuleInfo,
                   const TargetSubtargetInfo &Subtarget);

class LLVM_ABI LibcallLoweringInfoWrapper : public ImmutablePass {
  ModuleLibcallLoweringInfo Result;
  RuntimeLibraryInfoWrapper *RuntimeLibcallsWrapper = nullptr;

public:
  static char ID;
  LibcallLoweringInfoWrapper();

  const LibcallLoweringInfo &
  getLibcallLowering(const Module &M, const TargetSubtargetInfo &Subtarget);

  const ModuleLibcallLoweringInfo &getResult(const Module &M);

  void initializePass() override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;
};

} // end namespace llvm

#endif // LLVM_CODEGEN_LIBCALLLOWERINGINFO_H
