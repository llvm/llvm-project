//===- llvm/CodeGen/LiveDebugValuesPass.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEDEBUGVALUESPASS_H
#define LLVM_CODEGEN_LIVEDEBUGVALUESPASS_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class LiveDebugValuesPass : public PassInfoMixin<LiveDebugValuesPass> {
  const bool ShouldEmitDebugEntryValues;

public:
  LiveDebugValuesPass(bool ShouldEmitDebugEntryValues)
      : ShouldEmitDebugEntryValues(ShouldEmitDebugEntryValues) {}

  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);
};

} // namespace llvm

#endif // LLVM_CODEGEN_LIVEDEBUGVALUESPASS_H
