//===-- GCEmptyBasicBlocks.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class GCEmptyBasicBlocksPass : public PassInfoMixin<GCEmptyBasicBlocksPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // namespace llvm
