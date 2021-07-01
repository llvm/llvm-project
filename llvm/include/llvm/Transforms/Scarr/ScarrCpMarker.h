//===-- ScarrCpMarker.h - Transform IR with Checkpoint Info -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
///===----------------------------------------------------------------------===//
//
// Mark Basic Blocks into different ScaRR checkpoint types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCARR_SCARRCPMARKER_H
#define LLVM_TRANSFORMS_SCARR_SCARRCPMARKER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class ScarrCpMarkerPass : public PassInfoMixin<ScarrCpMarkerPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_SCARR_SCARRCPMARKER_H
