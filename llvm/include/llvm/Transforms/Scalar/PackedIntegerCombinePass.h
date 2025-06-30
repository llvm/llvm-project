//===- PackedIntegerCombinePass.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides the interface for LLVM's Packed Integer Combine pass.
/// This pass tries to treat integers as packed chunks of individual bytes,
/// and leverage this to coalesce needlessly fragmented
/// computations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_PACKEDINTCOMBINE_H
#define LLVM_TRANSFORMS_SCALAR_PACKEDINTCOMBINE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class PackedIntegerCombinePass
    : public PassInfoMixin<PackedIntegerCombinePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_PACKEDINTCOMBINE_H
