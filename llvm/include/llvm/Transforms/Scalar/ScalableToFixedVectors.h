//===- ScalableToFixedVectors.h - Convert scalable to fixed vectors -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts IR operations on scalable vector types to fixed-length
// vectors when the effective length is known and is less than the minimum
// possible scaled vector length. For a scalable vector type with
// element count VF (known min elements), if minvscale * VF > VL, the we can
// convert to a fixed length vector of length VL.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_SCALABLETOFIXEDVECTORS_H
#define LLVM_TRANSFORMS_SCALAR_SCALABLETOFIXEDVECTORS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class ScalableToFixedVectorsPass
    : public PassInfoMixin<ScalableToFixedVectorsPass> {

public:
  explicit ScalableToFixedVectorsPass() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_SCALABLETOFIXEDVECTORS_H
