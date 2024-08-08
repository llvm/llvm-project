//===-- ExpandVectorPredication.h - Expand vector predication ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_EXPANDVECTORPREDICATION_H
#define LLVM_CODEGEN_EXPANDVECTORPREDICATION_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class TargetTransformInfo;
class VPIntrinsic;

/// Expand a vector predication intrinsic. Returns true if the intrinsic was
/// removed/replaced.
bool expandVectorPredicationIntrinsic(VPIntrinsic &VPI,
                                      const TargetTransformInfo &TTI);

} // end namespace llvm

#endif // LLVM_CODEGEN_EXPANDVECTORPREDICATION_H
