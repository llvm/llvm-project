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

/// Represents the details the expansion of a VP intrinsic.
enum class VPExpansionDetails {
  /// No change happened during expansion.
  IntrinsicUnchanged,
  /// At least one operand was updated.
  IntrinsicUpdated,
  /// The whole intrinsic was replaced.
  IntrinsicReplaced,
};

/// Expand a vector predication intrinsic. Returns the kind of expansion
/// that was applied to the intrinsic.
VPExpansionDetails
expandVectorPredicationIntrinsic(VPIntrinsic &VPI,
                                 const TargetTransformInfo &TTI);

} // end namespace llvm

#endif // LLVM_CODEGEN_EXPANDVECTORPREDICATION_H
