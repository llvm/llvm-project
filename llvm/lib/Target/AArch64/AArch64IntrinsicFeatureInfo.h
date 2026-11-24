//===- AArch64IntrinsicFeatureInfo.h - Intrinsic feature table -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64INTRINSICFEATUREINFO_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64INTRINSICFEATUREINFO_H

#include "llvm/IR/GuardedIntrinsicInfo.h"

namespace llvm {

class AArch64Subtarget;
using AArch64GuardedIntrinsicInfo = GuardedIntrinsicInfo<AArch64Subtarget>;

/// Return the AArch64 intrinsic feature table shared by backend passes.
const AArch64GuardedIntrinsicInfo &getAArch64GuardedIntrinsicInfo();
FeatureBitset getAArch64StreamingFeatureBits(const AArch64Subtarget &ST);

} // namespace llvm

#endif // LLVM_LIB_TARGET_AARCH64_AARCH64INTRINSICFEATUREINFO_H
