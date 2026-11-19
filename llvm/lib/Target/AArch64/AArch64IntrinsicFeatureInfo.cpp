//===- AArch64IntrinsicFeatureInfo.cpp - Intrinsic feature table ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64IntrinsicFeatureInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/IR/GuardedIntrinsicInfo.h"
#include "llvm/IR/IntrinsicsAArch64.h"

using namespace llvm;

namespace {

#include "AArch64GenIntrinsicFeatureChecks.inc"

} // namespace

const GuardedIntrinsicInfo &llvm::getAArch64GuardedIntrinsicInfo() {
  static const GuardedIntrinsicInfo Info(
      AArch64IntrinsicFeatureClauses, AArch64IntrinsicOverloadTypes,
      AArch64IntrinsicGuardIndices, AArch64IntrinsicFeatureGuards,
      AArch64IntrinsicVariants, AArch64IntrinsicEntries);
  return Info;
}
