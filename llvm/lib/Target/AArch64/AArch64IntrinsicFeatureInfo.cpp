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

#include "AArch64GenIntrinsicFeatureChecks.inc"

llvm::FeatureBitset
llvm::getAArch64StreamingFeatureBits(const AArch64Subtarget &ST) {
  // A streaming-compatible function can enter streaming mode with SME.
  static constexpr FeatureBitset Implied = {
      AArch64::FeatureBF16, AArch64::FeatureFPARMv8, AArch64::FeatureFullFP16,
      AArch64::FeatureNEON, AArch64::FeatureSME};
  return ST.isStreamingCompatible() && !ST.isStreaming()
             ? ST.getFeatureBits() | Implied
             : ST.getFeatureBits();
}

const llvm::AArch64GuardedIntrinsicInfo &
llvm::getAArch64GuardedIntrinsicInfo() {
  return getAArch64GeneratedGuardedIntrinsicInfo<AArch64Subtarget>();
}
