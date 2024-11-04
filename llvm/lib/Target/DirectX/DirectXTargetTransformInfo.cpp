//===- DirectXTargetTransformInfo.cpp - DirectX TTI ---------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
//===----------------------------------------------------------------------===//

#include "DirectXTargetTransformInfo.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"

using namespace llvm;

bool DirectXTTIImpl::isTargetIntrinsicWithScalarOpAtArg(Intrinsic::ID ID,
                                                        unsigned ScalarOpdIdx) {
  switch (ID) {
  case Intrinsic::dx_wave_readlane:
    return ScalarOpdIdx == 1;
  default:
    return false;
  }
}

bool DirectXTTIImpl::isVectorIntrinsicWithOverloadTypeAtArg(
    Intrinsic::ID ID, unsigned ScalarOpdIdx, bool Default) {
  switch (ID) {
  default:
    return Default;
  }
}

bool DirectXTTIImpl::isTargetIntrinsicTriviallyScalarizable(
    Intrinsic::ID ID) const {
  switch (ID) {
  case Intrinsic::dx_frac:
  case Intrinsic::dx_rsqrt:
  case Intrinsic::dx_wave_readlane:
  case Intrinsic::dx_splitdouble:
    return true;
  default:
    return false;
  }
}
