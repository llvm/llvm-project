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

bool DirectXTTIImpl::isTargetIntrinsicWithScalarOpAtArg(
    Intrinsic::ID ID, unsigned ScalarOpdIdx) const {
  switch (ID) {
  case Intrinsic::dx_wave_readlane:
    return ScalarOpdIdx == 1;
  default:
    return false;
  }
}

bool DirectXTTIImpl::isTargetIntrinsicWithOverloadTypeAtArg(Intrinsic::ID ID,
                                                            int OpdIdx) const {
  switch (ID) {
  case Intrinsic::dx_asdouble:
  case Intrinsic::dx_firstbitlow:
  case Intrinsic::dx_firstbitshigh:
  case Intrinsic::dx_firstbituhigh:
  case Intrinsic::dx_isinf:
  case Intrinsic::dx_isnan:
  case Intrinsic::dx_legacyf16tof32:
  case Intrinsic::dx_legacyf32tof16:
  case Intrinsic::dx_wave_all_equal:
    return OpdIdx == 0;
  default:
    // All DX intrinsics are overloaded on return type unless specified
    // otherwise
    return OpdIdx == -1;
  }
}
