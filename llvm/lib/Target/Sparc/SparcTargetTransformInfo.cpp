//===-- SparcTargetTransformInfo.cpp - SPARC specific TTI -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SparcTargetTransformInfo.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "sparctti"

TargetTransformInfo::PopcntSupportKind
SparcTTIImpl::getPopcntSupport(unsigned TyWidth) const {
  assert(isPowerOf2_32(TyWidth) && "Type width must be power of 2");
  if (ST->usePopc())
    return TTI::PSK_FastHardware;
  return TTI::PSK_Software;
}
