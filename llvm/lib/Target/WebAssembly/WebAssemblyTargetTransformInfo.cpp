//===-- WebAssemblyTargetTransformInfo.cpp - WebAssembly-specific TTI -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines the WebAssembly-specific TargetTransformInfo
/// implementation.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyTargetTransformInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/CostTable.h"
using namespace llvm;

#define DEBUG_TYPE "wasmtti"

TargetTransformInfo::PopcntSupportKind
WebAssemblyTTIImpl::getPopcntSupport(unsigned TyWidth) const {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  return TargetTransformInfo::PSK_FastHardware;
}

bool
WebAssemblyTTIImpl::haveFastSqrt(Type *Ty) const {
  assert(Ty->isFPOrFPVectorTy() && "Ty must be floating point");
  return true;
}
