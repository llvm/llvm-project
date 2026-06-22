//===- IntegerLegalize.h - AGX-JIT integer/intrinsic legalization *- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_INTEGERLEGALIZE_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_INTEGERLEGALIZE_H

#include "llvm/IR/Module.h"

namespace llvm {
namespace metal {

void stripLifetimeIntrinsics(Module &M);

void expandWideIntegers(Module &M);

void lowerFreezeInsts(Module &M);

void stripDisjointFlags(Module &M);

void canonicalizeNNegZExt(Module &M);

void lowerCmpIntrinsics(Module &M);

}
}

#endif
