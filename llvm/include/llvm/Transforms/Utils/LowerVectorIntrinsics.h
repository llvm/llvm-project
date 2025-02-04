//===- llvm/Transforms/Utils/LowerVectorIntrinsics.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower intrinsics with a scalable vector arg to loops.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOWERVECTORINTRINSICS_H
#define LLVM_TRANSFORMS_UTILS_LOWERVECTORINTRINSICS_H

#include <cstdint>
#include <optional>

namespace llvm {

class CallInst;
class Module;

/// Lower \p CI as a loop. \p CI is a unary intrinsic with a vector argument and
/// is deleted and replaced with a loop.
bool lowerUnaryVectorIntrinsicAsLoop(Module &M, CallInst *CI);

} // namespace llvm

#endif
