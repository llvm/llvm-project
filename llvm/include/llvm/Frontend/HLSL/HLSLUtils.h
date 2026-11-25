//===- HLSLUtils.h - HLSL helper functions --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Various helpers for working with HLSL shaders.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_HLSLUTILS_H
#define LLVM_FRONTEND_HLSL_HLSLUTILS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
namespace hlsl {

LLVM_ABI void parseThreadCounts(StringRef Str, unsigned &Count1,
                                unsigned &Count2, unsigned &Count3);

} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLUTILS_H
