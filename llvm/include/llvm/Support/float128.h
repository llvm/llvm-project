//===-- llvm/Support/float128.h - Compiler abstraction support --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLOAT128
#define LLVM_FLOAT128

#include <cmath>

namespace llvm {

#ifdef HAS_LOGF128
#if !defined(__LONG_DOUBLE_IBM128__) && (__SIZEOF_INT128__ == 16)
typedef decltype(logf128(0.)) float128;
#define HAS_IEE754_FLOAT128
#endif
#endif // HAS_LOGF128

} // namespace llvm
#endif // LLVM_FLOAT128
