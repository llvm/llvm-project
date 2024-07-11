//===-- llvm/Support/float128.h - Compiler abstraction support --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLOAT128
#define LLVM_FLOAT128

namespace llvm {

#ifdef HAS_FLOAT128_LOGF128
typedef _Float128 float128;
#define HAS_IEE754_FLOAT128
#elif HAS__FLOAT128_LOGF128
typedef __float128 float128;
extern "C" {
float128 logf128(float128);
}
#define HAS_IEE754_FLOAT128
#elif HAS_LONG_DOUBLE_LOGF128
typedef long double float128;
#define HAS_IEE754_FLOAT128
#endif

} // namespace llvm
#endif // LLVM_FLOAT128
