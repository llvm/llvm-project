//===-- llvm/Support/float128.h - Compiler abstraction support --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLOAT128
#define LLVM_FLOAT128

#if defined(__clang__)
typedef __float128 float128;
#elif defined(__GNUC__) || defined(__GNUG__)
typedef _Float128 float128;
#else
typedef long double float128;
#endif

#endif // LLVM_FLOAT128
