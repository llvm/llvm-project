//===- llvm/ADT/STLArrayExtras.h - additions to <array> ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains some templates that are useful if you are working with the
// STL at all.
//
// No library is required when using these functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STLARRAYEXTRAS_H
#define LLVM_ADT_STLARRAYEXTRAS_H

#include <cstddef>

#include "llvm/Support/Compiler.h"

namespace llvm {

//===----------------------------------------------------------------------===//
//     Extra additions for arrays
//===----------------------------------------------------------------------===//

/// Find the length of an array.
template <class T, std::size_t N>
LLVM_DEPRECATED("Use std::size instead.", "std::size")
constexpr inline size_t array_lengthof(T (&)[N]) {
  return N;
}

} // end namespace llvm

#endif // LLVM_ADT_STLARRAYEXTRAS_H
