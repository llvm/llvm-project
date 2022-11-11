//===- STLForwardCompat.h - Library features from future STLs ------C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains library features backported from future STL versions.
///
/// These should be replaced with their STL counterparts as the C++ version LLVM
/// is compiled with is updated.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STLFORWARDCOMPAT_H
#define LLVM_ADT_STLFORWARDCOMPAT_H

#include <type_traits>

namespace llvm {

//===----------------------------------------------------------------------===//
//     Features from C++20
//===----------------------------------------------------------------------===//

template <typename T>
struct remove_cvref // NOLINT(readability-identifier-naming)
{
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <typename T>
using remove_cvref_t // NOLINT(readability-identifier-naming)
    = typename llvm::remove_cvref<T>::type;

} // namespace llvm

#endif // LLVM_ADT_STLFORWARDCOMPAT_H
