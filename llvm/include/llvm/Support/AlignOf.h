//===--- AlignOf.h - Portable calculation of type alignment -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AlignedCharArrayUnion class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ALIGNOF_H
#define LLVM_SUPPORT_ALIGNOF_H

#include <algorithm>

namespace llvm {

/// A suitably aligned and sized character array member which can hold elements
/// of any type.
template <typename T, typename... Ts> struct AlignedCharArrayUnion {
  // Work around "internal compiler error: Segmentation fault" with GCC 7.5,
  // apparently caused by alignas(Ts...).
  static constexpr std::size_t Align = std::max({alignof(T), alignof(Ts)...});
  alignas(Align) char buffer[std::max({sizeof(T), sizeof(Ts)...})];
};

} // end namespace llvm

#endif // LLVM_SUPPORT_ALIGNOF_H
