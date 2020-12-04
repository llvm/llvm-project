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

#include <type_traits>

namespace llvm {

/// A suitably aligned and sized character array member which can hold elements
/// of any type.
///
/// These types may be arrays, structs, or any other types. Underneath is a
/// char buffer member which can be used as suitable storage for a placement
/// new of any of these types.
template <typename T, typename... Ts>
using AlignedCharArrayUnion = std::aligned_union_t<1, T, Ts...>;

} // end namespace llvm

#endif // LLVM_SUPPORT_ALIGNOF_H
