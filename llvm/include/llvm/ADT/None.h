//===-- None.h - Simple null value for implicit construction ------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///  This file provides None, an enumerator for use in implicit constructors
///  of various (usually templated) types to make such construction more
///  terse.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_NONE_H
#define LLVM_ADT_NONE_H

#include "llvm/Support/Compiler.h"
#include <optional>

namespace llvm {
/// A simple null object to allow implicit construction of Optional<T>
/// and similar types without having to spell out the specialization's name.
LLVM_DEPRECATED("Use std::nullopt_t instead", "std::nullopt_t")
typedef std::nullopt_t NoneType;
inline constexpr std::nullopt_t None = std::nullopt;
}

#endif
