//===- Optional.h - Simple variant for passing optional values --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///  This file provides Optional, a template class modeled in the spirit of
///  OCaml's 'opt' variant.  The idea is to strongly type whether or not
///  a value can be optional.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_OPTIONAL_H
#define LLVM_ADT_OPTIONAL_H

#include <optional>

namespace llvm {
// Legacy alias of llvm::Optional to std::optional.
// FIXME: Remove this after LLVM 16.
template <class T> using Optional = std::optional<T>;
} // namespace llvm

#endif // LLVM_ADT_OPTIONAL_H
