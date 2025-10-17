//===- llvm/ADT/Identity.h - Provide std::identity from C++20 ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides an implementation of std::identity from C++20.
//
// No library is required when using these functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_IDENTITY_H
#define LLVM_ADT_IDENTITY_H

#include <utility>

namespace llvm {

// Our legacy llvm::identity, not quite the same as std::identity.
template <class Ty = void> struct identity {
  using is_transparent = void;
  using argument_type = Ty;

  Ty &operator()(Ty &self) const { return self; }
  const Ty &operator()(const Ty &self) const { return self; }
};

// Forward-ported from C++20.
//
// While we are migrating from the legacy version, we must refer to this
// template as identity<>.  Once the legacy version is removed, we can
// make this a non-template struct and remove the <>.
template <> struct identity<void> {
  using is_transparent = void;

  template <typename T> constexpr T &&operator()(T &&self) const {
    return std::forward<T>(self);
  }
};

} // namespace llvm

#endif // LLVM_ADT_IDENTITY_H
