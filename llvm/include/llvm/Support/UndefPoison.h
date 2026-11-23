//===-- llvm/Support/UndefPoison.h - Undef/Poison tracking ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the UndefPoisonKind enum and helper functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_UNDEFPOISON_H
#define LLVM_SUPPORT_UNDEFPOISON_H

namespace llvm {

/// Enumeration to track whether we are interested in Undef, Poison, or both.
enum class UndefPoisonKind {
  PoisonOnly = (1 << 0),
  UndefOnly = (1 << 1),
  UndefOrPoison = PoisonOnly | UndefOnly,
};

/// Returns true if \p Kind includes the Poison bit.
inline bool includesPoison(UndefPoisonKind Kind) {
  return (static_cast<unsigned>(Kind) &
          static_cast<unsigned>(UndefPoisonKind::PoisonOnly)) != 0;
}

/// Returns true if \p Kind includes the Undef bit.
inline bool includesUndef(UndefPoisonKind Kind) {
  return (static_cast<unsigned>(Kind) &
          static_cast<unsigned>(UndefPoisonKind::UndefOnly)) != 0;
}

} // namespace llvm

#endif // LLVM_SUPPORT_UNDEFPOISON_H
