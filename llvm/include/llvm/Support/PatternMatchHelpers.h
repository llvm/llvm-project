//===- PatternMatchHelpers.h - Helpers for PatternMatch -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides helpers that are used across the IR PatternMatch,
// ScalarEvolutionPatternMatch, and VPlanPatternMatch.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PATTERNMATCHHELPERS_H
#define LLVM_SUPPORT_PATTERNMATCHHELPERS_H

#include "llvm/Support/Casting.h"
#include <tuple>

namespace llvm::PatternMatchHelpers {
/// Matching or combinator.
template <typename... Ty> struct match_combine_or { // NOLINT
  std::tuple<Ty...> Ps;
  match_combine_or(const Ty &...Ps) : Ps(Ps...) {}
  template <typename ITy> bool match(ITy *V) const {
    return std::apply([V](auto &&...Ps) { return (Ps.match(V) || ...); }, Ps);
  }
};

/// Matching and combinator.
template <typename... Ty> struct match_combine_and { // NOLINT
  std::tuple<Ty...> Ps;
  match_combine_and(const Ty &...Ps) : Ps(Ps...) {}
  template <typename ITy> bool match(ITy *V) const {
    return std::apply([V](auto &&...Ps) { return (Ps.match(V) && ...); }, Ps);
  }
};

/// Combine pattern matchers matching any of Ps patterns.
template <typename... Ty>
inline match_combine_or<Ty...> m_CombineOr(const Ty &...Ps) { // NOLINT
  return {Ps...};
}

/// Combine pattern matchers matching all of Ps patterns.
template <typename... Ty>
inline match_combine_and<Ty...> m_CombineAnd(const Ty &...Ps) { // NOLINT
  return {Ps...};
}

/// A match-wrapper around isa.
template <typename... To> struct match_isa { // NOLINT
  template <typename ArgTy> bool match(const ArgTy *V) const {
    return isa<To...>(V);
  }
};

template <typename... To> inline match_isa<To...> m_Isa() { // NOLINT
  return match_isa<To...>();
}

/// A variant of m_Isa that also matches SubPattern.
template <typename... To, typename SubPattern>
inline auto m_Isa(const SubPattern &P) { // NOLINT
  return m_CombineAnd(m_Isa<To...>(), P);
}
} // namespace llvm::PatternMatchHelpers

#endif // LLVM_SUPPORT_PATTERNMATCHHELPERS_H
