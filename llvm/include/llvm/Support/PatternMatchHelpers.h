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

namespace llvm::PatternMatchHelpers {
/// Matching or combinator leaf case.
template <typename... Tys> struct match_combine_or { // NOLINT
  template <typename ITy> bool match(ITy *) const { return false; }
};

/// Matching or combinator.
template <typename Ty, typename... Tys>
struct match_combine_or<Ty, Tys...> : match_combine_or<Tys...> {
  Ty P;
  match_combine_or(const Ty &P, const Tys &...Ps)
      : match_combine_or<Tys...>(Ps...), P(P) {}

  template <typename ITy> bool match(ITy *V) const {
    return P.match(V) || match_combine_or<Tys...>::match(V);
  }
};

/// Matching and combinator leaf case.
template <typename... Tys> struct match_combine_and { // NOLINT
  template <typename ITy> bool match(ITy *) const { return true; }
};

/// Matching and combinator.
template <typename Ty, typename... Tys>
struct match_combine_and<Ty, Tys...> : match_combine_and<Tys...> {
  Ty P;
  match_combine_and(const Ty &P, const Tys &...Ps)
      : match_combine_and<Tys...>(Ps...), P(P) {}

  template <typename ITy> bool match(ITy *V) const {
    return P.match(V) && match_combine_and<Tys...>::match(V);
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

/// Matcher for a specific value, but stores a reference to the value, not the
/// value itself.
template <typename Ty> struct match_deferred { // NOLINT
  Ty *const &Val;
  match_deferred(Ty *const &V) : Val(V) {}
  template <typename ITy> bool match(ITy *const V) const { return V == Val; }
};

/// Matcher to bind the captured value.
template <typename Ty> struct match_bind { // NOLINT
  Ty *&VR;
  match_bind(Ty *&V) : VR(V) {}
  template <typename ITy> bool match(ITy *V) const {
    if (auto *CV = dyn_cast<Ty>(V)) {
      VR = CV;
      return true;
    }
    return false;
  }
};
} // namespace llvm::PatternMatchHelpers

#endif // LLVM_SUPPORT_PATTERNMATCHHELPERS_H
