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
#include <type_traits>
#include <utility>

namespace llvm::PatternMatchHelpers {
// A naive std::apply would cost more in compile-time.
template <typename TupleT, typename IndicesT> class PatternStorage;

template <typename TupleT, size_t... Indices>
class PatternStorage<TupleT, std::index_sequence<Indices...>> {
  TupleT Patterns;

public:
  template <typename... Ty>
  constexpr PatternStorage(Ty &&...Ps) : Patterns(std::forward<Ty>(Ps)...) {}
  template <typename ITy> bool combineAnd(ITy *V) const {
    return (std::get<Indices>(Patterns).match(V) && ...);
  }
  template <typename ITy> bool combineOr(ITy *V) const {
    return (std::get<Indices>(Patterns).match(V) || ...);
  }
};

/// Matching or combinator.
template <typename... Ty> struct match_combine_or { // NOLINT
  PatternStorage<std::tuple<std::decay_t<Ty>...>,
                 std::index_sequence_for<Ty...>>
      Storage;
  match_combine_or(const Ty &...Ps) : Storage(Ps...) {}
  template <typename ITy> bool match(ITy *V) const {
    return Storage.combineOr(V);
  }
};

/// Matching and combinator.
template <typename... Ty> struct match_combine_and { // NOLINT
  PatternStorage<std::tuple<std::decay_t<Ty>...>,
                 std::index_sequence_for<Ty...>>
      Storage;
  match_combine_and(const Ty &...Ps) : Storage(Ps...) {}
  template <typename ITy> bool match(ITy *V) const {
    return Storage.combineAnd(V);
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
