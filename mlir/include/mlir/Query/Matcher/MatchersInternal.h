//===- MatchersInternal.h - Structural query framework ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the base layer of the matcher framework.
//
// Matchers are methods that return a Matcher which provides a
// `match(...)` method whose parameters define the context of the match.
// Support includes simple (unary) matchers as well as matcher combinators
// (anyOf, allOf, etc.)
//
// This file contains the wrapper classes needed to construct matchers for
// mlir-query.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHER_MATCHERSINTERNAL_H
#define MLIR_TOOLS_MLIRQUERY_MATCHER_MATCHERSINTERNAL_H

#include "mlir/IR/Matchers.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace mlir::query::matcher {
class DynMatcher;
namespace internal {

bool allOfVariadicOperator(Operation *op, SetVector<Operation *> *matchedOps,
                           ArrayRef<DynMatcher> innerMatchers);
bool anyOfVariadicOperator(Operation *op, SetVector<Operation *> *matchedOps,
                           ArrayRef<DynMatcher> innerMatchers);

} // namespace internal

// Defaults to false if T has no match() method with the signature:
// match(Operation* op).
template <typename T, typename = void>
struct has_simple_match : std::false_type {};

// Specialized type trait that evaluates to true if T has a match() method
// with the signature: match(Operation* op).
template <typename T>
struct has_simple_match<T, std::void_t<decltype(std::declval<T>().match(
                               std::declval<Operation *>()))>>
    : std::true_type {};

// Defaults to false if T has no match() method with the signature:
// match(Operation* op, SetVector<Operation*>&).
template <typename T, typename = void>
struct has_bound_match : std::false_type {};

// Specialized type trait that evaluates to true if T has a match() method
// with the signature: match(Operation* op, SetVector<Operation*>&).
template <typename T>
struct has_bound_match<T, std::void_t<decltype(std::declval<T>().match(
                              std::declval<Operation *>(),
                              std::declval<SetVector<Operation *> &>()))>>
    : std::true_type {};

// Generic interface for matchers on an MLIR operation.
class MatcherInterface
    : public llvm::ThreadSafeRefCountedBase<MatcherInterface> {
public:
  virtual ~MatcherInterface() = default;

  virtual bool match(Operation *op) = 0;
  virtual bool match(Operation *op, SetVector<Operation *> &matchedOps) = 0;
};

// MatcherFnImpl takes a matcher function object and implements
// MatcherInterface.
template <typename MatcherFn>
class MatcherFnImpl : public MatcherInterface {
public:
  MatcherFnImpl(MatcherFn &matcherFn) : matcherFn(matcherFn) {}

  bool match(Operation *op) override {
    if constexpr (has_simple_match<MatcherFn>::value)
      return matcherFn.match(op);
    return false;
  }

  bool match(Operation *op, SetVector<Operation *> &matchedOps) override {
    if constexpr (has_bound_match<MatcherFn>::value)
      return matcherFn.match(op, matchedOps);
    return false;
  }

private:
  MatcherFn matcherFn;
};

// VariadicMatcher takes a vector of Matchers and returns true if any Matchers
// match the given operation.
using VariadicOperatorFunction = bool (*)(Operation *op,
                                          SetVector<Operation *> *matchedOps,
                                          ArrayRef<DynMatcher> innerMatchers);

template <VariadicOperatorFunction Func>
class VariadicMatcher : public MatcherInterface {
public:
  VariadicMatcher(std::vector<DynMatcher> matchers)
      : matchers(std::move(matchers)) {}

  bool match(Operation *op) override { return Func(op, nullptr, matchers); }
  bool match(Operation *op, SetVector<Operation *> &matchedOps) override {
    return Func(op, &matchedOps, matchers);
  }

private:
  std::vector<DynMatcher> matchers;
};

// Matcher wraps a MatcherInterface implementation and provides match()
// methods that redirect calls to the underlying implementation.
class DynMatcher {
public:
  // Takes ownership of the provided implementation pointer.
  DynMatcher(MatcherInterface *implementation)
      : implementation(implementation) {}

  // Construct from a variadic function.
  enum VariadicOperator {
    // Matches operations for which all provided matchers match.
    AllOf,
    // Matches operations for which at least one of the provided matchers
    // matches.
    AnyOf
  };

  static std::unique_ptr<DynMatcher>
  constructVariadic(VariadicOperator Op,
                    std::vector<DynMatcher> innerMatchers) {
    switch (Op) {
    case AllOf:
      return std::make_unique<DynMatcher>(
          new VariadicMatcher<internal::allOfVariadicOperator>(
              std::move(innerMatchers)));
    case AnyOf:
      return std::make_unique<DynMatcher>(
          new VariadicMatcher<internal::anyOfVariadicOperator>(
              std::move(innerMatchers)));
    }
    llvm_unreachable("Invalid Op value.");
  }

  template <typename MatcherFn>
  static std::unique_ptr<DynMatcher>
  constructDynMatcherFromMatcherFn(MatcherFn &matcherFn) {
    auto impl = std::make_unique<MatcherFnImpl<MatcherFn>>(matcherFn);
    return std::make_unique<DynMatcher>(impl.release());
  }

  bool match(Operation *op) const { return implementation->match(op); }
  bool match(Operation *op, SetVector<Operation *> &matchedOps) const {
    return implementation->match(op, matchedOps);
  }

  void setFunctionName(StringRef name) { functionName = name.str(); }
  bool hasFunctionName() const { return !functionName.empty(); }
  StringRef getFunctionName() const { return functionName; }

private:
  llvm::IntrusiveRefCntPtr<MatcherInterface> implementation;
  std::string functionName;
};

// VariadicOperatorMatcher related types.
template <typename... Ps>
class VariadicOperatorMatcher {
public:
  VariadicOperatorMatcher(DynMatcher::VariadicOperator varOp, Ps &&...params)
      : varOp(varOp), params(std::forward<Ps>(params)...) {}

  operator std::unique_ptr<DynMatcher>() const & {
    return DynMatcher::constructVariadic(
        varOp, getMatchers(std::index_sequence_for<Ps...>()));
  }

  operator std::unique_ptr<DynMatcher>() && {
    return DynMatcher::constructVariadic(
        varOp, std::move(*this).getMatchers(std::index_sequence_for<Ps...>()));
  }

private:
  // Helper method to unpack the tuple into a vector.
  template <std::size_t... Is>
  std::vector<DynMatcher> getMatchers(std::index_sequence<Is...>) const & {
    return {DynMatcher(std::get<Is>(params))...};
  }

  template <std::size_t... Is>
  std::vector<DynMatcher> getMatchers(std::index_sequence<Is...>) && {
    return {DynMatcher(std::get<Is>(std::move(params)))...};
  }

  const DynMatcher::VariadicOperator varOp;
  std::tuple<Ps...> params;
};

// Overloaded function object to generate VariadicOperatorMatcher objects from
// arbitrary matchers.
template <unsigned MinCount, unsigned MaxCount>
struct VariadicOperatorMatcherFunc {
  DynMatcher::VariadicOperator varOp;

  template <typename... Ms>
  VariadicOperatorMatcher<Ms...> operator()(Ms &&...Ps) const {
    static_assert(MinCount <= sizeof...(Ms) && sizeof...(Ms) <= MaxCount,
                  "invalid number of parameters for variadic matcher");
    return VariadicOperatorMatcher<Ms...>(varOp, std::forward<Ms>(Ps)...);
  }
};

namespace internal {
const VariadicOperatorMatcherFunc<1, std::numeric_limits<unsigned>::max()>
    anyOf = {DynMatcher::AnyOf};
const VariadicOperatorMatcherFunc<1, std::numeric_limits<unsigned>::max()>
    allOf = {DynMatcher::AllOf};
} // namespace internal
} // namespace mlir::query::matcher

#endif // MLIR_TOOLS_MLIRQUERY_MATCHER_MATCHERSINTERNAL_H
