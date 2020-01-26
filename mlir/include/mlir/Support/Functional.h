//===- Functional.h - Helpers for functional-style Combinators --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_FUNCTIONAL_H_
#define MLIR_SUPPORT_FUNCTIONAL_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

/// This file provides some simple template functional-style sugar to operate
/// on **value** types. Make sure when using that the stored type is cheap to
/// copy!
///
/// TODO(ntv): add some static_assert but we need proper traits for this.

namespace mlir {
namespace functional {

/// Map with iterators.
template <typename Fn, typename IterType>
auto map(Fn fun, IterType begin, IterType end)
    -> SmallVector<typename std::result_of<Fn(decltype(*begin))>::type, 8> {
  using R = typename std::result_of<Fn(decltype(*begin))>::type;
  SmallVector<R, 8> res;
  // auto i works with both pointer types and value types with an operator*.
  // auto *i only works for pointer types.
  for (auto i = begin; i != end; ++i) {
    res.push_back(fun(*i));
  }
  return res;
}

/// Map with templated container.
template <typename Fn, typename ContainerType>
auto map(Fn fun, ContainerType input)
    -> decltype(map(fun, std::begin(input), std::end(input))) {
  return map(fun, std::begin(input), std::end(input));
}

/// Zip map with 2 templated container, iterates to the min of the sizes of
/// the 2 containers.
/// TODO(ntv): make variadic when needed.
template <typename Fn, typename ContainerType1, typename ContainerType2>
auto zipMap(Fn fun, ContainerType1 input1, ContainerType2 input2)
    -> SmallVector<typename std::result_of<Fn(decltype(*input1.begin()),
                                              decltype(*input2.begin()))>::type,
                   8> {
  using R = typename std::result_of<Fn(decltype(*input1.begin()),
                                       decltype(*input2.begin()))>::type;
  SmallVector<R, 8> res;
  auto zipIter = llvm::zip(input1, input2);
  for (auto it : zipIter) {
    res.push_back(fun(std::get<0>(it), std::get<1>(it)));
  }
  return res;
}

/// Apply with iterators.
template <typename Fn, typename IterType>
void apply(Fn fun, IterType begin, IterType end) {
  // auto i works with both pointer types and value types with an operator*.
  // auto *i only works for pointer types.
  for (auto i = begin; i != end; ++i) {
    fun(*i);
  }
}

/// Apply with templated container.
template <typename Fn, typename ContainerType>
void apply(Fn fun, ContainerType input) {
  return apply(fun, std::begin(input), std::end(input));
}

/// Zip apply with 2 templated container, iterates to the min of the sizes of
/// the 2 containers.
/// TODO(ntv): make variadic when needed.
template <typename Fn, typename ContainerType1, typename ContainerType2>
void zipApply(Fn fun, ContainerType1 input1, ContainerType2 input2) {
  auto zipIter = llvm::zip(input1, input2);
  for (auto it : zipIter) {
    fun(std::get<0>(it), std::get<1>(it));
  }
}

/// Unwraps a pointer type to another type (possibly the same).
/// Used in particular to allow easier compositions of
///   Operation::operand_range types.
template <typename T, typename ToType = T>
inline std::function<ToType *(T *)> makePtrDynCaster() {
  return [](T *val) { return dyn_cast<ToType>(val); };
}

/// Simple ScopeGuard.
struct ScopeGuard {
  explicit ScopeGuard(std::function<void(void)> destruct)
      : destruct(destruct) {}
  ~ScopeGuard() { destruct(); }

private:
  std::function<void(void)> destruct;
};

} // namespace functional
} // namespace mlir

#endif // MLIR_SUPPORT_FUNCTIONAL_H_
