//===- Builders.h - MLIR Declarative Builder Classes ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides intuitive composable interfaces for building structured MLIR
// snippets in a declarative fashion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_EDSC_BUILDERS_H_
#define MLIR_DIALECT_AFFINE_EDSC_BUILDERS_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace edsc {

/// Creates a perfect nest of affine "for" loops, given the list of lower
/// bounds, upper bounds and steps. The three lists are expected to contain the
/// same number of elements. Uses the OpBuilder and Location stored in
/// ScopedContext and assumes they are non-null. The optional "bodyBuilderFn"
/// callback is called to construct the body of the innermost loop and is passed
/// the list of loop induction variables, in order from outermost to innermost.
/// The function is expected to use the builder and location stored in
/// ScopedContext at the moment of the call. The function should not create
/// the affine terminator op, which will be added regardless of the
/// "bodyBuilderFn" being present.
void affineLoopNestBuilder(
    ValueRange lbs, ValueRange ubs, ArrayRef<int64_t> steps,
    function_ref<void(ValueRange)> bodyBuilderFn = nullptr);

/// Creates a single affine "for" loop, iterating from max(lbs) to min(ubs) with
/// the given step. Uses the OpBuilder and Location stored in ScopedContext and
/// assumes they are non-null. The optional "bodyBuilderFn" callback is called
/// to construct the body of the loop and is passed the induction variable. The
/// function is expected to use the builder and location stored in ScopedContext
/// at the moment of the call. The function should not create the affine
/// terminator op, which will be added regardless of the "bodyBuilderFn" being
/// present.
void affineLoopBuilder(ValueRange lbs, ValueRange ubs, int64_t step,
                       function_ref<void(Value)> bodyBuilderFn = nullptr);

namespace op {

Value operator+(Value lhs, Value rhs);
Value operator-(Value lhs, Value rhs);
Value operator*(Value lhs, Value rhs);
Value operator/(Value lhs, Value rhs);
Value operator%(Value lhs, Value rhs);
Value floorDiv(Value lhs, Value rhs);
Value ceilDiv(Value lhs, Value rhs);

/// Logical operator overloadings.
Value negate(Value value);
Value operator&&(Value lhs, Value rhs);
Value operator||(Value lhs, Value rhs);
Value operator^(Value lhs, Value rhs);

/// Comparison operator overloadings.
Value eq(Value lhs, Value rhs);
Value ne(Value lhs, Value rhs);
Value operator<(Value lhs, Value rhs);
Value operator<=(Value lhs, Value rhs);
Value operator>(Value lhs, Value rhs);
Value operator>=(Value lhs, Value rhs);

} // namespace op

/// Arithmetic operator overloadings.
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator+(Value e) {
  using op::operator+;
  return static_cast<Value>(*this) + e;
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator-(Value e) {
  using op::operator-;
  return static_cast<Value>(*this) - e;
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator*(Value e) {
  using op::operator*;
  return static_cast<Value>(*this) * e;
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator/(Value e) {
  using op::operator/;
  return static_cast<Value>(*this) / e;
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator%(Value e) {
  using op::operator%;
  return static_cast<Value>(*this) % e;
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator^(Value e) {
  using op::operator^;
  return static_cast<Value>(*this) ^ e;
}

/// Assignment-arithmetic operator overloadings.
template <typename Load, typename Store>
Store TemplatedIndexedValue<Load, Store>::operator+=(Value e) {
  using op::operator+;
  return Store(*this + e, getBase(), indices);
}
template <typename Load, typename Store>
Store TemplatedIndexedValue<Load, Store>::operator-=(Value e) {
  using op::operator-;
  return Store(*this - e, getBase(), indices);
}
template <typename Load, typename Store>
Store TemplatedIndexedValue<Load, Store>::operator*=(Value e) {
  using op::operator*;
  return Store(*this * e, getBase(), indices);
}
template <typename Load, typename Store>
Store TemplatedIndexedValue<Load, Store>::operator/=(Value e) {
  using op::operator/;
  return Store(*this / e, getBase(), indices);
}
template <typename Load, typename Store>
Store TemplatedIndexedValue<Load, Store>::operator%=(Value e) {
  using op::operator%;
  return Store(*this % e, getBase(), indices);
}
template <typename Load, typename Store>
Store TemplatedIndexedValue<Load, Store>::operator^=(Value e) {
  using op::operator^;
  return Store(*this ^ e, getBase(), indices);
}

/// Logical operator overloadings.
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator&&(Value e) {
  using op::operator&&;
  return static_cast<Value>(*this) && e;
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator||(Value e) {
  using op::operator||;
  return static_cast<Value>(*this) || e;
}

/// Comparison operator overloadings.
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::eq(Value e) {
  return eq(value, e);
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::ne(Value e) {
  return ne(value, e);
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator<(Value e) {
  using op::operator<;
  return static_cast<Value>(*this) < e;
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator<=(Value e) {
  using op::operator<=;
  return static_cast<Value>(*this) <= e;
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator>(Value e) {
  using op::operator>;
  return static_cast<Value>(*this) > e;
}
template <typename Load, typename Store>
Value TemplatedIndexedValue<Load, Store>::operator>=(Value e) {
  using op::operator>=;
  return static_cast<Value>(*this) >= e;
}

} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_EDSC_BUILDERS_H_
