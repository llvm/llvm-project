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

#ifndef MLIR_DIALECT_AFFINEOPS_EDSC_BUILDERS_H_
#define MLIR_DIALECT_AFFINEOPS_EDSC_BUILDERS_H_

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace edsc {

/// Constructs a new AffineForOp and captures the associated induction
/// variable. A ValueHandle pointer is passed as the first argument and is the
/// *only* way to capture the loop induction variable.
LoopBuilder makeAffineLoopBuilder(ValueHandle *iv,
                                  ArrayRef<ValueHandle> lbHandles,
                                  ArrayRef<ValueHandle> ubHandles,
                                  int64_t step);

/// Explicit nested LoopBuilder. Offers a compressed multi-loop builder to avoid
/// explicitly writing all the loops in a nest. This simple functionality is
/// also useful to write rank-agnostic custom ops.
///
/// Usage:
///
/// ```c++
///    AffineLoopNestBuilder({&i, &j, &k}, {lb, lb, lb}, {ub, ub, ub}, {1, 1,
///    1})(
///      [&](){
///        ...
///      });
/// ```
///
/// ```c++
///    AffineLoopNestBuilder({&i}, {lb}, {ub}, {1})([&](){
///      AffineLoopNestBuilder({&j}, {lb}, {ub}, {1})([&](){
///        AffineLoopNestBuilder({&k}, {lb}, {ub}, {1})([&](){
///          ...
///        }),
///      }),
///    });
/// ```
class AffineLoopNestBuilder {
public:
  /// This entry point accommodates the fact that AffineForOp implicitly uses
  /// multiple `lbs` and `ubs` with one single `iv` and `step` to encode `max`
  /// and and `min` constraints respectively.
  AffineLoopNestBuilder(ValueHandle *iv, ArrayRef<ValueHandle> lbs,
                        ArrayRef<ValueHandle> ubs, int64_t step);
  AffineLoopNestBuilder(ArrayRef<ValueHandle *> ivs, ArrayRef<ValueHandle> lbs,
                        ArrayRef<ValueHandle> ubs, ArrayRef<int64_t> steps);

  void operator()(function_ref<void(void)> fun = nullptr);

private:
  SmallVector<LoopBuilder, 4> loops;
};

namespace op {

ValueHandle operator+(ValueHandle lhs, ValueHandle rhs);
ValueHandle operator-(ValueHandle lhs, ValueHandle rhs);
ValueHandle operator*(ValueHandle lhs, ValueHandle rhs);
ValueHandle operator/(ValueHandle lhs, ValueHandle rhs);
ValueHandle operator%(ValueHandle lhs, ValueHandle rhs);
ValueHandle floorDiv(ValueHandle lhs, ValueHandle rhs);
ValueHandle ceilDiv(ValueHandle lhs, ValueHandle rhs);

ValueHandle operator!(ValueHandle value);
ValueHandle operator&&(ValueHandle lhs, ValueHandle rhs);
ValueHandle operator||(ValueHandle lhs, ValueHandle rhs);
ValueHandle operator^(ValueHandle lhs, ValueHandle rhs);
ValueHandle operator==(ValueHandle lhs, ValueHandle rhs);
ValueHandle operator!=(ValueHandle lhs, ValueHandle rhs);
ValueHandle operator<(ValueHandle lhs, ValueHandle rhs);
ValueHandle operator<=(ValueHandle lhs, ValueHandle rhs);
ValueHandle operator>(ValueHandle lhs, ValueHandle rhs);
ValueHandle operator>=(ValueHandle lhs, ValueHandle rhs);

} // namespace op

/// Arithmetic operator overloadings.
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator+(ValueHandle e) {
  using op::operator+;
  return static_cast<ValueHandle>(*this) + e;
}
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator-(ValueHandle e) {
  using op::operator-;
  return static_cast<ValueHandle>(*this) - e;
}
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator*(ValueHandle e) {
  using op::operator*;
  return static_cast<ValueHandle>(*this) * e;
}
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator/(ValueHandle e) {
  using op::operator/;
  return static_cast<ValueHandle>(*this) / e;
}
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator%(ValueHandle e) {
  using op::operator%;
  return static_cast<ValueHandle>(*this) % e;
}
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator^(ValueHandle e) {
  using op::operator^;
  return static_cast<ValueHandle>(*this) ^ e;
}

/// Assignment-arithmetic operator overloadings.
template <typename Load, typename Store>
OperationHandle TemplatedIndexedValue<Load, Store>::operator+=(ValueHandle e) {
  using op::operator+;
  return Store(*this + e, getBase(), {indices.begin(), indices.end()});
}
template <typename Load, typename Store>
OperationHandle TemplatedIndexedValue<Load, Store>::operator-=(ValueHandle e) {
  using op::operator-;
  return Store(*this - e, getBase(), {indices.begin(), indices.end()});
}
template <typename Load, typename Store>
OperationHandle TemplatedIndexedValue<Load, Store>::operator*=(ValueHandle e) {
  using op::operator*;
  return Store(*this * e, getBase(), {indices.begin(), indices.end()});
}
template <typename Load, typename Store>
OperationHandle TemplatedIndexedValue<Load, Store>::operator/=(ValueHandle e) {
  using op::operator/;
  return Store(*this / e, getBase(), {indices.begin(), indices.end()});
}
template <typename Load, typename Store>
OperationHandle TemplatedIndexedValue<Load, Store>::operator%=(ValueHandle e) {
  using op::operator%;
  return Store(*this % e, getBase(), {indices.begin(), indices.end()});
}
template <typename Load, typename Store>
OperationHandle TemplatedIndexedValue<Load, Store>::operator^=(ValueHandle e) {
  using op::operator^;
  return Store(*this ^ e, getBase(), {indices.begin(), indices.end()});
}

/// Logical operator overloadings.
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator&&(ValueHandle e) {
  using op::operator&&;
  return static_cast<ValueHandle>(*this) && e;
}
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator||(ValueHandle e) {
  using op::operator||;
  return static_cast<ValueHandle>(*this) || e;
}

/// Comparison operator overloadings.
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator==(ValueHandle e) {
  using op::operator==;
  return static_cast<ValueHandle>(*this) == e;
}
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator!=(ValueHandle e) {
  using op::operator!=;
  return static_cast<ValueHandle>(*this) != e;
}
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator<(ValueHandle e) {
  using op::operator<;
  return static_cast<ValueHandle>(*this) < e;
}
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator<=(ValueHandle e) {
  using op::operator<=;
  return static_cast<ValueHandle>(*this) <= e;
}
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator>(ValueHandle e) {
  using op::operator>;
  return static_cast<ValueHandle>(*this) > e;
}
template <typename Load, typename Store>
ValueHandle TemplatedIndexedValue<Load, Store>::operator>=(ValueHandle e) {
  using op::operator>=;
  return static_cast<ValueHandle>(*this) >= e;
}

} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_AFFINEOPS_EDSC_BUILDERS_H_
