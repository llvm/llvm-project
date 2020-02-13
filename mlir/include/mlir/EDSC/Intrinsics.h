//===- Intrinsics.h - MLIR Operations for Declarative Builders ---*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides intuitive composable intrinsics for building snippets of MLIR
// declaratively.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EDSC_INTRINSICS_H_
#define MLIR_EDSC_INTRINSICS_H_

#include "mlir/EDSC/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

class MemRefType;
class Type;

namespace edsc {

/// Entry point to build multiple ValueHandle* from a mutable list `ivs`.
inline SmallVector<ValueHandle *, 8>
makeHandlePointers(MutableArrayRef<ValueHandle> ivs) {
  SmallVector<ValueHandle *, 8> pivs;
  pivs.reserve(ivs.size());
  for (auto &iv : ivs)
    pivs.push_back(&iv);
  return pivs;
}

/// Provides a set of first class intrinsics.
/// In the future, most of intrinsics related to Operation that don't contain
/// other operations should be Tablegen'd.
namespace intrinsics {
namespace detail {
/// Helper structure to be used with ValueBuilder / OperationBuilder.
/// It serves the purpose of removing boilerplate specialization for the sole
/// purpose of implicitly converting ArrayRef<ValueHandle> -> ArrayRef<Value>.
class ValueHandleArray {
public:
  ValueHandleArray(ArrayRef<ValueHandle> vals) {
    values.append(vals.begin(), vals.end());
  }
  operator ArrayRef<Value>() { return values; }

private:
  ValueHandleArray() = default;
  SmallVector<Value, 8> values;
};

template <typename T>
inline T unpack(T value) {
  return value;
}

inline detail::ValueHandleArray unpack(ArrayRef<ValueHandle> values) {
  return detail::ValueHandleArray(values);
}

} // namespace detail

/// Helper variadic abstraction to allow extending to any MLIR op without
/// boilerplate or Tablegen.
/// Arguably a builder is not a ValueHandle but in practice it is only used as
/// an alias to a notional ValueHandle<Op>.
/// Implementing it as a subclass allows it to compose all the way to Value.
/// Without subclassing, implicit conversion to Value would fail when composing
/// in patterns such as: `select(a, b, select(c, d, e))`.
template <typename Op>
struct ValueBuilder : public ValueHandle {
  // Builder-based
  template <typename... Args>
  ValueBuilder(Args... args)
      : ValueHandle(ValueHandle::create<Op>(detail::unpack(args)...)) {}
  ValueBuilder(ArrayRef<ValueHandle> vs)
      : ValueBuilder(ValueBuilder::create<Op>(detail::unpack(vs))) {}
  template <typename... Args>
  ValueBuilder(ArrayRef<ValueHandle> vs, Args... args)
      : ValueHandle(ValueHandle::create<Op>(detail::unpack(vs),
                                            detail::unpack(args)...)) {}
  template <typename T, typename... Args>
  ValueBuilder(T t, ArrayRef<ValueHandle> vs, Args... args)
      : ValueHandle(ValueHandle::create<Op>(
            detail::unpack(t), detail::unpack(vs), detail::unpack(args)...)) {}
  template <typename T1, typename T2, typename... Args>
  ValueBuilder(T1 t1, T2 t2, ArrayRef<ValueHandle> vs, Args... args)
      : ValueHandle(ValueHandle::create<Op>(
            detail::unpack(t1), detail::unpack(t2), detail::unpack(vs),
            detail::unpack(args)...)) {}

  ValueBuilder() : ValueHandle(ValueHandle::create<Op>()) {}
};

template <typename Op>
struct OperationBuilder : public OperationHandle {
  template <typename... Args>
  OperationBuilder(Args... args)
      : OperationHandle(OperationHandle::create<Op>(detail::unpack(args)...)) {}
  OperationBuilder(ArrayRef<ValueHandle> vs)
      : OperationHandle(OperationHandle::create<Op>(detail::unpack(vs))) {}
  template <typename... Args>
  OperationBuilder(ArrayRef<ValueHandle> vs, Args... args)
      : OperationHandle(OperationHandle::create<Op>(detail::unpack(vs),
                                                    detail::unpack(args)...)) {}
  template <typename T, typename... Args>
  OperationBuilder(T t, ArrayRef<ValueHandle> vs, Args... args)
      : OperationHandle(OperationHandle::create<Op>(
            detail::unpack(t), detail::unpack(vs), detail::unpack(args)...)) {}
  template <typename T1, typename T2, typename... Args>
  OperationBuilder(T1 t1, T2 t2, ArrayRef<ValueHandle> vs, Args... args)
      : OperationHandle(OperationHandle::create<Op>(
            detail::unpack(t1), detail::unpack(t2), detail::unpack(vs),
            detail::unpack(args)...)) {}
  OperationBuilder() : OperationHandle(OperationHandle::create<Op>()) {}
};

} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_EDSC_INTRINSICS_H_
