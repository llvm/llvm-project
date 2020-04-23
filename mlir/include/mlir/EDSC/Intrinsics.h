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

/// Provides a set of first class intrinsics.
/// In the future, most of intrinsics related to Operation that don't contain
/// other operations should be Tablegen'd.
namespace intrinsics {

template <typename Op>
struct OperationBuilder : public OperationHandle {
  template <typename... Args>
  OperationBuilder(Args... args)
      : OperationHandle(OperationHandle::create<Op>(args...)) {}
  OperationBuilder(ArrayRef<Value> vs)
      : OperationHandle(OperationHandle::create<Op>(vs)) {}
  template <typename... Args>
  OperationBuilder(ArrayRef<Value> vs, Args... args)
      : OperationHandle(OperationHandle::create<Op>(vs, args...)) {}
  template <typename T, typename... Args>
  OperationBuilder(T t, ArrayRef<Value> vs, Args... args)
      : OperationHandle(OperationHandle::create<Op>(t, vs, args...)) {}
  template <typename T1, typename T2, typename... Args>
  OperationBuilder(T1 t1, T2 t2, ArrayRef<Value> vs, Args... args)
      : OperationHandle(OperationHandle::create<Op>(t1, t2, vs, args...)) {}
  OperationBuilder() : OperationHandle(OperationHandle::create<Op>()) {}
};

} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_EDSC_INTRINSICS_H_
