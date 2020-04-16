//===- FoldedIntrinsics.h - MLIR EDSC Intrinsics for Linalg -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_LINALG_EDSC_FOLDEDINTRINSICS_H_
#define MLIR_DIALECT_LINALG_EDSC_FOLDEDINTRINSICS_H_

#include "mlir/Dialect/Linalg/EDSC/Builders.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"

#include "mlir/Transforms/FoldUtils.h"

namespace mlir {
namespace edsc {

template <typename Op, typename... Args>
ValueHandle ValueHandle::create(OperationFolder *folder, Args... args) {
  return folder ? ValueHandle(folder->create<Op>(ScopedContext::getBuilder(),
                                                 ScopedContext::getLocation(),
                                                 args...))
                : ValueHandle(ScopedContext::getBuilder().create<Op>(
                      ScopedContext::getLocation(), args...));
}

} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_EDSC_FOLDEDINTRINSICS_H_
