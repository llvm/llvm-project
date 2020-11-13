//===- PassDetail.h - Async Pass class details ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_ASYNC_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_ASYNC_TRANSFORMS_PASSDETAIL_H_

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace async {
class AsyncDialect;
} // namespace async

namespace scf {
class SCFDialect;
} // namespace scf

#define GEN_PASS_CLASSES
#include "mlir/Dialect/Async/Passes.h.inc"

} // namespace mlir

#endif // DIALECT_ASYNC_TRANSFORMS_PASSDETAIL_H_
