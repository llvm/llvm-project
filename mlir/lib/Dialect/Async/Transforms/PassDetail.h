//===- PassDetail.h - Async Pass class details ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_ASYNC_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_ASYNC_TRANSFORMS_PASSDETAIL_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace arith {
class ArithDialect;
} // namespace arith

namespace async {
class AsyncDialect;
} // namespace async

namespace scf {
class SCFDialect;
} // namespace scf

// -------------------------------------------------------------------------- //
// Utility functions shared by Async Transformations.
// -------------------------------------------------------------------------- //

// Forward declarations.
class OpBuilder;

namespace async {

/// Clone ConstantLike operations that are defined above the given region and
/// have users in the region into the region entry block. We do that to reduce
/// the number of function arguments when we outline `async.execute` and
/// `scf.parallel` operations body into functions.
void cloneConstantsIntoTheRegion(Region &region);
void cloneConstantsIntoTheRegion(Region &region, OpBuilder &builder);

} // namespace async

} // namespace mlir

#endif // DIALECT_ASYNC_TRANSFORMS_PASSDETAIL_H_
