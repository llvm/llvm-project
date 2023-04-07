//===- ValueBoundsOpInterfaceImpl.cpp - Impl. of ValueBoundsOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;

namespace mlir {
namespace linalg {
namespace {

/// Helper structure that iterates over all LinalgOps in `OpTys` and registers
/// the `ValueBoundsOpInterface` with each of them.
template <typename... Ops> struct LinalgValueBoundsOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (Ops::template attachInterface<DstValueBoundsOpInterfaceExternalModel<Ops>>(
         *ctx),
     ...);
  }
};

} // namespace
} // namespace linalg
} // namespace mlir

void mlir::linalg::registerValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    // Register all Linalg structured ops.
    LinalgValueBoundsOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >::registerOpInterface(ctx);
  });
}
