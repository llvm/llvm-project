//===- RuntimeOpVerification.cpp - Op Verification ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/RuntimeOpVerification.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/RuntimeVerifiableOpInterface.h"

namespace mlir {
namespace memref {
namespace {
struct ExpandShapeOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<ExpandShapeOpInterface,
                                                         ExpandShapeOp> {
  void generateRuntimeVerification(Operation *op, OpBuilder &builder,
                                   Location loc) const {
    auto expandShapeOp = cast<ExpandShapeOp>(op);

    // Verify that the expanded dim sizes are a product of the collapsed dim
    // size.
    for (const auto &it :
         llvm::enumerate(expandShapeOp.getReassociationIndices())) {
      Value srcDimSz =
          builder.create<DimOp>(loc, expandShapeOp.getSrc(), it.index());
      int64_t groupSz = 1;
      bool foundDynamicDim = false;
      for (int64_t resultDim : it.value()) {
        if (expandShapeOp.getResultType().isDynamicDim(resultDim)) {
          // Keep this assert here in case the op is extended in the future.
          assert(!foundDynamicDim &&
                 "more than one dynamic dim found in reassoc group");
          (void)foundDynamicDim;
          foundDynamicDim = true;
          continue;
        }
        groupSz *= expandShapeOp.getResultType().getDimSize(resultDim);
      }
      Value staticResultDimSz =
          builder.create<arith::ConstantIndexOp>(loc, groupSz);
      // staticResultDimSz must divide srcDimSz evenly.
      Value mod =
          builder.create<arith::RemSIOp>(loc, srcDimSz, staticResultDimSz);
      Value isModZero = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, mod,
          builder.create<arith::ConstantIndexOp>(loc, 0));
      builder.create<cf::AssertOp>(
          loc, isModZero,
          "static result dims in reassoc group do not divide src dim evenly");
    }
  }
};
} // namespace
} // namespace memref
} // namespace mlir

void mlir::memref::registerRuntimeVerifiableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    ExpandShapeOp::attachInterface<ExpandShapeOpInterface>(*ctx);

    // Load additional dialects of which ops may get created.
    ctx->loadDialect<arith::ArithDialect, cf::ControlFlowDialect>();
  });
}
