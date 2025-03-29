//===- ExpandRealloc.cpp - Expand memref.realloc ops into it's components -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_EXPANDREALLOC
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

namespace {

/// The `realloc` operation performs a conditional allocation and copy to
/// increase the size of a buffer if necessary. This pattern converts the
/// `realloc` operation into this sequence of simpler operations.

/// Example of an expansion:
/// ```mlir
/// %realloc = memref.realloc %alloc (%size) : memref<?xf32> to memref<?xf32>
/// ```
/// is expanded to
/// ```mlir
/// %c0 = arith.constant 0 : index
/// %dim = memref.dim %alloc, %c0 : memref<?xf32>
/// %is_old_smaller = arith.cmpi ult, %dim, %arg1
/// %realloc = scf.if %is_old_smaller -> (memref<?xf32>) {
///   %new_alloc = memref.alloc(%size) : memref<?xf32>
///   %subview = memref.subview %new_alloc[0] [%dim] [1]
///   memref.copy %alloc, %subview
///   memref.dealloc %alloc
///   scf.yield %alloc_0 : memref<?xf32>
/// } else {
///   %reinterpret_cast = memref.reinterpret_cast %alloc to
///     offset: [0], sizes: [%size], strides: [1]
///   scf.yield %reinterpret_cast : memref<?xf32>
/// }
/// ```
struct ExpandReallocOpPattern : public OpRewritePattern<memref::ReallocOp> {
  ExpandReallocOpPattern(MLIRContext *ctx, bool emitDeallocs)
      : OpRewritePattern(ctx), emitDeallocs(emitDeallocs) {}

  LogicalResult matchAndRewrite(memref::ReallocOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    assert(op.getType().getRank() == 1 &&
           "result MemRef must have exactly one rank");
    assert(op.getSource().getType().getRank() == 1 &&
           "source MemRef must have exactly one rank");
    assert(op.getType().getLayout().isIdentity() &&
           "result MemRef must have identity layout (or none)");
    assert(op.getSource().getType().getLayout().isIdentity() &&
           "source MemRef must have identity layout (or none)");

    // Get the size of the original buffer.
    int64_t inputSize =
        cast<BaseMemRefType>(op.getSource().getType()).getDimSize(0);
    OpFoldResult currSize = rewriter.getIndexAttr(inputSize);
    if (ShapedType::isDynamic(inputSize)) {
      Value dimZero = getValueOrCreateConstantIndexOp(rewriter, loc,
                                                      rewriter.getIndexAttr(0));
      currSize = rewriter.create<memref::DimOp>(loc, op.getSource(), dimZero)
                     .getResult();
    }

    // Get the requested size that the new buffer should have.
    int64_t outputSize =
        cast<BaseMemRefType>(op.getResult().getType()).getDimSize(0);
    OpFoldResult targetSize = ShapedType::isDynamic(outputSize)
                                  ? OpFoldResult{op.getDynamicResultSize()}
                                  : rewriter.getIndexAttr(outputSize);

    // Only allocate a new buffer and copy over the values in the old buffer if
    // the old buffer is smaller than the requested size.
    Value lhs = getValueOrCreateConstantIndexOp(rewriter, loc, currSize);
    Value rhs = getValueOrCreateConstantIndexOp(rewriter, loc, targetSize);
    Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                lhs, rhs);
    auto ifOp = rewriter.create<scf::IfOp>(
        loc, cond,
        [&](OpBuilder &builder, Location loc) {
          // Allocate the new buffer. If it is a dynamic memref we need to pass
          // an additional operand for the size at runtime, otherwise the static
          // size is encoded in the result type.
          SmallVector<Value> dynamicSizeOperands;
          if (op.getDynamicResultSize())
            dynamicSizeOperands.push_back(op.getDynamicResultSize());

          Value newAlloc = builder.create<memref::AllocOp>(
              loc, op.getResult().getType(), dynamicSizeOperands,
              op.getAlignmentAttr());

          // Take a subview of the new (bigger) buffer such that we can copy the
          // old values over (the copy operation requires both operands to have
          // the same shape).
          Value subview = builder.create<memref::SubViewOp>(
              loc, newAlloc, ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0)},
              ArrayRef<OpFoldResult>{currSize},
              ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
          builder.create<memref::CopyOp>(loc, op.getSource(), subview);

          // Insert the deallocation of the old buffer only if requested
          // (enabled by default).
          if (emitDeallocs)
            builder.create<memref::DeallocOp>(loc, op.getSource());

          builder.create<scf::YieldOp>(loc, newAlloc);
        },
        [&](OpBuilder &builder, Location loc) {
          // We need to reinterpret-cast here because either the input or output
          // type might be static, which means we need to cast from static to
          // dynamic or vice-versa. If both are static and the original buffer
          // is already bigger than the requested size, the cast represents a
          // subview operation.
          Value casted = builder.create<memref::ReinterpretCastOp>(
              loc, cast<MemRefType>(op.getResult().getType()), op.getSource(),
              rewriter.getIndexAttr(0), ArrayRef<OpFoldResult>{targetSize},
              ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
          builder.create<scf::YieldOp>(loc, casted);
        });

    rewriter.replaceOp(op, ifOp.getResult(0));
    return success();
  }

private:
  const bool emitDeallocs;
};

struct ExpandReallocPass
    : public memref::impl::ExpandReallocBase<ExpandReallocPass> {
  ExpandReallocPass(bool emitDeallocs)
      : memref::impl::ExpandReallocBase<ExpandReallocPass>() {
    this->emitDeallocs.setValue(emitDeallocs);
  }
  void runOnOperation() override {
    MLIRContext &ctx = getContext();

    RewritePatternSet patterns(&ctx);
    memref::populateExpandReallocPatterns(patterns, emitDeallocs.getValue());
    ConversionTarget target(ctx);

    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                           memref::MemRefDialect>();
    target.addIllegalOp<memref::ReallocOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::memref::populateExpandReallocPatterns(RewritePatternSet &patterns,
                                                 bool emitDeallocs) {
  patterns.add<ExpandReallocOpPattern>(patterns.getContext(), emitDeallocs);
}

std::unique_ptr<Pass> mlir::memref::createExpandReallocPass(bool emitDeallocs) {
  return std::make_unique<ExpandReallocPass>(emitDeallocs);
}
