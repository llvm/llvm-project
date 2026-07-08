//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/EmitC/Transforms/EmitCConversion.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"


struct ModuleOpConversion final : mlir::OpConversionPattern<mlir::ModuleOp> {
  using Base::Base;

  llvm::LogicalResult matchAndRewrite(
      mlir::ModuleOp moduleOp, 
      OpAdaptor adaptor, 
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Operation *parentOp = moduleOp->getParentOp();
    if (!parentOp || !llvm::isa<mlir::ModuleOp>(parentOp)) {
      return llvm::failure();
    }

    auto funcOps = moduleOp.getOps<mlir::func::FuncOp>();
    if (funcOps.empty()) {
      return llvm::failure();
    }

    mlir::emitc::ClassOp classOp = mlir::emitc::ClassOp::create(
        rewriter, 
        moduleOp.getLoc(), 
        moduleOp.getSymNameAttr()
    );

    mlir::Block *classBlock = rewriter.createBlock(&classOp.getBody());

    for (mlir::Operation &op : llvm::make_early_inc_range(moduleOp.getBody()->without_terminator())) {
      rewriter.moveOpBefore(&op, classBlock, classBlock->end());
    }

    rewriter.eraseOp(moduleOp);

    return llvm::success();
  }
};

void mlir::populateBuiltinModuleToEmitCPatterns(const EmitCTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<ModuleOpConversion>(typeConverter, patterns.getContext());
}