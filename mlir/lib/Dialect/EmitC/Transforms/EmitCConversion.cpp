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

  llvm::LogicalResult matchAndRewrite(mlir::ModuleOp moduleOp, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
    for (mlir::ModuleOp nestedModuleOp : llvm::make_early_inc_range(moduleOp.getOps<mlir::ModuleOp>())) {
      auto funcOps = nestedModuleOp.getOps<mlir::func::FuncOp>();
      if (!funcOps.empty()) {
        rewriter.setInsertionPoint(nestedModuleOp);
        mlir::emitc::ClassOp classOp = mlir::emitc::ClassOp::create(rewriter, nestedModuleOp->getLoc(), nestedModuleOp.getSymNameAttr());
        mlir::Block *classBlock = rewriter.createBlock(&classOp.getBody());
        for (mlir::Operation &op : llvm::make_early_inc_range(nestedModuleOp.getBody()->without_terminator())) {
          op.moveBefore(classBlock, classBlock->end());
        }
        rewriter.eraseOp(nestedModuleOp);
      }
    }

    return llvm::success();
  }
};

void mlir::populateBuiltinModuleToEmitCPatterns(const EmitCTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<ModuleOpConversion>(typeConverter, patterns.getContext());
}