//===- WrapFuncInClass.cpp - Wrap Emitc Funcs in classes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

using namespace mlir;
using namespace emitc;

namespace mlir {
namespace emitc {
#define GEN_PASS_DEF_WRAPFUNCINCLASSPASS
#include "mlir/Dialect/EmitC/Transforms/Passes.h.inc"

namespace {
struct WrapFuncInClassPass
    : public impl::WrapFuncInClassPassBase<WrapFuncInClassPass> {
  using WrapFuncInClassPassBase::WrapFuncInClassPassBase;
  void runOnOperation() override {
    Operation *rootOp = getOperation();

    RewritePatternSet patterns(&getContext());
    populateFuncPatterns(patterns, namedAttribute);

    walkAndApplyPatterns(rootOp, std::move(patterns));
  }
};

} // namespace
} // namespace emitc
} // namespace mlir

class WrapFuncInClass : public OpRewritePattern<emitc::FuncOp> {
public:
  WrapFuncInClass(MLIRContext *context, StringRef attrName)
      : OpRewritePattern<emitc::FuncOp>(context), attributeName(attrName) {}

  LogicalResult matchAndRewrite(emitc::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {

    auto className = funcOp.getSymNameAttr().str() + "Class";
    ClassOp newClassOp = ClassOp::create(rewriter, funcOp.getLoc(), className);

    SmallVector<std::pair<StringAttr, TypeAttr>> fields;
    rewriter.createBlock(&newClassOp.getBody());
    rewriter.setInsertionPointToStart(&newClassOp.getBody().front());

    auto argAttrs = funcOp.getArgAttrs();
    for (auto [idx, val] : llvm::enumerate(funcOp.getArguments())) {
      StringAttr fieldName;
      Attribute argAttr = nullptr;

      fieldName = rewriter.getStringAttr("fieldName" + std::to_string(idx));
      if (argAttrs && idx < argAttrs->size())
        argAttr = (*argAttrs)[idx];

      TypeAttr typeAttr = TypeAttr::get(val.getType());
      fields.push_back({fieldName, typeAttr});
      emitc::FieldOp::create(rewriter, funcOp.getLoc(), fieldName, typeAttr,
                             argAttr);
    }

    rewriter.setInsertionPointToEnd(&newClassOp.getBody().front());
    FunctionType funcType = funcOp.getFunctionType();
    Location loc = funcOp.getLoc();
    FuncOp newFuncOp =
        emitc::FuncOp::create(rewriter, loc, ("execute"), funcType);

    rewriter.createBlock(&newFuncOp.getBody());
    newFuncOp.getBody().takeBody(funcOp.getBody());

    rewriter.setInsertionPointToStart(&newFuncOp.getBody().front());
    std::vector<Value> newArguments;
    newArguments.reserve(fields.size());
    for (auto &[fieldName, attr] : fields) {
      GetFieldOp arg =
          emitc::GetFieldOp::create(rewriter, loc, attr.getValue(), fieldName);
      newArguments.push_back(arg);
    }

    for (auto [oldArg, newArg] :
         llvm::zip(newFuncOp.getArguments(), newArguments)) {
      rewriter.replaceAllUsesWith(oldArg, newArg);
    }

    llvm::BitVector argsToErase(newFuncOp.getNumArguments(), true);
    if (failed(newFuncOp.eraseArguments(argsToErase)))
      newFuncOp->emitOpError("failed to erase all arguments using BitVector");

    rewriter.replaceOp(funcOp, newClassOp);
    return success();
  }

private:
  StringRef attributeName;
};

void mlir::emitc::populateFuncPatterns(RewritePatternSet &patterns,
                                       StringRef namedAttribute) {
  patterns.add<WrapFuncInClass>(patterns.getContext(), namedAttribute);
}
