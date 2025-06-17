//===- ConvertFuncToClass.cpp - Convert functions to classes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Rewrite.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/LogicalResult.h"
#include <string>

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
    MLIRContext *context = rootOp->getContext();

    RewritePatternSet patterns(context);
    populateFuncPatterns(patterns, namedAttribute);

    if (failed(applyPatternsGreedily(rootOp, std::move(patterns))))
      return signalPassFailure();
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect>();
  }
};

} // namespace

} // namespace emitc
} // namespace mlir

using namespace mlir;
using namespace mlir::emitc;

class WrapFuncInClass : public OpRewritePattern<emitc::FuncOp> {
private:
  std::string attributeName;

public:
  WrapFuncInClass(MLIRContext *context, const std::string &attrName)
      : OpRewritePattern<emitc::FuncOp>(context), attributeName(attrName) {}

  LogicalResult matchAndRewrite(emitc::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    if (funcOp->getParentOfType<emitc::ClassOp>()) {
      return failure();
    }
    auto className = funcOp.getSymNameAttr().str() + "Class";
    mlir::emitc::ClassOp newClassOp =
        rewriter.create<emitc::ClassOp>(funcOp.getLoc(), className);

    SmallVector<std::pair<StringAttr, TypeAttr>> fields;
    rewriter.createBlock(&newClassOp.getBody());
    rewriter.setInsertionPointToStart(&newClassOp.getBody().front());

    auto argAttrs = funcOp.getArgAttrs();
    for (auto [idx, val] : llvm::enumerate(funcOp.getArguments())) {
      StringAttr fieldName;
      Attribute argAttr = nullptr;

      if (argAttrs && idx < argAttrs->size()) {
        if (DictionaryAttr dictAttr =
                dyn_cast<mlir::DictionaryAttr>((*argAttrs)[idx])) {
          if (auto namedAttr = dictAttr.getNamed(attributeName)) {
            Attribute nv = namedAttr->getValue();
            fieldName = cast<mlir::StringAttr>(cast<mlir::ArrayAttr>(nv)[0]);
            argAttr = (*argAttrs)[idx];
          }
        }
      }

      if (!fieldName) {
        fieldName = rewriter.getStringAttr("fieldName" + std::to_string(idx));
      }

      TypeAttr typeAttr = TypeAttr::get(val.getType());
      fields.push_back({fieldName, typeAttr});
      rewriter.create<emitc::FieldOp>(funcOp.getLoc(), fieldName, typeAttr,
                                      argAttr);
    }

    rewriter.setInsertionPointToEnd(&newClassOp.getBody().front());
    MLIRContext *funcContext = funcOp.getContext();
    ArrayRef<Type> inputTypes = funcOp.getFunctionType().getInputs();
    ArrayRef<Type> results = funcOp.getFunctionType().getResults();
    FunctionType funcType = FunctionType::get(funcContext, inputTypes, results);
    Location loc = funcOp.getLoc();
    FuncOp newFuncOp = rewriter.create<emitc::FuncOp>(
        loc, rewriter.getStringAttr("execute"), funcType);

    rewriter.createBlock(&newFuncOp.getBody());
    newFuncOp.getBody().takeBody(funcOp.getBody());

    rewriter.setInsertionPointToStart(&newFuncOp.getBody().front());
    std::vector<Value> newArguments;
    for (auto &[fieldName, attr] : fields) {
      GetFieldOp arg =
          rewriter.create<emitc::GetFieldOp>(loc, attr.getValue(), fieldName);
      newArguments.push_back(arg);
    }

    for (auto [oldArg, newArg] :
         llvm::zip(newFuncOp.getArguments(), newArguments)) {
      rewriter.replaceAllUsesWith(oldArg, newArg);
    }

    llvm::BitVector argsToErase(newFuncOp.getNumArguments(), true);
    if (failed(newFuncOp.eraseArguments(argsToErase))) {
      newFuncOp->emitOpError("Failed to erase all arguments using BitVector.");
    }

    rewriter.replaceOp(funcOp, newClassOp);
    return success();
  }
};

void mlir::emitc::populateFuncPatterns(RewritePatternSet &patterns,
                                       const std::string &namedAttribute) {
  patterns.add<WrapFuncInClass>(patterns.getContext(), namedAttribute);
}
