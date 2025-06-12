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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/GraphWriter.h"

namespace mlir {
namespace emitc {

#define GEN_PASS_DEF_WRAPFUNCINCLASSPASS
#include "mlir/Dialect/EmitC/Transforms/Passes.h.inc"

namespace {

struct WrapFuncInClassPass
    : public impl::WrapFuncInClassPassBase<WrapFuncInClassPass> {
  void runOnOperation() override {
    Operation *rootOp = getOperation();
    MLIRContext *context = rootOp->getContext();

    RewritePatternSet patterns(context);
    populateFuncPatterns(patterns);

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

static bool validOp(Operation &opToClone) {
  return isa<emitc::ConstantOp>(opToClone) ||
         isa<emitc::SubscriptOp>(opToClone) || isa<emitc::LoadOp>(opToClone) ||
         isa<emitc::AddOp>(opToClone) || isa<emitc::AssignOp>(opToClone) ||
         isa<emitc::ReturnOp>(opToClone);
}

class WrapFuncInClass : public OpRewritePattern<emitc::FuncOp> {
public:
  using OpRewritePattern<emitc::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(emitc::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    if (funcOp->getParentOfType<emitc::ClassOp>()) {
      return failure();
    }
    auto className = "My" + funcOp.getSymNameAttr().str() + "Class";
    mlir::emitc::ClassOp newClassOp =
        rewriter.create<emitc::ClassOp>(funcOp.getLoc(), className);

    SmallVector<std::pair<StringAttr, TypeAttr>> fields;
    rewriter.createBlock(&newClassOp.getBody());
    rewriter.setInsertionPointToStart(&newClassOp.getBody().front());

    auto argAttrs = funcOp.getArgAttrs();

    for (const auto &[arg, val] : (zip(*argAttrs, funcOp.getArguments()))) {
      // FIXME:How can we avoid hardcoding this name?
      // Should we loop through the dictionary and check for each named
      // attribute if attr.getName().getValue().contains("tf_saved_model")
      if (auto namedAttr = dyn_cast<mlir::DictionaryAttr>(arg).getNamed(
              "tf_saved_model.index_path")) {
        Attribute nv = namedAttr->getValue();
        StringAttr fieldName =
            cast<mlir::StringAttr>(cast<mlir::ArrayAttr>(nv)[0]);
        TypeAttr typeAttr = TypeAttr::get(val.getType());
        fields.push_back({fieldName, typeAttr});

        rewriter.create<emitc::FieldOp>(funcOp.getLoc(), fieldName, typeAttr,
                                        /* attributes*/ arg);
      } else
        funcOp->emitOpError("Only Covers TF models");
    }

    rewriter.setInsertionPointToEnd(&newClassOp.getBody().front());
    MLIRContext *funcContext = funcOp.getContext();
    ArrayRef<Type> inputTypes = funcOp.getFunctionType().getInputs();
    ArrayRef<Type> results = funcOp.getFunctionType().getResults();
    FunctionType funcType = FunctionType::get(funcContext, inputTypes, results);
    Location loc = funcOp.getLoc();
    FuncOp newFuncOp = rewriter.create<emitc::FuncOp>(
        loc, rewriter.getStringAttr("execute"), funcType);

    rewriter.setInsertionPointToStart(newFuncOp.addEntryBlock());

    std::vector<Value> newArguments;
    for (auto [fieldName, attr] : fields) {
      auto arg =
          rewriter.create<emitc::GetFieldOp>(loc, attr.getValue(), fieldName);
      newArguments.push_back(arg);
    }

    IRMapping mapper;
    for (auto [oldArg, newArg] :
         llvm::zip(funcOp.getArguments(), newArguments)) {
      mapper.map(oldArg, newArg);
    }

    while (!newFuncOp.getArguments().empty()) {
      if (failed(newFuncOp.eraseArgument(0))) {
        break;
      }
    }

    // TODO: The mapper is easier to use but cloning is more expensive than
    // moving the body. Working on changing this portion to move the body
    // instead
    auto body = llvm::make_early_inc_range(funcOp.getBody().front());
    for (Operation &opToClone : body) {
      if (validOp(opToClone)) {
        rewriter.clone(opToClone, mapper);
      } else {
        opToClone.emitOpError("Unsupported operation found");
      }
    }

    rewriter.replaceOp(funcOp, newClassOp);
    return funcOp->use_empty() ? success() : failure();
  }
};

void mlir::emitc::populateFuncPatterns(RewritePatternSet &patterns) {
  patterns.add<WrapFuncInClass>(patterns.getContext());
}
