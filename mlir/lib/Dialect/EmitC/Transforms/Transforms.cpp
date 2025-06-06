//===- Transforms.cpp - Patterns and transforms for the EmitC dialect -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/Transforms/Transforms.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace emitc {

ExpressionOp createExpression(Operation *op, OpBuilder &builder) {
  assert(op->hasTrait<OpTrait::emitc::CExpression>() &&
         "Expected a C expression");

  // Create an expression yielding the value returned by op.
  assert(op->getNumResults() == 1 && "Expected exactly one result");
  Value result = op->getResult(0);
  Type resultType = result.getType();
  Location loc = op->getLoc();

  builder.setInsertionPointAfter(op);
  auto expressionOp = builder.create<emitc::ExpressionOp>(loc, resultType);

  // Replace all op's uses with the new expression's result.
  result.replaceAllUsesWith(expressionOp.getResult());

  // Create an op to yield op's value.
  Region &region = expressionOp.getRegion();
  Block &block = region.emplaceBlock();
  builder.setInsertionPointToEnd(&block);
  auto yieldOp = builder.create<emitc::YieldOp>(loc, result);

  // Move op into the new expression.
  op->moveBefore(yieldOp);

  return expressionOp;
}

ClassOp createClass(FuncOp funcOp, OpBuilder &builder) {
  builder.setInsertionPoint(funcOp);

  // 2. Create the class
  auto classOp = builder.create<emitc::ClassOp>(
      funcOp.getLoc(), builder.getStringAttr("MyModelClass"));

  // Create a block inside the class body and set insertion point
  builder.createBlock(&classOp.getBody());
  builder.setInsertionPointToStart(&classOp.getBody().front());

  // 3. Extract input/output names from function arguments
  SmallVector<std::pair<StringRef, Type>> fields;
  llvm::SmallDenseMap<Value, Value> argToFieldMap;

  auto argAttrs = funcOp.getArgAttrs();
  if (argAttrs) {
    for (const auto [arg, val] : zip(*argAttrs, funcOp.getArguments())) {
      if (auto da = dyn_cast<mlir::DictionaryAttr>(arg)) {
        auto nv = da.getNamed("tf_saved_model.index_path")->getValue();
        auto fieldName = cast<mlir::StringAttr>(cast<mlir::ArrayAttr>(nv)[0]);
        auto fieldType = emitc::LValueType::get(emitc::PointerType::get(
            dyn_cast_or_null<emitc::ArrayType>(val.getType())
                .getElementType()));
        fields.push_back({fieldName.str(), fieldType});

        // 4.Create the class fields
        auto typeAttr = TypeAttr::get(val.getType());
        mlir::Attribute emptyAttr = builder.getAttr<mlir::UnitAttr>();
        auto dictAttr = DictionaryAttr::get(
            builder.getContext(),
            {builder.getNamedAttr(fieldName.str(), emptyAttr)});
        builder.create<emitc::FieldOp>(funcOp.getLoc(), fieldName, typeAttr,
                                       /* attributes*/ dictAttr);
        // 5. Get the pointers to the class fields
        auto pointer = emitc::PointerType::get(
            dyn_cast_or_null<emitc::ArrayType>(val.getType()).getElementType());
        auto ptr = builder.create<emitc::GetFieldOp>(
            funcOp.getLoc(), pointer, val, "MyModelClass", fieldName);
        argToFieldMap[val] = ptr;
      }
    }
  }

  // Create the new function inside the class
  auto funcContext = funcOp.getContext();
  auto inputTypes = funcOp.getFunctionType().getInputs();
  auto results = funcOp.getFunctionType().getResults();
  auto funcType = FunctionType::get(funcContext, inputTypes, results);
  auto loc = funcOp.getLoc();
  auto newFuncOp = builder.create<emitc::FuncOp>(
      loc, builder.getStringAttr("execute"), funcType);

  builder.createBlock(&newFuncOp.getBody());
  builder.setInsertionPointToStart(&newFuncOp.getBody().front());

  // 7. Remap original arguments to field pointers
  IRMapping mapper;

  // 8. move or clone operations from original function
  auto body = llvm::make_early_inc_range(funcOp.getBody().front());
  for (Operation &opToClone : body) {
    if (isa<emitc::ConstantOp>(opToClone) ||
        isa<emitc::SubscriptOp>(opToClone) || isa<emitc::LoadOp>(opToClone) ||
        isa<emitc::AddOp>(opToClone) || isa<emitc::AssignOp>(opToClone) ||
        isa<emitc::ReturnOp>(opToClone)) {
      builder.clone(opToClone, mapper);
    } else {
      opToClone.emitOpError("Unsupported operation found");
    }
  }

  // if (funcOp->use_empty()) funcOp->erase();

  return classOp;
}

} // namespace emitc
} // namespace mlir

using namespace mlir;
using namespace mlir::emitc;

namespace {

struct FoldExpressionOp : public OpRewritePattern<ExpressionOp> {
  using OpRewritePattern<ExpressionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ExpressionOp expressionOp,
                                PatternRewriter &rewriter) const override {
    bool anythingFolded = false;
    for (Operation &op : llvm::make_early_inc_range(
             expressionOp.getBody()->without_terminator())) {
      // Don't fold expressions whose result value has its address taken.
      auto applyOp = dyn_cast<emitc::ApplyOp>(op);
      if (applyOp && applyOp.getApplicableOperator() == "&")
        continue;

      for (Value operand : op.getOperands()) {
        auto usedExpression =
            dyn_cast_if_present<ExpressionOp>(operand.getDefiningOp());

        if (!usedExpression)
          continue;

        // Don't fold expressions with multiple users: assume any
        // re-materialization was done separately.
        if (!usedExpression.getResult().hasOneUse())
          continue;

        // Don't fold expressions with side effects.
        if (usedExpression.hasSideEffects())
          continue;

        // Fold the used expression into this expression by cloning all
        // instructions in the used expression just before the operation using
        // its value.
        rewriter.setInsertionPoint(&op);
        IRMapping mapper;
        for (Operation &opToClone :
             usedExpression.getBody()->without_terminator()) {
          Operation *clone = rewriter.clone(opToClone, mapper);
          mapper.map(&opToClone, clone);
        }

        Operation *expressionRoot = usedExpression.getRootOp();
        Operation *clonedExpressionRootOp = mapper.lookup(expressionRoot);
        assert(clonedExpressionRootOp &&
               "Expected cloned expression root to be in mapper");
        assert(clonedExpressionRootOp->getNumResults() == 1 &&
               "Expected cloned root to have a single result");

        rewriter.replaceOp(usedExpression, clonedExpressionRootOp);
        anythingFolded = true;
      }
    }
    return anythingFolded ? success() : failure();
  }
};

} // namespace

void mlir::emitc::populateExpressionPatterns(RewritePatternSet &patterns) {
  patterns.add<FoldExpressionOp>(patterns.getContext());
}
