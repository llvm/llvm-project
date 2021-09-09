//===- TosaInferShapes.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Propogate shapes forward along TOSA operations to resolve dynamic shape
// operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/PassDetail.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

void propagateShapesInRegion(Region &region);

void propagateShapesToTosaIf(Operation &op) {
  tosa::IfOp ifOp = dyn_cast<tosa::IfOp>(op);
  if (!ifOp)
    return;

  for (auto &region : op.getRegions()) {
    Block &frontBlock = region.front();
    if (frontBlock.getNumArguments() + 1 != ifOp.getNumOperands())
      return;

    for (int i = 0, e = frontBlock.getNumArguments(); i < e; i++) {
      ValueKnowledge operandKnowledge = ValueKnowledge::getKnowledgeFromType(
          ifOp.getOperand(i + 1).getType());
      ValueKnowledge blockKnowledge = ValueKnowledge::getKnowledgeFromType(
          frontBlock.getArgument(i).getType());
      ValueKnowledge joinedKnowledge =
          ValueKnowledge::join(operandKnowledge, blockKnowledge);
      if (!joinedKnowledge)
        continue;
      frontBlock.getArgument(i).setType(joinedKnowledge.getType());
    }

    propagateShapesInRegion(region);
  }

  return;
}

void propagateShapesInRegion(Region &region) {
  DenseMap<Value, ShapedTypeComponents> shapesStorage;
  auto setShapes = [&](Value val, Type t) {
    if (auto st = t.dyn_cast<ShapedType>())
      shapesStorage[val] = st;
    else
      shapesStorage[val] = t;
  };
  auto operandShape = [&](Value val) -> ShapeAdaptor {
    // Query the WIP mapping rather than the type if set.
    auto it = shapesStorage.find(val);
    if (it == shapesStorage.end())
      return nullptr;
    return it->second;
  };

  for (auto &block : region) {
    for (Operation &op : block) {
      if (op.getDialect()->getNamespace() !=
          tosa::TosaDialect::getDialectNamespace())
        continue;

      propagateShapesToTosaIf(op);

      InferShapedTypeOpInterface shapeInterface =
          dyn_cast<InferShapedTypeOpInterface>(op);
      if (!shapeInterface)
        continue;

      SmallVector<ShapedTypeComponents> returnedShapes;

      ValueShapeRange range(op.getOperands(), operandShape);
      if (shapeInterface
              .inferReturnTypeComponents(op.getContext(), op.getLoc(), range,
                                         op.getAttrDictionary(),
                                         op.getRegions(), returnedShapes)
              .succeeded()) {
        for (auto it : llvm::zip(op.getResults(), returnedShapes)) {
          Value result = std::get<0>(it);
          ShapedTypeComponents predictedShape = std::get<1>(it);

          // Check whether this use case is replaceable. We define an op as
          // being replaceable if it is used by a ReturnOp or a TosaOp.
          bool replaceable = true;
          for (auto user : result.getUsers()) {
            if (isa<ReturnOp>(user))
              continue;
            if (user->getDialect()->getNamespace() ==
                tosa::TosaDialect::getDialectNamespace())
              continue;

            replaceable = false;
          }

          // Determine the knowledge based on the output type.
          // TODO: should also query WIP type probably
          Type resultTy = result.getType();
          auto currentKnowledge =
              ValueKnowledge::getKnowledgeFromType(resultTy);

          // Compute the knowledge based on the inferred type.
          auto inferredKnowledge = ValueKnowledge::getPessimisticValueState();
          inferredKnowledge.dtype =
              resultTy.cast<ShapedType>().getElementType();
          inferredKnowledge.hasRank = predictedShape.hasRank();
          if (predictedShape.hasRank()) {
            for (auto dim : predictedShape.getDims()) {
              inferredKnowledge.sizes.push_back(dim);
            }
          }

          if (!replaceable)
            continue;

          // Compute the new type based on the joined version.
          auto newKnowledge =
              ValueKnowledge::join(currentKnowledge, inferredKnowledge);
          if (!newKnowledge)
            continue;
          setShapes(result, newKnowledge.getType());
        }
      }
    }
  }

  // Actually update types with updated shape knowledge.
  for (auto it : shapesStorage) {
    auto result = it.second;
    if (result.hasRank()) {
      Type t = it.first.getType().cast<ShapedType>().clone(result.getDims());
      it.first.setType(t);
    }
  }
}

/// Pass that performs shape propagation across TOSA operations. This includes
/// migrating to within the regions of if/while operations.
struct TosaInferShapes : public TosaInferShapesBase<TosaInferShapes> {
public:
  void runOnFunction() override {
    FuncOp func = getOperation();

    IRRewriter rewriter(func.getContext());

    propagateShapesInRegion(func.body());

    // Insert UnrealizedConversionCasts to guarantee ReturnOp agress with
    // the FuncOp type.
    func.walk([&](ReturnOp op) {
      FuncOp parent = dyn_cast<FuncOp>(op->getParentOp());
      if (!parent)
        return;

      rewriter.setInsertionPoint(op);
      FunctionType funcTy = func.getType();
      auto resultTys = funcTy.getResults();

      bool castAdded = false;
      SmallVector<Value> castedValues;
      for (auto it : llvm::zip(op->getOperands(), resultTys)) {
        auto operand = std::get<0>(it);
        auto currentTy = operand.getType();
        auto castTy = std::get<1>(it);
        if (currentTy == castTy) {
          castedValues.push_back(operand);
          continue;
        }

        castedValues.push_back(
            rewriter.create<tensor::CastOp>(op.getLoc(), castTy, operand)
                .getResult());

        castAdded = true;
      }

      if (castAdded) {
        rewriter.replaceOpWithNewOp<ReturnOp>(op, castedValues);
      }
    });
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::tosa::createTosaInferShapesPass() {
  return std::make_unique<TosaInferShapes>();
}
