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
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

// -----------------------------------------------------------------------------
// Analysis.
// -----------------------------------------------------------------------------

static Type joinElementTypes(Type lhs, Type rhs) {
  return lhs == rhs ? lhs : Type();
}

namespace {
// Statically known information for a particular Value.
//
// This struct currently tracks only information relevant for tensor/array-like
// shaped types. It is fine to associate a `ValueKnowledge` with a non-shaped
// type as long as it is in the default "no knowledge" state returned by
// `getPessimisticValueState`. The important invariant is that we cannot
// claim to know something about a value which is false.
//
// This class could also be called "dataflow facts", "lattice value", etc.
struct ValueKnowledge {
  ValueKnowledge() = delete;
  ValueKnowledge(bool hasSizes, std::vector<int64_t> sizes, Type dtype)
      : hasSizes(hasSizes), sizes(sizes), dtype(dtype) {
    assert(sizes.size() == 0 || hasSizes);
  }

  // Get the static knowledge intrinsic to `type`.
  static ValueKnowledge getKnowledgeFromType(Type type) {
    ValueKnowledge result = getPessimisticValueState(type.getContext());
    if (auto shapedType = type.dyn_cast<ShapedType>()) {
      if (shapedType.hasRank()) {
        result.hasSizes = true;
        result.sizes = shapedType.getShape();
      }
      result.dtype = shapedType.getElementType();
    }
    return result;
  }

  // Return a pessimistic/conservative value state without assuming any knowlege
  // about the IR.
  static ValueKnowledge getPessimisticValueState(MLIRContext *context) {
    return ValueKnowledge(false, {}, Type());
  }

  Type getType() const {
    if (hasSizes) {
      return RankedTensorType::get(llvm::makeArrayRef(sizes), dtype);
    }
    return UnrankedTensorType::get(dtype);
  }

  bool operator==(const ValueKnowledge &rhs) const {
    return std::make_tuple(hasSizes, sizes, dtype) ==
           std::make_tuple(rhs.hasSizes, rhs.sizes, rhs.dtype);
  }

  // Given two pieces of static knowledge, calculate conservatively the
  // information we can be sure about.
  static ValueKnowledge join(const ValueKnowledge &lhs,
                             const ValueKnowledge &rhs) {
    // Mental model: All conditions are checking how to change from the safe "no
    // knowledge" default-initialized state to a state with more knowledge
    // consistent with lhs and rhs.
    ValueKnowledge result = getPessimisticValueState(nullptr);

    if (lhs.hasSizes && !rhs.hasSizes) {
      result.hasSizes = true;
      result.sizes = lhs.sizes;
    } else if (!lhs.hasSizes && rhs.hasSizes) {
      result.hasSizes = true;
      result.sizes = rhs.sizes;
    } else if (lhs.hasSizes && rhs.hasSizes &&
               lhs.sizes.size() == rhs.sizes.size()) {
      result.hasSizes = true;
      result.sizes.resize(lhs.sizes.size(), ShapedType::kDynamicSize);
      for (int i = 0, e = result.sizes.size(); i != e; i++) {
        int64_t lhsSize = lhs.sizes[i];
        int64_t rhsSize = rhs.sizes[i];
        int64_t &resultSize = result.sizes[i];
        if (lhsSize == ShapedType::kDynamicSize) {
          resultSize = rhsSize;
        } else if (rhsSize == ShapedType::kDynamicSize) {
          resultSize = lhsSize;
        } else if (lhsSize == rhsSize) {
          resultSize = lhsSize;
        }
      }
    }

    result.dtype = joinElementTypes(lhs.dtype, rhs.dtype);
    return result;
  }

  // Whether the Value is known to have a list of sizes.
  bool hasSizes;
  // If `hasSizes`, the sizes along each rank. Unknown sizes are represented as
  // `ShapedType::kDynamicSize`.
  std::vector<int64_t> sizes;
  // The dtype of a tensor.
  // This is equal to nullptr if we don't know that it is a specific concrete
  // type.
  Type dtype;
};

} // namespace

/// Pass that enables broadcast by making all input arrays have the same
/// number of dimensions. Insert RESHAPE operations to lower rank operand
struct TosaInferShapes : public TosaInferShapesBase<TosaInferShapes> {
public:
  void runOnFunction() override {
    FuncOp func = getOperation();

    IRRewriter rewriter(func.getContext());

    func.walk([&](Operation *op) {
      if (op->getDialect()->getNamespace() !=
          tosa::TosaDialect::getDialectNamespace())
        return;
      InferShapedTypeOpInterface shapeInterface =
          dyn_cast<InferShapedTypeOpInterface>(op);
      if (!shapeInterface)
        return;

      SmallVector<ShapedTypeComponents> returnedShapes;
      if (shapeInterface
              .inferReturnTypeComponents(
                  op->getContext(), op->getLoc(), op->getOperands(),
                  op->getAttrDictionary(), op->getRegions(), returnedShapes)
              .succeeded()) {
        for (auto it : llvm::zip(op->getResults(), returnedShapes)) {
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
          Type resultTy = result.getType();
          auto currentKnowledge =
              ValueKnowledge::getKnowledgeFromType(resultTy);

          // Compute the knowledge based on the inferred type.
          auto inferredKnowledge =
              ValueKnowledge::getPessimisticValueState(op->getContext());
          inferredKnowledge.dtype =
              resultTy.cast<ShapedType>().getElementType();
          inferredKnowledge.hasSizes = predictedShape.hasRank();
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
          result.setType(newKnowledge.getType());
        }
      }
    });

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
