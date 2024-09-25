//===- TosaInferShapes.cpp ------------------------------------------------===//
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

#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAINFERSHAPES
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

// Check whether this use case is replaceable. We define an op as
// being replaceable if it is used by a TosaOp, or an op with a
// type-inference related interface.
// When a non-replaceable use is encountered, the value is wrapped in a
// cast back to the original type after inference.
bool canBeRefined(Operation *user) {
  if (!user->getDialect())
    return false;
  return user->getDialect()->getTypeID() == TypeID::get<TosaDialect>() ||
         isa<InferTypeOpInterface, InferShapedTypeOpInterface>(user);
}

// During type propagation, the types of values in the operator graph are
// updated. For the tosa.while_loop operation, types are speculatively updated
// within the body region to determine the output type of the while_loop. This
// process is performed until a fixed point is reached, then the types are
// rolled back.
//
// This class encapsulates the state information needed to perform the roll back
// process or to commit to the final changes.
class TypeModificationState {
public:
  TypeModificationState() = default;

  ~TypeModificationState() {
    // Ensure the recorded modifications are either committed or rolled back.
    assert(oldTypes.empty() && "unhandled type modifications");
  }

  // Update the state of the value and record the old type.
  void setType(Value value, Type type) {
    if (value.getType() != type) {
      oldTypes.emplace_back(value, value.getType());
      value.setType(type);
    }
  }

  // Roll back changes made to the types in the IR by setting all the affected
  // values to their old types.
  void rollBack() {
    for (auto [value, type] : oldTypes)
      value.setType(type);

    oldTypes.clear();
  }

  // Commit the changes to the types in the IR.
  // This requires inserting tensor.cast operations to mediate the newly
  // inferred result types with users that do not support type inference.
  void commit() {
    // For each use whose type changed, cast the value with the new type back to
    // the old type.
    for (auto [value, oldType] : oldTypes) {
      // The call to 'use->set()' in the body of the loop below invalidates the
      // iterator used to traverse op uses, so it is important to make a copy of
      // these first.
      llvm::SmallVector<OpOperand *> uses = llvm::map_to_vector(
          value.getUses(),
          [](OpOperand &use) -> OpOperand * {
            return &use;
          });

      // A 'tensor.cast' op is emitted only if needed. Once emitted, it is
      // cached and reused by all consumers.
      tensor::CastOp castValue;

      // Traverse all uses
      for (OpOperand *use : uses) {
        if (canBeRefined(use->getOwner()))
          continue;

        if (!castValue) {
          // Set the insertion point as far back as possible, since new
          // consumers of the 'tensor.cast' op generated in future iterations
          // are likely to be further up in the code due to the order in which
          // they appear in the use list.
          OpBuilder builder{value.getContext()};
          builder.setInsertionPointAfter(value.getDefiningOp());
          castValue =
              builder.create<tensor::CastOp>(value.getLoc(), oldType, value);
        }

        use->set(castValue);
      }
    }

    oldTypes.clear();
  }

private:
  // A record of each value whose type was updated along with that value's
  // previous type.
  llvm::SmallVector<std::pair<Value, Type>> oldTypes;
};

void propagateShapesInRegion(Region &region, TypeModificationState &state);

void propagateShapesToTosaIf(Operation &op, TypeModificationState &state) {
  IfOp ifOp = dyn_cast<IfOp>(op);
  if (!ifOp)
    return;

  for (auto &region : op.getRegions()) {
    Block &frontBlock = region.front();
    if (frontBlock.getNumArguments() + 1 != ifOp.getNumOperands())
      return;

    for (unsigned int i = 1, s = op.getNumOperands(); i < s; i++) {
      auto inferredTy = cast<ShapedType>(op.getOperand(i).getType());
      auto blockArg = frontBlock.getArgument(i - 1);
      auto oldType = cast<ShapedType>(blockArg.getType());

      if (inferredTy.hasRank()) {
        Type newType = oldType.clone(inferredTy.getShape());
        state.setType(blockArg, newType);
      }
    }

    for (int i = 0, e = frontBlock.getNumArguments(); i < e; i++) {
      ValueKnowledge operandKnowledge = ValueKnowledge::getKnowledgeFromType(
          ifOp.getOperand(i + 1).getType());
      ValueKnowledge blockKnowledge = ValueKnowledge::getKnowledgeFromType(
          frontBlock.getArgument(i).getType());
      ValueKnowledge joinedKnowledge =
          ValueKnowledge::join(operandKnowledge, blockKnowledge);
      if (!joinedKnowledge)
        continue;
      state.setType(frontBlock.getArgument(i), joinedKnowledge.getType());
    }

    propagateShapesInRegion(region, state);
  }
}

void propagateShapesToTosaWhile(Operation &op, TypeModificationState &state) {
  WhileOp whileOp = dyn_cast<WhileOp>(op);
  if (!whileOp)
    return;

  // Determine what the expected argument types are to the cond/body blocks.
  // The expected arguments should be compatible with ever iteration of the
  // loop body / condition for tosa.while.
  SmallVector<Type> argTypes = llvm::to_vector(op.getOperandTypes());

  bool hasNewTypes = true;
  while (hasNewTypes) {
    TypeModificationState localState;

    // Set types on the block args.
    Region &bodyRegion = op.getRegion(1);
    Block &block = bodyRegion.front();
    for (int i = 0, s = argTypes.size(); i < s; i++) {
      localState.setType(block.getArgument(i), argTypes[i]);
    }

    // Propagate to the end.
    propagateShapesInRegion(bodyRegion, localState);

    // Find all the tosa yield types and verify there is a single one.
    llvm::SmallVector<YieldOp> yieldOps;
    for (auto &block : bodyRegion)
      if (auto yieldOp = dyn_cast<YieldOp>(block.getTerminator()))
        yieldOps.push_back(yieldOp);

    assert(yieldOps.size() == 1 && "missing or non-unique yield op");
    // Using the new tosa.yield operand types, infer the new subtypes.
    llvm::SmallVector<ValueKnowledge> yieldTypeInfo;
    for (auto ty : argTypes) {
      yieldTypeInfo.push_back(ValueKnowledge::getKnowledgeFromType(ty));
    }

    for (auto yieldOp : yieldOps) {
      for (const auto &it : llvm::enumerate(yieldOp.getOperands())) {
        auto newKnowledge =
            ValueKnowledge::getKnowledgeFromType(it.value().getType());
        yieldTypeInfo[it.index()] =
            ValueKnowledge::meet(yieldTypeInfo[it.index()], newKnowledge);
      }
    }

    // This should never happen.
    if (yieldTypeInfo.size() != argTypes.size()) {
      op.emitWarning("has a tosa.yield with the incorrect number of operands");
      return;
    }

    // Determine the new block args and see if any changed.
    hasNewTypes = false;
    for (int i = 0, s = yieldTypeInfo.size(); i < s; i++) {
      Type newType = yieldTypeInfo[i].getType();
      hasNewTypes |= (newType != argTypes[i]);
      argTypes[i] = newType;
    }

    // Roll back all changes made during the speculative part of the algorithm.
    localState.rollBack();
  }

  // We now set the block arguments according to the most recent shape
  // inference results. This gives us the block arg types for the next
  // iteration.
  for (auto &region : op.getRegions()) {
    for (unsigned int i = 0, s = argTypes.size(); i < s; i++) {
      state.setType(region.front().getArgument(i), argTypes[i]);
    }

    propagateShapesInRegion(region, state);
  }
}

void propagateShapesInRegion(Region &region, TypeModificationState &state) {
  Dialect *tosaDialect = region.getContext()->getLoadedDialect<TosaDialect>();

  for (auto &block : region) {
    for (Operation &op : block) {
      if (op.getDialect() != tosaDialect)
        continue;

      propagateShapesToTosaIf(op, state);
      propagateShapesToTosaWhile(op, state);

      InferShapedTypeOpInterface shapeInterface =
          dyn_cast<InferShapedTypeOpInterface>(op);
      if (!shapeInterface)
        continue;

      SmallVector<ShapedTypeComponents> returnedShapes;

      if (shapeInterface
              .inferReturnTypeComponents(
                  op.getContext(), op.getLoc(), op.getOperands(),
                  op.getDiscardableAttrDictionary(), op.getPropertiesStorage(),
                  op.getRegions(), returnedShapes)
              .succeeded()) {
        for (auto it : llvm::zip(op.getResults(), returnedShapes)) {
          Value result = std::get<0>(it);
          ShapedTypeComponents predictedShape = std::get<1>(it);

          // Determine the knowledge based on the output type.
          // TODO: should also query WIP type probably
          Type resultTy = result.getType();
          auto currentKnowledge =
              ValueKnowledge::getKnowledgeFromType(resultTy);

          // Compute the knowledge based on the inferred type.
          auto inferredKnowledge = ValueKnowledge::getPessimisticValueState();
          inferredKnowledge.dtype = cast<ShapedType>(resultTy).getElementType();
          inferredKnowledge.hasRank = predictedShape.hasRank();
          if (predictedShape.hasRank()) {
            for (auto dim : predictedShape.getDims()) {
              inferredKnowledge.sizes.push_back(dim);
            }
          }

          // Compute the new type based on the joined version.
          auto newKnowledge =
              ValueKnowledge::join(currentKnowledge, inferredKnowledge);
          if (!newKnowledge)
            continue;

          // Set new type
          state.setType(result, newKnowledge.getType());
        }
      }
    }
  }
}

/// Pass that performs shape propagation across TOSA operations. This includes
/// migrating to within the regions of if/while operations.
struct TosaInferShapes
    : public tosa::impl::TosaInferShapesBase<TosaInferShapes> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    TypeModificationState state;
    propagateShapesInRegion(func.getBody(), state);
    state.commit();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaInferShapesPass() {
  return std::make_unique<TosaInferShapes>();
}
