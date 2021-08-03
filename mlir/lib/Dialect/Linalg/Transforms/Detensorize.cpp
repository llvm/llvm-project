//===- Detensorize.cpp - Linalg transformations as patterns ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iterator>
#include <memory>

using namespace mlir;
using namespace mlir::linalg;

static Value sourceMaterializationCallback(OpBuilder &builder, Type type,
                                           ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  // A detensored value is converted back by creating a new tensor from its
  // element(s).
  auto createNewTensorOp = builder.create<tensor::FromElementsOp>(
      loc, inputs[0].getType(), inputs[0]);

  // FromElementsOp results in a tensor<1xdtype>, we need to reshape that to
  // a tensor<dtype> instead.
  return builder.create<linalg::TensorCollapseShapeOp>(
      loc, type, createNewTensorOp, ArrayRef<ReassociationExprs>{});
}

namespace {
/// Defines the criteria a TensorType must follow in order to be considered
/// "detensorable".
///
/// NOTE: For now, only 0-D tensors are supported.
///
/// Returns true if tensorType can be detensored.
bool canBeDetensored(TensorType tensorType) {
  return tensorType.hasRank() && tensorType.getRank() == 0;
}

bool shouldBeDetensored(Operation *op, TypeConverter typeConverter) {
  GenericOp genericOp = dyn_cast_or_null<GenericOp>(op);
  return genericOp &&
         llvm::all_of(
             genericOp.getInputAndOutputOperands(), [&](OpOperand *opOperand) {
               return !typeConverter.isLegal(opOperand->get().getType());
             });
}

/// A conversion patttern for detensoring `linalg.generic` ops.
class DetensorizeGenericOp : public OpConversionPattern<GenericOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GenericOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Block *originalBlock = op->getBlock();

    // Gather some information about the op before inling its region.
    Block *opEntryBlock = &*op.region().begin();
    YieldOp yieldOp = dyn_cast<YieldOp>(op.region().back().getTerminator());

    // Split the op's region before the op. This way, we have a clear insertion
    // point in which the op can be inlined.
    Block *newBlock = originalBlock->splitBlock(op);
    rewriter.inlineRegionBefore(op.region(), newBlock);
    // Now that op's region is inlined, the operands of its YieldOp are mapped
    // to the materialized target values. Therefore, we can replace the op's
    // uses with those of its YielOp's operands.
    rewriter.replaceOp(op, yieldOp->getOperands());

    // No need for these intermediate blocks, merge them into 1.
    rewriter.mergeBlocks(opEntryBlock, originalBlock, operands);
    rewriter.mergeBlocks(newBlock, originalBlock, {});

    rewriter.eraseOp(&*Block::iterator(yieldOp));

    return success();
  }
};

/// A conversion pattern for detensoring internal (non-entry) blocks within a
/// function.
struct FunctionNonEntryBlockConversion : public ConversionPattern {
  FunctionNonEntryBlockConversion(StringRef functionLikeOpName,
                                  MLIRContext *ctx, TypeConverter &converter,
                                  DenseSet<BlockArgument> blockArgsToDetensor)
      : ConversionPattern(converter, functionLikeOpName, /*benefit=*/1, ctx),
        blockArgsToDetensor(blockArgsToDetensor) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    Region &region = function_like_impl::getFunctionBody(op);
    SmallVector<TypeConverter::SignatureConversion, 2> conversions;

    for (Block &block : llvm::drop_begin(region, 1)) {
      conversions.emplace_back(block.getNumArguments());
      TypeConverter::SignatureConversion &back = conversions.back();

      for (BlockArgument blockArgument : block.getArguments()) {
        int idx = blockArgument.getArgNumber();

        if (blockArgsToDetensor.count(blockArgument))
          back.addInputs(idx, {getTypeConverter()->convertType(
                                  block.getArgumentTypes()[idx])});
        else
          back.addInputs(idx, {block.getArgumentTypes()[idx]});
      }
    }

    if (failed(rewriter.convertNonEntryRegionTypes(&region, *typeConverter,
                                                   conversions))) {
      rewriter.cancelRootUpdate(op);
      return failure();
    }

    rewriter.finalizeRootUpdate(op);
    return success();
  }

private:
  const DenseSet<BlockArgument> blockArgsToDetensor;
};

class DetensorizeTypeConverter : public TypeConverter {
public:
  DetensorizeTypeConverter() {
    addConversion([](Type type) { return type; });

    // A TensorType that can be detensored, is converted to the underlying
    // element type.
    addConversion([](TensorType tensorType) -> Type {
      if (canBeDetensored(tensorType))
        return tensorType.getElementType();

      return tensorType;
    });

    // A tensor value is detensoried by extracting its element(s).
    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<tensor::ExtractOp>(loc, inputs[0], ValueRange{});
    });

    addSourceMaterialization(sourceMaterializationCallback);
    addArgumentMaterialization(sourceMaterializationCallback);
  }
};

/// Canonicalizes the pattern of the form
///
/// %tensor = tensor.from_elements(%element) : (i32) -> tensor<1xi32>
/// %reshaped_tensor = linalg.tensor_collapse_shape %tensor []
///     : tensor<1xi32> into tensor<i32>
/// %extracted_element = tensor.extract %reshaped_tensor[] : tensor<i32>
///
/// to just %element.
struct ExtractFromReshapeFromElements
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp extract,
                                PatternRewriter &rewriter) const final {
    if (!extract.indices().empty())
      return failure();

    auto tensorReshape =
        extract.tensor().getDefiningOp<TensorCollapseShapeOp>();
    if (tensorReshape == nullptr)
      return failure();

    auto tensorFromElements =
        tensorReshape.getOperand()
            .getDefiningOp<mlir::tensor::FromElementsOp>();
    if (tensorFromElements == nullptr)
      return failure();

    rewriter.replaceOp(extract, tensorFromElements.getOperand(0));
    return success();
  }
};

/// @see LinalgDetensorize in Linalg/Passes.td for more details.
struct LinalgDetensorize : public LinalgDetensorizeBase<LinalgDetensorize> {
  LinalgDetensorize() = default;
  LinalgDetensorize(const LinalgDetensorize &pass)
      : LinalgDetensorizeBase<LinalgDetensorize>() {}

  class CostModel {
  public:
    virtual ~CostModel() = default;

    /// A cost model algorithm computes the following outputs:
    ///
    /// - opsToDetensor: the list of linalg ops that should be
    /// detensored.
    ///
    /// - blockArgsToDetensor: since the operands and results of detensored
    /// linalg ops can cross the BB boundary (e.g. a linalg op's input can come
    /// from a BB argument and a linalg op's output can be passed to successor
    /// BBs), we need to maintain the sub-set of arguments that should be
    /// detensored (i.e. converted by typeConverter) for each affected BB.
    ///
    /// Example:
    ///
    /// For the following snippet:
    /// ...
    /// ^bb1(%6: tensor<i32>, %9: tensor<i32>):
    ///   %7 = linalg.init_tensor [] : tensor<i32>
    ///   %8 = linalg.generic #attrs
    ///     ins(%6, %6 : tensor<i32>, tensor<i32>)
    ///     outs(%7 : tensor<i32>) {
    ///     ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
    ///       %9 = addi %arg0, %arg1 : i32
    ///       linalg.yield %9 : i32
    ///   } -> tensor<i32>
    ///   %10 = "some.op"(%9)
    ///   br ^bb2(%8 : tensor<i32>)
    /// ...
    ///
    /// if the cost model decides that the linalg.generic op should be
    /// detensored, then:
    /// - opsToDetensor should be = {linalg.generic{add}}.
    /// - blockArgsToDetensor should be = {bb1 -> {0}, bb2 -> {0}}.
    virtual void compute(FuncOp func, DetensorizeTypeConverter typeConverter,
                         DenseSet<Operation *> &opsToDetensor,
                         DenseSet<BlockArgument> &blockArgsToDetensor) = 0;

    /// From the blockArgsToDetensor set computed by a CostModel
    /// implementation, this method computes the corresponding branch op
    /// detensoring. The result is a map from a branch op to a subset of indices
    /// of its operands. The indices specify which of the branch op's operands
    /// should be detensored.
    ///
    /// For the previous example, this method would compute: {bb2 -> {0}}.
    static DenseMap<Operation *, DenseSet<int>> computeBranchOpDetensoring(
        const DenseSet<BlockArgument> &blockArgsToDetensor) {
      DenseMap<Operation *, DenseSet<int>> detensorableBranchOps;

      for (auto blockArgumentElem : blockArgsToDetensor) {
        Block *block = blockArgumentElem.getOwner();

        for (PredecessorIterator pred = block->pred_begin();
             pred != block->pred_end(); ++pred) {
          BranchOpInterface terminator =
              dyn_cast<BranchOpInterface>((*pred)->getTerminator());
          auto blockOperands =
              terminator.getSuccessorOperands(pred.getSuccessorIndex());

          if (!blockOperands || blockOperands->empty())
            continue;

          detensorableBranchOps[terminator].insert(
              blockOperands->getBeginOperandIndex() +
              blockArgumentElem.getArgNumber());
        }
      }

      return detensorableBranchOps;
    }
  };

  /// Detensorize linalg ops involved in control-flow within a function.
  ///
  /// This model starts from CondBranchOps within a function. For each cond_br,
  /// the model then walks the use-def chain for the branch's condition
  /// backwards in order to understand where the condition's value comes from.
  /// If the condition value is (indirectly) computed by a linalg op that can be
  /// detensored, the model then continues walking the use-def chain in order to
  /// understand where the linalg op's operands come from. This leads to
  /// discovering a "detensoring component". A detensoring component is the set
  /// of operations + block arguments that are involved in control-flow AND can
  /// be detensored.
  ///
  /// For examples where this model succeeds to discover a detensoring
  /// component, see:
  /// - test/Dialect/Linalg/detensorize_while.mlir
  /// - test/Dialect/Linalg/detesorize_while_pure_cf.mlir.
  ///
  /// For an example where this model marks control-flow as "non-detensorable",
  /// see:
  /// - test/Dialect/Linalg/detensorize_while_failure.mlir
  class PureControlFlowDetectionModel : public CostModel {
  public:
    void compute(FuncOp func, DetensorizeTypeConverter typeConverter,
                 DenseSet<Operation *> &opsToDetensor,
                 DenseSet<BlockArgument> &blockArgsToDetensor) override {
      SmallVector<Value> workList;

      func.walk([&](CondBranchOp condBr) {
        for (auto operand : condBr.getOperands()) {
          workList.push_back(operand);
        }
      });

      func.walk([&](BranchOp br) {
        for (auto operand : br.getOperands()) {
          workList.push_back(operand);
        }
      });

      DenseSet<Value> visitedValues;
      DenseSet<Operation *> visitedOps;

      // For a (to-be-detesored) value, check if it "escapes" the block by being
      // passed to terminator. If it does, then workList is updated with the
      // corresponding argument to the successor block.
      auto updateWorkListWithSuccessorArguments =
          [&](Value value, BranchOpInterface terminator) {
            if (!terminator)
              return;

            for (auto operandIdx :
                 llvm::seq<unsigned>(0, terminator->getOperands().size())) {
              Value operand = terminator->getOperand(operandIdx);

              if (operand == value) {
                auto succBlockArg =
                    terminator.getSuccessorBlockArgument(operandIdx);

                if (succBlockArg && !blockArgsToDetensor.count(*succBlockArg))
                  workList.push_back(*succBlockArg);
              }
            }
          };

      while (!workList.empty()) {
        Value currentItem = workList.pop_back_val();

        if (!visitedValues.insert(currentItem).second)
          continue;

        // 1   - Look forward:
        // 1.1 - If currentItem escapes to one or more successors, add
        // the corresponding successor arguments to workList.
        updateWorkListWithSuccessorArguments(
            currentItem, dyn_cast<BranchOpInterface>(
                             currentItem.getParentBlock()->getTerminator()));

        // 1.2 - For each user of currentItem, add the defined values to
        // workList. This way, the user ops can be inspected later if they are
        // detensorable and if so, their operands will be added to workList to
        // potentially discover other parts of the detensorable component.
        for (auto *user : currentItem.getUsers())
          for (Value result : user->getResults())
            workList.push_back(result);

        // 2   - Look backward:
        // 2.1 - The current item is defined by a block argument. If the owner
        // block is a non-entry one, then:
        //       * Add the argument to blockArgsToDetensor.
        //       * Walk the use-def chain backwards to add each predecessor's
        //       terminator-operands corresponding to currentItem to workList.
        if (currentItem.dyn_cast<BlockArgument>()) {
          BlockArgument currentItemBlockArgument =
              currentItem.cast<BlockArgument>();
          Block *ownerBlock = currentItemBlockArgument.getOwner();

          // Function arguments are not detensored/converted.
          if (&*ownerBlock->getParent()->begin() == ownerBlock)
            continue;

          // This inner-block argument is involved in control-flow, it should be
          // detensored.
          blockArgsToDetensor.insert(currentItemBlockArgument);

          for (PredecessorIterator pred = ownerBlock->pred_begin();
               pred != ownerBlock->pred_end(); ++pred) {
            BranchOpInterface terminator =
                dyn_cast<BranchOpInterface>((*pred)->getTerminator());

            // TODO: For now, we give up if any of the control-flow components
            // in a function is not detensorable. Fix that.
            if (!terminator) {
              opsToDetensor.clear();
              blockArgsToDetensor.clear();
              return;
            }

            auto ownerBlockOperands =
                terminator.getSuccessorOperands(pred.getSuccessorIndex());

            if (!ownerBlockOperands || ownerBlockOperands->empty())
              continue;

            // For each predecessor, add the value it passes to that argument to
            // workList to find out how it's computed.
            workList.push_back(
                ownerBlockOperands
                    .getValue()[currentItemBlockArgument.getArgNumber()]);
          }

          continue;
        }

        Operation *currentItemDefiningOp = currentItem.getDefiningOp();

        if (!visitedOps.insert(currentItemDefiningOp).second)
          continue;

        // 2.2 - The current item is computed by a GenericOp. If the op should
        // be detensored, then:
        //       * Add it to opsToDetensor.
        //       * Add its operands to workList to discover other parts of the
        //       potentially detensorable component.
        if (auto genericOp = dyn_cast<GenericOp>(currentItemDefiningOp)) {
          // The op was encountered already, no need to inspect it again.
          if (opsToDetensor.count(genericOp))
            continue;

          // TODO: For now, we give up if any of the control-flow components
          // in a function is not detensorable. Fix that.
          if (!shouldBeDetensored(genericOp, typeConverter)) {
            opsToDetensor.clear();
            blockArgsToDetensor.clear();
            return;
          }

          opsToDetensor.insert(genericOp);

          for (Value genericOpOperand : genericOp.inputs())
            workList.push_back(genericOpOperand);

          continue;
        }

        // 2.3 - The current item is the result of a FromElementsOp, it will be
        // trivially detensored later as part of canonicalization patterns
        // applied at the end of detensoring.
        //
        // Note: No need to check whether the result type of this op is
        // detensorable since if it wasn't we wouldn't reach that point in the
        // work list.
        if (dyn_cast<tensor::FromElementsOp>(currentItemDefiningOp))
          continue;

        // 2.4 - The current item is the result of a scalar op, add all its
        // operands to the work list.
        if (llvm::all_of(
                currentItemDefiningOp->getResultTypes(),
                [&](Type resultType) { return resultType.isIntOrFloat(); }))
          for (Value scalarOpOperand : currentItemDefiningOp->getOperands())
            workList.push_back(scalarOpOperand);
      }
    }
  };

  /// Detensorize everything that can detensored.
  class AggressiveDetensoringModel : public CostModel {
  public:
    void compute(FuncOp func, DetensorizeTypeConverter typeConverter,
                 DenseSet<Operation *> &opsToDetensor,
                 DenseSet<BlockArgument> &blockArgsToDetensor) override {
      func.walk([&](GenericOp genericOp) {
        if (shouldBeDetensored(genericOp, typeConverter))
          opsToDetensor.insert(genericOp);
      });

      for (Block &block : llvm::drop_begin(func.getBody(), 1))
        for (BlockArgument blockArgument : block.getArguments())
          blockArgsToDetensor.insert(blockArgument);
    }
  };

  void runOnFunction() override {
    MLIRContext *context = &getContext();
    DetensorizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    DenseSet<Operation *> opsToDetensor;
    DenseMap<Operation *, DenseSet<int>> detensorableBranchOps;
    DenseSet<BlockArgument> blockArgsToDetensor;

    if (aggressiveMode.getValue()) {
      AggressiveDetensoringModel costModel;
      costModel.compute(getFunction(), typeConverter, opsToDetensor,
                        blockArgsToDetensor);

    } else {
      PureControlFlowDetectionModel costModel;
      costModel.compute(getFunction(), typeConverter, opsToDetensor,
                        blockArgsToDetensor);
    }

    detensorableBranchOps =
        CostModel::computeBranchOpDetensoring(blockArgsToDetensor);

    target.addDynamicallyLegalOp<GenericOp>(
        [&](GenericOp op) { return !opsToDetensor.count(op); });

    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      // A function is legal if all of its non-entry blocks are legal. We
      // don't legalize the entry block (i.e. the function's signature)
      // since detensoring can't happen along external calling convention
      // boundaries, which we conservatively approximate as all function
      // signatures.
      return llvm::all_of(llvm::drop_begin(op.getBody(), 1), [&](Block &block) {
        if (llvm::any_of(blockArgsToDetensor, [&](BlockArgument blockArgument) {
              return blockArgument.getOwner() == &block &&
                     !typeConverter.isLegal(blockArgument.getType());
            })) {
          return false;
        }
        return true;
      });
    });

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (isNotBranchOpInterfaceOrReturnLikeOp(op) ||
          isLegalForReturnOpTypeConversionPattern(op, typeConverter,
                                                  /*returnOpAlwaysLegal*/ true))
        return true;

      if (auto branchOp = dyn_cast<BranchOpInterface>(op)) {
        if (!detensorableBranchOps.count(branchOp))
          return true;

        for (auto operandIdx : detensorableBranchOps[branchOp])
          if (!typeConverter.isLegal(
                  branchOp->getOperand(operandIdx).getType()))
            return false;

        return true;
      }

      return false;
    });

    patterns.insert<DetensorizeGenericOp>(typeConverter, context);
    patterns.insert<FunctionNonEntryBlockConversion>(FuncOp::getOperationName(),
                                                     context, typeConverter,
                                                     blockArgsToDetensor);
    // Since non-entry block arguments get detensorized, we also need to
    // update the control flow inside the function to reflect the correct
    // types.
    auto shouldConvertBranchOperand = [&](BranchOpInterface branchOp,
                                          int operandIdx) -> bool {
      return detensorableBranchOps.count(branchOp) &&
             detensorableBranchOps[branchOp].count(operandIdx);
    };

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter,
                                                   shouldConvertBranchOperand);

    if (failed(applyFullConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();

    RewritePatternSet canonPatterns(context);
    canonPatterns.add<ExtractFromReshapeFromElements>(context);
    if (failed(applyPatternsAndFoldGreedily(getFunction(),
                                            std::move(canonPatterns))))
      signalPassFailure();
  }

  Option<bool> aggressiveMode{
      *this, "aggressive-mode",
      llvm::cl::desc("Detensorize all ops that qualify for detensoring along "
                     "with branch operands and basic-block arguments.")};
};
} // namespace

std::unique_ptr<Pass> mlir::createLinalgDetensorizePass() {
  return std::make_unique<LinalgDetensorize>();
}
