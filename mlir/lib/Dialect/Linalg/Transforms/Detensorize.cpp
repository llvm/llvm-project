//===- Detensorize.cpp - Linalg transformations as patterns ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iterator>
#include <memory>
#include <utility>

namespace mlir {
#define GEN_PASS_DEF_LINALGDETENSORIZEPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

static Value sourceMaterializationCallback(OpBuilder &builder, Type type,
                                           ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  auto inputType = inputs[0].getType();
  if (isa<TensorType>(inputType))
    return nullptr;

  // A detensored value is converted back by creating a new tensor from its
  // element(s).
  return builder.create<tensor::FromElementsOp>(
      loc, RankedTensorType::get({}, inputType), inputs[0]);
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
         llvm::all_of(genericOp->getOpOperands(), [&](OpOperand &opOperand) {
           return !typeConverter.isLegal(opOperand.get().getType());
         });
}

/// A conversion pattern for detensoring `linalg.generic` ops.
class DetensorizeGenericOp : public OpConversionPattern<GenericOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block *originalBlock = op->getBlock();

    // Gather some information about the op before inlining its region.
    Block *opEntryBlock = &*op.getRegion().begin();
    YieldOp yieldOp = dyn_cast<YieldOp>(op.getRegion().back().getTerminator());

    // Split the op's region before the op. This way, we have a clear insertion
    // point in which the op can be inlined.
    Block *newBlock = rewriter.splitBlock(originalBlock, Block::iterator(op));
    rewriter.inlineRegionBefore(op.getRegion(), newBlock);
    // Now that op's region is inlined, the operands of its YieldOp are mapped
    // to the materialized target values. Therefore, we can replace the op's
    // uses with those of its YielOp's operands.
    rewriter.replaceOp(op, yieldOp->getOperands());

    // No need for these intermediate blocks, merge them into 1.
    rewriter.mergeBlocks(opEntryBlock, originalBlock, adaptor.getOperands());
    rewriter.mergeBlocks(newBlock, originalBlock, {});

    rewriter.eraseOp(&*Block::iterator(yieldOp));

    return success();
  }
};

/// A conversion pattern for detensoring internal (non-entry) blocks within a
/// function.
struct FunctionNonEntryBlockConversion
    : public OpInterfaceConversionPattern<FunctionOpInterface> {
  FunctionNonEntryBlockConversion(MLIRContext *ctx, TypeConverter &converter,
                                  DenseSet<BlockArgument> blockArgsToDetensor)
      : OpInterfaceConversionPattern(converter, ctx),
        blockArgsToDetensor(std::move(blockArgsToDetensor)) {}

  LogicalResult
  matchAndRewrite(FunctionOpInterface op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.startOpModification(op);
    Region &region = op.getFunctionBody();

    for (Block &block :
         llvm::make_early_inc_range(llvm::drop_begin(region, 1))) {
      TypeConverter::SignatureConversion conversion(
          /*numOrigInputs=*/block.getNumArguments());

      for (BlockArgument blockArgument : block.getArguments()) {
        int idx = blockArgument.getArgNumber();

        if (blockArgsToDetensor.count(blockArgument))
          conversion.addInputs(idx, {getTypeConverter()->convertType(
                                        block.getArgumentTypes()[idx])});
        else
          conversion.addInputs(idx, {block.getArgumentTypes()[idx]});
      }

      rewriter.applySignatureConversion(&block, conversion, getTypeConverter());
    }

    rewriter.finalizeOpModification(op);
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

/// @see LinalgDetensorize in Linalg/Passes.td for more details.
struct LinalgDetensorize
    : public impl::LinalgDetensorizePassBase<LinalgDetensorize> {
  using impl::LinalgDetensorizePassBase<
      LinalgDetensorize>::LinalgDetensorizePassBase;
  LinalgDetensorize() = default;

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
    ///   %7 = tensor.empty() : tensor<i32>
    ///   %8 = linalg.generic #attrs
    ///     ins(%6, %6 : tensor<i32>, tensor<i32>)
    ///     outs(%7 : tensor<i32>) {
    ///     ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
    ///       %9 = arith.addi %arg0, %arg1 : i32
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
    virtual void compute(FunctionOpInterface func,
                         DetensorizeTypeConverter typeConverter,
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

          if (blockOperands.empty() ||
              blockOperands.isOperandProduced(blockArgumentElem.getArgNumber()))
            continue;

          detensorableBranchOps[terminator].insert(
              blockOperands.getOperandIndex(blockArgumentElem.getArgNumber()));
        }
      }

      return detensorableBranchOps;
    }
  };

  /// Detensorize linalg ops involved in control-flow within a function.
  ///
  /// This model starts from BranchOps and CondBranchOps within a function. For
  /// each such branch, the model then walks the use-def chain for the branch's
  /// condition backwards in order to understand where the condition's value
  /// comes from. If the condition value is (indirectly) computed by a linalg op
  /// that can be detensored, the model then continues walking the use-def chain
  /// in order to understand where the linalg op's operands come from. This
  /// leads to discovering a "detensoring component". A detensoring component is
  /// the set of operations + block arguments that are involved in control-flow
  /// AND can be detensored.
  class ControlFlowDetectionModel : public CostModel {
  public:
    void compute(FunctionOpInterface func,
                 DetensorizeTypeConverter typeConverter,
                 DenseSet<Operation *> &opsToDetensor,
                 DenseSet<BlockArgument> &blockArgsToDetensor) override {
      SmallVector<Value> workList;

      func->walk([&](cf::CondBranchOp condBr) {
        llvm::append_range(workList, condBr.getOperands());
      });

      func->walk([&](cf::BranchOp br) {
        llvm::append_range(workList, br.getOperands());
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
          llvm::append_range(workList, user->getResults());

        // 2   - Look backward:
        // 2.1 - The current item is defined by a block argument. If the owner
        // block is a non-entry one, then:
        //       * Add the argument to blockArgsToDetensor.
        //       * Walk the use-def chain backwards to add each predecessor's
        //       terminator-operands corresponding to currentItem to workList.
        if (dyn_cast<BlockArgument>(currentItem)) {
          BlockArgument currentItemBlockArgument =
              cast<BlockArgument>(currentItem);
          Block *ownerBlock = currentItemBlockArgument.getOwner();

          // Function arguments are not detensored/converted.
          if (&*ownerBlock->getParent()->begin() == ownerBlock)
            continue;

          // This inner-block argument is involved in control-flow, it should be
          // detensored.
          blockArgsToDetensor.insert(currentItemBlockArgument);

          for (PredecessorIterator pred = ownerBlock->pred_begin();
               pred != ownerBlock->pred_end(); ++pred) {
            BranchOpInterface predTerminator =
                dyn_cast<BranchOpInterface>((*pred)->getTerminator());

            // TODO: For now, we give up if any of the control-flow components
            // in a function is not detensorable. Fix that.
            if (!predTerminator) {
              opsToDetensor.clear();
              blockArgsToDetensor.clear();
              return;
            }

            auto ownerBlockOperands =
                predTerminator.getSuccessorOperands(pred.getSuccessorIndex());

            if (ownerBlockOperands.empty() ||
                ownerBlockOperands.isOperandProduced(
                    currentItemBlockArgument.getArgNumber()))
              continue;

            // For each predecessor, add the value it passes to that argument to
            // workList to find out how it's computed.
            workList.push_back(
                ownerBlockOperands[currentItemBlockArgument.getArgNumber()]);
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

          // The op should not be detensored, give up on it but continue with
          // discovering the rest of the control-flow component.
          if (!shouldBeDetensored(genericOp, typeConverter)) {
            continue;
          }

          opsToDetensor.insert(genericOp);
          llvm::append_range(workList, genericOp.getInputs());
          continue;
        }

        // 2.3 - The current item is the result of a FromElementsOp, it will be
        // trivially detensored later as part of canonicalization patterns
        // applied at the end of detensoring.
        //
        // Note: No need to check whether the result type of this op is
        // detensorable since if it wasn't we wouldn't reach that point in the
        // work list.
        if (isa<tensor::FromElementsOp>(currentItemDefiningOp))
          continue;

        // 2.4 - The current item is the result of a scalar op, add all its
        // operands to the work list.
        if (llvm::all_of(
                currentItemDefiningOp->getResultTypes(),
                [&](Type resultType) { return resultType.isIntOrFloat(); }))
          llvm::append_range(workList, currentItemDefiningOp->getOperands());
      }

      // Since the cost model gives up on some ops (see the details of step 2.2
      // above), block arguments that correspond to the values produced by those
      // ops should not be detensored as well.

      DenseSet<BlockArgument> blockArgsToRemove;

      for (auto &blockArg : blockArgsToDetensor) {
        Block *block = blockArg.getParentBlock();

        // For the potentially detensorable block argument, find the
        // correpsonding operands in predecessor blocks.
        for (PredecessorIterator pred = block->pred_begin();
             pred != block->pred_end(); ++pred) {
          BranchOpInterface terminator =
              dyn_cast<BranchOpInterface>((*pred)->getTerminator());
          auto blockOperands =
              terminator.getSuccessorOperands(pred.getSuccessorIndex());

          if (blockOperands.empty() ||
              blockOperands.isOperandProduced(blockArg.getArgNumber()))
            continue;

          Operation *definingOp =
              blockOperands[blockArg.getArgNumber()].getDefiningOp();

          // If the operand is defined by a GenericOp that will not be
          // detensored, then do not detensor the corresponding block argument.
          if (isa_and_nonnull<GenericOp>(definingOp) &&
              opsToDetensor.count(definingOp) == 0) {
            blockArgsToRemove.insert(blockArg);
            break;
          }
        }
      }

      for (auto &blockArg : blockArgsToRemove) {
        blockArgsToDetensor.erase(blockArg);
      }
    }
  };

  /// Detensorize everything that can detensored.
  class AggressiveDetensoringModel : public CostModel {
  public:
    void compute(FunctionOpInterface func,
                 DetensorizeTypeConverter typeConverter,
                 DenseSet<Operation *> &opsToDetensor,
                 DenseSet<BlockArgument> &blockArgsToDetensor) override {
      func->walk([&](GenericOp genericOp) {
        if (shouldBeDetensored(genericOp, typeConverter))
          opsToDetensor.insert(genericOp);
      });

      for (Block &block : llvm::drop_begin(func.getFunctionBody(), 1))
        for (BlockArgument blockArgument : block.getArguments())
          blockArgsToDetensor.insert(blockArgument);
    }
  };

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    DetensorizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    DenseSet<Operation *> opsToDetensor;
    DenseMap<Operation *, DenseSet<int>> detensorableBranchOps;
    DenseSet<BlockArgument> blockArgsToDetensor;
    FunctionOpInterface funcOp = getOperation();

    if (funcOp.getFunctionBody().empty())
      return;

    // Make sure the entry block of the function doesn't contain any Linalg ops.
    // Otherwise, it may lead to the signature of the block being changed by the
    // dialect conversion below, which would make the function op invalid
    // because its type shouldn't change.
    IRRewriter rewriter(funcOp->getContext());
    Block *entryBlock = &funcOp.getFunctionBody().front();
    Block *postEntryBlock =
        rewriter.splitBlock(entryBlock, entryBlock->begin());
    rewriter.setInsertionPointToStart(entryBlock);
    auto branch =
        rewriter.create<cf::BranchOp>(rewriter.getUnknownLoc(), postEntryBlock);

    if (aggressiveMode.getValue()) {
      AggressiveDetensoringModel costModel;
      costModel.compute(funcOp, typeConverter, opsToDetensor,
                        blockArgsToDetensor);
    } else {
      ControlFlowDetectionModel costModel;
      costModel.compute(funcOp, typeConverter, opsToDetensor,
                        blockArgsToDetensor);
    }

    detensorableBranchOps =
        CostModel::computeBranchOpDetensoring(blockArgsToDetensor);

    target.addDynamicallyLegalOp<GenericOp>(
        [&](GenericOp op) { return !opsToDetensor.count(op); });

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      // A function is legal if all of its non-entry blocks are legal. We
      // don't legalize the entry block (i.e. the function's signature)
      // since detensoring can't happen along external calling convention
      // boundaries, which we conservatively approximate as all function
      // signatures.
      if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
        Region &body = funcOp.getFunctionBody();
        return llvm::all_of(llvm::drop_begin(body, 1), [&](Block &block) {
          return !llvm::any_of(
              blockArgsToDetensor, [&](BlockArgument blockArgument) {
                return blockArgument.getOwner() == &block &&
                       !typeConverter.isLegal(blockArgument.getType());
              });
        });
      }

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

    patterns.add<DetensorizeGenericOp>(typeConverter, context);
    patterns.add<FunctionNonEntryBlockConversion>(context, typeConverter,
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

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();

    RewritePatternSet canonPatterns(context);
    tensor::FromElementsOp::getCanonicalizationPatterns(canonPatterns, context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(canonPatterns))))
      signalPassFailure();

    // Get rid of the dummy entry block we created in the beginning to work
    // around dialect conversion signature rewriting.
    rewriter.eraseOp(branch);
    rewriter.mergeBlocks(postEntryBlock, entryBlock);
  }
};
} // namespace
