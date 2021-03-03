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
  return builder.create<linalg::TensorReshapeOp>(
      loc, type, createNewTensorOp, ArrayRef<ReassociationExprs>{});
}

namespace {
/// Defines the criteria a TensorType must follow in order to be considered
/// "detensorable".
///
/// NOTE: For now, only 0-D are supported.
///
/// Returns true if tensorType can be detensored.
bool canBeDetensored(TensorType tensorType) {
  return tensorType.hasRank() && tensorType.getRank() == 0;
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
                                  MLIRContext *ctx, TypeConverter &converter)
      : ConversionPattern(functionLikeOpName, /*benefit=*/1, converter, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);

    if (failed(rewriter.convertNonEntryRegionTypes(
            &mlir::impl::getFunctionBody(op), *typeConverter))) {
      rewriter.cancelRootUpdate(op);
      return failure();
    }

    rewriter.finalizeRootUpdate(op);
    return success();
  }
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
/// %reshaped_tensor = linalg.tensor_reshape %tensor [] : tensor<1xi32> into
///   tensor<i32>
/// %extracted_element = tensor.extract %reshaped_tensor[] : tensor<i32>
///
/// to just %element.
struct ExtractFromReshapeFromElements
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp extract,
                                PatternRewriter &rewriter) const final {
    if (extract.indices().size() != 0)
      return failure();

    auto tensorReshape = extract.tensor().getDefiningOp<TensorReshapeOp>();
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
  void runOnFunction() override {
    auto *context = &getContext();
    DetensorizeTypeConverter typeConverter;
    OwningRewritePatternList patterns;
    ConversionTarget target(*context);

    target.addDynamicallyLegalOp<GenericOp>([&](GenericOp op) {
      // If any of the operands or results cannot be detensored (i.e. they are
      // all legal according the DetensorizeTypeConverter), the op is considered
      // legal and won't be detensored.
      return llvm::any_of(op.getShapedOperandTypes(),
                          [&](ShapedType shapedType) {
                            return typeConverter.isLegal(shapedType);
                          });
    });

    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      // A function is legal if all of its non-entry blocks are legal. We don't
      // legalize the entry block (i.e. the function's signature) since
      // detensoring can't happen along external calling convention boundaries,
      // which we conservatively approximate as all function signatures.
      return llvm::all_of(llvm::drop_begin(op.getBody(), 1), [&](Block &block) {
        return typeConverter.isLegal(block.getArgumentTypes());
      });
    });

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(
                 op, typeConverter, /*returnOpAlwaysLegal*/ true);
    });

    patterns.insert<DetensorizeGenericOp>(typeConverter, context);
    patterns.insert<FunctionNonEntryBlockConversion>(FuncOp::getOperationName(),
                                                     context, typeConverter);
    // Since non-entry block arguments get detensorized, we also need to update
    // the control flow inside the function to reflect the correct types.
    populateBranchOpInterfaceTypeConversionPattern(patterns, context,
                                                   typeConverter);

    if (failed(applyFullConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();

    OwningRewritePatternList canonPatterns;
    canonPatterns.insert<ExtractFromReshapeFromElements>(context);
    if (failed(applyPatternsAndFoldGreedily(getFunction(),
                                            std::move(canonPatterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createLinalgDetensorizePass() {
  return std::make_unique<LinalgDetensorize>();
}
