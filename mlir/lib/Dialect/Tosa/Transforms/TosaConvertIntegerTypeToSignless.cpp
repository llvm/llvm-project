//===- TosaConvertIntegerTypeToSignless.cpp
//-------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------------===//

// -----------
// Motivation:
// -----------

// The TOSA specification uses a signless type system, which means that
// information about signedness must be encapsulated by the operations
// themselves. For example, tosa.rescale provides the attributes
// `input_unsigned` and `output_unsigned` to indicate whether the input/output
// should be interpreted as unsigned or signed.

// The TOSA dialect, on the other hand, allows the use of signed or unsigned
// types in addition to signless. As such, when converting from TOSA dialect to
// other formats, we need to ensure that we conform to the TOSA specification.

// ---------
// Overview:
// ---------

// This pass converts signed or unsigned integer types to signless. It currently
// does this greedily for all operators and can also change the signature of the
// function. Should the signature of the entrypoint function change, it will be
// the responsibility of the user to carry signedness information of the inputs
// and outputs independently.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace tosa {

#define GEN_PASS_DEF_TOSACONVERTINTEGERTYPETOSIGNLESS
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"

namespace {
class ToSignlessTensorTypeConverter : public TypeConverter {
  static Type convertType(Type type) {
    const auto tensorType = dyn_cast<TensorType>(type);
    if (!tensorType)
      return type;

    const auto intType = dyn_cast<IntegerType>(tensorType.getElementType());
    if (!intType ||
        intType.getSignedness() == IntegerType::SignednessSemantics::Signless)
      return type;

    const auto signlessType = IntegerType::get(
        intType.getContext(), intType.getWidth(), IntegerType::Signless);
    return tensorType.cloneWith(std::nullopt, signlessType);
  }

public:
  explicit ToSignlessTensorTypeConverter() { addConversion(convertType); }
};

class ConvertGenericOpWithIntegerTensorType : public ConversionPattern {
public:
  ConvertGenericOpWithIntegerTensorType(TypeConverter &typeConverter,
                                        MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, 0, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Typically TOSA operators have a single result, but some have an
    // arbitrary number. 4 seems like a good balance as an optimization
    // hint for storing result types.
    constexpr unsigned int numResults = 4;

    // Convert integer types to signless
    SmallVector<Type, numResults> resultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
      return failure();

    // Create new op with replaced operands and results
    auto *newOp = Operation::create(
        op->getLoc(), op->getName(), resultTypes, operands, op->getAttrs(),
        op->getPropertiesStorage(), op->getSuccessors(), op->getNumRegions());

    // Handle regions in e.g. tosa.cond_if and tosa.while_loop
    for (auto regions : llvm::zip(op->getRegions(), newOp->getRegions())) {
      Region &before = std::get<0>(regions);
      Region &parent = std::get<1>(regions);
      rewriter.inlineRegionBefore(before, parent, parent.end());
      if (failed(rewriter.convertRegionTypes(&parent, *typeConverter)))
        return failure();
    }

    // Replace with rewritten op
    rewriter.insert(newOp);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

class TosaConvertIntegerTypeToSignless
    : public impl::TosaConvertIntegerTypeToSignlessBase<
          TosaConvertIntegerTypeToSignless> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    ToSignlessTensorTypeConverter typeConverter;

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return typeConverter.isLegal(op->getOperandTypes()) &&
             typeConverter.isLegal(op->getResultTypes());
    });

    RewritePatternSet patterns(context);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    patterns.add<ConvertGenericOpWithIntegerTensorType>(typeConverter, context);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace tosa
} // namespace mlir
