//===- RuntimeOpVerification.cpp - Op Verification ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/Transforms/RuntimeOpVerification.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/RuntimeVerifiableOpInterface.h"

using namespace mlir;

namespace mlir {
namespace tensor {
namespace {
/// Generate a runtime check for lb <= value < ub.
Value generateInBoundsCheck(OpBuilder &builder, Location loc, Value value,
                            Value lb, Value ub) {
  Value inBounds1 = builder.createOrFold<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, value, lb);
  Value inBounds2 = builder.createOrFold<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, value, ub);
  Value inBounds =
      builder.createOrFold<arith::AndIOp>(loc, inBounds1, inBounds2);
  return inBounds;
}

struct CastOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<CastOpInterface,
                                                         CastOp> {
  void
  generateRuntimeVerification(Operation *op, OpBuilder &builder, Location loc,
                              function_ref<std::string(Operation *, StringRef)>
                                  generateErrorMessage) const {
    auto castOp = cast<CastOp>(op);
    auto srcType = cast<TensorType>(castOp.getSource().getType());

    // Nothing to check if the result is an unranked tensor.
    auto resultType = dyn_cast<RankedTensorType>(castOp.getType());
    if (!resultType)
      return;

    if (isa<UnrankedTensorType>(srcType)) {
      // Check rank.
      Value srcRank = RankOp::create(builder, loc, castOp.getSource());
      Value resultRank =
          arith::ConstantIndexOp::create(builder, loc, resultType.getRank());
      Value isSameRank = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, srcRank, resultRank);
      cf::AssertOp::create(builder, loc, isSameRank,
                           generateErrorMessage(op, "rank mismatch"));
    }

    // Check dimension sizes.
    for (const auto &it : llvm::enumerate(resultType.getShape())) {
      // Static dim size -> static/dynamic dim size does not need verification.
      if (auto rankedSrcType = dyn_cast<RankedTensorType>(srcType))
        if (!rankedSrcType.isDynamicDim(it.index()))
          continue;

      // Static/dynamic dim size -> dynamic dim size does not need verification.
      if (resultType.isDynamicDim(it.index()))
        continue;

      Value srcDimSz =
          DimOp::create(builder, loc, castOp.getSource(), it.index());
      Value resultDimSz =
          arith::ConstantIndexOp::create(builder, loc, it.value());
      Value isSameSz = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, srcDimSz, resultDimSz);
      cf::AssertOp::create(
          builder, loc, isSameSz,
          generateErrorMessage(op, "size mismatch of dim " +
                                       std::to_string(it.index())));
    }
  }
};

struct DimOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<DimOpInterface,
                                                         DimOp> {
  void
  generateRuntimeVerification(Operation *op, OpBuilder &builder, Location loc,
                              function_ref<std::string(Operation *, StringRef)>
                                  generateErrorMessage) const {
    auto dimOp = cast<DimOp>(op);
    Value rank = RankOp::create(builder, loc, dimOp.getSource());
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    cf::AssertOp::create(
        builder, loc,
        generateInBoundsCheck(builder, loc, dimOp.getIndex(), zero, rank),
        generateErrorMessage(op, "index is out of bounds"));
  }
};

/// Verifies that the indices on extract/insert ops are in-bounds of the
/// tensor's index space: 0 <= index#i < dim#i
template <typename OpTy>
struct ExtractInsertOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<
          ExtractInsertOpInterface<OpTy>, OpTy> {
  void
  generateRuntimeVerification(Operation *op, OpBuilder &builder, Location loc,
                              function_ref<std::string(Operation *, StringRef)>
                                  generateErrorMessage) const {
    auto extractInsertOp = cast<OpTy>(op);

    Value tensor;
    if constexpr (std::is_same_v<OpTy, ExtractOp>) {
      tensor = extractInsertOp.getTensor();
    } else if constexpr (std::is_same_v<OpTy, InsertOp>) {
      tensor = extractInsertOp.getDest();
    } else {
      llvm_unreachable("invalid op");
    }
    auto tensorType = cast<RankedTensorType>(tensor.getType());
    auto rank = tensorType.getRank();
    if (rank == 0) {
      // Nothing to check for 0-d tensors.
      return;
    }

    auto indices = extractInsertOp.getIndices();
    auto zero = arith::ConstantIndexOp::create(builder, loc, 0);
    Value assertCond;
    for (auto i : llvm::seq<int64_t>(0, rank)) {
      Value dimOp = builder.createOrFold<tensor::DimOp>(loc, tensor, i);
      Value inBounds =
          generateInBoundsCheck(builder, loc, indices[i], zero, dimOp);
      assertCond =
          i > 0 ? builder.createOrFold<arith::AndIOp>(loc, assertCond, inBounds)
                : inBounds;
    }
    cf::AssertOp::create(builder, loc, assertCond,
                         generateErrorMessage(op, "out-of-bounds access"));
  }
};

struct ExtractSliceOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<
          ExtractSliceOpInterface, ExtractSliceOp> {
  void
  generateRuntimeVerification(Operation *op, OpBuilder &builder, Location loc,
                              function_ref<std::string(Operation *, StringRef)>
                                  generateErrorMessage) const {
    auto extractSliceOp = cast<ExtractSliceOp>(op);
    RankedTensorType sourceType = extractSliceOp.getSource().getType();

    // For each dimension, assert that:
    // For empty slices (size == 0)   : 0 <= offset <= dim_size
    // For non-empty slices (size > 0): 0 <= offset < dim_size
    //                                  0 <= offset + (size - 1) * stride <
    //                                  dim_size
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    Value one = arith::ConstantIndexOp::create(builder, loc, 1);

    for (int64_t i : llvm::seq<int64_t>(0, sourceType.getRank())) {

      builder.setInsertionPoint(extractSliceOp);

      Value offset = getValueOrCreateConstantIndexOp(
          builder, loc, extractSliceOp.getMixedOffsets()[i]);
      Value size = getValueOrCreateConstantIndexOp(
          builder, loc, extractSliceOp.getMixedSizes()[i]);
      Value stride = getValueOrCreateConstantIndexOp(
          builder, loc, extractSliceOp.getMixedStrides()[i]);
      Value dimSize = builder.createOrFold<tensor::DimOp>(
          loc, extractSliceOp.getSource(), i);

      // Verify that offset is in-bounds (conditional on slice size).
      Value sizeIsZero = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, size, zero);
      auto offsetCheckIf = scf::IfOp::create(
          builder, loc, sizeIsZero,
          [&](OpBuilder &b, Location loc) {
            // For empty slices, offset can be at the boundary: 0 <= offset <=
            // dimSize.
            Value offsetGEZero = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::sge, offset, zero);
            Value offsetLEDimSize = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::sle, offset, dimSize);
            Value emptyOffsetValid =
                arith::AndIOp::create(b, loc, offsetGEZero, offsetLEDimSize);
            scf::YieldOp::create(b, loc, emptyOffsetValid);
          },
          [&](OpBuilder &b, Location loc) {
            // For non-empty slices, offset must be a valid index: 0 <= offset <
            // dimSize.
            Value offsetInBounds =
                generateInBoundsCheck(b, loc, offset, zero, dimSize);
            scf::YieldOp::create(b, loc, offsetInBounds);
          });

      Value offsetCondition = offsetCheckIf.getResult(0);
      cf::AssertOp::create(builder, loc, offsetCondition,
                           generateErrorMessage(op, "offset " +
                                                        std::to_string(i) +
                                                        " is out-of-bounds"));

      // Verify that the slice endpoint is in-bounds (only for non-empty
      // slices).
      Value sizeIsNonZero = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::sgt, size, zero);
      auto ifOp = scf::IfOp::create(
          builder, loc, sizeIsNonZero,
          [&](OpBuilder &b, Location loc) {
            // Verify that slice does not run out-of-bounds.
            Value sizeMinusOne = arith::SubIOp::create(b, loc, size, one);
            Value sizeMinusOneTimesStride =
                arith::MulIOp::create(b, loc, sizeMinusOne, stride);
            Value lastPos =
                arith::AddIOp::create(b, loc, offset, sizeMinusOneTimesStride);
            Value lastPosInBounds =
                generateInBoundsCheck(b, loc, lastPos, zero, dimSize);
            scf::YieldOp::create(b, loc, lastPosInBounds);
          },
          [&](OpBuilder &b, Location loc) {
            Value trueVal =
                arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
            scf::YieldOp::create(b, loc, trueVal);
          });

      Value finalCondition = ifOp.getResult(0);
      cf::AssertOp::create(
          builder, loc, finalCondition,
          generateErrorMessage(
              op, "extract_slice runs out-of-bounds along dimension " +
                      std::to_string(i)));
    }
  }
};

struct ExpandShapeOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<ExpandShapeOpInterface,
                                                         ExpandShapeOp> {
  void
  generateRuntimeVerification(Operation *op, OpBuilder &builder, Location loc,
                              function_ref<std::string(Operation *, StringRef)>
                                  generateErrorMessage) const {
    auto expandShapeOp = cast<ExpandShapeOp>(op);

    SmallVector<OpFoldResult> outputShape = expandShapeOp.getMixedOutputShape();

    // Verify that the product of output dim sizes in each reassociation group
    // equals the corresponding input dim size.
    for (const auto &it :
         llvm::enumerate(expandShapeOp.getReassociationIndices())) {
      Value srcDimSz =
          DimOp::create(builder, loc, expandShapeOp.getSrc(), it.index());
      Value groupProduct = getValueOrCreateConstantIndexOp(
          builder, loc, outputShape[it.value().front()]);
      for (int64_t resultDim : llvm::drop_begin(it.value())) {
        Value dimSz = getValueOrCreateConstantIndexOp(builder, loc,
                                                      outputShape[resultDim]);
        groupProduct = arith::MulIOp::create(builder, loc, groupProduct, dimSz);
      }
      Value isEqual = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, groupProduct, srcDimSz);
      cf::AssertOp::create(
          builder, loc, isEqual,
          generateErrorMessage(op, "product of output dims in reassoc group "
                                   "does not equal input dim"));
    }
  }
};

struct ConcatOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<ConcatOpInterface,
                                                         ConcatOp> {
  void
  generateRuntimeVerification(Operation *op, OpBuilder &builder, Location loc,
                              function_ref<std::string(Operation *, StringRef)>
                                  generateErrorMessage) const {
    auto concatOp = cast<ConcatOp>(op);
    ValueRange inputs = concatOp.getInputs();
    int64_t concatDim = concatOp.getDim();
    int64_t rank = concatOp.getRank();
    RankedTensorType resultType = concatOp.getResultType();

    // Materialize the size of `tensor` along dimension `i`. Static sizes are
    // materialized as constants.
    auto getDimSize = [&](Value tensor, int64_t i) -> Value {
      auto tensorType = cast<RankedTensorType>(tensor.getType());
      if (!tensorType.isDynamicDim(i))
        return arith::ConstantIndexOp::create(builder, loc,
                                              tensorType.getDimSize(i));
      return DimOp::create(builder, loc, tensor, i);
    };

    // Check that all inputs have the same size along every non-concatenated
    // dimension, and that this size matches the static result size (if any).
    // Dynamic result dims are defined by the op itself and cannot be checked
    // here (the result is not yet defined at the insertion point).
    for (int64_t i : llvm::seq<int64_t>(0, rank)) {
      if (i == concatDim)
        continue;

      bool hasDynamicInput = llvm::any_of(inputs, [&](Value input) {
        return cast<RankedTensorType>(input.getType()).isDynamicDim(i);
      });
      if (!hasDynamicInput)
        continue;

      Value expectedSize = getDimSize(inputs.front(), i);
      bool frontIsDynamic =
          cast<RankedTensorType>(inputs.front().getType()).isDynamicDim(i);
      for (Value input : llvm::drop_begin(inputs)) {
        // Pairs of static sizes are verified statically.
        if (!frontIsDynamic &&
            !cast<RankedTensorType>(input.getType()).isDynamicDim(i))
          continue;
        Value size = getDimSize(input, i);
        Value isSameSize = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::eq, size, expectedSize);
        cf::AssertOp::create(
            builder, loc, isSameSize,
            generateErrorMessage(op, "input size mismatch along "
                                     "non-concatenated dimension " +
                                         std::to_string(i)));
      }

      if (!resultType.isDynamicDim(i)) {
        Value resultSize = arith::ConstantIndexOp::create(
            builder, loc, resultType.getDimSize(i));
        Value isSameSize = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::eq, expectedSize, resultSize);
        cf::AssertOp::create(
            builder, loc, isSameSize,
            generateErrorMessage(op, "input size does not match result size "
                                     "along non-concatenated dimension " +
                                         std::to_string(i)));
      }
    }

    // Check that the sum of the input sizes along the concatenation dimension
    // equals the static result size. Dynamic result sizes are reified as the
    // sum of the input sizes (ConcatOp::reifyResultShapes), so they cannot
    // mismatch after lowering.
    if (resultType.isDynamicDim(concatDim))
      return;

    bool hasDynamicInput = llvm::any_of(inputs, [&](Value input) {
      return cast<RankedTensorType>(input.getType()).isDynamicDim(concatDim);
    });
    if (!hasDynamicInput)
      return;

    Value sum = getDimSize(inputs.front(), concatDim);
    for (Value input : llvm::drop_begin(inputs))
      sum = arith::AddIOp::create(builder, loc, sum,
                                  getDimSize(input, concatDim));
    Value resultSize = arith::ConstantIndexOp::create(
        builder, loc, resultType.getDimSize(concatDim));
    Value isEqual = arith::CmpIOp::create(builder, loc,
                                          arith::CmpIPredicate::eq, sum,
                                          resultSize);
    cf::AssertOp::create(
        builder, loc, isEqual,
        generateErrorMessage(op, "sum of input sizes along concatenated "
                                 "dimension does not match result size"));
  }
};

struct ReshapeOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<ReshapeOpInterface,
                                                         ReshapeOp> {
  void
  generateRuntimeVerification(Operation *op, OpBuilder &builder, Location loc,
                              function_ref<std::string(Operation *, StringRef)>
                                  generateErrorMessage) const {
    auto reshapeOp = cast<ReshapeOp>(op);
    auto shapeType = cast<RankedTensorType>(reshapeOp.getShape().getType());
    int64_t shapeSize = shapeType.getDimSize(0);

    // A dynamic shape-operand length is only valid with an unranked result
    // (verified statically); there is then nothing to check here.
    if (ShapedType::isDynamic(shapeSize))
      return;

    // Extract the shape operand values and normalize them to index.
    Type indexType = builder.getIndexType();
    SmallVector<Value> shapeValues;
    shapeValues.reserve(shapeSize);
    for (int64_t i : llvm::seq<int64_t>(0, shapeSize)) {
      Value idx = arith::ConstantIndexOp::create(builder, loc, i);
      Value shapeValue =
          ExtractOp::create(builder, loc, reshapeOp.getShape(), idx);
      if (shapeValue.getType() != indexType)
        shapeValue = arith::IndexCastOp::create(builder, loc, indexType,
                                                shapeValue);
      shapeValues.push_back(shapeValue);
    }

    // Check that the shape operand values agree with the static result sizes.
    // The lowering of memref.reshape trusts the static result dimensions over
    // the shape operand values, so a mismatch yields an inconsistent memref
    // descriptor and out-of-bounds accesses.
    if (auto resultType =
            dyn_cast<RankedTensorType>(reshapeOp.getResult().getType())) {
      for (int64_t i : llvm::seq<int64_t>(0, resultType.getRank())) {
        if (resultType.isDynamicDim(i))
          continue;
        Value expectedSize = arith::ConstantIndexOp::create(
            builder, loc, resultType.getDimSize(i));
        Value isEqual = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::eq, shapeValues[i],
            expectedSize);
        cf::AssertOp::create(
            builder, loc, isEqual,
            generateErrorMessage(op, "shape operand value does not match "
                                     "static result size along dimension " +
                                         std::to_string(i)));
      }
    }

    // Check that the product of the shape operand values equals the number of
    // elements in the source. This requires a ranked source.
    auto sourceType =
        dyn_cast<RankedTensorType>(reshapeOp.getSource().getType());
    if (!sourceType)
      return;

    Value sourceNumElements;
    if (sourceType.hasStaticShape()) {
      sourceNumElements = arith::ConstantIndexOp::create(
          builder, loc, sourceType.getNumElements());
    } else {
      sourceNumElements = DimOp::create(builder, loc, reshapeOp.getSource(), 0);
      for (int64_t i : llvm::seq<int64_t>(1, sourceType.getRank())) {
        Value dimSize = DimOp::create(builder, loc, reshapeOp.getSource(), i);
        sourceNumElements =
            arith::MulIOp::create(builder, loc, sourceNumElements, dimSize);
      }
    }

    std::optional<Value> shapeProduct;
    for (Value shapeValue : shapeValues) {
      shapeProduct =
          shapeProduct ? arith::MulIOp::create(builder, loc, *shapeProduct,
                                               shapeValue)
                       : shapeValue;
    }
    if (!shapeProduct)
      shapeProduct = arith::ConstantIndexOp::create(builder, loc, 1);

    Value isEqual = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, *shapeProduct,
        sourceNumElements);
    cf::AssertOp::create(
        builder, loc, isEqual,
        generateErrorMessage(op, "product of shape operand values does not "
                                 "equal number of elements in source"));
  }
};

} // namespace
} // namespace tensor
} // namespace mlir

void mlir::tensor::registerRuntimeVerifiableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    CastOp::attachInterface<CastOpInterface>(*ctx);
    ConcatOp::attachInterface<ConcatOpInterface>(*ctx);
    DimOp::attachInterface<DimOpInterface>(*ctx);
    ExpandShapeOp::attachInterface<ExpandShapeOpInterface>(*ctx);
    ExtractOp::attachInterface<ExtractInsertOpInterface<ExtractOp>>(*ctx);
    ExtractSliceOp::attachInterface<ExtractSliceOpInterface>(*ctx);
    InsertOp::attachInterface<ExtractInsertOpInterface<InsertOp>>(*ctx);
    ReshapeOp::attachInterface<ReshapeOpInterface>(*ctx);

    // Load additional dialects of which ops may get created.
    ctx->loadDialect<arith::ArithDialect, cf::ControlFlowDialect>();
  });
}
