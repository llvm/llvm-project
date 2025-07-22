//===---- XeGPUUtils.cpp - MLIR Utilities for XeGPUOps   ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility methods for working with the XeGPU dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <numeric>

using namespace mlir;

/// convert ArrayRef<ValueRange> into SmallVector<Value>
SmallVector<Value> xegpu::flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> result;
  for (const auto &vals : values)
    llvm::append_range(result, vals);
  return result;
}

FailureOr<VectorType>
mlir::xegpu::getDistributedVectorType(xegpu::TensorDescType tdescTy) {
  auto layout = llvm::dyn_cast_if_present<LayoutAttr>(tdescTy.getLayout());
  // It only works for subgroup level layout, which only has lane_layout
  // and lane_data, and is to distribute a SIMD code into SIMT code.
  if (!layout || !layout.isSgLayout())
    return failure();

  SmallVector<int64_t> laneData(layout.getLaneData().asArrayRef());
  SmallVector<int64_t> laneLayout(layout.getLaneLayout().asArrayRef());
  auto tdescShape = tdescTy.getShape();
  auto elementType = tdescTy.getElementType();

  // compute sgSize by multiply elements of laneLayout
  // e.g. for 2D layout, sgSize = laneLayout[0] * laneLayout[1]
  // e.g. for 1D layout, sgSize = laneLayout[0]
  auto sgSize = std::accumulate(laneLayout.begin(), laneLayout.end(), 1,
                                std::multiplies<int64_t>());

  // Case 1: regular loads/stores
  auto scatterAttr = tdescTy.getEncodingAsScatterTensorDescAttr();
  if (scatterAttr) {
    auto chunkSize = scatterAttr.getChunkSize().getInt();
    // Verify if the first dimension of the tensor descriptor shape is
    // distributable.
    assert(tdescShape[0] == laneLayout[0] &&
           "tensor descriptor shape is not distributable");
    return VectorType::get({chunkSize}, elementType);
  }

  // Case 2: block loads/stores
  // Check if the tensor descriptor shape is distributable.
  int64_t tensorSize = 1;
  for (auto [tdescDim, laneDim, laneDataDim] :
       llvm::zip_equal(tdescShape, laneLayout, laneData)) {
    assert((tdescDim % (laneDim * laneDataDim) == 0) &&
           "tensor descriptor shape is not distributable");
    tensorSize *= tdescDim;
  }
  // tensorSize must be adjusted for array_length.
  tensorSize *= tdescTy.getArrayLength();

  return VectorType::get({tensorSize / sgSize}, elementType);
}

FailureOr<VectorType>
mlir::xegpu::getDistributedVectorType(VectorType originalType,
                                      xegpu::LayoutAttr layout) {
  int64_t rank = originalType.getRank();
  // Distributed vector type is only supported for 1D, 2D and 3D vectors.
  if (rank < 1 || rank > 3)
    return failure();
  ArrayRef<int64_t> shape = originalType.getShape();
  // arrayLength is 1 for 1D and 2D vectors, and equal to the first dimension
  // of the 3D vector.
  int arrayLength = 1;
  if (rank == 3) {
    arrayLength = shape[0];
    shape = shape.drop_front();
  }
  auto helperTdescTy = xegpu::TensorDescType::get(
      shape, originalType.getElementType(), arrayLength,
      /*boundary_check=*/true,
      /*memory_space=*/xegpu::MemorySpace::Global, layout);
  return xegpu::getDistributedVectorType(helperTdescTy);
}

std::string xegpu::getLayoutName(const OpOperand &operand) {
  const StringRef prefix("layout_operand_");
  unsigned idx = const_cast<OpOperand &>(operand).getOperandNumber();
  return llvm::formatv("{0}{1}", prefix, idx).str();
}

std::string xegpu::getLayoutName(const OpResult result) {
  const StringRef prefix = "layout_result_";
  return llvm::formatv("{0}{1}", prefix, result.getResultNumber()).str();
}

xegpu::LayoutAttr xegpu::getLayoutAttr(const Value value) {
  if (!value)
    return nullptr;

  if (auto tdescTy =
          dyn_cast_if_present<xegpu::TensorDescType>(value.getType()))
    return tdescTy.getLayoutAttr();

  if (auto result = dyn_cast<OpResult>(value)) {
    Operation *defOp = result.getDefiningOp();
    assert(defOp && "result must have a defining op");

    // for LoadNdOp, the layout is stored in the tensor descriptor
    if (auto loadNd = dyn_cast<xegpu::LoadNdOp>(defOp))
      return getLayoutAttr(loadNd.getTensorDesc());

    std::string layoutName = getLayoutName(result);
    if (defOp->hasAttr(layoutName))
      return defOp->getAttrOfType<xegpu::LayoutAttr>(layoutName);
  }

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto parentOp = arg.getOwner()->getParentOp();
    if (auto loop = dyn_cast<LoopLikeOpInterface>(parentOp)) {
      OpOperand *tiedInit = loop.getTiedLoopInit(arg);
      return getLayoutAttr(tiedInit->get());
    }
  }

  return nullptr;
}

xegpu::LayoutAttr xegpu::getLayoutAttr(const OpOperand &opr) {
  Operation *op = opr.getOwner();
  std::string layoutName = xegpu::getLayoutName(opr);
  if (op->hasAttr(layoutName))
    return op->getAttrOfType<xegpu::LayoutAttr>(layoutName);
  return getLayoutAttr(opr.get());
}

template <typename T, typename>
void xegpu::setLayoutAttr(const T &operandOrResult, const LayoutAttr layout) {
  Operation *owner = operandOrResult.getOwner();
  std::string name = xegpu::getLayoutName(operandOrResult);
  if (layout && !owner->hasAttrOfType<LayoutAttr>(name))
    owner->setAttr(name, layout);
}

// Explicit instantiation for OpResult
template void
xegpu::setLayoutAttr<mlir::OpResult>(const mlir::OpResult &result,
                                     const mlir::xegpu::LayoutAttr layout);

// Explicit instantiation for OpOperand
template void
xegpu::setLayoutAttr<mlir::OpOperand>(const mlir::OpOperand &operand,
                                      const mlir::xegpu::LayoutAttr layout);

void xegpu::setLayoutAttrs(Operation *op,
                           function_ref<LayoutAttr(Value)> getLayoutImpl) {
  op->walk([&](Operation *nestOp) {
    for (OpOperand &opr : nestOp->getOpOperands()) {
      auto layout = getLayoutImpl(opr.get());
      setLayoutAttr(opr, layout);
    }
    for (OpResult result : nestOp->getOpResults()) {
      auto layout = getLayoutImpl(result);
      setLayoutAttr(result, layout);
    }
  });
}

template <typename T, typename>
void xegpu::removeLayoutAttr(const T &operandOrResult) {
  Operation *owner = operandOrResult.getOwner();
  std::string name = xegpu::getLayoutName(operandOrResult);
  if (owner->hasAttrOfType<LayoutAttr>(name))
    owner->removeAttr(name);
}

// Explicit instantiation for OpResult
template void
xegpu::removeLayoutAttr<mlir::OpResult>(const mlir::OpResult &result);

// Explicit instantiation for OpOperand
template void
xegpu::removeLayoutAttr<mlir::OpOperand>(const mlir::OpOperand &operand);

void xegpu::removeLayoutAttrs(Operation *op) {
  op->walk([&](Operation *nestOp) {
    for (OpOperand &opr : nestOp->getOpOperands())
      removeLayoutAttr(opr);
    for (OpResult result : nestOp->getOpResults())
      removeLayoutAttr(result);
  });
}

SmallVector<Value>
xegpu::extractVectorsWithShapeFromValue(OpBuilder &builder, Location loc,
                                        Value value, ArrayRef<int64_t> shape) {
  auto vecTy = dyn_cast<VectorType>(value.getType());
  if (!vecTy)
    return {value};

  ArrayRef<int64_t> srcShape = vecTy.getShape();
  if (!computeShapeRatio(srcShape, shape))
    return {value};

  SmallVector<Value> result;
  for (SmallVector<int64_t> offsets : StaticTileOffsetRange(srcShape, shape)) {
    SmallVector<int64_t> staticStrides(offsets.size(), 1);
    result.push_back(vector::ExtractStridedSliceOp::create(
        builder, loc, value, offsets, shape, staticStrides));
  }

  return result;
}

Value xegpu::createVectorWithShapeFromValues(OpBuilder &builder, Location loc,
                                             ValueRange values,
                                             ArrayRef<int64_t> shape) {
  VectorType inputTy = dyn_cast<VectorType>(values[0].getType());
  assert(llvm::all_of(values.getTypes(),
                      [&](Type type) { return type == inputTy; }) &&
         "values must be of the same VectorType");

  Type elemTy = inputTy.getElementType();
  ArrayRef<int64_t> tileShape = inputTy.getShape();

  VectorType resultTy = VectorType::get(shape, elemTy);
  auto zeroAttr = builder.getZeroAttr(elemTy);
  Value result = arith::ConstantOp::create(
      builder, loc, resultTy, DenseElementsAttr::get(resultTy, zeroAttr));

  for (auto [src, offsets] :
       llvm::zip_equal(values, StaticTileOffsetRange(shape, tileShape))) {
    SmallVector<int64_t> staticStrides(offsets.size(), 1);
    result = vector::InsertStridedSliceOp::create(builder, loc, src, result,
                                                  offsets, staticStrides);
  }
  return result;
}

void xegpu::doSCFStructuralTypeConversionWithTensorType(
    Operation *op, TypeConverter converter) {
  MLIRContext *context = op->getContext();

  auto materializeCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                            Location loc) -> Value {
    return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
        .getResult(0);
  };

  { // convert VectorType to RankedTensorType for SCF Structural ops
    TypeConverter converter;
    converter.addConversion([](Type type) -> Type { return type; });
    converter.addConversion([](VectorType type) -> Type {
      return RankedTensorType::get(type.getShape(), type.getElementType());
    });
    converter.addSourceMaterialization(materializeCast);
    converter.addTargetMaterialization(materializeCast);

    mlir::ConversionTarget target(*context);
    target.addLegalOp<UnrealizedConversionCastOp>();

    mlir::RewritePatternSet patterns(context);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    (void)mlir::applyPartialConversion(op, target, std::move(patterns));
  }

  { // propagate the layout attribute to RankedTensorType by checking
    // BuiltInUnrealizedCastOps
    // for VectorType to RankedTensorType cast.
    op->walk([](UnrealizedConversionCastOp castOp) {
      if (castOp.getNumOperands() != 1 || castOp.getNumResults() != 1)
        return WalkResult::skip();

      Value input = castOp.getInputs()[0];
      Value result = castOp.getResults()[0];
      auto inputTy = dyn_cast<VectorType>(input.getType());
      auto resultTy = dyn_cast<RankedTensorType>(result.getType());

      // Only look at ops casting from VectorType to RankedTensorType
      if (!inputTy || !resultTy)
        return WalkResult::skip();

      xegpu::LayoutAttr layout = xegpu::getLayoutAttr(input);
      if (!layout)
        return WalkResult::skip();

      RankedTensorType newTy = resultTy.cloneWithEncoding(layout);
      result.setType(newTy);

      // update the arguments if user is a LoopLike op.
      for (OpOperand &use : result.getUses()) {
        if (auto loop = dyn_cast<LoopLikeOpInterface>(use.getOwner())) {
          BlockArgument arg = loop.getTiedLoopRegionIterArg(&use);
          arg.setType(newTy);
        }
        // whileOp has two regions, the BlockArgument of the after region
        // is not exposed by LoopLikeOpInterface
        if (auto whileOp = dyn_cast<scf::WhileOp>(use.getOwner())) {
          unsigned idx = use.getOperandNumber();
          BlockArgument arg = whileOp.getAfterArguments()[idx];
          arg.setType(newTy);
        }
      }
      return WalkResult::advance();
    });

    // using yieldOp as anchor to update the result type of its ParentOp
    op->walk([](scf::YieldOp yieldOp) {
      Operation *parentOp = yieldOp->getParentOp();
      for (OpResult r : parentOp->getOpResults()) {
        unsigned idx = r.getResultNumber();
        Type resultTy = r.getType();
        Type yieldTy = yieldOp.getResults()[idx].getType();
        if (isa<RankedTensorType>(resultTy) && yieldTy != resultTy)
          r.setType(yieldTy);
      }
    });
  }

  { // perform the conversion from RankedTensorType to VectorType based on the
    // LayoutAttr

    // Handle the UnrealizedConversionCastOp introduced by the first step.
    // For vector->RankedTensorType, it will simply forward the inputs.
    // For RankedTensorType->vector, it will update the inputs with the
    // one from the adaptor.
    class UnrealizedConversionCastOpPattern
        : public OpConversionPattern<mlir::UnrealizedConversionCastOp> {
      using OpConversionPattern<
          mlir::UnrealizedConversionCastOp>::OpConversionPattern;

      mlir::LogicalResult
      matchAndRewrite(mlir::UnrealizedConversionCastOp op,
                      OneToNOpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter) const override {
        auto inputs = op.getOperands();
        auto outputs = op.getOutputs();

        if (inputs.size() != 1 || outputs.size() != 1)
          return failure();

        auto inputTy = inputs[0].getType();
        auto outputTy = outputs[0].getType();

        if (isa<VectorType>(inputTy) && isa<RankedTensorType>(outputTy)) {
          rewriter.replaceOpWithMultiple(op, adaptor.getInputs());
          return success();
        }

        if (isa<RankedTensorType>(inputTy) && isa<VectorType>(outputTy)) {
          SmallVector<Value> values = xegpu::flattenValues(adaptor.getInputs());
          auto newOp = UnrealizedConversionCastOp::create(rewriter, op.getLoc(),
                                                          outputTy, values);
          rewriter.replaceOp(op, newOp);
          return success();
        }
        return failure();
      }
    };

    converter.addSourceMaterialization(materializeCast);
    converter.addTargetMaterialization([&](OpBuilder &builder, TypeRange type,
                                           ValueRange inputs, Location loc) {
      return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
          .getResults();
    });

    mlir::ConversionTarget target(*context);
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [](UnrealizedConversionCastOp op) {
          auto isTensorTy = [](Type type) {
            return isa<RankedTensorType>(type);
          };
          return llvm::none_of(op->getOperandTypes(), isTensorTy) &&
                 llvm::none_of(op->getResultTypes(), isTensorTy);
        });
    mlir::RewritePatternSet patterns(context);
    patterns.insert<UnrealizedConversionCastOpPattern>(context);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    (void)mlir::applyPartialConversion(op, target, std::move(patterns));
  }
}
