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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"
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
  if (!layout || !layout.isForSubgroup())
    return failure();

  SmallVector<int64_t> laneData(layout.getLaneData().asArrayRef());
  SmallVector<int64_t> laneLayout(layout.getLaneLayout().asArrayRef());
  auto tdescShape = tdescTy.getShape();
  auto elementType = tdescTy.getElementType();

  // compute sgSize by multiply elements of laneLayout
  // e.g. for 2D layout, sgSize = laneLayout[0] * laneLayout[1]
  // e.g. for 1D layout, sgSize = laneLayout[0]
  int64_t sgSize = llvm::product_of(laneLayout);

  // Case 1: regular loads/stores
  auto scatterAttr = tdescTy.getEncodingOfType<ScatterTensorDescAttr>();
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

std::string xegpu::getTemporaryLayoutName(const OpOperand &operand) {
  const StringRef prefix("layout_operand_");
  unsigned idx = const_cast<OpOperand &>(operand).getOperandNumber();
  return llvm::formatv("{0}{1}", prefix, idx).str();
}

std::string xegpu::getTemporaryLayoutName(const OpResult result) {
  const StringRef prefix = "layout_result_";
  return llvm::formatv("{0}{1}", prefix, result.getResultNumber()).str();
}

xegpu::DistributeLayoutAttr xegpu::getDistributeLayoutAttr(const Value value) {
  if (!value)
    return nullptr;

  if (auto tdescTy =
          dyn_cast_if_present<xegpu::TensorDescType>(value.getType()))
    return tdescTy.getLayoutAttr();

  if (auto result = dyn_cast<OpResult>(value)) {
    Operation *defOp = result.getDefiningOp();
    assert(defOp && "result must have a defining op");

    if (auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(defOp)) {
      auto layout = anchorOp.getAnchorLayout();
      return layout;
    }

    std::string layoutName = getTemporaryLayoutName(result);
    if (defOp->hasAttr(layoutName)) {
      auto layout =
          defOp->getAttrOfType<xegpu::DistributeLayoutAttr>(layoutName);
      return layout;
    }
  }

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto *parentOp = arg.getOwner()->getParentOp();
    if (auto loop = dyn_cast<LoopLikeOpInterface>(parentOp)) {
      OpOperand *tiedInit = loop.getTiedLoopInit(arg);
      if (tiedInit)
        return getDistributeLayoutAttr(tiedInit->get());
    }
  }

  return nullptr;
}
xegpu::DistributeLayoutAttr
xegpu::getDistributeLayoutAttr(const OpOperand &opr) {
  Operation *op = opr.getOwner();
  unsigned idx = const_cast<OpOperand &>(opr).getOperandNumber();

  if (auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(op)) {
    if (auto dpasOp = dyn_cast<xegpu::DpasOp>(op)) {
      if (idx == 0) {
        return dpasOp.getLayoutAAttr();
      } else if (idx == 1) {
        return dpasOp.getLayoutBAttr();
      } else if (idx == 2) {
        return dpasOp.getLayoutCdAttr();
      }
    }
    if (auto convertOp = dyn_cast<xegpu::ConvertLayoutOp>(op)) {
      return convertOp.getInputLayoutAttr();
    }
    auto layout = anchorOp.getAnchorLayout();

    if (idx == 0)
      return layout;

    // For store operations (StoreScatterOp, StoreNdOp, StoreMatrixOp),
    // the layout is valid for the first two operands: value and memref/tdesc.
    // For other operations, the layout applies to the first operand only.
    if (isa<xegpu::StoreScatterOp, xegpu::StoreNdOp, xegpu::StoreMatrixOp>(
            op) &&
        (idx < 2))
      return layout;
  }

  std::string layoutName = xegpu::getTemporaryLayoutName(opr);
  if (op->hasAttr(layoutName)) {
    auto layout = op->getAttrOfType<xegpu::DistributeLayoutAttr>(layoutName);
    return layout;
  }

  return nullptr;
}

// Returns the permanent layout attribute for the given result if it's
// available on the defining op. Otherwise returns the provided layout.
xegpu::DistributeLayoutAttr
maybePickPermanentLayout(xegpu::DistributeLayoutAttr layout,
                         const OpResult &result, mlir::Operation *owner,
                         const std::string &name) {
  xegpu::DistributeLayoutAttr candidate = layout;

  if (auto loadOp = dyn_cast<xegpu::LoadGatherOp>(owner)) {
    if (auto perm = loadOp.getLayoutAttr())
      candidate = perm;
  }

  return candidate;
}

// Returns the permanent layout attribute for the given operand if it's
// available on the defining op. Otherwise returns the provided layout.
xegpu::DistributeLayoutAttr
maybePickPermanentLayout(xegpu::DistributeLayoutAttr layout,
                         const OpOperand &operand, mlir::Operation *owner,
                         const std::string &name) {
  xegpu::DistributeLayoutAttr candidate = layout;
  unsigned idx = const_cast<OpOperand &>(operand).getOperandNumber();

  if (auto storeOp = dyn_cast<xegpu::StoreScatterOp>(owner)) {
    if (idx == 0) {
      if (auto perm = storeOp.getLayoutAttr())
        candidate = perm;
    }
  }

  return candidate;
}

// TODO-LayoutRefactor: Remove this function after replacing use
//  with setTemporaryLayout or setAnchorLayout
void xegpu::setDistributeLayoutAttr(
    const mlir::OpResult &result,
    const mlir::xegpu::DistributeLayoutAttr layout) {
  Operation *owner = result.getOwner();

  if (auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(owner)) {
    if (anchorOp.getAnchorLayout() == layout)
      return;
    anchorOp.setAnchorLayout(layout);
    return;
  }

  std::string name = xegpu::getTemporaryLayoutName(result);
  if (owner->hasAttrOfType<DistributeLayoutAttr>(name)) {
    return;
  }
  if (layout) {
    owner->setAttr(name, layout);
  }
}

// TODO-LayoutRefactor: Remove this function after replacing use
//  with setTemporaryLayout or setAnchorLayout
void xegpu::setDistributeLayoutAttr(const OpOperand &operand,
                                    const DistributeLayoutAttr layout) {
  Operation *owner = operand.getOwner();
  unsigned idx = const_cast<OpOperand &>(operand).getOperandNumber();

  if (!layout) {
    return;
  }
  if (auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(owner)) {
    if (auto dpasOp = dyn_cast<xegpu::DpasOp>(owner)) {
      if (idx == 0) {
        return dpasOp.setLayoutAAttr(layout);
      } else if (idx == 1) {
        return dpasOp.setLayoutBAttr(layout);
      } else if (idx == 2) {
        return dpasOp.setLayoutCdAttr(layout);
      }
    }
    if (auto convertOp = dyn_cast<xegpu::ConvertLayoutOp>(owner)) {
      return convertOp.setInputLayoutAttr(layout);
    }

    // For store operations (StoreScatterOp, StoreNdOp, StoreMatrixOp),
    // the layout is valid for the first two operands: value and memref/tdesc.
    // For other operations, the layout applies to the first operand only.
    if (isa<xegpu::StoreScatterOp, xegpu::StoreNdOp, xegpu::StoreMatrixOp>(
            owner)) {
      if (idx < 2) {
        anchorOp.setAnchorLayout(layout);
      }
    } else {
      if (idx == 0) {
        anchorOp.setAnchorLayout(layout);
      }
    }
  }

  std::string name = xegpu::getTemporaryLayoutName(operand);
  if (owner->hasAttrOfType<DistributeLayoutAttr>(name)) {
    return;
  }
  if (layout) {
    owner->setAttr(name, layout);
  }
}

template <typename T, typename>
xegpu::DistributeLayoutAttr
xegpu::getTemporaryLayout(const T &operandOrResult) {
  Operation *op = operandOrResult.getOwner();

  std::string layoutName = xegpu::getTemporaryLayoutName(operandOrResult);
  if (op->hasAttr(layoutName)) {
    auto layout = op->getAttrOfType<xegpu::DistributeLayoutAttr>(layoutName);
    return layout;
  }

  return nullptr;
}

template xegpu::DistributeLayoutAttr
xegpu::getTemporaryLayout<mlir::OpResult>(const OpResult &result);
template xegpu::DistributeLayoutAttr
xegpu::getTemporaryLayout<mlir::OpOperand>(const OpOperand &operand);

template <typename T, typename>
void xegpu::setTemporaryLayout(const T &operandOrResult,
                               const xegpu::DistributeLayoutAttr layout) {
  Operation *owner = operandOrResult.getOwner();
  std::string name = xegpu::getTemporaryLayoutName(operandOrResult);
  if (owner->hasAttrOfType<xegpu::DistributeLayoutAttr>(name)) {
    return;
  }
  if (layout) {
    owner->setAttr(name, layout);
  }
}

template void xegpu::setTemporaryLayout<mlir::OpResult>(
    const mlir::OpResult &result,
    const mlir::xegpu::DistributeLayoutAttr layout);

template void xegpu::setTemporaryLayout<mlir::OpOperand>(
    const mlir::OpOperand &operand,
    const mlir::xegpu::DistributeLayoutAttr layout);

void xegpu::recoverTemporaryLayoutsDeprecated(Operation *op) {
  op->walk([&](Operation *nestOp) {
    for (OpOperand &opr : nestOp->getOpOperands()) {
      auto layout = getDistributeLayoutAttr(opr.get());
      setDistributeLayoutAttr(opr, layout);
    }

    for (OpResult result : nestOp->getOpResults()) {
      auto layout = getDistributeLayoutAttr(result);
      setDistributeLayoutAttr(result, layout);
    }
  });
}

/// Attach layout attributes to all vector-type operands of operations within
/// the given operation's region. Reports an error if any vector operand lacks
/// a layout attribute.
bool xegpu::recoverTemporaryLayouts(Operation *rootOp) {
  auto result = rootOp->walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      // Layouts are needed for vector type only.
      if (!isa<VectorType>(operand.get().getType()))
        continue;
      auto layout = xegpu::getDistributeLayoutAttr(operand.get());
      if (!layout) {
        op->emitWarning("Could not find layout attribute for operand ")
            << operand.getOperandNumber() << " of operation " << op->getName();
        continue;
      }
      xegpu::setDistributeLayoutAttr(operand, layout);
    }
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

template <typename T, typename>
void xegpu::removeLayoutAttr(const T &operandOrResult) {
  Operation *owner = operandOrResult.getOwner();
  std::string name = xegpu::getTemporaryLayoutName(operandOrResult);
  if (owner->hasAttrOfType<DistributeLayoutAttr>(name))
    owner->removeAttr(name);
}

SmallVector<NamedAttribute>
xegpu::dropSgLayoutAndDataOnAttrs(ArrayRef<NamedAttribute> attrs) {
  SmallVector<NamedAttribute> out;
  out.reserve(attrs.size());

  for (auto attr : attrs) {
    if (auto dist = dyn_cast<xegpu::DistributeLayoutAttr>(attr.getValue())) {
      auto newLayout = dist.dropSgLayoutAndData();
      if (newLayout)
        out.emplace_back(attr.getName(), newLayout);
    } else {
      out.push_back(attr);
    }
  }

  return out;
}

SmallVector<NamedAttribute>
xegpu::dropInstDataOnAttrs(ArrayRef<NamedAttribute> attrs) {
  SmallVector<NamedAttribute> out;
  out.reserve(attrs.size());

  for (auto attr : attrs) {
    if (auto dist = dyn_cast<xegpu::DistributeLayoutAttr>(attr.getValue())) {
      auto newLayout = dist.dropInstData();
      if (newLayout)
        out.emplace_back(attr.getName(), newLayout);
    } else {
      out.push_back(attr);
    }
  }

  return out;
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
    if (op->hasAttrOfType<DistributeLayoutAttr>("layout"))
      op->removeAttr("layout");
    if (op->hasAttrOfType<DistributeLayoutAttr>("layout_a"))
      op->removeAttr("layout_a");
    if (op->hasAttrOfType<DistributeLayoutAttr>("layout_b"))
      op->removeAttr("layout_b");
    if (op->hasAttrOfType<DistributeLayoutAttr>("layout_cd"))
      op->removeAttr("layout_cd");
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

  int64_t srcShapeRank = srcShape.size();
  int64_t targetShapeRank = shape.size();

  SmallVector<int64_t> adjustedTargetShape(srcShape.size());
  int64_t rankDiff = srcShapeRank - targetShapeRank;
  std::fill(adjustedTargetShape.begin(), adjustedTargetShape.begin() + rankDiff,
            1);
  llvm::copy(shape, adjustedTargetShape.begin() + rankDiff);

  SmallVector<Value> result;
  for (SmallVector<int64_t> offsets :
       StaticTileOffsetRange(srcShape, adjustedTargetShape)) {
    SmallVector<int64_t> staticStrides(offsets.size(), 1);
    Value slice = vector::ExtractStridedSliceOp::create(
        builder, loc, value, offsets, adjustedTargetShape, staticStrides);

    // Reshape to remove leading unit dims if needed
    if (srcShapeRank > targetShapeRank) {
      auto targetTy = VectorType::get(shape, vecTy.getElementType());
      slice = vector::ShapeCastOp::create(builder, loc, targetTy, slice);
    }
    result.push_back(slice);
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
    SmallVector<int64_t> staticStrides(tileShape.size(), 1);
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

      xegpu::DistributeLayoutAttr layout =
          xegpu::getDistributeLayoutAttr(input);
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
    // DistributeLayoutAttr

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

std::optional<std::string> xegpu::getChipStr(Operation *op) {
  auto gpuModuleOp = op->getParentOfType<gpu::GPUModuleOp>();

  if (!gpuModuleOp)
    return std::nullopt;

  auto targetAttrs = gpuModuleOp.getTargets();
  if (targetAttrs) {
    for (auto &attr : *targetAttrs) {
      auto xevmAttr = llvm::dyn_cast<xevm::XeVMTargetAttr>(attr);
      if (xevmAttr)
        return xevmAttr.getChip().str();
    }
  }

  return std::nullopt;
}

/// Generates element-wise addition ops of two arrays with same length.
SmallVector<OpFoldResult> xegpu::addElementwise(OpBuilder &builder,
                                                Location loc,
                                                ArrayRef<OpFoldResult> lhs,
                                                ArrayRef<OpFoldResult> rhs) {
  assert(lhs.size() == rhs.size() && "lhs and rhs must have the same size");
  SmallVector<OpFoldResult> results;
  for (auto [l, r] : llvm::zip_equal(lhs, rhs)) {
    auto lval = getValueOrCreateConstantIndexOp(builder, loc, l);
    auto rval = getValueOrCreateConstantIndexOp(builder, loc, r);
    results.push_back(builder.createOrFold<arith::AddIOp>(loc, lval, rval));
  }
  return results;
}

/// Generates element-wise addition ops of two arrays with automatic alignment.
/// When the input arrays have different sizes, the shorter array is
/// right-aligned with the longer array, and the unmatched leading elements from
/// the longer array are preserved unchanged. This is commonly used for offset
/// computation where higher-dimensional offsets need to be added to
/// lower-dimensional adjustments.
///
/// Example:
///   lhs = [l1, l2, l3], rhs = [r1, r2]
///   Result: [11, l2+r1, l3+r2]
SmallVector<OpFoldResult>
xegpu::addWithRightAligned(OpBuilder &builder, Location loc,
                           ArrayRef<OpFoldResult> lhs,
                           ArrayRef<OpFoldResult> rhs) {
  // ensure a is longer than b
  ArrayRef<OpFoldResult> a = lhs.size() >= rhs.size() ? lhs : rhs;
  ArrayRef<OpFoldResult> b = lhs.size() >= rhs.size() ? rhs : lhs;
  SmallVector<OpFoldResult> results(a.take_front(a.size() - b.size()));
  a = a.slice(a.size() - b.size());
  results.append(addElementwise(builder, loc, a, b));
  return results;
}

template <typename T>
int xegpu::getLargestDivisor(T dim, ArrayRef<T> candidates,
                             ArrayRef<T> candidateMultiples) {
  static_assert(std::is_integral<T>::value, "T must be an integer type");
  int largest = -1;
  SmallVector<T> multiples = {1};
  if (!candidateMultiples.empty())
    multiples =
        SmallVector<T>(candidateMultiples.begin(), candidateMultiples.end());
  for (T candidate : candidates) {
    for (T multiple : multiples) {
      int value = static_cast<int>(candidate * multiple);
      if (value != 0 && dim % value == 0 && value > largest)
        largest = value;
    }
  }
  return largest;
}

/// Explicit instantiations
template int xegpu::getLargestDivisor<int>(int dim, ArrayRef<int> candidates,
                                           ArrayRef<int> candidateMultiples);
template int
xegpu::getLargestDivisor<unsigned>(unsigned dim, ArrayRef<unsigned> candidates,
                                   ArrayRef<unsigned> candidateMultiples);
