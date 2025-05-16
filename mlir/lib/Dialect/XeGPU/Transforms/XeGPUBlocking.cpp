//===---- XeGPUBlocking.cpp ---- XeGPU Instructionlize Pass ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUBLOCKING
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-blocking"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace {

void resolveUnrealizedConversionCastOp(UnrealizedConversionCastOp castOp) {
  ValueRange inputs = castOp.getInputs();
  ValueRange outputs = castOp.getOutputs();

  if (inputs.size() == 1 && outputs.size() == 1) {
    castOp->replaceAllUsesWith(inputs);
    castOp->erase();
  }

  VectorType inputTy = dyn_cast<VectorType>(inputs[0].getType());
  VectorType outputTy = dyn_cast<VectorType>(outputs[0].getType());
  if (inputTy && outputTy) {
    OpBuilder builder(castOp);
    // unpack
    if (inputs.size() > 1 && outputs.size() == 1) {
      ArrayRef<int64_t> shape = outputTy.getShape();
      Value result = xegpu::createVectorWithShapeFromValues(
          builder, castOp.getLoc(), inputs, shape);
      castOp->replaceAllUsesWith(ValueRange(result));
      castOp->erase();
    }

    // pack
    if (castOp.getNumResults() > 1 && castOp.getNumOperands() == 1) {
      ArrayRef<int64_t> tileShape = outputTy.getShape();
      SmallVector<Value> results = xegpu::extractVectorsWithShapeFromValue(
          builder, castOp.getLoc(), inputs[0], tileShape);
      castOp->replaceAllUsesWith(results);
      castOp->erase();
    }
  }
}

/// Unroll XeGPU ops to their instruction-level representation.
class XeGPUBlockingPass final
    : public xegpu::impl::XeGPUBlockingBase<XeGPUBlockingPass> {
public:
  void runOnOperation() override;

private:
  // Get the tile shape for a given value. If the value has a layout
  // attribute and it is an SG layout, return the inst_data as the tile shape
  // if inst_data is available; otherwise, return the original shape of the
  // value. If the value does not have an SG layout, return std::nullopt.
  std::optional<SmallVector<int64_t>>
  getTileShape(TypedValue<ShapedType> value) const;

  std::optional<SmallVector<int64_t>> getTileShape(OpOperand &operand) const;

  std::optional<SmallVector<int64_t>> getTileShape(OpResult result) const;

  // Get the tile shape for a given operation.
  std::optional<SmallVector<int64_t>> getTileShape(Operation *op) const;

  // Determine if the operation requires unrolling. Return false if all operands
  // and results have tile shapes identical to their original types. Otherwise,
  // return true.
  bool needsUnroll(Operation *op) const;
};
} // namespace

std::optional<SmallVector<int64_t>>
XeGPUBlockingPass::getTileShape(TypedValue<ShapedType> value) const {
  assert(value && "value must be non-null");
  xegpu::LayoutAttr layout = xegpu::getLayoutAttr(value);
  if (layout && layout.isSgLayout()) {
    if (auto inst_data = layout.getInstData())
      return llvm::to_vector_of<int64_t>(inst_data.asArrayRef());
    return llvm::to_vector(value.getType().getShape());
  }
  return std::nullopt;
}

std::optional<SmallVector<int64_t>>
XeGPUBlockingPass::getTileShape(OpOperand &operand) const {
  xegpu::LayoutAttr layout = xegpu::getLayoutAttr(operand);
  if (layout && layout.isSgLayout()) {
    if (auto inst_data = layout.getInstData())
      return llvm::to_vector_of<int64_t>(inst_data.asArrayRef());

    if (auto type = dyn_cast<ShapedType>(operand.get().getType()))
      return llvm::to_vector(type.getShape());
  }
  return std::nullopt;
}

std::optional<SmallVector<int64_t>>
XeGPUBlockingPass::getTileShape(OpResult result) const {
  xegpu::LayoutAttr layout = xegpu::getLayoutAttr(result);
  if (layout && layout.isSgLayout()) {
    if (auto inst_data = layout.getInstData())
      return llvm::to_vector_of<int64_t>(inst_data.asArrayRef());

    if (auto type = dyn_cast<ShapedType>(result.getType()))
      return llvm::to_vector(type.getShape());
  }
  return std::nullopt;
}

std::optional<SmallVector<int64_t>>
XeGPUBlockingPass::getTileShape(Operation *op) const {
  if (isa<xegpu::CreateNdDescOp, xegpu::UpdateNdOffsetOp>(op))
    return getTileShape(op->getOpResult(0));
  if (isa<xegpu::PrefetchNdOp, xegpu::LoadNdOp>(op))
    return getTileShape(op->getOpOperand(0));
  if (isa<xegpu::StoreNdOp>(op))
    return getTileShape(op->getOpOperand(1));

  if (isa<xegpu::DpasOp>(op)) {
    std::optional<SmallVector<int64_t>> aTile =
        getTileShape(op->getOpOperand(0));
    std::optional<SmallVector<int64_t>> bTile =
        getTileShape(op->getOpOperand(1));

    if (!aTile || aTile->size() != 2 || !bTile || bTile->size() != 2)
      return std::nullopt;

    // semantic check for A and B
    if ((*aTile)[1] != (*bTile)[0])
      return std::nullopt;

    // semantic check for C
    if (op->getNumOperands() == 3) {
      std::optional<SmallVector<int64_t>> cTile =
          getTileShape(op->getOpOperand(2));
      int64_t expectedCTile[2] = {(*aTile)[0], (*bTile)[1]};
      if (!cTile || !llvm::equal(*cTile, expectedCTile))
        return std::nullopt;
    }

    return SmallVector<int64_t>({(*aTile)[0], (*aTile)[1], (*bTile)[1]});
  }

  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1)
    return getTileShape(op->getOpResult(0));

  return std::nullopt;
}

bool XeGPUBlockingPass::needsUnroll(Operation *op) const {
  if (isa<LoopLikeOpInterface>(op))
    return false;

  for (auto &opr : op->getOpOperands()) {
    std::optional<SmallVector<int64_t>> tileShape = getTileShape(opr);
    auto shapedType = dyn_cast<ShapedType>(opr.get().getType());
    if (!shapedType)
      continue;

    if (tileShape && !llvm::equal(*tileShape, shapedType.getShape()))
      return true;
  }

  for (auto result : op->getOpResults()) {
    std::optional<SmallVector<int64_t>> tileShape = getTileShape(result);
    auto shapedType = dyn_cast<ShapedType>(result.getType());
    if (!shapedType)
      continue;

    if (tileShape && !llvm::equal(*tileShape, shapedType.getShape()))
      return true;
  }
  return false;
}

void XeGPUBlockingPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  Operation *mod = getOperation();

  // Preserve the LayoutAttr for each operand to the owner's DictionaryAttr.
  // This ensures that the LayoutAttr remains accessible even if the defining
  // operation is replaced.
  xegpu::setLayoutAttrs(mod, [&](Value v) { return xegpu::getLayoutAttr(v); });

  // Perform type conversion for SCF control folow ops
  xegpu::doSCFStructuralTypeConversionWithTensorType(mod);

  xegpu::UnrollOptions options;
  options.setFilterConstraint([&](Operation *op) -> LogicalResult {
    return needsUnroll(op) ? success() : failure();
  });

  options.setNativeShapeFn([&](Operation *op) { return getTileShape(op); });

  options.setUnrolledTypesFn([&](ShapedType type, ArrayRef<int64_t> tileShape) {
    Type elemTy = type.getElementType();
    Type newTy;

    if (auto tdescTy = dyn_cast<xegpu::TensorDescType>(type))
      newTy = xegpu::TensorDescType::get(
          ctx, tileShape, elemTy, tdescTy.getEncoding(),
          tdescTy.getLayoutAttr().dropInstData());
    else
      newTy = type.clone(tileShape, elemTy);

    std::optional<SmallVector<int64_t>> ratio =
        computeShapeRatio(type.getShape(), tileShape);
    assert(ratio && "The shape of the type must be a multiple of tileShape.");
    return SmallVector<Type>(computeProduct(*ratio), newTy);
  });

  RewritePatternSet patterns(ctx);

  vector::UnrollVectorOptions vectorOptions;
  vectorOptions.setNativeShapeFn(options.nativeShape);

  populateXeGPUUnrollPatterns(patterns, options);
  vector::populateVectorUnrollPatterns(patterns, vectorOptions);

  (void)applyPatternsGreedily(mod, std::move(patterns));

  mod->walk([&](Operation *op) {
    if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op))
      resolveUnrealizedConversionCastOp(castOp);

    for (OpOperand &opr : op->getOpOperands()) {
      std::string name = xegpu::getLayoutName(opr);
      if (auto layout = op->getAttrOfType<xegpu::LayoutAttr>(name))
        op->removeAttr(name);
    }

    for (OpResult result : op->getOpResults()) {
      std::string name = xegpu::getLayoutName(result);
      if (auto layout = op->getAttrOfType<xegpu::LayoutAttr>(name)) {
        op->removeAttr(name);
        if (!isa<LoopLikeOpInterface>(op))
          xegpu::setLayoutAttr(result, layout.dropInstData());
      }
    }
  });
}
