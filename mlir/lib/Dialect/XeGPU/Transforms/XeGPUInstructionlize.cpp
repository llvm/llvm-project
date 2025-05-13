//===---- XeGPUInstructionlize.cpp -- XeGPU Instructionlize Pass ----------===//
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
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUINSTRUCTIONLIZE
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-instructionlize"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace {

/// Unroll XeGPU ops to their instruction-level representation.
class XeGPUInstructionlizePass final
    : public xegpu::impl::XeGPUInstructionlizeBase<XeGPUInstructionlizePass> {
public:
  void runOnOperation() override;

private:
  SmallVector<int64_t> getTileShape(TypedValue<ShapedType> value) const;
  std::optional<SmallVector<int64_t>> getTileShape(Operation *op) const;
  bool needsUnroll(Operation *op) const;
};
} // namespace

SmallVector<int64_t>
XeGPUInstructionlizePass::getTileShape(TypedValue<ShapedType> value) const {
  assert(value && "value must be non-null");
  xegpu::LayoutAttr layout = xegpu::getLayoutAttr(value);
  if (layout && layout.isSgLayout()) {
    if (auto inst_data = layout.getInstData())
      return llvm::to_vector_of<int64_t>(inst_data.asArrayRef());
  }
  return llvm::to_vector(value.getType().getShape());
}

std::optional<SmallVector<int64_t>>
XeGPUInstructionlizePass::getTileShape(Operation *op) const {
  if (isa<xegpu::CreateNdDescOp, xegpu::UpdateNdOffsetOp>(op))
    return getTileShape(cast<TypedValue<ShapedType>>(op->getResult(0)));
  if (isa<xegpu::PrefetchNdOp, xegpu::LoadNdOp>(op))
    return getTileShape(cast<TypedValue<ShapedType>>(op->getOperand(0)));
  if (isa<xegpu::StoreNdOp>(op))
    return getTileShape(cast<TypedValue<ShapedType>>(op->getOperand(1)));

  if (isa<xegpu::DpasOp>(op)) {
    auto a = cast<TypedValue<ShapedType>>(op->getOperand(0));
    auto b = cast<TypedValue<ShapedType>>(op->getOperand(1));
    SmallVector<int64_t> aTileShape = getTileShape(a);
    SmallVector<int64_t> bTileShape = getTileShape(b);

    if (aTileShape.size() != 2 || bTileShape.size() != 2)
      return std::nullopt;

    // semantic check for A and B
    if (aTileShape[1] != bTileShape[0])
      return std::nullopt;

    // semantic check for C
    if (op->getNumOperands() == 3) {
      auto c = cast<TypedValue<ShapedType>>(op->getOperand(2));
      SmallVector<int64_t> cTileShape = getTileShape(c);
      int64_t expectedShape[2] = {aTileShape[0], bTileShape[1]};
      if (!llvm::equal(cTileShape, expectedShape))
        return std::nullopt;
    }

    return SmallVector<int64_t>({aTileShape[0], aTileShape[1], bTileShape[1]});
  }
  return std::nullopt;
}

bool XeGPUInstructionlizePass::needsUnroll(Operation *op) const {
  for (Value opr : op->getOperands()) {
    if (auto value = dyn_cast<TypedValue<ShapedType>>(opr)) {
      auto tileShape = getTileShape(value);
      // the tile should have the same rank as the origial type
      if (tileShape.size() != static_cast<size_t>(value.getType().getRank()))
        return false;
      if (!llvm::equal(tileShape, value.getType().getShape()))
        return true;
    }
  }
  return false;
}

void XeGPUInstructionlizePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  xegpu::UnrollOptions options;
  options.setFilterConstraint([&](Operation *op) -> LogicalResult {
    return needsUnroll(op) ? success() : failure();
  });

  options.setNativeShapeFn(
      [&](Operation *op) -> std::optional<SmallVector<int64_t>> {
        return getTileShape(op);
      });

  options.setUnrolledTypesFn(
      [&](ShapedType type, ArrayRef<int64_t> tileShape) -> SmallVector<Type> {
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
        assert(ratio &&
               "The shape of the type must be a multiple of tileShape.");
        return SmallVector<Type>(computeProduct(*ratio), newTy);
      });

  RewritePatternSet patterns(ctx);

  populateXeGPUUnrollPatterns(patterns, options);
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}
