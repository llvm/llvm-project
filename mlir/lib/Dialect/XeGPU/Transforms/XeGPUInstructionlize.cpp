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
  // Get the tile shape for a given value. If the value has a layout
  // attribute and it is an SG layout, return the inst_data as the tile shape
  // if inst_data is available; otherwise, return the original shape of the
  // value. If the value does not have an SG layout, return std::nullopt.
  std::optional<SmallVector<int64_t>>
  getTileShape(TypedValue<ShapedType> value) const;

  // Get the tile shape for a given operation.
  std::optional<SmallVector<int64_t>> getTileShape(Operation *op) const;

  // Determine if the operation requires unrolling. Return false if all operands
  // and results have tile shapes identical to their original types. Otherwise,
  // return true.
  bool needsUnroll(Operation *op) const;
};
} // namespace

std::optional<SmallVector<int64_t>>
XeGPUInstructionlizePass::getTileShape(TypedValue<ShapedType> value) const {
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
    std::optional<SmallVector<int64_t>> aTile = getTileShape(a);
    std::optional<SmallVector<int64_t>> bTile = getTileShape(b);

    if (!aTile || aTile->size() != 2 || !bTile || bTile->size() != 2)
      return std::nullopt;

    // semantic check for A and B
    if ((*aTile)[1] != (*bTile)[0])
      return std::nullopt;

    // semantic check for C
    if (op->getNumOperands() == 3) {
      auto c = cast<TypedValue<ShapedType>>(op->getOperand(2));
      std::optional<SmallVector<int64_t>> cTile = getTileShape(c);
      int64_t expectedCTile[2] = {(*aTile)[0], (*bTile)[1]};
      if (!cTile || !llvm::equal(*cTile, expectedCTile))
        return std::nullopt;
    }

    return SmallVector<int64_t>({(*aTile)[0], (*aTile)[1], (*bTile)[1]});
  }
  return std::nullopt;
}

bool XeGPUInstructionlizePass::needsUnroll(Operation *op) const {
  for (Value opr : op->getOperands()) {
    if (auto value = dyn_cast<TypedValue<ShapedType>>(opr)) {
      std::optional<SmallVector<int64_t>> tileShape = getTileShape(value);
      // the tile should have the same rank as the origial type
      if (!tileShape ||
          tileShape->size() != static_cast<size_t>(value.getType().getRank()))
        return false;
      if (!llvm::equal(*tileShape, value.getType().getShape()))
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
