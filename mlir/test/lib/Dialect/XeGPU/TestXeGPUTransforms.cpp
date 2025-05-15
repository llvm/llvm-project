//===- TestXeGPUTransforms.cpp -- Test Vector transforms and lowerings ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::xegpu;

namespace {

struct TestXeGPUUnrollingPatterns
    : public PassWrapper<TestXeGPUUnrollingPatterns,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestXeGPUUnrollingPatterns)

  StringRef getArgument() const final {
    return "test-xegpu-unrolling-patterns";
  }

  StringRef getDescription() const final {
    return "Test lowering patterns to unroll ops in the xegpu dialect";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    registry.insert<xegpu::XeGPUDialect>();
    registry.insert<vector::VectorDialect>();
  }

  TestXeGPUUnrollingPatterns() = default;
  TestXeGPUUnrollingPatterns(const TestXeGPUUnrollingPatterns &pass)
      : PassWrapper(pass) {}

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    xegpu::UnrollOptions options;
    options.setNativeShapeFn(
        [&](Operation *op) -> std::optional<SmallVector<int64_t>> {
          if (isa<xegpu::CreateNdDescOp, xegpu::UpdateNdOffsetOp,
                  xegpu::PrefetchNdOp, xegpu::LoadNdOp, xegpu::StoreNdOp>(op)) {
            xegpu::TensorDescType tdescTy;
            if (auto createNdOp = dyn_cast<xegpu::CreateNdDescOp>(op)) {
              tdescTy = createNdOp.getType();
            } else if (auto updateNdOp =
                           dyn_cast<xegpu::UpdateNdOffsetOp>(op)) {
              tdescTy = updateNdOp.getTensorDescType();
            } else if (auto prefetchNdOp = dyn_cast<xegpu::PrefetchNdOp>(op)) {
              tdescTy = prefetchNdOp.getTensorDescType();
            } else if (auto loadNdOp = dyn_cast<xegpu::LoadNdOp>(op)) {
              tdescTy = loadNdOp.getTensorDescType();
            } else if (auto storeNdOp = dyn_cast<xegpu::StoreNdOp>(op)) {
              tdescTy = storeNdOp.getTensorDescType();
            }

            if (auto layout = tdescTy.getLayoutAttr()) {
              auto inst_data = layout.getInstData();
              if (inst_data && layout.isSgLayout())
                return SmallVector<int64_t>(inst_data.asArrayRef().begin(),
                                            inst_data.asArrayRef().end());
            }
          }

          if (isa<xegpu::DpasOp>(op))
            return SmallVector<int64_t>{8, 16, 16};

          return std::nullopt;
        });

    options.setUnrolledTypesFn(
        [&](ShapedType type, ArrayRef<int64_t> tileShape) -> SmallVector<Type> {
          Type elemTy = type.getElementType();
          Type newTy;

          // TensorDescType needs to drop the inst_data field in the layout
          // attribute
          if (auto tdescTy = dyn_cast<xegpu::TensorDescType>(type)) {
            Attribute encoding = tdescTy.getEncoding();
            auto layout = llvm::dyn_cast_if_present<xegpu::LayoutAttr>(
                tdescTy.getLayout());
            if (layout) {
              if (layout.getLaneLayout() == nullptr)
                layout = xegpu::LayoutAttr();
              else
                layout = layout.dropInstData();
            }
            newTy = xegpu::TensorDescType::get(ctx, tileShape, elemTy, encoding,
                                               layout);
          } else {
            newTy = type.clone(tileShape, elemTy);
          }

          std::optional<SmallVector<int64_t>> ratio =
              computeShapeRatio(type.getShape(), tileShape);
          assert(ratio && "Expecting the ratio to be valid.");
          return SmallVector<Type>(computeProduct(*ratio), newTy);
        });

    RewritePatternSet patterns(ctx);

    populateXeGPUUnrollPatterns(patterns, options);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestXeGPULowerings() {
  PassRegistration<TestXeGPUUnrollingPatterns>();
}
} // namespace test
} // namespace mlir