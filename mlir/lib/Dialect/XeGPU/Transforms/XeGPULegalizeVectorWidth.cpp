//===- XeGPULegalizeVectorWidth.cpp - Split wide elementwise vector ops ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/uArch/uArchBase.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPULEGALIZEVECTORWIDTH
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-legalize-vector-width"

using namespace mlir;

namespace {

/// Returns the largest divisor of `numComponents` that does not exceed
/// `maxComponents`, so that unrolling produces only whole tiles and never a
/// ragged tail.
static int64_t getUnrollFactor(int64_t numComponents, int64_t maxComponents) {
  for (int64_t candidate = maxComponents; candidate > 1; --candidate)
    if (numComponents % candidate == 0)
      return candidate;
  return 1;
}

/// Returns true for vectors of sub-byte, non-boolean elements (`f4E2M1FN`,
/// `i4`, ...).
///
/// These are packed payload types: they are bit-packed into byte or word
/// vectors when lowered, and SPIR-V has no corresponding scalar type at all.
/// An elementwise op producing or consuming one is a quantization boundary
/// whose own lowering already handles width (see `TruncfToXeVMPattern`, which
/// splits into `xevm.truncf` instruction groups and concatenates the results as
/// `i8` vectors). Unrolling such an op here would instead glue the pieces back
/// together with `insert_strided_slice` on the sub-byte type, materializing
/// sub-byte vector data movement that cannot be translated.
///
/// `i1` is excluded because vector masks are ordinary compute values.
static bool isSubBytePayload(Type type) {
  auto vecType = dyn_cast<VectorType>(type);
  if (!vecType)
    return false;
  Type elemType = vecType.getElementType();
  if (!elemType.isIntOrFloat())
    return false;
  unsigned width = elemType.getIntOrFloatBitWidth();
  return width < 8 && width != 1;
}

/// Native shape function driving `vector::populateVectorUnrollPatterns`.
///
/// Only elementwise, single-result operations are legalized. Everything else
/// -- `vector.shuffle`, length-changing `vector.bitcast`, `xegpu.load_nd`,
/// `xegpu.dpas_mx`, ... -- carries packed payloads whose component count is not
/// a compute width and must be preserved.
static std::optional<SmallVector<int64_t>>
getNativeVectorShape(Operation *op, int64_t maxComponents) {
  if (!OpTrait::hasElementwiseMappableTraits(op) || op->getNumResults() != 1)
    return std::nullopt;

  if (llvm::any_of(op->getOperandTypes(), isSubBytePayload) ||
      llvm::any_of(op->getResultTypes(), isSubBytePayload))
    return std::nullopt;

  auto vecType = dyn_cast<VectorType>(op->getResultTypes()[0]);
  if (!vecType || vecType.getRank() == 0 || vecType.isScalable())
    return std::nullopt;

  // Elementwise ops have matching shapes across operands and result, so the
  // result alone determines legality.
  int64_t trailing = vecType.getShape().back();
  if (trailing <= maxComponents)
    return std::nullopt;

  SmallVector<int64_t> nativeShape(vecType.getRank(), 1);
  nativeShape.back() = getUnrollFactor(trailing, maxComponents);
  if (nativeShape.back() == trailing)
    return std::nullopt;

  return nativeShape;
}

struct XeGPULegalizeVectorWidthPass final
    : public xegpu::impl::XeGPULegalizeVectorWidthBase<
          XeGPULegalizeVectorWidthPass> {
  using XeGPULegalizeVectorWidthBase::XeGPULegalizeVectorWidthBase;

  void runOnOperation() override {
    // A pass option of 0 means "use the target default". Once `uArch` grows a
    // vector-width query this is the single place that needs to consult it.
    int64_t maxComponents = maxVectorComponents
                                ? static_cast<int64_t>(maxVectorComponents)
                                : xegpu::uArch::kDefaultMaxVectorComponents;
    if (maxComponents < 1) {
      getOperation()->emitError(
          "max-vector-components must be greater than zero");
      return signalPassFailure();
    }

    RewritePatternSet patterns(&getContext());
    vector::UnrollVectorOptions options;
    options.setNativeShapeFn(
        [maxComponents](Operation *op) -> std::optional<SmallVector<int64_t>> {
          return getNativeVectorShape(op, maxComponents);
        });
    vector::populateVectorUnrollPatterns(patterns, options);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
