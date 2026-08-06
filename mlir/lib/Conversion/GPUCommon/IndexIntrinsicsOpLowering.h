//===- IndexIntrinsicsOpLowering.h - GPU Index Op Lowering ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUCOMMON_INDEXINTRINSICSOPLOWERING_H_
#define MLIR_CONVERSION_GPUCOMMON_INDEXINTRINSICSOPLOWERING_H_

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace gpu {
namespace index_lowering {
// Alias so existing call sites don't need updating.
using IndexKind = gpu::DimensionKind;

enum class IntrType : uint32_t {
  None = 0,
  Id = 1,
  Dim = 2,
};

/// Returns a ConstantRangeAttr for a GPU index op, or nullptr if no bounds
/// are found. `bitWidth` controls the width of the returned range.
/// Checks the provided upper_bound from the op (highest priority), inherent
/// attrs on enclosing `gpu.func`s, and discardable attributes on other
/// enclosing function ops (lowest priority). However, in the case where
/// a dimension is known to have a constant value, returns a range indicating
/// that value.
LLVM::ConstantRangeAttr getIndexOpRange(Operation *op, gpu::Dimension dim,
                                        std::optional<uint32_t> opUpperBound,
                                        IndexKind indexKind, IntrType intrType,
                                        unsigned bitWidth);

// Rewriting that replaces Op with XOp, YOp, or ZOp depending on the dimension
// that Op operates on.  Op is assumed to return an `index` value and
// XOp, YOp and ZOp are assumed to return an `llvm.i32` value.  Depending on
// `indexBitwidth`, sign-extend or truncate the resulting value to match the
// bitwidth expected by the consumers of the value.
template <typename Op, typename XOp, typename YOp, typename ZOp>
struct OpLowering : public ConvertOpToLLVMPattern<Op> {
private:
  unsigned indexBitwidth;
  IndexKind indexKind;
  IntrType intrType;

public:
  explicit OpLowering(const LLVMTypeConverter &typeConverter,
                      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<Op>(typeConverter, benefit),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()),
        indexKind(IndexKind::Other), intrType(IntrType::None) {}

  explicit OpLowering(const LLVMTypeConverter &typeConverter,
                      IndexKind indexKind, IntrType intrType,
                      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<Op>(typeConverter, benefit),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()),
        indexKind(indexKind), intrType(intrType) {}

  // Convert the kernel arguments to an LLVM type, preserve the rest.
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    Operation *newOp;
    switch (op.getDimension()) {
    case gpu::Dimension::x:
      newOp = XOp::create(rewriter, loc, IntegerType::get(context, 32));
      break;
    case gpu::Dimension::y:
      newOp = YOp::create(rewriter, loc, IntegerType::get(context, 32));
      break;
    case gpu::Dimension::z:
      newOp = ZOp::create(rewriter, loc, IntegerType::get(context, 32));
      break;
    }

    std::optional<uint32_t> opBound;
    if (auto bound = op.getUpperBound())
      opBound = static_cast<uint32_t>(bound->getZExtValue());
    if (auto range = getIndexOpRange(op, op.getDimension(), opBound, indexKind,
                                     intrType, /*bitWidth=*/32))
      newOp->setAttr("range", range);

    if (indexBitwidth > 32) {
      newOp = LLVM::SExtOp::create(rewriter, loc,
                                   IntegerType::get(context, indexBitwidth),
                                   newOp->getResult(0));
    } else if (indexBitwidth < 32) {
      newOp = LLVM::TruncOp::create(rewriter, loc,
                                    IntegerType::get(context, indexBitwidth),
                                    newOp->getResult(0));
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};
} // namespace index_lowering
} // namespace gpu
} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_INDEXINTRINSICSOPLOWERING_H_
