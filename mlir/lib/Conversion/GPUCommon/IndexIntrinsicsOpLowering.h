//===- IndexIntrinsicsOpLowering.h - GPU IndexOps Lowering class *- C++ -*-===//
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
enum class IndexKind : uint32_t { Other = 0, Block = 1, Grid = 2 };
enum class IntrType : uint32_t {
  None = 0,
  Id = 1,
  Dim = 2,
};

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
  explicit OpLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<Op>(typeConverter),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()),
        indexKind(IndexKind::Other), intrType(IntrType::None) {}

  explicit OpLowering(LLVMTypeConverter &typeConverter, IndexKind indexKind,
                      IntrType intrType)
      : ConvertOpToLLVMPattern<Op>(typeConverter),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()),
        indexKind(indexKind), intrType(intrType) {}

  // Convert the kernel arguments to an LLVM type, preserve the rest.
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    Operation *newOp;
    switch (op.getDimension()) {
    case gpu::Dimension::x:
      newOp = rewriter.create<XOp>(loc, IntegerType::get(context, 32));
      break;
    case gpu::Dimension::y:
      newOp = rewriter.create<YOp>(loc, IntegerType::get(context, 32));
      break;
    case gpu::Dimension::z:
      newOp = rewriter.create<ZOp>(loc, IntegerType::get(context, 32));
      break;
    }

    // Order of priority for bounds:
    // 1. The upper_bound attribute
    // 2. Inherent attributes on a surrounding gpu.func
    // 3. Discardable attributes on a surrounding function of any kind
    // The below code handles these in reverse order so that more important
    // sources overwrite less important ones.
    DenseI32ArrayAttr funcBounds = nullptr;
    if (auto funcOp = op->template getParentOfType<FunctionOpInterface>()) {
      switch (indexKind) {
      case IndexKind::Block: {
        auto blockHelper =
            gpu::GPUDialect::KnownBlockSizeAttrHelper(op.getContext());
        if (blockHelper.isAttrPresent(funcOp))
          funcBounds = blockHelper.getAttr(funcOp);
        break;
      }
      case IndexKind::Grid: {
        auto gridHelper =
            gpu::GPUDialect::KnownGridSizeAttrHelper(op.getContext());
        if (gridHelper.isAttrPresent(funcOp))
          funcBounds = gridHelper.getAttr(funcOp);
        break;
      }
      case IndexKind::Other:
        break;
      }
    }
    if (auto gpuFunc = op->template getParentOfType<gpu::GPUFuncOp>()) {
      switch (indexKind) {
      case IndexKind::Block:
        funcBounds = gpuFunc.getKnownBlockSizeAttr();
        break;
      case IndexKind::Grid:
        funcBounds = gpuFunc.getKnownGridSizeAttr();
        break;
      case IndexKind::Other:
        break;
      }
    }
    std::optional<int32_t> upperBound;
    if (funcBounds)
      upperBound =
          funcBounds.asArrayRef()[static_cast<uint32_t>(op.getDimension())];
    if (auto opBound = op.getUpperBound())
      upperBound = opBound->getZExtValue();

    if (upperBound && intrType != IntrType::None) {
      int32_t min = (intrType == IntrType::Dim ? 1 : 0);
      int32_t max = *upperBound - (intrType == IntrType::Id ? 0 : 1);
      newOp->setAttr(
          "range", DenseI32ArrayAttr::get(op.getContext(), ArrayRef{min, max}));
    }
    if (indexBitwidth > 32) {
      newOp = rewriter.create<LLVM::SExtOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp->getResult(0));
    } else if (indexBitwidth < 32) {
      newOp = rewriter.create<LLVM::TruncOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp->getResult(0));
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};
} // namespace index_lowering
} // namespace gpu
} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_INDEXINTRINSICSOPLOWERING_H_
