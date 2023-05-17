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

// Rewriting that replaces Op with XOp, YOp, or ZOp depending on the dimension
// that Op operates on.  Op is assumed to return an `index` value and
// XOp, YOp and ZOp are assumed to return an `llvm.i32` value.  Depending on
// `indexBitwidth`, sign-extend or truncate the resulting value to match the
// bitwidth expected by the consumers of the value.
template <typename Op, typename XOp, typename YOp, typename ZOp>
struct GPUIndexIntrinsicOpLowering : public ConvertOpToLLVMPattern<Op> {
private:
  unsigned indexBitwidth;
  StringRef boundsAttrName;

public:
  explicit GPUIndexIntrinsicOpLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<Op>(typeConverter),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()),
        boundsAttrName("") {}

  explicit GPUIndexIntrinsicOpLowering(LLVMTypeConverter &typeConverter,
                                       StringRef boundsAttrName)
      : ConvertOpToLLVMPattern<Op>(typeConverter),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()),
        boundsAttrName(boundsAttrName) {}

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

    Operation *function;
    if (auto gpuFunc = op->template getParentOfType<gpu::GPUFuncOp>())
      function = gpuFunc;
    if (auto llvmFunc = op->template getParentOfType<LLVM::LLVMFuncOp>())
      function = llvmFunc;
    if (!boundsAttrName.empty() && function) {
      if (auto attr = function->template getAttrOfType<DenseI32ArrayAttr>(
              boundsAttrName)) {
        int32_t maximum = attr[static_cast<uint32_t>(op.getDimension())];
        newOp->setAttr("range", rewriter.getDenseI32ArrayAttr({0, maximum}));
      }
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

} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_INDEXINTRINSICSOPLOWERING_H_
