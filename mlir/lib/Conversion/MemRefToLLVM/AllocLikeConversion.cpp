//===- AllocLikeConversion.cpp - LLVM conversion for alloc operations -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToLLVM/AllocLikeConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

Value AllocLikeOpLLVMLowering::createAligned(
    ConversionPatternRewriter &rewriter, Location loc, Value input,
    Value alignment) {
  Value one = createIndexAttrConstant(rewriter, loc, alignment.getType(), 1);
  Value bump = rewriter.create<LLVM::SubOp>(loc, alignment, one);
  Value bumped = rewriter.create<LLVM::AddOp>(loc, input, bump);
  Value mod = rewriter.create<LLVM::URemOp>(loc, bumped, alignment);
  return rewriter.create<LLVM::SubOp>(loc, bumped, mod);
}

LogicalResult AllocLikeOpLLVMLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  MemRefType memRefType = getMemRefResultType(op);
  if (!isConvertibleAndHasIdentityMaps(memRefType))
    return rewriter.notifyMatchFailure(op, "incompatible memref type");
  auto loc = op->getLoc();

  // Get actual sizes of the memref as values: static sizes are constant
  // values and dynamic sizes are passed to 'alloc' as operands.  In case of
  // zero-dimensional memref, assume a scalar (size 1).
  SmallVector<Value, 4> sizes;
  SmallVector<Value, 4> strides;
  Value sizeBytes;
  this->getMemRefDescriptorSizes(loc, memRefType, operands, rewriter, sizes,
                                 strides, sizeBytes);

  // Allocate the underlying buffer.
  Value allocatedPtr;
  Value alignedPtr;
  std::tie(allocatedPtr, alignedPtr) =
      this->allocateBuffer(rewriter, loc, sizeBytes, op);

  // Create the MemRef descriptor.
  auto memRefDescriptor = this->createMemRefDescriptor(
      loc, memRefType, allocatedPtr, alignedPtr, sizes, strides, rewriter);

  // Return the final value of the descriptor.
  rewriter.replaceOp(op, {memRefDescriptor});
  return success();
}
