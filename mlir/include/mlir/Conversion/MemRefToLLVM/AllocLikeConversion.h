//===- AllocLikeConversion.h - Convert allocation ops to LLVM ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MEMREFTOLLVM_ALLOCLIKECONVERSION_H
#define MLIR_CONVERSION_MEMREFTOLLVM_ALLOCLIKECONVERSION_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"

namespace mlir {

/// Lowering for AllocOp and AllocaOp.
struct AllocLikeOpLLVMLowering : public ConvertToLLVMPattern {
  using ConvertToLLVMPattern::createIndexConstant;
  using ConvertToLLVMPattern::getIndexType;
  using ConvertToLLVMPattern::getVoidPtrType;

  explicit AllocLikeOpLLVMLowering(StringRef opName,
                                   LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(opName, &converter.getContext(), converter) {}

protected:
  // Returns 'input' aligned up to 'alignment'. Computes
  // bumped = input + alignement - 1
  // aligned = bumped - bumped % alignment
  static Value createAligned(ConversionPatternRewriter &rewriter, Location loc,
                             Value input, Value alignment);

  /// Allocates the underlying buffer. Returns the allocated pointer and the
  /// aligned pointer.
  virtual std::tuple<Value, Value>
  allocateBuffer(ConversionPatternRewriter &rewriter, Location loc,
                 Value sizeBytes, Operation *op) const = 0;

private:
  static MemRefType getMemRefResultType(Operation *op) {
    return op->getResult(0).getType().cast<MemRefType>();
  }

  // An `alloc` is converted into a definition of a memref descriptor value and
  // a call to `malloc` to allocate the underlying data buffer.  The memref
  // descriptor is of the LLVM structure type where:
  //   1. the first element is a pointer to the allocated (typed) data buffer,
  //   2. the second element is a pointer to the (typed) payload, aligned to the
  //      specified alignment,
  //   3. the remaining elements serve to store all the sizes and strides of the
  //      memref using LLVM-converted `index` type.
  //
  // Alignment is performed by allocating `alignment` more bytes than
  // requested and shifting the aligned pointer relative to the allocated
  // memory. Note: `alignment - <minimum malloc alignment>` would actually be
  // sufficient. If alignment is unspecified, the two pointers are equal.

  // An `alloca` is converted into a definition of a memref descriptor value and
  // an llvm.alloca to allocate the underlying data buffer.
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace mlir

#endif // MLIR_CONVERSION_MEMREFTOLLVM_ALLOCLIKECONVERSION_H
