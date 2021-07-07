//===- ConvertStandardToLLVM.h - Convert to the LLVM dialect ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a dialect conversion targeting the LLVM IR dialect.  By default, it
// converts Standard ops and types and provides hooks for dialect-specific
// extensions to the conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVM_H
#define MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVM_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;

/// Collect a set of patterns to convert memory-related operations from the
/// Standard dialect to the LLVM dialect, excluding non-memory-related
/// operations and FuncOp.
void populateStdToLLVMMemoryConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns);

/// Collect a set of patterns to convert from the Standard dialect to the LLVM
/// dialect, excluding the memory-related operations.
void populateStdToLLVMNonMemoryConversionPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns);

/// Collect the default pattern to convert a FuncOp to the LLVM dialect. If
/// `emitCWrappers` is set, the pattern will also produce functions
/// that pass memref descriptors by pointer-to-structure in addition to the
/// default unpacked form.
void populateStdToLLVMFuncOpConversionPattern(LLVMTypeConverter &converter,
                                              RewritePatternSet &patterns);

/// Collect the patterns to convert from the Standard dialect to LLVM. The
/// conversion patterns capture the LLVMTypeConverter and the LowerToLLVMOptions
/// by reference meaning the references have to remain alive during the entire
/// pattern lifetime.
void populateStdToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

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

/// Derived class that automatically populates legalization information for
/// different LLVM ops.
class LLVMConversionTarget : public ConversionTarget {
public:
  explicit LLVMConversionTarget(MLIRContext &ctx);
};

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVM_H
