//===- Pattern.h - Pattern for conversion to the LLVM dialect ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LLVMCOMMON_PATTERN_H
#define MLIR_CONVERSION_LLVMCOMMON_PATTERN_H

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class CallOpInterface;

namespace LLVM {
namespace detail {
/// Replaces the given operation "op" with a new operation of type "targetOp"
/// and given operands.
LogicalResult oneToOneRewrite(Operation *op, StringRef targetOp,
                              ValueRange operands,
                              LLVMTypeConverter &typeConverter,
                              ConversionPatternRewriter &rewriter);
} // namespace detail
} // namespace LLVM

/// Base class for operation conversions targeting the LLVM IR dialect. It
/// provides the conversion patterns with access to the LLVMTypeConverter and
/// the LowerToLLVMOptions. The class captures the LLVMTypeConverter and the
/// LowerToLLVMOptions by reference meaning the references have to remain alive
/// during the entire pattern lifetime.
class ConvertToLLVMPattern : public ConversionPattern {
public:
  ConvertToLLVMPattern(StringRef rootOpName, MLIRContext *context,
                       LLVMTypeConverter &typeConverter,
                       PatternBenefit benefit = 1);

protected:
  /// Returns the LLVM dialect.
  LLVM::LLVMDialect &getDialect() const;

  LLVMTypeConverter *getTypeConverter() const;

  /// Gets the MLIR type wrapping the LLVM integer type whose bit width is
  /// defined by the used type converter.
  Type getIndexType() const;

  /// Gets the MLIR type wrapping the LLVM integer type whose bit width
  /// corresponds to that of a LLVM pointer type.
  Type getIntPtrType(unsigned addressSpace = 0) const;

  /// Gets the MLIR type wrapping the LLVM void type.
  Type getVoidType() const;

  /// Get the MLIR type wrapping the LLVM i8* type.
  Type getVoidPtrType() const;

  /// Create a constant Op producing a value of `resultType` from an index-typed
  /// integer attribute.
  static Value createIndexAttrConstant(OpBuilder &builder, Location loc,
                                       Type resultType, int64_t value);

  /// Create an LLVM dialect operation defining the given index constant.
  Value createIndexConstant(ConversionPatternRewriter &builder, Location loc,
                            uint64_t value) const;

  // This is a strided getElementPtr variant that linearizes subscripts as:
  //   `base_offset + index_0 * stride_0 + ... + index_n * stride_n`.
  Value getStridedElementPtr(Location loc, MemRefType type, Value memRefDesc,
                             ValueRange indices,
                             ConversionPatternRewriter &rewriter) const;

  /// Returns if the given memref has identity maps and the element type is
  /// convertible to LLVM.
  bool isConvertibleAndHasIdentityMaps(MemRefType type) const;

  /// Returns the type of a pointer to an element of the memref.
  Type getElementPtrType(MemRefType type) const;

  /// Computes sizes, strides and buffer size in bytes of `memRefType` with
  /// identity layout. Emits constant ops for the static sizes of `memRefType`,
  /// and uses `dynamicSizes` for the others. Emits instructions to compute
  /// strides and buffer size from these sizes.
  ///
  /// For example, memref<4x?xf32> emits:
  /// `sizes[0]`   = llvm.mlir.constant(4 : index) : i64
  /// `sizes[1]`   = `dynamicSizes[0]`
  /// `strides[1]` = llvm.mlir.constant(1 : index) : i64
  /// `strides[0]` = `sizes[0]`
  /// %size        = llvm.mul `sizes[0]`, `sizes[1]` : i64
  /// %nullptr     = llvm.mlir.null : !llvm.ptr<f32>
  /// %gep         = llvm.getelementptr %nullptr[%size]
  ///                  : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  /// `sizeBytes`  = llvm.ptrtoint %gep : !llvm.ptr<f32> to i64
  void getMemRefDescriptorSizes(Location loc, MemRefType memRefType,
                                ValueRange dynamicSizes,
                                ConversionPatternRewriter &rewriter,
                                SmallVectorImpl<Value> &sizes,
                                SmallVectorImpl<Value> &strides,
                                Value &sizeBytes) const;

  /// Computes the size of type in bytes.
  Value getSizeInBytes(Location loc, Type type,
                       ConversionPatternRewriter &rewriter) const;

  /// Computes total number of elements for the given shape.
  Value getNumElements(Location loc, ArrayRef<Value> shape,
                       ConversionPatternRewriter &rewriter) const;

  /// Creates and populates a canonical memref descriptor struct.
  MemRefDescriptor
  createMemRefDescriptor(Location loc, MemRefType memRefType,
                         Value allocatedPtr, Value alignedPtr,
                         ArrayRef<Value> sizes, ArrayRef<Value> strides,
                         ConversionPatternRewriter &rewriter) const;

  /// Copies the memory descriptor for any operands that were unranked
  /// descriptors originally to heap-allocated memory (if toDynamic is true) or
  /// to stack-allocated memory (otherwise). Also frees the previously used
  /// memory (that is assumed to be heap-allocated) if toDynamic is false.
  LogicalResult copyUnrankedDescriptors(OpBuilder &builder, Location loc,
                                        TypeRange origTypes,
                                        SmallVectorImpl<Value> &operands,
                                        bool toDynamic) const;
};

/// Utility class for operation conversions targeting the LLVM dialect that
/// match exactly one source operation.
template <typename SourceOp>
class ConvertOpToLLVMPattern : public ConvertToLLVMPattern {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ConvertOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                  PatternBenefit benefit = 1)
      : ConvertToLLVMPattern(SourceOp::getOperationName(),
                             &typeConverter.getContext(), typeConverter,
                             benefit) {}

  /// Wrappers around the RewritePattern methods that pass the derived op type.
  void rewrite(Operation *op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const final {
    rewrite(cast<SourceOp>(op), OpAdaptor(operands, op->getAttrDictionary()),
            rewriter);
  }
  LogicalResult match(Operation *op) const final {
    return match(cast<SourceOp>(op));
  }
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    return matchAndRewrite(cast<SourceOp>(op),
                           OpAdaptor(operands, op->getAttrDictionary()),
                           rewriter);
  }

  /// Rewrite and Match methods that operate on the SourceOp type. These must be
  /// overridden by the derived pattern class.
  virtual LogicalResult match(SourceOp op) const {
    llvm_unreachable("must override match or matchAndRewrite");
  }
  virtual void rewrite(SourceOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("must override rewrite or matchAndRewrite");
  }
  virtual LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    if (failed(match(op)))
      return failure();
    rewrite(op, adaptor, rewriter);
    return success();
  }

private:
  using ConvertToLLVMPattern::match;
  using ConvertToLLVMPattern::matchAndRewrite;
};

/// Generic implementation of one-to-one conversion from "SourceOp" to
/// "TargetOp" where the latter belongs to the LLVM dialect or an equivalent.
/// Upholds a convention that multi-result operations get converted into an
/// operation returning the LLVM IR structure type, in which case individual
/// values must be extracted from using LLVM::ExtractValueOp before being used.
template <typename SourceOp, typename TargetOp>
class OneToOneConvertToLLVMPattern : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using Super = OneToOneConvertToLLVMPattern<SourceOp, TargetOp>;

  /// Converts the type of the result to an LLVM type, pass operands as is,
  /// preserve attributes.
  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return LLVM::detail::oneToOneRewrite(op, TargetOp::getOperationName(),
                                         adaptor.getOperands(),
                                         *this->getTypeConverter(), rewriter);
  }
};

} // namespace mlir

#endif // MLIR_CONVERSION_LLVMCOMMON_PATTERN_H
