//===- PtrToLLVM.cpp - Ptr to LLVM dialect conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PtrToLLVM/PtrToLLVM.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/TypeUtilities.h"
#include <type_traits>

using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// FromPtrOpConversion
//===----------------------------------------------------------------------===//
struct FromPtrOpConversion : public ConvertOpToLLVMPattern<ptr::FromPtrOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ptr::FromPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// GetMetadataOpConversion
//===----------------------------------------------------------------------===//
struct GetMetadataOpConversion
    : public ConvertOpToLLVMPattern<ptr::GetMetadataOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ptr::GetMetadataOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// PtrAddOpConversion
//===----------------------------------------------------------------------===//
struct PtrAddOpConversion : public ConvertOpToLLVMPattern<ptr::PtrAddOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ptr::PtrAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// ToPtrOpConversion
//===----------------------------------------------------------------------===//
struct ToPtrOpConversion : public ConvertOpToLLVMPattern<ptr::ToPtrOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ptr::ToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// TypeOffsetOpConversion
//===----------------------------------------------------------------------===//
struct TypeOffsetOpConversion
    : public ConvertOpToLLVMPattern<ptr::TypeOffsetOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(ptr::TypeOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Internal functions
//===----------------------------------------------------------------------===//

// Function to create an LLVM struct type representing a memref metadata.
static FailureOr<LLVM::LLVMStructType>
createMemRefMetadataType(MemRefType type,
                         const LLVMTypeConverter &typeConverter) {
  MLIRContext *context = type.getContext();
  // Get the address space.
  FailureOr<unsigned> addressSpace = typeConverter.getMemRefAddressSpace(type);
  if (failed(addressSpace))
    return failure();

  // Get pointer type (using address space 0 by default)
  auto ptrType = LLVM::LLVMPointerType::get(context, *addressSpace);

  // Get the strides offsets and shape.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return failure();
  ArrayRef<int64_t> shape = type.getShape();

  // Use index type from the type converter for the descriptor elements
  Type indexType = typeConverter.getIndexType();

  // For a ranked memref, the descriptor contains:
  // 1. The pointer to the allocated data
  // 2. The pointer to the aligned data
  // 3. The dynamic offset?
  // 4. The dynamic sizes?
  // 5. The dynamic strides?
  SmallVector<Type, 5> elements;

  // Allocated pointer.
  elements.push_back(ptrType);

  // Potentially add the dynamic offset.
  if (offset == ShapedType::kDynamic)
    elements.push_back(indexType);

  // Potentially add the dynamic sizes.
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic)
      elements.push_back(indexType);
  }

  // Potentially add the dynamic strides.
  for (int64_t stride : strides) {
    if (stride == ShapedType::kDynamic)
      elements.push_back(indexType);
  }
  return LLVM::LLVMStructType::getLiteral(context, elements);
}

//===----------------------------------------------------------------------===//
// FromPtrOpConversion
//===----------------------------------------------------------------------===//

LogicalResult FromPtrOpConversion::matchAndRewrite(
    ptr::FromPtrOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Get the target memref type
  auto mTy = dyn_cast<MemRefType>(op.getResult().getType());
  if (!mTy)
    return rewriter.notifyMatchFailure(op, "Expected memref result type");

  if (!op.getMetadata() && op.getType().hasPtrMetadata()) {
    return rewriter.notifyMatchFailure(
        op, "Can convert only memrefs with metadata");
  }

  // Convert the result type
  Type descriptorTy = getTypeConverter()->convertType(mTy);
  if (!descriptorTy)
    return rewriter.notifyMatchFailure(op, "Failed to convert result type");

  // Get the strides, offsets and shape.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(mTy.getStridesAndOffset(strides, offset))) {
    return rewriter.notifyMatchFailure(op,
                                       "Failed to get the strides and offset");
  }
  ArrayRef<int64_t> shape = mTy.getShape();

  // Create a new memref descriptor
  Location loc = op.getLoc();
  auto desc = MemRefDescriptor::poison(rewriter, loc, descriptorTy);

  // Set the allocated and aligned pointers.
  desc.setAllocatedPtr(
      rewriter, loc,
      LLVM::ExtractValueOp::create(rewriter, loc, adaptor.getMetadata(), 0));
  desc.setAlignedPtr(rewriter, loc, adaptor.getPtr());

  // Extract metadata from the passed struct.
  unsigned fieldIdx = 1;

  // Set dynamic offset if needed.
  if (offset == ShapedType::kDynamic) {
    Value offsetValue = LLVM::ExtractValueOp::create(
        rewriter, loc, adaptor.getMetadata(), fieldIdx++);
    desc.setOffset(rewriter, loc, offsetValue);
  } else {
    desc.setConstantOffset(rewriter, loc, offset);
  }

  // Set dynamic sizes if needed.
  for (auto [i, dim] : llvm::enumerate(shape)) {
    if (dim == ShapedType::kDynamic) {
      Value sizeValue = LLVM::ExtractValueOp::create(
          rewriter, loc, adaptor.getMetadata(), fieldIdx++);
      desc.setSize(rewriter, loc, i, sizeValue);
    } else {
      desc.setConstantSize(rewriter, loc, i, dim);
    }
  }

  // Set dynamic strides if needed.
  for (auto [i, stride] : llvm::enumerate(strides)) {
    if (stride == ShapedType::kDynamic) {
      Value strideValue = LLVM::ExtractValueOp::create(
          rewriter, loc, adaptor.getMetadata(), fieldIdx++);
      desc.setStride(rewriter, loc, i, strideValue);
    } else {
      desc.setConstantStride(rewriter, loc, i, stride);
    }
  }

  rewriter.replaceOp(op, static_cast<Value>(desc));
  return success();
}

//===----------------------------------------------------------------------===//
// GetMetadataOpConversion
//===----------------------------------------------------------------------===//

LogicalResult GetMetadataOpConversion::matchAndRewrite(
    ptr::GetMetadataOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto mTy = dyn_cast<MemRefType>(op.getPtr().getType());
  if (!mTy)
    return rewriter.notifyMatchFailure(op, "Only memref metadata is supported");

  // Get the metadata type.
  FailureOr<LLVM::LLVMStructType> mdTy =
      createMemRefMetadataType(mTy, *getTypeConverter());
  if (failed(mdTy)) {
    return rewriter.notifyMatchFailure(op,
                                       "Failed to create the metadata type");
  }

  // Get the memref descriptor.
  MemRefDescriptor descriptor(adaptor.getPtr());

  // Get the strides offsets and shape.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(mTy.getStridesAndOffset(strides, offset))) {
    return rewriter.notifyMatchFailure(op,
                                       "Failed to get the strides and offset");
  }
  ArrayRef<int64_t> shape = mTy.getShape();

  // Create a new LLVM struct to hold the metadata
  Location loc = op.getLoc();
  Value sV = LLVM::UndefOp::create(rewriter, loc, *mdTy);

  // First element is the allocated pointer.
  sV = LLVM::InsertValueOp::create(
      rewriter, loc, sV, descriptor.allocatedPtr(rewriter, loc), int64_t{0});

  // Track the current field index.
  unsigned fieldIdx = 1;

  // Add dynamic offset if needed.
  if (offset == ShapedType::kDynamic) {
    sV = LLVM::InsertValueOp::create(
        rewriter, loc, sV, descriptor.offset(rewriter, loc), fieldIdx++);
  }

  // Add dynamic sizes if needed.
  for (auto [i, dim] : llvm::enumerate(shape)) {
    if (dim != ShapedType::kDynamic)
      continue;
    sV = LLVM::InsertValueOp::create(
        rewriter, loc, sV, descriptor.size(rewriter, loc, i), fieldIdx++);
  }

  // Add dynamic strides if needed
  for (auto [i, stride] : llvm::enumerate(strides)) {
    if (stride != ShapedType::kDynamic)
      continue;
    sV = LLVM::InsertValueOp::create(
        rewriter, loc, sV, descriptor.stride(rewriter, loc, i), fieldIdx++);
  }
  rewriter.replaceOp(op, sV);
  return success();
}

//===----------------------------------------------------------------------===//
// PtrAddOpConversion
//===----------------------------------------------------------------------===//

LogicalResult
PtrAddOpConversion::matchAndRewrite(ptr::PtrAddOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  // Get and check the base.
  Value base = adaptor.getBase();
  if (!isa<LLVM::LLVMPointerType>(base.getType()))
    return rewriter.notifyMatchFailure(op, "Incompatible pointer type");

  // Get the offset.
  Value offset = adaptor.getOffset();

  // Ptr assumes the offset is in bytes.
  Type elementType = IntegerType::get(rewriter.getContext(), 8);

  // Convert the `ptradd` flags.
  LLVM::GEPNoWrapFlags flags;
  switch (op.getFlags()) {
  case ptr::PtrAddFlags::none:
    flags = LLVM::GEPNoWrapFlags::none;
    break;
  case ptr::PtrAddFlags::nusw:
    flags = LLVM::GEPNoWrapFlags::nusw;
    break;
  case ptr::PtrAddFlags::nuw:
    flags = LLVM::GEPNoWrapFlags::nuw;
    break;
  case ptr::PtrAddFlags::inbounds:
    flags = LLVM::GEPNoWrapFlags::inbounds;
    break;
  }

  // Create the GEP operation with appropriate arguments
  rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, base.getType(), elementType,
                                           base, ValueRange{offset}, flags);
  return success();
}

//===----------------------------------------------------------------------===//
// ToPtrOpConversion
//===----------------------------------------------------------------------===//

LogicalResult
ToPtrOpConversion::matchAndRewrite(ptr::ToPtrOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  // Bail if it's not a memref.
  if (!isa<MemRefType>(op.getPtr().getType()))
    return rewriter.notifyMatchFailure(op, "Expected a memref input");

  // Extract the aligned pointer from the memref descriptor.
  rewriter.replaceOp(
      op, MemRefDescriptor(adaptor.getPtr()).alignedPtr(rewriter, op.getLoc()));
  return success();
}

//===----------------------------------------------------------------------===//
// TypeOffsetOpConversion
//===----------------------------------------------------------------------===//

LogicalResult TypeOffsetOpConversion::matchAndRewrite(
    ptr::TypeOffsetOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Convert the type attribute.
  Type type = getTypeConverter()->convertType(op.getElementType());
  if (!type)
    return rewriter.notifyMatchFailure(op, "Couldn't convert the type");

  // Convert the result type.
  Type rTy = getTypeConverter()->convertType(op.getResult().getType());
  if (!rTy)
    return rewriter.notifyMatchFailure(op, "Couldn't convert the result type");

  // TODO: Use MLIR's data layout. We don't use it because overall support is
  // still flaky.

  // Create an LLVM pointer type for the GEP operation.
  auto ptrTy = LLVM::LLVMPointerType::get(getContext());

  // Create a GEP operation to compute the offset of the type.
  auto offset =
      LLVM::GEPOp::create(rewriter, op.getLoc(), ptrTy, type,
                          LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrTy),
                          ArrayRef<LLVM::GEPArg>({LLVM::GEPArg(1)}));

  // Replace the original op with a PtrToIntOp using the computed offset.
  rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, rTy, offset.getRes());
  return success();
}

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

namespace {
/// Implement the interface to convert Ptr to LLVM.
struct PtrToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &converter,
      RewritePatternSet &patterns) const final {
    ptr::populatePtrToLLVMConversionPatterns(converter, patterns);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

void mlir::ptr::populatePtrToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // Add address space conversions.
  converter.addTypeAttributeConversion(
      [&](PtrLikeTypeInterface type, ptr::GenericSpaceAttr memorySpace)
          -> TypeConverter::AttributeConversionResult {
        if (type.getMemorySpace() != memorySpace)
          return TypeConverter::AttributeConversionResult::na();
        return IntegerAttr::get(IntegerType::get(type.getContext(), 32), 0);
      });

  // Add type conversions.
  converter.addConversion([&](ptr::PtrType type) -> Type {
    std::optional<Attribute> maybeAttr =
        converter.convertTypeAttribute(type, type.getMemorySpace());
    auto memSpace =
        maybeAttr ? dyn_cast_or_null<IntegerAttr>(*maybeAttr) : IntegerAttr();
    if (!memSpace)
      return {};
    return LLVM::LLVMPointerType::get(type.getContext(),
                                      memSpace.getValue().getSExtValue());
  });

  // Convert ptr metadata of memref type.
  converter.addConversion([&](ptr::PtrMetadataType type) -> Type {
    auto mTy = dyn_cast<MemRefType>(type.getType());
    if (!mTy)
      return {};
    FailureOr<LLVM::LLVMStructType> res =
        createMemRefMetadataType(mTy, converter);
    return failed(res) ? Type() : res.value();
  });

  // Add conversion patterns.
  patterns.add<FromPtrOpConversion, GetMetadataOpConversion, PtrAddOpConversion,
               ToPtrOpConversion, TypeOffsetOpConversion>(converter);
}

void mlir::ptr::registerConvertPtrToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ptr::PtrDialect *dialect) {
    dialect->addInterfaces<PtrToLLVMDialectInterface>();
  });
}
