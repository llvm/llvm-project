//===- SPIRVToLLVM.cpp - SPIR-V to LLVM Patterns --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert SPIR-V dialect to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SPIRVCommon/AttrToLLVMConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Utils/LayoutUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "spirv-to-llvm-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Returns true if the given type is a signed integer or vector type.
static bool isSignedIntegerOrVector(Type type) {
  if (type.isSignedInteger())
    return true;
  if (auto vecType = dyn_cast<VectorType>(type))
    return vecType.getElementType().isSignedInteger();
  return false;
}

/// Returns true if the given type is an unsigned integer or vector type
static bool isUnsignedIntegerOrVector(Type type) {
  if (type.isUnsignedInteger())
    return true;
  if (auto vecType = dyn_cast<VectorType>(type))
    return vecType.getElementType().isUnsignedInteger();
  return false;
}

/// Returns the width of an integer or of the element type of an integer vector,
/// if applicable.
static std::optional<uint64_t> getIntegerOrVectorElementWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (auto vecType = dyn_cast<VectorType>(type))
    if (auto intType = dyn_cast<IntegerType>(vecType.getElementType()))
      return intType.getWidth();
  return std::nullopt;
}

/// Returns the bit width of integer, float or vector of float or integer values
static unsigned getBitWidth(Type type) {
  assert((type.isIntOrFloat() || isa<VectorType>(type)) &&
         "bitwidth is not supported for this type");
  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth();
  auto vecType = dyn_cast<VectorType>(type);
  auto elementType = vecType.getElementType();
  assert(elementType.isIntOrFloat() &&
         "only integers and floats have a bitwidth");
  return elementType.getIntOrFloatBitWidth();
}

/// Returns the bit width of LLVMType integer or vector.
static unsigned getLLVMTypeBitWidth(Type type) {
  return cast<IntegerType>((LLVM::isCompatibleVectorType(type)
                                ? LLVM::getVectorElementType(type)
                                : type))
      .getWidth();
}

/// Creates `IntegerAttribute` with all bits set for given type
static IntegerAttr minusOneIntegerAttribute(Type type, Builder builder) {
  if (auto vecType = dyn_cast<VectorType>(type)) {
    auto integerType = cast<IntegerType>(vecType.getElementType());
    return builder.getIntegerAttr(integerType, -1);
  }
  auto integerType = cast<IntegerType>(type);
  return builder.getIntegerAttr(integerType, -1);
}

/// Creates `llvm.mlir.constant` with all bits set for the given type.
static Value createConstantAllBitsSet(Location loc, Type srcType, Type dstType,
                                      PatternRewriter &rewriter) {
  if (isa<VectorType>(srcType)) {
    return rewriter.create<LLVM::ConstantOp>(
        loc, dstType,
        SplatElementsAttr::get(cast<ShapedType>(srcType),
                               minusOneIntegerAttribute(srcType, rewriter)));
  }
  return rewriter.create<LLVM::ConstantOp>(
      loc, dstType, minusOneIntegerAttribute(srcType, rewriter));
}

/// Creates `llvm.mlir.constant` with a floating-point scalar or vector value.
static Value createFPConstant(Location loc, Type srcType, Type dstType,
                              PatternRewriter &rewriter, double value) {
  if (auto vecType = dyn_cast<VectorType>(srcType)) {
    auto floatType = cast<FloatType>(vecType.getElementType());
    return rewriter.create<LLVM::ConstantOp>(
        loc, dstType,
        SplatElementsAttr::get(vecType,
                               rewriter.getFloatAttr(floatType, value)));
  }
  auto floatType = cast<FloatType>(srcType);
  return rewriter.create<LLVM::ConstantOp>(
      loc, dstType, rewriter.getFloatAttr(floatType, value));
}

/// Utility function for bitfield ops:
///   - `BitFieldInsert`
///   - `BitFieldSExtract`
///   - `BitFieldUExtract`
/// Truncates or extends the value. If the bitwidth of the value is the same as
/// `llvmType` bitwidth, the value remains unchanged.
static Value optionallyTruncateOrExtend(Location loc, Value value,
                                        Type llvmType,
                                        PatternRewriter &rewriter) {
  auto srcType = value.getType();
  unsigned targetBitWidth = getLLVMTypeBitWidth(llvmType);
  unsigned valueBitWidth = LLVM::isCompatibleType(srcType)
                               ? getLLVMTypeBitWidth(srcType)
                               : getBitWidth(srcType);

  if (valueBitWidth < targetBitWidth)
    return rewriter.create<LLVM::ZExtOp>(loc, llvmType, value);
  // If the bit widths of `Count` and `Offset` are greater than the bit width
  // of the target type, they are truncated. Truncation is safe since `Count`
  // and `Offset` must be no more than 64 for op behaviour to be defined. Hence,
  // both values can be expressed in 8 bits.
  if (valueBitWidth > targetBitWidth)
    return rewriter.create<LLVM::TruncOp>(loc, llvmType, value);
  return value;
}

/// Broadcasts the value to vector with `numElements` number of elements.
static Value broadcast(Location loc, Value toBroadcast, unsigned numElements,
                       const TypeConverter &typeConverter,
                       ConversionPatternRewriter &rewriter) {
  auto vectorType = VectorType::get(numElements, toBroadcast.getType());
  auto llvmVectorType = typeConverter.convertType(vectorType);
  auto llvmI32Type = typeConverter.convertType(rewriter.getIntegerType(32));
  Value broadcasted = rewriter.create<LLVM::UndefOp>(loc, llvmVectorType);
  for (unsigned i = 0; i < numElements; ++i) {
    auto index = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type, rewriter.getI32IntegerAttr(i));
    broadcasted = rewriter.create<LLVM::InsertElementOp>(
        loc, llvmVectorType, broadcasted, toBroadcast, index);
  }
  return broadcasted;
}

/// Broadcasts the value. If `srcType` is a scalar, the value remains unchanged.
static Value optionallyBroadcast(Location loc, Value value, Type srcType,
                                 const TypeConverter &typeConverter,
                                 ConversionPatternRewriter &rewriter) {
  if (auto vectorType = dyn_cast<VectorType>(srcType)) {
    unsigned numElements = vectorType.getNumElements();
    return broadcast(loc, value, numElements, typeConverter, rewriter);
  }
  return value;
}

/// Utility function for bitfield ops: `BitFieldInsert`, `BitFieldSExtract` and
/// `BitFieldUExtract`.
/// Broadcast `Offset` and `Count` to match the type of `Base`. If `Base` is of
/// a vector type, construct a vector that has:
///  - same number of elements as `Base`
///  - each element has the type that is the same as the type of `Offset` or
///    `Count`
///  - each element has the same value as `Offset` or `Count`
/// Then cast `Offset` and `Count` if their bit width is different
/// from `Base` bit width.
static Value processCountOrOffset(Location loc, Value value, Type srcType,
                                  Type dstType, const TypeConverter &converter,
                                  ConversionPatternRewriter &rewriter) {
  Value broadcasted =
      optionallyBroadcast(loc, value, srcType, converter, rewriter);
  return optionallyTruncateOrExtend(loc, broadcasted, dstType, rewriter);
}

/// Converts SPIR-V struct with a regular (according to `VulkanLayoutUtils`)
/// offset to LLVM struct. Otherwise, the conversion is not supported.
static Type convertStructTypeWithOffset(spirv::StructType type,
                                        const TypeConverter &converter) {
  if (type != VulkanLayoutUtils::decorateType(type))
    return nullptr;

  SmallVector<Type> elementsVector;
  if (failed(converter.convertTypes(type.getElementTypes(), elementsVector)))
    return nullptr;
  return LLVM::LLVMStructType::getLiteral(type.getContext(), elementsVector,
                                          /*isPacked=*/false);
}

/// Converts SPIR-V struct with no offset to packed LLVM struct.
static Type convertStructTypePacked(spirv::StructType type,
                                    const TypeConverter &converter) {
  SmallVector<Type> elementsVector;
  if (failed(converter.convertTypes(type.getElementTypes(), elementsVector)))
    return nullptr;
  return LLVM::LLVMStructType::getLiteral(type.getContext(), elementsVector,
                                          /*isPacked=*/true);
}

/// Creates LLVM dialect constant with the given value.
static Value createI32ConstantOf(Location loc, PatternRewriter &rewriter,
                                 unsigned value) {
  return rewriter.create<LLVM::ConstantOp>(
      loc, IntegerType::get(rewriter.getContext(), 32),
      rewriter.getIntegerAttr(rewriter.getI32Type(), value));
}

/// Utility for `spirv.Load` and `spirv.Store` conversion.
static LogicalResult replaceWithLoadOrStore(Operation *op, ValueRange operands,
                                            ConversionPatternRewriter &rewriter,
                                            const TypeConverter &typeConverter,
                                            unsigned alignment, bool isVolatile,
                                            bool isNonTemporal) {
  if (auto loadOp = dyn_cast<spirv::LoadOp>(op)) {
    auto dstType = typeConverter.convertType(loadOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        loadOp, dstType, spirv::LoadOpAdaptor(operands).getPtr(), alignment,
        isVolatile, isNonTemporal);
    return success();
  }
  auto storeOp = cast<spirv::StoreOp>(op);
  spirv::StoreOpAdaptor adaptor(operands);
  rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeOp, adaptor.getValue(),
                                             adaptor.getPtr(), alignment,
                                             isVolatile, isNonTemporal);
  return success();
}

//===----------------------------------------------------------------------===//
// Type conversion
//===----------------------------------------------------------------------===//

/// Converts SPIR-V array type to LLVM array. Natural stride (according to
/// `VulkanLayoutUtils`) is also mapped to LLVM array. This has to be respected
/// when converting ops that manipulate array types.
static std::optional<Type> convertArrayType(spirv::ArrayType type,
                                            TypeConverter &converter) {
  unsigned stride = type.getArrayStride();
  Type elementType = type.getElementType();
  auto sizeInBytes = cast<spirv::SPIRVType>(elementType).getSizeInBytes();
  if (stride != 0 && (!sizeInBytes || *sizeInBytes != stride))
    return std::nullopt;

  auto llvmElementType = converter.convertType(elementType);
  unsigned numElements = type.getNumElements();
  return LLVM::LLVMArrayType::get(llvmElementType, numElements);
}

/// Converts SPIR-V pointer type to LLVM pointer. Pointer's storage class is not
/// modelled at the moment.
static Type convertPointerType(spirv::PointerType type,
                               const TypeConverter &converter,
                               spirv::ClientAPI clientAPI) {
  unsigned addressSpace =
      storageClassToAddressSpace(clientAPI, type.getStorageClass());
  return LLVM::LLVMPointerType::get(type.getContext(), addressSpace);
}

/// Converts SPIR-V runtime array to LLVM array. Since LLVM allows indexing over
/// the bounds, the runtime array is converted to a 0-sized LLVM array. There is
/// no modelling of array stride at the moment.
static std::optional<Type> convertRuntimeArrayType(spirv::RuntimeArrayType type,
                                                   TypeConverter &converter) {
  if (type.getArrayStride() != 0)
    return std::nullopt;
  auto elementType = converter.convertType(type.getElementType());
  return LLVM::LLVMArrayType::get(elementType, 0);
}

/// Converts SPIR-V struct to LLVM struct. There is no support of structs with
/// member decorations. Also, only natural offset is supported.
static Type convertStructType(spirv::StructType type,
                              const TypeConverter &converter) {
  SmallVector<spirv::StructType::MemberDecorationInfo, 4> memberDecorations;
  type.getMemberDecorations(memberDecorations);
  if (!memberDecorations.empty())
    return nullptr;
  if (type.hasOffset())
    return convertStructTypeWithOffset(type, converter);
  return convertStructTypePacked(type, converter);
}

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

namespace {

class AccessChainPattern : public SPIRVToLLVMConversion<spirv::AccessChainOp> {
public:
  using SPIRVToLLVMConversion<spirv::AccessChainOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::AccessChainOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType =
        getTypeConverter()->convertType(op.getComponentPtr().getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    // To use GEP we need to add a first 0 index to go through the pointer.
    auto indices = llvm::to_vector<4>(adaptor.getIndices());
    Type indexType = op.getIndices().front().getType();
    auto llvmIndexType = getTypeConverter()->convertType(indexType);
    if (!llvmIndexType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), llvmIndexType, rewriter.getIntegerAttr(indexType, 0));
    indices.insert(indices.begin(), zero);

    auto elementType = getTypeConverter()->convertType(
        cast<spirv::PointerType>(op.getBasePtr().getType()).getPointeeType());
    if (!elementType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, dstType, elementType,
                                             adaptor.getBasePtr(), indices);
    return success();
  }
};

class AddressOfPattern : public SPIRVToLLVMConversion<spirv::AddressOfOp> {
public:
  using SPIRVToLLVMConversion<spirv::AddressOfOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::AddressOfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = getTypeConverter()->convertType(op.getPointer().getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(op, dstType,
                                                   op.getVariable());
    return success();
  }
};

class BitFieldInsertPattern
    : public SPIRVToLLVMConversion<spirv::BitFieldInsertOp> {
public:
  using SPIRVToLLVMConversion<spirv::BitFieldInsertOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::BitFieldInsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = op.getType();
    auto dstType = getTypeConverter()->convertType(srcType);
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    Location loc = op.getLoc();

    // Process `Offset` and `Count`: broadcast and extend/truncate if needed.
    Value offset = processCountOrOffset(loc, op.getOffset(), srcType, dstType,
                                        *getTypeConverter(), rewriter);
    Value count = processCountOrOffset(loc, op.getCount(), srcType, dstType,
                                       *getTypeConverter(), rewriter);

    // Create a mask with bits set outside [Offset, Offset + Count - 1].
    Value minusOne = createConstantAllBitsSet(loc, srcType, dstType, rewriter);
    Value maskShiftedByCount =
        rewriter.create<LLVM::ShlOp>(loc, dstType, minusOne, count);
    Value negated = rewriter.create<LLVM::XOrOp>(loc, dstType,
                                                 maskShiftedByCount, minusOne);
    Value maskShiftedByCountAndOffset =
        rewriter.create<LLVM::ShlOp>(loc, dstType, negated, offset);
    Value mask = rewriter.create<LLVM::XOrOp>(
        loc, dstType, maskShiftedByCountAndOffset, minusOne);

    // Extract unchanged bits from the `Base`  that are outside of
    // [Offset, Offset + Count - 1]. Then `or` with shifted `Insert`.
    Value baseAndMask =
        rewriter.create<LLVM::AndOp>(loc, dstType, op.getBase(), mask);
    Value insertShiftedByOffset =
        rewriter.create<LLVM::ShlOp>(loc, dstType, op.getInsert(), offset);
    rewriter.replaceOpWithNewOp<LLVM::OrOp>(op, dstType, baseAndMask,
                                            insertShiftedByOffset);
    return success();
  }
};

/// Converts SPIR-V ConstantOp with scalar or vector type.
class ConstantScalarAndVectorPattern
    : public SPIRVToLLVMConversion<spirv::ConstantOp> {
public:
  using SPIRVToLLVMConversion<spirv::ConstantOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = constOp.getType();
    if (!isa<VectorType>(srcType) && !srcType.isIntOrFloat())
      return failure();

    auto dstType = getTypeConverter()->convertType(srcType);
    if (!dstType)
      return rewriter.notifyMatchFailure(constOp, "type conversion failed");

    // SPIR-V constant can be a signed/unsigned integer, which has to be
    // casted to signless integer when converting to LLVM dialect. Removing the
    // sign bit may have unexpected behaviour. However, it is better to handle
    // it case-by-case, given that the purpose of the conversion is not to
    // cover all possible corner cases.
    if (isSignedIntegerOrVector(srcType) ||
        isUnsignedIntegerOrVector(srcType)) {
      auto signlessType = rewriter.getIntegerType(getBitWidth(srcType));

      if (isa<VectorType>(srcType)) {
        auto dstElementsAttr = cast<DenseIntElementsAttr>(constOp.getValue());
        rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
            constOp, dstType,
            dstElementsAttr.mapValues(
                signlessType, [&](const APInt &value) { return value; }));
        return success();
      }
      auto srcAttr = cast<IntegerAttr>(constOp.getValue());
      auto dstAttr = rewriter.getIntegerAttr(signlessType, srcAttr.getValue());
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(constOp, dstType, dstAttr);
      return success();
    }
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        constOp, dstType, adaptor.getOperands(), constOp->getAttrs());
    return success();
  }
};

class BitFieldSExtractPattern
    : public SPIRVToLLVMConversion<spirv::BitFieldSExtractOp> {
public:
  using SPIRVToLLVMConversion<spirv::BitFieldSExtractOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::BitFieldSExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = op.getType();
    auto dstType = getTypeConverter()->convertType(srcType);
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    Location loc = op.getLoc();

    // Process `Offset` and `Count`: broadcast and extend/truncate if needed.
    Value offset = processCountOrOffset(loc, op.getOffset(), srcType, dstType,
                                        *getTypeConverter(), rewriter);
    Value count = processCountOrOffset(loc, op.getCount(), srcType, dstType,
                                       *getTypeConverter(), rewriter);

    // Create a constant that holds the size of the `Base`.
    IntegerType integerType;
    if (auto vecType = dyn_cast<VectorType>(srcType))
      integerType = cast<IntegerType>(vecType.getElementType());
    else
      integerType = cast<IntegerType>(srcType);

    auto baseSize = rewriter.getIntegerAttr(integerType, getBitWidth(srcType));
    Value size =
        isa<VectorType>(srcType)
            ? rewriter.create<LLVM::ConstantOp>(
                  loc, dstType,
                  SplatElementsAttr::get(cast<ShapedType>(srcType), baseSize))
            : rewriter.create<LLVM::ConstantOp>(loc, dstType, baseSize);

    // Shift `Base` left by [sizeof(Base) - (Count + Offset)], so that the bit
    // at Offset + Count - 1 is the most significant bit now.
    Value countPlusOffset =
        rewriter.create<LLVM::AddOp>(loc, dstType, count, offset);
    Value amountToShiftLeft =
        rewriter.create<LLVM::SubOp>(loc, dstType, size, countPlusOffset);
    Value baseShiftedLeft = rewriter.create<LLVM::ShlOp>(
        loc, dstType, op.getBase(), amountToShiftLeft);

    // Shift the result right, filling the bits with the sign bit.
    Value amountToShiftRight =
        rewriter.create<LLVM::AddOp>(loc, dstType, offset, amountToShiftLeft);
    rewriter.replaceOpWithNewOp<LLVM::AShrOp>(op, dstType, baseShiftedLeft,
                                              amountToShiftRight);
    return success();
  }
};

class BitFieldUExtractPattern
    : public SPIRVToLLVMConversion<spirv::BitFieldUExtractOp> {
public:
  using SPIRVToLLVMConversion<spirv::BitFieldUExtractOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::BitFieldUExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = op.getType();
    auto dstType = getTypeConverter()->convertType(srcType);
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    Location loc = op.getLoc();

    // Process `Offset` and `Count`: broadcast and extend/truncate if needed.
    Value offset = processCountOrOffset(loc, op.getOffset(), srcType, dstType,
                                        *getTypeConverter(), rewriter);
    Value count = processCountOrOffset(loc, op.getCount(), srcType, dstType,
                                       *getTypeConverter(), rewriter);

    // Create a mask with bits set at [0, Count - 1].
    Value minusOne = createConstantAllBitsSet(loc, srcType, dstType, rewriter);
    Value maskShiftedByCount =
        rewriter.create<LLVM::ShlOp>(loc, dstType, minusOne, count);
    Value mask = rewriter.create<LLVM::XOrOp>(loc, dstType, maskShiftedByCount,
                                              minusOne);

    // Shift `Base` by `Offset` and apply the mask on it.
    Value shiftedBase =
        rewriter.create<LLVM::LShrOp>(loc, dstType, op.getBase(), offset);
    rewriter.replaceOpWithNewOp<LLVM::AndOp>(op, dstType, shiftedBase, mask);
    return success();
  }
};

class BranchConversionPattern : public SPIRVToLLVMConversion<spirv::BranchOp> {
public:
  using SPIRVToLLVMConversion<spirv::BranchOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::BranchOp branchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::BrOp>(branchOp, adaptor.getOperands(),
                                            branchOp.getTarget());
    return success();
  }
};

class BranchConditionalConversionPattern
    : public SPIRVToLLVMConversion<spirv::BranchConditionalOp> {
public:
  using SPIRVToLLVMConversion<
      spirv::BranchConditionalOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::BranchConditionalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If branch weights exist, map them to 32-bit integer vector.
    DenseI32ArrayAttr branchWeights = nullptr;
    if (auto weights = op.getBranchWeights()) {
      SmallVector<int32_t> weightValues;
      for (auto weight : weights->getAsRange<IntegerAttr>())
        weightValues.push_back(weight.getInt());
      branchWeights = DenseI32ArrayAttr::get(getContext(), weightValues);
    }

    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
        op, op.getCondition(), op.getTrueBlockArguments(),
        op.getFalseBlockArguments(), branchWeights, op.getTrueBlock(),
        op.getFalseBlock());
    return success();
  }
};

/// Converts `spirv.getCompositeExtract` to `llvm.extractvalue` if the container
/// type is an aggregate type (struct or array). Otherwise, converts to
/// `llvm.extractelement` that operates on vectors.
class CompositeExtractPattern
    : public SPIRVToLLVMConversion<spirv::CompositeExtractOp> {
public:
  using SPIRVToLLVMConversion<spirv::CompositeExtractOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::CompositeExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = this->getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    Type containerType = op.getComposite().getType();
    if (isa<VectorType>(containerType)) {
      Location loc = op.getLoc();
      IntegerAttr value = cast<IntegerAttr>(op.getIndices()[0]);
      Value index = createI32ConstantOf(loc, rewriter, value.getInt());
      rewriter.replaceOpWithNewOp<LLVM::ExtractElementOp>(
          op, dstType, adaptor.getComposite(), index);
      return success();
    }

    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        op, adaptor.getComposite(),
        LLVM::convertArrayToIndices(op.getIndices()));
    return success();
  }
};

/// Converts `spirv.getCompositeInsert` to `llvm.insertvalue` if the container
/// type is an aggregate type (struct or array). Otherwise, converts to
/// `llvm.insertelement` that operates on vectors.
class CompositeInsertPattern
    : public SPIRVToLLVMConversion<spirv::CompositeInsertOp> {
public:
  using SPIRVToLLVMConversion<spirv::CompositeInsertOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::CompositeInsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = this->getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    Type containerType = op.getComposite().getType();
    if (isa<VectorType>(containerType)) {
      Location loc = op.getLoc();
      IntegerAttr value = cast<IntegerAttr>(op.getIndices()[0]);
      Value index = createI32ConstantOf(loc, rewriter, value.getInt());
      rewriter.replaceOpWithNewOp<LLVM::InsertElementOp>(
          op, dstType, adaptor.getComposite(), adaptor.getObject(), index);
      return success();
    }

    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, adaptor.getComposite(), adaptor.getObject(),
        LLVM::convertArrayToIndices(op.getIndices()));
    return success();
  }
};

/// Converts SPIR-V operations that have straightforward LLVM equivalent
/// into LLVM dialect operations.
template <typename SPIRVOp, typename LLVMOp>
class DirectConversionPattern : public SPIRVToLLVMConversion<SPIRVOp> {
public:
  using SPIRVToLLVMConversion<SPIRVOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(SPIRVOp op, typename SPIRVOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = this->getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    rewriter.template replaceOpWithNewOp<LLVMOp>(
        op, dstType, adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

/// Converts `spirv.ExecutionMode` into a global struct constant that holds
/// execution mode information.
class ExecutionModePattern
    : public SPIRVToLLVMConversion<spirv::ExecutionModeOp> {
public:
  using SPIRVToLLVMConversion<spirv::ExecutionModeOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::ExecutionModeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // First, create the global struct's name that would be associated with
    // this entry point's execution mode. We set it to be:
    //   __spv__{SPIR-V module name}_{function name}_execution_mode_info_{mode}
    ModuleOp module = op->getParentOfType<ModuleOp>();
    spirv::ExecutionModeAttr executionModeAttr = op.getExecutionModeAttr();
    std::string moduleName;
    if (module.getName().has_value())
      moduleName = "_" + module.getName()->str();
    else
      moduleName = "";
    std::string executionModeInfoName = llvm::formatv(
        "__spv_{0}_{1}_execution_mode_info_{2}", moduleName, op.getFn().str(),
        static_cast<uint32_t>(executionModeAttr.getValue()));

    MLIRContext *context = rewriter.getContext();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());

    // Create a struct type, corresponding to the C struct below.
    // struct {
    //   int32_t executionMode;
    //   int32_t values[];          // optional values
    // };
    auto llvmI32Type = IntegerType::get(context, 32);
    SmallVector<Type, 2> fields;
    fields.push_back(llvmI32Type);
    ArrayAttr values = op.getValues();
    if (!values.empty()) {
      auto arrayType = LLVM::LLVMArrayType::get(llvmI32Type, values.size());
      fields.push_back(arrayType);
    }
    auto structType = LLVM::LLVMStructType::getLiteral(context, fields);

    // Create `llvm.mlir.global` with initializer region containing one block.
    auto global = rewriter.create<LLVM::GlobalOp>(
        UnknownLoc::get(context), structType, /*isConstant=*/true,
        LLVM::Linkage::External, executionModeInfoName, Attribute(),
        /*alignment=*/0);
    Location loc = global.getLoc();
    Region &region = global.getInitializerRegion();
    Block *block = rewriter.createBlock(&region);

    // Initialize the struct and set the execution mode value.
    rewriter.setInsertionPointToStart(block);
    Value structValue = rewriter.create<LLVM::UndefOp>(loc, structType);
    Value executionMode = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type,
        rewriter.getI32IntegerAttr(
            static_cast<uint32_t>(executionModeAttr.getValue())));
    structValue = rewriter.create<LLVM::InsertValueOp>(loc, structValue,
                                                       executionMode, 0);

    // Insert extra operands if they exist into execution mode info struct.
    for (unsigned i = 0, e = values.size(); i < e; ++i) {
      auto attr = values.getValue()[i];
      Value entry = rewriter.create<LLVM::ConstantOp>(loc, llvmI32Type, attr);
      structValue = rewriter.create<LLVM::InsertValueOp>(
          loc, structValue, entry, ArrayRef<int64_t>({1, i}));
    }
    rewriter.create<LLVM::ReturnOp>(loc, ArrayRef<Value>({structValue}));
    rewriter.eraseOp(op);
    return success();
  }
};

/// Converts `spirv.GlobalVariable` to `llvm.mlir.global`. Note that SPIR-V
/// global returns a pointer, whereas in LLVM dialect the global holds an actual
/// value. This difference is handled by `spirv.mlir.addressof` and
/// `llvm.mlir.addressof`ops that both return a pointer.
class GlobalVariablePattern
    : public SPIRVToLLVMConversion<spirv::GlobalVariableOp> {
public:
  template <typename... Args>
  GlobalVariablePattern(spirv::ClientAPI clientAPI, Args &&...args)
      : SPIRVToLLVMConversion<spirv::GlobalVariableOp>(
            std::forward<Args>(args)...),
        clientAPI(clientAPI) {}

  LogicalResult
  matchAndRewrite(spirv::GlobalVariableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Currently, there is no support of initialization with a constant value in
    // SPIR-V dialect. Specialization constants are not considered as well.
    if (op.getInitializer())
      return failure();

    auto srcType = cast<spirv::PointerType>(op.getType());
    auto dstType = getTypeConverter()->convertType(srcType.getPointeeType());
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    // Limit conversion to the current invocation only or `StorageBuffer`
    // required by SPIR-V runner.
    // This is okay because multiple invocations are not supported yet.
    auto storageClass = srcType.getStorageClass();
    switch (storageClass) {
    case spirv::StorageClass::Input:
    case spirv::StorageClass::Private:
    case spirv::StorageClass::Output:
    case spirv::StorageClass::StorageBuffer:
    case spirv::StorageClass::UniformConstant:
      break;
    default:
      return failure();
    }

    // LLVM dialect spec: "If the global value is a constant, storing into it is
    // not allowed.". This corresponds to SPIR-V 'Input' and 'UniformConstant'
    // storage class that is read-only.
    bool isConstant = (storageClass == spirv::StorageClass::Input) ||
                      (storageClass == spirv::StorageClass::UniformConstant);
    // SPIR-V spec: "By default, functions and global variables are private to a
    // module and cannot be accessed by other modules. However, a module may be
    // written to export or import functions and global (module scope)
    // variables.". Therefore, map 'Private' storage class to private linkage,
    // 'Input' and 'Output' to external linkage.
    auto linkage = storageClass == spirv::StorageClass::Private
                       ? LLVM::Linkage::Private
                       : LLVM::Linkage::External;
    auto newGlobalOp = rewriter.replaceOpWithNewOp<LLVM::GlobalOp>(
        op, dstType, isConstant, linkage, op.getSymName(), Attribute(),
        /*alignment=*/0, storageClassToAddressSpace(clientAPI, storageClass));

    // Attach location attribute if applicable
    if (op.getLocationAttr())
      newGlobalOp->setAttr(op.getLocationAttrName(), op.getLocationAttr());

    return success();
  }

private:
  spirv::ClientAPI clientAPI;
};

/// Converts SPIR-V cast ops that do not have straightforward LLVM
/// equivalent in LLVM dialect.
template <typename SPIRVOp, typename LLVMExtOp, typename LLVMTruncOp>
class IndirectCastPattern : public SPIRVToLLVMConversion<SPIRVOp> {
public:
  using SPIRVToLLVMConversion<SPIRVOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(SPIRVOp op, typename SPIRVOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type fromType = op.getOperand().getType();
    Type toType = op.getType();

    auto dstType = this->getTypeConverter()->convertType(toType);
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    if (getBitWidth(fromType) < getBitWidth(toType)) {
      rewriter.template replaceOpWithNewOp<LLVMExtOp>(op, dstType,
                                                      adaptor.getOperands());
      return success();
    }
    if (getBitWidth(fromType) > getBitWidth(toType)) {
      rewriter.template replaceOpWithNewOp<LLVMTruncOp>(op, dstType,
                                                        adaptor.getOperands());
      return success();
    }
    return failure();
  }
};

class FunctionCallPattern
    : public SPIRVToLLVMConversion<spirv::FunctionCallOp> {
public:
  using SPIRVToLLVMConversion<spirv::FunctionCallOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::FunctionCallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (callOp.getNumResults() == 0) {
      auto newOp = rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          callOp, std::nullopt, adaptor.getOperands(), callOp->getAttrs());
      newOp.getProperties().operandSegmentSizes = {
          static_cast<int32_t>(adaptor.getOperands().size()), 0};
      newOp.getProperties().op_bundle_sizes = rewriter.getDenseI32ArrayAttr({});
      return success();
    }

    // Function returns a single result.
    auto dstType = getTypeConverter()->convertType(callOp.getType(0));
    if (!dstType)
      return rewriter.notifyMatchFailure(callOp, "type conversion failed");
    auto newOp = rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        callOp, dstType, adaptor.getOperands(), callOp->getAttrs());
    newOp.getProperties().operandSegmentSizes = {
        static_cast<int32_t>(adaptor.getOperands().size()), 0};
    newOp.getProperties().op_bundle_sizes = rewriter.getDenseI32ArrayAttr({});
    return success();
  }
};

/// Converts SPIR-V floating-point comparisons to llvm.fcmp "predicate"
template <typename SPIRVOp, LLVM::FCmpPredicate predicate>
class FComparePattern : public SPIRVToLLVMConversion<SPIRVOp> {
public:
  using SPIRVToLLVMConversion<SPIRVOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(SPIRVOp op, typename SPIRVOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto dstType = this->getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    rewriter.template replaceOpWithNewOp<LLVM::FCmpOp>(
        op, dstType, predicate, op.getOperand1(), op.getOperand2());
    return success();
  }
};

/// Converts SPIR-V integer comparisons to llvm.icmp "predicate"
template <typename SPIRVOp, LLVM::ICmpPredicate predicate>
class IComparePattern : public SPIRVToLLVMConversion<SPIRVOp> {
public:
  using SPIRVToLLVMConversion<SPIRVOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(SPIRVOp op, typename SPIRVOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto dstType = this->getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    rewriter.template replaceOpWithNewOp<LLVM::ICmpOp>(
        op, dstType, predicate, op.getOperand1(), op.getOperand2());
    return success();
  }
};

class InverseSqrtPattern
    : public SPIRVToLLVMConversion<spirv::GLInverseSqrtOp> {
public:
  using SPIRVToLLVMConversion<spirv::GLInverseSqrtOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::GLInverseSqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = op.getType();
    auto dstType = getTypeConverter()->convertType(srcType);
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    Location loc = op.getLoc();
    Value one = createFPConstant(loc, srcType, dstType, rewriter, 1.0);
    Value sqrt = rewriter.create<LLVM::SqrtOp>(loc, dstType, op.getOperand());
    rewriter.replaceOpWithNewOp<LLVM::FDivOp>(op, dstType, one, sqrt);
    return success();
  }
};

/// Converts `spirv.Load` and `spirv.Store` to LLVM dialect.
template <typename SPIRVOp>
class LoadStorePattern : public SPIRVToLLVMConversion<SPIRVOp> {
public:
  using SPIRVToLLVMConversion<SPIRVOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(SPIRVOp op, typename SPIRVOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getMemoryAccess()) {
      return replaceWithLoadOrStore(op, adaptor.getOperands(), rewriter,
                                    *this->getTypeConverter(), /*alignment=*/0,
                                    /*isVolatile=*/false,
                                    /*isNonTemporal=*/false);
    }
    auto memoryAccess = *op.getMemoryAccess();
    switch (memoryAccess) {
    case spirv::MemoryAccess::Aligned:
    case spirv::MemoryAccess::None:
    case spirv::MemoryAccess::Nontemporal:
    case spirv::MemoryAccess::Volatile: {
      unsigned alignment =
          memoryAccess == spirv::MemoryAccess::Aligned ? *op.getAlignment() : 0;
      bool isNonTemporal = memoryAccess == spirv::MemoryAccess::Nontemporal;
      bool isVolatile = memoryAccess == spirv::MemoryAccess::Volatile;
      return replaceWithLoadOrStore(op, adaptor.getOperands(), rewriter,
                                    *this->getTypeConverter(), alignment,
                                    isVolatile, isNonTemporal);
    }
    default:
      // There is no support of other memory access attributes.
      return failure();
    }
  }
};

/// Converts `spirv.Not` and `spirv.LogicalNot` into LLVM dialect.
template <typename SPIRVOp>
class NotPattern : public SPIRVToLLVMConversion<SPIRVOp> {
public:
  using SPIRVToLLVMConversion<SPIRVOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(SPIRVOp notOp, typename SPIRVOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = notOp.getType();
    auto dstType = this->getTypeConverter()->convertType(srcType);
    if (!dstType)
      return rewriter.notifyMatchFailure(notOp, "type conversion failed");

    Location loc = notOp.getLoc();
    IntegerAttr minusOne = minusOneIntegerAttribute(srcType, rewriter);
    auto mask =
        isa<VectorType>(srcType)
            ? rewriter.create<LLVM::ConstantOp>(
                  loc, dstType,
                  SplatElementsAttr::get(cast<VectorType>(srcType), minusOne))
            : rewriter.create<LLVM::ConstantOp>(loc, dstType, minusOne);
    rewriter.template replaceOpWithNewOp<LLVM::XOrOp>(notOp, dstType,
                                                      notOp.getOperand(), mask);
    return success();
  }
};

/// A template pattern that erases the given `SPIRVOp`.
template <typename SPIRVOp>
class ErasePattern : public SPIRVToLLVMConversion<SPIRVOp> {
public:
  using SPIRVToLLVMConversion<SPIRVOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(SPIRVOp op, typename SPIRVOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class ReturnPattern : public SPIRVToLLVMConversion<spirv::ReturnOp> {
public:
  using SPIRVToLLVMConversion<spirv::ReturnOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(returnOp, ArrayRef<Type>(),
                                                ArrayRef<Value>());
    return success();
  }
};

class ReturnValuePattern : public SPIRVToLLVMConversion<spirv::ReturnValueOp> {
public:
  using SPIRVToLLVMConversion<spirv::ReturnValueOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::ReturnValueOp returnValueOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(returnValueOp, ArrayRef<Type>(),
                                                adaptor.getOperands());
    return success();
  }
};

static LLVM::LLVMFuncOp lookupOrCreateSPIRVFn(Operation *symbolTable,
                                              StringRef name,
                                              ArrayRef<Type> paramTypes,
                                              Type resultType,
                                              bool convergent = true) {
  auto func = dyn_cast_or_null<LLVM::LLVMFuncOp>(
      SymbolTable::lookupSymbolIn(symbolTable, name));
  if (func)
    return func;

  OpBuilder b(symbolTable->getRegion(0));
  func = b.create<LLVM::LLVMFuncOp>(
      symbolTable->getLoc(), name,
      LLVM::LLVMFunctionType::get(resultType, paramTypes));
  func.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
  func.setConvergent(convergent);
  func.setNoUnwind(true);
  func.setWillReturn(true);
  return func;
}

static LLVM::CallOp createSPIRVBuiltinCall(Location loc, OpBuilder &builder,
                                           LLVM::LLVMFuncOp func,
                                           ValueRange args) {
  auto call = builder.create<LLVM::CallOp>(loc, func, args);
  call.setCConv(func.getCConv());
  call.setConvergentAttr(func.getConvergentAttr());
  call.setNoUnwindAttr(func.getNoUnwindAttr());
  call.setWillReturnAttr(func.getWillReturnAttr());
  return call;
}

template <typename BarrierOpTy>
class ControlBarrierPattern : public SPIRVToLLVMConversion<BarrierOpTy> {
public:
  using OpAdaptor = typename SPIRVToLLVMConversion<BarrierOpTy>::OpAdaptor;

  using SPIRVToLLVMConversion<BarrierOpTy>::SPIRVToLLVMConversion;

  static constexpr StringRef getFuncName();

  LogicalResult
  matchAndRewrite(BarrierOpTy controlBarrierOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    constexpr StringRef funcName = getFuncName();
    Operation *symbolTable =
        controlBarrierOp->template getParentWithTrait<OpTrait::SymbolTable>();

    Type i32 = rewriter.getI32Type();

    Type voidTy = rewriter.getType<LLVM::LLVMVoidType>();
    LLVM::LLVMFuncOp func =
        lookupOrCreateSPIRVFn(symbolTable, funcName, {i32, i32, i32}, voidTy);

    Location loc = controlBarrierOp->getLoc();
    Value execution = rewriter.create<LLVM::ConstantOp>(
        loc, i32, static_cast<int32_t>(adaptor.getExecutionScope()));
    Value memory = rewriter.create<LLVM::ConstantOp>(
        loc, i32, static_cast<int32_t>(adaptor.getMemoryScope()));
    Value semantics = rewriter.create<LLVM::ConstantOp>(
        loc, i32, static_cast<int32_t>(adaptor.getMemorySemantics()));

    auto call = createSPIRVBuiltinCall(loc, rewriter, func,
                                       {execution, memory, semantics});

    rewriter.replaceOp(controlBarrierOp, call);
    return success();
  }
};

namespace {

StringRef getTypeMangling(Type type, bool isSigned) {
  return llvm::TypeSwitch<Type, StringRef>(type)
      .Case<Float16Type>([](auto) { return "Dh"; })
      .Case<Float32Type>([](auto) { return "f"; })
      .Case<Float64Type>([](auto) { return "d"; })
      .Case<IntegerType>([isSigned](IntegerType intTy) {
        switch (intTy.getWidth()) {
        case 1:
          return "b";
        case 8:
          return (isSigned) ? "a" : "c";
        case 16:
          return (isSigned) ? "s" : "t";
        case 32:
          return (isSigned) ? "i" : "j";
        case 64:
          return (isSigned) ? "l" : "m";
        default:
          llvm_unreachable("Unsupported integer width");
        }
      })
      .Default([](auto) {
        llvm_unreachable("No mangling defined");
        return "";
      });
}

template <typename ReduceOp>
constexpr StringLiteral getGroupFuncName();

template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupIAddOp>() {
  return "_Z17__spirv_GroupIAddii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupFAddOp>() {
  return "_Z17__spirv_GroupFAddii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupSMinOp>() {
  return "_Z17__spirv_GroupSMinii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupUMinOp>() {
  return "_Z17__spirv_GroupUMinii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupFMinOp>() {
  return "_Z17__spirv_GroupFMinii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupSMaxOp>() {
  return "_Z17__spirv_GroupSMaxii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupUMaxOp>() {
  return "_Z17__spirv_GroupUMaxii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupFMaxOp>() {
  return "_Z17__spirv_GroupFMaxii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformIAddOp>() {
  return "_Z27__spirv_GroupNonUniformIAddii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformFAddOp>() {
  return "_Z27__spirv_GroupNonUniformFAddii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformIMulOp>() {
  return "_Z27__spirv_GroupNonUniformIMulii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformFMulOp>() {
  return "_Z27__spirv_GroupNonUniformFMulii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformSMinOp>() {
  return "_Z27__spirv_GroupNonUniformSMinii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformUMinOp>() {
  return "_Z27__spirv_GroupNonUniformUMinii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformFMinOp>() {
  return "_Z27__spirv_GroupNonUniformFMinii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformSMaxOp>() {
  return "_Z27__spirv_GroupNonUniformSMaxii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformUMaxOp>() {
  return "_Z27__spirv_GroupNonUniformUMaxii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformFMaxOp>() {
  return "_Z27__spirv_GroupNonUniformFMaxii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformBitwiseAndOp>() {
  return "_Z33__spirv_GroupNonUniformBitwiseAndii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformBitwiseOrOp>() {
  return "_Z32__spirv_GroupNonUniformBitwiseOrii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformBitwiseXorOp>() {
  return "_Z33__spirv_GroupNonUniformBitwiseXorii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformLogicalAndOp>() {
  return "_Z33__spirv_GroupNonUniformLogicalAndii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformLogicalOrOp>() {
  return "_Z32__spirv_GroupNonUniformLogicalOrii";
}
template <>
constexpr StringLiteral getGroupFuncName<spirv::GroupNonUniformLogicalXorOp>() {
  return "_Z33__spirv_GroupNonUniformLogicalXorii";
}
} // namespace

template <typename ReduceOp, bool Signed = false, bool NonUniform = false>
class GroupReducePattern : public SPIRVToLLVMConversion<ReduceOp> {
public:
  using SPIRVToLLVMConversion<ReduceOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(ReduceOp op, typename ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type retTy = op.getResult().getType();
    if (!retTy.isIntOrFloat()) {
      return failure();
    }
    SmallString<36> funcName = getGroupFuncName<ReduceOp>();
    funcName += getTypeMangling(retTy, false);

    Type i32Ty = rewriter.getI32Type();
    SmallVector<Type> paramTypes{i32Ty, i32Ty, retTy};
    if constexpr (NonUniform) {
      if (adaptor.getClusterSize()) {
        funcName += "j";
        paramTypes.push_back(i32Ty);
      }
    }

    Operation *symbolTable =
        op->template getParentWithTrait<OpTrait::SymbolTable>();

    LLVM::LLVMFuncOp func = lookupOrCreateSPIRVFn(
        symbolTable, funcName, paramTypes, retTy, !NonUniform);

    Location loc = op.getLoc();
    Value scope = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, static_cast<int32_t>(adaptor.getExecutionScope()));
    Value groupOp = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, static_cast<int32_t>(adaptor.getGroupOperation()));
    SmallVector<Value> operands{scope, groupOp};
    operands.append(adaptor.getOperands().begin(), adaptor.getOperands().end());

    auto call = createSPIRVBuiltinCall(loc, rewriter, func, operands);
    rewriter.replaceOp(op, call);
    return success();
  }
};

template <>
constexpr StringRef
ControlBarrierPattern<spirv::ControlBarrierOp>::getFuncName() {
  return "_Z22__spirv_ControlBarrieriii";
}

template <>
constexpr StringRef
ControlBarrierPattern<spirv::INTELControlBarrierArriveOp>::getFuncName() {
  return "_Z33__spirv_ControlBarrierArriveINTELiii";
}

template <>
constexpr StringRef
ControlBarrierPattern<spirv::INTELControlBarrierWaitOp>::getFuncName() {
  return "_Z31__spirv_ControlBarrierWaitINTELiii";
}

/// Converts `spirv.mlir.loop` to LLVM dialect. All blocks within selection
/// should be reachable for conversion to succeed. The structure of the loop in
/// LLVM dialect will be the following:
///
///      +------------------------------------+
///      | <code before spirv.mlir.loop>        |
///      | llvm.br ^header                    |
///      +------------------------------------+
///                           |
///   +----------------+      |
///   |                |      |
///   |                V      V
///   |  +------------------------------------+
///   |  | ^header:                           |
///   |  |   <header code>                    |
///   |  |   llvm.cond_br %cond, ^body, ^exit |
///   |  +------------------------------------+
///   |                    |
///   |                    |----------------------+
///   |                    |                      |
///   |                    V                      |
///   |  +------------------------------------+   |
///   |  | ^body:                             |   |
///   |  |   <body code>                      |   |
///   |  |   llvm.br ^continue                |   |
///   |  +------------------------------------+   |
///   |                    |                      |
///   |                    V                      |
///   |  +------------------------------------+   |
///   |  | ^continue:                         |   |
///   |  |   <continue code>                  |   |
///   |  |   llvm.br ^header                  |   |
///   |  +------------------------------------+   |
///   |               |                           |
///   +---------------+    +----------------------+
///                        |
///                        V
///      +------------------------------------+
///      | ^exit:                             |
///      |   llvm.br ^remaining               |
///      +------------------------------------+
///                        |
///                        V
///      +------------------------------------+
///      | ^remaining:                        |
///      |   <code after spirv.mlir.loop>       |
///      +------------------------------------+
///
class LoopPattern : public SPIRVToLLVMConversion<spirv::LoopOp> {
public:
  using SPIRVToLLVMConversion<spirv::LoopOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::LoopOp loopOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // There is no support of loop control at the moment.
    if (loopOp.getLoopControl() != spirv::LoopControl::None)
      return failure();

    // `spirv.mlir.loop` with empty region is redundant and should be erased.
    if (loopOp.getBody().empty()) {
      rewriter.eraseOp(loopOp);
      return success();
    }

    Location loc = loopOp.getLoc();

    // Split the current block after `spirv.mlir.loop`. The remaining ops will
    // be used in `endBlock`.
    Block *currentBlock = rewriter.getBlock();
    auto position = Block::iterator(loopOp);
    Block *endBlock = rewriter.splitBlock(currentBlock, position);

    // Remove entry block and create a branch in the current block going to the
    // header block.
    Block *entryBlock = loopOp.getEntryBlock();
    assert(entryBlock->getOperations().size() == 1);
    auto brOp = dyn_cast<spirv::BranchOp>(entryBlock->getOperations().front());
    if (!brOp)
      return failure();
    Block *headerBlock = loopOp.getHeaderBlock();
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::BrOp>(loc, brOp.getBlockArguments(), headerBlock);
    rewriter.eraseBlock(entryBlock);

    // Branch from merge block to end block.
    Block *mergeBlock = loopOp.getMergeBlock();
    Operation *terminator = mergeBlock->getTerminator();
    ValueRange terminatorOperands = terminator->getOperands();
    rewriter.setInsertionPointToEnd(mergeBlock);
    rewriter.create<LLVM::BrOp>(loc, terminatorOperands, endBlock);

    rewriter.inlineRegionBefore(loopOp.getBody(), endBlock);
    rewriter.replaceOp(loopOp, endBlock->getArguments());
    return success();
  }
};

/// Converts `spirv.mlir.selection` with `spirv.BranchConditional` in its header
/// block. All blocks within selection should be reachable for conversion to
/// succeed.
class SelectionPattern : public SPIRVToLLVMConversion<spirv::SelectionOp> {
public:
  using SPIRVToLLVMConversion<spirv::SelectionOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::SelectionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // There is no support for `Flatten` or `DontFlatten` selection control at
    // the moment. This are just compiler hints and can be performed during the
    // optimization passes.
    if (op.getSelectionControl() != spirv::SelectionControl::None)
      return failure();

    // `spirv.mlir.selection` should have at least two blocks: one selection
    // header block and one merge block. If no blocks are present, or control
    // flow branches straight to merge block (two blocks are present), the op is
    // redundant and it is erased.
    if (op.getBody().getBlocks().size() <= 2) {
      rewriter.eraseOp(op);
      return success();
    }

    Location loc = op.getLoc();

    // Split the current block after `spirv.mlir.selection`. The remaining ops
    // will be used in `continueBlock`.
    auto *currentBlock = rewriter.getInsertionBlock();
    rewriter.setInsertionPointAfter(op);
    auto position = rewriter.getInsertionPoint();
    auto *continueBlock = rewriter.splitBlock(currentBlock, position);

    // Extract conditional branch information from the header block. By SPIR-V
    // dialect spec, it should contain `spirv.BranchConditional` or
    // `spirv.Switch` op. Note that `spirv.Switch op` is not supported at the
    // moment in the SPIR-V dialect. Remove this block when finished.
    auto *headerBlock = op.getHeaderBlock();
    assert(headerBlock->getOperations().size() == 1);
    auto condBrOp = dyn_cast<spirv::BranchConditionalOp>(
        headerBlock->getOperations().front());
    if (!condBrOp)
      return failure();
    rewriter.eraseBlock(headerBlock);

    // Branch from merge block to continue block.
    auto *mergeBlock = op.getMergeBlock();
    Operation *terminator = mergeBlock->getTerminator();
    ValueRange terminatorOperands = terminator->getOperands();
    rewriter.setInsertionPointToEnd(mergeBlock);
    rewriter.create<LLVM::BrOp>(loc, terminatorOperands, continueBlock);

    // Link current block to `true` and `false` blocks within the selection.
    Block *trueBlock = condBrOp.getTrueBlock();
    Block *falseBlock = condBrOp.getFalseBlock();
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, condBrOp.getCondition(), trueBlock,
                                    condBrOp.getTrueTargetOperands(),
                                    falseBlock,
                                    condBrOp.getFalseTargetOperands());

    rewriter.inlineRegionBefore(op.getBody(), continueBlock);
    rewriter.replaceOp(op, continueBlock->getArguments());
    return success();
  }
};

/// Converts SPIR-V shift ops to LLVM shift ops. Since LLVM dialect
/// puts a restriction on `Shift` and `Base` to have the same bit width,
/// `Shift` is zero or sign extended to match this specification. Cases when
/// `Shift` bit width > `Base` bit width are considered to be illegal.
template <typename SPIRVOp, typename LLVMOp>
class ShiftPattern : public SPIRVToLLVMConversion<SPIRVOp> {
public:
  using SPIRVToLLVMConversion<SPIRVOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(SPIRVOp op, typename SPIRVOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto dstType = this->getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    Type op1Type = op.getOperand1().getType();
    Type op2Type = op.getOperand2().getType();

    if (op1Type == op2Type) {
      rewriter.template replaceOpWithNewOp<LLVMOp>(op, dstType,
                                                   adaptor.getOperands());
      return success();
    }

    std::optional<uint64_t> dstTypeWidth =
        getIntegerOrVectorElementWidth(dstType);
    std::optional<uint64_t> op2TypeWidth =
        getIntegerOrVectorElementWidth(op2Type);

    if (!dstTypeWidth || !op2TypeWidth)
      return failure();

    Location loc = op.getLoc();
    Value extended;
    if (op2TypeWidth < dstTypeWidth) {
      if (isUnsignedIntegerOrVector(op2Type)) {
        extended = rewriter.template create<LLVM::ZExtOp>(
            loc, dstType, adaptor.getOperand2());
      } else {
        extended = rewriter.template create<LLVM::SExtOp>(
            loc, dstType, adaptor.getOperand2());
      }
    } else if (op2TypeWidth == dstTypeWidth) {
      extended = adaptor.getOperand2();
    } else {
      return failure();
    }

    Value result = rewriter.template create<LLVMOp>(
        loc, dstType, adaptor.getOperand1(), extended);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class TanPattern : public SPIRVToLLVMConversion<spirv::GLTanOp> {
public:
  using SPIRVToLLVMConversion<spirv::GLTanOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::GLTanOp tanOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = getTypeConverter()->convertType(tanOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(tanOp, "type conversion failed");

    Location loc = tanOp.getLoc();
    Value sin = rewriter.create<LLVM::SinOp>(loc, dstType, tanOp.getOperand());
    Value cos = rewriter.create<LLVM::CosOp>(loc, dstType, tanOp.getOperand());
    rewriter.replaceOpWithNewOp<LLVM::FDivOp>(tanOp, dstType, sin, cos);
    return success();
  }
};

/// Convert `spirv.Tanh` to
///
///   exp(2x) - 1
///   -----------
///   exp(2x) + 1
///
class TanhPattern : public SPIRVToLLVMConversion<spirv::GLTanhOp> {
public:
  using SPIRVToLLVMConversion<spirv::GLTanhOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::GLTanhOp tanhOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = tanhOp.getType();
    auto dstType = getTypeConverter()->convertType(srcType);
    if (!dstType)
      return rewriter.notifyMatchFailure(tanhOp, "type conversion failed");

    Location loc = tanhOp.getLoc();
    Value two = createFPConstant(loc, srcType, dstType, rewriter, 2.0);
    Value multiplied =
        rewriter.create<LLVM::FMulOp>(loc, dstType, two, tanhOp.getOperand());
    Value exponential = rewriter.create<LLVM::ExpOp>(loc, dstType, multiplied);
    Value one = createFPConstant(loc, srcType, dstType, rewriter, 1.0);
    Value numerator =
        rewriter.create<LLVM::FSubOp>(loc, dstType, exponential, one);
    Value denominator =
        rewriter.create<LLVM::FAddOp>(loc, dstType, exponential, one);
    rewriter.replaceOpWithNewOp<LLVM::FDivOp>(tanhOp, dstType, numerator,
                                              denominator);
    return success();
  }
};

class VariablePattern : public SPIRVToLLVMConversion<spirv::VariableOp> {
public:
  using SPIRVToLLVMConversion<spirv::VariableOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::VariableOp varOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = varOp.getType();
    // Initialization is supported for scalars and vectors only.
    auto pointerTo = cast<spirv::PointerType>(srcType).getPointeeType();
    auto init = varOp.getInitializer();
    if (init && !pointerTo.isIntOrFloat() && !isa<VectorType>(pointerTo))
      return failure();

    auto dstType = getTypeConverter()->convertType(srcType);
    if (!dstType)
      return rewriter.notifyMatchFailure(varOp, "type conversion failed");

    Location loc = varOp.getLoc();
    Value size = createI32ConstantOf(loc, rewriter, 1);
    if (!init) {
      auto elementType = getTypeConverter()->convertType(pointerTo);
      if (!elementType)
        return rewriter.notifyMatchFailure(varOp, "type conversion failed");
      rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(varOp, dstType, elementType,
                                                  size);
      return success();
    }
    auto elementType = getTypeConverter()->convertType(pointerTo);
    if (!elementType)
      return rewriter.notifyMatchFailure(varOp, "type conversion failed");
    Value allocated =
        rewriter.create<LLVM::AllocaOp>(loc, dstType, elementType, size);
    rewriter.create<LLVM::StoreOp>(loc, adaptor.getInitializer(), allocated);
    rewriter.replaceOp(varOp, allocated);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BitcastOp conversion
//===----------------------------------------------------------------------===//

class BitcastConversionPattern
    : public SPIRVToLLVMConversion<spirv::BitcastOp> {
public:
  using SPIRVToLLVMConversion<spirv::BitcastOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::BitcastOp bitcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = getTypeConverter()->convertType(bitcastOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(bitcastOp, "type conversion failed");

    // LLVM's opaque pointers do not require bitcasts.
    if (isa<LLVM::LLVMPointerType>(dstType)) {
      rewriter.replaceOp(bitcastOp, adaptor.getOperand());
      return success();
    }

    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(
        bitcastOp, dstType, adaptor.getOperands(), bitcastOp->getAttrs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FuncOp conversion
//===----------------------------------------------------------------------===//

class FuncConversionPattern : public SPIRVToLLVMConversion<spirv::FuncOp> {
public:
  using SPIRVToLLVMConversion<spirv::FuncOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Convert function signature. At the moment LLVMType converter is enough
    // for currently supported types.
    auto funcType = funcOp.getFunctionType();
    TypeConverter::SignatureConversion signatureConverter(
        funcType.getNumInputs());
    auto llvmType = static_cast<const LLVMTypeConverter *>(getTypeConverter())
                        ->convertFunctionSignature(
                            funcType, /*isVariadic=*/false,
                            /*useBarePtrCallConv=*/false, signatureConverter);
    if (!llvmType)
      return failure();

    // Create a new `LLVMFuncOp`
    Location loc = funcOp.getLoc();
    StringRef name = funcOp.getName();
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(loc, name, llvmType);

    // Convert SPIR-V Function Control to equivalent LLVM function attribute
    MLIRContext *context = funcOp.getContext();
    switch (funcOp.getFunctionControl()) {
    case spirv::FunctionControl::Inline:
      newFuncOp.setAlwaysInline(true);
      break;
    case spirv::FunctionControl::DontInline:
      newFuncOp.setNoInline(true);
      break;

#define DISPATCH(functionControl, llvmAttr)                                    \
  case functionControl:                                                        \
    newFuncOp->setAttr("passthrough", ArrayAttr::get(context, {llvmAttr}));    \
    break;

      DISPATCH(spirv::FunctionControl::Pure,
               StringAttr::get(context, "readonly"));
      DISPATCH(spirv::FunctionControl::Const,
               StringAttr::get(context, "readnone"));

#undef DISPATCH

    // Default: if `spirv::FunctionControl::None`, then no attributes are
    // needed.
    default:
      break;
    }

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(
            &newFuncOp.getBody(), *getTypeConverter(), &signatureConverter))) {
      return failure();
    }
    rewriter.eraseOp(funcOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ModuleOp conversion
//===----------------------------------------------------------------------===//

class ModuleConversionPattern : public SPIRVToLLVMConversion<spirv::ModuleOp> {
public:
  using SPIRVToLLVMConversion<spirv::ModuleOp>::SPIRVToLLVMConversion;

  LogicalResult
  matchAndRewrite(spirv::ModuleOp spvModuleOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto newModuleOp =
        rewriter.create<ModuleOp>(spvModuleOp.getLoc(), spvModuleOp.getName());
    rewriter.inlineRegionBefore(spvModuleOp.getRegion(), newModuleOp.getBody());

    // Remove the terminator block that was automatically added by builder
    rewriter.eraseBlock(&newModuleOp.getBodyRegion().back());
    rewriter.eraseOp(spvModuleOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// VectorShuffleOp conversion
//===----------------------------------------------------------------------===//

class VectorShufflePattern
    : public SPIRVToLLVMConversion<spirv::VectorShuffleOp> {
public:
  using SPIRVToLLVMConversion<spirv::VectorShuffleOp>::SPIRVToLLVMConversion;
  LogicalResult
  matchAndRewrite(spirv::VectorShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto components = adaptor.getComponents();
    auto vector1 = adaptor.getVector1();
    auto vector2 = adaptor.getVector2();
    int vector1Size = cast<VectorType>(vector1.getType()).getNumElements();
    int vector2Size = cast<VectorType>(vector2.getType()).getNumElements();
    if (vector1Size == vector2Size) {
      rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(
          op, vector1, vector2,
          LLVM::convertArrayToIndices<int32_t>(components));
      return success();
    }

    auto dstType = getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    auto scalarType = cast<VectorType>(dstType).getElementType();
    auto componentsArray = components.getValue();
    auto *context = rewriter.getContext();
    auto llvmI32Type = IntegerType::get(context, 32);
    Value targetOp = rewriter.create<LLVM::UndefOp>(loc, dstType);
    for (unsigned i = 0; i < componentsArray.size(); i++) {
      if (!isa<IntegerAttr>(componentsArray[i]))
        return op.emitError("unable to support non-constant component");

      int indexVal = cast<IntegerAttr>(componentsArray[i]).getInt();
      if (indexVal == -1)
        continue;

      int offsetVal = 0;
      Value baseVector = vector1;
      if (indexVal >= vector1Size) {
        offsetVal = vector1Size;
        baseVector = vector2;
      }

      Value dstIndex = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI32Type, rewriter.getIntegerAttr(rewriter.getI32Type(), i));
      Value index = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI32Type,
          rewriter.getIntegerAttr(rewriter.getI32Type(), indexVal - offsetVal));

      auto extractOp = rewriter.create<LLVM::ExtractElementOp>(
          loc, scalarType, baseVector, index);
      targetOp = rewriter.create<LLVM::InsertElementOp>(loc, dstType, targetOp,
                                                        extractOp, dstIndex);
    }
    rewriter.replaceOp(op, targetOp);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateSPIRVToLLVMTypeConversion(LLVMTypeConverter &typeConverter,
                                             spirv::ClientAPI clientAPI) {
  typeConverter.addConversion([&](spirv::ArrayType type) {
    return convertArrayType(type, typeConverter);
  });
  typeConverter.addConversion([&, clientAPI](spirv::PointerType type) {
    return convertPointerType(type, typeConverter, clientAPI);
  });
  typeConverter.addConversion([&](spirv::RuntimeArrayType type) {
    return convertRuntimeArrayType(type, typeConverter);
  });
  typeConverter.addConversion([&](spirv::StructType type) {
    return convertStructType(type, typeConverter);
  });
}

void mlir::populateSPIRVToLLVMConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    spirv::ClientAPI clientAPI) {
  patterns.add<
      // Arithmetic ops
      DirectConversionPattern<spirv::IAddOp, LLVM::AddOp>,
      DirectConversionPattern<spirv::IMulOp, LLVM::MulOp>,
      DirectConversionPattern<spirv::ISubOp, LLVM::SubOp>,
      DirectConversionPattern<spirv::FAddOp, LLVM::FAddOp>,
      DirectConversionPattern<spirv::FDivOp, LLVM::FDivOp>,
      DirectConversionPattern<spirv::FMulOp, LLVM::FMulOp>,
      DirectConversionPattern<spirv::FNegateOp, LLVM::FNegOp>,
      DirectConversionPattern<spirv::FRemOp, LLVM::FRemOp>,
      DirectConversionPattern<spirv::FSubOp, LLVM::FSubOp>,
      DirectConversionPattern<spirv::SDivOp, LLVM::SDivOp>,
      DirectConversionPattern<spirv::SRemOp, LLVM::SRemOp>,
      DirectConversionPattern<spirv::UDivOp, LLVM::UDivOp>,
      DirectConversionPattern<spirv::UModOp, LLVM::URemOp>,

      // Bitwise ops
      BitFieldInsertPattern, BitFieldUExtractPattern, BitFieldSExtractPattern,
      DirectConversionPattern<spirv::BitCountOp, LLVM::CtPopOp>,
      DirectConversionPattern<spirv::BitReverseOp, LLVM::BitReverseOp>,
      DirectConversionPattern<spirv::BitwiseAndOp, LLVM::AndOp>,
      DirectConversionPattern<spirv::BitwiseOrOp, LLVM::OrOp>,
      DirectConversionPattern<spirv::BitwiseXorOp, LLVM::XOrOp>,
      NotPattern<spirv::NotOp>,

      // Cast ops
      BitcastConversionPattern,
      DirectConversionPattern<spirv::ConvertFToSOp, LLVM::FPToSIOp>,
      DirectConversionPattern<spirv::ConvertFToUOp, LLVM::FPToUIOp>,
      DirectConversionPattern<spirv::ConvertSToFOp, LLVM::SIToFPOp>,
      DirectConversionPattern<spirv::ConvertUToFOp, LLVM::UIToFPOp>,
      IndirectCastPattern<spirv::FConvertOp, LLVM::FPExtOp, LLVM::FPTruncOp>,
      IndirectCastPattern<spirv::SConvertOp, LLVM::SExtOp, LLVM::TruncOp>,
      IndirectCastPattern<spirv::UConvertOp, LLVM::ZExtOp, LLVM::TruncOp>,

      // Comparison ops
      IComparePattern<spirv::IEqualOp, LLVM::ICmpPredicate::eq>,
      IComparePattern<spirv::INotEqualOp, LLVM::ICmpPredicate::ne>,
      FComparePattern<spirv::FOrdEqualOp, LLVM::FCmpPredicate::oeq>,
      FComparePattern<spirv::FOrdGreaterThanOp, LLVM::FCmpPredicate::ogt>,
      FComparePattern<spirv::FOrdGreaterThanEqualOp, LLVM::FCmpPredicate::oge>,
      FComparePattern<spirv::FOrdLessThanEqualOp, LLVM::FCmpPredicate::ole>,
      FComparePattern<spirv::FOrdLessThanOp, LLVM::FCmpPredicate::olt>,
      FComparePattern<spirv::FOrdNotEqualOp, LLVM::FCmpPredicate::one>,
      FComparePattern<spirv::FUnordEqualOp, LLVM::FCmpPredicate::ueq>,
      FComparePattern<spirv::FUnordGreaterThanOp, LLVM::FCmpPredicate::ugt>,
      FComparePattern<spirv::FUnordGreaterThanEqualOp,
                      LLVM::FCmpPredicate::uge>,
      FComparePattern<spirv::FUnordLessThanEqualOp, LLVM::FCmpPredicate::ule>,
      FComparePattern<spirv::FUnordLessThanOp, LLVM::FCmpPredicate::ult>,
      FComparePattern<spirv::FUnordNotEqualOp, LLVM::FCmpPredicate::une>,
      IComparePattern<spirv::SGreaterThanOp, LLVM::ICmpPredicate::sgt>,
      IComparePattern<spirv::SGreaterThanEqualOp, LLVM::ICmpPredicate::sge>,
      IComparePattern<spirv::SLessThanEqualOp, LLVM::ICmpPredicate::sle>,
      IComparePattern<spirv::SLessThanOp, LLVM::ICmpPredicate::slt>,
      IComparePattern<spirv::UGreaterThanOp, LLVM::ICmpPredicate::ugt>,
      IComparePattern<spirv::UGreaterThanEqualOp, LLVM::ICmpPredicate::uge>,
      IComparePattern<spirv::ULessThanEqualOp, LLVM::ICmpPredicate::ule>,
      IComparePattern<spirv::ULessThanOp, LLVM::ICmpPredicate::ult>,

      // Constant op
      ConstantScalarAndVectorPattern,

      // Control Flow ops
      BranchConversionPattern, BranchConditionalConversionPattern,
      FunctionCallPattern, LoopPattern, SelectionPattern,
      ErasePattern<spirv::MergeOp>,

      // Entry points and execution mode are handled separately.
      ErasePattern<spirv::EntryPointOp>, ExecutionModePattern,

      // GLSL extended instruction set ops
      DirectConversionPattern<spirv::GLCeilOp, LLVM::FCeilOp>,
      DirectConversionPattern<spirv::GLCosOp, LLVM::CosOp>,
      DirectConversionPattern<spirv::GLExpOp, LLVM::ExpOp>,
      DirectConversionPattern<spirv::GLFAbsOp, LLVM::FAbsOp>,
      DirectConversionPattern<spirv::GLFloorOp, LLVM::FFloorOp>,
      DirectConversionPattern<spirv::GLFMaxOp, LLVM::MaxNumOp>,
      DirectConversionPattern<spirv::GLFMinOp, LLVM::MinNumOp>,
      DirectConversionPattern<spirv::GLLogOp, LLVM::LogOp>,
      DirectConversionPattern<spirv::GLSinOp, LLVM::SinOp>,
      DirectConversionPattern<spirv::GLSMaxOp, LLVM::SMaxOp>,
      DirectConversionPattern<spirv::GLSMinOp, LLVM::SMinOp>,
      DirectConversionPattern<spirv::GLSqrtOp, LLVM::SqrtOp>,
      InverseSqrtPattern, TanPattern, TanhPattern,

      // Logical ops
      DirectConversionPattern<spirv::LogicalAndOp, LLVM::AndOp>,
      DirectConversionPattern<spirv::LogicalOrOp, LLVM::OrOp>,
      IComparePattern<spirv::LogicalEqualOp, LLVM::ICmpPredicate::eq>,
      IComparePattern<spirv::LogicalNotEqualOp, LLVM::ICmpPredicate::ne>,
      NotPattern<spirv::LogicalNotOp>,

      // Memory ops
      AccessChainPattern, AddressOfPattern, LoadStorePattern<spirv::LoadOp>,
      LoadStorePattern<spirv::StoreOp>, VariablePattern,

      // Miscellaneous ops
      CompositeExtractPattern, CompositeInsertPattern,
      DirectConversionPattern<spirv::SelectOp, LLVM::SelectOp>,
      DirectConversionPattern<spirv::UndefOp, LLVM::UndefOp>,
      VectorShufflePattern,

      // Shift ops
      ShiftPattern<spirv::ShiftRightArithmeticOp, LLVM::AShrOp>,
      ShiftPattern<spirv::ShiftRightLogicalOp, LLVM::LShrOp>,
      ShiftPattern<spirv::ShiftLeftLogicalOp, LLVM::ShlOp>,

      // Return ops
      ReturnPattern, ReturnValuePattern,

      // Barrier ops
      ControlBarrierPattern<spirv::ControlBarrierOp>,
      ControlBarrierPattern<spirv::INTELControlBarrierArriveOp>,
      ControlBarrierPattern<spirv::INTELControlBarrierWaitOp>,

      // Group reduction operations
      GroupReducePattern<spirv::GroupIAddOp>,
      GroupReducePattern<spirv::GroupFAddOp>,
      GroupReducePattern<spirv::GroupFMinOp>,
      GroupReducePattern<spirv::GroupUMinOp>,
      GroupReducePattern<spirv::GroupSMinOp, /*Signed=*/true>,
      GroupReducePattern<spirv::GroupFMaxOp>,
      GroupReducePattern<spirv::GroupUMaxOp>,
      GroupReducePattern<spirv::GroupSMaxOp, /*Signed=*/true>,
      GroupReducePattern<spirv::GroupNonUniformIAddOp, /*Signed=*/false,
                         /*NonUniform=*/true>,
      GroupReducePattern<spirv::GroupNonUniformFAddOp, /*Signed=*/false,
                         /*NonUniform=*/true>,
      GroupReducePattern<spirv::GroupNonUniformIMulOp, /*Signed=*/false,
                         /*NonUniform=*/true>,
      GroupReducePattern<spirv::GroupNonUniformFMulOp, /*Signed=*/false,
                         /*NonUniform=*/true>,
      GroupReducePattern<spirv::GroupNonUniformSMinOp, /*Signed=*/true,
                         /*NonUniform=*/true>,
      GroupReducePattern<spirv::GroupNonUniformUMinOp, /*Signed=*/false,
                         /*NonUniform=*/true>,
      GroupReducePattern<spirv::GroupNonUniformFMinOp, /*Signed=*/false,
                         /*NonUniform=*/true>,
      GroupReducePattern<spirv::GroupNonUniformSMaxOp, /*Signed=*/true,
                         /*NonUniform=*/true>,
      GroupReducePattern<spirv::GroupNonUniformUMaxOp, /*Signed=*/false,
                         /*NonUniform=*/true>,
      GroupReducePattern<spirv::GroupNonUniformFMaxOp, /*Signed=*/false,
                         /*NonUniform=*/true>,
      GroupReducePattern<spirv::GroupNonUniformBitwiseAndOp, /*Signed=*/false,
                         /*NonUniform=*/true>,
      GroupReducePattern<spirv::GroupNonUniformBitwiseOrOp, /*Signed=*/false,
                         /*NonUniform=*/true>,
      GroupReducePattern<spirv::GroupNonUniformBitwiseXorOp, /*Signed=*/false,
                         /*NonUniform=*/true>,
      GroupReducePattern<spirv::GroupNonUniformLogicalAndOp, /*Signed=*/false,
                         /*NonUniform=*/true>,
      GroupReducePattern<spirv::GroupNonUniformLogicalOrOp, /*Signed=*/false,
                         /*NonUniform=*/true>,
      GroupReducePattern<spirv::GroupNonUniformLogicalXorOp, /*Signed=*/false,
                         /*NonUniform=*/true>>(patterns.getContext(),
                                               typeConverter);

  patterns.add<GlobalVariablePattern>(clientAPI, patterns.getContext(),
                                      typeConverter);
}

void mlir::populateSPIRVToLLVMFunctionConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<FuncConversionPattern>(patterns.getContext(), typeConverter);
}

void mlir::populateSPIRVToLLVMModuleConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<ModuleConversionPattern>(patterns.getContext(), typeConverter);
}

//===----------------------------------------------------------------------===//
// Pre-conversion hooks
//===----------------------------------------------------------------------===//

/// Hook for descriptor set and binding number encoding.
static constexpr StringRef kBinding = "binding";
static constexpr StringRef kDescriptorSet = "descriptor_set";
void mlir::encodeBindAttribute(ModuleOp module) {
  auto spvModules = module.getOps<spirv::ModuleOp>();
  for (auto spvModule : spvModules) {
    spvModule.walk([&](spirv::GlobalVariableOp op) {
      IntegerAttr descriptorSet =
          op->getAttrOfType<IntegerAttr>(kDescriptorSet);
      IntegerAttr binding = op->getAttrOfType<IntegerAttr>(kBinding);
      // For every global variable in the module, get the ones with descriptor
      // set and binding numbers.
      if (descriptorSet && binding) {
        // Encode these numbers into the variable's symbolic name. If the
        // SPIR-V module has a name, add it at the beginning.
        auto moduleAndName =
            spvModule.getName().has_value()
                ? spvModule.getName()->str() + "_" + op.getSymName().str()
                : op.getSymName().str();
        std::string name =
            llvm::formatv("{0}_descriptor_set{1}_binding{2}", moduleAndName,
                          std::to_string(descriptorSet.getInt()),
                          std::to_string(binding.getInt()));
        auto nameAttr = StringAttr::get(op->getContext(), name);

        // Replace all symbol uses and set the new symbol name. Finally, remove
        // descriptor set and binding attributes.
        if (failed(SymbolTable::replaceAllSymbolUses(op, nameAttr, spvModule)))
          op.emitError("unable to replace all symbol uses for ") << name;
        SymbolTable::setSymbolName(op, nameAttr);
        op->removeAttr(kDescriptorSet);
        op->removeAttr(kBinding);
      }
    });
  }
}
