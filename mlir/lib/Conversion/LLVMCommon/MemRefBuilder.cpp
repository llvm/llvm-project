//===- MemRefBuilder.cpp - Helper for LLVM MemRef equivalents -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "MemRefDescriptor.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/MathExtras.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// MemRefDescriptor implementation
//===----------------------------------------------------------------------===//

/// Construct a helper for the given descriptor value.
MemRefDescriptor::MemRefDescriptor(Value descriptor)
    : StructBuilder(descriptor) {
  assert(value != nullptr && "value cannot be null");
  indexType = value.getType()
                  .cast<LLVM::LLVMStructType>()
                  .getBody()[kOffsetPosInMemRefDescriptor];
}

/// Builds IR creating an `undef` value of the descriptor type.
MemRefDescriptor MemRefDescriptor::undef(OpBuilder &builder, Location loc,
                                         Type descriptorType) {

  Value descriptor = builder.create<LLVM::UndefOp>(loc, descriptorType);
  return MemRefDescriptor(descriptor);
}

/// Builds IR creating a MemRef descriptor that represents `type` and
/// populates it with static shape and stride information extracted from the
/// type.
MemRefDescriptor
MemRefDescriptor::fromStaticShape(OpBuilder &builder, Location loc,
                                  LLVMTypeConverter &typeConverter,
                                  MemRefType type, Value memory) {
  assert(type.hasStaticShape() && "unexpected dynamic shape");

  // Extract all strides and offsets and verify they are static.
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto result = getStridesAndOffset(type, strides, offset);
  (void)result;
  assert(succeeded(result) && "unexpected failure in stride computation");
  assert(!ShapedType::isDynamicStrideOrOffset(offset) &&
         "expected static offset");
  assert(!llvm::any_of(strides, ShapedType::isDynamicStrideOrOffset) &&
         "expected static strides");

  auto convertedType = typeConverter.convertType(type);
  assert(convertedType && "unexpected failure in memref type conversion");

  auto descr = MemRefDescriptor::undef(builder, loc, convertedType);
  descr.setAllocatedPtr(builder, loc, memory);
  descr.setAlignedPtr(builder, loc, memory);
  descr.setConstantOffset(builder, loc, offset);

  // Fill in sizes and strides
  for (unsigned i = 0, e = type.getRank(); i != e; ++i) {
    descr.setConstantSize(builder, loc, i, type.getDimSize(i));
    descr.setConstantStride(builder, loc, i, strides[i]);
  }
  return descr;
}

/// Builds IR extracting the allocated pointer from the descriptor.
Value MemRefDescriptor::allocatedPtr(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kAllocatedPtrPosInMemRefDescriptor);
}

/// Builds IR inserting the allocated pointer into the descriptor.
void MemRefDescriptor::setAllocatedPtr(OpBuilder &builder, Location loc,
                                       Value ptr) {
  setPtr(builder, loc, kAllocatedPtrPosInMemRefDescriptor, ptr);
}

/// Builds IR extracting the aligned pointer from the descriptor.
Value MemRefDescriptor::alignedPtr(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kAlignedPtrPosInMemRefDescriptor);
}

/// Builds IR inserting the aligned pointer into the descriptor.
void MemRefDescriptor::setAlignedPtr(OpBuilder &builder, Location loc,
                                     Value ptr) {
  setPtr(builder, loc, kAlignedPtrPosInMemRefDescriptor, ptr);
}

// Creates a constant Op producing a value of `resultType` from an index-typed
// integer attribute.
static Value createIndexAttrConstant(OpBuilder &builder, Location loc,
                                     Type resultType, int64_t value) {
  return builder.create<LLVM::ConstantOp>(
      loc, resultType, builder.getIntegerAttr(builder.getIndexType(), value));
}

/// Builds IR extracting the offset from the descriptor.
Value MemRefDescriptor::offset(OpBuilder &builder, Location loc) {
  return builder.create<LLVM::ExtractValueOp>(
      loc, indexType, value,
      builder.getI64ArrayAttr(kOffsetPosInMemRefDescriptor));
}

/// Builds IR inserting the offset into the descriptor.
void MemRefDescriptor::setOffset(OpBuilder &builder, Location loc,
                                 Value offset) {
  value = builder.create<LLVM::InsertValueOp>(
      loc, structType, value, offset,
      builder.getI64ArrayAttr(kOffsetPosInMemRefDescriptor));
}

/// Builds IR inserting the offset into the descriptor.
void MemRefDescriptor::setConstantOffset(OpBuilder &builder, Location loc,
                                         uint64_t offset) {
  setOffset(builder, loc,
            createIndexAttrConstant(builder, loc, indexType, offset));
}

/// Builds IR extracting the pos-th size from the descriptor.
Value MemRefDescriptor::size(OpBuilder &builder, Location loc, unsigned pos) {
  return builder.create<LLVM::ExtractValueOp>(
      loc, indexType, value,
      builder.getI64ArrayAttr({kSizePosInMemRefDescriptor, pos}));
}

Value MemRefDescriptor::size(OpBuilder &builder, Location loc, Value pos,
                             int64_t rank) {
  auto indexPtrTy = LLVM::LLVMPointerType::get(indexType);
  auto arrayTy = LLVM::LLVMArrayType::get(indexType, rank);
  auto arrayPtrTy = LLVM::LLVMPointerType::get(arrayTy);

  // Copy size values to stack-allocated memory.
  auto zero = createIndexAttrConstant(builder, loc, indexType, 0);
  auto one = createIndexAttrConstant(builder, loc, indexType, 1);
  auto sizes = builder.create<LLVM::ExtractValueOp>(
      loc, arrayTy, value,
      builder.getI64ArrayAttr({kSizePosInMemRefDescriptor}));
  auto sizesPtr =
      builder.create<LLVM::AllocaOp>(loc, arrayPtrTy, one, /*alignment=*/0);
  builder.create<LLVM::StoreOp>(loc, sizes, sizesPtr);

  // Load an return size value of interest.
  auto resultPtr = builder.create<LLVM::GEPOp>(loc, indexPtrTy, sizesPtr,
                                               ValueRange({zero, pos}));
  return builder.create<LLVM::LoadOp>(loc, resultPtr);
}

/// Builds IR inserting the pos-th size into the descriptor
void MemRefDescriptor::setSize(OpBuilder &builder, Location loc, unsigned pos,
                               Value size) {
  value = builder.create<LLVM::InsertValueOp>(
      loc, structType, value, size,
      builder.getI64ArrayAttr({kSizePosInMemRefDescriptor, pos}));
}

void MemRefDescriptor::setConstantSize(OpBuilder &builder, Location loc,
                                       unsigned pos, uint64_t size) {
  setSize(builder, loc, pos,
          createIndexAttrConstant(builder, loc, indexType, size));
}

/// Builds IR extracting the pos-th stride from the descriptor.
Value MemRefDescriptor::stride(OpBuilder &builder, Location loc, unsigned pos) {
  return builder.create<LLVM::ExtractValueOp>(
      loc, indexType, value,
      builder.getI64ArrayAttr({kStridePosInMemRefDescriptor, pos}));
}

/// Builds IR inserting the pos-th stride into the descriptor
void MemRefDescriptor::setStride(OpBuilder &builder, Location loc, unsigned pos,
                                 Value stride) {
  value = builder.create<LLVM::InsertValueOp>(
      loc, structType, value, stride,
      builder.getI64ArrayAttr({kStridePosInMemRefDescriptor, pos}));
}

void MemRefDescriptor::setConstantStride(OpBuilder &builder, Location loc,
                                         unsigned pos, uint64_t stride) {
  setStride(builder, loc, pos,
            createIndexAttrConstant(builder, loc, indexType, stride));
}

LLVM::LLVMPointerType MemRefDescriptor::getElementPtrType() {
  return value.getType()
      .cast<LLVM::LLVMStructType>()
      .getBody()[kAlignedPtrPosInMemRefDescriptor]
      .cast<LLVM::LLVMPointerType>();
}

/// Creates a MemRef descriptor structure from a list of individual values
/// composing that descriptor, in the following order:
/// - allocated pointer;
/// - aligned pointer;
/// - offset;
/// - <rank> sizes;
/// - <rank> shapes;
/// where <rank> is the MemRef rank as provided in `type`.
Value MemRefDescriptor::pack(OpBuilder &builder, Location loc,
                             LLVMTypeConverter &converter, MemRefType type,
                             ValueRange values) {
  Type llvmType = converter.convertType(type);
  auto d = MemRefDescriptor::undef(builder, loc, llvmType);

  d.setAllocatedPtr(builder, loc, values[kAllocatedPtrPosInMemRefDescriptor]);
  d.setAlignedPtr(builder, loc, values[kAlignedPtrPosInMemRefDescriptor]);
  d.setOffset(builder, loc, values[kOffsetPosInMemRefDescriptor]);

  int64_t rank = type.getRank();
  for (unsigned i = 0; i < rank; ++i) {
    d.setSize(builder, loc, i, values[kSizePosInMemRefDescriptor + i]);
    d.setStride(builder, loc, i, values[kSizePosInMemRefDescriptor + rank + i]);
  }

  return d;
}

/// Builds IR extracting individual elements of a MemRef descriptor structure
/// and returning them as `results` list.
void MemRefDescriptor::unpack(OpBuilder &builder, Location loc, Value packed,
                              MemRefType type,
                              SmallVectorImpl<Value> &results) {
  int64_t rank = type.getRank();
  results.reserve(results.size() + getNumUnpackedValues(type));

  MemRefDescriptor d(packed);
  results.push_back(d.allocatedPtr(builder, loc));
  results.push_back(d.alignedPtr(builder, loc));
  results.push_back(d.offset(builder, loc));
  for (int64_t i = 0; i < rank; ++i)
    results.push_back(d.size(builder, loc, i));
  for (int64_t i = 0; i < rank; ++i)
    results.push_back(d.stride(builder, loc, i));
}

/// Returns the number of non-aggregate values that would be produced by
/// `unpack`.
unsigned MemRefDescriptor::getNumUnpackedValues(MemRefType type) {
  // Two pointers, offset, <rank> sizes, <rank> shapes.
  return 3 + 2 * type.getRank();
}

//===----------------------------------------------------------------------===//
// MemRefDescriptorView implementation.
//===----------------------------------------------------------------------===//

MemRefDescriptorView::MemRefDescriptorView(ValueRange range)
    : rank((range.size() - kSizePosInMemRefDescriptor) / 2), elements(range) {}

Value MemRefDescriptorView::allocatedPtr() {
  return elements[kAllocatedPtrPosInMemRefDescriptor];
}

Value MemRefDescriptorView::alignedPtr() {
  return elements[kAlignedPtrPosInMemRefDescriptor];
}

Value MemRefDescriptorView::offset() {
  return elements[kOffsetPosInMemRefDescriptor];
}

Value MemRefDescriptorView::size(unsigned pos) {
  return elements[kSizePosInMemRefDescriptor + pos];
}

Value MemRefDescriptorView::stride(unsigned pos) {
  return elements[kSizePosInMemRefDescriptor + rank + pos];
}

//===----------------------------------------------------------------------===//
// UnrankedMemRefDescriptor implementation
//===----------------------------------------------------------------------===//

/// Construct a helper for the given descriptor value.
UnrankedMemRefDescriptor::UnrankedMemRefDescriptor(Value descriptor)
    : StructBuilder(descriptor) {}

/// Builds IR creating an `undef` value of the descriptor type.
UnrankedMemRefDescriptor UnrankedMemRefDescriptor::undef(OpBuilder &builder,
                                                         Location loc,
                                                         Type descriptorType) {
  Value descriptor = builder.create<LLVM::UndefOp>(loc, descriptorType);
  return UnrankedMemRefDescriptor(descriptor);
}
Value UnrankedMemRefDescriptor::rank(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kRankInUnrankedMemRefDescriptor);
}
void UnrankedMemRefDescriptor::setRank(OpBuilder &builder, Location loc,
                                       Value v) {
  setPtr(builder, loc, kRankInUnrankedMemRefDescriptor, v);
}
Value UnrankedMemRefDescriptor::memRefDescPtr(OpBuilder &builder,
                                              Location loc) {
  return extractPtr(builder, loc, kPtrInUnrankedMemRefDescriptor);
}
void UnrankedMemRefDescriptor::setMemRefDescPtr(OpBuilder &builder,
                                                Location loc, Value v) {
  setPtr(builder, loc, kPtrInUnrankedMemRefDescriptor, v);
}

/// Builds IR populating an unranked MemRef descriptor structure from a list
/// of individual constituent values in the following order:
/// - rank of the memref;
/// - pointer to the memref descriptor.
Value UnrankedMemRefDescriptor::pack(OpBuilder &builder, Location loc,
                                     LLVMTypeConverter &converter,
                                     UnrankedMemRefType type,
                                     ValueRange values) {
  Type llvmType = converter.convertType(type);
  auto d = UnrankedMemRefDescriptor::undef(builder, loc, llvmType);

  d.setRank(builder, loc, values[kRankInUnrankedMemRefDescriptor]);
  d.setMemRefDescPtr(builder, loc, values[kPtrInUnrankedMemRefDescriptor]);
  return d;
}

/// Builds IR extracting individual elements that compose an unranked memref
/// descriptor and returns them as `results` list.
void UnrankedMemRefDescriptor::unpack(OpBuilder &builder, Location loc,
                                      Value packed,
                                      SmallVectorImpl<Value> &results) {
  UnrankedMemRefDescriptor d(packed);
  results.reserve(results.size() + 2);
  results.push_back(d.rank(builder, loc));
  results.push_back(d.memRefDescPtr(builder, loc));
}

void UnrankedMemRefDescriptor::computeSizes(
    OpBuilder &builder, Location loc, LLVMTypeConverter &typeConverter,
    ArrayRef<UnrankedMemRefDescriptor> values, SmallVectorImpl<Value> &sizes) {
  if (values.empty())
    return;

  // Cache the index type.
  Type indexType = typeConverter.getIndexType();

  // Initialize shared constants.
  Value one = createIndexAttrConstant(builder, loc, indexType, 1);
  Value two = createIndexAttrConstant(builder, loc, indexType, 2);
  Value pointerSize = createIndexAttrConstant(
      builder, loc, indexType, ceilDiv(typeConverter.getPointerBitwidth(), 8));
  Value indexSize =
      createIndexAttrConstant(builder, loc, indexType,
                              ceilDiv(typeConverter.getIndexTypeBitwidth(), 8));

  sizes.reserve(sizes.size() + values.size());
  for (UnrankedMemRefDescriptor desc : values) {
    // Emit IR computing the memory necessary to store the descriptor. This
    // assumes the descriptor to be
    //   { type*, type*, index, index[rank], index[rank] }
    // and densely packed, so the total size is
    //   2 * sizeof(pointer) + (1 + 2 * rank) * sizeof(index).
    // TODO: consider including the actual size (including eventual padding due
    // to data layout) into the unranked descriptor.
    Value doublePointerSize =
        builder.create<LLVM::MulOp>(loc, indexType, two, pointerSize);

    // (1 + 2 * rank) * sizeof(index)
    Value rank = desc.rank(builder, loc);
    Value doubleRank = builder.create<LLVM::MulOp>(loc, indexType, two, rank);
    Value doubleRankIncremented =
        builder.create<LLVM::AddOp>(loc, indexType, doubleRank, one);
    Value rankIndexSize = builder.create<LLVM::MulOp>(
        loc, indexType, doubleRankIncremented, indexSize);

    // Total allocation size.
    Value allocationSize = builder.create<LLVM::AddOp>(
        loc, indexType, doublePointerSize, rankIndexSize);
    sizes.push_back(allocationSize);
  }
}

Value UnrankedMemRefDescriptor::allocatedPtr(OpBuilder &builder, Location loc,
                                             Value memRefDescPtr,
                                             Type elemPtrPtrType) {

  Value elementPtrPtr =
      builder.create<LLVM::BitcastOp>(loc, elemPtrPtrType, memRefDescPtr);
  return builder.create<LLVM::LoadOp>(loc, elementPtrPtr);
}

void UnrankedMemRefDescriptor::setAllocatedPtr(OpBuilder &builder, Location loc,
                                               Value memRefDescPtr,
                                               Type elemPtrPtrType,
                                               Value allocatedPtr) {
  Value elementPtrPtr =
      builder.create<LLVM::BitcastOp>(loc, elemPtrPtrType, memRefDescPtr);
  builder.create<LLVM::StoreOp>(loc, allocatedPtr, elementPtrPtr);
}

Value UnrankedMemRefDescriptor::alignedPtr(OpBuilder &builder, Location loc,
                                           LLVMTypeConverter &typeConverter,
                                           Value memRefDescPtr,
                                           Type elemPtrPtrType) {
  Value elementPtrPtr =
      builder.create<LLVM::BitcastOp>(loc, elemPtrPtrType, memRefDescPtr);

  Value one =
      createIndexAttrConstant(builder, loc, typeConverter.getIndexType(), 1);
  Value alignedGep = builder.create<LLVM::GEPOp>(
      loc, elemPtrPtrType, elementPtrPtr, ValueRange({one}));
  return builder.create<LLVM::LoadOp>(loc, alignedGep);
}

void UnrankedMemRefDescriptor::setAlignedPtr(OpBuilder &builder, Location loc,
                                             LLVMTypeConverter &typeConverter,
                                             Value memRefDescPtr,
                                             Type elemPtrPtrType,
                                             Value alignedPtr) {
  Value elementPtrPtr =
      builder.create<LLVM::BitcastOp>(loc, elemPtrPtrType, memRefDescPtr);

  Value one =
      createIndexAttrConstant(builder, loc, typeConverter.getIndexType(), 1);
  Value alignedGep = builder.create<LLVM::GEPOp>(
      loc, elemPtrPtrType, elementPtrPtr, ValueRange({one}));
  builder.create<LLVM::StoreOp>(loc, alignedPtr, alignedGep);
}

Value UnrankedMemRefDescriptor::offset(OpBuilder &builder, Location loc,
                                       LLVMTypeConverter &typeConverter,
                                       Value memRefDescPtr,
                                       Type elemPtrPtrType) {
  Value elementPtrPtr =
      builder.create<LLVM::BitcastOp>(loc, elemPtrPtrType, memRefDescPtr);

  Value two =
      createIndexAttrConstant(builder, loc, typeConverter.getIndexType(), 2);
  Value offsetGep = builder.create<LLVM::GEPOp>(
      loc, elemPtrPtrType, elementPtrPtr, ValueRange({two}));
  offsetGep = builder.create<LLVM::BitcastOp>(
      loc, LLVM::LLVMPointerType::get(typeConverter.getIndexType()), offsetGep);
  return builder.create<LLVM::LoadOp>(loc, offsetGep);
}

void UnrankedMemRefDescriptor::setOffset(OpBuilder &builder, Location loc,
                                         LLVMTypeConverter &typeConverter,
                                         Value memRefDescPtr,
                                         Type elemPtrPtrType, Value offset) {
  Value elementPtrPtr =
      builder.create<LLVM::BitcastOp>(loc, elemPtrPtrType, memRefDescPtr);

  Value two =
      createIndexAttrConstant(builder, loc, typeConverter.getIndexType(), 2);
  Value offsetGep = builder.create<LLVM::GEPOp>(
      loc, elemPtrPtrType, elementPtrPtr, ValueRange({two}));
  offsetGep = builder.create<LLVM::BitcastOp>(
      loc, LLVM::LLVMPointerType::get(typeConverter.getIndexType()), offsetGep);
  builder.create<LLVM::StoreOp>(loc, offset, offsetGep);
}

Value UnrankedMemRefDescriptor::sizeBasePtr(
    OpBuilder &builder, Location loc, LLVMTypeConverter &typeConverter,
    Value memRefDescPtr, LLVM::LLVMPointerType elemPtrPtrType) {
  Type elemPtrTy = elemPtrPtrType.getElementType();
  Type indexTy = typeConverter.getIndexType();
  Type structPtrTy =
      LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getLiteral(
          indexTy.getContext(), {elemPtrTy, elemPtrTy, indexTy, indexTy}));
  Value structPtr =
      builder.create<LLVM::BitcastOp>(loc, structPtrTy, memRefDescPtr);

  Type int32Type = typeConverter.convertType(builder.getI32Type());
  Value zero =
      createIndexAttrConstant(builder, loc, typeConverter.getIndexType(), 0);
  Value three = builder.create<LLVM::ConstantOp>(loc, int32Type,
                                                 builder.getI32IntegerAttr(3));
  return builder.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(indexTy),
                                     structPtr, ValueRange({zero, three}));
}

Value UnrankedMemRefDescriptor::size(OpBuilder &builder, Location loc,
                                     LLVMTypeConverter &typeConverter,
                                     Value sizeBasePtr, Value index) {
  Type indexPtrTy = LLVM::LLVMPointerType::get(typeConverter.getIndexType());
  Value sizeStoreGep = builder.create<LLVM::GEPOp>(loc, indexPtrTy, sizeBasePtr,
                                                   ValueRange({index}));
  return builder.create<LLVM::LoadOp>(loc, sizeStoreGep);
}

void UnrankedMemRefDescriptor::setSize(OpBuilder &builder, Location loc,
                                       LLVMTypeConverter &typeConverter,
                                       Value sizeBasePtr, Value index,
                                       Value size) {
  Type indexPtrTy = LLVM::LLVMPointerType::get(typeConverter.getIndexType());
  Value sizeStoreGep = builder.create<LLVM::GEPOp>(loc, indexPtrTy, sizeBasePtr,
                                                   ValueRange({index}));
  builder.create<LLVM::StoreOp>(loc, size, sizeStoreGep);
}

Value UnrankedMemRefDescriptor::strideBasePtr(OpBuilder &builder, Location loc,
                                              LLVMTypeConverter &typeConverter,
                                              Value sizeBasePtr, Value rank) {
  Type indexPtrTy = LLVM::LLVMPointerType::get(typeConverter.getIndexType());
  return builder.create<LLVM::GEPOp>(loc, indexPtrTy, sizeBasePtr,
                                     ValueRange({rank}));
}

Value UnrankedMemRefDescriptor::stride(OpBuilder &builder, Location loc,
                                       LLVMTypeConverter &typeConverter,
                                       Value strideBasePtr, Value index,
                                       Value stride) {
  Type indexPtrTy = LLVM::LLVMPointerType::get(typeConverter.getIndexType());
  Value strideStoreGep = builder.create<LLVM::GEPOp>(
      loc, indexPtrTy, strideBasePtr, ValueRange({index}));
  return builder.create<LLVM::LoadOp>(loc, strideStoreGep);
}

void UnrankedMemRefDescriptor::setStride(OpBuilder &builder, Location loc,
                                         LLVMTypeConverter &typeConverter,
                                         Value strideBasePtr, Value index,
                                         Value stride) {
  Type indexPtrTy = LLVM::LLVMPointerType::get(typeConverter.getIndexType());
  Value strideStoreGep = builder.create<LLVM::GEPOp>(
      loc, indexPtrTy, strideBasePtr, ValueRange({index}));
  builder.create<LLVM::StoreOp>(loc, stride, strideStoreGep);
}
