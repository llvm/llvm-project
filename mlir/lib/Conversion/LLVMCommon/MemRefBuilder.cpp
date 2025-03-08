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
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// MemRefDescriptor implementation
//===----------------------------------------------------------------------===//

/// Construct a helper for the given descriptor value.
MemRefDescriptor::MemRefDescriptor(ValueRange elements) : elements(elements) {
  indexType = elements[kOffsetPosInMemRefDescriptor].getType();
}

/// Builds IR creating an `undef` value of the descriptor type.
MemRefDescriptor MemRefDescriptor::poison(OpBuilder &builder, Location loc,
                                          TypeRange descriptorTypes) {
  DenseMap<Type, Value> poisonValues;
  SmallVector<Value> elements;
  for (Type t : descriptorTypes) {
    auto it = poisonValues.find(t);
    if (it == poisonValues.end()) {
      poisonValues[t] = builder.create<LLVM::PoisonOp>(loc, t);
    }
    elements.push_back(poisonValues[t]);
  }
  return MemRefDescriptor(elements);
}

/// Builds IR creating a MemRef descriptor that represents `type` and
/// populates it with static shape and stride information extracted from the
/// type.
MemRefDescriptor
MemRefDescriptor::fromStaticShape(OpBuilder &builder, Location loc,
                                  const LLVMTypeConverter &typeConverter,
                                  MemRefType type, Value memory) {
  return fromStaticShape(builder, loc, typeConverter, type, memory, memory);
}

MemRefDescriptor MemRefDescriptor::fromStaticShape(
    OpBuilder &builder, Location loc, const LLVMTypeConverter &typeConverter,
    MemRefType type, Value memory, Value alignedMemory) {
  assert(type.hasStaticShape() && "unexpected dynamic shape");

  // Extract all strides and offsets and verify they are static.
  auto [strides, offset] = type.getStridesAndOffset();
  assert(!ShapedType::isDynamic(offset) && "expected static offset");
  assert(!llvm::any_of(strides, ShapedType::isDynamic) &&
         "expected static strides");

  SmallVector<Type> convertedTypes;
  LogicalResult status = typeConverter.convertType(type, convertedTypes);
  assert(succeeded(status) && "unexpected failure in memref type conversion");

  auto descr = MemRefDescriptor::poison(builder, loc, convertedTypes);
  descr.setAllocatedPtr(builder, loc, memory);
  descr.setAlignedPtr(builder, loc, alignedMemory);
  descr.setConstantOffset(builder, loc, offset);

  // Fill in sizes and strides
  for (unsigned i = 0, e = type.getRank(); i != e; ++i) {
    descr.setConstantSize(builder, loc, i, type.getDimSize(i));
    descr.setConstantStride(builder, loc, i, strides[i]);
  }
  return descr;
}

static Value extractStructElement(OpBuilder &builder, Location loc,
                                  Value packed, ArrayRef<int64_t> idx) {
  return builder.create<LLVM::ExtractValueOp>(loc, packed, idx);
}

static Value insertStructElement(OpBuilder &builder, Location loc, Value packed,
                                 Value val, ArrayRef<int64_t> idx) {
  return builder.create<LLVM::InsertValueOp>(loc, packed, val, idx);
}
MemRefDescriptor MemRefDescriptor::fromPackedStruct(OpBuilder &builder,
                                                    Location loc,
                                                    Value packed) {
  auto llvmStruct = cast<LLVM::LLVMStructType>(packed.getType());
  auto llvmArray = cast<LLVM::LLVMArrayType>(llvmStruct.getBody()[2]);
  int64_t rank = llvmArray.getNumElements();
  SmallVector<Value> elements;
  elements.push_back(extractStructElement(builder, loc, packed, 0));
  elements.push_back(extractStructElement(builder, loc, packed, 1));
  elements.push_back(extractStructElement(builder, loc, packed, 2));
  for (int i = 0; i < rank; ++i)
    elements.push_back(extractStructElement(builder, loc, packed, {3, i}));
  for (int i = 0; i < rank; ++i)
    elements.push_back(extractStructElement(builder, loc, packed, {4, i}));
  return MemRefDescriptor(elements);
}

Value MemRefDescriptor::packStruct(OpBuilder &builder, Location loc) {
  auto llvmArray = LLVM::LLVMArrayType::get(builder.getContext(),
                                            elements[3].getType(), getRank());
  SmallVector<Type> fields;
  fields.push_back(elements[0].getType());
  fields.push_back(elements[1].getType());
  fields.push_back(elements[2].getType());
  fields.push_back(llvmArray);
  fields.push_back(llvmArray);
  Value desc = builder.create<LLVM::UndefOp>(
      loc, LLVM::LLVMStructType::getLiteral(builder.getContext(), fields));
  desc = insertStructElement(builder, loc, desc, elements[0], 0);
  desc = insertStructElement(builder, loc, desc, elements[1], 1);
  desc = insertStructElement(builder, loc, desc, elements[2], 2);
  for (int i = 0; i < getRank(); ++i)
    desc = insertStructElement(builder, loc, desc, elements[3 + i], {3, i});
  for (int i = 0; i < getRank(); ++i)
    desc = insertStructElement(builder, loc, desc, elements[3 + getRank() + i],
                               {4, i});
  return desc;
}

/// Builds IR extracting the allocated pointer from the descriptor.
Value MemRefDescriptor::allocatedPtr(OpBuilder &builder, Location loc) {
  return elements[kAllocatedPtrPosInMemRefDescriptor];
}

/// Builds IR inserting the allocated pointer into the descriptor.
void MemRefDescriptor::setAllocatedPtr(OpBuilder &builder, Location loc,
                                       Value ptr) {
  elements[kAllocatedPtrPosInMemRefDescriptor] = ptr;
}

/// Builds IR extracting the aligned pointer from the descriptor.
Value MemRefDescriptor::alignedPtr(OpBuilder &builder, Location loc) {
  return elements[kAlignedPtrPosInMemRefDescriptor];
}

/// Builds IR inserting the aligned pointer into the descriptor.
void MemRefDescriptor::setAlignedPtr(OpBuilder &builder, Location loc,
                                     Value ptr) {
  elements[kAlignedPtrPosInMemRefDescriptor] = ptr;
}

// Creates a constant Op producing a value of `resultType` from an index-typed
// integer attribute.
static Value createIndexAttrConstant(OpBuilder &builder, Location loc,
                                     Type resultType, int64_t value) {
  return builder.create<LLVM::ConstantOp>(loc, resultType,
                                          builder.getIndexAttr(value));
}

/// Builds IR extracting the offset from the descriptor.
Value MemRefDescriptor::offset(OpBuilder &builder, Location loc) {
  return elements[kOffsetPosInMemRefDescriptor];
}

/// Builds IR inserting the offset into the descriptor.
void MemRefDescriptor::setOffset(OpBuilder &builder, Location loc,
                                 Value offset) {
  elements[kOffsetPosInMemRefDescriptor] = offset;
}

/// Builds IR inserting the offset into the descriptor.
void MemRefDescriptor::setConstantOffset(OpBuilder &builder, Location loc,
                                         uint64_t offset) {
  elements[kOffsetPosInMemRefDescriptor] =
      createIndexAttrConstant(builder, loc, indexType, offset);
}

/// Builds IR extracting the pos-th size from the descriptor.
Value MemRefDescriptor::size(OpBuilder &builder, Location loc, unsigned pos) {
  return elements[kSizePosInMemRefDescriptor + pos];
}

Value MemRefDescriptor::size(OpBuilder &builder, Location loc, Value pos,
                             int64_t rank) {
  auto arrayTy = LLVM::LLVMArrayType::get(indexType, rank);

  auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());

  // Copy size values to stack-allocated memory.
  auto one = createIndexAttrConstant(builder, loc, indexType, 1);
  SmallVector<Type> structElems(rank, indexType);
  Value sizes = builder.create<LLVM::UndefOp>(
      loc, LLVM::LLVMStructType::getLiteral(builder.getContext(), structElems));
  ValueRange sizeVals =
      ValueRange(elements).slice(kSizePosInMemRefDescriptor, rank);
  for (auto it : llvm::enumerate(sizeVals))
    sizes =
        builder.create<LLVM::InsertValueOp>(loc, sizes, it.value(), it.index());
  auto sizesPtr = builder.create<LLVM::AllocaOp>(loc, ptrTy, arrayTy, one,
                                                 /*alignment=*/0);
  builder.create<LLVM::StoreOp>(loc, sizes, sizesPtr);

  // Load an return size value of interest.
  auto resultPtr = builder.create<LLVM::GEPOp>(loc, ptrTy, arrayTy, sizesPtr,
                                               ArrayRef<LLVM::GEPArg>{0, pos});
  return builder.create<LLVM::LoadOp>(loc, indexType, resultPtr);
}

/// Builds IR inserting the pos-th size into the descriptor
void MemRefDescriptor::setSize(OpBuilder &builder, Location loc, unsigned pos,
                               Value size) {
  elements[kSizePosInMemRefDescriptor] = size;
}

void MemRefDescriptor::setConstantSize(OpBuilder &builder, Location loc,
                                       unsigned pos, uint64_t size) {
  elements[kSizePosInMemRefDescriptor] =
      createIndexAttrConstant(builder, loc, indexType, size);
}

/// Builds IR extracting the pos-th stride from the descriptor.
Value MemRefDescriptor::stride(OpBuilder &builder, Location loc, unsigned pos) {
  return elements[kSizePosInMemRefDescriptor + getRank() + pos];
}

/// Builds IR inserting the pos-th stride into the descriptor
void MemRefDescriptor::setStride(OpBuilder &builder, Location loc, unsigned pos,
                                 Value stride) {
  elements[kSizePosInMemRefDescriptor + getRank() + pos] = stride;
}

void MemRefDescriptor::setConstantStride(OpBuilder &builder, Location loc,
                                         unsigned pos, uint64_t stride) {
  elements[kSizePosInMemRefDescriptor + getRank() + pos] =
      createIndexAttrConstant(builder, loc, indexType, stride);
}

LLVM::LLVMPointerType MemRefDescriptor::getElementPtrType() {
  return cast<LLVM::LLVMPointerType>(
      elements[kAlignedPtrPosInMemRefDescriptor].getType());
}

Value MemRefDescriptor::bufferPtr(OpBuilder &builder, Location loc,
                                  const LLVMTypeConverter &converter,
                                  MemRefType type) {
  // When we convert to LLVM, the input memref must have been normalized
  // beforehand. Hence, this call is guaranteed to work.
  auto [strides, offsetCst] = type.getStridesAndOffset();

  Value ptr = alignedPtr(builder, loc);
  // For zero offsets, we already have the base pointer.
  if (offsetCst == 0)
    return ptr;

  // Otherwise add the offset to the aligned base.
  Type indexType = converter.getIndexType();
  Value offsetVal =
      ShapedType::isDynamic(offsetCst)
          ? offset(builder, loc)
          : createIndexAttrConstant(builder, loc, indexType, offsetCst);
  Type elementType = converter.convertType(type.getElementType());
  ptr = builder.create<LLVM::GEPOp>(loc, ptr.getType(), elementType, ptr,
                                    offsetVal);
  return ptr;
}

/// Returns the number of non-aggregate values that would be produced by
/// `unpack`.
unsigned MemRefDescriptor::getNumUnpackedValues(MemRefType type) {
  // Two pointers, offset, <rank> sizes, <rank> strides.
  return 3 + 2 * type.getRank();
}

int64_t MemRefDescriptor::getRank() { return (elements.size() - 3) / 2; }

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
UnrankedMemRefDescriptor::UnrankedMemRefDescriptor(ValueRange elements)
    : elements(elements) {}

/// Builds IR creating an `undef` value of the descriptor type.
UnrankedMemRefDescriptor
UnrankedMemRefDescriptor::poison(OpBuilder &builder, Location loc,
                                 TypeRange descriptorTypes) {
  DenseMap<Type, Value> poisonValues;
  SmallVector<Value> elements;
  for (Type t : descriptorTypes) {
    auto it = poisonValues.find(t);
    if (it == poisonValues.end()) {
      poisonValues[t] = builder.create<LLVM::PoisonOp>(loc, t);
    }
    elements.push_back(poisonValues[t]);
  }
  return UnrankedMemRefDescriptor(elements);
}

/// Builds IR extracting individual elements of a MemRef descriptor structure
/// and returning them as `results` list.
UnrankedMemRefDescriptor
UnrankedMemRefDescriptor::fromPackedStruct(OpBuilder &builder, Location loc,
                                           Value packed) {
  SmallVector<Value> elements;
  elements.push_back(extractStructElement(builder, loc, packed, 0));
  elements.push_back(extractStructElement(builder, loc, packed, 1));
  return UnrankedMemRefDescriptor(elements);
}

Value UnrankedMemRefDescriptor::packStruct(OpBuilder &builder, Location loc) {
  SmallVector<Type> fields;
  fields.push_back(elements[0].getType());
  fields.push_back(elements[1].getType());
  Value desc = builder.create<LLVM::UndefOp>(
      loc, LLVM::LLVMStructType::getLiteral(builder.getContext(), fields));
  desc = insertStructElement(builder, loc, desc, elements[0], 0);
  desc = insertStructElement(builder, loc, desc, elements[1], 1);
  return desc;
}

Value UnrankedMemRefDescriptor::rank(OpBuilder &builder, Location loc) const {
  return elements[kRankInUnrankedMemRefDescriptor];
}
void UnrankedMemRefDescriptor::setRank(OpBuilder &builder, Location loc,
                                       Value v) {
  elements[kRankInUnrankedMemRefDescriptor] = v;
}
Value UnrankedMemRefDescriptor::memRefDescPtr(OpBuilder &builder,
                                              Location loc) const {
  return elements[kPtrInUnrankedMemRefDescriptor];
}
void UnrankedMemRefDescriptor::setMemRefDescPtr(OpBuilder &builder,
                                                Location loc, Value v) {
  elements[kPtrInUnrankedMemRefDescriptor] = v;
}

void UnrankedMemRefDescriptor::computeSizes(
    OpBuilder &builder, Location loc, const LLVMTypeConverter &typeConverter,
    ArrayRef<UnrankedMemRefDescriptor> values, ArrayRef<unsigned> addressSpaces,
    SmallVectorImpl<Value> &sizes) {
  if (values.empty())
    return;
  assert(values.size() == addressSpaces.size() &&
         "must provide address space for each descriptor");
  // Cache the index type.
  Type indexType = typeConverter.getIndexType();

  // Initialize shared constants.
  Value one = createIndexAttrConstant(builder, loc, indexType, 1);
  Value two = createIndexAttrConstant(builder, loc, indexType, 2);
  Value indexSize = createIndexAttrConstant(
      builder, loc, indexType,
      llvm::divideCeil(typeConverter.getIndexTypeBitwidth(), 8));

  sizes.reserve(sizes.size() + values.size());
  for (auto [desc, addressSpace] : llvm::zip(values, addressSpaces)) {
    // Emit IR computing the memory necessary to store the descriptor. This
    // assumes the descriptor to be
    //   { type*, type*, index, index[rank], index[rank] }
    // and densely packed, so the total size is
    //   2 * sizeof(pointer) + (1 + 2 * rank) * sizeof(index).
    // TODO: consider including the actual size (including eventual padding due
    // to data layout) into the unranked descriptor.
    Value pointerSize = createIndexAttrConstant(
        builder, loc, indexType,
        llvm::divideCeil(typeConverter.getPointerBitwidth(addressSpace), 8));
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

Value UnrankedMemRefDescriptor::allocatedPtr(
    OpBuilder &builder, Location loc, Value memRefDescPtr,
    LLVM::LLVMPointerType elemPtrType) {
  return builder.create<LLVM::LoadOp>(loc, elemPtrType, memRefDescPtr);
}

void UnrankedMemRefDescriptor::setAllocatedPtr(
    OpBuilder &builder, Location loc, Value memRefDescPtr,
    LLVM::LLVMPointerType elemPtrType, Value allocatedPtr) {
  builder.create<LLVM::StoreOp>(loc, allocatedPtr, memRefDescPtr);
}

static std::pair<Value, Type>
castToElemPtrPtr(OpBuilder &builder, Location loc, Value memRefDescPtr,
                 LLVM::LLVMPointerType elemPtrType) {
  auto elemPtrPtrType = LLVM::LLVMPointerType::get(builder.getContext());
  return {memRefDescPtr, elemPtrPtrType};
}

Value UnrankedMemRefDescriptor::alignedPtr(
    OpBuilder &builder, Location loc, const LLVMTypeConverter &typeConverter,
    Value memRefDescPtr, LLVM::LLVMPointerType elemPtrType) {
  auto [elementPtrPtr, elemPtrPtrType] =
      castToElemPtrPtr(builder, loc, memRefDescPtr, elemPtrType);

  Value alignedGep =
      builder.create<LLVM::GEPOp>(loc, elemPtrPtrType, elemPtrType,
                                  elementPtrPtr, ArrayRef<LLVM::GEPArg>{1});
  return builder.create<LLVM::LoadOp>(loc, elemPtrType, alignedGep);
}

void UnrankedMemRefDescriptor::setAlignedPtr(
    OpBuilder &builder, Location loc, const LLVMTypeConverter &typeConverter,
    Value memRefDescPtr, LLVM::LLVMPointerType elemPtrType, Value alignedPtr) {
  auto [elementPtrPtr, elemPtrPtrType] =
      castToElemPtrPtr(builder, loc, memRefDescPtr, elemPtrType);

  Value alignedGep =
      builder.create<LLVM::GEPOp>(loc, elemPtrPtrType, elemPtrType,
                                  elementPtrPtr, ArrayRef<LLVM::GEPArg>{1});
  builder.create<LLVM::StoreOp>(loc, alignedPtr, alignedGep);
}

Value UnrankedMemRefDescriptor::offsetBasePtr(
    OpBuilder &builder, Location loc, const LLVMTypeConverter &typeConverter,
    Value memRefDescPtr, LLVM::LLVMPointerType elemPtrType) {
  auto [elementPtrPtr, elemPtrPtrType] =
      castToElemPtrPtr(builder, loc, memRefDescPtr, elemPtrType);

  return builder.create<LLVM::GEPOp>(loc, elemPtrPtrType, elemPtrType,
                                     elementPtrPtr, ArrayRef<LLVM::GEPArg>{2});
}

Value UnrankedMemRefDescriptor::offset(OpBuilder &builder, Location loc,
                                       const LLVMTypeConverter &typeConverter,
                                       Value memRefDescPtr,
                                       LLVM::LLVMPointerType elemPtrType) {
  Value offsetPtr =
      offsetBasePtr(builder, loc, typeConverter, memRefDescPtr, elemPtrType);
  return builder.create<LLVM::LoadOp>(loc, typeConverter.getIndexType(),
                                      offsetPtr);
}

void UnrankedMemRefDescriptor::setOffset(OpBuilder &builder, Location loc,
                                         const LLVMTypeConverter &typeConverter,
                                         Value memRefDescPtr,
                                         LLVM::LLVMPointerType elemPtrType,
                                         Value offset) {
  Value offsetPtr =
      offsetBasePtr(builder, loc, typeConverter, memRefDescPtr, elemPtrType);
  builder.create<LLVM::StoreOp>(loc, offset, offsetPtr);
}

Value UnrankedMemRefDescriptor::sizeBasePtr(
    OpBuilder &builder, Location loc, const LLVMTypeConverter &typeConverter,
    Value memRefDescPtr, LLVM::LLVMPointerType elemPtrType) {
  Type indexTy = typeConverter.getIndexType();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      indexTy.getContext(), {elemPtrType, elemPtrType, indexTy, indexTy});
  auto resultType = LLVM::LLVMPointerType::get(builder.getContext());
  return builder.create<LLVM::GEPOp>(loc, resultType, structTy, memRefDescPtr,
                                     ArrayRef<LLVM::GEPArg>{0, 3});
}

Value UnrankedMemRefDescriptor::size(OpBuilder &builder, Location loc,
                                     const LLVMTypeConverter &typeConverter,
                                     Value sizeBasePtr, Value index) {

  Type indexTy = typeConverter.getIndexType();
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());

  Value sizeStoreGep =
      builder.create<LLVM::GEPOp>(loc, ptrType, indexTy, sizeBasePtr, index);
  return builder.create<LLVM::LoadOp>(loc, indexTy, sizeStoreGep);
}

void UnrankedMemRefDescriptor::setSize(OpBuilder &builder, Location loc,
                                       const LLVMTypeConverter &typeConverter,
                                       Value sizeBasePtr, Value index,
                                       Value size) {
  Type indexTy = typeConverter.getIndexType();
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());

  Value sizeStoreGep =
      builder.create<LLVM::GEPOp>(loc, ptrType, indexTy, sizeBasePtr, index);
  builder.create<LLVM::StoreOp>(loc, size, sizeStoreGep);
}

Value UnrankedMemRefDescriptor::strideBasePtr(
    OpBuilder &builder, Location loc, const LLVMTypeConverter &typeConverter,
    Value sizeBasePtr, Value rank) {
  Type indexTy = typeConverter.getIndexType();
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());

  return builder.create<LLVM::GEPOp>(loc, ptrType, indexTy, sizeBasePtr, rank);
}

Value UnrankedMemRefDescriptor::stride(OpBuilder &builder, Location loc,
                                       const LLVMTypeConverter &typeConverter,
                                       Value strideBasePtr, Value index,
                                       Value stride) {
  Type indexTy = typeConverter.getIndexType();
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());

  Value strideStoreGep =
      builder.create<LLVM::GEPOp>(loc, ptrType, indexTy, strideBasePtr, index);
  return builder.create<LLVM::LoadOp>(loc, indexTy, strideStoreGep);
}

void UnrankedMemRefDescriptor::setStride(OpBuilder &builder, Location loc,
                                         const LLVMTypeConverter &typeConverter,
                                         Value strideBasePtr, Value index,
                                         Value stride) {
  Type indexTy = typeConverter.getIndexType();
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());

  Value strideStoreGep =
      builder.create<LLVM::GEPOp>(loc, ptrType, indexTy, strideBasePtr, index);
  builder.create<LLVM::StoreOp>(loc, stride, strideStoreGep);
}
