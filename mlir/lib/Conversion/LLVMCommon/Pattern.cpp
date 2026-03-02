//===- Pattern.cpp - Conversion pattern to the LLVM dialect ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ConvertToLLVMPattern
//===----------------------------------------------------------------------===//

ConvertToLLVMPattern::ConvertToLLVMPattern(
    StringRef rootOpName, MLIRContext *context,
    const LLVMTypeConverter &typeConverter, PatternBenefit benefit)
    : ConversionPattern(typeConverter, rootOpName, benefit, context) {}

const LLVMTypeConverter *ConvertToLLVMPattern::getTypeConverter() const {
  return static_cast<const LLVMTypeConverter *>(
      ConversionPattern::getTypeConverter());
}

LLVM::LLVMDialect &ConvertToLLVMPattern::getDialect() const {
  return *getTypeConverter()->getDialect();
}

Type ConvertToLLVMPattern::getIndexType() const {
  return getTypeConverter()->getIndexType();
}

Type ConvertToLLVMPattern::getIntPtrType(unsigned addressSpace) const {
  return IntegerType::get(&getTypeConverter()->getContext(),
                          getTypeConverter()->getPointerBitwidth(addressSpace));
}

Type ConvertToLLVMPattern::getVoidType() const {
  return LLVM::LLVMVoidType::get(&getTypeConverter()->getContext());
}

Type ConvertToLLVMPattern::getPtrType(unsigned addressSpace) const {
  return LLVM::LLVMPointerType::get(&getTypeConverter()->getContext(),
                                    addressSpace);
}

Type ConvertToLLVMPattern::getVoidPtrType() const { return getPtrType(); }

Value ConvertToLLVMPattern::createIndexAttrConstant(OpBuilder &builder,
                                                    Location loc,
                                                    Type resultType,
                                                    int64_t value) {
  return LLVM::ConstantOp::create(builder, loc, resultType,
                                  builder.getIndexAttr(value));
}

Value ConvertToLLVMPattern::getStridedElementPtr(
    ConversionPatternRewriter &rewriter, Location loc, MemRefType type,
    Value memRefDesc, ValueRange indices,
    LLVM::GEPNoWrapFlags noWrapFlags) const {
  return LLVM::getStridedElementPtr(rewriter, loc, *getTypeConverter(), type,
                                    memRefDesc, indices, noWrapFlags);
}

// Check if the MemRefType `type` is supported by the lowering. We currently
// only support memrefs with identity maps.
bool ConvertToLLVMPattern::isConvertibleAndHasIdentityMaps(
    MemRefType type) const {
  if (!type.getLayout().isIdentity())
    return false;
  return static_cast<bool>(typeConverter->convertType(type));
}

Type ConvertToLLVMPattern::getElementPtrType(MemRefType type) const {
  auto addressSpace = getTypeConverter()->getMemRefAddressSpace(type);
  if (failed(addressSpace))
    return {};
  return LLVM::LLVMPointerType::get(type.getContext(), *addressSpace);
}

void ConvertToLLVMPattern::getMemRefDescriptorSizes(
    Location loc, MemRefType memRefType, ValueRange dynamicSizes,
    ConversionPatternRewriter &rewriter, SmallVectorImpl<Value> &sizes,
    SmallVectorImpl<Value> &strides, Value &size, bool sizeInBytes) const {
  assert(isConvertibleAndHasIdentityMaps(memRefType) &&
         "layout maps must have been normalized away");
  assert(count(memRefType.getShape(), ShapedType::kDynamic) ==
             static_cast<ssize_t>(dynamicSizes.size()) &&
         "dynamicSizes size doesn't match dynamic sizes count in memref shape");

  sizes.reserve(memRefType.getRank());
  unsigned dynamicIndex = 0;
  Type indexType = getIndexType();
  for (int64_t size : memRefType.getShape()) {
    sizes.push_back(
        size == ShapedType::kDynamic
            ? dynamicSizes[dynamicIndex++]
            : createIndexAttrConstant(rewriter, loc, indexType, size));
  }

  // Strides: iterate sizes in reverse order and multiply.
  int64_t stride = 1;
  Value runningStride = createIndexAttrConstant(rewriter, loc, indexType, 1);
  strides.resize(memRefType.getRank());
  for (auto i = memRefType.getRank(); i-- > 0;) {
    strides[i] = runningStride;

    int64_t staticSize = memRefType.getShape()[i];
    bool useSizeAsStride = stride == 1;
    if (staticSize == ShapedType::kDynamic)
      stride = ShapedType::kDynamic;
    if (stride != ShapedType::kDynamic)
      stride *= staticSize;

    if (useSizeAsStride)
      runningStride = sizes[i];
    else if (stride == ShapedType::kDynamic)
      runningStride =
          LLVM::MulOp::create(rewriter, loc, runningStride, sizes[i]);
    else
      runningStride = createIndexAttrConstant(rewriter, loc, indexType, stride);
  }
  if (sizeInBytes) {
    // Buffer size in bytes.
    Type elementType = typeConverter->convertType(memRefType.getElementType());
    auto elementPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, elementPtrType);
    Value gepPtr = LLVM::GEPOp::create(rewriter, loc, elementPtrType,
                                       elementType, nullPtr, runningStride);
    size = LLVM::PtrToIntOp::create(rewriter, loc, getIndexType(), gepPtr);
  } else {
    size = runningStride;
  }
}

Value ConvertToLLVMPattern::getSizeInBytes(
    Location loc, Type type, ConversionPatternRewriter &rewriter) const {
  // Compute the size of an individual element. This emits the MLIR equivalent
  // of the following sizeof(...) implementation in LLVM IR:
  //   %0 = getelementptr %elementType* null, %indexType 1
  //   %1 = ptrtoint %elementType* %0 to %indexType
  // which is a common pattern of getting the size of a type in bytes.
  Type llvmType = typeConverter->convertType(type);
  auto convertedPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto nullPtr = LLVM::ZeroOp::create(rewriter, loc, convertedPtrType);
  auto gep = LLVM::GEPOp::create(rewriter, loc, convertedPtrType, llvmType,
                                 nullPtr, ArrayRef<LLVM::GEPArg>{1});
  return LLVM::PtrToIntOp::create(rewriter, loc, getIndexType(), gep);
}

Value ConvertToLLVMPattern::getNumElements(
    Location loc, MemRefType memRefType, ValueRange dynamicSizes,
    ConversionPatternRewriter &rewriter) const {
  assert(count(memRefType.getShape(), ShapedType::kDynamic) ==
             static_cast<ssize_t>(dynamicSizes.size()) &&
         "dynamicSizes size doesn't match dynamic sizes count in memref shape");

  Type indexType = getIndexType();
  Value numElements = memRefType.getRank() == 0
                          ? createIndexAttrConstant(rewriter, loc, indexType, 1)
                          : nullptr;
  unsigned dynamicIndex = 0;

  // Compute the total number of memref elements.
  for (int64_t staticSize : memRefType.getShape()) {
    if (numElements) {
      Value size =
          staticSize == ShapedType::kDynamic
              ? dynamicSizes[dynamicIndex++]
              : createIndexAttrConstant(rewriter, loc, indexType, staticSize);
      numElements = LLVM::MulOp::create(rewriter, loc, numElements, size);
    } else {
      numElements =
          staticSize == ShapedType::kDynamic
              ? dynamicSizes[dynamicIndex++]
              : createIndexAttrConstant(rewriter, loc, indexType, staticSize);
    }
  }
  return numElements;
}

/// Creates and populates the memref descriptor struct given all its fields.
MemRefDescriptor ConvertToLLVMPattern::createMemRefDescriptor(
    Location loc, MemRefType memRefType, Value allocatedPtr, Value alignedPtr,
    ArrayRef<Value> sizes, ArrayRef<Value> strides,
    ConversionPatternRewriter &rewriter) const {
  auto structType = typeConverter->convertType(memRefType);
  auto memRefDescriptor = MemRefDescriptor::poison(rewriter, loc, structType);

  // Field 1: Allocated pointer, used for malloc/free.
  memRefDescriptor.setAllocatedPtr(rewriter, loc, allocatedPtr);

  // Field 2: Actual aligned pointer to payload.
  memRefDescriptor.setAlignedPtr(rewriter, loc, alignedPtr);

  // Field 3: Offset in aligned pointer.
  Type indexType = getIndexType();
  memRefDescriptor.setOffset(
      rewriter, loc, createIndexAttrConstant(rewriter, loc, indexType, 0));

  // Fields 4: Sizes.
  for (const auto &en : llvm::enumerate(sizes))
    memRefDescriptor.setSize(rewriter, loc, en.index(), en.value());

  // Field 5: Strides.
  for (const auto &en : llvm::enumerate(strides))
    memRefDescriptor.setStride(rewriter, loc, en.index(), en.value());

  return memRefDescriptor;
}

Value ConvertToLLVMPattern::copyUnrankedDescriptor(
    OpBuilder &builder, Location loc, UnrankedMemRefType memRefType,
    Value operand, bool toDynamic) const {
  // Convert memory space.
  FailureOr<unsigned> addressSpace =
      getTypeConverter()->getMemRefAddressSpace(memRefType);
  if (failed(addressSpace))
    return {};

  // Get frequently used types.
  Type indexType = getTypeConverter()->getIndexType();

  // Find the malloc and free, or declare them if necessary.
  auto module = builder.getInsertionPoint()->getParentOfType<ModuleOp>();
  FailureOr<LLVM::LLVMFuncOp> freeFunc, mallocFunc;
  if (toDynamic) {
    mallocFunc = LLVM::lookupOrCreateMallocFn(builder, module, indexType);
    if (failed(mallocFunc))
      return {};
  }
  if (!toDynamic) {
    freeFunc = LLVM::lookupOrCreateFreeFn(builder, module);
    if (failed(freeFunc))
      return {};
  }

  UnrankedMemRefDescriptor desc(operand);
  Value allocationSize = UnrankedMemRefDescriptor::computeSize(
      builder, loc, *getTypeConverter(), desc, *addressSpace);

  // Allocate memory, copy, and free the source if necessary.
  Value memory = toDynamic
                     ? LLVM::CallOp::create(builder, loc, mallocFunc.value(),
                                            allocationSize)
                           .getResult()
                     : LLVM::AllocaOp::create(builder, loc, getPtrType(),
                                              IntegerType::get(getContext(), 8),
                                              allocationSize,
                                              /*alignment=*/0);
  Value source = desc.memRefDescPtr(builder, loc);
  LLVM::MemcpyOp::create(builder, loc, memory, source, allocationSize, false);
  if (!toDynamic)
    LLVM::CallOp::create(builder, loc, freeFunc.value(), source);

  // Create a new descriptor. The same descriptor can be returned multiple
  // times, attempting to modify its pointer can lead to memory leaks
  // (allocated twice and overwritten) or double frees (the caller does not
  // know if the descriptor points to the same memory).
  Type descriptorType = getTypeConverter()->convertType(memRefType);
  if (!descriptorType)
    return {};
  auto updatedDesc =
      UnrankedMemRefDescriptor::poison(builder, loc, descriptorType);
  Value rank = desc.rank(builder, loc);
  updatedDesc.setRank(builder, loc, rank);
  updatedDesc.setMemRefDescPtr(builder, loc, memory);
  return updatedDesc;
}

LogicalResult ConvertToLLVMPattern::copyUnrankedDescriptors(
    OpBuilder &builder, Location loc, TypeRange origTypes,
    SmallVectorImpl<Value> &operands, bool toDynamic) const {
  assert(origTypes.size() == operands.size() &&
         "expected as may original types as operands");
  for (unsigned i = 0, e = operands.size(); i < e; ++i) {
    if (auto memRefType = dyn_cast<UnrankedMemRefType>(origTypes[i])) {
      Value updatedDesc = copyUnrankedDescriptor(builder, loc, memRefType,
                                                 operands[i], toDynamic);
      if (!updatedDesc)
        return failure();
      operands[i] = updatedDesc;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Detail methods
//===----------------------------------------------------------------------===//

/// Replaces the given operation "op" with a new operation of type "targetOp"
/// and given operands.
LogicalResult LLVM::detail::oneToOneRewrite(
    Operation *op, StringRef targetOp, ValueRange operands,
    ArrayRef<NamedAttribute> targetAttrs, Attribute propertiesAttr,
    const LLVMTypeConverter &typeConverter,
    ConversionPatternRewriter &rewriter) {
  unsigned numResults = op->getNumResults();

  SmallVector<Type> resultTypes;
  if (numResults != 0) {
    resultTypes.push_back(
        typeConverter.packOperationResults(op->getResultTypes()));
    if (!resultTypes.back())
      return failure();
  }

  // Create the operation through state since we don't know its C++ type.
  OperationState state(op->getLoc(), rewriter.getStringAttr(targetOp), operands,
                       resultTypes, targetAttrs);
  state.propertiesAttr = propertiesAttr;
  Operation *newOp = rewriter.create(state);

  // If the operation produced 0 or 1 result, return them immediately.
  if (numResults == 0)
    return rewriter.eraseOp(op), success();
  if (numResults == 1)
    return rewriter.replaceOp(op, newOp->getResult(0)), success();

  // Otherwise, it had been converted to an operation producing a structure.
  // Extract individual results from the structure and return them as list.
  SmallVector<Value, 4> results;
  results.reserve(numResults);
  for (unsigned i = 0; i < numResults; ++i) {
    results.push_back(LLVM::ExtractValueOp::create(rewriter, op->getLoc(),
                                                   newOp->getResult(0), i));
  }
  rewriter.replaceOp(op, results);
  return success();
}

LogicalResult LLVM::detail::intrinsicRewrite(
    Operation *op, StringRef intrinsic, ValueRange operands,
    const LLVMTypeConverter &typeConverter, RewriterBase &rewriter) {
  auto loc = op->getLoc();

  if (!llvm::all_of(operands, [](Value value) {
        return LLVM::isCompatibleType(value.getType());
      }))
    return failure();

  unsigned numResults = op->getNumResults();
  Type resType;
  if (numResults != 0)
    resType = typeConverter.packOperationResults(op->getResultTypes());

  auto callIntrOp = LLVM::CallIntrinsicOp::create(
      rewriter, loc, resType, rewriter.getStringAttr(intrinsic), operands);
  // Propagate attributes.
  callIntrOp->setAttrs(op->getAttrDictionary());

  if (numResults <= 1) {
    // Directly replace the original op.
    rewriter.replaceOp(op, callIntrOp);
    return success();
  }

  // Extract individual results from packed structure and use them as
  // replacements.
  SmallVector<Value, 4> results;
  results.reserve(numResults);
  Value intrRes = callIntrOp.getResults();
  for (unsigned i = 0; i < numResults; ++i)
    results.push_back(LLVM::ExtractValueOp::create(rewriter, loc, intrRes, i));
  rewriter.replaceOp(op, results);

  return success();
}

static unsigned getBitWidth(Type type) {
  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth();

  auto vec = cast<VectorType>(type);
  assert(!vec.isScalable() && "scalable vectors are not supported");
  return vec.getNumElements() * getBitWidth(vec.getElementType());
}

/// Returns true if every leaf in `type` (recursing through LLVM arrays and
/// structs) is either equal to `dstType` or has a fixed bit width.
static bool isFixedSizeAggregate(Type type, Type dstType) {
  if (type == dstType)
    return true;
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(type))
    return isFixedSizeAggregate(arrayType.getElementType(), dstType);
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(type))
    return llvm::all_of(structType.getBody(), [&](Type fieldType) {
      return isFixedSizeAggregate(fieldType, dstType);
    });
  if (auto vecTy = dyn_cast<VectorType>(type))
    return !vecTy.isScalable();
  return type.isIntOrFloat();
}

static Value createI32Constant(OpBuilder &builder, Location loc,
                               int32_t value) {
  Type i32 = builder.getI32Type();
  return LLVM::ConstantOp::create(builder, loc, i32, value);
}

/// Recursive implementation of decomposeValue. When
/// `permitVariablySizedScalars` is false, callers must ensure
/// isFixedSizeAggregate() holds before calling this.
static void decomposeValueImpl(OpBuilder &builder, Location loc, Value src,
                               Type dstType, SmallVectorImpl<Value> &result) {
  Type srcType = src.getType();
  if (srcType == dstType) {
    result.push_back(src);
    return;
  }

  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(srcType)) {
    for (auto i : llvm::seq(arrayType.getNumElements())) {
      Value elem = LLVM::ExtractValueOp::create(builder, loc, src, i);
      decomposeValueImpl(builder, loc, elem, dstType, result);
    }
    return;
  }

  if (auto structType = dyn_cast<LLVM::LLVMStructType>(srcType)) {
    for (auto [i, fieldType] : llvm::enumerate(structType.getBody())) {
      Value field = LLVM::ExtractValueOp::create(builder, loc, src,
                                                 static_cast<int64_t>(i));
      decomposeValueImpl(builder, loc, field, dstType, result);
    }
    return;
  }

  // Variably sized leaf types (e.g., ptr) — pass through as-is.
  if (!srcType.isIntOrFloat() && !isa<VectorType>(srcType)) {
    result.push_back(src);
    return;
  }

  unsigned srcBitWidth = getBitWidth(srcType);
  unsigned dstBitWidth = getBitWidth(dstType);
  if (srcBitWidth == dstBitWidth) {
    Value cast = LLVM::BitcastOp::create(builder, loc, dstType, src);
    result.push_back(cast);
    return;
  }

  if (dstBitWidth > srcBitWidth) {
    auto smallerInt = builder.getIntegerType(srcBitWidth);
    if (srcType != smallerInt)
      src = LLVM::BitcastOp::create(builder, loc, smallerInt, src);

    auto largerInt = builder.getIntegerType(dstBitWidth);
    Value res = LLVM::ZExtOp::create(builder, loc, largerInt, src);
    result.push_back(res);
    return;
  }
  int64_t numElements = llvm::divideCeil(srcBitWidth, dstBitWidth);
  int64_t roundedBitWidth = numElements * dstBitWidth;

  // Pad out values that don't decompose evenly before creating a vector.
  if (roundedBitWidth != srcBitWidth) {
    auto srcInt = builder.getIntegerType(srcBitWidth);
    if (srcType != srcInt)
      src = LLVM::BitcastOp::create(builder, loc, srcInt, src);
    auto roundedInt = builder.getIntegerType(roundedBitWidth);
    src = LLVM::ZExtOp::create(builder, loc, roundedInt, src);
  }

  auto vecType = VectorType::get(numElements, dstType);
  src = LLVM::BitcastOp::create(builder, loc, vecType, src);

  for (auto i : llvm::seq(numElements)) {
    Value idx = createI32Constant(builder, loc, i);
    Value elem = LLVM::ExtractElementOp::create(builder, loc, src, idx);
    result.push_back(elem);
  }
}

LogicalResult mlir::LLVM::decomposeValue(OpBuilder &builder, Location loc,
                                         Value src, Type dstType,
                                         SmallVectorImpl<Value> &result,
                                         bool permitVariablySizedScalars) {
  // Check the type tree before emitting any IR, so that a failing pattern
  // leaves the IR unmodified.
  if (!permitVariablySizedScalars &&
      !isFixedSizeAggregate(src.getType(), dstType))
    return failure();

  decomposeValueImpl(builder, loc, src, dstType, result);
  return success();
}

/// Recursive implementation of composeValue. Consumes elements from `src`
/// starting at `offset`, advancing it past the consumed elements.
static Value composeValueImpl(OpBuilder &builder, Location loc, ValueRange src,
                              size_t &offset, Type dstType) {
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(dstType)) {
    Value result = LLVM::PoisonOp::create(builder, loc, arrayType);
    Type elemType = arrayType.getElementType();
    for (auto i : llvm::seq(arrayType.getNumElements())) {
      Value elem = composeValueImpl(builder, loc, src, offset, elemType);
      result = LLVM::InsertValueOp::create(builder, loc, result, elem, i);
    }
    return result;
  }

  if (auto structType = dyn_cast<LLVM::LLVMStructType>(dstType)) {
    Value result = LLVM::PoisonOp::create(builder, loc, structType);
    for (auto [i, fieldType] : llvm::enumerate(structType.getBody())) {
      Value field = composeValueImpl(builder, loc, src, offset, fieldType);
      result = LLVM::InsertValueOp::create(builder, loc, result, field,
                                           static_cast<int64_t>(i));
    }
    return result;
  }

  // Variably sized leaf types (e.g., ptr) — consume and return as-is.
  if (!dstType.isIntOrFloat() && !isa<VectorType>(dstType))
    return src[offset++];

  unsigned dstBitWidth = getBitWidth(dstType);

  Value front = src[offset];
  if (front.getType() == dstType) {
    ++offset;
    return front;
  }

  // Single element wider than or equal to dst: bitcast/trunc.
  if (front.getType().isIntOrFloat() || isa<VectorType>(front.getType())) {
    unsigned srcBitWidth = getBitWidth(front.getType());
    if (srcBitWidth >= dstBitWidth) {
      ++offset;
      Value res = front;
      if (dstBitWidth < srcBitWidth) {
        auto largerInt = builder.getIntegerType(srcBitWidth);
        if (res.getType() != largerInt)
          res = LLVM::BitcastOp::create(builder, loc, largerInt, res);

        auto smallerInt = builder.getIntegerType(dstBitWidth);
        res = LLVM::TruncOp::create(builder, loc, smallerInt, res);
      }
      if (res.getType() != dstType)
        res = LLVM::BitcastOp::create(builder, loc, dstType, res);
      return res;
    }
  }

  // Multiple elements narrower than dst: gather into a vector and bitcast.
  unsigned elemBitWidth = getBitWidth(front.getType());
  int64_t numElements = llvm::divideCeil(dstBitWidth, elemBitWidth);
  int64_t roundedBitWidth = numElements * elemBitWidth;

  auto vecType = VectorType::get(numElements, front.getType());
  Value res = LLVM::PoisonOp::create(builder, loc, vecType);
  for (auto i : llvm::seq(numElements)) {
    Value idx = createI32Constant(builder, loc, i);
    res = LLVM::InsertElementOp::create(builder, loc, vecType, res,
                                        src[offset++], idx);
  }

  // Undo any padding decomposition might have introduced.
  if (roundedBitWidth != dstBitWidth) {
    auto roundedInt = builder.getIntegerType(roundedBitWidth);
    res = LLVM::BitcastOp::create(builder, loc, roundedInt, res);
    auto dstInt = builder.getIntegerType(dstBitWidth);
    res = LLVM::TruncOp::create(builder, loc, dstInt, res);
    if (dstType != dstInt)
      res = LLVM::BitcastOp::create(builder, loc, dstType, res);
  } else {
    if (res.getType() != dstType)
      res = LLVM::BitcastOp::create(builder, loc, dstType, res);
  }

  return res;
}

Value mlir::LLVM::composeValue(OpBuilder &builder, Location loc, ValueRange src,
                               Type dstType) {
  assert(!src.empty() && "src range must not be empty");
  size_t offset = 0;
  Value result = composeValueImpl(builder, loc, src, offset, dstType);
  assert(offset == src.size() && "not all decomposed values were consumed");
  return result;
}

Value mlir::LLVM::getStridedElementPtr(OpBuilder &builder, Location loc,
                                       const LLVMTypeConverter &converter,
                                       MemRefType type, Value memRefDesc,
                                       ValueRange indices,
                                       LLVM::GEPNoWrapFlags noWrapFlags) {
  auto [strides, offset] = type.getStridesAndOffset();

  MemRefDescriptor memRefDescriptor(memRefDesc);
  // Use a canonical representation of the start address so that later
  // optimizations have a longer sequence of instructions to CSE.
  // If we don't do that we would sprinkle the memref.offset in various
  // position of the different address computations.
  Value base = memRefDescriptor.bufferPtr(builder, loc, converter, type);

  LLVM::IntegerOverflowFlags intOverflowFlags =
      LLVM::IntegerOverflowFlags::none;
  if (LLVM::bitEnumContainsAny(noWrapFlags, LLVM::GEPNoWrapFlags::nusw)) {
    intOverflowFlags = intOverflowFlags | LLVM::IntegerOverflowFlags::nsw;
  }
  if (LLVM::bitEnumContainsAny(noWrapFlags, LLVM::GEPNoWrapFlags::nuw)) {
    intOverflowFlags = intOverflowFlags | LLVM::IntegerOverflowFlags::nuw;
  }

  Type indexType = converter.getIndexType();
  Value index;
  for (int i = 0, e = indices.size(); i < e; ++i) {
    Value increment = indices[i];
    if (strides[i] != 1) { // Skip if stride is 1.
      Value stride =
          ShapedType::isDynamic(strides[i])
              ? memRefDescriptor.stride(builder, loc, i)
              : LLVM::ConstantOp::create(builder, loc, indexType,
                                         builder.getIndexAttr(strides[i]));
      increment = LLVM::MulOp::create(builder, loc, increment, stride,
                                      intOverflowFlags);
    }
    index = index ? LLVM::AddOp::create(builder, loc, index, increment,
                                        intOverflowFlags)
                  : increment;
  }

  Type elementPtrType = memRefDescriptor.getElementPtrType();
  return index
             ? LLVM::GEPOp::create(builder, loc, elementPtrType,
                                   converter.convertType(type.getElementType()),
                                   base, index, noWrapFlags)
             : base;
}

/// Return the given type if it's a floating point type. If the given type is
/// a vector type, return its element type if it's a floating point type.
static FloatType getFloatingPointType(Type type) {
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType;
  if (auto vecType = dyn_cast<VectorType>(type))
    return dyn_cast<FloatType>(vecType.getElementType());
  return nullptr;
}

bool LLVM::detail::isUnsupportedFloatingPointType(
    const TypeConverter &typeConverter, Type type) {
  FloatType floatType = getFloatingPointType(type);
  if (!floatType)
    return false;
  Type convertedType = typeConverter.convertType(floatType);
  if (!convertedType)
    return true;
  return !isa<FloatType>(convertedType);
}

bool LLVM::detail::opHasUnsupportedFloatingPointTypes(
    Operation *op, const TypeConverter &typeConverter) {
  for (Value operand : op->getOperands())
    if (isUnsupportedFloatingPointType(typeConverter, operand.getType()))
      return true;
  return llvm::any_of(op->getResults(), [&typeConverter](OpResult r) {
    return isUnsupportedFloatingPointType(typeConverter, r.getType());
  });
}
