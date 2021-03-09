//===- StandardToLLVM.cpp - Standard to LLVM dialect conversion -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert MLIR standard and builtin dialects
// into the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include <functional>

using namespace mlir;

#define PASS_NAME "convert-std-to-llvm"

// Extract an LLVM IR type from the LLVM IR dialect type.
static Type unwrap(Type type) {
  if (!type)
    return nullptr;
  auto *mlirContext = type.getContext();
  if (!LLVM::isCompatibleType(type))
    emitError(UnknownLoc::get(mlirContext),
              "conversion resulted in a non-LLVM type ")
        << type;
  return type;
}

/// Callback to convert function argument types. It converts a MemRef function
/// argument to a list of non-aggregate types containing descriptor
/// information, and an UnrankedmemRef function argument to a list containing
/// the rank and a pointer to a descriptor struct.
LogicalResult mlir::structFuncArgTypeConverter(LLVMTypeConverter &converter,
                                               Type type,
                                               SmallVectorImpl<Type> &result) {
  if (auto memref = type.dyn_cast<MemRefType>()) {
    // In signatures, Memref descriptors are expanded into lists of
    // non-aggregate values.
    auto converted =
        converter.getMemRefDescriptorFields(memref, /*unpackAggregates=*/true);
    if (converted.empty())
      return failure();
    result.append(converted.begin(), converted.end());
    return success();
  }
  if (type.isa<UnrankedMemRefType>()) {
    auto converted = converter.getUnrankedMemRefDescriptorFields();
    if (converted.empty())
      return failure();
    result.append(converted.begin(), converted.end());
    return success();
  }
  auto converted = converter.convertType(type);
  if (!converted)
    return failure();
  result.push_back(converted);
  return success();
}

/// Callback to convert function argument types. It converts MemRef function
/// arguments to bare pointers to the MemRef element type.
LogicalResult mlir::barePtrFuncArgTypeConverter(LLVMTypeConverter &converter,
                                                Type type,
                                                SmallVectorImpl<Type> &result) {
  auto llvmTy = converter.convertCallingConventionType(type);
  if (!llvmTy)
    return failure();

  result.push_back(llvmTy);
  return success();
}

/// Create an LLVMTypeConverter using default LowerToLLVMOptions.
LLVMTypeConverter::LLVMTypeConverter(MLIRContext *ctx)
    : LLVMTypeConverter(ctx, LowerToLLVMOptions::getDefaultOptions()) {}

/// Create an LLVMTypeConverter using custom LowerToLLVMOptions.
LLVMTypeConverter::LLVMTypeConverter(MLIRContext *ctx,
                                     const LowerToLLVMOptions &options)
    : llvmDialect(ctx->getOrLoadDialect<LLVM::LLVMDialect>()),
      options(options) {
  assert(llvmDialect && "LLVM IR dialect is not registered");
  if (options.indexBitwidth == kDeriveIndexBitwidthFromDataLayout)
    this->options.indexBitwidth = options.dataLayout.getPointerSizeInBits();

  // Register conversions for the builtin types.
  addConversion([&](ComplexType type) { return convertComplexType(type); });
  addConversion([&](FloatType type) { return convertFloatType(type); });
  addConversion([&](FunctionType type) { return convertFunctionType(type); });
  addConversion([&](IndexType type) { return convertIndexType(type); });
  addConversion([&](IntegerType type) { return convertIntegerType(type); });
  addConversion([&](MemRefType type) { return convertMemRefType(type); });
  addConversion(
      [&](UnrankedMemRefType type) { return convertUnrankedMemRefType(type); });
  addConversion([&](VectorType type) { return convertVectorType(type); });

  // LLVM-compatible types are legal, so add a pass-through conversion.
  addConversion([](Type type) {
    return LLVM::isCompatibleType(type) ? llvm::Optional<Type>(type)
                                        : llvm::None;
  });

  // Materialization for memrefs creates descriptor structs from individual
  // values constituting them, when descriptors are used, i.e. more than one
  // value represents a memref.
  addArgumentMaterialization(
      [&](OpBuilder &builder, UnrankedMemRefType resultType, ValueRange inputs,
          Location loc) -> Optional<Value> {
        if (inputs.size() == 1)
          return llvm::None;
        return UnrankedMemRefDescriptor::pack(builder, loc, *this, resultType,
                                              inputs);
      });
  addArgumentMaterialization([&](OpBuilder &builder, MemRefType resultType,
                                 ValueRange inputs,
                                 Location loc) -> Optional<Value> {
    if (inputs.size() == 1)
      return llvm::None;
    return MemRefDescriptor::pack(builder, loc, *this, resultType, inputs);
  });
  // Add generic source and target materializations to handle cases where
  // non-LLVM types persist after an LLVM conversion.
  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return llvm::None;
    // FIXME: These should check LLVM::DialectCastOp can actually be constructed
    // from the input and result.
    return builder.create<LLVM::DialectCastOp>(loc, resultType, inputs[0])
        .getResult();
  });
  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return llvm::None;
    // FIXME: These should check LLVM::DialectCastOp can actually be constructed
    // from the input and result.
    return builder.create<LLVM::DialectCastOp>(loc, resultType, inputs[0])
        .getResult();
  });
}

/// Returns the MLIR context.
MLIRContext &LLVMTypeConverter::getContext() {
  return *getDialect()->getContext();
}

Type LLVMTypeConverter::getIndexType() {
  return IntegerType::get(&getContext(), getIndexTypeBitwidth());
}

unsigned LLVMTypeConverter::getPointerBitwidth(unsigned addressSpace) {
  return options.dataLayout.getPointerSizeInBits(addressSpace);
}

Type LLVMTypeConverter::convertIndexType(IndexType type) {
  return getIndexType();
}

Type LLVMTypeConverter::convertIntegerType(IntegerType type) {
  return IntegerType::get(&getContext(), type.getWidth());
}

Type LLVMTypeConverter::convertFloatType(FloatType type) { return type; }

// Convert a `ComplexType` to an LLVM type. The result is a complex number
// struct with entries for the
//   1. real part and for the
//   2. imaginary part.
static constexpr unsigned kRealPosInComplexNumberStruct = 0;
static constexpr unsigned kImaginaryPosInComplexNumberStruct = 1;
Type LLVMTypeConverter::convertComplexType(ComplexType type) {
  auto elementType = convertType(type.getElementType());
  return LLVM::LLVMStructType::getLiteral(&getContext(),
                                          {elementType, elementType});
}

// Except for signatures, MLIR function types are converted into LLVM
// pointer-to-function types.
Type LLVMTypeConverter::convertFunctionType(FunctionType type) {
  SignatureConversion conversion(type.getNumInputs());
  Type converted =
      convertFunctionSignature(type, /*isVariadic=*/false, conversion);
  return LLVM::LLVMPointerType::get(converted);
}

// Function types are converted to LLVM Function types by recursively converting
// argument and result types.  If MLIR Function has zero results, the LLVM
// Function has one VoidType result.  If MLIR Function has more than one result,
// they are into an LLVM StructType in their order of appearance.
Type LLVMTypeConverter::convertFunctionSignature(
    FunctionType funcTy, bool isVariadic,
    LLVMTypeConverter::SignatureConversion &result) {
  // Select the argument converter depending on the calling convention.
  auto funcArgConverter = options.useBarePtrCallConv
                              ? barePtrFuncArgTypeConverter
                              : structFuncArgTypeConverter;
  // Convert argument types one by one and check for errors.
  for (auto &en : llvm::enumerate(funcTy.getInputs())) {
    Type type = en.value();
    SmallVector<Type, 8> converted;
    if (failed(funcArgConverter(*this, type, converted)))
      return {};
    result.addInputs(en.index(), converted);
  }

  SmallVector<Type, 8> argTypes;
  argTypes.reserve(llvm::size(result.getConvertedTypes()));
  for (Type type : result.getConvertedTypes())
    argTypes.push_back(unwrap(type));

  // If function does not return anything, create the void result type,
  // if it returns on element, convert it, otherwise pack the result types into
  // a struct.
  Type resultType = funcTy.getNumResults() == 0
                        ? LLVM::LLVMVoidType::get(&getContext())
                        : unwrap(packFunctionResults(funcTy.getResults()));
  if (!resultType)
    return {};
  return LLVM::LLVMFunctionType::get(resultType, argTypes, isVariadic);
}

/// Converts the function type to a C-compatible format, in particular using
/// pointers to memref descriptors for arguments.
Type LLVMTypeConverter::convertFunctionTypeCWrapper(FunctionType type) {
  SmallVector<Type, 4> inputs;

  for (Type t : type.getInputs()) {
    auto converted = convertType(t);
    if (!converted || !LLVM::isCompatibleType(converted))
      return {};
    if (t.isa<MemRefType, UnrankedMemRefType>())
      converted = LLVM::LLVMPointerType::get(converted);
    inputs.push_back(converted);
  }

  Type resultType = type.getNumResults() == 0
                        ? LLVM::LLVMVoidType::get(&getContext())
                        : unwrap(packFunctionResults(type.getResults()));
  if (!resultType)
    return {};

  return LLVM::LLVMFunctionType::get(resultType, inputs);
}

static constexpr unsigned kAllocatedPtrPosInMemRefDescriptor = 0;
static constexpr unsigned kAlignedPtrPosInMemRefDescriptor = 1;
static constexpr unsigned kOffsetPosInMemRefDescriptor = 2;
static constexpr unsigned kSizePosInMemRefDescriptor = 3;
static constexpr unsigned kStridePosInMemRefDescriptor = 4;

/// Convert a memref type into a list of LLVM IR types that will form the
/// memref descriptor. The result contains the following types:
///  1. The pointer to the allocated data buffer, followed by
///  2. The pointer to the aligned data buffer, followed by
///  3. A lowered `index`-type integer containing the distance between the
///  beginning of the buffer and the first element to be accessed through the
///  view, followed by
///  4. An array containing as many `index`-type integers as the rank of the
///  MemRef: the array represents the size, in number of elements, of the memref
///  along the given dimension. For constant MemRef dimensions, the
///  corresponding size entry is a constant whose runtime value must match the
///  static value, followed by
///  5. A second array containing as many `index`-type integers as the rank of
///  the MemRef: the second array represents the "stride" (in tensor abstraction
///  sense), i.e. the number of consecutive elements of the underlying buffer.
///  TODO: add assertions for the static cases.
///
///  If `unpackAggregates` is set to true, the arrays described in (4) and (5)
///  are expanded into individual index-type elements.
///
///  template <typename Elem, typename Index, size_t Rank>
///  struct {
///    Elem *allocatedPtr;
///    Elem *alignedPtr;
///    Index offset;
///    Index sizes[Rank]; // omitted when rank == 0
///    Index strides[Rank]; // omitted when rank == 0
///  };
SmallVector<Type, 5>
LLVMTypeConverter::getMemRefDescriptorFields(MemRefType type,
                                             bool unpackAggregates) {
  assert(isStrided(type) &&
         "Non-strided layout maps must have been normalized away");

  Type elementType = unwrap(convertType(type.getElementType()));
  if (!elementType)
    return {};
  auto ptrTy =
      LLVM::LLVMPointerType::get(elementType, type.getMemorySpaceAsInt());
  auto indexTy = getIndexType();

  SmallVector<Type, 5> results = {ptrTy, ptrTy, indexTy};
  auto rank = type.getRank();
  if (rank == 0)
    return results;

  if (unpackAggregates)
    results.insert(results.end(), 2 * rank, indexTy);
  else
    results.insert(results.end(), 2, LLVM::LLVMArrayType::get(indexTy, rank));
  return results;
}

/// Converts MemRefType to LLVMType. A MemRefType is converted to a struct that
/// packs the descriptor fields as defined by `getMemRefDescriptorFields`.
Type LLVMTypeConverter::convertMemRefType(MemRefType type) {
  // When converting a MemRefType to a struct with descriptor fields, do not
  // unpack the `sizes` and `strides` arrays.
  SmallVector<Type, 5> types =
      getMemRefDescriptorFields(type, /*unpackAggregates=*/false);
  return LLVM::LLVMStructType::getLiteral(&getContext(), types);
}

static constexpr unsigned kRankInUnrankedMemRefDescriptor = 0;
static constexpr unsigned kPtrInUnrankedMemRefDescriptor = 1;

/// Convert an unranked memref type into a list of non-aggregate LLVM IR types
/// that will form the unranked memref descriptor. In particular, the fields
/// for an unranked memref descriptor are:
/// 1. index-typed rank, the dynamic rank of this MemRef
/// 2. void* ptr, pointer to the static ranked MemRef descriptor. This will be
///    stack allocated (alloca) copy of a MemRef descriptor that got casted to
///    be unranked.
SmallVector<Type, 2> LLVMTypeConverter::getUnrankedMemRefDescriptorFields() {
  return {getIndexType(),
          LLVM::LLVMPointerType::get(IntegerType::get(&getContext(), 8))};
}

Type LLVMTypeConverter::convertUnrankedMemRefType(UnrankedMemRefType type) {
  return LLVM::LLVMStructType::getLiteral(&getContext(),
                                          getUnrankedMemRefDescriptorFields());
}

/// Convert a memref type to a bare pointer to the memref element type.
Type LLVMTypeConverter::convertMemRefToBarePtr(BaseMemRefType type) {
  if (type.isa<UnrankedMemRefType>())
    // Unranked memref is not supported in the bare pointer calling convention.
    return {};

  // Check that the memref has static shape, strides and offset. Otherwise, it
  // cannot be lowered to a bare pointer.
  auto memrefTy = type.cast<MemRefType>();
  if (!memrefTy.hasStaticShape())
    return {};

  int64_t offset = 0;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(memrefTy, strides, offset)))
    return {};

  for (int64_t stride : strides)
    if (ShapedType::isDynamicStrideOrOffset(stride))
      return {};

  if (ShapedType::isDynamicStrideOrOffset(offset))
    return {};

  Type elementType = unwrap(convertType(type.getElementType()));
  if (!elementType)
    return {};
  return LLVM::LLVMPointerType::get(elementType, type.getMemorySpaceAsInt());
}

/// Convert an n-D vector type to an LLVM vector type via (n-1)-D array type
/// when n > 1. For example, `vector<4 x f32>` remains as is while,
/// `vector<4x8x16xf32>` converts to `!llvm.array<4xarray<8 x vector<16xf32>>>`.
Type LLVMTypeConverter::convertVectorType(VectorType type) {
  auto elementType = unwrap(convertType(type.getElementType()));
  if (!elementType)
    return {};
  Type vectorType = VectorType::get(type.getShape().back(), elementType);
  assert(LLVM::isCompatibleVectorType(vectorType) &&
         "expected vector type compatible with the LLVM dialect");
  auto shape = type.getShape();
  for (int i = shape.size() - 2; i >= 0; --i)
    vectorType = LLVM::LLVMArrayType::get(vectorType, shape[i]);
  return vectorType;
}

/// Convert a type in the context of the default or bare pointer calling
/// convention. Calling convention sensitive types, such as MemRefType and
/// UnrankedMemRefType, are converted following the specific rules for the
/// calling convention. Calling convention independent types are converted
/// following the default LLVM type conversions.
Type LLVMTypeConverter::convertCallingConventionType(Type type) {
  if (options.useBarePtrCallConv)
    if (auto memrefTy = type.dyn_cast<BaseMemRefType>())
      return convertMemRefToBarePtr(memrefTy);

  return convertType(type);
}

/// Promote the bare pointers in 'values' that resulted from memrefs to
/// descriptors. 'stdTypes' holds they types of 'values' before the conversion
/// to the LLVM-IR dialect (i.e., MemRefType, or any other builtin type).
void LLVMTypeConverter::promoteBarePtrsToDescriptors(
    ConversionPatternRewriter &rewriter, Location loc, ArrayRef<Type> stdTypes,
    SmallVectorImpl<Value> &values) {
  assert(stdTypes.size() == values.size() &&
         "The number of types and values doesn't match");
  for (unsigned i = 0, end = values.size(); i < end; ++i)
    if (auto memrefTy = stdTypes[i].dyn_cast<MemRefType>())
      values[i] = MemRefDescriptor::fromStaticShape(rewriter, loc, *this,
                                                    memrefTy, values[i]);
}

ConvertToLLVMPattern::ConvertToLLVMPattern(StringRef rootOpName,
                                           MLIRContext *context,
                                           LLVMTypeConverter &typeConverter,
                                           PatternBenefit benefit)
    : ConversionPattern(rootOpName, benefit, typeConverter, context) {}

//===----------------------------------------------------------------------===//
// StructBuilder implementation
//===----------------------------------------------------------------------===//

StructBuilder::StructBuilder(Value v) : value(v), structType(v.getType()) {
  assert(value != nullptr && "value cannot be null");
  assert(LLVM::isCompatibleType(structType) && "expected llvm type");
}

Value StructBuilder::extractPtr(OpBuilder &builder, Location loc,
                                unsigned pos) {
  Type type = structType.cast<LLVM::LLVMStructType>().getBody()[pos];
  return builder.create<LLVM::ExtractValueOp>(loc, type, value,
                                              builder.getI64ArrayAttr(pos));
}

void StructBuilder::setPtr(OpBuilder &builder, Location loc, unsigned pos,
                           Value ptr) {
  value = builder.create<LLVM::InsertValueOp>(loc, structType, value, ptr,
                                              builder.getI64ArrayAttr(pos));
}

//===----------------------------------------------------------------------===//
// ComplexStructBuilder implementation
//===----------------------------------------------------------------------===//

ComplexStructBuilder ComplexStructBuilder::undef(OpBuilder &builder,
                                                 Location loc, Type type) {
  Value val = builder.create<LLVM::UndefOp>(loc, type);
  return ComplexStructBuilder(val);
}

void ComplexStructBuilder::setReal(OpBuilder &builder, Location loc,
                                   Value real) {
  setPtr(builder, loc, kRealPosInComplexNumberStruct, real);
}

Value ComplexStructBuilder::real(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kRealPosInComplexNumberStruct);
}

void ComplexStructBuilder::setImaginary(OpBuilder &builder, Location loc,
                                        Value imaginary) {
  setPtr(builder, loc, kImaginaryPosInComplexNumberStruct, imaginary);
}

Value ComplexStructBuilder::imaginary(OpBuilder &builder, Location loc) {
  return extractPtr(builder, loc, kImaginaryPosInComplexNumberStruct);
}

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
  assert(!MemRefType::isDynamicStrideOrOffset(offset) &&
         "expected static offset");
  assert(!llvm::any_of(strides, [](int64_t stride) {
    return MemRefType::isDynamicStrideOrOffset(stride);
  }) && "expected static strides");

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

  Type int32_type = unwrap(typeConverter.convertType(builder.getI32Type()));
  Value zero =
      createIndexAttrConstant(builder, loc, typeConverter.getIndexType(), 0);
  Value three = builder.create<LLVM::ConstantOp>(loc, int32_type,
                                                 builder.getI32IntegerAttr(3));
  return builder.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(indexTy),
                                     structPtr, ValueRange({zero, three}));
}

Value UnrankedMemRefDescriptor::size(OpBuilder &builder, Location loc,
                                     LLVMTypeConverter typeConverter,
                                     Value sizeBasePtr, Value index) {
  Type indexPtrTy = LLVM::LLVMPointerType::get(typeConverter.getIndexType());
  Value sizeStoreGep = builder.create<LLVM::GEPOp>(loc, indexPtrTy, sizeBasePtr,
                                                   ValueRange({index}));
  return builder.create<LLVM::LoadOp>(loc, sizeStoreGep);
}

void UnrankedMemRefDescriptor::setSize(OpBuilder &builder, Location loc,
                                       LLVMTypeConverter typeConverter,
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
                                       LLVMTypeConverter typeConverter,
                                       Value strideBasePtr, Value index,
                                       Value stride) {
  Type indexPtrTy = LLVM::LLVMPointerType::get(typeConverter.getIndexType());
  Value strideStoreGep = builder.create<LLVM::GEPOp>(
      loc, indexPtrTy, strideBasePtr, ValueRange({index}));
  return builder.create<LLVM::LoadOp>(loc, strideStoreGep);
}

void UnrankedMemRefDescriptor::setStride(OpBuilder &builder, Location loc,
                                         LLVMTypeConverter typeConverter,
                                         Value strideBasePtr, Value index,
                                         Value stride) {
  Type indexPtrTy = LLVM::LLVMPointerType::get(typeConverter.getIndexType());
  Value strideStoreGep = builder.create<LLVM::GEPOp>(
      loc, indexPtrTy, strideBasePtr, ValueRange({index}));
  builder.create<LLVM::StoreOp>(loc, stride, strideStoreGep);
}

LLVMTypeConverter *ConvertToLLVMPattern::getTypeConverter() const {
  return static_cast<LLVMTypeConverter *>(
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

Type ConvertToLLVMPattern::getVoidPtrType() const {
  return LLVM::LLVMPointerType::get(
      IntegerType::get(&getTypeConverter()->getContext(), 8));
}

Value ConvertToLLVMPattern::createIndexConstant(
    ConversionPatternRewriter &builder, Location loc, uint64_t value) const {
  return createIndexAttrConstant(builder, loc, getIndexType(), value);
}

Value ConvertToLLVMPattern::getStridedElementPtr(
    Location loc, MemRefType type, Value memRefDesc, ValueRange indices,
    ConversionPatternRewriter &rewriter) const {

  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto successStrides = getStridesAndOffset(type, strides, offset);
  assert(succeeded(successStrides) && "unexpected non-strided memref");
  (void)successStrides;

  MemRefDescriptor memRefDescriptor(memRefDesc);
  Value base = memRefDescriptor.alignedPtr(rewriter, loc);

  Value index;
  if (offset != 0) // Skip if offset is zero.
    index = MemRefType::isDynamicStrideOrOffset(offset)
                ? memRefDescriptor.offset(rewriter, loc)
                : createIndexConstant(rewriter, loc, offset);

  for (int i = 0, e = indices.size(); i < e; ++i) {
    Value increment = indices[i];
    if (strides[i] != 1) { // Skip if stride is 1.
      Value stride = MemRefType::isDynamicStrideOrOffset(strides[i])
                         ? memRefDescriptor.stride(rewriter, loc, i)
                         : createIndexConstant(rewriter, loc, strides[i]);
      increment = rewriter.create<LLVM::MulOp>(loc, increment, stride);
    }
    index =
        index ? rewriter.create<LLVM::AddOp>(loc, index, increment) : increment;
  }

  Type elementPtrType = memRefDescriptor.getElementPtrType();
  return index ? rewriter.create<LLVM::GEPOp>(loc, elementPtrType, base, index)
               : base;
}

// Check if the MemRefType `type` is supported by the lowering. We currently
// only support memrefs with identity maps.
bool ConvertToLLVMPattern::isConvertibleAndHasIdentityMaps(
    MemRefType type) const {
  if (!typeConverter->convertType(type.getElementType()))
    return false;
  return type.getAffineMaps().empty() ||
         llvm::all_of(type.getAffineMaps(),
                      [](AffineMap map) { return map.isIdentity(); });
}

Type ConvertToLLVMPattern::getElementPtrType(MemRefType type) const {
  auto elementType = type.getElementType();
  auto structElementType = unwrap(typeConverter->convertType(elementType));
  return LLVM::LLVMPointerType::get(structElementType,
                                    type.getMemorySpaceAsInt());
}

void ConvertToLLVMPattern::getMemRefDescriptorSizes(
    Location loc, MemRefType memRefType, ValueRange dynamicSizes,
    ConversionPatternRewriter &rewriter, SmallVectorImpl<Value> &sizes,
    SmallVectorImpl<Value> &strides, Value &sizeBytes) const {
  assert(isConvertibleAndHasIdentityMaps(memRefType) &&
         "layout maps must have been normalized away");
  assert(count(memRefType.getShape(), ShapedType::kDynamicSize) ==
             static_cast<ssize_t>(dynamicSizes.size()) &&
         "dynamicSizes size doesn't match dynamic sizes count in memref shape");

  sizes.reserve(memRefType.getRank());
  unsigned dynamicIndex = 0;
  for (int64_t size : memRefType.getShape()) {
    sizes.push_back(size == ShapedType::kDynamicSize
                        ? dynamicSizes[dynamicIndex++]
                        : createIndexConstant(rewriter, loc, size));
  }

  // Strides: iterate sizes in reverse order and multiply.
  int64_t stride = 1;
  Value runningStride = createIndexConstant(rewriter, loc, 1);
  strides.resize(memRefType.getRank());
  for (auto i = memRefType.getRank(); i-- > 0;) {
    strides[i] = runningStride;

    int64_t size = memRefType.getShape()[i];
    if (size == 0)
      continue;
    bool useSizeAsStride = stride == 1;
    if (size == ShapedType::kDynamicSize)
      stride = ShapedType::kDynamicSize;
    if (stride != ShapedType::kDynamicSize)
      stride *= size;

    if (useSizeAsStride)
      runningStride = sizes[i];
    else if (stride == ShapedType::kDynamicSize)
      runningStride =
          rewriter.create<LLVM::MulOp>(loc, runningStride, sizes[i]);
    else
      runningStride = createIndexConstant(rewriter, loc, stride);
  }

  // Buffer size in bytes.
  Type elementPtrType = getElementPtrType(memRefType);
  Value nullPtr = rewriter.create<LLVM::NullOp>(loc, elementPtrType);
  Value gepPtr = rewriter.create<LLVM::GEPOp>(
      loc, elementPtrType, ArrayRef<Value>{nullPtr, runningStride});
  sizeBytes = rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);
}

Value ConvertToLLVMPattern::getSizeInBytes(
    Location loc, Type type, ConversionPatternRewriter &rewriter) const {
  // Compute the size of an individual element. This emits the MLIR equivalent
  // of the following sizeof(...) implementation in LLVM IR:
  //   %0 = getelementptr %elementType* null, %indexType 1
  //   %1 = ptrtoint %elementType* %0 to %indexType
  // which is a common pattern of getting the size of a type in bytes.
  auto convertedPtrType =
      LLVM::LLVMPointerType::get(typeConverter->convertType(type));
  auto nullPtr = rewriter.create<LLVM::NullOp>(loc, convertedPtrType);
  auto gep = rewriter.create<LLVM::GEPOp>(
      loc, convertedPtrType,
      ArrayRef<Value>{nullPtr, createIndexConstant(rewriter, loc, 1)});
  return rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(), gep);
}

Value ConvertToLLVMPattern::getNumElements(
    Location loc, ArrayRef<Value> shape,
    ConversionPatternRewriter &rewriter) const {
  // Compute the total number of memref elements.
  Value numElements =
      shape.empty() ? createIndexConstant(rewriter, loc, 1) : shape.front();
  for (unsigned i = 1, e = shape.size(); i < e; ++i)
    numElements = rewriter.create<LLVM::MulOp>(loc, numElements, shape[i]);
  return numElements;
}

/// Creates and populates the memref descriptor struct given all its fields.
MemRefDescriptor ConvertToLLVMPattern::createMemRefDescriptor(
    Location loc, MemRefType memRefType, Value allocatedPtr, Value alignedPtr,
    ArrayRef<Value> sizes, ArrayRef<Value> strides,
    ConversionPatternRewriter &rewriter) const {
  auto structType = typeConverter->convertType(memRefType);
  auto memRefDescriptor = MemRefDescriptor::undef(rewriter, loc, structType);

  // Field 1: Allocated pointer, used for malloc/free.
  memRefDescriptor.setAllocatedPtr(rewriter, loc, allocatedPtr);

  // Field 2: Actual aligned pointer to payload.
  memRefDescriptor.setAlignedPtr(rewriter, loc, alignedPtr);

  // Field 3: Offset in aligned pointer.
  memRefDescriptor.setOffset(rewriter, loc,
                             createIndexConstant(rewriter, loc, 0));

  // Fields 4: Sizes.
  for (auto en : llvm::enumerate(sizes))
    memRefDescriptor.setSize(rewriter, loc, en.index(), en.value());

  // Field 5: Strides.
  for (auto en : llvm::enumerate(strides))
    memRefDescriptor.setStride(rewriter, loc, en.index(), en.value());

  return memRefDescriptor;
}

/// Only retain those attributes that are not constructed by
/// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
/// attributes.
static void filterFuncAttributes(ArrayRef<NamedAttribute> attrs,
                                 bool filterArgAttrs,
                                 SmallVectorImpl<NamedAttribute> &result) {
  for (const auto &attr : attrs) {
    if (attr.first == SymbolTable::getSymbolAttrName() ||
        attr.first == impl::getTypeAttrName() || attr.first == "std.varargs" ||
        (filterArgAttrs && impl::isArgAttrName(attr.first.strref())))
      continue;
    result.push_back(attr);
  }
}

/// Creates an auxiliary function with pointer-to-memref-descriptor-struct
/// arguments instead of unpacked arguments. This function can be called from C
/// by passing a pointer to a C struct corresponding to a memref descriptor.
/// Internally, the auxiliary function unpacks the descriptor into individual
/// components and forwards them to `newFuncOp`.
static void wrapForExternalCallers(OpBuilder &rewriter, Location loc,
                                   LLVMTypeConverter &typeConverter,
                                   FuncOp funcOp, LLVM::LLVMFuncOp newFuncOp) {
  auto type = funcOp.getType();
  SmallVector<NamedAttribute, 4> attributes;
  filterFuncAttributes(funcOp->getAttrs(), /*filterArgAttrs=*/false,
                       attributes);
  auto wrapperFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
      loc, llvm::formatv("_mlir_ciface_{0}", funcOp.getName()).str(),
      typeConverter.convertFunctionTypeCWrapper(type), LLVM::Linkage::External,
      attributes);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(wrapperFuncOp.addEntryBlock());

  SmallVector<Value, 8> args;
  for (auto &en : llvm::enumerate(type.getInputs())) {
    Value arg = wrapperFuncOp.getArgument(en.index());
    if (auto memrefType = en.value().dyn_cast<MemRefType>()) {
      Value loaded = rewriter.create<LLVM::LoadOp>(loc, arg);
      MemRefDescriptor::unpack(rewriter, loc, loaded, memrefType, args);
      continue;
    }
    if (en.value().isa<UnrankedMemRefType>()) {
      Value loaded = rewriter.create<LLVM::LoadOp>(loc, arg);
      UnrankedMemRefDescriptor::unpack(rewriter, loc, loaded, args);
      continue;
    }

    args.push_back(wrapperFuncOp.getArgument(en.index()));
  }
  auto call = rewriter.create<LLVM::CallOp>(loc, newFuncOp, args);
  rewriter.create<LLVM::ReturnOp>(loc, call.getResults());
}

/// Creates an auxiliary function with pointer-to-memref-descriptor-struct
/// arguments instead of unpacked arguments. Creates a body for the (external)
/// `newFuncOp` that allocates a memref descriptor on stack, packs the
/// individual arguments into this descriptor and passes a pointer to it into
/// the auxiliary function. This auxiliary external function is now compatible
/// with functions defined in C using pointers to C structs corresponding to a
/// memref descriptor.
static void wrapExternalFunction(OpBuilder &builder, Location loc,
                                 LLVMTypeConverter &typeConverter,
                                 FuncOp funcOp, LLVM::LLVMFuncOp newFuncOp) {
  OpBuilder::InsertionGuard guard(builder);

  Type wrapperType =
      typeConverter.convertFunctionTypeCWrapper(funcOp.getType());
  // This conversion can only fail if it could not convert one of the argument
  // types. But since it has been applies to a non-wrapper function before, it
  // should have failed earlier and not reach this point at all.
  assert(wrapperType && "unexpected type conversion failure");

  SmallVector<NamedAttribute, 4> attributes;
  filterFuncAttributes(funcOp->getAttrs(), /*filterArgAttrs=*/false,
                       attributes);

  // Create the auxiliary function.
  auto wrapperFunc = builder.create<LLVM::LLVMFuncOp>(
      loc, llvm::formatv("_mlir_ciface_{0}", funcOp.getName()).str(),
      wrapperType, LLVM::Linkage::External, attributes);

  builder.setInsertionPointToStart(newFuncOp.addEntryBlock());

  // Get a ValueRange containing arguments.
  FunctionType type = funcOp.getType();
  SmallVector<Value, 8> args;
  args.reserve(type.getNumInputs());
  ValueRange wrapperArgsRange(newFuncOp.getArguments());

  // Iterate over the inputs of the original function and pack values into
  // memref descriptors if the original type is a memref.
  for (auto &en : llvm::enumerate(type.getInputs())) {
    Value arg;
    int numToDrop = 1;
    auto memRefType = en.value().dyn_cast<MemRefType>();
    auto unrankedMemRefType = en.value().dyn_cast<UnrankedMemRefType>();
    if (memRefType || unrankedMemRefType) {
      numToDrop = memRefType
                      ? MemRefDescriptor::getNumUnpackedValues(memRefType)
                      : UnrankedMemRefDescriptor::getNumUnpackedValues();
      Value packed =
          memRefType
              ? MemRefDescriptor::pack(builder, loc, typeConverter, memRefType,
                                       wrapperArgsRange.take_front(numToDrop))
              : UnrankedMemRefDescriptor::pack(
                    builder, loc, typeConverter, unrankedMemRefType,
                    wrapperArgsRange.take_front(numToDrop));

      auto ptrTy = LLVM::LLVMPointerType::get(packed.getType());
      Value one = builder.create<LLVM::ConstantOp>(
          loc, typeConverter.convertType(builder.getIndexType()),
          builder.getIntegerAttr(builder.getIndexType(), 1));
      Value allocated =
          builder.create<LLVM::AllocaOp>(loc, ptrTy, one, /*alignment=*/0);
      builder.create<LLVM::StoreOp>(loc, packed, allocated);
      arg = allocated;
    } else {
      arg = wrapperArgsRange[0];
    }

    args.push_back(arg);
    wrapperArgsRange = wrapperArgsRange.drop_front(numToDrop);
  }
  assert(wrapperArgsRange.empty() && "did not map some of the arguments");

  auto call = builder.create<LLVM::CallOp>(loc, wrapperFunc, args);
  builder.create<LLVM::ReturnOp>(loc, call.getResults());
}

namespace {

struct FuncOpConversionBase : public ConvertOpToLLVMPattern<FuncOp> {
protected:
  using ConvertOpToLLVMPattern<FuncOp>::ConvertOpToLLVMPattern;

  // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
  // to this legalization pattern.
  LLVM::LLVMFuncOp
  convertFuncOpToLLVMFuncOp(FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) const {
    // Convert the original function arguments. They are converted using the
    // LLVMTypeConverter provided to this legalization pattern.
    auto varargsAttr = funcOp->getAttrOfType<BoolAttr>("std.varargs");
    TypeConverter::SignatureConversion result(funcOp.getNumArguments());
    auto llvmType = getTypeConverter()->convertFunctionSignature(
        funcOp.getType(), varargsAttr && varargsAttr.getValue(), result);
    if (!llvmType)
      return nullptr;

    // Propagate argument attributes to all converted arguments obtained after
    // converting a given original argument.
    SmallVector<NamedAttribute, 4> attributes;
    filterFuncAttributes(funcOp->getAttrs(), /*filterArgAttrs=*/true,
                         attributes);
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      auto attr = impl::getArgAttrDict(funcOp, i);
      if (!attr)
        continue;

      auto mapping = result.getInputMapping(i);
      assert(mapping.hasValue() && "unexpected deletion of function argument");

      SmallString<8> name;
      for (size_t j = 0; j < mapping->size; ++j) {
        impl::getArgAttrName(mapping->inputNo + j, name);
        attributes.push_back(rewriter.getNamedAttr(name, attr));
      }
    }

    // Create an LLVM function, use external linkage by default until MLIR
    // functions have linkage.
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getName(), llvmType, LLVM::Linkage::External,
        attributes);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &result)))
      return nullptr;

    return newFuncOp;
  }
};

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
static constexpr StringRef kEmitIfaceAttrName = "llvm.emit_c_interface";
struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(LLVMTypeConverter &converter)
      : FuncOpConversionBase(converter) {}

  LogicalResult
  matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (!newFuncOp)
      return failure();

    if (getTypeConverter()->getOptions().emitCWrappers ||
        funcOp->getAttrOfType<UnitAttr>(kEmitIfaceAttrName)) {
      if (newFuncOp.isExternal())
        wrapExternalFunction(rewriter, funcOp.getLoc(), *getTypeConverter(),
                             funcOp, newFuncOp);
      else
        wrapForExternalCallers(rewriter, funcOp.getLoc(), *getTypeConverter(),
                               funcOp, newFuncOp);
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

/// FuncOp legalization pattern that converts MemRef arguments to bare pointers
/// to the MemRef element type. This will impact the calling convention and ABI.
struct BarePtrFuncOpConversion : public FuncOpConversionBase {
  using FuncOpConversionBase::FuncOpConversionBase;

  LogicalResult
  matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Store the type of memref-typed arguments before the conversion so that we
    // can promote them to MemRef descriptor at the beginning of the function.
    SmallVector<Type, 8> oldArgTypes =
        llvm::to_vector<8>(funcOp.getType().getInputs());

    auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (!newFuncOp)
      return failure();
    if (newFuncOp.getBody().empty()) {
      rewriter.eraseOp(funcOp);
      return success();
    }

    // Promote bare pointers from memref arguments to memref descriptors at the
    // beginning of the function so that all the memrefs in the function have a
    // uniform representation.
    Block *entryBlock = &newFuncOp.getBody().front();
    auto blockArgs = entryBlock->getArguments();
    assert(blockArgs.size() == oldArgTypes.size() &&
           "The number of arguments and types doesn't match");

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);
    for (auto it : llvm::zip(blockArgs, oldArgTypes)) {
      BlockArgument arg = std::get<0>(it);
      Type argTy = std::get<1>(it);

      // Unranked memrefs are not supported in the bare pointer calling
      // convention. We should have bailed out before in the presence of
      // unranked memrefs.
      assert(!argTy.isa<UnrankedMemRefType>() &&
             "Unranked memref is not supported");
      auto memrefTy = argTy.dyn_cast<MemRefType>();
      if (!memrefTy)
        continue;

      // Replace barePtr with a placeholder (undef), promote barePtr to a ranked
      // or unranked memref descriptor and replace placeholder with the last
      // instruction of the memref descriptor.
      // TODO: The placeholder is needed to avoid replacing barePtr uses in the
      // MemRef descriptor instructions. We may want to have a utility in the
      // rewriter to properly handle this use case.
      Location loc = funcOp.getLoc();
      auto placeholder = rewriter.create<LLVM::UndefOp>(loc, memrefTy);
      rewriter.replaceUsesOfBlockArgument(arg, placeholder);

      Value desc = MemRefDescriptor::fromStaticShape(
          rewriter, loc, *getTypeConverter(), memrefTy, arg);
      rewriter.replaceOp(placeholder, {desc});
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

//////////////// Support for Lowering operations on n-D vectors ////////////////
// Helper struct to "unroll" operations on n-D vectors in terms of operations on
// 1-D LLVM vectors.
struct NDVectorTypeInfo {
  // LLVM array struct which encodes n-D vectors.
  Type llvmNDVectorTy;
  // LLVM vector type which encodes the inner 1-D vector type.
  Type llvm1DVectorTy;
  // Multiplicity of llvmNDVectorTy to llvm1DVectorTy.
  SmallVector<int64_t, 4> arraySizes;
};
} // namespace

// For >1-D vector types, extracts the necessary information to iterate over all
// 1-D subvectors in the underlying llrepresentation of the n-D vector
// Iterates on the llvm array type until we hit a non-array type (which is
// asserted to be an llvm vector type).
static NDVectorTypeInfo extractNDVectorTypeInfo(VectorType vectorType,
                                                LLVMTypeConverter &converter) {
  assert(vectorType.getRank() > 1 && "expected >1D vector type");
  NDVectorTypeInfo info;
  info.llvmNDVectorTy = converter.convertType(vectorType);
  if (!info.llvmNDVectorTy || !LLVM::isCompatibleType(info.llvmNDVectorTy)) {
    info.llvmNDVectorTy = nullptr;
    return info;
  }
  info.arraySizes.reserve(vectorType.getRank() - 1);
  auto llvmTy = info.llvmNDVectorTy;
  while (llvmTy.isa<LLVM::LLVMArrayType>()) {
    info.arraySizes.push_back(
        llvmTy.cast<LLVM::LLVMArrayType>().getNumElements());
    llvmTy = llvmTy.cast<LLVM::LLVMArrayType>().getElementType();
  }
  if (!LLVM::isCompatibleVectorType(llvmTy))
    return info;
  info.llvm1DVectorTy = llvmTy;
  return info;
}

// Express `linearIndex` in terms of coordinates of `basis`.
// Returns the empty vector when linearIndex is out of the range [0, P] where
// P is the product of all the basis coordinates.
//
// Prerequisites:
//   Basis is an array of nonnegative integers (signed type inherited from
//   vector shape type).
static SmallVector<int64_t, 4> getCoordinates(ArrayRef<int64_t> basis,
                                              unsigned linearIndex) {
  SmallVector<int64_t, 4> res;
  res.reserve(basis.size());
  for (unsigned basisElement : llvm::reverse(basis)) {
    res.push_back(linearIndex % basisElement);
    linearIndex = linearIndex / basisElement;
  }
  if (linearIndex > 0)
    return {};
  std::reverse(res.begin(), res.end());
  return res;
}

// Iterate of linear index, convert to coords space and insert splatted 1-D
// vector in each position.
template <typename Lambda>
void nDVectorIterate(const NDVectorTypeInfo &info, OpBuilder &builder,
                     Lambda fun) {
  unsigned ub = 1;
  for (auto s : info.arraySizes)
    ub *= s;
  for (unsigned linearIndex = 0; linearIndex < ub; ++linearIndex) {
    auto coords = getCoordinates(info.arraySizes, linearIndex);
    // Linear index is out of bounds, we are done.
    if (coords.empty())
      break;
    assert(coords.size() == info.arraySizes.size());
    auto position = builder.getI64ArrayAttr(coords);
    fun(position);
  }
}
////////////// End Support for Lowering operations on n-D vectors //////////////

/// Replaces the given operation "op" with a new operation of type "targetOp"
/// and given operands.
LogicalResult LLVM::detail::oneToOneRewrite(
    Operation *op, StringRef targetOp, ValueRange operands,
    LLVMTypeConverter &typeConverter, ConversionPatternRewriter &rewriter) {
  unsigned numResults = op->getNumResults();

  Type packedType;
  if (numResults != 0) {
    packedType = typeConverter.packFunctionResults(op->getResultTypes());
    if (!packedType)
      return failure();
  }

  // Create the operation through state since we don't know its C++ type.
  OperationState state(op->getLoc(), targetOp);
  state.addTypes(packedType);
  state.addOperands(operands);
  state.addAttributes(op->getAttrs());
  Operation *newOp = rewriter.createOperation(state);

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
    auto type = typeConverter.convertType(op->getResult(i).getType());
    results.push_back(rewriter.create<LLVM::ExtractValueOp>(
        op->getLoc(), type, newOp->getResult(0), rewriter.getI64ArrayAttr(i)));
  }
  rewriter.replaceOp(op, results);
  return success();
}

static LogicalResult handleMultidimensionalVectors(
    Operation *op, ValueRange operands, LLVMTypeConverter &typeConverter,
    std::function<Value(Type, ValueRange)> createOperand,
    ConversionPatternRewriter &rewriter) {
  auto operandNDVectorType = op->getOperand(0).getType().dyn_cast<VectorType>();
  auto resultNDVectorType = op->getResult(0).getType().dyn_cast<VectorType>();
  assert(operandNDVectorType && resultNDVectorType && "expected vector types");

  auto resultTypeInfo =
      extractNDVectorTypeInfo(resultNDVectorType, typeConverter);
  auto operandTypeInfo =
      extractNDVectorTypeInfo(operandNDVectorType, typeConverter);
  auto result1DVectorTy = resultTypeInfo.llvm1DVectorTy;
  auto operand1DVectorTy = operandTypeInfo.llvm1DVectorTy;
  auto resultNDVectoryTy = resultTypeInfo.llvmNDVectorTy;
  auto loc = op->getLoc();
  Value desc = rewriter.create<LLVM::UndefOp>(loc, resultNDVectoryTy);
  nDVectorIterate(resultTypeInfo, rewriter, [&](ArrayAttr position) {
    // For this unrolled `position` corresponding to the `linearIndex`^th
    // element, extract operand vectors
    SmallVector<Value, 4> extractedOperands;
    for (auto operand : operands)
      extractedOperands.push_back(rewriter.create<LLVM::ExtractValueOp>(
          loc, operand1DVectorTy, operand, position));
    Value newVal = createOperand(result1DVectorTy, extractedOperands);
    desc = rewriter.create<LLVM::InsertValueOp>(loc, resultNDVectoryTy, desc,
                                                newVal, position);
  });
  rewriter.replaceOp(op, desc);
  return success();
}

LogicalResult LLVM::detail::vectorOneToOneRewrite(
    Operation *op, StringRef targetOp, ValueRange operands,
    LLVMTypeConverter &typeConverter, ConversionPatternRewriter &rewriter) {
  assert(!operands.empty());

  // Cannot convert ops if their operands are not of LLVM type.
  if (!llvm::all_of(operands.getTypes(),
                    [](Type t) { return isCompatibleType(t); }))
    return failure();

  auto llvmNDVectorTy = operands[0].getType();
  if (!llvmNDVectorTy.isa<LLVM::LLVMArrayType>())
    return oneToOneRewrite(op, targetOp, operands, typeConverter, rewriter);

  auto callback = [op, targetOp, &rewriter](Type llvm1DVectorTy,
                                            ValueRange operands) {
    OperationState state(op->getLoc(), targetOp);
    state.addTypes(llvm1DVectorTy);
    state.addOperands(operands);
    state.addAttributes(op->getAttrs());
    return rewriter.createOperation(state)->getResult(0);
  };

  return handleMultidimensionalVectors(op, operands, typeConverter, callback,
                                       rewriter);
}

namespace {
// Straightforward lowerings.
using AbsFOpLowering = VectorConvertToLLVMPattern<AbsFOp, LLVM::FAbsOp>;
using AddFOpLowering = VectorConvertToLLVMPattern<AddFOp, LLVM::FAddOp>;
using AddIOpLowering = VectorConvertToLLVMPattern<AddIOp, LLVM::AddOp>;
using AndOpLowering = VectorConvertToLLVMPattern<AndOp, LLVM::AndOp>;
using CeilFOpLowering = VectorConvertToLLVMPattern<CeilFOp, LLVM::FCeilOp>;
using CopySignOpLowering =
    VectorConvertToLLVMPattern<CopySignOp, LLVM::CopySignOp>;
using CosOpLowering = VectorConvertToLLVMPattern<math::CosOp, LLVM::CosOp>;
using DivFOpLowering = VectorConvertToLLVMPattern<DivFOp, LLVM::FDivOp>;
using ExpOpLowering = VectorConvertToLLVMPattern<math::ExpOp, LLVM::ExpOp>;
using Exp2OpLowering = VectorConvertToLLVMPattern<math::Exp2Op, LLVM::Exp2Op>;
using FloorFOpLowering = VectorConvertToLLVMPattern<FloorFOp, LLVM::FFloorOp>;
using FmaFOpLowering = VectorConvertToLLVMPattern<FmaFOp, LLVM::FMAOp>;
using Log10OpLowering =
    VectorConvertToLLVMPattern<math::Log10Op, LLVM::Log10Op>;
using Log2OpLowering = VectorConvertToLLVMPattern<math::Log2Op, LLVM::Log2Op>;
using LogOpLowering = VectorConvertToLLVMPattern<math::LogOp, LLVM::LogOp>;
using MulFOpLowering = VectorConvertToLLVMPattern<MulFOp, LLVM::FMulOp>;
using MulIOpLowering = VectorConvertToLLVMPattern<MulIOp, LLVM::MulOp>;
using NegFOpLowering = VectorConvertToLLVMPattern<NegFOp, LLVM::FNegOp>;
using OrOpLowering = VectorConvertToLLVMPattern<OrOp, LLVM::OrOp>;
using PowFOpLowering = VectorConvertToLLVMPattern<math::PowFOp, LLVM::PowOp>;
using RemFOpLowering = VectorConvertToLLVMPattern<RemFOp, LLVM::FRemOp>;
using SelectOpLowering = OneToOneConvertToLLVMPattern<SelectOp, LLVM::SelectOp>;
using SignExtendIOpLowering =
    VectorConvertToLLVMPattern<SignExtendIOp, LLVM::SExtOp>;
using ShiftLeftOpLowering =
    OneToOneConvertToLLVMPattern<ShiftLeftOp, LLVM::ShlOp>;
using SignedDivIOpLowering =
    VectorConvertToLLVMPattern<SignedDivIOp, LLVM::SDivOp>;
using SignedRemIOpLowering =
    VectorConvertToLLVMPattern<SignedRemIOp, LLVM::SRemOp>;
using SignedShiftRightOpLowering =
    OneToOneConvertToLLVMPattern<SignedShiftRightOp, LLVM::AShrOp>;
using SinOpLowering = VectorConvertToLLVMPattern<math::SinOp, LLVM::SinOp>;
using SqrtOpLowering = VectorConvertToLLVMPattern<math::SqrtOp, LLVM::SqrtOp>;
using SubFOpLowering = VectorConvertToLLVMPattern<SubFOp, LLVM::FSubOp>;
using SubIOpLowering = VectorConvertToLLVMPattern<SubIOp, LLVM::SubOp>;
using UnsignedDivIOpLowering =
    VectorConvertToLLVMPattern<UnsignedDivIOp, LLVM::UDivOp>;
using UnsignedRemIOpLowering =
    VectorConvertToLLVMPattern<UnsignedRemIOp, LLVM::URemOp>;
using UnsignedShiftRightOpLowering =
    OneToOneConvertToLLVMPattern<UnsignedShiftRightOp, LLVM::LShrOp>;
using XOrOpLowering = VectorConvertToLLVMPattern<XOrOp, LLVM::XOrOp>;
using ZeroExtendIOpLowering =
    VectorConvertToLLVMPattern<ZeroExtendIOp, LLVM::ZExtOp>;

/// Lower `std.assert`. The default lowering calls the `abort` function if the
/// assertion is violated and has no effect otherwise. The failure message is
/// ignored by the default lowering but should be propagated by any custom
/// lowering.
struct AssertOpLowering : public ConvertOpToLLVMPattern<AssertOp> {
  using ConvertOpToLLVMPattern<AssertOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AssertOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    AssertOp::Adaptor transformed(operands);

    // Insert the `abort` declaration if necessary.
    auto module = op->getParentOfType<ModuleOp>();
    auto abortFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("abort");
    if (!abortFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto abortFuncTy = LLVM::LLVMFunctionType::get(getVoidType(), {});
      abortFunc = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                    "abort", abortFuncTy);
    }

    // Split block at `assert` operation.
    Block *opBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    Block *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

    // Generate IR to call `abort`.
    Block *failureBlock = rewriter.createBlock(opBlock->getParent());
    rewriter.create<LLVM::CallOp>(loc, abortFunc, llvm::None);
    rewriter.create<LLVM::UnreachableOp>(loc);

    // Generate assertion test.
    rewriter.setInsertionPointToEnd(opBlock);
    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
        op, transformed.arg(), continuationBlock, failureBlock);

    return success();
  }
};

struct ConstantOpLowering : public ConvertOpToLLVMPattern<ConstantOp> {
  using ConvertOpToLLVMPattern<ConstantOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // If constant refers to a function, convert it to "addressof".
    if (auto symbolRef = op.getValue().dyn_cast<FlatSymbolRefAttr>()) {
      auto type = typeConverter->convertType(op.getResult().getType());
      if (!type || !LLVM::isCompatibleType(type))
        return rewriter.notifyMatchFailure(op, "failed to convert result type");

      auto newOp = rewriter.create<LLVM::AddressOfOp>(op.getLoc(), type,
                                                      symbolRef.getValue());
      for (const NamedAttribute &attr : op->getAttrs()) {
        if (attr.first.strref() == "value")
          continue;
        newOp->setAttr(attr.first, attr.second);
      }
      rewriter.replaceOp(op, newOp->getResults());
      return success();
    }

    // Calling into other scopes (non-flat reference) is not supported in LLVM.
    if (op.getValue().isa<SymbolRefAttr>())
      return rewriter.notifyMatchFailure(
          op, "referring to a symbol outside of the current module");

    return LLVM::detail::oneToOneRewrite(
        op, LLVM::ConstantOp::getOperationName(), operands, *getTypeConverter(),
        rewriter);
  }
};

/// Lowering for AllocOp and AllocaOp.
struct AllocLikeOpLowering : public ConvertToLLVMPattern {
  using ConvertToLLVMPattern::createIndexConstant;
  using ConvertToLLVMPattern::getIndexType;
  using ConvertToLLVMPattern::getVoidPtrType;

  explicit AllocLikeOpLowering(StringRef opName, LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(opName, &converter.getContext(), converter) {}

protected:
  // Returns 'input' aligned up to 'alignment'. Computes
  // bumped = input + alignement - 1
  // aligned = bumped - bumped % alignment
  static Value createAligned(ConversionPatternRewriter &rewriter, Location loc,
                             Value input, Value alignment) {
    Value one = createIndexAttrConstant(rewriter, loc, alignment.getType(), 1);
    Value bump = rewriter.create<LLVM::SubOp>(loc, alignment, one);
    Value bumped = rewriter.create<LLVM::AddOp>(loc, input, bump);
    Value mod = rewriter.create<LLVM::URemOp>(loc, bumped, alignment);
    return rewriter.create<LLVM::SubOp>(loc, bumped, mod);
  }

  /// Allocates the underlying buffer. Returns the allocated pointer and the
  /// aligned pointer.
  virtual std::tuple<Value, Value>
  allocateBuffer(ConversionPatternRewriter &rewriter, Location loc,
                 Value sizeBytes, Operation *op) const = 0;

private:
  static MemRefType getMemRefResultType(Operation *op) {
    return op->getResult(0).getType().cast<MemRefType>();
  }

  LogicalResult match(Operation *op) const override {
    MemRefType memRefType = getMemRefResultType(op);
    return success(isConvertibleAndHasIdentityMaps(memRefType));
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
  void rewrite(Operation *op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override {
    MemRefType memRefType = getMemRefResultType(op);
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
  }
};

struct AllocOpLowering : public AllocLikeOpLowering {
  AllocOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLowering(AllocOp::getOperationName(), converter) {}

  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {
    // Heap allocations.
    AllocOp allocOp = cast<AllocOp>(op);
    MemRefType memRefType = allocOp.getType();

    Value alignment;
    if (auto alignmentAttr = allocOp.alignment()) {
      alignment = createIndexConstant(rewriter, loc, *alignmentAttr);
    } else if (!memRefType.getElementType().isSignlessIntOrIndexOrFloat()) {
      // In the case where no alignment is specified, we may want to override
      // `malloc's` behavior. `malloc` typically aligns at the size of the
      // biggest scalar on a target HW. For non-scalars, use the natural
      // alignment of the LLVM type given by the LLVM DataLayout.
      alignment = getSizeInBytes(loc, memRefType.getElementType(), rewriter);
    }

    if (alignment) {
      // Adjust the allocation size to consider alignment.
      sizeBytes = rewriter.create<LLVM::AddOp>(loc, sizeBytes, alignment);
    }

    // Allocate the underlying buffer and store a pointer to it in the MemRef
    // descriptor.
    Type elementPtrType = this->getElementPtrType(memRefType);
    auto allocFuncOp = LLVM::lookupOrCreateMallocFn(
        allocOp->getParentOfType<ModuleOp>(), getIndexType());
    auto results = createLLVMCall(rewriter, loc, allocFuncOp, {sizeBytes},
                                  getVoidPtrType());
    Value allocatedPtr =
        rewriter.create<LLVM::BitcastOp>(loc, elementPtrType, results[0]);

    Value alignedPtr = allocatedPtr;
    if (alignment) {
      auto intPtrType = getIntPtrType(memRefType.getMemorySpaceAsInt());
      // Compute the aligned type pointer.
      Value allocatedInt =
          rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, allocatedPtr);
      Value alignmentInt =
          createAligned(rewriter, loc, allocatedInt, alignment);
      alignedPtr =
          rewriter.create<LLVM::IntToPtrOp>(loc, elementPtrType, alignmentInt);
    }

    return std::make_tuple(allocatedPtr, alignedPtr);
  }
};

struct AlignedAllocOpLowering : public AllocLikeOpLowering {
  AlignedAllocOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLowering(AllocOp::getOperationName(), converter) {}

  /// Returns the memref's element size in bytes.
  // TODO: there are other places where this is used. Expose publicly?
  static unsigned getMemRefEltSizeInBytes(MemRefType memRefType) {
    auto elementType = memRefType.getElementType();

    unsigned sizeInBits;
    if (elementType.isIntOrFloat()) {
      sizeInBits = elementType.getIntOrFloatBitWidth();
    } else {
      auto vectorType = elementType.cast<VectorType>();
      sizeInBits =
          vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
    }
    return llvm::divideCeil(sizeInBits, 8);
  }

  /// Returns true if the memref size in bytes is known to be a multiple of
  /// factor.
  static bool isMemRefSizeMultipleOf(MemRefType type, uint64_t factor) {
    uint64_t sizeDivisor = getMemRefEltSizeInBytes(type);
    for (unsigned i = 0, e = type.getRank(); i < e; i++) {
      if (type.isDynamic(type.getDimSize(i)))
        continue;
      sizeDivisor = sizeDivisor * type.getDimSize(i);
    }
    return sizeDivisor % factor == 0;
  }

  /// Returns the alignment to be used for the allocation call itself.
  /// aligned_alloc requires the allocation size to be a power of two, and the
  /// allocation size to be a multiple of alignment,
  int64_t getAllocationAlignment(AllocOp allocOp) const {
    if (Optional<uint64_t> alignment = allocOp.alignment())
      return *alignment;

    // Whenever we don't have alignment set, we will use an alignment
    // consistent with the element type; since the allocation size has to be a
    // power of two, we will bump to the next power of two if it already isn't.
    auto eltSizeBytes = getMemRefEltSizeInBytes(allocOp.getType());
    return std::max(kMinAlignedAllocAlignment,
                    llvm::PowerOf2Ceil(eltSizeBytes));
  }

  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {
    // Heap allocations.
    AllocOp allocOp = cast<AllocOp>(op);
    MemRefType memRefType = allocOp.getType();
    int64_t alignment = getAllocationAlignment(allocOp);
    Value allocAlignment = createIndexConstant(rewriter, loc, alignment);

    // aligned_alloc requires size to be a multiple of alignment; we will pad
    // the size to the next multiple if necessary.
    if (!isMemRefSizeMultipleOf(memRefType, alignment))
      sizeBytes = createAligned(rewriter, loc, sizeBytes, allocAlignment);

    Type elementPtrType = this->getElementPtrType(memRefType);
    auto allocFuncOp = LLVM::lookupOrCreateAlignedAllocFn(
        allocOp->getParentOfType<ModuleOp>(), getIndexType());
    auto results =
        createLLVMCall(rewriter, loc, allocFuncOp, {allocAlignment, sizeBytes},
                       getVoidPtrType());
    Value allocatedPtr =
        rewriter.create<LLVM::BitcastOp>(loc, elementPtrType, results[0]);

    return std::make_tuple(allocatedPtr, allocatedPtr);
  }

  /// The minimum alignment to use with aligned_alloc (has to be a power of 2).
  static constexpr uint64_t kMinAlignedAllocAlignment = 16UL;
};

// Out of line definition, required till C++17.
constexpr uint64_t AlignedAllocOpLowering::kMinAlignedAllocAlignment;

struct AllocaOpLowering : public AllocLikeOpLowering {
  AllocaOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLowering(AllocaOp::getOperationName(), converter) {}

  /// Allocates the underlying buffer using the right call. `allocatedBytePtr`
  /// is set to null for stack allocations. `accessAlignment` is set if
  /// alignment is needed post allocation (for eg. in conjunction with malloc).
  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {

    // With alloca, one gets a pointer to the element type right away.
    // For stack allocations.
    auto allocaOp = cast<AllocaOp>(op);
    auto elementPtrType = this->getElementPtrType(allocaOp.getType());

    auto allocatedElementPtr = rewriter.create<LLVM::AllocaOp>(
        loc, elementPtrType, sizeBytes,
        allocaOp.alignment() ? *allocaOp.alignment() : 0);

    return std::make_tuple(allocatedElementPtr, allocatedElementPtr);
  }
};

/// Copies the shaped descriptor part to (if `toDynamic` is set) or from
/// (otherwise) the dynamically allocated memory for any operands that were
/// unranked descriptors originally.
static LogicalResult copyUnrankedDescriptors(OpBuilder &builder, Location loc,
                                             LLVMTypeConverter &typeConverter,
                                             TypeRange origTypes,
                                             SmallVectorImpl<Value> &operands,
                                             bool toDynamic) {
  assert(origTypes.size() == operands.size() &&
         "expected as may original types as operands");

  // Find operands of unranked memref type and store them.
  SmallVector<UnrankedMemRefDescriptor, 4> unrankedMemrefs;
  for (unsigned i = 0, e = operands.size(); i < e; ++i)
    if (origTypes[i].isa<UnrankedMemRefType>())
      unrankedMemrefs.emplace_back(operands[i]);

  if (unrankedMemrefs.empty())
    return success();

  // Compute allocation sizes.
  SmallVector<Value, 4> sizes;
  UnrankedMemRefDescriptor::computeSizes(builder, loc, typeConverter,
                                         unrankedMemrefs, sizes);

  // Get frequently used types.
  MLIRContext *context = builder.getContext();
  Type voidPtrType = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  auto i1Type = IntegerType::get(context, 1);
  Type indexType = typeConverter.getIndexType();

  // Find the malloc and free, or declare them if necessary.
  auto module = builder.getInsertionPoint()->getParentOfType<ModuleOp>();
  LLVM::LLVMFuncOp freeFunc, mallocFunc;
  if (toDynamic)
    mallocFunc = LLVM::lookupOrCreateMallocFn(module, indexType);
  if (!toDynamic)
    freeFunc = LLVM::lookupOrCreateFreeFn(module);

  // Initialize shared constants.
  Value zero =
      builder.create<LLVM::ConstantOp>(loc, i1Type, builder.getBoolAttr(false));

  unsigned unrankedMemrefPos = 0;
  for (unsigned i = 0, e = operands.size(); i < e; ++i) {
    Type type = origTypes[i];
    if (!type.isa<UnrankedMemRefType>())
      continue;
    Value allocationSize = sizes[unrankedMemrefPos++];
    UnrankedMemRefDescriptor desc(operands[i]);

    // Allocate memory, copy, and free the source if necessary.
    Value memory =
        toDynamic
            ? builder.create<LLVM::CallOp>(loc, mallocFunc, allocationSize)
                  .getResult(0)
            : builder.create<LLVM::AllocaOp>(loc, voidPtrType, allocationSize,
                                             /*alignment=*/0);

    Value source = desc.memRefDescPtr(builder, loc);
    builder.create<LLVM::MemcpyOp>(loc, memory, source, allocationSize, zero);
    if (!toDynamic)
      builder.create<LLVM::CallOp>(loc, freeFunc, source);

    // Create a new descriptor. The same descriptor can be returned multiple
    // times, attempting to modify its pointer can lead to memory leaks
    // (allocated twice and overwritten) or double frees (the caller does not
    // know if the descriptor points to the same memory).
    Type descriptorType = typeConverter.convertType(type);
    if (!descriptorType)
      return failure();
    auto updatedDesc =
        UnrankedMemRefDescriptor::undef(builder, loc, descriptorType);
    Value rank = desc.rank(builder, loc);
    updatedDesc.setRank(builder, loc, rank);
    updatedDesc.setMemRefDescPtr(builder, loc, memory);

    operands[i] = updatedDesc;
  }

  return success();
}

// A CallOp automatically promotes MemRefType to a sequence of alloca/store and
// passes the pointer to the MemRef across function boundaries.
template <typename CallOpType>
struct CallOpInterfaceLowering : public ConvertOpToLLVMPattern<CallOpType> {
  using ConvertOpToLLVMPattern<CallOpType>::ConvertOpToLLVMPattern;
  using Super = CallOpInterfaceLowering<CallOpType>;
  using Base = ConvertOpToLLVMPattern<CallOpType>;

  LogicalResult
  matchAndRewrite(CallOpType callOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename CallOpType::Adaptor transformed(operands);

    // Pack the result types into a struct.
    Type packedResult = nullptr;
    unsigned numResults = callOp.getNumResults();
    auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());

    if (numResults != 0) {
      if (!(packedResult =
                this->getTypeConverter()->packFunctionResults(resultTypes)))
        return failure();
    }

    auto promoted = this->getTypeConverter()->promoteOperands(
        callOp.getLoc(), /*opOperands=*/callOp->getOperands(), operands,
        rewriter);
    auto newOp = rewriter.create<LLVM::CallOp>(
        callOp.getLoc(), packedResult ? TypeRange(packedResult) : TypeRange(),
        promoted, callOp->getAttrs());

    SmallVector<Value, 4> results;
    if (numResults < 2) {
      // If < 2 results, packing did not do anything and we can just return.
      results.append(newOp.result_begin(), newOp.result_end());
    } else {
      // Otherwise, it had been converted to an operation producing a structure.
      // Extract individual results from the structure and return them as list.
      results.reserve(numResults);
      for (unsigned i = 0; i < numResults; ++i) {
        auto type =
            this->typeConverter->convertType(callOp.getResult(i).getType());
        results.push_back(rewriter.create<LLVM::ExtractValueOp>(
            callOp.getLoc(), type, newOp->getResult(0),
            rewriter.getI64ArrayAttr(i)));
      }
    }

    if (this->getTypeConverter()->getOptions().useBarePtrCallConv) {
      // For the bare-ptr calling convention, promote memref results to
      // descriptors.
      assert(results.size() == resultTypes.size() &&
             "The number of arguments and types doesn't match");
      this->getTypeConverter()->promoteBarePtrsToDescriptors(
          rewriter, callOp.getLoc(), resultTypes, results);
    } else if (failed(copyUnrankedDescriptors(rewriter, callOp.getLoc(),
                                              *this->getTypeConverter(),
                                              resultTypes, results,
                                              /*toDynamic=*/false))) {
      return failure();
    }

    rewriter.replaceOp(callOp, results);
    return success();
  }
};

struct CallOpLowering : public CallOpInterfaceLowering<CallOp> {
  using Super::Super;
};

struct CallIndirectOpLowering : public CallOpInterfaceLowering<CallIndirectOp> {
  using Super::Super;
};

// A `dealloc` is converted into a call to `free` on the underlying data buffer.
// The memref descriptor being an SSA value, there is no need to clean it up
// in any way.
struct DeallocOpLowering : public ConvertOpToLLVMPattern<DeallocOp> {
  using ConvertOpToLLVMPattern<DeallocOp>::ConvertOpToLLVMPattern;

  explicit DeallocOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<DeallocOp>(converter) {}

  LogicalResult
  matchAndRewrite(DeallocOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 1 && "dealloc takes one operand");
    DeallocOp::Adaptor transformed(operands);

    // Insert the `free` declaration if it is not already present.
    auto freeFunc = LLVM::lookupOrCreateFreeFn(op->getParentOfType<ModuleOp>());
    MemRefDescriptor memref(transformed.memref());
    Value casted = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(), getVoidPtrType(),
        memref.allocatedPtr(rewriter, op.getLoc()));
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange(), rewriter.getSymbolRefAttr(freeFunc), casted);
    return success();
  }
};

/// Returns the LLVM type of the global variable given the memref type `type`.
static Type convertGlobalMemrefTypeToLLVM(MemRefType type,
                                          LLVMTypeConverter &typeConverter) {
  // LLVM type for a global memref will be a multi-dimension array. For
  // declarations or uninitialized global memrefs, we can potentially flatten
  // this to a 1D array. However, for global_memref's with an initial value,
  // we do not intend to flatten the ElementsAttribute when going from std ->
  // LLVM dialect, so the LLVM type needs to me a multi-dimension array.
  Type elementType = unwrap(typeConverter.convertType(type.getElementType()));
  Type arrayTy = elementType;
  // Shape has the outermost dim at index 0, so need to walk it backwards
  for (int64_t dim : llvm::reverse(type.getShape()))
    arrayTy = LLVM::LLVMArrayType::get(arrayTy, dim);
  return arrayTy;
}

/// GlobalMemrefOp is lowered to a LLVM Global Variable.
struct GlobalMemrefOpLowering : public ConvertOpToLLVMPattern<GlobalMemrefOp> {
  using ConvertOpToLLVMPattern<GlobalMemrefOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GlobalMemrefOp global, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = global.type().cast<MemRefType>();
    if (!isConvertibleAndHasIdentityMaps(type))
      return failure();

    Type arrayTy = convertGlobalMemrefTypeToLLVM(type, *getTypeConverter());

    LLVM::Linkage linkage =
        global.isPublic() ? LLVM::Linkage::External : LLVM::Linkage::Private;

    Attribute initialValue = nullptr;
    if (!global.isExternal() && !global.isUninitialized()) {
      auto elementsAttr = global.initial_value()->cast<ElementsAttr>();
      initialValue = elementsAttr;

      // For scalar memrefs, the global variable created is of the element type,
      // so unpack the elements attribute to extract the value.
      if (type.getRank() == 0)
        initialValue = elementsAttr.getValue({});
    }

    rewriter.replaceOpWithNewOp<LLVM::GlobalOp>(
        global, arrayTy, global.constant(), linkage, global.sym_name(),
        initialValue, type.getMemorySpaceAsInt());
    return success();
  }
};

/// GetGlobalMemrefOp is lowered into a Memref descriptor with the pointer to
/// the first element stashed into the descriptor. This reuses
/// `AllocLikeOpLowering` to reuse the Memref descriptor construction.
struct GetGlobalMemrefOpLowering : public AllocLikeOpLowering {
  GetGlobalMemrefOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLowering(GetGlobalMemrefOp::getOperationName(), converter) {}

  /// Buffer "allocation" for get_global_memref op is getting the address of
  /// the global variable referenced.
  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {
    auto getGlobalOp = cast<GetGlobalMemrefOp>(op);
    MemRefType type = getGlobalOp.result().getType().cast<MemRefType>();
    unsigned memSpace = type.getMemorySpaceAsInt();

    Type arrayTy = convertGlobalMemrefTypeToLLVM(type, *getTypeConverter());
    auto addressOf = rewriter.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(arrayTy, memSpace), getGlobalOp.name());

    // Get the address of the first element in the array by creating a GEP with
    // the address of the GV as the base, and (rank + 1) number of 0 indices.
    Type elementType =
        unwrap(typeConverter->convertType(type.getElementType()));
    Type elementPtrType = LLVM::LLVMPointerType::get(elementType, memSpace);

    SmallVector<Value, 4> operands = {addressOf};
    operands.insert(operands.end(), type.getRank() + 1,
                    createIndexConstant(rewriter, loc, 0));
    auto gep = rewriter.create<LLVM::GEPOp>(loc, elementPtrType, operands);

    // We do not expect the memref obtained using `get_global_memref` to be
    // ever deallocated. Set the allocated pointer to be known bad value to
    // help debug if that ever happens.
    auto intPtrType = getIntPtrType(memSpace);
    Value deadBeefConst =
        createIndexAttrConstant(rewriter, op->getLoc(), intPtrType, 0xdeadbeef);
    auto deadBeefPtr =
        rewriter.create<LLVM::IntToPtrOp>(loc, elementPtrType, deadBeefConst);

    // Both allocated and aligned pointers are same. We could potentially stash
    // a nullptr for the allocated pointer since we do not expect any dealloc.
    return std::make_tuple(deadBeefPtr, gep);
  }
};

// A `rsqrt` is converted into `1 / sqrt`.
struct RsqrtOpLowering : public ConvertOpToLLVMPattern<math::RsqrtOp> {
  using ConvertOpToLLVMPattern<math::RsqrtOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::RsqrtOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    math::RsqrtOp::Adaptor transformed(operands);
    auto operandType = transformed.operand().getType();

    if (!operandType || !LLVM::isCompatibleType(operandType))
      return failure();

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto floatType = getElementTypeOrSelf(resultType).cast<FloatType>();
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);

    if (!operandType.isa<LLVM::LLVMArrayType>()) {
      LLVM::ConstantOp one;
      if (LLVM::isCompatibleVectorType(operandType)) {
        one = rewriter.create<LLVM::ConstantOp>(
            loc, operandType,
            SplatElementsAttr::get(resultType.cast<ShapedType>(), floatOne));
      } else {
        one = rewriter.create<LLVM::ConstantOp>(loc, operandType, floatOne);
      }
      auto sqrt = rewriter.create<LLVM::SqrtOp>(loc, transformed.operand());
      rewriter.replaceOpWithNewOp<LLVM::FDivOp>(op, operandType, one, sqrt);
      return success();
    }

    auto vectorType = resultType.dyn_cast<VectorType>();
    if (!vectorType)
      return failure();

    return handleMultidimensionalVectors(
        op.getOperation(), operands, *getTypeConverter(),
        [&](Type llvm1DVectorTy, ValueRange operands) {
          auto splatAttr = SplatElementsAttr::get(
              mlir::VectorType::get(
                  {LLVM::getVectorNumElements(llvm1DVectorTy).getFixedValue()},
                  floatType),
              floatOne);
          auto one =
              rewriter.create<LLVM::ConstantOp>(loc, llvm1DVectorTy, splatAttr);
          auto sqrt =
              rewriter.create<LLVM::SqrtOp>(loc, llvm1DVectorTy, operands[0]);
          return rewriter.create<LLVM::FDivOp>(loc, llvm1DVectorTy, one, sqrt);
        },
        rewriter);
  }
};

struct MemRefCastOpLowering : public ConvertOpToLLVMPattern<MemRefCastOp> {
  using ConvertOpToLLVMPattern<MemRefCastOp>::ConvertOpToLLVMPattern;

  LogicalResult match(MemRefCastOp memRefCastOp) const override {
    Type srcType = memRefCastOp.getOperand().getType();
    Type dstType = memRefCastOp.getType();

    // MemRefCastOp reduce to bitcast in the ranked MemRef case and can be used
    // for type erasure. For now they must preserve underlying element type and
    // require source and result type to have the same rank. Therefore, perform
    // a sanity check that the underlying structs are the same. Once op
    // semantics are relaxed we can revisit.
    if (srcType.isa<MemRefType>() && dstType.isa<MemRefType>())
      return success(typeConverter->convertType(srcType) ==
                     typeConverter->convertType(dstType));

    // At least one of the operands is unranked type
    assert(srcType.isa<UnrankedMemRefType>() ||
           dstType.isa<UnrankedMemRefType>());

    // Unranked to unranked cast is disallowed
    return !(srcType.isa<UnrankedMemRefType>() &&
             dstType.isa<UnrankedMemRefType>())
               ? success()
               : failure();
  }

  void rewrite(MemRefCastOp memRefCastOp, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override {
    MemRefCastOp::Adaptor transformed(operands);

    auto srcType = memRefCastOp.getOperand().getType();
    auto dstType = memRefCastOp.getType();
    auto targetStructType = typeConverter->convertType(memRefCastOp.getType());
    auto loc = memRefCastOp.getLoc();

    // For ranked/ranked case, just keep the original descriptor.
    if (srcType.isa<MemRefType>() && dstType.isa<MemRefType>())
      return rewriter.replaceOp(memRefCastOp, {transformed.source()});

    if (srcType.isa<MemRefType>() && dstType.isa<UnrankedMemRefType>()) {
      // Casting ranked to unranked memref type
      // Set the rank in the destination from the memref type
      // Allocate space on the stack and copy the src memref descriptor
      // Set the ptr in the destination to the stack space
      auto srcMemRefType = srcType.cast<MemRefType>();
      int64_t rank = srcMemRefType.getRank();
      // ptr = AllocaOp sizeof(MemRefDescriptor)
      auto ptr = getTypeConverter()->promoteOneMemRefDescriptor(
          loc, transformed.source(), rewriter);
      // voidptr = BitCastOp srcType* to void*
      auto voidPtr =
          rewriter.create<LLVM::BitcastOp>(loc, getVoidPtrType(), ptr)
              .getResult();
      // rank = ConstantOp srcRank
      auto rankVal = rewriter.create<LLVM::ConstantOp>(
          loc, typeConverter->convertType(rewriter.getIntegerType(64)),
          rewriter.getI64IntegerAttr(rank));
      // undef = UndefOp
      UnrankedMemRefDescriptor memRefDesc =
          UnrankedMemRefDescriptor::undef(rewriter, loc, targetStructType);
      // d1 = InsertValueOp undef, rank, 0
      memRefDesc.setRank(rewriter, loc, rankVal);
      // d2 = InsertValueOp d1, voidptr, 1
      memRefDesc.setMemRefDescPtr(rewriter, loc, voidPtr);
      rewriter.replaceOp(memRefCastOp, (Value)memRefDesc);

    } else if (srcType.isa<UnrankedMemRefType>() && dstType.isa<MemRefType>()) {
      // Casting from unranked type to ranked.
      // The operation is assumed to be doing a correct cast. If the destination
      // type mismatches the unranked the type, it is undefined behavior.
      UnrankedMemRefDescriptor memRefDesc(transformed.source());
      // ptr = ExtractValueOp src, 1
      auto ptr = memRefDesc.memRefDescPtr(rewriter, loc);
      // castPtr = BitCastOp i8* to structTy*
      auto castPtr =
          rewriter
              .create<LLVM::BitcastOp>(
                  loc, LLVM::LLVMPointerType::get(targetStructType), ptr)
              .getResult();
      // struct = LoadOp castPtr
      auto loadOp = rewriter.create<LLVM::LoadOp>(loc, castPtr);
      rewriter.replaceOp(memRefCastOp, loadOp.getResult());
    } else {
      llvm_unreachable("Unsupported unranked memref to unranked memref cast");
    }
  }
};

/// Extracts allocated, aligned pointers and offset from a ranked or unranked
/// memref type. In unranked case, the fields are extracted from the underlying
/// ranked descriptor.
static void extractPointersAndOffset(Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     LLVMTypeConverter &typeConverter,
                                     Value originalOperand,
                                     Value convertedOperand,
                                     Value *allocatedPtr, Value *alignedPtr,
                                     Value *offset = nullptr) {
  Type operandType = originalOperand.getType();
  if (operandType.isa<MemRefType>()) {
    MemRefDescriptor desc(convertedOperand);
    *allocatedPtr = desc.allocatedPtr(rewriter, loc);
    *alignedPtr = desc.alignedPtr(rewriter, loc);
    if (offset != nullptr)
      *offset = desc.offset(rewriter, loc);
    return;
  }

  unsigned memorySpace =
      operandType.cast<UnrankedMemRefType>().getMemorySpaceAsInt();
  Type elementType = operandType.cast<UnrankedMemRefType>().getElementType();
  Type llvmElementType = unwrap(typeConverter.convertType(elementType));
  Type elementPtrPtrType = LLVM::LLVMPointerType::get(
      LLVM::LLVMPointerType::get(llvmElementType, memorySpace));

  // Extract pointer to the underlying ranked memref descriptor and cast it to
  // ElemType**.
  UnrankedMemRefDescriptor unrankedDesc(convertedOperand);
  Value underlyingDescPtr = unrankedDesc.memRefDescPtr(rewriter, loc);

  *allocatedPtr = UnrankedMemRefDescriptor::allocatedPtr(
      rewriter, loc, underlyingDescPtr, elementPtrPtrType);
  *alignedPtr = UnrankedMemRefDescriptor::alignedPtr(
      rewriter, loc, typeConverter, underlyingDescPtr, elementPtrPtrType);
  if (offset != nullptr) {
    *offset = UnrankedMemRefDescriptor::offset(
        rewriter, loc, typeConverter, underlyingDescPtr, elementPtrPtrType);
  }
}

struct MemRefReinterpretCastOpLowering
    : public ConvertOpToLLVMPattern<MemRefReinterpretCastOp> {
  using ConvertOpToLLVMPattern<MemRefReinterpretCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MemRefReinterpretCastOp castOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefReinterpretCastOp::Adaptor adaptor(operands,
                                             castOp->getAttrDictionary());
    Type srcType = castOp.source().getType();

    Value descriptor;
    if (failed(convertSourceMemRefToDescriptor(rewriter, srcType, castOp,
                                               adaptor, &descriptor)))
      return failure();
    rewriter.replaceOp(castOp, {descriptor});
    return success();
  }

private:
  LogicalResult
  convertSourceMemRefToDescriptor(ConversionPatternRewriter &rewriter,
                                  Type srcType, MemRefReinterpretCastOp castOp,
                                  MemRefReinterpretCastOp::Adaptor adaptor,
                                  Value *descriptor) const {
    MemRefType targetMemRefType =
        castOp.getResult().getType().cast<MemRefType>();
    auto llvmTargetDescriptorTy = typeConverter->convertType(targetMemRefType)
                                      .dyn_cast_or_null<LLVM::LLVMStructType>();
    if (!llvmTargetDescriptorTy)
      return failure();

    // Create descriptor.
    Location loc = castOp.getLoc();
    auto desc = MemRefDescriptor::undef(rewriter, loc, llvmTargetDescriptorTy);

    // Set allocated and aligned pointers.
    Value allocatedPtr, alignedPtr;
    extractPointersAndOffset(loc, rewriter, *getTypeConverter(),
                             castOp.source(), adaptor.source(), &allocatedPtr,
                             &alignedPtr);
    desc.setAllocatedPtr(rewriter, loc, allocatedPtr);
    desc.setAlignedPtr(rewriter, loc, alignedPtr);

    // Set offset.
    if (castOp.isDynamicOffset(0))
      desc.setOffset(rewriter, loc, adaptor.offsets()[0]);
    else
      desc.setConstantOffset(rewriter, loc, castOp.getStaticOffset(0));

    // Set sizes and strides.
    unsigned dynSizeId = 0;
    unsigned dynStrideId = 0;
    for (unsigned i = 0, e = targetMemRefType.getRank(); i < e; ++i) {
      if (castOp.isDynamicSize(i))
        desc.setSize(rewriter, loc, i, adaptor.sizes()[dynSizeId++]);
      else
        desc.setConstantSize(rewriter, loc, i, castOp.getStaticSize(i));

      if (castOp.isDynamicStride(i))
        desc.setStride(rewriter, loc, i, adaptor.strides()[dynStrideId++]);
      else
        desc.setConstantStride(rewriter, loc, i, castOp.getStaticStride(i));
    }
    *descriptor = desc;
    return success();
  }
};

struct MemRefReshapeOpLowering
    : public ConvertOpToLLVMPattern<MemRefReshapeOp> {
  using ConvertOpToLLVMPattern<MemRefReshapeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MemRefReshapeOp reshapeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *op = reshapeOp.getOperation();
    MemRefReshapeOp::Adaptor adaptor(operands, op->getAttrDictionary());
    Type srcType = reshapeOp.source().getType();

    Value descriptor;
    if (failed(convertSourceMemRefToDescriptor(rewriter, srcType, reshapeOp,
                                               adaptor, &descriptor)))
      return failure();
    rewriter.replaceOp(op, {descriptor});
    return success();
  }

private:
  LogicalResult
  convertSourceMemRefToDescriptor(ConversionPatternRewriter &rewriter,
                                  Type srcType, MemRefReshapeOp reshapeOp,
                                  MemRefReshapeOp::Adaptor adaptor,
                                  Value *descriptor) const {
    // Conversion for statically-known shape args is performed via
    // `memref_reinterpret_cast`.
    auto shapeMemRefType = reshapeOp.shape().getType().cast<MemRefType>();
    if (shapeMemRefType.hasStaticShape())
      return failure();

    // The shape is a rank-1 tensor with unknown length.
    Location loc = reshapeOp.getLoc();
    MemRefDescriptor shapeDesc(adaptor.shape());
    Value resultRank = shapeDesc.size(rewriter, loc, 0);

    // Extract address space and element type.
    auto targetType =
        reshapeOp.getResult().getType().cast<UnrankedMemRefType>();
    unsigned addressSpace = targetType.getMemorySpaceAsInt();
    Type elementType = targetType.getElementType();

    // Create the unranked memref descriptor that holds the ranked one. The
    // inner descriptor is allocated on stack.
    auto targetDesc = UnrankedMemRefDescriptor::undef(
        rewriter, loc, unwrap(typeConverter->convertType(targetType)));
    targetDesc.setRank(rewriter, loc, resultRank);
    SmallVector<Value, 4> sizes;
    UnrankedMemRefDescriptor::computeSizes(rewriter, loc, *getTypeConverter(),
                                           targetDesc, sizes);
    Value underlyingDescPtr = rewriter.create<LLVM::AllocaOp>(
        loc, getVoidPtrType(), sizes.front(), llvm::None);
    targetDesc.setMemRefDescPtr(rewriter, loc, underlyingDescPtr);

    // Extract pointers and offset from the source memref.
    Value allocatedPtr, alignedPtr, offset;
    extractPointersAndOffset(loc, rewriter, *getTypeConverter(),
                             reshapeOp.source(), adaptor.source(),
                             &allocatedPtr, &alignedPtr, &offset);

    // Set pointers and offset.
    Type llvmElementType = unwrap(typeConverter->convertType(elementType));
    auto elementPtrPtrType = LLVM::LLVMPointerType::get(
        LLVM::LLVMPointerType::get(llvmElementType, addressSpace));
    UnrankedMemRefDescriptor::setAllocatedPtr(rewriter, loc, underlyingDescPtr,
                                              elementPtrPtrType, allocatedPtr);
    UnrankedMemRefDescriptor::setAlignedPtr(rewriter, loc, *getTypeConverter(),
                                            underlyingDescPtr,
                                            elementPtrPtrType, alignedPtr);
    UnrankedMemRefDescriptor::setOffset(rewriter, loc, *getTypeConverter(),
                                        underlyingDescPtr, elementPtrPtrType,
                                        offset);

    // Use the offset pointer as base for further addressing. Copy over the new
    // shape and compute strides. For this, we create a loop from rank-1 to 0.
    Value targetSizesBase = UnrankedMemRefDescriptor::sizeBasePtr(
        rewriter, loc, *getTypeConverter(), underlyingDescPtr,
        elementPtrPtrType);
    Value targetStridesBase = UnrankedMemRefDescriptor::strideBasePtr(
        rewriter, loc, *getTypeConverter(), targetSizesBase, resultRank);
    Value shapeOperandPtr = shapeDesc.alignedPtr(rewriter, loc);
    Value oneIndex = createIndexConstant(rewriter, loc, 1);
    Value resultRankMinusOne =
        rewriter.create<LLVM::SubOp>(loc, resultRank, oneIndex);

    Block *initBlock = rewriter.getInsertionBlock();
    Type indexType = getTypeConverter()->getIndexType();
    Block::iterator remainingOpsIt = std::next(rewriter.getInsertionPoint());

    Block *condBlock = rewriter.createBlock(initBlock->getParent(), {},
                                            {indexType, indexType});

    // Iterate over the remaining ops in initBlock and move them to condBlock.
    BlockAndValueMapping map;
    for (auto it = remainingOpsIt, e = initBlock->end(); it != e; ++it) {
      rewriter.clone(*it, map);
      rewriter.eraseOp(&*it);
    }

    rewriter.setInsertionPointToEnd(initBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange({resultRankMinusOne, oneIndex}),
                                condBlock);
    rewriter.setInsertionPointToStart(condBlock);
    Value indexArg = condBlock->getArgument(0);
    Value strideArg = condBlock->getArgument(1);

    Value zeroIndex = createIndexConstant(rewriter, loc, 0);
    Value pred = rewriter.create<LLVM::ICmpOp>(
        loc, IntegerType::get(rewriter.getContext(), 1),
        LLVM::ICmpPredicate::sge, indexArg, zeroIndex);

    Block *bodyBlock =
        rewriter.splitBlock(condBlock, rewriter.getInsertionPoint());
    rewriter.setInsertionPointToStart(bodyBlock);

    // Copy size from shape to descriptor.
    Type llvmIndexPtrType = LLVM::LLVMPointerType::get(indexType);
    Value sizeLoadGep = rewriter.create<LLVM::GEPOp>(
        loc, llvmIndexPtrType, shapeOperandPtr, ValueRange{indexArg});
    Value size = rewriter.create<LLVM::LoadOp>(loc, sizeLoadGep);
    UnrankedMemRefDescriptor::setSize(rewriter, loc, *getTypeConverter(),
                                      targetSizesBase, indexArg, size);

    // Write stride value and compute next one.
    UnrankedMemRefDescriptor::setStride(rewriter, loc, *getTypeConverter(),
                                        targetStridesBase, indexArg, strideArg);
    Value nextStride = rewriter.create<LLVM::MulOp>(loc, strideArg, size);

    // Decrement loop counter and branch back.
    Value decrement = rewriter.create<LLVM::SubOp>(loc, indexArg, oneIndex);
    rewriter.create<LLVM::BrOp>(loc, ValueRange({decrement, nextStride}),
                                condBlock);

    Block *remainder =
        rewriter.splitBlock(bodyBlock, rewriter.getInsertionPoint());

    // Hook up the cond exit to the remainder.
    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, bodyBlock, llvm::None, remainder,
                                    llvm::None);

    // Reset position to beginning of new remainder block.
    rewriter.setInsertionPointToStart(remainder);

    *descriptor = targetDesc;
    return success();
  }
};

struct DialectCastOpLowering
    : public ConvertOpToLLVMPattern<LLVM::DialectCastOp> {
  using ConvertOpToLLVMPattern<LLVM::DialectCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LLVM::DialectCastOp castOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::DialectCastOp::Adaptor transformed(operands);
    if (transformed.in().getType() !=
        typeConverter->convertType(castOp.getType())) {
      return failure();
    }
    rewriter.replaceOp(castOp, transformed.in());
    return success();
  }
};

// A `dim` is converted to a constant for static sizes and to an access to the
// size stored in the memref descriptor for dynamic sizes.
struct DimOpLowering : public ConvertOpToLLVMPattern<DimOp> {
  using ConvertOpToLLVMPattern<DimOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(DimOp dimOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type operandType = dimOp.memrefOrTensor().getType();
    if (operandType.isa<UnrankedMemRefType>()) {
      rewriter.replaceOp(dimOp, {extractSizeOfUnrankedMemRef(
                                    operandType, dimOp, operands, rewriter)});

      return success();
    }
    if (operandType.isa<MemRefType>()) {
      rewriter.replaceOp(dimOp, {extractSizeOfRankedMemRef(
                                    operandType, dimOp, operands, rewriter)});
      return success();
    }
    return failure();
  }

private:
  Value extractSizeOfUnrankedMemRef(Type operandType, DimOp dimOp,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) const {
    Location loc = dimOp.getLoc();
    DimOp::Adaptor transformed(operands);

    auto unrankedMemRefType = operandType.cast<UnrankedMemRefType>();
    auto scalarMemRefType =
        MemRefType::get({}, unrankedMemRefType.getElementType());
    unsigned addressSpace = unrankedMemRefType.getMemorySpaceAsInt();

    // Extract pointer to the underlying ranked descriptor and bitcast it to a
    // memref<element_type> descriptor pointer to minimize the number of GEP
    // operations.
    UnrankedMemRefDescriptor unrankedDesc(transformed.memrefOrTensor());
    Value underlyingRankedDesc = unrankedDesc.memRefDescPtr(rewriter, loc);
    Value scalarMemRefDescPtr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(typeConverter->convertType(scalarMemRefType),
                                   addressSpace),
        underlyingRankedDesc);

    // Get pointer to offset field of memref<element_type> descriptor.
    Type indexPtrTy = LLVM::LLVMPointerType::get(
        getTypeConverter()->getIndexType(), addressSpace);
    Value two = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter->convertType(rewriter.getI32Type()),
        rewriter.getI32IntegerAttr(2));
    Value offsetPtr = rewriter.create<LLVM::GEPOp>(
        loc, indexPtrTy, scalarMemRefDescPtr,
        ValueRange({createIndexConstant(rewriter, loc, 0), two}));

    // The size value that we have to extract can be obtained using GEPop with
    // `dimOp.index() + 1` index argument.
    Value idxPlusOne = rewriter.create<LLVM::AddOp>(
        loc, createIndexConstant(rewriter, loc, 1), transformed.index());
    Value sizePtr = rewriter.create<LLVM::GEPOp>(loc, indexPtrTy, offsetPtr,
                                                 ValueRange({idxPlusOne}));
    return rewriter.create<LLVM::LoadOp>(loc, sizePtr);
  }

  Value extractSizeOfRankedMemRef(Type operandType, DimOp dimOp,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
    Location loc = dimOp.getLoc();
    DimOp::Adaptor transformed(operands);
    // Take advantage if index is constant.
    MemRefType memRefType = operandType.cast<MemRefType>();
    if (Optional<int64_t> index = dimOp.getConstantIndex()) {
      int64_t i = index.getValue();
      if (memRefType.isDynamicDim(i)) {
        // extract dynamic size from the memref descriptor.
        MemRefDescriptor descriptor(transformed.memrefOrTensor());
        return descriptor.size(rewriter, loc, i);
      }
      // Use constant for static size.
      int64_t dimSize = memRefType.getDimSize(i);
      return createIndexConstant(rewriter, loc, dimSize);
    }
    Value index = dimOp.index();
    int64_t rank = memRefType.getRank();
    MemRefDescriptor memrefDescriptor(transformed.memrefOrTensor());
    return memrefDescriptor.size(rewriter, loc, index, rank);
  }
};

struct RankOpLowering : public ConvertOpToLLVMPattern<RankOp> {
  using ConvertOpToLLVMPattern<RankOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RankOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type operandType = op.memrefOrTensor().getType();
    if (auto unrankedMemRefType = operandType.dyn_cast<UnrankedMemRefType>()) {
      UnrankedMemRefDescriptor desc(RankOp::Adaptor(operands).memrefOrTensor());
      rewriter.replaceOp(op, {desc.rank(rewriter, loc)});
      return success();
    }
    if (auto rankedMemRefType = operandType.dyn_cast<MemRefType>()) {
      rewriter.replaceOp(
          op, {createIndexConstant(rewriter, loc, rankedMemRefType.getRank())});
      return success();
    }
    return failure();
  }
};

// Common base for load and store operations on MemRefs.  Restricts the match
// to supported MemRef types.  Provides functionality to emit code accessing a
// specific element of the underlying data buffer.
template <typename Derived>
struct LoadStoreOpLowering : public ConvertOpToLLVMPattern<Derived> {
  using ConvertOpToLLVMPattern<Derived>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<Derived>::isConvertibleAndHasIdentityMaps;
  using Base = LoadStoreOpLowering<Derived>;

  LogicalResult match(Derived op) const override {
    MemRefType type = op.getMemRefType();
    return isConvertibleAndHasIdentityMaps(type) ? success() : failure();
  }
};

// Load operation is lowered to obtaining a pointer to the indexed element
// and loading it.
struct LoadOpLowering : public LoadStoreOpLowering<LoadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(LoadOp loadOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    LoadOp::Adaptor transformed(operands);
    auto type = loadOp.getMemRefType();

    Value dataPtr =
        getStridedElementPtr(loadOp.getLoc(), type, transformed.memref(),
                             transformed.indices(), rewriter);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(loadOp, dataPtr);
    return success();
  }
};

// Store operation is lowered to obtaining a pointer to the indexed element,
// and storing the given value to it.
struct StoreOpLowering : public LoadStoreOpLowering<StoreOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(StoreOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = op.getMemRefType();
    StoreOp::Adaptor transformed(operands);

    Value dataPtr =
        getStridedElementPtr(op.getLoc(), type, transformed.memref(),
                             transformed.indices(), rewriter);
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, transformed.value(),
                                               dataPtr);
    return success();
  }
};

// The prefetch operation is lowered in a way similar to the load operation
// except that the llvm.prefetch operation is used for replacement.
struct PrefetchOpLowering : public LoadStoreOpLowering<PrefetchOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(PrefetchOp prefetchOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    PrefetchOp::Adaptor transformed(operands);
    auto type = prefetchOp.getMemRefType();
    auto loc = prefetchOp.getLoc();

    Value dataPtr = getStridedElementPtr(loc, type, transformed.memref(),
                                         transformed.indices(), rewriter);

    // Replace with llvm.prefetch.
    auto llvmI32Type = typeConverter->convertType(rewriter.getIntegerType(32));
    auto isWrite = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type, rewriter.getI32IntegerAttr(prefetchOp.isWrite()));
    auto localityHint = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type,
        rewriter.getI32IntegerAttr(prefetchOp.localityHint()));
    auto isData = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type, rewriter.getI32IntegerAttr(prefetchOp.isDataCache()));

    rewriter.replaceOpWithNewOp<LLVM::Prefetch>(prefetchOp, dataPtr, isWrite,
                                                localityHint, isData);
    return success();
  }
};

// The lowering of index_cast becomes an integer conversion since index becomes
// an integer.  If the bit width of the source and target integer types is the
// same, just erase the cast.  If the target type is wider, sign-extend the
// value, otherwise truncate it.
struct IndexCastOpLowering : public ConvertOpToLLVMPattern<IndexCastOp> {
  using ConvertOpToLLVMPattern<IndexCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IndexCastOp indexCastOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    IndexCastOpAdaptor transformed(operands);

    auto targetType =
        typeConverter->convertType(indexCastOp.getResult().getType())
            .cast<IntegerType>();
    auto sourceType = transformed.in().getType().cast<IntegerType>();
    unsigned targetBits = targetType.getWidth();
    unsigned sourceBits = sourceType.getWidth();

    if (targetBits == sourceBits)
      rewriter.replaceOp(indexCastOp, transformed.in());
    else if (targetBits < sourceBits)
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(indexCastOp, targetType,
                                                 transformed.in());
    else
      rewriter.replaceOpWithNewOp<LLVM::SExtOp>(indexCastOp, targetType,
                                                transformed.in());
    return success();
  }
};

// Convert std.cmp predicate into the LLVM dialect CmpPredicate.  The two
// enums share the numerical values so just cast.
template <typename LLVMPredType, typename StdPredType>
static LLVMPredType convertCmpPredicate(StdPredType pred) {
  return static_cast<LLVMPredType>(pred);
}

struct CmpIOpLowering : public ConvertOpToLLVMPattern<CmpIOp> {
  using ConvertOpToLLVMPattern<CmpIOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CmpIOp cmpiOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    CmpIOpAdaptor transformed(operands);

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        cmpiOp, typeConverter->convertType(cmpiOp.getResult().getType()),
        rewriter.getI64IntegerAttr(static_cast<int64_t>(
            convertCmpPredicate<LLVM::ICmpPredicate>(cmpiOp.getPredicate()))),
        transformed.lhs(), transformed.rhs());

    return success();
  }
};

struct CmpFOpLowering : public ConvertOpToLLVMPattern<CmpFOp> {
  using ConvertOpToLLVMPattern<CmpFOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CmpFOp cmpfOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    CmpFOpAdaptor transformed(operands);

    auto fmf = LLVM::FMFAttr::get(cmpfOp.getContext(), {});
    rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
        cmpfOp, typeConverter->convertType(cmpfOp.getResult().getType()),
        rewriter.getI64IntegerAttr(static_cast<int64_t>(
            convertCmpPredicate<LLVM::FCmpPredicate>(cmpfOp.getPredicate()))),
        transformed.lhs(), transformed.rhs(), fmf);

    return success();
  }
};

struct SIToFPLowering
    : public OneToOneConvertToLLVMPattern<SIToFPOp, LLVM::SIToFPOp> {
  using Super::Super;
};

struct UIToFPLowering
    : public OneToOneConvertToLLVMPattern<UIToFPOp, LLVM::UIToFPOp> {
  using Super::Super;
};

struct FPExtLowering
    : public OneToOneConvertToLLVMPattern<FPExtOp, LLVM::FPExtOp> {
  using Super::Super;
};

struct FPToSILowering
    : public OneToOneConvertToLLVMPattern<FPToSIOp, LLVM::FPToSIOp> {
  using Super::Super;
};

struct FPToUILowering
    : public OneToOneConvertToLLVMPattern<FPToUIOp, LLVM::FPToUIOp> {
  using Super::Super;
};

struct FPTruncLowering
    : public OneToOneConvertToLLVMPattern<FPTruncOp, LLVM::FPTruncOp> {
  using Super::Super;
};

struct TruncateIOpLowering
    : public OneToOneConvertToLLVMPattern<TruncateIOp, LLVM::TruncOp> {
  using Super::Super;
};

// Base class for LLVM IR lowering terminator operations with successors.
template <typename SourceOp, typename TargetOp>
struct OneToOneLLVMTerminatorLowering
    : public ConvertOpToLLVMPattern<SourceOp> {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using Super = OneToOneLLVMTerminatorLowering<SourceOp, TargetOp>;

  LogicalResult
  matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, operands, op->getSuccessors(),
                                          op->getAttrs());
    return success();
  }
};

// Special lowering pattern for `ReturnOps`.  Unlike all other operations,
// `ReturnOp` interacts with the function signature and must have as many
// operands as the function has return values.  Because in LLVM IR, functions
// can only return 0 or 1 value, we pack multiple values into a structure type.
// Emit `UndefOp` followed by `InsertValueOp`s to create such structure if
// necessary before returning it
struct ReturnOpLowering : public ConvertOpToLLVMPattern<ReturnOp> {
  using ConvertOpToLLVMPattern<ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned numArguments = op.getNumOperands();
    SmallVector<Value, 4> updatedOperands;

    if (getTypeConverter()->getOptions().useBarePtrCallConv) {
      // For the bare-ptr calling convention, extract the aligned pointer to
      // be returned from the memref descriptor.
      for (auto it : llvm::zip(op->getOperands(), operands)) {
        Type oldTy = std::get<0>(it).getType();
        Value newOperand = std::get<1>(it);
        if (oldTy.isa<MemRefType>()) {
          MemRefDescriptor memrefDesc(newOperand);
          newOperand = memrefDesc.alignedPtr(rewriter, loc);
        } else if (oldTy.isa<UnrankedMemRefType>()) {
          // Unranked memref is not supported in the bare pointer calling
          // convention.
          return failure();
        }
        updatedOperands.push_back(newOperand);
      }
    } else {
      updatedOperands = llvm::to_vector<4>(operands);
      (void)copyUnrankedDescriptors(rewriter, loc, *getTypeConverter(),
                                    op.getOperands().getTypes(),
                                    updatedOperands,
                                    /*toDynamic=*/true);
    }

    // If ReturnOp has 0 or 1 operand, create it and return immediately.
    if (numArguments == 0) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), ValueRange(),
                                                  op->getAttrs());
      return success();
    }
    if (numArguments == 1) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
          op, TypeRange(), updatedOperands, op->getAttrs());
      return success();
    }

    // Otherwise, we need to pack the arguments into an LLVM struct type before
    // returning.
    auto packedType = getTypeConverter()->packFunctionResults(
        llvm::to_vector<4>(op.getOperandTypes()));

    Value packed = rewriter.create<LLVM::UndefOp>(loc, packedType);
    for (unsigned i = 0; i < numArguments; ++i) {
      packed = rewriter.create<LLVM::InsertValueOp>(
          loc, packedType, packed, updatedOperands[i],
          rewriter.getI64ArrayAttr(i));
    }
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), packed,
                                                op->getAttrs());
    return success();
  }
};

// FIXME: this should be tablegen'ed as well.
struct BranchOpLowering
    : public OneToOneLLVMTerminatorLowering<BranchOp, LLVM::BrOp> {
  using Super::Super;
};
struct CondBranchOpLowering
    : public OneToOneLLVMTerminatorLowering<CondBranchOp, LLVM::CondBrOp> {
  using Super::Super;
};

// The Splat operation is lowered to an insertelement + a shufflevector
// operation. Splat to only 1-d vector result types are lowered.
struct SplatOpLowering : public ConvertOpToLLVMPattern<SplatOp> {
  using ConvertOpToLLVMPattern<SplatOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SplatOp splatOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = splatOp.getType().dyn_cast<VectorType>();
    if (!resultType || resultType.getRank() != 1)
      return failure();

    // First insert it into an undef vector so we can shuffle it.
    auto vectorType = typeConverter->convertType(splatOp.getType());
    Value undef = rewriter.create<LLVM::UndefOp>(splatOp.getLoc(), vectorType);
    auto zero = rewriter.create<LLVM::ConstantOp>(
        splatOp.getLoc(),
        typeConverter->convertType(rewriter.getIntegerType(32)),
        rewriter.getZeroAttr(rewriter.getIntegerType(32)));

    auto v = rewriter.create<LLVM::InsertElementOp>(
        splatOp.getLoc(), vectorType, undef, splatOp.getOperand(), zero);

    int64_t width = splatOp.getType().cast<VectorType>().getDimSize(0);
    SmallVector<int32_t, 4> zeroValues(width, 0);

    // Shuffle the value across the desired number of elements.
    ArrayAttr zeroAttrs = rewriter.getI32ArrayAttr(zeroValues);
    rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(splatOp, v, undef,
                                                       zeroAttrs);
    return success();
  }
};

// The Splat operation is lowered to an insertelement + a shufflevector
// operation. Splat to only 2+-d vector result types are lowered by the
// SplatNdOpLowering, the 1-d case is handled by SplatOpLowering.
struct SplatNdOpLowering : public ConvertOpToLLVMPattern<SplatOp> {
  using ConvertOpToLLVMPattern<SplatOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SplatOp splatOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SplatOp::Adaptor adaptor(operands);
    VectorType resultType = splatOp.getType().dyn_cast<VectorType>();
    if (!resultType || resultType.getRank() == 1)
      return failure();

    // First insert it into an undef vector so we can shuffle it.
    auto loc = splatOp.getLoc();
    auto vectorTypeInfo =
        extractNDVectorTypeInfo(resultType, *getTypeConverter());
    auto llvmNDVectorTy = vectorTypeInfo.llvmNDVectorTy;
    auto llvm1DVectorTy = vectorTypeInfo.llvm1DVectorTy;
    if (!llvmNDVectorTy || !llvm1DVectorTy)
      return failure();

    // Construct returned value.
    Value desc = rewriter.create<LLVM::UndefOp>(loc, llvmNDVectorTy);

    // Construct a 1-D vector with the splatted value that we insert in all the
    // places within the returned descriptor.
    Value vdesc = rewriter.create<LLVM::UndefOp>(loc, llvm1DVectorTy);
    auto zero = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter->convertType(rewriter.getIntegerType(32)),
        rewriter.getZeroAttr(rewriter.getIntegerType(32)));
    Value v = rewriter.create<LLVM::InsertElementOp>(loc, llvm1DVectorTy, vdesc,
                                                     adaptor.input(), zero);

    // Shuffle the value across the desired number of elements.
    int64_t width = resultType.getDimSize(resultType.getRank() - 1);
    SmallVector<int32_t, 4> zeroValues(width, 0);
    ArrayAttr zeroAttrs = rewriter.getI32ArrayAttr(zeroValues);
    v = rewriter.create<LLVM::ShuffleVectorOp>(loc, v, v, zeroAttrs);

    // Iterate of linear index, convert to coords space and insert splatted 1-D
    // vector in each position.
    nDVectorIterate(vectorTypeInfo, rewriter, [&](ArrayAttr position) {
      desc = rewriter.create<LLVM::InsertValueOp>(loc, llvmNDVectorTy, desc, v,
                                                  position);
    });
    rewriter.replaceOp(splatOp, desc);
    return success();
  }
};

/// Helper function extracts int64_t from the assumedArrayAttr of IntegerAttr.
static SmallVector<int64_t, 4> extractFromI64ArrayAttr(Attribute attr) {
  return llvm::to_vector<4>(
      llvm::map_range(attr.cast<ArrayAttr>(), [](Attribute a) -> int64_t {
        return a.cast<IntegerAttr>().getInt();
      }));
}

/// Conversion pattern that transforms a subview op into:
///   1. An `llvm.mlir.undef` operation to create a memref descriptor
///   2. Updates to the descriptor to introduce the data ptr, offset, size
///      and stride.
/// The subview op is replaced by the descriptor.
struct SubViewOpLowering : public ConvertOpToLLVMPattern<SubViewOp> {
  using ConvertOpToLLVMPattern<SubViewOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SubViewOp subViewOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = subViewOp.getLoc();

    auto sourceMemRefType = subViewOp.source().getType().cast<MemRefType>();
    auto sourceElementTy =
        typeConverter->convertType(sourceMemRefType.getElementType());

    auto viewMemRefType = subViewOp.getType();
    auto inferredType = SubViewOp::inferResultType(
                            subViewOp.getSourceType(),
                            extractFromI64ArrayAttr(subViewOp.static_offsets()),
                            extractFromI64ArrayAttr(subViewOp.static_sizes()),
                            extractFromI64ArrayAttr(subViewOp.static_strides()))
                            .cast<MemRefType>();
    auto targetElementTy =
        typeConverter->convertType(viewMemRefType.getElementType());
    auto targetDescTy = typeConverter->convertType(viewMemRefType);
    if (!sourceElementTy || !targetDescTy || !targetElementTy ||
        !LLVM::isCompatibleType(sourceElementTy) ||
        !LLVM::isCompatibleType(targetElementTy) ||
        !LLVM::isCompatibleType(targetDescTy))
      return failure();

    // Extract the offset and strides from the type.
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(inferredType, strides, offset);
    if (failed(successStrides))
      return failure();

    // Create the descriptor.
    if (!LLVM::isCompatibleType(operands.front().getType()))
      return failure();
    MemRefDescriptor sourceMemRef(operands.front());
    auto targetMemRef = MemRefDescriptor::undef(rewriter, loc, targetDescTy);

    // Copy the buffer pointer from the old descriptor to the new one.
    Value extracted = sourceMemRef.allocatedPtr(rewriter, loc);
    Value bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(targetElementTy,
                                   viewMemRefType.getMemorySpaceAsInt()),
        extracted);
    targetMemRef.setAllocatedPtr(rewriter, loc, bitcastPtr);

    // Copy the aligned pointer from the old descriptor to the new one.
    extracted = sourceMemRef.alignedPtr(rewriter, loc);
    bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(targetElementTy,
                                   viewMemRefType.getMemorySpaceAsInt()),
        extracted);
    targetMemRef.setAlignedPtr(rewriter, loc, bitcastPtr);

    auto shape = viewMemRefType.getShape();
    auto inferredShape = inferredType.getShape();
    size_t inferredShapeRank = inferredShape.size();
    size_t resultShapeRank = shape.size();
    llvm::SmallDenseSet<unsigned> unusedDims =
        computeRankReductionMask(inferredShape, shape).getValue();

    // Extract strides needed to compute offset.
    SmallVector<Value, 4> strideValues;
    strideValues.reserve(inferredShapeRank);
    for (unsigned i = 0; i < inferredShapeRank; ++i)
      strideValues.push_back(sourceMemRef.stride(rewriter, loc, i));

    // Offset.
    auto llvmIndexType = typeConverter->convertType(rewriter.getIndexType());
    if (!ShapedType::isDynamicStrideOrOffset(offset)) {
      targetMemRef.setConstantOffset(rewriter, loc, offset);
    } else {
      Value baseOffset = sourceMemRef.offset(rewriter, loc);
      // `inferredShapeRank` may be larger than the number of offset operands
      // because of trailing semantics. In this case, the offset is guaranteed
      // to be interpreted as 0 and we can just skip the extra dimensions.
      for (unsigned i = 0, e = std::min(inferredShapeRank,
                                        subViewOp.getMixedOffsets().size());
           i < e; ++i) {
        Value offset =
            // TODO: need OpFoldResult ODS adaptor to clean this up.
            subViewOp.isDynamicOffset(i)
                ? operands[subViewOp.getIndexOfDynamicOffset(i)]
                : rewriter.create<LLVM::ConstantOp>(
                      loc, llvmIndexType,
                      rewriter.getI64IntegerAttr(subViewOp.getStaticOffset(i)));
        Value mul = rewriter.create<LLVM::MulOp>(loc, offset, strideValues[i]);
        baseOffset = rewriter.create<LLVM::AddOp>(loc, baseOffset, mul);
      }
      targetMemRef.setOffset(rewriter, loc, baseOffset);
    }

    // Update sizes and strides.
    SmallVector<OpFoldResult> mixedSizes = subViewOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = subViewOp.getMixedStrides();
    assert(mixedSizes.size() == mixedStrides.size() &&
           "expected sizes and strides of equal length");
    for (int i = inferredShapeRank - 1, j = resultShapeRank - 1;
         i >= 0 && j >= 0; --i) {
      if (unusedDims.contains(i))
        continue;

      // `i` may overflow subViewOp.getMixedSizes because of trailing semantics.
      // In this case, the size is guaranteed to be interpreted as Dim and the
      // stride as 1.
      Value size, stride;
      if (static_cast<unsigned>(i) >= mixedSizes.size()) {
        size = rewriter.create<LLVM::DialectCastOp>(
            loc, llvmIndexType,
            rewriter.create<DimOp>(loc, subViewOp.source(), i));
        stride = rewriter.create<LLVM::ConstantOp>(
            loc, llvmIndexType, rewriter.getI64IntegerAttr(1));
      } else {
        // TODO: need OpFoldResult ODS adaptor to clean this up.
        size =
            subViewOp.isDynamicSize(i)
                ? operands[subViewOp.getIndexOfDynamicSize(i)]
                : rewriter.create<LLVM::ConstantOp>(
                      loc, llvmIndexType,
                      rewriter.getI64IntegerAttr(subViewOp.getStaticSize(i)));
        if (!ShapedType::isDynamicStrideOrOffset(strides[i])) {
          stride = rewriter.create<LLVM::ConstantOp>(
              loc, llvmIndexType, rewriter.getI64IntegerAttr(strides[i]));
        } else {
          stride = subViewOp.isDynamicStride(i)
                       ? operands[subViewOp.getIndexOfDynamicStride(i)]
                       : rewriter.create<LLVM::ConstantOp>(
                             loc, llvmIndexType,
                             rewriter.getI64IntegerAttr(
                                 subViewOp.getStaticStride(i)));
          stride = rewriter.create<LLVM::MulOp>(loc, stride, strideValues[i]);
        }
      }
      targetMemRef.setSize(rewriter, loc, j, size);
      targetMemRef.setStride(rewriter, loc, j, stride);
      j--;
    }

    rewriter.replaceOp(subViewOp, {targetMemRef});
    return success();
  }
};

/// Conversion pattern that transforms a transpose op into:
///   1. A function entry `alloca` operation to allocate a ViewDescriptor.
///   2. A load of the ViewDescriptor from the pointer allocated in 1.
///   3. Updates to the ViewDescriptor to introduce the data ptr, offset, size
///      and stride. Size and stride are permutations of the original values.
///   4. A store of the resulting ViewDescriptor to the alloca'ed pointer.
/// The transpose op is replaced by the alloca'ed pointer.
class TransposeOpLowering : public ConvertOpToLLVMPattern<TransposeOp> {
public:
  using ConvertOpToLLVMPattern<TransposeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TransposeOp transposeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = transposeOp.getLoc();
    TransposeOpAdaptor adaptor(operands);
    MemRefDescriptor viewMemRef(adaptor.in());

    // No permutation, early exit.
    if (transposeOp.permutation().isIdentity())
      return rewriter.replaceOp(transposeOp, {viewMemRef}), success();

    auto targetMemRef = MemRefDescriptor::undef(
        rewriter, loc, typeConverter->convertType(transposeOp.getShapedType()));

    // Copy the base and aligned pointers from the old descriptor to the new
    // one.
    targetMemRef.setAllocatedPtr(rewriter, loc,
                                 viewMemRef.allocatedPtr(rewriter, loc));
    targetMemRef.setAlignedPtr(rewriter, loc,
                               viewMemRef.alignedPtr(rewriter, loc));

    // Copy the offset pointer from the old descriptor to the new one.
    targetMemRef.setOffset(rewriter, loc, viewMemRef.offset(rewriter, loc));

    // Iterate over the dimensions and apply size/stride permutation.
    for (auto en : llvm::enumerate(transposeOp.permutation().getResults())) {
      int sourcePos = en.index();
      int targetPos = en.value().cast<AffineDimExpr>().getPosition();
      targetMemRef.setSize(rewriter, loc, targetPos,
                           viewMemRef.size(rewriter, loc, sourcePos));
      targetMemRef.setStride(rewriter, loc, targetPos,
                             viewMemRef.stride(rewriter, loc, sourcePos));
    }

    rewriter.replaceOp(transposeOp, {targetMemRef});
    return success();
  }
};

/// Conversion pattern that transforms an op into:
///   1. An `llvm.mlir.undef` operation to create a memref descriptor
///   2. Updates to the descriptor to introduce the data ptr, offset, size
///      and stride.
/// The view op is replaced by the descriptor.
struct ViewOpLowering : public ConvertOpToLLVMPattern<ViewOp> {
  using ConvertOpToLLVMPattern<ViewOp>::ConvertOpToLLVMPattern;

  // Build and return the value for the idx^th shape dimension, either by
  // returning the constant shape dimension or counting the proper dynamic size.
  Value getSize(ConversionPatternRewriter &rewriter, Location loc,
                ArrayRef<int64_t> shape, ValueRange dynamicSizes,
                unsigned idx) const {
    assert(idx < shape.size());
    if (!ShapedType::isDynamic(shape[idx]))
      return createIndexConstant(rewriter, loc, shape[idx]);
    // Count the number of dynamic dims in range [0, idx]
    unsigned nDynamic = llvm::count_if(shape.take_front(idx), [](int64_t v) {
      return ShapedType::isDynamic(v);
    });
    return dynamicSizes[nDynamic];
  }

  // Build and return the idx^th stride, either by returning the constant stride
  // or by computing the dynamic stride from the current `runningStride` and
  // `nextSize`. The caller should keep a running stride and update it with the
  // result returned by this function.
  Value getStride(ConversionPatternRewriter &rewriter, Location loc,
                  ArrayRef<int64_t> strides, Value nextSize,
                  Value runningStride, unsigned idx) const {
    assert(idx < strides.size());
    if (!MemRefType::isDynamicStrideOrOffset(strides[idx]))
      return createIndexConstant(rewriter, loc, strides[idx]);
    if (nextSize)
      return runningStride
                 ? rewriter.create<LLVM::MulOp>(loc, runningStride, nextSize)
                 : nextSize;
    assert(!runningStride);
    return createIndexConstant(rewriter, loc, 1);
  }

  LogicalResult
  matchAndRewrite(ViewOp viewOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = viewOp.getLoc();
    ViewOpAdaptor adaptor(operands);

    auto viewMemRefType = viewOp.getType();
    auto targetElementTy =
        typeConverter->convertType(viewMemRefType.getElementType());
    auto targetDescTy = typeConverter->convertType(viewMemRefType);
    if (!targetDescTy || !targetElementTy ||
        !LLVM::isCompatibleType(targetElementTy) ||
        !LLVM::isCompatibleType(targetDescTy))
      return viewOp.emitWarning("Target descriptor type not converted to LLVM"),
             failure();

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(viewMemRefType, strides, offset);
    if (failed(successStrides))
      return viewOp.emitWarning("cannot cast to non-strided shape"), failure();
    assert(offset == 0 && "expected offset to be 0");

    // Create the descriptor.
    MemRefDescriptor sourceMemRef(adaptor.source());
    auto targetMemRef = MemRefDescriptor::undef(rewriter, loc, targetDescTy);

    // Field 1: Copy the allocated pointer, used for malloc/free.
    Value allocatedPtr = sourceMemRef.allocatedPtr(rewriter, loc);
    auto srcMemRefType = viewOp.source().getType().cast<MemRefType>();
    Value bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(targetElementTy,
                                   srcMemRefType.getMemorySpaceAsInt()),
        allocatedPtr);
    targetMemRef.setAllocatedPtr(rewriter, loc, bitcastPtr);

    // Field 2: Copy the actual aligned pointer to payload.
    Value alignedPtr = sourceMemRef.alignedPtr(rewriter, loc);
    alignedPtr = rewriter.create<LLVM::GEPOp>(loc, alignedPtr.getType(),
                                              alignedPtr, adaptor.byte_shift());
    bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(targetElementTy,
                                   srcMemRefType.getMemorySpaceAsInt()),
        alignedPtr);
    targetMemRef.setAlignedPtr(rewriter, loc, bitcastPtr);

    // Field 3: The offset in the resulting type must be 0. This is because of
    // the type change: an offset on srcType* may not be expressible as an
    // offset on dstType*.
    targetMemRef.setOffset(rewriter, loc,
                           createIndexConstant(rewriter, loc, offset));

    // Early exit for 0-D corner case.
    if (viewMemRefType.getRank() == 0)
      return rewriter.replaceOp(viewOp, {targetMemRef}), success();

    // Fields 4 and 5: Update sizes and strides.
    if (strides.back() != 1)
      return viewOp.emitWarning("cannot cast to non-contiguous shape"),
             failure();
    Value stride = nullptr, nextSize = nullptr;
    for (int i = viewMemRefType.getRank() - 1; i >= 0; --i) {
      // Update size.
      Value size =
          getSize(rewriter, loc, viewMemRefType.getShape(), adaptor.sizes(), i);
      targetMemRef.setSize(rewriter, loc, i, size);
      // Update stride.
      stride = getStride(rewriter, loc, strides, nextSize, stride, i);
      targetMemRef.setStride(rewriter, loc, i, stride);
      nextSize = size;
    }

    rewriter.replaceOp(viewOp, {targetMemRef});
    return success();
  }
};

struct AssumeAlignmentOpLowering
    : public ConvertOpToLLVMPattern<AssumeAlignmentOp> {
  using ConvertOpToLLVMPattern<AssumeAlignmentOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AssumeAlignmentOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    AssumeAlignmentOp::Adaptor transformed(operands);
    Value memref = transformed.memref();
    unsigned alignment = op.alignment();
    auto loc = op.getLoc();

    MemRefDescriptor memRefDescriptor(memref);
    Value ptr = memRefDescriptor.alignedPtr(rewriter, memref.getLoc());

    // Emit llvm.assume(memref.alignedPtr & (alignment - 1) == 0). Notice that
    // the asserted memref.alignedPtr isn't used anywhere else, as the real
    // users like load/store/views always re-extract memref.alignedPtr as they
    // get lowered.
    //
    // This relies on LLVM's CSE optimization (potentially after SROA), since
    // after CSE all memref.alignedPtr instances get de-duplicated into the same
    // pointer SSA value.
    auto intPtrType =
        getIntPtrType(memRefDescriptor.getElementPtrType().getAddressSpace());
    Value zero = createIndexAttrConstant(rewriter, loc, intPtrType, 0);
    Value mask =
        createIndexAttrConstant(rewriter, loc, intPtrType, alignment - 1);
    Value ptrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, ptr);
    rewriter.create<LLVM::AssumeOp>(
        loc, rewriter.create<LLVM::ICmpOp>(
                 loc, LLVM::ICmpPredicate::eq,
                 rewriter.create<LLVM::AndOp>(loc, ptrValue, mask), zero));

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

/// Try to match the kind of a std.atomic_rmw to determine whether to use a
/// lowering to llvm.atomicrmw or fallback to llvm.cmpxchg.
static Optional<LLVM::AtomicBinOp> matchSimpleAtomicOp(AtomicRMWOp atomicOp) {
  switch (atomicOp.kind()) {
  case AtomicRMWKind::addf:
    return LLVM::AtomicBinOp::fadd;
  case AtomicRMWKind::addi:
    return LLVM::AtomicBinOp::add;
  case AtomicRMWKind::assign:
    return LLVM::AtomicBinOp::xchg;
  case AtomicRMWKind::maxs:
    return LLVM::AtomicBinOp::max;
  case AtomicRMWKind::maxu:
    return LLVM::AtomicBinOp::umax;
  case AtomicRMWKind::mins:
    return LLVM::AtomicBinOp::min;
  case AtomicRMWKind::minu:
    return LLVM::AtomicBinOp::umin;
  default:
    return llvm::None;
  }
  llvm_unreachable("Invalid AtomicRMWKind");
}

namespace {

struct AtomicRMWOpLowering : public LoadStoreOpLowering<AtomicRMWOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(AtomicRMWOp atomicOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(match(atomicOp)))
      return failure();
    auto maybeKind = matchSimpleAtomicOp(atomicOp);
    if (!maybeKind)
      return failure();
    AtomicRMWOp::Adaptor adaptor(operands);
    auto resultType = adaptor.value().getType();
    auto memRefType = atomicOp.getMemRefType();
    auto dataPtr =
        getStridedElementPtr(atomicOp.getLoc(), memRefType, adaptor.memref(),
                             adaptor.indices(), rewriter);
    rewriter.replaceOpWithNewOp<LLVM::AtomicRMWOp>(
        atomicOp, resultType, *maybeKind, dataPtr, adaptor.value(),
        LLVM::AtomicOrdering::acq_rel);
    return success();
  }
};

/// Wrap a llvm.cmpxchg operation in a while loop so that the operation can be
/// retried until it succeeds in atomically storing a new value into memory.
///
///      +---------------------------------+
///      |   <code before the AtomicRMWOp> |
///      |   <compute initial %loaded>     |
///      |   br loop(%loaded)              |
///      +---------------------------------+
///             |
///  -------|   |
///  |      v   v
///  |   +--------------------------------+
///  |   | loop(%loaded):                 |
///  |   |   <body contents>              |
///  |   |   %pair = cmpxchg              |
///  |   |   %ok = %pair[0]               |
///  |   |   %new = %pair[1]              |
///  |   |   cond_br %ok, end, loop(%new) |
///  |   +--------------------------------+
///  |          |        |
///  |-----------        |
///                      v
///      +--------------------------------+
///      | end:                           |
///      |   <code after the AtomicRMWOp> |
///      +--------------------------------+
///
struct GenericAtomicRMWOpLowering
    : public LoadStoreOpLowering<GenericAtomicRMWOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(GenericAtomicRMWOp atomicOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = atomicOp.getLoc();
    GenericAtomicRMWOp::Adaptor adaptor(operands);
    Type valueType = typeConverter->convertType(atomicOp.getResult().getType());

    // Split the block into initial, loop, and ending parts.
    auto *initBlock = rewriter.getInsertionBlock();
    auto *loopBlock =
        rewriter.createBlock(initBlock->getParent(),
                             std::next(Region::iterator(initBlock)), valueType);
    auto *endBlock = rewriter.createBlock(
        loopBlock->getParent(), std::next(Region::iterator(loopBlock)));

    // Operations range to be moved to `endBlock`.
    auto opsToMoveStart = atomicOp->getIterator();
    auto opsToMoveEnd = initBlock->back().getIterator();

    // Compute the loaded value and branch to the loop block.
    rewriter.setInsertionPointToEnd(initBlock);
    auto memRefType = atomicOp.memref().getType().cast<MemRefType>();
    auto dataPtr = getStridedElementPtr(loc, memRefType, adaptor.memref(),
                                        adaptor.indices(), rewriter);
    Value init = rewriter.create<LLVM::LoadOp>(loc, dataPtr);
    rewriter.create<LLVM::BrOp>(loc, init, loopBlock);

    // Prepare the body of the loop block.
    rewriter.setInsertionPointToStart(loopBlock);

    // Clone the GenericAtomicRMWOp region and extract the result.
    auto loopArgument = loopBlock->getArgument(0);
    BlockAndValueMapping mapping;
    mapping.map(atomicOp.getCurrentValue(), loopArgument);
    Block &entryBlock = atomicOp.body().front();
    for (auto &nestedOp : entryBlock.without_terminator()) {
      Operation *clone = rewriter.clone(nestedOp, mapping);
      mapping.map(nestedOp.getResults(), clone->getResults());
    }
    Value result = mapping.lookup(entryBlock.getTerminator()->getOperand(0));

    // Prepare the epilog of the loop block.
    // Append the cmpxchg op to the end of the loop block.
    auto successOrdering = LLVM::AtomicOrdering::acq_rel;
    auto failureOrdering = LLVM::AtomicOrdering::monotonic;
    auto boolType = IntegerType::get(rewriter.getContext(), 1);
    auto pairType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                     {valueType, boolType});
    auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
        loc, pairType, dataPtr, loopArgument, result, successOrdering,
        failureOrdering);
    // Extract the %new_loaded and %ok values from the pair.
    Value newLoaded = rewriter.create<LLVM::ExtractValueOp>(
        loc, valueType, cmpxchg, rewriter.getI64ArrayAttr({0}));
    Value ok = rewriter.create<LLVM::ExtractValueOp>(
        loc, boolType, cmpxchg, rewriter.getI64ArrayAttr({1}));

    // Conditionally branch to the end or back to the loop depending on %ok.
    rewriter.create<LLVM::CondBrOp>(loc, ok, endBlock, ArrayRef<Value>(),
                                    loopBlock, newLoaded);

    rewriter.setInsertionPointToEnd(endBlock);
    moveOpsRange(atomicOp.getResult(), newLoaded, std::next(opsToMoveStart),
                 std::next(opsToMoveEnd), rewriter);

    // The 'result' of the atomic_rmw op is the newly loaded value.
    rewriter.replaceOp(atomicOp, {newLoaded});

    return success();
  }

private:
  // Clones a segment of ops [start, end) and erases the original.
  void moveOpsRange(ValueRange oldResult, ValueRange newResult,
                    Block::iterator start, Block::iterator end,
                    ConversionPatternRewriter &rewriter) const {
    BlockAndValueMapping mapping;
    mapping.map(oldResult, newResult);
    SmallVector<Operation *, 2> opsToErase;
    for (auto it = start; it != end; ++it) {
      rewriter.clone(*it, mapping);
      opsToErase.push_back(&*it);
    }
    for (auto *it : opsToErase)
      rewriter.eraseOp(it);
  }
};

} // namespace

/// Collect a set of patterns to convert from the Standard dialect to LLVM.
void mlir::populateStdToLLVMNonMemoryConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  // FIXME: this should be tablegen'ed
  // clang-format off
  patterns.insert<
      AbsFOpLowering,
      AddFOpLowering,
      AddIOpLowering,
      AllocaOpLowering,
      AndOpLowering,
      AssertOpLowering,
      AtomicRMWOpLowering,
      BranchOpLowering,
      CallIndirectOpLowering,
      CallOpLowering,
      CeilFOpLowering,
      CmpFOpLowering,
      CmpIOpLowering,
      CondBranchOpLowering,
      CopySignOpLowering,
      CosOpLowering,
      ConstantOpLowering,
      DialectCastOpLowering,
      DivFOpLowering,
      ExpOpLowering,
      Exp2OpLowering,
      FloorFOpLowering,
      FmaFOpLowering,
      GenericAtomicRMWOpLowering,
      LogOpLowering,
      Log10OpLowering,
      Log2OpLowering,
      FPExtLowering,
      FPToSILowering,
      FPToUILowering,
      FPTruncLowering,
      IndexCastOpLowering,
      MulFOpLowering,
      MulIOpLowering,
      NegFOpLowering,
      OrOpLowering,
      PowFOpLowering,
      PrefetchOpLowering,
      RemFOpLowering,
      ReturnOpLowering,
      RsqrtOpLowering,
      SIToFPLowering,
      SelectOpLowering,
      ShiftLeftOpLowering,
      SignExtendIOpLowering,
      SignedDivIOpLowering,
      SignedRemIOpLowering,
      SignedShiftRightOpLowering,
      SinOpLowering,
      SplatOpLowering,
      SplatNdOpLowering,
      SqrtOpLowering,
      SubFOpLowering,
      SubIOpLowering,
      TruncateIOpLowering,
      UIToFPLowering,
      UnsignedDivIOpLowering,
      UnsignedRemIOpLowering,
      UnsignedShiftRightOpLowering,
      XOrOpLowering,
      ZeroExtendIOpLowering>(converter);
  // clang-format on
}

void mlir::populateStdToLLVMMemoryConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  // clang-format off
  patterns.insert<
      AssumeAlignmentOpLowering,
      DeallocOpLowering,
      DimOpLowering,
      GlobalMemrefOpLowering,
      GetGlobalMemrefOpLowering,
      LoadOpLowering,
      MemRefCastOpLowering,
      MemRefReinterpretCastOpLowering,
      MemRefReshapeOpLowering,
      RankOpLowering,
      StoreOpLowering,
      SubViewOpLowering,
      TransposeOpLowering,
      ViewOpLowering>(converter);
  // clang-format on
  if (converter.getOptions().useAlignedAlloc)
    patterns.insert<AlignedAllocOpLowering>(converter);
  else
    patterns.insert<AllocOpLowering>(converter);
}

void mlir::populateStdToLLVMFuncOpConversionPattern(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  if (converter.getOptions().useBarePtrCallConv)
    patterns.insert<BarePtrFuncOpConversion>(converter);
  else
    patterns.insert<FuncOpConversion>(converter);
}

void mlir::populateStdToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  populateStdToLLVMFuncOpConversionPattern(converter, patterns);
  populateStdToLLVMNonMemoryConversionPatterns(converter, patterns);
  populateStdToLLVMMemoryConversionPatterns(converter, patterns);
}

/// Convert a non-empty list of types to be returned from a function into a
/// supported LLVM IR type.  In particular, if more than one value is returned,
/// create an LLVM IR structure type with elements that correspond to each of
/// the MLIR types converted with `convertType`.
Type LLVMTypeConverter::packFunctionResults(TypeRange types) {
  assert(!types.empty() && "expected non-empty list of type");

  if (types.size() == 1)
    return convertCallingConventionType(types.front());

  SmallVector<Type, 8> resultTypes;
  resultTypes.reserve(types.size());
  for (auto t : types) {
    auto converted = convertCallingConventionType(t);
    if (!converted || !LLVM::isCompatibleType(converted))
      return {};
    resultTypes.push_back(converted);
  }

  return LLVM::LLVMStructType::getLiteral(&getContext(), resultTypes);
}

Value LLVMTypeConverter::promoteOneMemRefDescriptor(Location loc, Value operand,
                                                    OpBuilder &builder) {
  auto *context = builder.getContext();
  auto int64Ty = IntegerType::get(builder.getContext(), 64);
  auto indexType = IndexType::get(context);
  // Alloca with proper alignment. We do not expect optimizations of this
  // alloca op and so we omit allocating at the entry block.
  auto ptrType = LLVM::LLVMPointerType::get(operand.getType());
  Value one = builder.create<LLVM::ConstantOp>(loc, int64Ty,
                                               IntegerAttr::get(indexType, 1));
  Value allocated =
      builder.create<LLVM::AllocaOp>(loc, ptrType, one, /*alignment=*/0);
  // Store into the alloca'ed descriptor.
  builder.create<LLVM::StoreOp>(loc, operand, allocated);
  return allocated;
}

SmallVector<Value, 4> LLVMTypeConverter::promoteOperands(Location loc,
                                                         ValueRange opOperands,
                                                         ValueRange operands,
                                                         OpBuilder &builder) {
  SmallVector<Value, 4> promotedOperands;
  promotedOperands.reserve(operands.size());
  for (auto it : llvm::zip(opOperands, operands)) {
    auto operand = std::get<0>(it);
    auto llvmOperand = std::get<1>(it);

    if (options.useBarePtrCallConv) {
      // For the bare-ptr calling convention, we only have to extract the
      // aligned pointer of a memref.
      if (auto memrefType = operand.getType().dyn_cast<MemRefType>()) {
        MemRefDescriptor desc(llvmOperand);
        llvmOperand = desc.alignedPtr(builder, loc);
      } else if (operand.getType().isa<UnrankedMemRefType>()) {
        llvm_unreachable("Unranked memrefs are not supported");
      }
    } else {
      if (operand.getType().isa<UnrankedMemRefType>()) {
        UnrankedMemRefDescriptor::unpack(builder, loc, llvmOperand,
                                         promotedOperands);
        continue;
      }
      if (auto memrefType = operand.getType().dyn_cast<MemRefType>()) {
        MemRefDescriptor::unpack(builder, loc, llvmOperand, memrefType,
                                 promotedOperands);
        continue;
      }
    }

    promotedOperands.push_back(llvmOperand);
  }
  return promotedOperands;
}

namespace {
/// A pass converting MLIR operations into the LLVM IR dialect.
struct LLVMLoweringPass : public ConvertStandardToLLVMBase<LLVMLoweringPass> {
  LLVMLoweringPass() = default;
  LLVMLoweringPass(bool useBarePtrCallConv, bool emitCWrappers,
                   unsigned indexBitwidth, bool useAlignedAlloc,
                   const llvm::DataLayout &dataLayout) {
    this->useBarePtrCallConv = useBarePtrCallConv;
    this->emitCWrappers = emitCWrappers;
    this->indexBitwidth = indexBitwidth;
    this->useAlignedAlloc = useAlignedAlloc;
    this->dataLayout = dataLayout.getStringRepresentation();
  }

  /// Run the dialect converter on the module.
  void runOnOperation() override {
    if (useBarePtrCallConv && emitCWrappers) {
      getOperation().emitError()
          << "incompatible conversion options: bare-pointer calling convention "
             "and C wrapper emission";
      signalPassFailure();
      return;
    }
    if (failed(LLVM::LLVMDialect::verifyDataLayoutString(
            this->dataLayout, [this](const Twine &message) {
              getOperation().emitError() << message.str();
            }))) {
      signalPassFailure();
      return;
    }

    ModuleOp m = getOperation();

    LowerToLLVMOptions options = {useBarePtrCallConv, emitCWrappers,
                                  indexBitwidth, useAlignedAlloc,
                                  llvm::DataLayout(this->dataLayout)};
    LLVMTypeConverter typeConverter(&getContext(), options);

    OwningRewritePatternList patterns;
    populateStdToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
    m->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
               StringAttr::get(m.getContext(), this->dataLayout));
  }
};
} // end namespace

mlir::LLVMConversionTarget::LLVMConversionTarget(MLIRContext &ctx)
    : ConversionTarget(ctx) {
  this->addLegalDialect<LLVM::LLVMDialect>();
  this->addIllegalOp<LLVM::DialectCastOp>();
  this->addIllegalOp<math::TanhOp>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createLowerToLLVMPass(const LowerToLLVMOptions &options) {
  return std::make_unique<LLVMLoweringPass>(
      options.useBarePtrCallConv, options.emitCWrappers, options.indexBitwidth,
      options.useAlignedAlloc, options.dataLayout);
}
