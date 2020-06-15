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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
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
static LLVM::LLVMType unwrap(Type type) {
  if (!type)
    return nullptr;
  auto *mlirContext = type.getContext();
  auto wrappedLLVMType = type.dyn_cast<LLVM::LLVMType>();
  if (!wrappedLLVMType)
    emitError(UnknownLoc::get(mlirContext),
              "conversion resulted in a non-LLVM type");
  return wrappedLLVMType;
}

/// Initialize customization to default callbacks.
LLVMTypeConverterCustomization::LLVMTypeConverterCustomization()
    : funcArgConverter(structFuncArgTypeConverter),
      indexBitwidth(kDeriveIndexBitwidthFromDataLayout) {}

/// Callback to convert function argument types. It converts a MemRef function
/// argument to a list of non-aggregate types containing descriptor
/// information, and an UnrankedmemRef function argument to a list containing
/// the rank and a pointer to a descriptor struct.
LogicalResult mlir::structFuncArgTypeConverter(LLVMTypeConverter &converter,
                                               Type type,
                                               SmallVectorImpl<Type> &result) {
  if (auto memref = type.dyn_cast<MemRefType>()) {
    auto converted = converter.convertMemRefSignature(memref);
    if (converted.empty())
      return failure();
    result.append(converted.begin(), converted.end());
    return success();
  }
  if (type.isa<UnrankedMemRefType>()) {
    auto converted = converter.convertUnrankedMemRefSignature();
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

/// Convert a MemRef type to a bare pointer to the MemRef element type.
static Type convertMemRefTypeToBarePtr(LLVMTypeConverter &converter,
                                       MemRefType type) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(type, strides, offset)))
    return {};

  LLVM::LLVMType elementType =
      unwrap(converter.convertType(type.getElementType()));
  if (!elementType)
    return {};
  return elementType.getPointerTo(type.getMemorySpace());
}

/// Callback to convert function argument types. It converts MemRef function
/// arguments to bare pointers to the MemRef element type.
LogicalResult mlir::barePtrFuncArgTypeConverter(LLVMTypeConverter &converter,
                                                Type type,
                                                SmallVectorImpl<Type> &result) {
  // TODO: Add support for unranked memref.
  if (auto memrefTy = type.dyn_cast<MemRefType>()) {
    auto llvmTy = convertMemRefTypeToBarePtr(converter, memrefTy);
    if (!llvmTy)
      return failure();

    result.push_back(llvmTy);
    return success();
  }

  auto llvmTy = converter.convertType(type);
  if (!llvmTy)
    return failure();

  result.push_back(llvmTy);
  return success();
}

/// Create an LLVMTypeConverter using default LLVMTypeConverterCustomization.
LLVMTypeConverter::LLVMTypeConverter(MLIRContext *ctx)
    : LLVMTypeConverter(ctx, LLVMTypeConverterCustomization()) {}

/// Create an LLVMTypeConverter using 'custom' customizations.
LLVMTypeConverter::LLVMTypeConverter(
    MLIRContext *ctx, const LLVMTypeConverterCustomization &customs)
    : llvmDialect(ctx->getRegisteredDialect<LLVM::LLVMDialect>()),
      customizations(customs) {
  assert(llvmDialect && "LLVM IR dialect is not registered");
  module = &llvmDialect->getLLVMModule();
  if (customizations.indexBitwidth == kDeriveIndexBitwidthFromDataLayout)
    customizations.indexBitwidth =
        module->getDataLayout().getPointerSizeInBits();

  // Register conversions for the standard types.
  addConversion([&](ComplexType type) { return convertComplexType(type); });
  addConversion([&](FloatType type) { return convertFloatType(type); });
  addConversion([&](FunctionType type) { return convertFunctionType(type); });
  addConversion([&](IndexType type) { return convertIndexType(type); });
  addConversion([&](IntegerType type) { return convertIntegerType(type); });
  addConversion([&](MemRefType type) { return convertMemRefType(type); });
  addConversion(
      [&](UnrankedMemRefType type) { return convertUnrankedMemRefType(type); });
  addConversion([&](VectorType type) { return convertVectorType(type); });

  // LLVMType is legal, so add a pass-through conversion.
  addConversion([](LLVM::LLVMType type) { return type; });

  // Materialization for memrefs creates descriptor structs from individual
  // values constituting them, when descriptors are used, i.e. more than one
  // value represents a memref.
  addMaterialization([&](PatternRewriter &rewriter,
                         UnrankedMemRefType resultType, ValueRange inputs,
                         Location loc) -> Optional<Value> {
    if (inputs.size() == 1)
      return llvm::None;
    return UnrankedMemRefDescriptor::pack(rewriter, loc, *this, resultType,
                                          inputs);
  });
  addMaterialization([&](PatternRewriter &rewriter, MemRefType resultType,
                         ValueRange inputs, Location loc) -> Optional<Value> {
    if (inputs.size() == 1)
      return llvm::None;
    return MemRefDescriptor::pack(rewriter, loc, *this, resultType, inputs);
  });
}

/// Returns the MLIR context.
MLIRContext &LLVMTypeConverter::getContext() {
  return *getDialect()->getContext();
}

/// Get the LLVM context.
llvm::LLVMContext &LLVMTypeConverter::getLLVMContext() {
  return module->getContext();
}

LLVM::LLVMType LLVMTypeConverter::getIndexType() {
  return LLVM::LLVMType::getIntNTy(llvmDialect, getIndexTypeBitwidth());
}

Type LLVMTypeConverter::convertIndexType(IndexType type) {
  return getIndexType();
}

Type LLVMTypeConverter::convertIntegerType(IntegerType type) {
  return LLVM::LLVMType::getIntNTy(llvmDialect, type.getWidth());
}

Type LLVMTypeConverter::convertFloatType(FloatType type) {
  switch (type.getKind()) {
  case mlir::StandardTypes::F32:
    return LLVM::LLVMType::getFloatTy(llvmDialect);
  case mlir::StandardTypes::F64:
    return LLVM::LLVMType::getDoubleTy(llvmDialect);
  case mlir::StandardTypes::F16:
    return LLVM::LLVMType::getHalfTy(llvmDialect);
  case mlir::StandardTypes::BF16: {
    return LLVM::LLVMType::getBFloatTy(llvmDialect);
  }
  default:
    llvm_unreachable("non-float type in convertFloatType");
  }
}

// Convert a `ComplexType` to an LLVM type. The result is a complex number
// struct with entries for the
//   1. real part and for the
//   2. imaginary part.
static constexpr unsigned kRealPosInComplexNumberStruct = 0;
static constexpr unsigned kImaginaryPosInComplexNumberStruct = 1;
Type LLVMTypeConverter::convertComplexType(ComplexType type) {
  auto elementType = convertType(type.getElementType()).cast<LLVM::LLVMType>();
  return LLVM::LLVMType::getStructTy(llvmDialect, {elementType, elementType});
}

// Except for signatures, MLIR function types are converted into LLVM
// pointer-to-function types.
Type LLVMTypeConverter::convertFunctionType(FunctionType type) {
  SignatureConversion conversion(type.getNumInputs());
  LLVM::LLVMType converted =
      convertFunctionSignature(type, /*isVariadic=*/false, conversion);
  return converted.getPointerTo();
}

/// In signatures, MemRef descriptors are expanded into lists of non-aggregate
/// values.
SmallVector<Type, 5>
LLVMTypeConverter::convertMemRefSignature(MemRefType type) {
  SmallVector<Type, 5> results;
  assert(isStrided(type) &&
         "Non-strided layout maps must have been normalized away");

  LLVM::LLVMType elementType = unwrap(convertType(type.getElementType()));
  if (!elementType)
    return {};
  auto indexTy = getIndexType();

  results.insert(results.begin(), 2,
                 elementType.getPointerTo(type.getMemorySpace()));
  results.push_back(indexTy);
  auto rank = type.getRank();
  results.insert(results.end(), 2 * rank, indexTy);
  return results;
}

/// In signatures, unranked MemRef descriptors are expanded into a pair "rank,
/// pointer to descriptor".
SmallVector<Type, 2> LLVMTypeConverter::convertUnrankedMemRefSignature() {
  return {getIndexType(), LLVM::LLVMType::getInt8PtrTy(llvmDialect)};
}

// Function types are converted to LLVM Function types by recursively converting
// argument and result types.  If MLIR Function has zero results, the LLVM
// Function has one VoidType result.  If MLIR Function has more than one result,
// they are into an LLVM StructType in their order of appearance.
LLVM::LLVMType LLVMTypeConverter::convertFunctionSignature(
    FunctionType type, bool isVariadic,
    LLVMTypeConverter::SignatureConversion &result) {
  // Convert argument types one by one and check for errors.
  for (auto &en : llvm::enumerate(type.getInputs())) {
    Type type = en.value();
    SmallVector<Type, 8> converted;
    if (failed(customizations.funcArgConverter(*this, type, converted)))
      return {};
    result.addInputs(en.index(), converted);
  }

  SmallVector<LLVM::LLVMType, 8> argTypes;
  argTypes.reserve(llvm::size(result.getConvertedTypes()));
  for (Type type : result.getConvertedTypes())
    argTypes.push_back(unwrap(type));

  // If function does not return anything, create the void result type,
  // if it returns on element, convert it, otherwise pack the result types into
  // a struct.
  LLVM::LLVMType resultType =
      type.getNumResults() == 0
          ? LLVM::LLVMType::getVoidTy(llvmDialect)
          : unwrap(packFunctionResults(type.getResults()));
  if (!resultType)
    return {};
  return LLVM::LLVMType::getFunctionTy(resultType, argTypes, isVariadic);
}

/// Converts the function type to a C-compatible format, in particular using
/// pointers to memref descriptors for arguments.
LLVM::LLVMType
LLVMTypeConverter::convertFunctionTypeCWrapper(FunctionType type) {
  SmallVector<LLVM::LLVMType, 4> inputs;

  for (Type t : type.getInputs()) {
    auto converted = convertType(t).dyn_cast_or_null<LLVM::LLVMType>();
    if (!converted)
      return {};
    if (t.isa<MemRefType>() || t.isa<UnrankedMemRefType>())
      converted = converted.getPointerTo();
    inputs.push_back(converted);
  }

  LLVM::LLVMType resultType =
      type.getNumResults() == 0
          ? LLVM::LLVMType::getVoidTy(llvmDialect)
          : unwrap(packFunctionResults(type.getResults()));
  if (!resultType)
    return {};

  return LLVM::LLVMType::getFunctionTy(resultType, inputs, false);
}

// Convert a MemRef to an LLVM type. The result is a MemRef descriptor which
// contains:
//   1. the pointer to the data buffer, followed by
//   2.  a lowered `index`-type integer containing the distance between the
//   beginning of the buffer and the first element to be accessed through the
//   view, followed by
//   3. an array containing as many `index`-type integers as the rank of the
//   MemRef: the array represents the size, in number of elements, of the memref
//   along the given dimension. For constant MemRef dimensions, the
//   corresponding size entry is a constant whose runtime value must match the
//   static value, followed by
//   4. a second array containing as many `index`-type integers as the rank of
//   the MemRef: the second array represents the "stride" (in tensor abstraction
//   sense), i.e. the number of consecutive elements of the underlying buffer.
//   TODO(ntv, zinenko): add assertions for the static cases.
//
// template <typename Elem, size_t Rank>
// struct {
//   Elem *allocatedPtr;
//   Elem *alignedPtr;
//   int64_t offset;
//   int64_t sizes[Rank]; // omitted when rank == 0
//   int64_t strides[Rank]; // omitted when rank == 0
// };
static constexpr unsigned kAllocatedPtrPosInMemRefDescriptor = 0;
static constexpr unsigned kAlignedPtrPosInMemRefDescriptor = 1;
static constexpr unsigned kOffsetPosInMemRefDescriptor = 2;
static constexpr unsigned kSizePosInMemRefDescriptor = 3;
static constexpr unsigned kStridePosInMemRefDescriptor = 4;
Type LLVMTypeConverter::convertMemRefType(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  bool strideSuccess = succeeded(getStridesAndOffset(type, strides, offset));
  assert(strideSuccess &&
         "Non-strided layout maps must have been normalized away");
  (void)strideSuccess;
  LLVM::LLVMType elementType = unwrap(convertType(type.getElementType()));
  if (!elementType)
    return {};
  auto ptrTy = elementType.getPointerTo(type.getMemorySpace());
  auto indexTy = getIndexType();
  auto rank = type.getRank();
  if (rank > 0) {
    auto arrayTy = LLVM::LLVMType::getArrayTy(indexTy, type.getRank());
    return LLVM::LLVMType::getStructTy(ptrTy, ptrTy, indexTy, arrayTy, arrayTy);
  }
  return LLVM::LLVMType::getStructTy(ptrTy, ptrTy, indexTy);
}

// Converts UnrankedMemRefType to LLVMType. The result is a descriptor which
// contains:
// 1. int64_t rank, the dynamic rank of this MemRef
// 2. void* ptr, pointer to the static ranked MemRef descriptor. This will be
//    stack allocated (alloca) copy of a MemRef descriptor that got casted to
//    be unranked.

static constexpr unsigned kRankInUnrankedMemRefDescriptor = 0;
static constexpr unsigned kPtrInUnrankedMemRefDescriptor = 1;

Type LLVMTypeConverter::convertUnrankedMemRefType(UnrankedMemRefType type) {
  auto rankTy = LLVM::LLVMType::getInt64Ty(llvmDialect);
  auto ptrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  return LLVM::LLVMType::getStructTy(rankTy, ptrTy);
}

// Convert an n-D vector type to an LLVM vector type via (n-1)-D array type when
// n > 1.
// For example, `vector<4 x f32>` converts to `!llvm.type<"<4 x float>">` and
// `vector<4 x 8 x 16 f32>` converts to `!llvm<"[4 x [8 x <16 x float>]]">`.
Type LLVMTypeConverter::convertVectorType(VectorType type) {
  auto elementType = unwrap(convertType(type.getElementType()));
  if (!elementType)
    return {};
  auto vectorType =
      LLVM::LLVMType::getVectorTy(elementType, type.getShape().back());
  auto shape = type.getShape();
  for (int i = shape.size() - 2; i >= 0; --i)
    vectorType = LLVM::LLVMType::getArrayTy(vectorType, shape[i]);
  return vectorType;
}

ConvertToLLVMPattern::ConvertToLLVMPattern(StringRef rootOpName,
                                           MLIRContext *context,
                                           LLVMTypeConverter &typeConverter_,
                                           PatternBenefit benefit)
    : ConversionPattern(rootOpName, benefit, context),
      typeConverter(typeConverter_) {}

/*============================================================================*/
/* StructBuilder implementation                                               */
/*============================================================================*/

StructBuilder::StructBuilder(Value v) : value(v) {
  assert(value != nullptr && "value cannot be null");
  structType = value.getType().dyn_cast<LLVM::LLVMType>();
  assert(structType && "expected llvm type");
}

Value StructBuilder::extractPtr(OpBuilder &builder, Location loc,
                                unsigned pos) {
  Type type = structType.cast<LLVM::LLVMType>().getStructElementType(pos);
  return builder.create<LLVM::ExtractValueOp>(loc, type, value,
                                              builder.getI64ArrayAttr(pos));
}

void StructBuilder::setPtr(OpBuilder &builder, Location loc, unsigned pos,
                           Value ptr) {
  value = builder.create<LLVM::InsertValueOp>(loc, structType, value, ptr,
                                              builder.getI64ArrayAttr(pos));
}

/*============================================================================*/
/* ComplexStructBuilder implementation                                        */
/*============================================================================*/

ComplexStructBuilder ComplexStructBuilder::undef(OpBuilder &builder,
                                                 Location loc, Type type) {
  Value val = builder.create<LLVM::UndefOp>(loc, type.cast<LLVM::LLVMType>());
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

/*============================================================================*/
/* MemRefDescriptor implementation                                            */
/*============================================================================*/

/// Construct a helper for the given descriptor value.
MemRefDescriptor::MemRefDescriptor(Value descriptor)
    : StructBuilder(descriptor) {
  assert(value != nullptr && "value cannot be null");
  indexType = value.getType().cast<LLVM::LLVMType>().getStructElementType(
      kOffsetPosInMemRefDescriptor);
}

/// Builds IR creating an `undef` value of the descriptor type.
MemRefDescriptor MemRefDescriptor::undef(OpBuilder &builder, Location loc,
                                         Type descriptorType) {

  Value descriptor =
      builder.create<LLVM::UndefOp>(loc, descriptorType.cast<LLVM::LLVMType>());
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
  assert(offset != MemRefType::getDynamicStrideOrOffset() &&
         "expected static offset");
  assert(!llvm::is_contained(strides, MemRefType::getDynamicStrideOrOffset()) &&
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

/// Builds IR inserting the pos-th size into the descriptor
void MemRefDescriptor::setSize(OpBuilder &builder, Location loc, unsigned pos,
                               Value size) {
  value = builder.create<LLVM::InsertValueOp>(
      loc, structType, value, size,
      builder.getI64ArrayAttr({kSizePosInMemRefDescriptor, pos}));
}

/// Builds IR inserting the pos-th size into the descriptor
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

/// Builds IR inserting the pos-th stride into the descriptor
void MemRefDescriptor::setConstantStride(OpBuilder &builder, Location loc,
                                         unsigned pos, uint64_t stride) {
  setStride(builder, loc, pos,
            createIndexAttrConstant(builder, loc, indexType, stride));
}

LLVM::LLVMType MemRefDescriptor::getElementType() {
  return value.getType().cast<LLVM::LLVMType>().getStructElementType(
      kAlignedPtrPosInMemRefDescriptor);
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

/*============================================================================*/
/* MemRefDescriptorView implementation.                                       */
/*============================================================================*/

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

/*============================================================================*/
/* UnrankedMemRefDescriptor implementation                                    */
/*============================================================================*/

/// Construct a helper for the given descriptor value.
UnrankedMemRefDescriptor::UnrankedMemRefDescriptor(Value descriptor)
    : StructBuilder(descriptor) {}

/// Builds IR creating an `undef` value of the descriptor type.
UnrankedMemRefDescriptor UnrankedMemRefDescriptor::undef(OpBuilder &builder,
                                                         Location loc,
                                                         Type descriptorType) {
  Value descriptor =
      builder.create<LLVM::UndefOp>(loc, descriptorType.cast<LLVM::LLVMType>());
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

LLVM::LLVMDialect &ConvertToLLVMPattern::getDialect() const {
  return *typeConverter.getDialect();
}

llvm::LLVMContext &ConvertToLLVMPattern::getContext() const {
  return typeConverter.getLLVMContext();
}

llvm::Module &ConvertToLLVMPattern::getModule() const {
  return getDialect().getLLVMModule();
}

LLVM::LLVMType ConvertToLLVMPattern::getIndexType() const {
  return typeConverter.getIndexType();
}

LLVM::LLVMType ConvertToLLVMPattern::getVoidType() const {
  return LLVM::LLVMType::getVoidTy(&getDialect());
}

LLVM::LLVMType ConvertToLLVMPattern::getVoidPtrType() const {
  return LLVM::LLVMType::getInt8PtrTy(&getDialect());
}

Value ConvertToLLVMPattern::createIndexConstant(
    ConversionPatternRewriter &builder, Location loc, uint64_t value) const {
  return createIndexAttrConstant(builder, loc, getIndexType(), value);
}

Value ConvertToLLVMPattern::linearizeSubscripts(
    ConversionPatternRewriter &builder, Location loc, ArrayRef<Value> indices,
    ArrayRef<Value> allocSizes) const {
  assert(indices.size() == allocSizes.size() &&
         "mismatching number of indices and allocation sizes");
  assert(!indices.empty() && "cannot linearize a 0-dimensional access");

  Value linearized = indices.front();
  for (int i = 1, nSizes = allocSizes.size(); i < nSizes; ++i) {
    linearized = builder.create<LLVM::MulOp>(
        loc, this->getIndexType(), ArrayRef<Value>{linearized, allocSizes[i]});
    linearized = builder.create<LLVM::AddOp>(
        loc, this->getIndexType(), ArrayRef<Value>{linearized, indices[i]});
  }
  return linearized;
}

Value ConvertToLLVMPattern::getStridedElementPtr(
    Location loc, Type elementTypePtr, Value descriptor, ValueRange indices,
    ArrayRef<int64_t> strides, int64_t offset,
    ConversionPatternRewriter &rewriter) const {
  MemRefDescriptor memRefDescriptor(descriptor);

  Value base = memRefDescriptor.alignedPtr(rewriter, loc);
  Value offsetValue = offset == MemRefType::getDynamicStrideOrOffset()
                          ? memRefDescriptor.offset(rewriter, loc)
                          : this->createIndexConstant(rewriter, loc, offset);

  for (int i = 0, e = indices.size(); i < e; ++i) {
    Value stride = strides[i] == MemRefType::getDynamicStrideOrOffset()
                       ? memRefDescriptor.stride(rewriter, loc, i)
                       : this->createIndexConstant(rewriter, loc, strides[i]);
    Value additionalOffset =
        rewriter.create<LLVM::MulOp>(loc, indices[i], stride);
    offsetValue =
        rewriter.create<LLVM::AddOp>(loc, offsetValue, additionalOffset);
  }
  return rewriter.create<LLVM::GEPOp>(loc, elementTypePtr, base, offsetValue);
}

Value ConvertToLLVMPattern::getDataPtr(Location loc, MemRefType type,
                                       Value memRefDesc, ValueRange indices,
                                       ConversionPatternRewriter &rewriter,
                                       llvm::Module &module) const {
  LLVM::LLVMType ptrType = MemRefDescriptor(memRefDesc).getElementType();
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto successStrides = getStridesAndOffset(type, strides, offset);
  assert(succeeded(successStrides) && "unexpected non-strided memref");
  (void)successStrides;
  return getStridedElementPtr(loc, ptrType, memRefDesc, indices, strides,
                              offset, rewriter);
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
  filterFuncAttributes(funcOp.getAttrs(), /*filterArgAttrs=*/false, attributes);
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

  LLVM::LLVMType wrapperType =
      typeConverter.convertFunctionTypeCWrapper(funcOp.getType());
  // This conversion can only fail if it could not convert one of the argument
  // types. But since it has been applies to a non-wrapper function before, it
  // should have failed earlier and not reach this point at all.
  assert(wrapperType && "unexpected type conversion failure");

  SmallVector<NamedAttribute, 4> attributes;
  filterFuncAttributes(funcOp.getAttrs(), /*filterArgAttrs=*/false, attributes);

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

      auto ptrTy = packed.getType().cast<LLVM::LLVMType>().getPointerTo();
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
  using UnsignedTypePair = std::pair<unsigned, Type>;

  // Gather the positions and types of memref-typed arguments in a given
  // FunctionType.
  void getMemRefArgIndicesAndTypes(
      FunctionType type, SmallVectorImpl<UnsignedTypePair> &argsInfo) const {
    argsInfo.reserve(type.getNumInputs());
    for (auto en : llvm::enumerate(type.getInputs())) {
      if (en.value().isa<MemRefType>() || en.value().isa<UnrankedMemRefType>())
        argsInfo.push_back({en.index(), en.value()});
    }
  }

  // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
  // to this legalization pattern.
  LLVM::LLVMFuncOp
  convertFuncOpToLLVMFuncOp(FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) const {
    // Convert the original function arguments. They are converted using the
    // LLVMTypeConverter provided to this legalization pattern.
    auto varargsAttr = funcOp.getAttrOfType<BoolAttr>("std.varargs");
    TypeConverter::SignatureConversion result(funcOp.getNumArguments());
    auto llvmType = typeConverter.convertFunctionSignature(
        funcOp.getType(), varargsAttr && varargsAttr.getValue(), result);

    // Propagate argument attributes to all converted arguments obtained after
    // converting a given original argument.
    SmallVector<NamedAttribute, 4> attributes;
    filterFuncAttributes(funcOp.getAttrs(), /*filterArgAttrs=*/true,
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
    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newFuncOp.getBody(), result);

    return newFuncOp;
  }
};

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
static constexpr StringRef kEmitIfaceAttrName = "llvm.emit_c_interface";
struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(LLVMTypeConverter &converter, bool emitCWrappers)
      : FuncOpConversionBase(converter), emitWrappers(emitCWrappers) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = cast<FuncOp>(op);

    auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (emitWrappers || funcOp.getAttrOfType<UnitAttr>(kEmitIfaceAttrName)) {
      if (newFuncOp.isExternal())
        wrapExternalFunction(rewriter, op->getLoc(), typeConverter, funcOp,
                             newFuncOp);
      else
        wrapForExternalCallers(rewriter, op->getLoc(), typeConverter, funcOp,
                               newFuncOp);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  /// If true, also create the adaptor functions having signatures compatible
  /// with those produced by clang.
  const bool emitWrappers;
};

/// FuncOp legalization pattern that converts MemRef arguments to bare pointers
/// to the MemRef element type. This will impact the calling convention and ABI.
struct BarePtrFuncOpConversion : public FuncOpConversionBase {
  using FuncOpConversionBase::FuncOpConversionBase;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = cast<FuncOp>(op);

    // Store the positions and type of memref-typed arguments so that we can
    // promote them to MemRef descriptor structs at the beginning of the
    // function.
    SmallVector<UnsignedTypePair, 4> promotedArgsInfo;
    getMemRefArgIndicesAndTypes(funcOp.getType(), promotedArgsInfo);

    auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (newFuncOp.getBody().empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    // Promote bare pointers from MemRef arguments to a MemRef descriptor struct
    // at the beginning of the function so that all the MemRefs in the function
    // have a uniform representation.
    Block *firstBlock = &newFuncOp.getBody().front();
    rewriter.setInsertionPoint(firstBlock, firstBlock->begin());
    auto funcLoc = funcOp.getLoc();
    for (const auto &argInfo : promotedArgsInfo) {
      // TODO: Add support for unranked MemRefs.
      if (auto memrefType = argInfo.second.dyn_cast<MemRefType>()) {
        // Replace argument with a placeholder (undef), promote argument to a
        // MemRef descriptor and replace placeholder with the last instruction
        // of the MemRef descriptor. The placeholder is needed to avoid
        // replacing argument uses in the MemRef descriptor instructions.
        BlockArgument arg = firstBlock->getArgument(argInfo.first);
        Value placeHolder =
            rewriter.create<LLVM::UndefOp>(funcLoc, arg.getType());
        rewriter.replaceUsesOfBlockArgument(arg, placeHolder);
        auto desc = MemRefDescriptor::fromStaticShape(
            rewriter, funcLoc, typeConverter, memrefType, arg);
        rewriter.replaceOp(placeHolder.getDefiningOp(), {desc});
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//////////////// Support for Lowering operations on n-D vectors ////////////////
// Helper struct to "unroll" operations on n-D vectors in terms of operations on
// 1-D LLVM vectors.
struct NDVectorTypeInfo {
  // LLVM array struct which encodes n-D vectors.
  LLVM::LLVMType llvmArrayTy;
  // LLVM vector type which encodes the inner 1-D vector type.
  LLVM::LLVMType llvmVectorTy;
  // Multiplicity of llvmArrayTy to llvmVectorTy.
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
  info.llvmArrayTy =
      converter.convertType(vectorType).dyn_cast<LLVM::LLVMType>();
  if (!info.llvmArrayTy)
    return info;
  info.arraySizes.reserve(vectorType.getRank() - 1);
  auto llvmTy = info.llvmArrayTy;
  while (llvmTy.isArrayTy()) {
    info.arraySizes.push_back(llvmTy.getArrayNumElements());
    llvmTy = llvmTy.getArrayElementType();
  }
  if (!llvmTy.isVectorTy())
    return info;
  info.llvmVectorTy = llvmTy;
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
    std::function<Value(LLVM::LLVMType, ValueRange)> createOperand,
    ConversionPatternRewriter &rewriter) {
  auto vectorType = op->getResult(0).getType().dyn_cast<VectorType>();
  if (!vectorType)
    return failure();
  auto vectorTypeInfo = extractNDVectorTypeInfo(vectorType, typeConverter);
  auto llvmVectorTy = vectorTypeInfo.llvmVectorTy;
  auto llvmArrayTy = operands[0].getType().cast<LLVM::LLVMType>();
  if (!llvmVectorTy || llvmArrayTy != vectorTypeInfo.llvmArrayTy)
    return failure();

  auto loc = op->getLoc();
  Value desc = rewriter.create<LLVM::UndefOp>(loc, llvmArrayTy);
  nDVectorIterate(vectorTypeInfo, rewriter, [&](ArrayAttr position) {
    // For this unrolled `position` corresponding to the `linearIndex`^th
    // element, extract operand vectors
    SmallVector<Value, 4> extractedOperands;
    for (auto operand : operands)
      extractedOperands.push_back(rewriter.create<LLVM::ExtractValueOp>(
          loc, llvmVectorTy, operand, position));
    Value newVal = createOperand(llvmVectorTy, extractedOperands);
    desc = rewriter.create<LLVM::InsertValueOp>(loc, llvmArrayTy, desc, newVal,
                                                position);
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
                    [](Type t) { return t.isa<LLVM::LLVMType>(); }))
    return failure();

  auto llvmArrayTy = operands[0].getType().cast<LLVM::LLVMType>();
  if (!llvmArrayTy.isArrayTy())
    return oneToOneRewrite(op, targetOp, operands, typeConverter, rewriter);

  auto callback = [op, targetOp, &rewriter](LLVM::LLVMType llvmVectorTy,
                                            ValueRange operands) {
    OperationState state(op->getLoc(), targetOp);
    state.addTypes(llvmVectorTy);
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
using ConstLLVMOpLowering =
    OneToOneConvertToLLVMPattern<ConstantOp, LLVM::ConstantOp>;
using CopySignOpLowering =
    VectorConvertToLLVMPattern<CopySignOp, LLVM::CopySignOp>;
using CosOpLowering = VectorConvertToLLVMPattern<CosOp, LLVM::CosOp>;
using DivFOpLowering = VectorConvertToLLVMPattern<DivFOp, LLVM::FDivOp>;
using ExpOpLowering = VectorConvertToLLVMPattern<ExpOp, LLVM::ExpOp>;
using Exp2OpLowering = VectorConvertToLLVMPattern<Exp2Op, LLVM::Exp2Op>;
using Log10OpLowering = VectorConvertToLLVMPattern<Log10Op, LLVM::Log10Op>;
using Log2OpLowering = VectorConvertToLLVMPattern<Log2Op, LLVM::Log2Op>;
using LogOpLowering = VectorConvertToLLVMPattern<LogOp, LLVM::LogOp>;
using MulFOpLowering = VectorConvertToLLVMPattern<MulFOp, LLVM::FMulOp>;
using MulIOpLowering = VectorConvertToLLVMPattern<MulIOp, LLVM::MulOp>;
using NegFOpLowering = VectorConvertToLLVMPattern<NegFOp, LLVM::FNegOp>;
using OrOpLowering = VectorConvertToLLVMPattern<OrOp, LLVM::OrOp>;
using RemFOpLowering = VectorConvertToLLVMPattern<RemFOp, LLVM::FRemOp>;
using SelectOpLowering = OneToOneConvertToLLVMPattern<SelectOp, LLVM::SelectOp>;
using ShiftLeftOpLowering =
    OneToOneConvertToLLVMPattern<ShiftLeftOp, LLVM::ShlOp>;
using SignedDivIOpLowering =
    VectorConvertToLLVMPattern<SignedDivIOp, LLVM::SDivOp>;
using SignedRemIOpLowering =
    VectorConvertToLLVMPattern<SignedRemIOp, LLVM::SRemOp>;
using SignedShiftRightOpLowering =
    OneToOneConvertToLLVMPattern<SignedShiftRightOp, LLVM::AShrOp>;
using SinOpLowering = VectorConvertToLLVMPattern<SinOp, LLVM::SinOp>;
using SqrtOpLowering = VectorConvertToLLVMPattern<SqrtOp, LLVM::SqrtOp>;
using SubFOpLowering = VectorConvertToLLVMPattern<SubFOp, LLVM::FSubOp>;
using SubIOpLowering = VectorConvertToLLVMPattern<SubIOp, LLVM::SubOp>;
using UnsignedDivIOpLowering =
    VectorConvertToLLVMPattern<UnsignedDivIOp, LLVM::UDivOp>;
using UnsignedRemIOpLowering =
    VectorConvertToLLVMPattern<UnsignedRemIOp, LLVM::URemOp>;
using UnsignedShiftRightOpLowering =
    OneToOneConvertToLLVMPattern<UnsignedShiftRightOp, LLVM::LShrOp>;
using XOrOpLowering = VectorConvertToLLVMPattern<XOrOp, LLVM::XOrOp>;

// Lowerings for operations on complex numbers.

struct CreateComplexOpLowering
    : public ConvertOpToLLVMPattern<CreateComplexOp> {
  using ConvertOpToLLVMPattern<CreateComplexOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto complexOp = cast<CreateComplexOp>(op);
    CreateComplexOp::Adaptor transformed(operands);

    // Pack real and imaginary part in a complex number struct.
    auto loc = op->getLoc();
    auto structType = typeConverter.convertType(complexOp.getType());
    auto complexStruct = ComplexStructBuilder::undef(rewriter, loc, structType);
    complexStruct.setReal(rewriter, loc, transformed.real());
    complexStruct.setImaginary(rewriter, loc, transformed.imaginary());

    rewriter.replaceOp(op, {complexStruct});
    return success();
  }
};

struct ReOpLowering : public ConvertOpToLLVMPattern<ReOp> {
  using ConvertOpToLLVMPattern<ReOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    ReOp::Adaptor transformed(operands);

    // Extract real part from the complex number struct.
    ComplexStructBuilder complexStruct(transformed.complex());
    Value real = complexStruct.real(rewriter, op->getLoc());
    rewriter.replaceOp(op, real);

    return success();
  }
};

struct ImOpLowering : public ConvertOpToLLVMPattern<ImOp> {
  using ConvertOpToLLVMPattern<ImOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    ImOp::Adaptor transformed(operands);

    // Extract imaginary part from the complex number struct.
    ComplexStructBuilder complexStruct(transformed.complex());
    Value imaginary = complexStruct.imaginary(rewriter, op->getLoc());
    rewriter.replaceOp(op, imaginary);

    return success();
  }
};

struct BinaryComplexOperands {
  std::complex<Value> lhs, rhs;
};

template <typename OpTy>
BinaryComplexOperands
unpackBinaryComplexOperands(OpTy op, ArrayRef<Value> operands,
                            ConversionPatternRewriter &rewriter) {
  auto bop = cast<OpTy>(op);
  auto loc = bop.getLoc();
  typename OpTy::Adaptor transformed(operands);

  // Extract real and imaginary values from operands.
  BinaryComplexOperands unpacked;
  ComplexStructBuilder lhs(transformed.lhs());
  unpacked.lhs.real(lhs.real(rewriter, loc));
  unpacked.lhs.imag(lhs.imaginary(rewriter, loc));
  ComplexStructBuilder rhs(transformed.rhs());
  unpacked.rhs.real(rhs.real(rewriter, loc));
  unpacked.rhs.imag(rhs.imaginary(rewriter, loc));

  return unpacked;
}

struct AddCFOpLowering : public ConvertOpToLLVMPattern<AddCFOp> {
  using ConvertOpToLLVMPattern<AddCFOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto op = cast<AddCFOp>(operation);
    auto loc = op.getLoc();
    BinaryComplexOperands arg =
        unpackBinaryComplexOperands<AddCFOp>(op, operands, rewriter);

    // Initialize complex number struct for result.
    auto structType = this->typeConverter.convertType(op.getType());
    auto result = ComplexStructBuilder::undef(rewriter, loc, structType);

    // Emit IR to add complex numbers.
    Value real =
        rewriter.create<LLVM::FAddOp>(loc, arg.lhs.real(), arg.rhs.real());
    Value imag =
        rewriter.create<LLVM::FAddOp>(loc, arg.lhs.imag(), arg.rhs.imag());
    result.setReal(rewriter, loc, real);
    result.setImaginary(rewriter, loc, imag);

    rewriter.replaceOp(op, {result});
    return success();
  }
};

struct SubCFOpLowering : public ConvertOpToLLVMPattern<SubCFOp> {
  using ConvertOpToLLVMPattern<SubCFOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto op = cast<SubCFOp>(operation);
    auto loc = op.getLoc();
    BinaryComplexOperands arg =
        unpackBinaryComplexOperands<SubCFOp>(op, operands, rewriter);

    // Initialize complex number struct for result.
    auto structType = this->typeConverter.convertType(op.getType());
    auto result = ComplexStructBuilder::undef(rewriter, loc, structType);

    // Emit IR to substract complex numbers.
    Value real =
        rewriter.create<LLVM::FSubOp>(loc, arg.lhs.real(), arg.rhs.real());
    Value imag =
        rewriter.create<LLVM::FSubOp>(loc, arg.lhs.imag(), arg.rhs.imag());
    result.setReal(rewriter, loc, real);
    result.setImaginary(rewriter, loc, imag);

    rewriter.replaceOp(op, {result});
    return success();
  }
};

// Check if the MemRefType `type` is supported by the lowering. We currently
// only support memrefs with identity maps.
static bool isSupportedMemRefType(MemRefType type) {
  return type.getAffineMaps().empty() ||
         llvm::all_of(type.getAffineMaps(),
                      [](AffineMap map) { return map.isIdentity(); });
}

/// Lowering for AllocOp and AllocaOp.
template <typename AllocLikeOp>
struct AllocLikeOpLowering : public ConvertOpToLLVMPattern<AllocLikeOp> {
  using ConvertOpToLLVMPattern<AllocLikeOp>::createIndexConstant;
  using ConvertOpToLLVMPattern<AllocLikeOp>::getIndexType;
  using ConvertOpToLLVMPattern<AllocLikeOp>::typeConverter;
  using ConvertOpToLLVMPattern<AllocLikeOp>::getVoidPtrType;

  explicit AllocLikeOpLowering(LLVMTypeConverter &converter,
                               bool useAlignedAlloc = false)
      : ConvertOpToLLVMPattern<AllocLikeOp>(converter),
        useAlignedAlloc(useAlignedAlloc) {}

  LogicalResult match(Operation *op) const override {
    MemRefType memRefType = cast<AllocLikeOp>(op).getType();
    if (isSupportedMemRefType(memRefType))
      return success();

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(memRefType, strides, offset);
    if (failed(successStrides))
      return failure();

    // Dynamic strides are ok if they can be deduced from dynamic sizes (which
    // is guaranteed when succeeded(successStrides)). Dynamic offset however can
    // never be alloc'ed.
    if (offset == MemRefType::getDynamicStrideOrOffset())
      return failure();

    return success();
  }

  // Returns bump = (alignment - (input % alignment))% alignment, which is the
  // increment necessary to align `input` to `alignment` boundary.
  // TODO: this can be made more efficient by just using a single addition
  // and two bit shifts: (ptr + align - 1)/align, align is always power of 2.
  Value createBumpToAlign(Location loc, OpBuilder b, Value input,
                          Value alignment) const {
    Value modAlign = b.create<LLVM::URemOp>(loc, input, alignment);
    Value diff = b.create<LLVM::SubOp>(loc, alignment, modAlign);
    Value shift = b.create<LLVM::URemOp>(loc, diff, alignment);
    return shift;
  }

  /// Creates and populates the memref descriptor struct given all its fields.
  /// This method also performs any post allocation alignment needed for heap
  /// allocations when `accessAlignment` is non null. This is used with
  /// allocators that do not support alignment.
  MemRefDescriptor createMemRefDescriptor(
      Location loc, ConversionPatternRewriter &rewriter, MemRefType memRefType,
      Value allocatedTypePtr, Value allocatedBytePtr, Value accessAlignment,
      uint64_t offset, ArrayRef<int64_t> strides, ArrayRef<Value> sizes) const {
    auto elementPtrType = getElementPtrType(memRefType);
    auto structType = typeConverter.convertType(memRefType);
    auto memRefDescriptor = MemRefDescriptor::undef(rewriter, loc, structType);

    // Field 1: Allocated pointer, used for malloc/free.
    memRefDescriptor.setAllocatedPtr(rewriter, loc, allocatedTypePtr);

    // Field 2: Actual aligned pointer to payload.
    Value alignedBytePtr = allocatedTypePtr;
    if (accessAlignment) {
      // offset = (align - (ptr % align))% align
      Value intVal = rewriter.create<LLVM::PtrToIntOp>(
          loc, this->getIndexType(), allocatedBytePtr);
      Value offset = createBumpToAlign(loc, rewriter, intVal, accessAlignment);
      Value aligned = rewriter.create<LLVM::GEPOp>(
          loc, allocatedBytePtr.getType(), allocatedBytePtr, offset);
      alignedBytePtr = rewriter.create<LLVM::BitcastOp>(
          loc, elementPtrType, ArrayRef<Value>(aligned));
    }
    memRefDescriptor.setAlignedPtr(rewriter, loc, alignedBytePtr);

    // Field 3: Offset in aligned pointer.
    memRefDescriptor.setOffset(rewriter, loc,
                               createIndexConstant(rewriter, loc, offset));

    if (memRefType.getRank() == 0)
      // No size/stride descriptor in memref, return the descriptor value.
      return memRefDescriptor;

    // Fields 4 and 5: sizes and strides of the strided MemRef.
    // Store all sizes in the descriptor. Only dynamic sizes are passed in as
    // operands to AllocOp.
    Value runningStride = nullptr;
    // Iterate strides in reverse order, compute runningStride and strideValues.
    auto nStrides = strides.size();
    SmallVector<Value, 4> strideValues(nStrides, nullptr);
    for (unsigned i = 0; i < nStrides; ++i) {
      int64_t index = nStrides - 1 - i;
      if (strides[index] == MemRefType::getDynamicStrideOrOffset())
        // Identity layout map is enforced in the match function, so we compute:
        //   `runningStride *= sizes[index + 1]`
        runningStride = runningStride
                            ? rewriter.create<LLVM::MulOp>(loc, runningStride,
                                                           sizes[index + 1])
                            : createIndexConstant(rewriter, loc, 1);
      else
        runningStride = createIndexConstant(rewriter, loc, strides[index]);
      strideValues[index] = runningStride;
    }
    // Fill size and stride descriptors in memref.
    for (auto indexedSize : llvm::enumerate(sizes)) {
      int64_t index = indexedSize.index();
      memRefDescriptor.setSize(rewriter, loc, index, indexedSize.value());
      memRefDescriptor.setStride(rewriter, loc, index, strideValues[index]);
    }
    return memRefDescriptor;
  }

  /// Determines sizes to be used in the memref descriptor.
  void getSizes(Location loc, MemRefType memRefType, ArrayRef<Value> operands,
                ConversionPatternRewriter &rewriter,
                SmallVectorImpl<Value> &sizes, Value &cumulativeSize,
                Value &one) const {
    sizes.reserve(memRefType.getRank());
    unsigned i = 0;
    for (int64_t s : memRefType.getShape())
      sizes.push_back(s == -1 ? operands[i++]
                              : createIndexConstant(rewriter, loc, s));
    if (sizes.empty())
      sizes.push_back(createIndexConstant(rewriter, loc, 1));

    // Compute the total number of memref elements.
    cumulativeSize = sizes.front();
    for (unsigned i = 1, e = sizes.size(); i < e; ++i)
      cumulativeSize = rewriter.create<LLVM::MulOp>(
          loc, getIndexType(), ArrayRef<Value>{cumulativeSize, sizes[i]});

    // Compute the size of an individual element. This emits the MLIR equivalent
    // of the following sizeof(...) implementation in LLVM IR:
    //   %0 = getelementptr %elementType* null, %indexType 1
    //   %1 = ptrtoint %elementType* %0 to %indexType
    // which is a common pattern of getting the size of a type in bytes.
    auto elementType = memRefType.getElementType();
    auto convertedPtrType = typeConverter.convertType(elementType)
                                .template cast<LLVM::LLVMType>()
                                .getPointerTo();
    auto nullPtr = rewriter.create<LLVM::NullOp>(loc, convertedPtrType);
    one = createIndexConstant(rewriter, loc, 1);
    auto gep = rewriter.create<LLVM::GEPOp>(loc, convertedPtrType,
                                            ArrayRef<Value>{nullPtr, one});
    auto elementSize =
        rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(), gep);
    cumulativeSize = rewriter.create<LLVM::MulOp>(
        loc, getIndexType(), ArrayRef<Value>{cumulativeSize, elementSize});
  }

  /// Returns the type of a pointer to an element of the memref.
  Type getElementPtrType(MemRefType memRefType) const {
    auto elementType = memRefType.getElementType();
    auto structElementType = typeConverter.convertType(elementType);
    return structElementType.template cast<LLVM::LLVMType>().getPointerTo(
        memRefType.getMemorySpace());
  }

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

  /// Returns the alignment to be used for the allocation call itself.
  /// aligned_alloc requires the allocation size to be a power of two, and the
  /// allocation size to be a multiple of alignment,
  Optional<int64_t> getAllocationAlignment(AllocOp allocOp) const {
    // No alignment can be used for the 'malloc' call itself.
    if (!useAlignedAlloc)
      return None;

    if (allocOp.alignment())
      return allocOp.alignment().getValue().getSExtValue();

    // Whenever we don't have alignment set, we will use an alignment
    // consistent with the element type; since the allocation size has to be a
    // power of two, we will bump to the next power of two if it already isn't.
    auto eltSizeBytes = getMemRefEltSizeInBytes(allocOp.getType());
    return std::max(kMinAlignedAllocAlignment,
                    llvm::PowerOf2Ceil(eltSizeBytes));
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

  /// Allocates the underlying buffer using the right call. `allocatedBytePtr`
  /// is set to null for stack allocations. `accessAlignment` is set if
  /// alignment is needed post allocation (for eg. in conjunction with malloc).
  Value allocateBuffer(Location loc, Value cumulativeSize, Operation *op,
                       MemRefType memRefType, Value one, Value &accessAlignment,
                       Value &allocatedBytePtr,
                       ConversionPatternRewriter &rewriter) const {
    auto elementPtrType = getElementPtrType(memRefType);

    // With alloca, one gets a pointer to the element type right away.
    // For stack allocations.
    if (auto allocaOp = dyn_cast<AllocaOp>(op)) {
      allocatedBytePtr = nullptr;
      accessAlignment = nullptr;
      return rewriter.create<LLVM::AllocaOp>(
          loc, elementPtrType, cumulativeSize,
          allocaOp.alignment() ? allocaOp.alignment().getValue().getSExtValue()
                               : 0);
    }

    // Heap allocations.
    AllocOp allocOp = cast<AllocOp>(op);

    Optional<int64_t> allocationAlignment = getAllocationAlignment(allocOp);
    // Whether to use std lib function aligned_alloc that supports alignment.
    bool useAlignedAlloc = allocationAlignment.hasValue();

    // Insert the malloc/aligned_alloc declaration if it is not already present.
    auto allocFuncName = useAlignedAlloc ? "aligned_alloc" : "malloc";
    auto module = allocOp.getParentOfType<ModuleOp>();
    auto allocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(allocFuncName);
    if (!allocFunc) {
      OpBuilder moduleBuilder(op->getParentOfType<ModuleOp>().getBodyRegion());
      SmallVector<LLVM::LLVMType, 2> callArgTypes = {getIndexType()};
      // aligned_alloc(size_t alignment, size_t size)
      if (useAlignedAlloc)
        callArgTypes.push_back(getIndexType());
      allocFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), allocFuncName,
          LLVM::LLVMType::getFunctionTy(getVoidPtrType(), callArgTypes,
                                        /*isVarArg=*/false));
    }

    // Allocate the underlying buffer and store a pointer to it in the MemRef
    // descriptor.
    SmallVector<Value, 2> callArgs;
    if (useAlignedAlloc) {
      // Use aligned_alloc.
      assert(allocationAlignment && "allocation alignment should be present");
      auto alignedAllocAlignmentValue = rewriter.create<LLVM::ConstantOp>(
          loc, typeConverter.convertType(rewriter.getIntegerType(64)),
          rewriter.getI64IntegerAttr(allocationAlignment.getValue()));
      // aligned_alloc requires size to be a multiple of alignment; we will pad
      // the size to the next multiple if necessary.
      if (!isMemRefSizeMultipleOf(memRefType, allocationAlignment.getValue())) {
        Value bump = createBumpToAlign(loc, rewriter, cumulativeSize,
                                       alignedAllocAlignmentValue);
        cumulativeSize =
            rewriter.create<LLVM::AddOp>(loc, cumulativeSize, bump);
      }
      callArgs = {alignedAllocAlignmentValue, cumulativeSize};
    } else {
      // Adjust the allocation size to consider alignment.
      if (allocOp.alignment()) {
        accessAlignment = createIndexConstant(
            rewriter, loc, allocOp.alignment().getValue().getSExtValue());
        cumulativeSize = rewriter.create<LLVM::SubOp>(
            loc,
            rewriter.create<LLVM::AddOp>(loc, cumulativeSize, accessAlignment),
            one);
      }
      callArgs.push_back(cumulativeSize);
    }
    auto allocFuncSymbol = rewriter.getSymbolRefAttr(allocFunc);
    allocatedBytePtr = rewriter
                           .create<LLVM::CallOp>(loc, getVoidPtrType(),
                                                 allocFuncSymbol, callArgs)
                           .getResult(0);
    // For heap allocations, the allocated pointer is a cast of the byte pointer
    // to the type pointer.
    return rewriter.create<LLVM::BitcastOp>(loc, elementPtrType,
                                            allocatedBytePtr);
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
  // Alignment is performed by allocating `alignment - 1` more bytes than
  // requested and shifting the aligned pointer relative to the allocated
  // memory. If alignment is unspecified, the two pointers are equal.

  // An `alloca` is converted into a definition of a memref descriptor value and
  // an llvm.alloca to allocate the underlying data buffer.
  void rewrite(Operation *op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override {
    MemRefType memRefType = cast<AllocLikeOp>(op).getType();
    auto loc = op->getLoc();

    // Get actual sizes of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.  In case of
    // zero-dimensional memref, assume a scalar (size 1).
    SmallVector<Value, 4> sizes;
    Value cumulativeSize, one;
    getSizes(loc, memRefType, operands, rewriter, sizes, cumulativeSize, one);

    // Allocate the underlying buffer.
    // Value holding the alignment that has to be performed post allocation
    // (in conjunction with allocators that do not support alignment, eg.
    // malloc); nullptr if no such adjustment needs to be performed.
    Value accessAlignment;
    // Byte pointer to the allocated buffer.
    Value allocatedBytePtr;
    Value allocatedTypePtr =
        allocateBuffer(loc, cumulativeSize, op, memRefType, one,
                       accessAlignment, allocatedBytePtr, rewriter);

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(memRefType, strides, offset);
    (void)successStrides;
    assert(succeeded(successStrides) && "unexpected non-strided memref");
    assert(offset != MemRefType::getDynamicStrideOrOffset() &&
           "unexpected dynamic offset");

    // 0-D memref corner case: they have size 1.
    assert(
        ((memRefType.getRank() == 0 && strides.empty() && sizes.size() == 1) ||
         (strides.size() == sizes.size())) &&
        "unexpected number of strides");

    // Create the MemRef descriptor.
    auto memRefDescriptor = createMemRefDescriptor(
        loc, rewriter, memRefType, allocatedTypePtr, allocatedBytePtr,
        accessAlignment, offset, strides, sizes);

    // Return the final value of the descriptor.
    rewriter.replaceOp(op, {memRefDescriptor});
  }

protected:
  /// Use aligned_alloc instead of malloc for all heap allocations.
  bool useAlignedAlloc;
  /// The minimum alignment to use with aligned_alloc (has to be a power of 2).
  uint64_t kMinAlignedAllocAlignment = 16UL;
};

struct AllocOpLowering : public AllocLikeOpLowering<AllocOp> {
  explicit AllocOpLowering(LLVMTypeConverter &converter,
                           bool useAlignedAlloc = false)
      : AllocLikeOpLowering<AllocOp>(converter, useAlignedAlloc) {}
};

using AllocaOpLowering = AllocLikeOpLowering<AllocaOp>;

// A CallOp automatically promotes MemRefType to a sequence of alloca/store and
// passes the pointer to the MemRef across function boundaries.
template <typename CallOpType>
struct CallOpInterfaceLowering : public ConvertOpToLLVMPattern<CallOpType> {
  using ConvertOpToLLVMPattern<CallOpType>::ConvertOpToLLVMPattern;
  using Super = CallOpInterfaceLowering<CallOpType>;
  using Base = ConvertOpToLLVMPattern<CallOpType>;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename CallOpType::Adaptor transformed(operands);
    auto callOp = cast<CallOpType>(op);

    // Pack the result types into a struct.
    Type packedResult;
    unsigned numResults = callOp.getNumResults();
    auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());

    for (Type resType : resultTypes) {
      assert(!resType.isa<UnrankedMemRefType>() &&
             "Returning unranked memref is not supported. Pass result as an"
             "argument instead.");
      (void)resType;
    }

    if (numResults != 0) {
      if (!(packedResult =
                this->typeConverter.packFunctionResults(resultTypes)))
        return failure();
    }

    auto promoted = this->typeConverter.promoteMemRefDescriptors(
        op->getLoc(), /*opOperands=*/op->getOperands(), operands, rewriter);
    auto newOp = rewriter.create<LLVM::CallOp>(op->getLoc(), packedResult,
                                               promoted, op->getAttrs());

    // If < 2 results, packing did not do anything and we can just return.
    if (numResults < 2) {
      rewriter.replaceOp(op, newOp.getResults());
      return success();
    }

    // Otherwise, it had been converted to an operation producing a structure.
    // Extract individual results from the structure and return them as list.
    // TODO(aminim, ntv, riverriddle, zinenko): this seems like patching around
    // a particular interaction between MemRefType and CallOp lowering. Find a
    // way to avoid special casing.
    SmallVector<Value, 4> results;
    results.reserve(numResults);
    for (unsigned i = 0; i < numResults; ++i) {
      auto type = this->typeConverter.convertType(op->getResult(i).getType());
      results.push_back(rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), type, newOp.getOperation()->getResult(0),
          rewriter.getI64ArrayAttr(i)));
    }
    rewriter.replaceOp(op, results);

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
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 1 && "dealloc takes one operand");
    DeallocOp::Adaptor transformed(operands);

    // Insert the `free` declaration if it is not already present.
    auto freeFunc =
        op->getParentOfType<ModuleOp>().lookupSymbol<LLVM::LLVMFuncOp>("free");
    if (!freeFunc) {
      OpBuilder moduleBuilder(op->getParentOfType<ModuleOp>().getBodyRegion());
      freeFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), "free",
          LLVM::LLVMType::getFunctionTy(getVoidType(), getVoidPtrType(),
                                        /*isVarArg=*/false));
    }

    MemRefDescriptor memref(transformed.memref());
    Value casted = rewriter.create<LLVM::BitcastOp>(
        op->getLoc(), getVoidPtrType(),
        memref.allocatedPtr(rewriter, op->getLoc()));
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, ArrayRef<Type>(), rewriter.getSymbolRefAttr(freeFunc), casted);
    return success();
  }
};

// A `rsqrt` is converted into `1 / sqrt`.
struct RsqrtOpLowering : public ConvertOpToLLVMPattern<RsqrtOp> {
  using ConvertOpToLLVMPattern<RsqrtOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    RsqrtOp::Adaptor transformed(operands);
    auto operandType =
        transformed.operand().getType().dyn_cast<LLVM::LLVMType>();

    if (!operandType)
      return failure();

    auto loc = op->getLoc();
    auto resultType = *op->result_type_begin();
    auto floatType = getElementTypeOrSelf(resultType).cast<FloatType>();
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);

    if (!operandType.isArrayTy()) {
      LLVM::ConstantOp one;
      if (operandType.isVectorTy()) {
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
        op, operands, typeConverter,
        [&](LLVM::LLVMType llvmVectorTy, ValueRange operands) {
          auto splatAttr = SplatElementsAttr::get(
              mlir::VectorType::get(
                  {cast<llvm::VectorType>(llvmVectorTy.getUnderlyingType())
                       ->getNumElements()},
                  floatType),
              floatOne);
          auto one =
              rewriter.create<LLVM::ConstantOp>(loc, llvmVectorTy, splatAttr);
          auto sqrt =
              rewriter.create<LLVM::SqrtOp>(loc, llvmVectorTy, operands[0]);
          return rewriter.create<LLVM::FDivOp>(loc, llvmVectorTy, one, sqrt);
        },
        rewriter);
  }
};

struct MemRefCastOpLowering : public ConvertOpToLLVMPattern<MemRefCastOp> {
  using ConvertOpToLLVMPattern<MemRefCastOp>::ConvertOpToLLVMPattern;

  LogicalResult match(Operation *op) const override {
    auto memRefCastOp = cast<MemRefCastOp>(op);
    Type srcType = memRefCastOp.getOperand().getType();
    Type dstType = memRefCastOp.getType();

    // MemRefCastOp reduce to bitcast in the ranked MemRef case and can be used
    // for type erasure. For now they must preserve underlying element type and
    // require source and result type to have the same rank. Therefore, perform
    // a sanity check that the underlying structs are the same. Once op
    // semantics are relaxed we can revisit.
    if (srcType.isa<MemRefType>() && dstType.isa<MemRefType>())
      return success(typeConverter.convertType(srcType) ==
                     typeConverter.convertType(dstType));

    // At least one of the operands is unranked type
    assert(srcType.isa<UnrankedMemRefType>() ||
           dstType.isa<UnrankedMemRefType>());

    // Unranked to unranked cast is disallowed
    return !(srcType.isa<UnrankedMemRefType>() &&
             dstType.isa<UnrankedMemRefType>())
               ? success()
               : failure();
  }

  void rewrite(Operation *op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override {
    auto memRefCastOp = cast<MemRefCastOp>(op);
    MemRefCastOp::Adaptor transformed(operands);

    auto srcType = memRefCastOp.getOperand().getType();
    auto dstType = memRefCastOp.getType();
    auto targetStructType = typeConverter.convertType(memRefCastOp.getType());
    auto loc = op->getLoc();

    // MemRefCastOp reduce to bitcast in the ranked MemRef case.
    if (srcType.isa<MemRefType>() && dstType.isa<MemRefType>()) {
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, targetStructType,
                                                   transformed.source());
    } else if (srcType.isa<MemRefType>() && dstType.isa<UnrankedMemRefType>()) {
      // Casting ranked to unranked memref type
      // Set the rank in the destination from the memref type
      // Allocate space on the stack and copy the src memref descriptor
      // Set the ptr in the destination to the stack space
      auto srcMemRefType = srcType.cast<MemRefType>();
      int64_t rank = srcMemRefType.getRank();
      // ptr = AllocaOp sizeof(MemRefDescriptor)
      auto ptr = typeConverter.promoteOneMemRefDescriptor(
          loc, transformed.source(), rewriter);
      // voidptr = BitCastOp srcType* to void*
      auto voidPtr =
          rewriter.create<LLVM::BitcastOp>(loc, getVoidPtrType(), ptr)
              .getResult();
      // rank = ConstantOp srcRank
      auto rankVal = rewriter.create<LLVM::ConstantOp>(
          loc, typeConverter.convertType(rewriter.getIntegerType(64)),
          rewriter.getI64IntegerAttr(rank));
      // undef = UndefOp
      UnrankedMemRefDescriptor memRefDesc =
          UnrankedMemRefDescriptor::undef(rewriter, loc, targetStructType);
      // d1 = InsertValueOp undef, rank, 0
      memRefDesc.setRank(rewriter, loc, rankVal);
      // d2 = InsertValueOp d1, voidptr, 1
      memRefDesc.setMemRefDescPtr(rewriter, loc, voidPtr);
      rewriter.replaceOp(op, (Value)memRefDesc);

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
                  loc, targetStructType.cast<LLVM::LLVMType>().getPointerTo(),
                  ptr)
              .getResult();
      // struct = LoadOp castPtr
      auto loadOp = rewriter.create<LLVM::LoadOp>(loc, castPtr);
      rewriter.replaceOp(op, loadOp.getResult());
    } else {
      llvm_unreachable("Unsupported unranked memref to unranked memref cast");
    }
  }
};

struct DialectCastOpLowering
    : public ConvertOpToLLVMPattern<LLVM::DialectCastOp> {
  using ConvertOpToLLVMPattern<LLVM::DialectCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto castOp = cast<LLVM::DialectCastOp>(op);
    LLVM::DialectCastOp::Adaptor transformed(operands);
    if (transformed.in().getType() !=
        typeConverter.convertType(castOp.getType())) {
      return failure();
    }
    rewriter.replaceOp(op, transformed.in());
    return success();
  }
};

// A `dim` is converted to a constant for static sizes and to an access to the
// size stored in the memref descriptor for dynamic sizes.
struct DimOpLowering : public ConvertOpToLLVMPattern<DimOp> {
  using ConvertOpToLLVMPattern<DimOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto dimOp = cast<DimOp>(op);
    DimOp::Adaptor transformed(operands);
    MemRefType type = dimOp.memrefOrTensor().getType().cast<MemRefType>();

    Optional<int64_t> index = dimOp.getConstantIndex();
    if (!index.hasValue()) {
      // TODO: Implement this lowering.
      return failure();
    }

    int64_t i = index.getValue();
    // Extract dynamic size from the memref descriptor.
    if (type.isDynamicDim(i))
      rewriter.replaceOp(op, {MemRefDescriptor(transformed.memrefOrTensor())
                                  .size(rewriter, op->getLoc(), i)});
    else
      // Use constant for static size.
      rewriter.replaceOp(
          op, createIndexConstant(rewriter, op->getLoc(), type.getDimSize(i)));

    return success();
  }
};

// Common base for load and store operations on MemRefs.  Restricts the match
// to supported MemRef types.  Provides functionality to emit code accessing a
// specific element of the underlying data buffer.
template <typename Derived>
struct LoadStoreOpLowering : public ConvertOpToLLVMPattern<Derived> {
  using ConvertOpToLLVMPattern<Derived>::ConvertOpToLLVMPattern;
  using Base = LoadStoreOpLowering<Derived>;

  LogicalResult match(Operation *op) const override {
    MemRefType type = cast<Derived>(op).getMemRefType();
    return isSupportedMemRefType(type) ? success() : failure();
  }
};

// Load operation is lowered to obtaining a pointer to the indexed element
// and loading it.
struct LoadOpLowering : public LoadStoreOpLowering<LoadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loadOp = cast<LoadOp>(op);
    LoadOp::Adaptor transformed(operands);
    auto type = loadOp.getMemRefType();

    Value dataPtr = getDataPtr(op->getLoc(), type, transformed.memref(),
                               transformed.indices(), rewriter, getModule());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, dataPtr);
    return success();
  }
};

// Store operation is lowered to obtaining a pointer to the indexed element,
// and storing the given value to it.
struct StoreOpLowering : public LoadStoreOpLowering<StoreOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = cast<StoreOp>(op).getMemRefType();
    StoreOp::Adaptor transformed(operands);

    Value dataPtr = getDataPtr(op->getLoc(), type, transformed.memref(),
                               transformed.indices(), rewriter, getModule());
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
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto prefetchOp = cast<PrefetchOp>(op);
    PrefetchOp::Adaptor transformed(operands);
    auto type = prefetchOp.getMemRefType();

    Value dataPtr = getDataPtr(op->getLoc(), type, transformed.memref(),
                               transformed.indices(), rewriter, getModule());

    // Replace with llvm.prefetch.
    auto llvmI32Type = typeConverter.convertType(rewriter.getIntegerType(32));
    auto isWrite = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), llvmI32Type,
        rewriter.getI32IntegerAttr(prefetchOp.isWrite()));
    auto localityHint = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), llvmI32Type,
        rewriter.getI32IntegerAttr(prefetchOp.localityHint().getZExtValue()));
    auto isData = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), llvmI32Type,
        rewriter.getI32IntegerAttr(prefetchOp.isDataCache()));

    rewriter.replaceOpWithNewOp<LLVM::Prefetch>(op, dataPtr, isWrite,
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
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    IndexCastOpAdaptor transformed(operands);
    auto indexCastOp = cast<IndexCastOp>(op);

    auto targetType =
        this->typeConverter.convertType(indexCastOp.getResult().getType())
            .cast<LLVM::LLVMType>();
    auto sourceType = transformed.in().getType().cast<LLVM::LLVMType>();
    unsigned targetBits = targetType.getUnderlyingType()->getIntegerBitWidth();
    unsigned sourceBits = sourceType.getUnderlyingType()->getIntegerBitWidth();

    if (targetBits == sourceBits)
      rewriter.replaceOp(op, transformed.in());
    else if (targetBits < sourceBits)
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, targetType,
                                                 transformed.in());
    else
      rewriter.replaceOpWithNewOp<LLVM::SExtOp>(op, targetType,
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
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto cmpiOp = cast<CmpIOp>(op);
    CmpIOpAdaptor transformed(operands);

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, typeConverter.convertType(cmpiOp.getResult().getType()),
        rewriter.getI64IntegerAttr(static_cast<int64_t>(
            convertCmpPredicate<LLVM::ICmpPredicate>(cmpiOp.getPredicate()))),
        transformed.lhs(), transformed.rhs());

    return success();
  }
};

struct CmpFOpLowering : public ConvertOpToLLVMPattern<CmpFOp> {
  using ConvertOpToLLVMPattern<CmpFOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto cmpfOp = cast<CmpFOp>(op);
    CmpFOpAdaptor transformed(operands);

    rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
        op, typeConverter.convertType(cmpfOp.getResult().getType()),
        rewriter.getI64IntegerAttr(static_cast<int64_t>(
            convertCmpPredicate<LLVM::FCmpPredicate>(cmpfOp.getPredicate()))),
        transformed.lhs(), transformed.rhs());

    return success();
  }
};

struct SIToFPLowering
    : public OneToOneConvertToLLVMPattern<SIToFPOp, LLVM::SIToFPOp> {
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

struct FPTruncLowering
    : public OneToOneConvertToLLVMPattern<FPTruncOp, LLVM::FPTruncOp> {
  using Super::Super;
};

struct SignExtendIOpLowering
    : public OneToOneConvertToLLVMPattern<SignExtendIOp, LLVM::SExtOp> {
  using Super::Super;
};

struct TruncateIOpLowering
    : public OneToOneConvertToLLVMPattern<TruncateIOp, LLVM::TruncOp> {
  using Super::Super;
};

struct ZeroExtendIOpLowering
    : public OneToOneConvertToLLVMPattern<ZeroExtendIOp, LLVM::ZExtOp> {
  using Super::Super;
};

// Base class for LLVM IR lowering terminator operations with successors.
template <typename SourceOp, typename TargetOp>
struct OneToOneLLVMTerminatorLowering
    : public ConvertOpToLLVMPattern<SourceOp> {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using Super = OneToOneLLVMTerminatorLowering<SourceOp, TargetOp>;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
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
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numArguments = op->getNumOperands();

    // If ReturnOp has 0 or 1 operand, create it and return immediately.
    if (numArguments == 0) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
          op, ArrayRef<Type>(), ArrayRef<Value>(), op->getAttrs());
      return success();
    }
    if (numArguments == 1) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
          op, ArrayRef<Type>(), operands.front(), op->getAttrs());
      return success();
    }

    // Otherwise, we need to pack the arguments into an LLVM struct type before
    // returning.
    auto packedType = typeConverter.packFunctionResults(
        llvm::to_vector<4>(op->getOperandTypes()));

    Value packed = rewriter.create<LLVM::UndefOp>(op->getLoc(), packedType);
    for (unsigned i = 0; i < numArguments; ++i) {
      packed = rewriter.create<LLVM::InsertValueOp>(
          op->getLoc(), packedType, packed, operands[i],
          rewriter.getI64ArrayAttr(i));
    }
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ArrayRef<Type>(), packed,
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
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto splatOp = cast<SplatOp>(op);
    VectorType resultType = splatOp.getType().dyn_cast<VectorType>();
    if (!resultType || resultType.getRank() != 1)
      return failure();

    // First insert it into an undef vector so we can shuffle it.
    auto vectorType = typeConverter.convertType(splatOp.getType());
    Value undef = rewriter.create<LLVM::UndefOp>(op->getLoc(), vectorType);
    auto zero = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), typeConverter.convertType(rewriter.getIntegerType(32)),
        rewriter.getZeroAttr(rewriter.getIntegerType(32)));

    auto v = rewriter.create<LLVM::InsertElementOp>(
        op->getLoc(), vectorType, undef, splatOp.getOperand(), zero);

    int64_t width = splatOp.getType().cast<VectorType>().getDimSize(0);
    SmallVector<int32_t, 4> zeroValues(width, 0);

    // Shuffle the value across the desired number of elements.
    ArrayAttr zeroAttrs = rewriter.getI32ArrayAttr(zeroValues);
    rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(op, v, undef, zeroAttrs);
    return success();
  }
};

// The Splat operation is lowered to an insertelement + a shufflevector
// operation. Splat to only 2+-d vector result types are lowered by the
// SplatNdOpLowering, the 1-d case is handled by SplatOpLowering.
struct SplatNdOpLowering : public ConvertOpToLLVMPattern<SplatOp> {
  using ConvertOpToLLVMPattern<SplatOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto splatOp = cast<SplatOp>(op);
    SplatOp::Adaptor adaptor(operands);
    VectorType resultType = splatOp.getType().dyn_cast<VectorType>();
    if (!resultType || resultType.getRank() == 1)
      return failure();

    // First insert it into an undef vector so we can shuffle it.
    auto loc = op->getLoc();
    auto vectorTypeInfo = extractNDVectorTypeInfo(resultType, typeConverter);
    auto llvmArrayTy = vectorTypeInfo.llvmArrayTy;
    auto llvmVectorTy = vectorTypeInfo.llvmVectorTy;
    if (!llvmArrayTy || !llvmVectorTy)
      return failure();

    // Construct returned value.
    Value desc = rewriter.create<LLVM::UndefOp>(loc, llvmArrayTy);

    // Construct a 1-D vector with the splatted value that we insert in all the
    // places within the returned descriptor.
    Value vdesc = rewriter.create<LLVM::UndefOp>(loc, llvmVectorTy);
    auto zero = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter.convertType(rewriter.getIntegerType(32)),
        rewriter.getZeroAttr(rewriter.getIntegerType(32)));
    Value v = rewriter.create<LLVM::InsertElementOp>(loc, llvmVectorTy, vdesc,
                                                     adaptor.input(), zero);

    // Shuffle the value across the desired number of elements.
    int64_t width = resultType.getDimSize(resultType.getRank() - 1);
    SmallVector<int32_t, 4> zeroValues(width, 0);
    ArrayAttr zeroAttrs = rewriter.getI32ArrayAttr(zeroValues);
    v = rewriter.create<LLVM::ShuffleVectorOp>(loc, v, v, zeroAttrs);

    // Iterate of linear index, convert to coords space and insert splatted 1-D
    // vector in each position.
    nDVectorIterate(vectorTypeInfo, rewriter, [&](ArrayAttr position) {
      desc = rewriter.create<LLVM::InsertValueOp>(loc, llvmArrayTy, desc, v,
                                                  position);
    });
    rewriter.replaceOp(op, desc);
    return success();
  }
};

/// Conversion pattern that transforms a subview op into:
///   1. An `llvm.mlir.undef` operation to create a memref descriptor
///   2. Updates to the descriptor to introduce the data ptr, offset, size
///      and stride.
/// The subview op is replaced by the descriptor.
struct SubViewOpLowering : public ConvertOpToLLVMPattern<SubViewOp> {
  using ConvertOpToLLVMPattern<SubViewOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto subViewOp = cast<SubViewOp>(op);

    auto sourceMemRefType = subViewOp.source().getType().cast<MemRefType>();
    auto sourceElementTy =
        typeConverter.convertType(sourceMemRefType.getElementType())
            .dyn_cast_or_null<LLVM::LLVMType>();

    auto viewMemRefType = subViewOp.getType();
    auto targetElementTy =
        typeConverter.convertType(viewMemRefType.getElementType())
            .dyn_cast<LLVM::LLVMType>();
    auto targetDescTy = typeConverter.convertType(viewMemRefType)
                            .dyn_cast_or_null<LLVM::LLVMType>();
    if (!sourceElementTy || !targetDescTy)
      return failure();

    // Extract the offset and strides from the type.
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(viewMemRefType, strides, offset);
    if (failed(successStrides))
      return failure();

    // Create the descriptor.
    if (!operands.front().getType().isa<LLVM::LLVMType>())
      return failure();
    MemRefDescriptor sourceMemRef(operands.front());
    auto targetMemRef = MemRefDescriptor::undef(rewriter, loc, targetDescTy);

    // Copy the buffer pointer from the old descriptor to the new one.
    Value extracted = sourceMemRef.allocatedPtr(rewriter, loc);
    Value bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc, targetElementTy.getPointerTo(viewMemRefType.getMemorySpace()),
        extracted);
    targetMemRef.setAllocatedPtr(rewriter, loc, bitcastPtr);

    // Copy the buffer pointer from the old descriptor to the new one.
    extracted = sourceMemRef.alignedPtr(rewriter, loc);
    bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc, targetElementTy.getPointerTo(viewMemRefType.getMemorySpace()),
        extracted);
    targetMemRef.setAlignedPtr(rewriter, loc, bitcastPtr);

    // Extract strides needed to compute offset.
    SmallVector<Value, 4> strideValues;
    strideValues.reserve(viewMemRefType.getRank());
    for (int i = 0, e = viewMemRefType.getRank(); i < e; ++i)
      strideValues.push_back(sourceMemRef.stride(rewriter, loc, i));

    // Offset.
    auto llvmIndexType = typeConverter.convertType(rewriter.getIndexType());
    if (!ShapedType::isDynamicStrideOrOffset(offset)) {
      targetMemRef.setConstantOffset(rewriter, loc, offset);
    } else {
      Value baseOffset = sourceMemRef.offset(rewriter, loc);
      for (unsigned i = 0, e = viewMemRefType.getRank(); i < e; ++i) {
        Value offset =
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
    for (int i = viewMemRefType.getRank() - 1; i >= 0; --i) {
      Value size =
          subViewOp.isDynamicSize(i)
              ? operands[subViewOp.getIndexOfDynamicSize(i)]
              : rewriter.create<LLVM::ConstantOp>(
                    loc, llvmIndexType,
                    rewriter.getI64IntegerAttr(subViewOp.getStaticSize(i)));
      targetMemRef.setSize(rewriter, loc, i, size);
      Value stride;
      if (!ShapedType::isDynamicStrideOrOffset(strides[i])) {
        stride = rewriter.create<LLVM::ConstantOp>(
            loc, llvmIndexType, rewriter.getI64IntegerAttr(strides[i]));
      } else {
        stride =
            subViewOp.isDynamicStride(i)
                ? operands[subViewOp.getIndexOfDynamicStride(i)]
                : rewriter.create<LLVM::ConstantOp>(
                      loc, llvmIndexType,
                      rewriter.getI64IntegerAttr(subViewOp.getStaticStride(i)));
        stride = rewriter.create<LLVM::MulOp>(loc, stride, strideValues[i]);
      }
      targetMemRef.setStride(rewriter, loc, i, stride);
    }

    rewriter.replaceOp(op, {targetMemRef});
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
    if (strides[idx] != MemRefType::getDynamicStrideOrOffset())
      return createIndexConstant(rewriter, loc, strides[idx]);
    if (nextSize)
      return runningStride
                 ? rewriter.create<LLVM::MulOp>(loc, runningStride, nextSize)
                 : nextSize;
    assert(!runningStride);
    return createIndexConstant(rewriter, loc, 1);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto viewOp = cast<ViewOp>(op);
    ViewOpAdaptor adaptor(operands);

    auto viewMemRefType = viewOp.getType();
    auto targetElementTy =
        typeConverter.convertType(viewMemRefType.getElementType())
            .dyn_cast<LLVM::LLVMType>();
    auto targetDescTy =
        typeConverter.convertType(viewMemRefType).dyn_cast<LLVM::LLVMType>();
    if (!targetDescTy)
      return op->emitWarning("Target descriptor type not converted to LLVM"),
             failure();

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(viewMemRefType, strides, offset);
    if (failed(successStrides))
      return op->emitWarning("cannot cast to non-strided shape"), failure();
    assert(offset == 0 && "expected offset to be 0");

    // Create the descriptor.
    MemRefDescriptor sourceMemRef(adaptor.source());
    auto targetMemRef = MemRefDescriptor::undef(rewriter, loc, targetDescTy);

    // Field 1: Copy the allocated pointer, used for malloc/free.
    Value allocatedPtr = sourceMemRef.allocatedPtr(rewriter, loc);
    Value bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc, targetElementTy.getPointerTo(), allocatedPtr);
    targetMemRef.setAllocatedPtr(rewriter, loc, bitcastPtr);

    // Field 2: Copy the actual aligned pointer to payload.
    Value alignedPtr = sourceMemRef.alignedPtr(rewriter, loc);
    alignedPtr = rewriter.create<LLVM::GEPOp>(loc, alignedPtr.getType(),
                                              alignedPtr, adaptor.byte_shift());
    bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc, targetElementTy.getPointerTo(), alignedPtr);
    targetMemRef.setAlignedPtr(rewriter, loc, bitcastPtr);

    // Field 3: The offset in the resulting type must be 0. This is because of
    // the type change: an offset on srcType* may not be expressible as an
    // offset on dstType*.
    targetMemRef.setOffset(rewriter, loc,
                           createIndexConstant(rewriter, loc, offset));

    // Early exit for 0-D corner case.
    if (viewMemRefType.getRank() == 0)
      return rewriter.replaceOp(op, {targetMemRef}), success();

    // Fields 4 and 5: Update sizes and strides.
    if (strides.back() != 1)
      return op->emitWarning("cannot cast to non-contiguous shape"), failure();
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

    rewriter.replaceOp(op, {targetMemRef});
    return success();
  }
};

struct AssumeAlignmentOpLowering
    : public ConvertOpToLLVMPattern<AssumeAlignmentOp> {
  using ConvertOpToLLVMPattern<AssumeAlignmentOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    AssumeAlignmentOp::Adaptor transformed(operands);
    Value memref = transformed.memref();
    unsigned alignment = cast<AssumeAlignmentOp>(op).alignment().getZExtValue();

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
    Value zero =
        createIndexAttrConstant(rewriter, op->getLoc(), getIndexType(), 0);
    Value mask = createIndexAttrConstant(rewriter, op->getLoc(), getIndexType(),
                                         alignment - 1);
    Value ptrValue =
        rewriter.create<LLVM::PtrToIntOp>(op->getLoc(), getIndexType(), ptr);
    rewriter.create<LLVM::AssumeOp>(
        op->getLoc(),
        rewriter.create<LLVM::ICmpOp>(
            op->getLoc(), LLVM::ICmpPredicate::eq,
            rewriter.create<LLVM::AndOp>(op->getLoc(), ptrValue, mask), zero));

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
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto atomicOp = cast<AtomicRMWOp>(op);
    auto maybeKind = matchSimpleAtomicOp(atomicOp);
    if (!maybeKind)
      return failure();
    AtomicRMWOp::Adaptor adaptor(operands);
    auto resultType = adaptor.value().getType();
    auto memRefType = atomicOp.getMemRefType();
    auto dataPtr = getDataPtr(op->getLoc(), memRefType, adaptor.memref(),
                              adaptor.indices(), rewriter, getModule());
    rewriter.replaceOpWithNewOp<LLVM::AtomicRMWOp>(
        op, resultType, *maybeKind, dataPtr, adaptor.value(),
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
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto atomicOp = cast<GenericAtomicRMWOp>(op);

    auto loc = op->getLoc();
    GenericAtomicRMWOp::Adaptor adaptor(operands);
    LLVM::LLVMType valueType =
        typeConverter.convertType(atomicOp.getResult().getType())
            .cast<LLVM::LLVMType>();

    // Split the block into initial, loop, and ending parts.
    auto *initBlock = rewriter.getInsertionBlock();
    auto *loopBlock =
        rewriter.createBlock(initBlock->getParent(),
                             std::next(Region::iterator(initBlock)), valueType);
    auto *endBlock = rewriter.createBlock(
        loopBlock->getParent(), std::next(Region::iterator(loopBlock)));

    // Operations range to be moved to `endBlock`.
    auto opsToMoveStart = atomicOp.getOperation()->getIterator();
    auto opsToMoveEnd = initBlock->back().getIterator();

    // Compute the loaded value and branch to the loop block.
    rewriter.setInsertionPointToEnd(initBlock);
    auto memRefType = atomicOp.memref().getType().cast<MemRefType>();
    auto dataPtr = getDataPtr(loc, memRefType, adaptor.memref(),
                              adaptor.indices(), rewriter, getModule());
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
    auto boolType = LLVM::LLVMType::getInt1Ty(&getDialect());
    auto pairType = LLVM::LLVMType::getStructTy(valueType, boolType);
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
    MoveOpsRange(atomicOp.getResult(), newLoaded, std::next(opsToMoveStart),
                 std::next(opsToMoveEnd), rewriter);

    // The 'result' of the atomic_rmw op is the newly loaded value.
    rewriter.replaceOp(op, {newLoaded});

    return success();
  }

private:
  // Clones a segment of ops [start, end) and erases the original.
  void MoveOpsRange(ValueRange oldResult, ValueRange newResult,
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
      AddCFOpLowering,
      AddFOpLowering,
      AddIOpLowering,
      AllocaOpLowering,
      AndOpLowering,
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
      ConstLLVMOpLowering,
      CreateComplexOpLowering,
      DialectCastOpLowering,
      DivFOpLowering,
      ExpOpLowering,
      Exp2OpLowering,
      GenericAtomicRMWOpLowering,
      LogOpLowering,
      Log10OpLowering,
      Log2OpLowering,
      FPExtLowering,
      FPToSILowering,
      FPTruncLowering,
      ImOpLowering,
      IndexCastOpLowering,
      MulFOpLowering,
      MulIOpLowering,
      NegFOpLowering,
      OrOpLowering,
      PrefetchOpLowering,
      ReOpLowering,
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
      SubCFOpLowering,
      SubFOpLowering,
      SubIOpLowering,
      TruncateIOpLowering,
      UnsignedDivIOpLowering,
      UnsignedRemIOpLowering,
      UnsignedShiftRightOpLowering,
      XOrOpLowering,
      ZeroExtendIOpLowering>(converter);
  // clang-format on
}

void mlir::populateStdToLLVMMemoryConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    bool useAlignedAlloc) {
  // clang-format off
  patterns.insert<
      AssumeAlignmentOpLowering,
      DeallocOpLowering,
      DimOpLowering,
      LoadOpLowering,
      MemRefCastOpLowering,
      StoreOpLowering,
      SubViewOpLowering,
      ViewOpLowering>(converter);
  patterns.insert<
      AllocOpLowering
      >(converter, useAlignedAlloc);
  // clang-format on
}

void mlir::populateStdToLLVMDefaultFuncOpConversionPattern(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    bool emitCWrappers) {
  patterns.insert<FuncOpConversion>(converter, emitCWrappers);
}

void mlir::populateStdToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    bool emitCWrappers, bool useAlignedAlloc) {
  populateStdToLLVMDefaultFuncOpConversionPattern(converter, patterns,
                                                  emitCWrappers);
  populateStdToLLVMNonMemoryConversionPatterns(converter, patterns);
  populateStdToLLVMMemoryConversionPatterns(converter, patterns,
                                            useAlignedAlloc);
}

static void populateStdToLLVMBarePtrFuncOpConversionPattern(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  patterns.insert<BarePtrFuncOpConversion>(converter);
}

void mlir::populateStdToLLVMBarePtrConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    bool useAlignedAlloc) {
  populateStdToLLVMBarePtrFuncOpConversionPattern(converter, patterns);
  populateStdToLLVMNonMemoryConversionPatterns(converter, patterns);
  populateStdToLLVMMemoryConversionPatterns(converter, patterns,
                                            useAlignedAlloc);
}

// Create an LLVM IR structure type if there is more than one result.
Type LLVMTypeConverter::packFunctionResults(ArrayRef<Type> types) {
  assert(!types.empty() && "expected non-empty list of type");

  if (types.size() == 1)
    return convertType(types.front());

  SmallVector<LLVM::LLVMType, 8> resultTypes;
  resultTypes.reserve(types.size());
  for (auto t : types) {
    auto converted = convertType(t).dyn_cast<LLVM::LLVMType>();
    if (!converted)
      return {};
    resultTypes.push_back(converted);
  }

  return LLVM::LLVMType::getStructTy(llvmDialect, resultTypes);
}

Value LLVMTypeConverter::promoteOneMemRefDescriptor(Location loc, Value operand,
                                                    OpBuilder &builder) {
  auto *context = builder.getContext();
  auto int64Ty = LLVM::LLVMType::getInt64Ty(getDialect());
  auto indexType = IndexType::get(context);
  // Alloca with proper alignment. We do not expect optimizations of this
  // alloca op and so we omit allocating at the entry block.
  auto ptrType = operand.getType().cast<LLVM::LLVMType>().getPointerTo();
  Value one = builder.create<LLVM::ConstantOp>(loc, int64Ty,
                                               IntegerAttr::get(indexType, 1));
  Value allocated =
      builder.create<LLVM::AllocaOp>(loc, ptrType, one, /*alignment=*/0);
  // Store into the alloca'ed descriptor.
  builder.create<LLVM::StoreOp>(loc, operand, allocated);
  return allocated;
}

SmallVector<Value, 4>
LLVMTypeConverter::promoteMemRefDescriptors(Location loc, ValueRange opOperands,
                                            ValueRange operands,
                                            OpBuilder &builder) {
  SmallVector<Value, 4> promotedOperands;
  promotedOperands.reserve(operands.size());
  for (auto it : llvm::zip(opOperands, operands)) {
    auto operand = std::get<0>(it);
    auto llvmOperand = std::get<1>(it);

    if (operand.getType().isa<UnrankedMemRefType>()) {
      UnrankedMemRefDescriptor::unpack(builder, loc, llvmOperand,
                                       promotedOperands);
      continue;
    }
    if (auto memrefType = operand.getType().dyn_cast<MemRefType>()) {
      MemRefDescriptor::unpack(builder, loc, llvmOperand,
                               operand.getType().cast<MemRefType>(),
                               promotedOperands);
      continue;
    }

    promotedOperands.push_back(operand);
  }
  return promotedOperands;
}

namespace {
/// A pass converting MLIR operations into the LLVM IR dialect.
struct LLVMLoweringPass : public ConvertStandardToLLVMBase<LLVMLoweringPass> {
  LLVMLoweringPass() = default;
  LLVMLoweringPass(bool useBarePtrCallConv, bool emitCWrappers,
                   unsigned indexBitwidth, bool useAlignedAlloc) {
    this->useBarePtrCallConv = useBarePtrCallConv;
    this->emitCWrappers = emitCWrappers;
    this->indexBitwidth = indexBitwidth;
    this->useAlignedAlloc = useAlignedAlloc;
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

    ModuleOp m = getOperation();

    LLVMTypeConverterCustomization customs;
    customs.funcArgConverter = useBarePtrCallConv ? barePtrFuncArgTypeConverter
                                                  : structFuncArgTypeConverter;
    customs.indexBitwidth = indexBitwidth;
    LLVMTypeConverter typeConverter(&getContext(), customs);

    OwningRewritePatternList patterns;
    if (useBarePtrCallConv)
      populateStdToLLVMBarePtrConversionPatterns(typeConverter, patterns,
                                                 useAlignedAlloc);
    else
      populateStdToLLVMConversionPatterns(typeConverter, patterns,
                                          emitCWrappers, useAlignedAlloc);

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(m, target, patterns, &typeConverter)))
      signalPassFailure();
  }
};
} // end namespace

mlir::LLVMConversionTarget::LLVMConversionTarget(MLIRContext &ctx)
    : ConversionTarget(ctx) {
  this->addLegalDialect<LLVM::LLVMDialect>();
  this->addIllegalOp<LLVM::DialectCastOp>();
  this->addIllegalOp<TanhOp>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createLowerToLLVMPass(const LowerToLLVMOptions &options) {
  return std::make_unique<LLVMLoweringPass>(
      options.useBarePtrCallConv, options.emitCWrappers, options.indexBitwidth,
      options.useAlignedAlloc);
}
