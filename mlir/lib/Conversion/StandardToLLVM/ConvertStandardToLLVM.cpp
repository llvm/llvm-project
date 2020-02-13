//===- ConvertStandardToLLVM.cpp - Standard to LLVM dialect conversion-----===//
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

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"

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
LLVMTypeConverterCustomization::LLVMTypeConverterCustomization() {
  funcArgConverter = structFuncArgTypeConverter;
}

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
}

/// Get the LLVM context.
llvm::LLVMContext &LLVMTypeConverter::getLLVMContext() {
  return module->getContext();
}

LLVM::LLVMType LLVMTypeConverter::getIndexType() {
  return LLVM::LLVMType::getIntNTy(
      llvmDialect, module->getDataLayout().getPointerSizeInBits());
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
    auto *mlirContext = llvmDialect->getContext();
    return emitError(UnknownLoc::get(mlirContext), "unsupported type: BF16"),
           Type();
  }
  default:
    llvm_unreachable("non-float type in convertFloatType");
  }
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

/// Creates descriptor structs from individual values constituting them.
Operation *LLVMTypeConverter::materializeConversion(PatternRewriter &rewriter,
                                                    Type type,
                                                    ArrayRef<Value> values,
                                                    Location loc) {
  if (auto unrankedMemRefType = type.dyn_cast<UnrankedMemRefType>())
    return UnrankedMemRefDescriptor::pack(rewriter, loc, *this,
                                          unrankedMemRefType, values)
        .getDefiningOp();

  auto memRefType = type.dyn_cast<MemRefType>();
  assert(memRefType && "1->N conversion is only supported for memrefs");
  return MemRefDescriptor::pack(rewriter, loc, *this, memRefType, values)
      .getDefiningOp();
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

// Dispatch based on the actual type.  Return null type on error.
Type LLVMTypeConverter::convertStandardType(Type t) {
  return TypeSwitch<Type, Type>(t)
      .Case([&](FloatType type) { return convertFloatType(type); })
      .Case([&](FunctionType type) { return convertFunctionType(type); })
      .Case([&](IndexType type) { return convertIndexType(type); })
      .Case([&](IntegerType type) { return convertIntegerType(type); })
      .Case([&](MemRefType type) { return convertMemRefType(type); })
      .Case([&](UnrankedMemRefType type) {
        return convertUnrankedMemRefType(type);
      })
      .Case([&](VectorType type) { return convertVectorType(type); })
      .Case([](LLVM::LLVMType type) { return type; })
      .Default([](Type) { return Type(); });
}

LLVMOpLowering::LLVMOpLowering(StringRef rootOpName, MLIRContext *context,
                               LLVMTypeConverter &lowering_,
                               PatternBenefit benefit)
    : ConversionPattern(rootOpName, benefit, context), lowering(lowering_) {}

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

/// Builds IR extracting the pos-th size from the descriptor.
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

namespace {
// Base class for Standard to LLVM IR op conversions.  Matches the Op type
// provided as template argument.  Carries a reference to the LLVM dialect in
// case it is necessary for rewriters.
template <typename SourceOp>
class LLVMLegalizationPattern : public LLVMOpLowering {
public:
  // Construct a conversion pattern.
  explicit LLVMLegalizationPattern(LLVM::LLVMDialect &dialect_,
                                   LLVMTypeConverter &lowering_)
      : LLVMOpLowering(SourceOp::getOperationName(), dialect_.getContext(),
                       lowering_),
        dialect(dialect_) {}

  // Get the LLVM IR dialect.
  LLVM::LLVMDialect &getDialect() const { return dialect; }
  // Get the LLVM context.
  llvm::LLVMContext &getContext() const { return dialect.getLLVMContext(); }
  // Get the LLVM module in which the types are constructed.
  llvm::Module &getModule() const { return dialect.getLLVMModule(); }

  // Get the MLIR type wrapping the LLVM integer type whose bit width is defined
  // by the pointer size used in the LLVM module.
  LLVM::LLVMType getIndexType() const {
    return LLVM::LLVMType::getIntNTy(
        &dialect, getModule().getDataLayout().getPointerSizeInBits());
  }

  LLVM::LLVMType getVoidType() const {
    return LLVM::LLVMType::getVoidTy(&dialect);
  }

  // Get the MLIR type wrapping the LLVM i8* type.
  LLVM::LLVMType getVoidPtrType() const {
    return LLVM::LLVMType::getInt8PtrTy(&dialect);
  }

  // Create an LLVM IR pseudo-operation defining the given index constant.
  Value createIndexConstant(ConversionPatternRewriter &builder, Location loc,
                            uint64_t value) const {
    return createIndexAttrConstant(builder, loc, getIndexType(), value);
  }

protected:
  LLVM::LLVMDialect &dialect;
};

/// Only retain those attributes that are not constructed by
/// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
/// attributes.
static void filterFuncAttributes(ArrayRef<NamedAttribute> attrs,
                                 bool filterArgAttrs,
                                 SmallVectorImpl<NamedAttribute> &result) {
  for (const auto &attr : attrs) {
    if (attr.first.is(SymbolTable::getSymbolAttrName()) ||
        attr.first.is(impl::getTypeAttrName()) ||
        attr.first.is("std.varargs") ||
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

  // Get a ValueRange containing argument types. Note that ValueRange is
  // currently not constructible from a pair of iterators pointing to
  // BlockArgument.
  FunctionType type = funcOp.getType();
  SmallVector<Value, 8> args;
  args.reserve(type.getNumInputs());
  auto wrapperArgIters = newFuncOp.getArguments();
  SmallVector<Value, 8> wrapperArgs(wrapperArgIters.begin(),
                                    wrapperArgIters.end());
  ValueRange wrapperArgsRange(wrapperArgs);

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

struct FuncOpConversionBase : public LLVMLegalizationPattern<FuncOp> {
protected:
  using LLVMLegalizationPattern<FuncOp>::LLVMLegalizationPattern;
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
    auto llvmType = lowering.convertFunctionSignature(
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
      for (size_t j = mapping->inputNo; j < mapping->size; ++j) {
        impl::getArgAttrName(j, name);
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
struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(LLVM::LLVMDialect &dialect, LLVMTypeConverter &converter,
                   bool emitCWrappers)
      : FuncOpConversionBase(dialect, converter), emitWrappers(emitCWrappers) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = cast<FuncOp>(op);

    auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (emitWrappers) {
      if (newFuncOp.isExternal())
        wrapExternalFunction(rewriter, op->getLoc(), lowering, funcOp,
                             newFuncOp);
      else
        wrapForExternalCallers(rewriter, op->getLoc(), lowering, funcOp,
                               newFuncOp);
    }

    rewriter.eraseOp(op);
    return matchSuccess();
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

  PatternMatchResult
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
      return matchSuccess();
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
            rewriter, funcLoc, lowering, memrefType, arg);
        rewriter.replaceOp(placeHolder.getDefiningOp(), {desc});
      }
    }

    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

//////////////// Support for Lowering operations on n-D vectors ////////////////
namespace {
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

// Basic lowering implementation for one-to-one rewriting from Standard Ops to
// LLVM Dialect Ops.
template <typename SourceOp, typename TargetOp>
struct OneToOneLLVMOpLowering : public LLVMLegalizationPattern<SourceOp> {
  using LLVMLegalizationPattern<SourceOp>::LLVMLegalizationPattern;
  using Super = OneToOneLLVMOpLowering<SourceOp, TargetOp>;

  // Convert the type of the result to an LLVM type, pass operands as is,
  // preserve attributes.
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numResults = op->getNumResults();

    Type packedType;
    if (numResults != 0) {
      packedType = this->lowering.packFunctionResults(op->getResultTypes());
      if (!packedType)
        return this->matchFailure();
    }

    auto newOp = rewriter.create<TargetOp>(op->getLoc(), packedType, operands,
                                           op->getAttrs());

    // If the operation produced 0 or 1 result, return them immediately.
    if (numResults == 0)
      return rewriter.eraseOp(op), this->matchSuccess();
    if (numResults == 1)
      return rewriter.replaceOp(op, newOp.getOperation()->getResult(0)),
             this->matchSuccess();

    // Otherwise, it had been converted to an operation producing a structure.
    // Extract individual results from the structure and return them as list.
    SmallVector<Value, 4> results;
    results.reserve(numResults);
    for (unsigned i = 0; i < numResults; ++i) {
      auto type = this->lowering.convertType(op->getResult(i).getType());
      results.push_back(rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), type, newOp.getOperation()->getResult(0),
          rewriter.getI64ArrayAttr(i)));
    }
    rewriter.replaceOp(op, results);
    return this->matchSuccess();
  }
};

template <typename SourceOp, unsigned OpCount> struct OpCountValidator {
  static_assert(
      std::is_base_of<
          typename OpTrait::NOperands<OpCount>::template Impl<SourceOp>,
          SourceOp>::value,
      "wrong operand count");
};

template <typename SourceOp> struct OpCountValidator<SourceOp, 1> {
  static_assert(std::is_base_of<OpTrait::OneOperand<SourceOp>, SourceOp>::value,
                "expected a single operand");
};

template <typename SourceOp, unsigned OpCount> void ValidateOpCount() {
  OpCountValidator<SourceOp, OpCount>();
}

// Basic lowering implementation for rewriting from Standard Ops to LLVM Dialect
// Ops for N-ary ops with one result. This supports higher-dimensional vector
// types.
template <typename SourceOp, typename TargetOp, unsigned OpCount>
struct NaryOpLLVMOpLowering : public LLVMLegalizationPattern<SourceOp> {
  using LLVMLegalizationPattern<SourceOp>::LLVMLegalizationPattern;
  using Super = NaryOpLLVMOpLowering<SourceOp, TargetOp, OpCount>;

  // Convert the type of the result to an LLVM type, pass operands as is,
  // preserve attributes.
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    ValidateOpCount<SourceOp, OpCount>();
    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");
    static_assert(std::is_base_of<OpTrait::SameOperandsAndResultType<SourceOp>,
                                  SourceOp>::value,
                  "expected same operands and result type");

    // Cannot convert ops if their operands are not of LLVM type.
    for (Value operand : operands) {
      if (!operand || !operand.getType().isa<LLVM::LLVMType>())
        return this->matchFailure();
    }

    auto loc = op->getLoc();
    auto llvmArrayTy = operands[0].getType().cast<LLVM::LLVMType>();

    if (!llvmArrayTy.isArrayTy()) {
      auto newOp = rewriter.create<TargetOp>(
          op->getLoc(), operands[0].getType(), operands, op->getAttrs());
      rewriter.replaceOp(op, newOp.getResult());
      return this->matchSuccess();
    }

    auto vectorType = op->getResult(0).getType().dyn_cast<VectorType>();
    if (!vectorType)
      return this->matchFailure();
    auto vectorTypeInfo = extractNDVectorTypeInfo(vectorType, this->lowering);
    auto llvmVectorTy = vectorTypeInfo.llvmVectorTy;
    if (!llvmVectorTy || llvmArrayTy != vectorTypeInfo.llvmArrayTy)
      return this->matchFailure();

    Value desc = rewriter.create<LLVM::UndefOp>(loc, llvmArrayTy);
    nDVectorIterate(vectorTypeInfo, rewriter, [&](ArrayAttr position) {
      // For this unrolled `position` corresponding to the `linearIndex`^th
      // element, extract operand vectors
      SmallVector<Value, OpCount> extractedOperands;
      for (unsigned i = 0; i < OpCount; ++i) {
        extractedOperands.push_back(rewriter.create<LLVM::ExtractValueOp>(
            loc, llvmVectorTy, operands[i], position));
      }
      Value newVal = rewriter.create<TargetOp>(
          loc, llvmVectorTy, extractedOperands, op->getAttrs());
      desc = rewriter.create<LLVM::InsertValueOp>(loc, llvmArrayTy, desc,
                                                  newVal, position);
    });
    rewriter.replaceOp(op, desc);
    return this->matchSuccess();
  }
};

template <typename SourceOp, typename TargetOp>
using UnaryOpLLVMOpLowering = NaryOpLLVMOpLowering<SourceOp, TargetOp, 1>;
template <typename SourceOp, typename TargetOp>
using BinaryOpLLVMOpLowering = NaryOpLLVMOpLowering<SourceOp, TargetOp, 2>;

// Specific lowerings.
// FIXME: this should be tablegen'ed.
struct AbsFOpLowering : public UnaryOpLLVMOpLowering<AbsFOp, LLVM::FAbsOp> {
  using Super::Super;
};
struct CeilFOpLowering : public UnaryOpLLVMOpLowering<CeilFOp, LLVM::FCeilOp> {
  using Super::Super;
};
struct CosOpLowering : public UnaryOpLLVMOpLowering<CosOp, LLVM::CosOp> {
  using Super::Super;
};
struct ExpOpLowering : public UnaryOpLLVMOpLowering<ExpOp, LLVM::ExpOp> {
  using Super::Super;
};
struct LogOpLowering : public UnaryOpLLVMOpLowering<LogOp, LLVM::LogOp> {
  using Super::Super;
};
struct Log10OpLowering : public UnaryOpLLVMOpLowering<Log10Op, LLVM::Log10Op> {
  using Super::Super;
};
struct Log2OpLowering : public UnaryOpLLVMOpLowering<Log2Op, LLVM::Log2Op> {
  using Super::Super;
};
struct NegFOpLowering : public UnaryOpLLVMOpLowering<NegFOp, LLVM::FNegOp> {
  using Super::Super;
};
struct AddIOpLowering : public BinaryOpLLVMOpLowering<AddIOp, LLVM::AddOp> {
  using Super::Super;
};
struct SubIOpLowering : public BinaryOpLLVMOpLowering<SubIOp, LLVM::SubOp> {
  using Super::Super;
};
struct MulIOpLowering : public BinaryOpLLVMOpLowering<MulIOp, LLVM::MulOp> {
  using Super::Super;
};
struct SignedDivIOpLowering
    : public BinaryOpLLVMOpLowering<SignedDivIOp, LLVM::SDivOp> {
  using Super::Super;
};
struct SqrtOpLowering : public UnaryOpLLVMOpLowering<SqrtOp, LLVM::SqrtOp> {
  using Super::Super;
};
struct UnsignedDivIOpLowering
    : public BinaryOpLLVMOpLowering<UnsignedDivIOp, LLVM::UDivOp> {
  using Super::Super;
};
struct SignedRemIOpLowering
    : public BinaryOpLLVMOpLowering<SignedRemIOp, LLVM::SRemOp> {
  using Super::Super;
};
struct UnsignedRemIOpLowering
    : public BinaryOpLLVMOpLowering<UnsignedRemIOp, LLVM::URemOp> {
  using Super::Super;
};
struct AndOpLowering : public BinaryOpLLVMOpLowering<AndOp, LLVM::AndOp> {
  using Super::Super;
};
struct OrOpLowering : public BinaryOpLLVMOpLowering<OrOp, LLVM::OrOp> {
  using Super::Super;
};
struct XOrOpLowering : public BinaryOpLLVMOpLowering<XOrOp, LLVM::XOrOp> {
  using Super::Super;
};
struct AddFOpLowering : public BinaryOpLLVMOpLowering<AddFOp, LLVM::FAddOp> {
  using Super::Super;
};
struct SubFOpLowering : public BinaryOpLLVMOpLowering<SubFOp, LLVM::FSubOp> {
  using Super::Super;
};
struct MulFOpLowering : public BinaryOpLLVMOpLowering<MulFOp, LLVM::FMulOp> {
  using Super::Super;
};
struct DivFOpLowering : public BinaryOpLLVMOpLowering<DivFOp, LLVM::FDivOp> {
  using Super::Super;
};
struct RemFOpLowering : public BinaryOpLLVMOpLowering<RemFOp, LLVM::FRemOp> {
  using Super::Super;
};
struct CopySignOpLowering
    : public BinaryOpLLVMOpLowering<CopySignOp, LLVM::CopySignOp> {
  using Super::Super;
};
struct SelectOpLowering
    : public OneToOneLLVMOpLowering<SelectOp, LLVM::SelectOp> {
  using Super::Super;
};
struct ConstLLVMOpLowering
    : public OneToOneLLVMOpLowering<ConstantOp, LLVM::ConstantOp> {
  using Super::Super;
};
struct ShiftLeftOpLowering
    : public OneToOneLLVMOpLowering<ShiftLeftOp, LLVM::ShlOp> {
  using Super::Super;
};
struct SignedShiftRightOpLowering
    : public OneToOneLLVMOpLowering<SignedShiftRightOp, LLVM::AShrOp> {
  using Super::Super;
};
struct UnsignedShiftRightOpLowering
    : public OneToOneLLVMOpLowering<UnsignedShiftRightOp, LLVM::LShrOp> {
  using Super::Super;
};

// Check if the MemRefType `type` is supported by the lowering. We currently
// only support memrefs with identity maps.
static bool isSupportedMemRefType(MemRefType type) {
  return type.getAffineMaps().empty() ||
         llvm::all_of(type.getAffineMaps(),
                      [](AffineMap map) { return map.isIdentity(); });
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
// Alignment is obtained by allocating `alignment - 1` more bytes than requested
// and shifting the aligned pointer relative to the allocated memory. If
// alignment is unspecified, the two pointers are equal.
struct AllocOpLowering : public LLVMLegalizationPattern<AllocOp> {
  using LLVMLegalizationPattern<AllocOp>::LLVMLegalizationPattern;

  AllocOpLowering(LLVM::LLVMDialect &dialect_, LLVMTypeConverter &converter,
                  bool useAlloca = false)
      : LLVMLegalizationPattern<AllocOp>(dialect_, converter),
        useAlloca(useAlloca) {}

  PatternMatchResult match(Operation *op) const override {
    MemRefType type = cast<AllocOp>(op).getType();
    if (isSupportedMemRefType(type))
      return matchSuccess();

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(type, strides, offset);
    if (failed(successStrides))
      return matchFailure();

    // Dynamic strides are ok if they can be deduced from dynamic sizes (which
    // is guaranteed when succeeded(successStrides)). Dynamic offset however can
    // never be alloc'ed.
    if (offset == MemRefType::getDynamicStrideOrOffset())
      return matchFailure();

    return matchSuccess();
  }

  void rewrite(Operation *op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto allocOp = cast<AllocOp>(op);
    MemRefType type = allocOp.getType();

    // Get actual sizes of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.  In case of
    // zero-dimensional memref, assume a scalar (size 1).
    SmallVector<Value, 4> sizes;
    sizes.reserve(type.getRank());
    unsigned i = 0;
    for (int64_t s : type.getShape())
      sizes.push_back(s == -1 ? operands[i++]
                              : createIndexConstant(rewriter, loc, s));
    if (sizes.empty())
      sizes.push_back(createIndexConstant(rewriter, loc, 1));

    // Compute the total number of memref elements.
    Value cumulativeSize = sizes.front();
    for (unsigned i = 1, e = sizes.size(); i < e; ++i)
      cumulativeSize = rewriter.create<LLVM::MulOp>(
          loc, getIndexType(), ArrayRef<Value>{cumulativeSize, sizes[i]});

    // Compute the size of an individual element. This emits the MLIR equivalent
    // of the following sizeof(...) implementation in LLVM IR:
    //   %0 = getelementptr %elementType* null, %indexType 1
    //   %1 = ptrtoint %elementType* %0 to %indexType
    // which is a common pattern of getting the size of a type in bytes.
    auto elementType = type.getElementType();
    auto convertedPtrType =
        lowering.convertType(elementType).cast<LLVM::LLVMType>().getPointerTo();
    auto nullPtr = rewriter.create<LLVM::NullOp>(loc, convertedPtrType);
    auto one = createIndexConstant(rewriter, loc, 1);
    auto gep = rewriter.create<LLVM::GEPOp>(loc, convertedPtrType,
                                            ArrayRef<Value>{nullPtr, one});
    auto elementSize =
        rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(), gep);
    cumulativeSize = rewriter.create<LLVM::MulOp>(
        loc, getIndexType(), ArrayRef<Value>{cumulativeSize, elementSize});

    // Allocate the underlying buffer and store a pointer to it in the MemRef
    // descriptor.
    Value allocated = nullptr;
    int alignment = 0;
    Value alignmentValue = nullptr;
    if (auto alignAttr = allocOp.alignment())
      alignment = alignAttr.getValue().getSExtValue();

    if (useAlloca) {
      allocated = rewriter.create<LLVM::AllocaOp>(loc, getVoidPtrType(),
                                                  cumulativeSize, alignment);
    } else {
      // Insert the `malloc` declaration if it is not already present.
      auto module = op->getParentOfType<ModuleOp>();
      auto mallocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("malloc");
      if (!mallocFunc) {
        OpBuilder moduleBuilder(
            op->getParentOfType<ModuleOp>().getBodyRegion());
        mallocFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
            rewriter.getUnknownLoc(), "malloc",
            LLVM::LLVMType::getFunctionTy(getVoidPtrType(), getIndexType(),
                                          /*isVarArg=*/false));
      }
      if (alignment != 0) {
        alignmentValue = createIndexConstant(rewriter, loc, alignment);
        cumulativeSize = rewriter.create<LLVM::SubOp>(
            loc,
            rewriter.create<LLVM::AddOp>(loc, cumulativeSize, alignmentValue),
            one);
      }
      allocated = rewriter
                      .create<LLVM::CallOp>(
                          loc, getVoidPtrType(),
                          rewriter.getSymbolRefAttr(mallocFunc), cumulativeSize)
                      .getResult(0);
    }

    auto structElementType = lowering.convertType(elementType);
    auto elementPtrType = structElementType.cast<LLVM::LLVMType>().getPointerTo(
        type.getMemorySpace());
    Value bitcastAllocated = rewriter.create<LLVM::BitcastOp>(
        loc, elementPtrType, ArrayRef<Value>(allocated));

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(type, strides, offset);
    assert(succeeded(successStrides) && "unexpected non-strided memref");
    (void)successStrides;
    assert(offset != MemRefType::getDynamicStrideOrOffset() &&
           "unexpected dynamic offset");

    // 0-D memref corner case: they have size 1 ...
    assert(((type.getRank() == 0 && strides.empty() && sizes.size() == 1) ||
            (strides.size() == sizes.size())) &&
           "unexpected number of strides");

    // Create the MemRef descriptor.
    auto structType = lowering.convertType(type);
    auto memRefDescriptor = MemRefDescriptor::undef(rewriter, loc, structType);
    // Field 1: Allocated pointer, used for malloc/free.
    memRefDescriptor.setAllocatedPtr(rewriter, loc, bitcastAllocated);

    // Field 2: Actual aligned pointer to payload.
    Value bitcastAligned = bitcastAllocated;
    if (!useAlloca && alignment != 0) {
      assert(alignmentValue);
      // offset = (align - (ptr % align))% align
      Value intVal = rewriter.create<LLVM::PtrToIntOp>(
          loc, this->getIndexType(), allocated);
      Value ptrModAlign =
          rewriter.create<LLVM::URemOp>(loc, intVal, alignmentValue);
      Value subbed =
          rewriter.create<LLVM::SubOp>(loc, alignmentValue, ptrModAlign);
      Value offset = rewriter.create<LLVM::URemOp>(loc, subbed, alignmentValue);
      Value aligned = rewriter.create<LLVM::GEPOp>(loc, allocated.getType(),
                                                   allocated, offset);
      bitcastAligned = rewriter.create<LLVM::BitcastOp>(
          loc, elementPtrType, ArrayRef<Value>(aligned));
    }
    memRefDescriptor.setAlignedPtr(rewriter, loc, bitcastAligned);

    // Field 3: Offset in aligned pointer.
    memRefDescriptor.setOffset(rewriter, loc,
                               createIndexConstant(rewriter, loc, offset));

    if (type.getRank() == 0)
      // No size/stride descriptor in memref, return the descriptor value.
      return rewriter.replaceOp(op, {memRefDescriptor});

    // Fields 4 and 5: Sizes and strides of the strided MemRef.
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
        runningStride =
            runningStride
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

    // Return the final value of the descriptor.
    rewriter.replaceOp(op, {memRefDescriptor});
  }

  bool useAlloca;
};

// A CallOp automatically promotes MemRefType to a sequence of alloca/store and
// passes the pointer to the MemRef across function boundaries.
template <typename CallOpType>
struct CallOpInterfaceLowering : public LLVMLegalizationPattern<CallOpType> {
  using LLVMLegalizationPattern<CallOpType>::LLVMLegalizationPattern;
  using Super = CallOpInterfaceLowering<CallOpType>;
  using Base = LLVMLegalizationPattern<CallOpType>;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CallOpType> transformed(operands);
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
      if (!(packedResult = this->lowering.packFunctionResults(resultTypes)))
        return this->matchFailure();
    }

    auto promoted = this->lowering.promoteMemRefDescriptors(
        op->getLoc(), /*opOperands=*/op->getOperands(), operands, rewriter);
    auto newOp = rewriter.create<LLVM::CallOp>(op->getLoc(), packedResult,
                                               promoted, op->getAttrs());

    // If < 2 results, packing did not do anything and we can just return.
    if (numResults < 2) {
      rewriter.replaceOp(op, newOp.getResults());
      return this->matchSuccess();
    }

    // Otherwise, it had been converted to an operation producing a structure.
    // Extract individual results from the structure and return them as list.
    // TODO(aminim, ntv, riverriddle, zinenko): this seems like patching around
    // a particular interaction between MemRefType and CallOp lowering. Find a
    // way to avoid special casing.
    SmallVector<Value, 4> results;
    results.reserve(numResults);
    for (unsigned i = 0; i < numResults; ++i) {
      auto type = this->lowering.convertType(op->getResult(i).getType());
      results.push_back(rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), type, newOp.getOperation()->getResult(0),
          rewriter.getI64ArrayAttr(i)));
    }
    rewriter.replaceOp(op, results);

    return this->matchSuccess();
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
struct DeallocOpLowering : public LLVMLegalizationPattern<DeallocOp> {
  using LLVMLegalizationPattern<DeallocOp>::LLVMLegalizationPattern;

  DeallocOpLowering(LLVM::LLVMDialect &dialect_, LLVMTypeConverter &converter,
                    bool useAlloca = false)
      : LLVMLegalizationPattern<DeallocOp>(dialect_, converter),
        useAlloca(useAlloca) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (useAlloca)
      return rewriter.eraseOp(op), matchSuccess();

    assert(operands.size() == 1 && "dealloc takes one operand");
    OperandAdaptor<DeallocOp> transformed(operands);

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
    return matchSuccess();
  }

  bool useAlloca;
};

// A `tanh` is converted into a call to the `tanh` function.
struct TanhOpLowering : public LLVMLegalizationPattern<TanhOp> {
  using LLVMLegalizationPattern<TanhOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    using LLVMFuncOpT = LLVM::LLVMFuncOp;
    using LLVMTypeT = LLVM::LLVMType;

    OperandAdaptor<TanhOp> transformed(operands);
    LLVMTypeT operandType =
        transformed.operand().getType().dyn_cast_or_null<LLVM::LLVMType>();

    if (!operandType)
      return matchFailure();

    std::string functionName;
    if (operandType.isFloatTy())
      functionName = "tanhf";
    else if (operandType.isDoubleTy())
      functionName = "tanh";
    else
      return matchFailure();

    // Get a reference to the tanh function, inserting it if necessary.
    Operation *tanhFunc =
        SymbolTable::lookupNearestSymbolFrom(op, functionName);

    LLVMFuncOpT tanhLLVMFunc;
    if (tanhFunc) {
      tanhLLVMFunc = cast<LLVMFuncOpT>(tanhFunc);
    } else {
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      auto module = op->getParentOfType<ModuleOp>();
      rewriter.setInsertionPointToStart(module.getBody());
      tanhLLVMFunc = rewriter.create<LLVMFuncOpT>(
          module.getLoc(), functionName,
          LLVMTypeT::getFunctionTy(operandType, operandType,
                                   /*isVarArg=*/false));
    }

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, operandType, rewriter.getSymbolRefAttr(tanhLLVMFunc),
        transformed.operand());
    return matchSuccess();
  }
};

struct MemRefCastOpLowering : public LLVMLegalizationPattern<MemRefCastOp> {
  using LLVMLegalizationPattern<MemRefCastOp>::LLVMLegalizationPattern;

  PatternMatchResult match(Operation *op) const override {
    auto memRefCastOp = cast<MemRefCastOp>(op);
    Type srcType = memRefCastOp.getOperand().getType();
    Type dstType = memRefCastOp.getType();

    if (srcType.isa<MemRefType>() && dstType.isa<MemRefType>()) {
      MemRefType sourceType =
          memRefCastOp.getOperand().getType().cast<MemRefType>();
      MemRefType targetType = memRefCastOp.getType().cast<MemRefType>();
      return (isSupportedMemRefType(targetType) &&
              isSupportedMemRefType(sourceType))
                 ? matchSuccess()
                 : matchFailure();
    }

    // At least one of the operands is unranked type
    assert(srcType.isa<UnrankedMemRefType>() ||
           dstType.isa<UnrankedMemRefType>());

    // Unranked to unranked cast is disallowed
    return !(srcType.isa<UnrankedMemRefType>() &&
             dstType.isa<UnrankedMemRefType>())
               ? matchSuccess()
               : matchFailure();
  }

  void rewrite(Operation *op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override {
    auto memRefCastOp = cast<MemRefCastOp>(op);
    OperandAdaptor<MemRefCastOp> transformed(operands);

    auto srcType = memRefCastOp.getOperand().getType();
    auto dstType = memRefCastOp.getType();
    auto targetStructType = lowering.convertType(memRefCastOp.getType());
    auto loc = op->getLoc();

    if (srcType.isa<MemRefType>() && dstType.isa<MemRefType>()) {
      // memref_cast is defined for source and destination memref types with the
      // same element type, same mappings, same address space and same rank.
      // Therefore a simple bitcast suffices. If not it is undefined behavior.
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
      auto ptr = lowering.promoteOneMemRefDescriptor(loc, transformed.source(),
                                                     rewriter);
      // voidptr = BitCastOp srcType* to void*
      auto voidPtr =
          rewriter.create<LLVM::BitcastOp>(loc, getVoidPtrType(), ptr)
              .getResult();
      // rank = ConstantOp srcRank
      auto rankVal = rewriter.create<LLVM::ConstantOp>(
          loc, lowering.convertType(rewriter.getIntegerType(64)),
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
      llvm_unreachable("Unsuppored unranked memref to unranked memref cast");
    }
  }
};

// A `dim` is converted to a constant for static sizes and to an access to the
// size stored in the memref descriptor for dynamic sizes.
struct DimOpLowering : public LLVMLegalizationPattern<DimOp> {
  using LLVMLegalizationPattern<DimOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto dimOp = cast<DimOp>(op);
    OperandAdaptor<DimOp> transformed(operands);
    MemRefType type = dimOp.getOperand().getType().cast<MemRefType>();

    auto shape = type.getShape();
    int64_t index = dimOp.getIndex();
    // Extract dynamic size from the memref descriptor.
    if (ShapedType::isDynamic(shape[index]))
      rewriter.replaceOp(op, {MemRefDescriptor(transformed.memrefOrTensor())
                                  .size(rewriter, op->getLoc(), index)});
    else
      // Use constant for static size.
      rewriter.replaceOp(
          op, createIndexConstant(rewriter, op->getLoc(), shape[index]));
    return matchSuccess();
  }
};

// Common base for load and store operations on MemRefs.  Restricts the match
// to supported MemRef types.  Provides functionality to emit code accessing a
// specific element of the underlying data buffer.
template <typename Derived>
struct LoadStoreOpLowering : public LLVMLegalizationPattern<Derived> {
  using LLVMLegalizationPattern<Derived>::LLVMLegalizationPattern;
  using Base = LoadStoreOpLowering<Derived>;

  PatternMatchResult match(Operation *op) const override {
    MemRefType type = cast<Derived>(op).getMemRefType();
    return isSupportedMemRefType(type) ? this->matchSuccess()
                                       : this->matchFailure();
  }

  // Given subscript indices and array sizes in row-major order,
  //   i_n, i_{n-1}, ..., i_1
  //   s_n, s_{n-1}, ..., s_1
  // obtain a value that corresponds to the linearized subscript
  //   \sum_k i_k * \prod_{j=1}^{k-1} s_j
  // by accumulating the running linearized value.
  // Note that `indices` and `allocSizes` are passed in the same order as they
  // appear in load/store operations and memref type declarations.
  Value linearizeSubscripts(ConversionPatternRewriter &builder, Location loc,
                            ArrayRef<Value> indices,
                            ArrayRef<Value> allocSizes) const {
    assert(indices.size() == allocSizes.size() &&
           "mismatching number of indices and allocation sizes");
    assert(!indices.empty() && "cannot linearize a 0-dimensional access");

    Value linearized = indices.front();
    for (int i = 1, nSizes = allocSizes.size(); i < nSizes; ++i) {
      linearized = builder.create<LLVM::MulOp>(
          loc, this->getIndexType(),
          ArrayRef<Value>{linearized, allocSizes[i]});
      linearized = builder.create<LLVM::AddOp>(
          loc, this->getIndexType(), ArrayRef<Value>{linearized, indices[i]});
    }
    return linearized;
  }

  // This is a strided getElementPtr variant that linearizes subscripts as:
  //   `base_offset + index_0 * stride_0 + ... + index_n * stride_n`.
  Value getStridedElementPtr(Location loc, Type elementTypePtr,
                             Value descriptor, ArrayRef<Value> indices,
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

  Value getDataPtr(Location loc, MemRefType type, Value memRefDesc,
                   ArrayRef<Value> indices, ConversionPatternRewriter &rewriter,
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
};

// Load operation is lowered to obtaining a pointer to the indexed element
// and loading it.
struct LoadOpLowering : public LoadStoreOpLowering<LoadOp> {
  using Base::Base;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loadOp = cast<LoadOp>(op);
    OperandAdaptor<LoadOp> transformed(operands);
    auto type = loadOp.getMemRefType();

    Value dataPtr = getDataPtr(op->getLoc(), type, transformed.memref(),
                               transformed.indices(), rewriter, getModule());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, dataPtr);
    return matchSuccess();
  }
};

// Store operation is lowered to obtaining a pointer to the indexed element,
// and storing the given value to it.
struct StoreOpLowering : public LoadStoreOpLowering<StoreOp> {
  using Base::Base;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = cast<StoreOp>(op).getMemRefType();
    OperandAdaptor<StoreOp> transformed(operands);

    Value dataPtr = getDataPtr(op->getLoc(), type, transformed.memref(),
                               transformed.indices(), rewriter, getModule());
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, transformed.value(),
                                               dataPtr);
    return matchSuccess();
  }
};

// The prefetch operation is lowered in a way similar to the load operation
// except that the llvm.prefetch operation is used for replacement.
struct PrefetchOpLowering : public LoadStoreOpLowering<PrefetchOp> {
  using Base::Base;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto prefetchOp = cast<PrefetchOp>(op);
    OperandAdaptor<PrefetchOp> transformed(operands);
    auto type = prefetchOp.getMemRefType();

    Value dataPtr = getDataPtr(op->getLoc(), type, transformed.memref(),
                               transformed.indices(), rewriter, getModule());

    // Replace with llvm.prefetch.
    auto llvmI32Type = lowering.convertType(rewriter.getIntegerType(32));
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
    return matchSuccess();
  }
};

// The lowering of index_cast becomes an integer conversion since index becomes
// an integer.  If the bit width of the source and target integer types is the
// same, just erase the cast.  If the target type is wider, sign-extend the
// value, otherwise truncate it.
struct IndexCastOpLowering : public LLVMLegalizationPattern<IndexCastOp> {
  using LLVMLegalizationPattern<IndexCastOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    IndexCastOpOperandAdaptor transformed(operands);
    auto indexCastOp = cast<IndexCastOp>(op);

    auto targetType =
        this->lowering.convertType(indexCastOp.getResult().getType())
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
    return matchSuccess();
  }
};

// Convert std.cmp predicate into the LLVM dialect CmpPredicate.  The two
// enums share the numerical values so just cast.
template <typename LLVMPredType, typename StdPredType>
static LLVMPredType convertCmpPredicate(StdPredType pred) {
  return static_cast<LLVMPredType>(pred);
}

struct CmpIOpLowering : public LLVMLegalizationPattern<CmpIOp> {
  using LLVMLegalizationPattern<CmpIOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto cmpiOp = cast<CmpIOp>(op);
    CmpIOpOperandAdaptor transformed(operands);

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, lowering.convertType(cmpiOp.getResult().getType()),
        rewriter.getI64IntegerAttr(static_cast<int64_t>(
            convertCmpPredicate<LLVM::ICmpPredicate>(cmpiOp.getPredicate()))),
        transformed.lhs(), transformed.rhs());

    return matchSuccess();
  }
};

struct CmpFOpLowering : public LLVMLegalizationPattern<CmpFOp> {
  using LLVMLegalizationPattern<CmpFOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto cmpfOp = cast<CmpFOp>(op);
    CmpFOpOperandAdaptor transformed(operands);

    rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
        op, lowering.convertType(cmpfOp.getResult().getType()),
        rewriter.getI64IntegerAttr(static_cast<int64_t>(
            convertCmpPredicate<LLVM::FCmpPredicate>(cmpfOp.getPredicate()))),
        transformed.lhs(), transformed.rhs());

    return matchSuccess();
  }
};

struct SIToFPLowering
    : public OneToOneLLVMOpLowering<SIToFPOp, LLVM::SIToFPOp> {
  using Super::Super;
};

struct FPExtLowering : public OneToOneLLVMOpLowering<FPExtOp, LLVM::FPExtOp> {
  using Super::Super;
};

struct FPTruncLowering
    : public OneToOneLLVMOpLowering<FPTruncOp, LLVM::FPTruncOp> {
  using Super::Super;
};

struct SignExtendIOpLowering
    : public OneToOneLLVMOpLowering<SignExtendIOp, LLVM::SExtOp> {
  using Super::Super;
};

struct TruncateIOpLowering
    : public OneToOneLLVMOpLowering<TruncateIOp, LLVM::TruncOp> {
  using Super::Super;
};

struct ZeroExtendIOpLowering
    : public OneToOneLLVMOpLowering<ZeroExtendIOp, LLVM::ZExtOp> {
  using Super::Super;
};

// Base class for LLVM IR lowering terminator operations with successors.
template <typename SourceOp, typename TargetOp>
struct OneToOneLLVMTerminatorLowering
    : public LLVMLegalizationPattern<SourceOp> {
  using LLVMLegalizationPattern<SourceOp>::LLVMLegalizationPattern;
  using Super = OneToOneLLVMTerminatorLowering<SourceOp, TargetOp>;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> properOperands,
                  ArrayRef<Block *> destinations,
                  ArrayRef<ArrayRef<Value>> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<ValueRange, 2> operandRanges(operands.begin(), operands.end());
    rewriter.replaceOpWithNewOp<TargetOp>(op, properOperands, destinations,
                                          operandRanges, op->getAttrs());
    return this->matchSuccess();
  }
};

// Special lowering pattern for `ReturnOps`.  Unlike all other operations,
// `ReturnOp` interacts with the function signature and must have as many
// operands as the function has return values.  Because in LLVM IR, functions
// can only return 0 or 1 value, we pack multiple values into a structure type.
// Emit `UndefOp` followed by `InsertValueOp`s to create such structure if
// necessary before returning it
struct ReturnOpLowering : public LLVMLegalizationPattern<ReturnOp> {
  using LLVMLegalizationPattern<ReturnOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numArguments = op->getNumOperands();

    // If ReturnOp has 0 or 1 operand, create it and return immediately.
    if (numArguments == 0) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
          op, ArrayRef<Value>(), ArrayRef<Block *>(), op->getAttrs());
      return matchSuccess();
    }
    if (numArguments == 1) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
          op, ArrayRef<Value>(operands.front()), ArrayRef<Block *>(),
          op->getAttrs());
      return matchSuccess();
    }

    // Otherwise, we need to pack the arguments into an LLVM struct type before
    // returning.
    auto packedType =
        lowering.packFunctionResults(llvm::to_vector<4>(op->getOperandTypes()));

    Value packed = rewriter.create<LLVM::UndefOp>(op->getLoc(), packedType);
    for (unsigned i = 0; i < numArguments; ++i) {
      packed = rewriter.create<LLVM::InsertValueOp>(
          op->getLoc(), packedType, packed, operands[i],
          rewriter.getI64ArrayAttr(i));
    }
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
        op, llvm::makeArrayRef(packed), ArrayRef<Block *>(), op->getAttrs());
    return matchSuccess();
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
struct SplatOpLowering : public LLVMLegalizationPattern<SplatOp> {
  using LLVMLegalizationPattern<SplatOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto splatOp = cast<SplatOp>(op);
    VectorType resultType = splatOp.getType().dyn_cast<VectorType>();
    if (!resultType || resultType.getRank() != 1)
      return matchFailure();

    // First insert it into an undef vector so we can shuffle it.
    auto vectorType = lowering.convertType(splatOp.getType());
    Value undef = rewriter.create<LLVM::UndefOp>(op->getLoc(), vectorType);
    auto zero = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), lowering.convertType(rewriter.getIntegerType(32)),
        rewriter.getZeroAttr(rewriter.getIntegerType(32)));

    auto v = rewriter.create<LLVM::InsertElementOp>(
        op->getLoc(), vectorType, undef, splatOp.getOperand(), zero);

    int64_t width = splatOp.getType().cast<VectorType>().getDimSize(0);
    SmallVector<int32_t, 4> zeroValues(width, 0);

    // Shuffle the value across the desired number of elements.
    ArrayAttr zeroAttrs = rewriter.getI32ArrayAttr(zeroValues);
    rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(op, v, undef, zeroAttrs);
    return matchSuccess();
  }
};

// The Splat operation is lowered to an insertelement + a shufflevector
// operation. Splat to only 2+-d vector result types are lowered by the
// SplatNdOpLowering, the 1-d case is handled by SplatOpLowering.
struct SplatNdOpLowering : public LLVMLegalizationPattern<SplatOp> {
  using LLVMLegalizationPattern<SplatOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto splatOp = cast<SplatOp>(op);
    OperandAdaptor<SplatOp> adaptor(operands);
    VectorType resultType = splatOp.getType().dyn_cast<VectorType>();
    if (!resultType || resultType.getRank() == 1)
      return matchFailure();

    // First insert it into an undef vector so we can shuffle it.
    auto loc = op->getLoc();
    auto vectorTypeInfo = extractNDVectorTypeInfo(resultType, lowering);
    auto llvmArrayTy = vectorTypeInfo.llvmArrayTy;
    auto llvmVectorTy = vectorTypeInfo.llvmVectorTy;
    if (!llvmArrayTy || !llvmVectorTy)
      return matchFailure();

    // Construct returned value.
    Value desc = rewriter.create<LLVM::UndefOp>(loc, llvmArrayTy);

    // Construct a 1-D vector with the splatted value that we insert in all the
    // places within the returned descriptor.
    Value vdesc = rewriter.create<LLVM::UndefOp>(loc, llvmVectorTy);
    auto zero = rewriter.create<LLVM::ConstantOp>(
        loc, lowering.convertType(rewriter.getIntegerType(32)),
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
    return matchSuccess();
  }
};

/// Conversion pattern that transforms a subview op into:
///   1. An `llvm.mlir.undef` operation to create a memref descriptor
///   2. Updates to the descriptor to introduce the data ptr, offset, size
///      and stride.
/// The subview op is replaced by the descriptor.
struct SubViewOpLowering : public LLVMLegalizationPattern<SubViewOp> {
  using LLVMLegalizationPattern<SubViewOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto viewOp = cast<SubViewOp>(op);
    // TODO(b/144779634, ravishankarm) : After Tblgen is adapted to support
    // having multiple variadic operands where each operand can have different
    // number of entries, clean all of this up.
    SmallVector<Value, 2> dynamicOffsets(
        std::next(operands.begin()),
        std::next(operands.begin(), 1 + viewOp.getNumOffsets()));
    SmallVector<Value, 2> dynamicSizes(
        std::next(operands.begin(), 1 + viewOp.getNumOffsets()),
        std::next(operands.begin(),
                  1 + viewOp.getNumOffsets() + viewOp.getNumSizes()));
    SmallVector<Value, 2> dynamicStrides(
        std::next(operands.begin(),
                  1 + viewOp.getNumOffsets() + viewOp.getNumSizes()),
        operands.end());

    auto sourceMemRefType = viewOp.source().getType().cast<MemRefType>();
    auto sourceElementTy =
        lowering.convertType(sourceMemRefType.getElementType())
            .dyn_cast_or_null<LLVM::LLVMType>();

    auto viewMemRefType = viewOp.getType();
    auto targetElementTy = lowering.convertType(viewMemRefType.getElementType())
                               .dyn_cast<LLVM::LLVMType>();
    auto targetDescTy =
        lowering.convertType(viewMemRefType).dyn_cast_or_null<LLVM::LLVMType>();
    if (!sourceElementTy || !targetDescTy)
      return matchFailure();

    // Currently, only rank > 0 and full or no operands are supported. Fail to
    // convert otherwise.
    unsigned rank = sourceMemRefType.getRank();
    if (viewMemRefType.getRank() == 0 ||
        (!dynamicOffsets.empty() && rank != dynamicOffsets.size()) ||
        (!dynamicSizes.empty() && rank != dynamicSizes.size()) ||
        (!dynamicStrides.empty() && rank != dynamicStrides.size()))
      return matchFailure();

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(viewMemRefType, strides, offset);
    if (failed(successStrides))
      return matchFailure();

    // Fail to convert if neither a dynamic nor static offset is available.
    if (dynamicOffsets.empty() &&
        offset == MemRefType::getDynamicStrideOrOffset())
      return matchFailure();

    // Create the descriptor.
    if (!operands.front().getType().isa<LLVM::LLVMType>())
      return matchFailure();
    MemRefDescriptor sourceMemRef(operands.front());
    auto targetMemRef = MemRefDescriptor::undef(rewriter, loc, targetDescTy);

    // Copy the buffer pointer from the old descriptor to the new one.
    Value extracted = sourceMemRef.allocatedPtr(rewriter, loc);
    Value bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc, targetElementTy.getPointerTo(), extracted);
    targetMemRef.setAllocatedPtr(rewriter, loc, bitcastPtr);

    extracted = sourceMemRef.alignedPtr(rewriter, loc);
    bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc, targetElementTy.getPointerTo(), extracted);
    targetMemRef.setAlignedPtr(rewriter, loc, bitcastPtr);

    // Extract strides needed to compute offset.
    SmallVector<Value, 4> strideValues;
    strideValues.reserve(viewMemRefType.getRank());
    for (int i = 0, e = viewMemRefType.getRank(); i < e; ++i)
      strideValues.push_back(sourceMemRef.stride(rewriter, loc, i));

    // Fill in missing dynamic sizes.
    auto llvmIndexType = lowering.convertType(rewriter.getIndexType());
    if (dynamicSizes.empty()) {
      dynamicSizes.reserve(viewMemRefType.getRank());
      auto shape = viewMemRefType.getShape();
      for (auto extent : shape) {
        dynamicSizes.push_back(rewriter.create<LLVM::ConstantOp>(
            loc, llvmIndexType, rewriter.getI64IntegerAttr(extent)));
      }
    }

    // Offset.
    if (dynamicOffsets.empty()) {
      targetMemRef.setConstantOffset(rewriter, loc, offset);
    } else {
      Value baseOffset = sourceMemRef.offset(rewriter, loc);
      for (int i = 0, e = viewMemRefType.getRank(); i < e; ++i) {
        Value min = dynamicOffsets[i];
        baseOffset = rewriter.create<LLVM::AddOp>(
            loc, baseOffset,
            rewriter.create<LLVM::MulOp>(loc, min, strideValues[i]));
      }
      targetMemRef.setOffset(rewriter, loc, baseOffset);
    }

    // Update sizes and strides.
    for (int i = viewMemRefType.getRank() - 1; i >= 0; --i) {
      targetMemRef.setSize(rewriter, loc, i, dynamicSizes[i]);
      Value newStride;
      if (dynamicStrides.empty())
        newStride = rewriter.create<LLVM::ConstantOp>(
            loc, llvmIndexType, rewriter.getI64IntegerAttr(strides[i]));
      else
        newStride = rewriter.create<LLVM::MulOp>(loc, dynamicStrides[i],
                                                 strideValues[i]);
      targetMemRef.setStride(rewriter, loc, i, newStride);
    }

    rewriter.replaceOp(op, {targetMemRef});
    return matchSuccess();
  }
};

/// Conversion pattern that transforms a op into:
///   1. An `llvm.mlir.undef` operation to create a memref descriptor
///   2. Updates to the descriptor to introduce the data ptr, offset, size
///      and stride.
/// The view op is replaced by the descriptor.
struct ViewOpLowering : public LLVMLegalizationPattern<ViewOp> {
  using LLVMLegalizationPattern<ViewOp>::LLVMLegalizationPattern;

  // Build and return the value for the idx^th shape dimension, either by
  // returning the constant shape dimension or counting the proper dynamic size.
  Value getSize(ConversionPatternRewriter &rewriter, Location loc,
                ArrayRef<int64_t> shape, ArrayRef<Value> dynamicSizes,
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

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto viewOp = cast<ViewOp>(op);
    ViewOpOperandAdaptor adaptor(operands);

    auto viewMemRefType = viewOp.getType();
    auto targetElementTy = lowering.convertType(viewMemRefType.getElementType())
                               .dyn_cast<LLVM::LLVMType>();
    auto targetDescTy =
        lowering.convertType(viewMemRefType).dyn_cast<LLVM::LLVMType>();
    if (!targetDescTy)
      return op->emitWarning("Target descriptor type not converted to LLVM"),
             matchFailure();

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(viewMemRefType, strides, offset);
    if (failed(successStrides))
      return op->emitWarning("cannot cast to non-strided shape"),
             matchFailure();

    // Create the descriptor.
    MemRefDescriptor sourceMemRef(adaptor.source());
    auto targetMemRef = MemRefDescriptor::undef(rewriter, loc, targetDescTy);

    // Field 1: Copy the allocated pointer, used for malloc/free.
    Value extracted = sourceMemRef.allocatedPtr(rewriter, loc);
    Value bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc, targetElementTy.getPointerTo(), extracted);
    targetMemRef.setAllocatedPtr(rewriter, loc, bitcastPtr);

    // Field 2: Copy the actual aligned pointer to payload.
    extracted = sourceMemRef.alignedPtr(rewriter, loc);
    bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc, targetElementTy.getPointerTo(), extracted);
    targetMemRef.setAlignedPtr(rewriter, loc, bitcastPtr);

    // Field 3: Copy the offset in aligned pointer.
    unsigned numDynamicSizes = llvm::size(viewOp.getDynamicSizes());
    (void)numDynamicSizes;
    bool hasDynamicOffset = offset == MemRefType::getDynamicStrideOrOffset();
    auto sizeAndOffsetOperands = adaptor.operands();
    assert(llvm::size(sizeAndOffsetOperands) ==
           numDynamicSizes + (hasDynamicOffset ? 1 : 0));
    Value baseOffset = !hasDynamicOffset
                           ? createIndexConstant(rewriter, loc, offset)
                           // TODO(ntv): better adaptor.
                           : sizeAndOffsetOperands.front();
    targetMemRef.setOffset(rewriter, loc, baseOffset);

    // Early exit for 0-D corner case.
    if (viewMemRefType.getRank() == 0)
      return rewriter.replaceOp(op, {targetMemRef}), matchSuccess();

    // Fields 4 and 5: Update sizes and strides.
    if (strides.back() != 1)
      return op->emitWarning("cannot cast to non-contiguous shape"),
             matchFailure();
    Value stride = nullptr, nextSize = nullptr;
    // Drop the dynamic stride from the operand list, if present.
    ArrayRef<Value> sizeOperands(sizeAndOffsetOperands);
    if (hasDynamicOffset)
      sizeOperands = sizeOperands.drop_front();
    for (int i = viewMemRefType.getRank() - 1; i >= 0; --i) {
      // Update size.
      Value size =
          getSize(rewriter, loc, viewMemRefType.getShape(), sizeOperands, i);
      targetMemRef.setSize(rewriter, loc, i, size);
      // Update stride.
      stride = getStride(rewriter, loc, strides, nextSize, stride, i);
      targetMemRef.setStride(rewriter, loc, i, stride);
      nextSize = size;
    }

    rewriter.replaceOp(op, {targetMemRef});
    return matchSuccess();
  }
};

} // namespace

static void ensureDistinctSuccessors(Block &bb) {
  auto *terminator = bb.getTerminator();

  // Find repeated successors with arguments.
  llvm::SmallDenseMap<Block *, SmallVector<int, 4>> successorPositions;
  for (int i = 0, e = terminator->getNumSuccessors(); i < e; ++i) {
    Block *successor = terminator->getSuccessor(i);
    // Blocks with no arguments are safe even if they appear multiple times
    // because they don't need PHI nodes.
    if (successor->getNumArguments() == 0)
      continue;
    successorPositions[successor].push_back(i);
  }

  // If a successor appears for the second or more time in the terminator,
  // create a new dummy block that unconditionally branches to the original
  // destination, and retarget the terminator to branch to this new block.
  // There is no need to pass arguments to the dummy block because it will be
  // dominated by the original block and can therefore use any values defined in
  // the original block.
  for (const auto &successor : successorPositions) {
    const auto &positions = successor.second;
    // Start from the second occurrence of a block in the successor list.
    for (auto position = std::next(positions.begin()), end = positions.end();
         position != end; ++position) {
      auto *dummyBlock = new Block();
      bb.getParent()->push_back(dummyBlock);
      auto builder = OpBuilder(dummyBlock);
      SmallVector<Value, 8> operands(
          terminator->getSuccessorOperands(*position));
      builder.create<BranchOp>(terminator->getLoc(), successor.first, operands);
      terminator->setSuccessor(dummyBlock, *position);
      for (int i = 0, e = terminator->getNumSuccessorOperands(*position); i < e;
           ++i)
        terminator->eraseSuccessorOperand(*position, i);
    }
  }
}

void mlir::LLVM::ensureDistinctSuccessors(ModuleOp m) {
  for (auto f : m.getOps<FuncOp>()) {
    for (auto &bb : f.getBlocks()) {
      ::ensureDistinctSuccessors(bb);
    }
  }
}

/// Collect a set of patterns to convert from the Standard dialect to LLVM.
void mlir::populateStdToLLVMNonMemoryConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  // FIXME: this should be tablegen'ed
  // clang-format off
  patterns.insert<
      AbsFOpLowering,
      AddFOpLowering,
      AddIOpLowering,
      AndOpLowering,
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
      DivFOpLowering,
      ExpOpLowering,
      LogOpLowering,
      Log10OpLowering,
      Log2OpLowering,
      FPExtLowering,
      FPTruncLowering,
      IndexCastOpLowering,
      MulFOpLowering,
      MulIOpLowering,
      NegFOpLowering,
      OrOpLowering,
      PrefetchOpLowering,
      RemFOpLowering,
      ReturnOpLowering,
      SIToFPLowering,
      SelectOpLowering,
      ShiftLeftOpLowering,
      SignExtendIOpLowering,
      SignedDivIOpLowering,
      SignedRemIOpLowering,
      SignedShiftRightOpLowering,
      SplatOpLowering,
      SplatNdOpLowering,
      SqrtOpLowering,
      SubFOpLowering,
      SubIOpLowering,
      TanhOpLowering,
      TruncateIOpLowering,
      UnsignedDivIOpLowering,
      UnsignedRemIOpLowering,
      UnsignedShiftRightOpLowering,
      XOrOpLowering,
      ZeroExtendIOpLowering>(*converter.getDialect(), converter);
  // clang-format on
}

void mlir::populateStdToLLVMMemoryConversionPatters(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    bool useAlloca) {
  // clang-format off
  patterns.insert<
      DimOpLowering,
      LoadOpLowering,
      MemRefCastOpLowering,
      StoreOpLowering,
      SubViewOpLowering,
      ViewOpLowering>(*converter.getDialect(), converter);
  patterns.insert<
      AllocOpLowering,
      DeallocOpLowering>(
        *converter.getDialect(), converter, useAlloca);
  // clang-format on
}

void mlir::populateStdToLLVMDefaultFuncOpConversionPattern(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    bool emitCWrappers) {
  patterns.insert<FuncOpConversion>(*converter.getDialect(), converter,
                                    emitCWrappers);
}

void mlir::populateStdToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    bool useAlloca, bool emitCWrappers) {
  populateStdToLLVMDefaultFuncOpConversionPattern(converter, patterns,
                                                  emitCWrappers);
  populateStdToLLVMNonMemoryConversionPatterns(converter, patterns);
  populateStdToLLVMMemoryConversionPatters(converter, patterns, useAlloca);
}

static void populateStdToLLVMBarePtrFuncOpConversionPattern(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  patterns.insert<BarePtrFuncOpConversion>(*converter.getDialect(), converter);
}

void mlir::populateStdToLLVMBarePtrConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    bool useAlloca) {
  populateStdToLLVMBarePtrFuncOpConversionPattern(converter, patterns);
  populateStdToLLVMNonMemoryConversionPatterns(converter, patterns);
  populateStdToLLVMMemoryConversionPatters(converter, patterns, useAlloca);
}

// Convert types using the stored LLVM IR module.
Type LLVMTypeConverter::convertType(Type t) { return convertStandardType(t); }

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
struct LLVMLoweringPass : public ModulePass<LLVMLoweringPass> {
  /// Creates an LLVM lowering pass.
  explicit LLVMLoweringPass(bool useAlloca, bool useBarePtrCallConv,
                            bool emitCWrappers) {
    this->useAlloca = useAlloca;
    this->useBarePtrCallConv = useBarePtrCallConv;
    this->emitCWrappers = emitCWrappers;
  }
  explicit LLVMLoweringPass() {}
  LLVMLoweringPass(const LLVMLoweringPass &pass) {}

  /// Run the dialect converter on the module.
  void runOnModule() override {
    if (useBarePtrCallConv && emitCWrappers) {
      getModule().emitError()
          << "incompatible conversion options: bare-pointer calling convention "
             "and C wrapper emission";
      signalPassFailure();
      return;
    }

    ModuleOp m = getModule();
    LLVM::ensureDistinctSuccessors(m);

    LLVMTypeConverterCustomization customs;
    customs.funcArgConverter = useBarePtrCallConv ? barePtrFuncArgTypeConverter
                                                  : structFuncArgTypeConverter;
    LLVMTypeConverter typeConverter(&getContext(), customs);

    OwningRewritePatternList patterns;
    if (useBarePtrCallConv)
      populateStdToLLVMBarePtrConversionPatterns(typeConverter, patterns,
                                                 useAlloca);
    else
      populateStdToLLVMConversionPatterns(typeConverter, patterns, useAlloca,
                                          emitCWrappers);

    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    if (failed(applyPartialConversion(m, target, patterns, &typeConverter)))
      signalPassFailure();
  }

  /// Use `alloca` instead of `call @malloc` for converting std.alloc.
  Option<bool> useAlloca{
      *this, "use-alloca",
      llvm::cl::desc("Replace emission of malloc/free by alloca"),
      llvm::cl::init(false)};

  /// Convert memrefs to bare pointers in function signatures.
  Option<bool> useBarePtrCallConv{
      *this, "use-bare-ptr-memref-call-conv",
      llvm::cl::desc("Replace FuncOp's MemRef arguments with "
                     "bare pointers to the MemRef element types"),
      llvm::cl::init(false)};

  /// Emit wrappers for C-compatible pointer-to-struct memref descriptors.
  Option<bool> emitCWrappers{
      *this, "emit-c-wrappers",
      llvm::cl::desc("Emit C-compatible wrapper functions"),
      llvm::cl::init(false)};
};
} // end namespace

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::createLowerToLLVMPass(bool useAlloca, bool useBarePtrCallConv,
                            bool emitCWrappers) {
  return std::make_unique<LLVMLoweringPass>(useAlloca, useBarePtrCallConv,
                                            emitCWrappers);
}

static PassRegistration<LLVMLoweringPass>
    pass(PASS_NAME, "Convert scalar and vector operations from the "
                    "Standard to the LLVM dialect");
