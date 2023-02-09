//===- TypeConverter.cpp - Convert builtin to LLVM dialect types ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "MemRefDescriptor.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

using namespace mlir;

/// Create an LLVMTypeConverter using default LowerToLLVMOptions.
LLVMTypeConverter::LLVMTypeConverter(MLIRContext *ctx,
                                     const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, LowerToLLVMOptions(ctx), analysis) {}

/// Create an LLVMTypeConverter using custom LowerToLLVMOptions.
LLVMTypeConverter::LLVMTypeConverter(MLIRContext *ctx,
                                     const LowerToLLVMOptions &options,
                                     const DataLayoutAnalysis *analysis)
    : llvmDialect(ctx->getOrLoadDialect<LLVM::LLVMDialect>()), options(options),
      dataLayoutAnalysis(analysis) {
  assert(llvmDialect && "LLVM IR dialect is not registered");

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

  // LLVM-compatible types are legal, so add a pass-through conversion. Do this
  // before the conversions below since conversions are attempted in reverse
  // order and those should take priority.
  addConversion([](Type type) {
    return LLVM::isCompatibleType(type) ? std::optional<Type>(type)
                                        : std::nullopt;
  });

  // LLVM container types may (recursively) contain other types that must be
  // converted even when the outer type is compatible.
  addConversion([&](LLVM::LLVMPointerType type) -> std::optional<Type> {
    if (type.isOpaque())
      return type;
    if (auto pointee = convertType(type.getElementType()))
      return LLVM::LLVMPointerType::get(pointee, type.getAddressSpace());
    return std::nullopt;
  });
  addConversion([&](LLVM::LLVMStructType type, SmallVectorImpl<Type> &results,
                    ArrayRef<Type> callStack) -> std::optional<LogicalResult> {
    // Fastpath for types that won't be converted by this callback anyway.
    if (LLVM::isCompatibleType(type)) {
      results.push_back(type);
      return success();
    }

    if (type.isIdentified()) {
      auto convertedType = LLVM::LLVMStructType::getIdentified(
          type.getContext(), ("_Converted_" + type.getName()).str());
      unsigned counter = 1;
      while (convertedType.isInitialized()) {
        assert(counter != UINT_MAX &&
               "about to overflow struct renaming counter in conversion");
        convertedType = LLVM::LLVMStructType::getIdentified(
            type.getContext(),
            ("_Converted_" + std::to_string(counter) + type.getName()).str());
      }
      if (llvm::count(callStack, type) > 1) {
        results.push_back(convertedType);
        return success();
      }

      SmallVector<Type> convertedElemTypes;
      convertedElemTypes.reserve(type.getBody().size());
      if (failed(convertTypes(type.getBody(), convertedElemTypes)))
        return std::nullopt;

      if (failed(convertedType.setBody(convertedElemTypes, type.isPacked())))
        return failure();
      results.push_back(convertedType);
      return success();
    }

    SmallVector<Type> convertedSubtypes;
    convertedSubtypes.reserve(type.getBody().size());
    if (failed(convertTypes(type.getBody(), convertedSubtypes)))
      return std::nullopt;

    results.push_back(LLVM::LLVMStructType::getLiteral(
        type.getContext(), convertedSubtypes, type.isPacked()));
    return success();
  });
  addConversion([&](LLVM::LLVMArrayType type) -> std::optional<Type> {
    if (auto element = convertType(type.getElementType()))
      return LLVM::LLVMArrayType::get(element, type.getNumElements());
    return std::nullopt;
  });
  addConversion([&](LLVM::LLVMFunctionType type) -> std::optional<Type> {
    Type convertedResType = convertType(type.getReturnType());
    if (!convertedResType)
      return std::nullopt;

    SmallVector<Type> convertedArgTypes;
    convertedArgTypes.reserve(type.getNumParams());
    if (failed(convertTypes(type.getParams(), convertedArgTypes)))
      return std::nullopt;

    return LLVM::LLVMFunctionType::get(convertedResType, convertedArgTypes,
                                       type.isVarArg());
  });

  // Materialization for memrefs creates descriptor structs from individual
  // values constituting them, when descriptors are used, i.e. more than one
  // value represents a memref.
  addArgumentMaterialization(
      [&](OpBuilder &builder, UnrankedMemRefType resultType, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        if (inputs.size() == 1)
          return std::nullopt;
        return UnrankedMemRefDescriptor::pack(builder, loc, *this, resultType,
                                              inputs);
      });
  addArgumentMaterialization([&](OpBuilder &builder, MemRefType resultType,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
    // TODO: bare ptr conversion could be handled here but we would need a way
    // to distinguish between FuncOp and other regions.
    if (inputs.size() == 1)
      return std::nullopt;
    return MemRefDescriptor::pack(builder, loc, *this, resultType, inputs);
  });
  // Add generic source and target materializations to handle cases where
  // non-LLVM types persist after an LLVM conversion.
  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

/// Returns the MLIR context.
MLIRContext &LLVMTypeConverter::getContext() {
  return *getDialect()->getContext();
}

Type LLVMTypeConverter::getIndexType() {
  return IntegerType::get(&getContext(), getIndexTypeBitwidth());
}

LLVM::LLVMPointerType
LLVMTypeConverter::getPointerType(Type elementType, unsigned int addressSpace) {
  if (useOpaquePointers())
    return LLVM::LLVMPointerType::get(&getContext(), addressSpace);
  return LLVM::LLVMPointerType::get(elementType, addressSpace);
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
  if (!converted)
    return {};
  return getPointerType(converted);
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

  // If function does not return anything, create the void result type,
  // if it returns on element, convert it, otherwise pack the result types into
  // a struct.
  Type resultType = funcTy.getNumResults() == 0
                        ? LLVM::LLVMVoidType::get(&getContext())
                        : packFunctionResults(funcTy.getResults());
  if (!resultType)
    return {};
  return LLVM::LLVMFunctionType::get(resultType, result.getConvertedTypes(),
                                     isVariadic);
}

/// Converts the function type to a C-compatible format, in particular using
/// pointers to memref descriptors for arguments.
std::pair<LLVM::LLVMFunctionType, LLVM::LLVMStructType>
LLVMTypeConverter::convertFunctionTypeCWrapper(FunctionType type) {
  SmallVector<Type, 4> inputs;

  Type resultType = type.getNumResults() == 0
                        ? LLVM::LLVMVoidType::get(&getContext())
                        : packFunctionResults(type.getResults());
  if (!resultType)
    return {};

  auto structType = resultType.dyn_cast<LLVM::LLVMStructType>();
  if (structType) {
    // Struct types cannot be safely returned via C interface. Make this a
    // pointer argument, instead.
    inputs.push_back(getPointerType(structType));
    resultType = LLVM::LLVMVoidType::get(&getContext());
  }

  for (Type t : type.getInputs()) {
    auto converted = convertType(t);
    if (!converted || !LLVM::isCompatibleType(converted))
      return {};
    if (t.isa<MemRefType, UnrankedMemRefType>())
      converted = getPointerType(converted);
    inputs.push_back(converted);
  }

  return {LLVM::LLVMFunctionType::get(resultType, inputs), structType};
}

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
  if (!isStrided(type)) {
    emitError(
        UnknownLoc::get(type.getContext()),
        "conversion to strided form failed either due to non-strided layout "
        "maps (which should have been normalized away) or other reasons");
    return {};
  }

  Type elementType = convertType(type.getElementType());
  if (!elementType)
    return {};

  LLVM::LLVMPointerType ptrTy =
      getPointerType(elementType, type.getMemorySpaceAsInt());
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

unsigned LLVMTypeConverter::getMemRefDescriptorSize(MemRefType type,
                                                    const DataLayout &layout) {
  // Compute the descriptor size given that of its components indicated above.
  unsigned space = type.getMemorySpaceAsInt();
  return 2 * llvm::divideCeil(getPointerBitwidth(space), 8) +
         (1 + 2 * type.getRank()) * layout.getTypeSize(getIndexType());
}

/// Converts MemRefType to LLVMType. A MemRefType is converted to a struct that
/// packs the descriptor fields as defined by `getMemRefDescriptorFields`.
Type LLVMTypeConverter::convertMemRefType(MemRefType type) {
  // When converting a MemRefType to a struct with descriptor fields, do not
  // unpack the `sizes` and `strides` arrays.
  SmallVector<Type, 5> types =
      getMemRefDescriptorFields(type, /*unpackAggregates=*/false);
  if (types.empty())
    return {};
  return LLVM::LLVMStructType::getLiteral(&getContext(), types);
}

/// Convert an unranked memref type into a list of non-aggregate LLVM IR types
/// that will form the unranked memref descriptor. In particular, the fields
/// for an unranked memref descriptor are:
/// 1. index-typed rank, the dynamic rank of this MemRef
/// 2. void* ptr, pointer to the static ranked MemRef descriptor. This will be
///    stack allocated (alloca) copy of a MemRef descriptor that got casted to
///    be unranked.
SmallVector<Type, 2> LLVMTypeConverter::getUnrankedMemRefDescriptorFields() {
  return {getIndexType(), getPointerType(IntegerType::get(&getContext(), 8))};
}

unsigned
LLVMTypeConverter::getUnrankedMemRefDescriptorSize(UnrankedMemRefType type,
                                                   const DataLayout &layout) {
  // Compute the descriptor size given that of its components indicated above.
  unsigned space = type.getMemorySpaceAsInt();
  return layout.getTypeSize(getIndexType()) +
         llvm::divideCeil(getPointerBitwidth(space), 8);
}

Type LLVMTypeConverter::convertUnrankedMemRefType(UnrankedMemRefType type) {
  if (!convertType(type.getElementType()))
    return {};
  return LLVM::LLVMStructType::getLiteral(&getContext(),
                                          getUnrankedMemRefDescriptorFields());
}

// Check if a memref type can be converted to a bare pointer.
bool LLVMTypeConverter::canConvertToBarePtr(BaseMemRefType type) {
  if (type.isa<UnrankedMemRefType>())
    // Unranked memref is not supported in the bare pointer calling convention.
    return false;

  // Check that the memref has static shape, strides and offset. Otherwise, it
  // cannot be lowered to a bare pointer.
  auto memrefTy = type.cast<MemRefType>();
  if (!memrefTy.hasStaticShape())
    return false;

  int64_t offset = 0;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(memrefTy, strides, offset)))
    return false;

  for (int64_t stride : strides)
    if (ShapedType::isDynamic(stride))
      return false;

  return !ShapedType::isDynamic(offset);
}

/// Convert a memref type to a bare pointer to the memref element type.
Type LLVMTypeConverter::convertMemRefToBarePtr(BaseMemRefType type) {
  if (!canConvertToBarePtr(type))
    return {};
  Type elementType = convertType(type.getElementType());
  if (!elementType)
    return {};
  return getPointerType(elementType, type.getMemorySpaceAsInt());
}

/// Convert an n-D vector type to an LLVM vector type:
///  * 0-D `vector<T>` are converted to vector<1xT>
///  * 1-D `vector<axT>` remains as is while,
///  * n>1 `vector<ax...xkxT>` convert via an (n-1)-D array type to
///    `!llvm.array<ax...array<jxvector<kxT>>>`.
Type LLVMTypeConverter::convertVectorType(VectorType type) {
  auto elementType = convertType(type.getElementType());
  if (!elementType)
    return {};
  if (type.getShape().empty())
    return VectorType::get({1}, elementType);
  Type vectorType = VectorType::get(type.getShape().back(), elementType,
                                    type.getNumScalableDims());
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
  // Alloca with proper alignment. We do not expect optimizations of this
  // alloca op and so we omit allocating at the entry block.
  auto ptrType = getPointerType(operand.getType());
  Value one = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                               builder.getIndexAttr(1));
  Value allocated =
      builder.create<LLVM::AllocaOp>(loc, ptrType, operand.getType(), one);
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
