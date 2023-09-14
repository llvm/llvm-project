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
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Threading.h"
#include <memory>
#include <mutex>
#include <optional>

using namespace mlir;

SmallVector<Type> &LLVMTypeConverter::getCurrentThreadRecursiveStack() {
  {
    // Most of the time, the entry already exists in the map.
    std::shared_lock<decltype(callStackMutex)> lock(callStackMutex,
                                                    std::defer_lock);
    if (getContext().isMultithreadingEnabled())
      lock.lock();
    auto recursiveStack = conversionCallStack.find(llvm::get_threadid());
    if (recursiveStack != conversionCallStack.end())
      return *recursiveStack->second;
  }

  // First time this thread gets here, we have to get an exclusive access to
  // inset in the map
  std::unique_lock<decltype(callStackMutex)> lock(callStackMutex);
  auto recursiveStackInserted = conversionCallStack.insert(std::make_pair(
      llvm::get_threadid(), std::make_unique<SmallVector<Type>>()));
  return *recursiveStackInserted.first->second;
}

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
  addConversion([&](VectorType type) -> std::optional<Type> {
    FailureOr<Type> llvmType = convertVectorType(type);
    if (failed(llvmType))
      return std::nullopt;
    return llvmType;
  });

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

  addConversion([&](LLVM::LLVMStructType type, SmallVectorImpl<Type> &results)
                    -> std::optional<LogicalResult> {
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

      SmallVectorImpl<Type> &recursiveStack = getCurrentThreadRecursiveStack();
      if (llvm::count(recursiveStack, type)) {
        results.push_back(convertedType);
        return success();
      }
      recursiveStack.push_back(type);
      auto popConversionCallStack = llvm::make_scope_exit(
          [&recursiveStack]() { recursiveStack.pop_back(); });

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

  // Integer memory spaces map to themselves.
  addTypeAttributeConversion(
      [](BaseMemRefType memref, IntegerAttr addrspace) { return addrspace; });
}

/// Returns the MLIR context.
MLIRContext &LLVMTypeConverter::getContext() const {
  return *getDialect()->getContext();
}

Type LLVMTypeConverter::getIndexType() const {
  return IntegerType::get(&getContext(), getIndexTypeBitwidth());
}

LLVM::LLVMPointerType
LLVMTypeConverter::getPointerType(Type elementType,
                                  unsigned int addressSpace) const {
  if (useOpaquePointers())
    return LLVM::LLVMPointerType::get(&getContext(), addressSpace);
  return LLVM::LLVMPointerType::get(elementType, addressSpace);
}

unsigned LLVMTypeConverter::getPointerBitwidth(unsigned addressSpace) const {
  return options.dataLayout.getPointerSizeInBits(addressSpace);
}

Type LLVMTypeConverter::convertIndexType(IndexType type) const {
  return getIndexType();
}

Type LLVMTypeConverter::convertIntegerType(IntegerType type) const {
  return IntegerType::get(&getContext(), type.getWidth());
}

Type LLVMTypeConverter::convertFloatType(FloatType type) const {
  if (type.isFloat8E5M2() || type.isFloat8E4M3FN() || type.isFloat8E5M2FNUZ() ||
      type.isFloat8E4M3FNUZ() || type.isFloat8E4M3B11FNUZ())
    return IntegerType::get(&getContext(), type.getWidth());
  return type;
}

// Convert a `ComplexType` to an LLVM type. The result is a complex number
// struct with entries for the
//   1. real part and for the
//   2. imaginary part.
Type LLVMTypeConverter::convertComplexType(ComplexType type) const {
  auto elementType = convertType(type.getElementType());
  return LLVM::LLVMStructType::getLiteral(&getContext(),
                                          {elementType, elementType});
}

// Except for signatures, MLIR function types are converted into LLVM
// pointer-to-function types.
Type LLVMTypeConverter::convertFunctionType(FunctionType type) const {
  SignatureConversion conversion(type.getNumInputs());
  Type converted = convertFunctionSignature(
      type, /*isVariadic=*/false, options.useBarePtrCallConv, conversion);
  if (!converted)
    return {};
  return getPointerType(converted);
}

// Function types are converted to LLVM Function types by recursively converting
// argument and result types.  If MLIR Function has zero results, the LLVM
// Function has one VoidType result.  If MLIR Function has more than one result,
// they are into an LLVM StructType in their order of appearance.
Type LLVMTypeConverter::convertFunctionSignature(
    FunctionType funcTy, bool isVariadic, bool useBarePtrCallConv,
    LLVMTypeConverter::SignatureConversion &result) const {
  // Select the argument converter depending on the calling convention.
  useBarePtrCallConv = useBarePtrCallConv || options.useBarePtrCallConv;
  auto funcArgConverter = useBarePtrCallConv ? barePtrFuncArgTypeConverter
                                             : structFuncArgTypeConverter;
  // Convert argument types one by one and check for errors.
  for (auto [idx, type] : llvm::enumerate(funcTy.getInputs())) {
    SmallVector<Type, 8> converted;
    if (failed(funcArgConverter(*this, type, converted)))
      return {};
    result.addInputs(idx, converted);
  }

  // If function does not return anything, create the void result type,
  // if it returns on element, convert it, otherwise pack the result types into
  // a struct.
  Type resultType =
      funcTy.getNumResults() == 0
          ? LLVM::LLVMVoidType::get(&getContext())
          : packFunctionResults(funcTy.getResults(), useBarePtrCallConv);
  if (!resultType)
    return {};
  return LLVM::LLVMFunctionType::get(resultType, result.getConvertedTypes(),
                                     isVariadic);
}

/// Converts the function type to a C-compatible format, in particular using
/// pointers to memref descriptors for arguments.
std::pair<LLVM::LLVMFunctionType, LLVM::LLVMStructType>
LLVMTypeConverter::convertFunctionTypeCWrapper(FunctionType type) const {
  SmallVector<Type, 4> inputs;

  Type resultType = type.getNumResults() == 0
                        ? LLVM::LLVMVoidType::get(&getContext())
                        : packFunctionResults(type.getResults());
  if (!resultType)
    return {};

  auto structType = dyn_cast<LLVM::LLVMStructType>(resultType);
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
    if (isa<MemRefType, UnrankedMemRefType>(t))
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
                                             bool unpackAggregates) const {
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

  FailureOr<unsigned> addressSpace = getMemRefAddressSpace(type);
  if (failed(addressSpace)) {
    emitError(UnknownLoc::get(type.getContext()),
              "conversion of memref memory space ")
        << type.getMemorySpace()
        << " to integer address space "
           "failed. Consider adding memory space conversions.";
    return {};
  }
  auto ptrTy = getPointerType(elementType, *addressSpace);

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

unsigned
LLVMTypeConverter::getMemRefDescriptorSize(MemRefType type,
                                           const DataLayout &layout) const {
  // Compute the descriptor size given that of its components indicated above.
  unsigned space = *getMemRefAddressSpace(type);
  return 2 * llvm::divideCeil(getPointerBitwidth(space), 8) +
         (1 + 2 * type.getRank()) * layout.getTypeSize(getIndexType());
}

/// Converts MemRefType to LLVMType. A MemRefType is converted to a struct that
/// packs the descriptor fields as defined by `getMemRefDescriptorFields`.
Type LLVMTypeConverter::convertMemRefType(MemRefType type) const {
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
SmallVector<Type, 2>
LLVMTypeConverter::getUnrankedMemRefDescriptorFields() const {
  return {getIndexType(), getPointerType(IntegerType::get(&getContext(), 8))};
}

unsigned LLVMTypeConverter::getUnrankedMemRefDescriptorSize(
    UnrankedMemRefType type, const DataLayout &layout) const {
  // Compute the descriptor size given that of its components indicated above.
  unsigned space = *getMemRefAddressSpace(type);
  return layout.getTypeSize(getIndexType()) +
         llvm::divideCeil(getPointerBitwidth(space), 8);
}

Type LLVMTypeConverter::convertUnrankedMemRefType(
    UnrankedMemRefType type) const {
  if (!convertType(type.getElementType()))
    return {};
  return LLVM::LLVMStructType::getLiteral(&getContext(),
                                          getUnrankedMemRefDescriptorFields());
}

FailureOr<unsigned>
LLVMTypeConverter::getMemRefAddressSpace(BaseMemRefType type) const {
  if (!type.getMemorySpace()) // Default memory space -> 0.
    return 0;
  std::optional<Attribute> converted =
      convertTypeAttribute(type, type.getMemorySpace());
  if (!converted)
    return failure();
  if (!(*converted)) // Conversion to default is 0.
    return 0;
  if (auto explicitSpace = llvm::dyn_cast_if_present<IntegerAttr>(*converted))
    return explicitSpace.getInt();
  return failure();
}

// Check if a memref type can be converted to a bare pointer.
bool LLVMTypeConverter::canConvertToBarePtr(BaseMemRefType type) {
  if (isa<UnrankedMemRefType>(type))
    // Unranked memref is not supported in the bare pointer calling convention.
    return false;

  // Check that the memref has static shape, strides and offset. Otherwise, it
  // cannot be lowered to a bare pointer.
  auto memrefTy = cast<MemRefType>(type);
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
Type LLVMTypeConverter::convertMemRefToBarePtr(BaseMemRefType type) const {
  if (!canConvertToBarePtr(type))
    return {};
  Type elementType = convertType(type.getElementType());
  if (!elementType)
    return {};
  FailureOr<unsigned> addressSpace = getMemRefAddressSpace(type);
  if (failed(addressSpace))
    return {};
  return getPointerType(elementType, *addressSpace);
}

/// Convert an n-D vector type to an LLVM vector type:
///  * 0-D `vector<T>` are converted to vector<1xT>
///  * 1-D `vector<axT>` remains as is while,
///  * n>1 `vector<ax...xkxT>` convert via an (n-1)-D array type to
///    `!llvm.array<ax...array<jxvector<kxT>>>`.
/// Returns failure for n-D scalable vector types as LLVM does not support
/// arrays of scalable vectors.
FailureOr<Type> LLVMTypeConverter::convertVectorType(VectorType type) const {
  auto elementType = convertType(type.getElementType());
  if (!elementType)
    return {};
  if (type.getShape().empty())
    return VectorType::get({1}, elementType);
  Type vectorType = VectorType::get(type.getShape().back(), elementType,
                                    type.getScalableDims().back());
  assert(LLVM::isCompatibleVectorType(vectorType) &&
         "expected vector type compatible with the LLVM dialect");
  if (type.isScalable() && (type.getRank() > 1))
    return failure();
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
Type LLVMTypeConverter::convertCallingConventionType(
    Type type, bool useBarePtrCallConv) const {
  if (useBarePtrCallConv)
    if (auto memrefTy = dyn_cast<BaseMemRefType>(type))
      return convertMemRefToBarePtr(memrefTy);

  return convertType(type);
}

/// Promote the bare pointers in 'values' that resulted from memrefs to
/// descriptors. 'stdTypes' holds they types of 'values' before the conversion
/// to the LLVM-IR dialect (i.e., MemRefType, or any other builtin type).
void LLVMTypeConverter::promoteBarePtrsToDescriptors(
    ConversionPatternRewriter &rewriter, Location loc, ArrayRef<Type> stdTypes,
    SmallVectorImpl<Value> &values) const {
  assert(stdTypes.size() == values.size() &&
         "The number of types and values doesn't match");
  for (unsigned i = 0, end = values.size(); i < end; ++i)
    if (auto memrefTy = dyn_cast<MemRefType>(stdTypes[i]))
      values[i] = MemRefDescriptor::fromStaticShape(rewriter, loc, *this,
                                                    memrefTy, values[i]);
}

/// Convert a non-empty list of types of values produced by an operation into an
/// LLVM-compatible type. In particular, if more than one value is
/// produced, create a literal structure with elements that correspond to each
/// of the types converted with `convertType`.
Type LLVMTypeConverter::packOperationResults(TypeRange types) const {
  assert(!types.empty() && "expected non-empty list of type");
  if (types.size() == 1)
    return convertType(types[0]);

  SmallVector<Type> resultTypes;
  resultTypes.reserve(types.size());
  for (Type type : types) {
    Type converted = convertType(type);
    if (!converted || !LLVM::isCompatibleType(converted))
      return {};
    resultTypes.push_back(converted);
  }

  return LLVM::LLVMStructType::getLiteral(&getContext(), resultTypes);
}

/// Convert a non-empty list of types to be returned from a function into an
/// LLVM-compatible type. In particular, if more than one value is returned,
/// create an LLVM dialect structure type with elements that correspond to each
/// of the types converted with `convertCallingConventionType`.
Type LLVMTypeConverter::packFunctionResults(TypeRange types,
                                            bool useBarePtrCallConv) const {
  assert(!types.empty() && "expected non-empty list of type");

  useBarePtrCallConv |= options.useBarePtrCallConv;
  if (types.size() == 1)
    return convertCallingConventionType(types.front(), useBarePtrCallConv);

  SmallVector<Type> resultTypes;
  resultTypes.reserve(types.size());
  for (auto t : types) {
    auto converted = convertCallingConventionType(t, useBarePtrCallConv);
    if (!converted || !LLVM::isCompatibleType(converted))
      return {};
    resultTypes.push_back(converted);
  }

  return LLVM::LLVMStructType::getLiteral(&getContext(), resultTypes);
}

Value LLVMTypeConverter::promoteOneMemRefDescriptor(Location loc, Value operand,
                                                    OpBuilder &builder) const {
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

SmallVector<Value, 4>
LLVMTypeConverter::promoteOperands(Location loc, ValueRange opOperands,
                                   ValueRange operands, OpBuilder &builder,
                                   bool useBarePtrCallConv) const {
  SmallVector<Value, 4> promotedOperands;
  promotedOperands.reserve(operands.size());
  useBarePtrCallConv |= options.useBarePtrCallConv;
  for (auto it : llvm::zip(opOperands, operands)) {
    auto operand = std::get<0>(it);
    auto llvmOperand = std::get<1>(it);

    if (useBarePtrCallConv) {
      // For the bare-ptr calling convention, we only have to extract the
      // aligned pointer of a memref.
      if (dyn_cast<MemRefType>(operand.getType())) {
        MemRefDescriptor desc(llvmOperand);
        llvmOperand = desc.alignedPtr(builder, loc);
      } else if (isa<UnrankedMemRefType>(operand.getType())) {
        llvm_unreachable("Unranked memrefs are not supported");
      }
    } else {
      if (isa<UnrankedMemRefType>(operand.getType())) {
        UnrankedMemRefDescriptor::unpack(builder, loc, llvmOperand,
                                         promotedOperands);
        continue;
      }
      if (auto memrefType = dyn_cast<MemRefType>(operand.getType())) {
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
LogicalResult
mlir::structFuncArgTypeConverter(const LLVMTypeConverter &converter, Type type,
                                 SmallVectorImpl<Type> &result) {
  if (auto memref = dyn_cast<MemRefType>(type)) {
    // In signatures, Memref descriptors are expanded into lists of
    // non-aggregate values.
    auto converted =
        converter.getMemRefDescriptorFields(memref, /*unpackAggregates=*/true);
    if (converted.empty())
      return failure();
    result.append(converted.begin(), converted.end());
    return success();
  }
  if (isa<UnrankedMemRefType>(type)) {
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
LogicalResult
mlir::barePtrFuncArgTypeConverter(const LLVMTypeConverter &converter, Type type,
                                  SmallVectorImpl<Type> &result) {
  auto llvmTy = converter.convertCallingConventionType(
      type, /*useBarePointerCallConv=*/true);
  if (!llvmTy)
    return failure();

  result.push_back(llvmTy);
  return success();
}
