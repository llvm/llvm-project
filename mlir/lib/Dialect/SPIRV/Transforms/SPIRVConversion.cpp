//===- SPIRVConversion.cpp - SPIR-V Conversion Utilities ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities used to lower to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

#include <functional>

#define DEBUG_TYPE "mlir-spirv-conversion"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Checks that `candidates` extension requirements are possible to be satisfied
/// with the given `targetEnv`.
///
///  `candidates` is a vector of vector for extension requirements following
/// ((Extension::A OR Extension::B) AND (Extension::C OR Extension::D))
/// convention.
template <typename LabelT>
static LogicalResult checkExtensionRequirements(
    LabelT label, const spirv::TargetEnv &targetEnv,
    const spirv::SPIRVType::ExtensionArrayRefVector &candidates) {
  for (const auto &ors : candidates) {
    if (targetEnv.allows(ors))
      continue;

    LLVM_DEBUG({
      SmallVector<StringRef> extStrings;
      for (spirv::Extension ext : ors)
        extStrings.push_back(spirv::stringifyExtension(ext));

      llvm::dbgs() << label << " illegal: requires at least one extension in ["
                   << llvm::join(extStrings, ", ")
                   << "] but none allowed in target environment\n";
    });
    return failure();
  }
  return success();
}

/// Checks that `candidates`capability requirements are possible to be satisfied
/// with the given `isAllowedFn`.
///
///  `candidates` is a vector of vector for capability requirements following
/// ((Capability::A OR Capability::B) AND (Capability::C OR Capability::D))
/// convention.
template <typename LabelT>
static LogicalResult checkCapabilityRequirements(
    LabelT label, const spirv::TargetEnv &targetEnv,
    const spirv::SPIRVType::CapabilityArrayRefVector &candidates) {
  for (const auto &ors : candidates) {
    if (targetEnv.allows(ors))
      continue;

    LLVM_DEBUG({
      SmallVector<StringRef> capStrings;
      for (spirv::Capability cap : ors)
        capStrings.push_back(spirv::stringifyCapability(cap));

      llvm::dbgs() << label << " illegal: requires at least one capability in ["
                   << llvm::join(capStrings, ", ")
                   << "] but none allowed in target environment\n";
    });
    return failure();
  }
  return success();
}

/// Returns true if the given `storageClass` needs explicit layout when used in
/// Shader environments.
static bool needsExplicitLayout(spirv::StorageClass storageClass) {
  switch (storageClass) {
  case spirv::StorageClass::PhysicalStorageBuffer:
  case spirv::StorageClass::PushConstant:
  case spirv::StorageClass::StorageBuffer:
  case spirv::StorageClass::Uniform:
    return true;
  default:
    return false;
  }
}

/// Wraps the given `elementType` in a struct and gets the pointer to the
/// struct. This is used to satisfy Vulkan interface requirements.
static spirv::PointerType
wrapInStructAndGetPointer(Type elementType, spirv::StorageClass storageClass) {
  auto structType = needsExplicitLayout(storageClass)
                        ? spirv::StructType::get(elementType, /*offsetInfo=*/0)
                        : spirv::StructType::get(elementType);
  return spirv::PointerType::get(structType, storageClass);
}

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

Type SPIRVTypeConverter::getIndexType() const {
  return IntegerType::get(getContext(), options.use64bitIndex ? 64 : 32);
}

/// Mapping between SPIR-V storage classes to memref memory spaces.
///
/// Note: memref does not have a defined semantics for each memory space; it
/// depends on the context where it is used. There are no particular reasons
/// behind the number assignments; we try to follow NVVM conventions and largely
/// give common storage classes a smaller number. The hope is use symbolic
/// memory space representation eventually after memref supports it.
// TODO: swap Generic and StorageBuffer assignment to be more akin
// to NVVM.
#define STORAGE_SPACE_MAP_LIST(MAP_FN)                                         \
  MAP_FN(spirv::StorageClass::Generic, 1)                                      \
  MAP_FN(spirv::StorageClass::StorageBuffer, 0)                                \
  MAP_FN(spirv::StorageClass::Workgroup, 3)                                    \
  MAP_FN(spirv::StorageClass::Uniform, 4)                                      \
  MAP_FN(spirv::StorageClass::Private, 5)                                      \
  MAP_FN(spirv::StorageClass::Function, 6)                                     \
  MAP_FN(spirv::StorageClass::PushConstant, 7)                                 \
  MAP_FN(spirv::StorageClass::UniformConstant, 8)                              \
  MAP_FN(spirv::StorageClass::Input, 9)                                        \
  MAP_FN(spirv::StorageClass::Output, 10)                                      \
  MAP_FN(spirv::StorageClass::CrossWorkgroup, 11)                              \
  MAP_FN(spirv::StorageClass::AtomicCounter, 12)                               \
  MAP_FN(spirv::StorageClass::Image, 13)                                       \
  MAP_FN(spirv::StorageClass::CallableDataNV, 14)                              \
  MAP_FN(spirv::StorageClass::IncomingCallableDataNV, 15)                      \
  MAP_FN(spirv::StorageClass::RayPayloadNV, 16)                                \
  MAP_FN(spirv::StorageClass::HitAttributeNV, 17)                              \
  MAP_FN(spirv::StorageClass::IncomingRayPayloadNV, 18)                        \
  MAP_FN(spirv::StorageClass::ShaderRecordBufferNV, 19)                        \
  MAP_FN(spirv::StorageClass::PhysicalStorageBuffer, 20)

unsigned
SPIRVTypeConverter::getMemorySpaceForStorageClass(spirv::StorageClass storage) {
#define STORAGE_SPACE_MAP_FN(storage, space)                                   \
  case storage:                                                                \
    return space;

  switch (storage) { STORAGE_SPACE_MAP_LIST(STORAGE_SPACE_MAP_FN) }
#undef STORAGE_SPACE_MAP_FN
  llvm_unreachable("unhandled storage class!");
}

Optional<spirv::StorageClass>
SPIRVTypeConverter::getStorageClassForMemorySpace(unsigned space) {
#define STORAGE_SPACE_MAP_FN(storage, space)                                   \
  case space:                                                                  \
    return storage;

  switch (space) {
    STORAGE_SPACE_MAP_LIST(STORAGE_SPACE_MAP_FN)
  default:
    return llvm::None;
  }
#undef STORAGE_SPACE_MAP_FN
}

const SPIRVTypeConverter::Options &SPIRVTypeConverter::getOptions() const {
  return options;
}

MLIRContext *SPIRVTypeConverter::getContext() const {
  return targetEnv.getAttr().getContext();
}

#undef STORAGE_SPACE_MAP_LIST

// TODO: This is a utility function that should probably be exposed by the
// SPIR-V dialect. Keeping it local till the use case arises.
static Optional<int64_t>
getTypeNumBytes(const SPIRVTypeConverter::Options &options, Type type) {
  if (type.isa<spirv::ScalarType>()) {
    auto bitWidth = type.getIntOrFloatBitWidth();
    // According to the SPIR-V spec:
    // "There is no physical size or bit pattern defined for values with boolean
    // type. If they are stored (in conjunction with OpVariable), they can only
    // be used with logical addressing operations, not physical, and only with
    // non-externally visible shader Storage Classes: Workgroup, CrossWorkgroup,
    // Private, Function, Input, and Output."
    if (bitWidth == 1)
      return llvm::None;
    return bitWidth / 8;
  }

  if (auto vecType = type.dyn_cast<VectorType>()) {
    auto elementSize = getTypeNumBytes(options, vecType.getElementType());
    if (!elementSize)
      return llvm::None;
    return vecType.getNumElements() * elementSize.getValue();
  }

  if (auto memRefType = type.dyn_cast<MemRefType>()) {
    // TODO: Layout should also be controlled by the ABI attributes. For now
    // using the layout from MemRef.
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    if (!memRefType.hasStaticShape() ||
        failed(getStridesAndOffset(memRefType, strides, offset)))
      return llvm::None;

    // To get the size of the memref object in memory, the total size is the
    // max(stride * dimension-size) computed for all dimensions times the size
    // of the element.
    auto elementSize = getTypeNumBytes(options, memRefType.getElementType());
    if (!elementSize)
      return llvm::None;

    if (memRefType.getRank() == 0)
      return elementSize;

    auto dims = memRefType.getShape();
    if (llvm::is_contained(dims, ShapedType::kDynamicSize) ||
        offset == MemRefType::getDynamicStrideOrOffset() ||
        llvm::is_contained(strides, MemRefType::getDynamicStrideOrOffset()))
      return llvm::None;

    int64_t memrefSize = -1;
    for (auto shape : enumerate(dims))
      memrefSize = std::max(memrefSize, shape.value() * strides[shape.index()]);

    return (offset + memrefSize) * elementSize.getValue();
  }

  if (auto tensorType = type.dyn_cast<TensorType>()) {
    if (!tensorType.hasStaticShape())
      return llvm::None;

    auto elementSize = getTypeNumBytes(options, tensorType.getElementType());
    if (!elementSize)
      return llvm::None;

    int64_t size = elementSize.getValue();
    for (auto shape : tensorType.getShape())
      size *= shape;

    return size;
  }

  // TODO: Add size computation for other types.
  return llvm::None;
}

/// Converts a scalar `type` to a suitable type under the given `targetEnv`.
static Type convertScalarType(const spirv::TargetEnv &targetEnv,
                              const SPIRVTypeConverter::Options &options,
                              spirv::ScalarType type,
                              Optional<spirv::StorageClass> storageClass = {}) {
  // Get extension and capability requirements for the given type.
  SmallVector<ArrayRef<spirv::Extension>, 1> extensions;
  SmallVector<ArrayRef<spirv::Capability>, 2> capabilities;
  type.getExtensions(extensions, storageClass);
  type.getCapabilities(capabilities, storageClass);

  // If all requirements are met, then we can accept this type as-is.
  if (succeeded(checkCapabilityRequirements(type, targetEnv, capabilities)) &&
      succeeded(checkExtensionRequirements(type, targetEnv, extensions)))
    return type;

  // Otherwise we need to adjust the type, which really means adjusting the
  // bitwidth given this is a scalar type.

  if (!options.emulateNon32BitScalarTypes)
    return nullptr;

  if (auto floatType = type.dyn_cast<FloatType>()) {
    LLVM_DEBUG(llvm::dbgs() << type << " converted to 32-bit for SPIR-V\n");
    return Builder(targetEnv.getContext()).getF32Type();
  }

  auto intType = type.cast<IntegerType>();
  LLVM_DEBUG(llvm::dbgs() << type << " converted to 32-bit for SPIR-V\n");
  return IntegerType::get(targetEnv.getContext(), /*width=*/32,
                          intType.getSignedness());
}

/// Converts a vector `type` to a suitable type under the given `targetEnv`.
static Type convertVectorType(const spirv::TargetEnv &targetEnv,
                              const SPIRVTypeConverter::Options &options,
                              VectorType type,
                              Optional<spirv::StorageClass> storageClass = {}) {
  if (type.getRank() == 1 && type.getNumElements() == 1)
    return type.getElementType();

  if (!spirv::CompositeType::isValid(type)) {
    // TODO: Vector types with more than four elements can be translated into
    // array types.
    LLVM_DEBUG(llvm::dbgs() << type << " illegal: > 4-element unimplemented\n");
    return nullptr;
  }

  // Get extension and capability requirements for the given type.
  SmallVector<ArrayRef<spirv::Extension>, 1> extensions;
  SmallVector<ArrayRef<spirv::Capability>, 2> capabilities;
  type.cast<spirv::CompositeType>().getExtensions(extensions, storageClass);
  type.cast<spirv::CompositeType>().getCapabilities(capabilities, storageClass);

  // If all requirements are met, then we can accept this type as-is.
  if (succeeded(checkCapabilityRequirements(type, targetEnv, capabilities)) &&
      succeeded(checkExtensionRequirements(type, targetEnv, extensions)))
    return type;

  auto elementType = convertScalarType(
      targetEnv, options, type.getElementType().cast<spirv::ScalarType>(),
      storageClass);
  if (elementType)
    return VectorType::get(type.getShape(), elementType);
  return nullptr;
}

/// Converts a tensor `type` to a suitable type under the given `targetEnv`.
///
/// Note that this is mainly for lowering constant tensors. In SPIR-V one can
/// create composite constants with OpConstantComposite to embed relative large
/// constant values and use OpCompositeExtract and OpCompositeInsert to
/// manipulate, like what we do for vectors.
static Type convertTensorType(const spirv::TargetEnv &targetEnv,
                              const SPIRVTypeConverter::Options &options,
                              TensorType type) {
  // TODO: Handle dynamic shapes.
  if (!type.hasStaticShape()) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: dynamic shape unimplemented\n");
    return nullptr;
  }

  auto scalarType = type.getElementType().dyn_cast<spirv::ScalarType>();
  if (!scalarType) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot convert non-scalar element type\n");
    return nullptr;
  }

  Optional<int64_t> scalarSize = getTypeNumBytes(options, scalarType);
  Optional<int64_t> tensorSize = getTypeNumBytes(options, type);
  if (!scalarSize || !tensorSize) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot deduce element count\n");
    return nullptr;
  }

  auto arrayElemCount = *tensorSize / *scalarSize;
  auto arrayElemType = convertScalarType(targetEnv, options, scalarType);
  if (!arrayElemType)
    return nullptr;
  Optional<int64_t> arrayElemSize = getTypeNumBytes(options, arrayElemType);
  if (!arrayElemSize) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot deduce converted element size\n");
    return nullptr;
  }

  return spirv::ArrayType::get(arrayElemType, arrayElemCount, *arrayElemSize);
}

static Type convertBoolMemrefType(const spirv::TargetEnv &targetEnv,
                                  const SPIRVTypeConverter::Options &options,
                                  MemRefType type) {
  Optional<spirv::StorageClass> storageClass =
      SPIRVTypeConverter::getStorageClassForMemorySpace(
          type.getMemorySpaceAsInt());
  if (!storageClass) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot convert memory space\n");
    return nullptr;
  }

  unsigned numBoolBits = options.boolNumBits;
  if (numBoolBits != 8) {
    LLVM_DEBUG(llvm::dbgs()
               << "using non-8-bit storage for bool types unimplemented");
    return nullptr;
  }
  auto elementType = IntegerType::get(type.getContext(), numBoolBits)
                         .dyn_cast<spirv::ScalarType>();
  if (!elementType)
    return nullptr;
  Type arrayElemType =
      convertScalarType(targetEnv, options, elementType, storageClass);
  if (!arrayElemType)
    return nullptr;
  Optional<int64_t> arrayElemSize = getTypeNumBytes(options, arrayElemType);
  if (!arrayElemSize) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot deduce converted element size\n");
    return nullptr;
  }

  if (!type.hasStaticShape()) {
    auto arrayType =
        spirv::RuntimeArrayType::get(arrayElemType, *arrayElemSize);
    return wrapInStructAndGetPointer(arrayType, *storageClass);
  }

  int64_t memrefSize = (type.getNumElements() * numBoolBits + 7) / 8;
  auto arrayElemCount = (memrefSize + *arrayElemSize - 1) / *arrayElemSize;
  auto arrayType =
      spirv::ArrayType::get(arrayElemType, arrayElemCount, *arrayElemSize);

  return wrapInStructAndGetPointer(arrayType, *storageClass);
}

static Type convertMemrefType(const spirv::TargetEnv &targetEnv,
                              const SPIRVTypeConverter::Options &options,
                              MemRefType type) {
  if (type.getElementType().isa<IntegerType>() &&
      type.getElementTypeBitWidth() == 1) {
    return convertBoolMemrefType(targetEnv, options, type);
  }

  Optional<spirv::StorageClass> storageClass =
      SPIRVTypeConverter::getStorageClassForMemorySpace(
          type.getMemorySpaceAsInt());
  if (!storageClass) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot convert memory space\n");
    return nullptr;
  }

  Type arrayElemType;
  Type elementType = type.getElementType();
  if (auto vecType = elementType.dyn_cast<VectorType>()) {
    arrayElemType =
        convertVectorType(targetEnv, options, vecType, storageClass);
  } else if (auto scalarType = elementType.dyn_cast<spirv::ScalarType>()) {
    arrayElemType =
        convertScalarType(targetEnv, options, scalarType, storageClass);
  } else {
    LLVM_DEBUG(
        llvm::dbgs()
        << type
        << " unhandled: can only convert scalar or vector element type\n");
    return nullptr;
  }
  if (!arrayElemType)
    return nullptr;

  Optional<int64_t> elementSize = getTypeNumBytes(options, elementType);
  if (!elementSize) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot deduce element size\n");
    return nullptr;
  }

  Optional<int64_t> arrayElemSize = getTypeNumBytes(options, arrayElemType);
  if (!arrayElemSize) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot deduce converted element size\n");
    return nullptr;
  }

  if (!type.hasStaticShape()) {
    auto arrayType =
        spirv::RuntimeArrayType::get(arrayElemType, *arrayElemSize);
    return wrapInStructAndGetPointer(arrayType, *storageClass);
  }

  Optional<int64_t> memrefSize = getTypeNumBytes(options, type);
  if (!memrefSize) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot deduce element count\n");
    return nullptr;
  }

  auto arrayElemCount = *memrefSize / *elementSize;


  auto arrayType =
      spirv::ArrayType::get(arrayElemType, arrayElemCount, *arrayElemSize);

  return wrapInStructAndGetPointer(arrayType, *storageClass);
}

SPIRVTypeConverter::SPIRVTypeConverter(spirv::TargetEnvAttr targetAttr,
                                       Options options)
    : targetEnv(targetAttr), options(options) {
  // Add conversions. The order matters here: later ones will be tried earlier.

  // Allow all SPIR-V dialect specific types. This assumes all builtin types
  // adopted in the SPIR-V dialect (i.e., IntegerType, FloatType, VectorType)
  // were tried before.
  //
  // TODO: this assumes that the SPIR-V types are valid to use in
  // the given target environment, which should be the case if the whole
  // pipeline is driven by the same target environment. Still, we probably still
  // want to validate and convert to be safe.
  addConversion([](spirv::SPIRVType type) { return type; });

  addConversion([this](IndexType /*indexType*/) { return getIndexType(); });

  addConversion([this](IntegerType intType) -> Optional<Type> {
    if (auto scalarType = intType.dyn_cast<spirv::ScalarType>())
      return convertScalarType(this->targetEnv, this->options, scalarType);
    return Type();
  });

  addConversion([this](FloatType floatType) -> Optional<Type> {
    if (auto scalarType = floatType.dyn_cast<spirv::ScalarType>())
      return convertScalarType(this->targetEnv, this->options, scalarType);
    return Type();
  });

  addConversion([this](VectorType vectorType) {
    return convertVectorType(this->targetEnv, this->options, vectorType);
  });

  addConversion([this](TensorType tensorType) {
    return convertTensorType(this->targetEnv, this->options, tensorType);
  });

  addConversion([this](MemRefType memRefType) {
    return convertMemrefType(this->targetEnv, this->options, memRefType);
  });
}

//===----------------------------------------------------------------------===//
// FuncOp Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
/// A pattern for rewriting function signature to convert arguments of functions
/// to be of valid SPIR-V types.
class FuncOpConversion final : public OpConversionPattern<FuncOp> {
public:
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
FuncOpConversion::matchAndRewrite(FuncOp funcOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  auto fnType = funcOp.getType();
  if (fnType.getNumResults() > 1)
    return failure();

  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  for (auto argType : enumerate(fnType.getInputs())) {
    auto convertedType = getTypeConverter()->convertType(argType.value());
    if (!convertedType)
      return failure();
    signatureConverter.addInputs(argType.index(), convertedType);
  }

  Type resultType;
  if (fnType.getNumResults() == 1) {
    resultType = getTypeConverter()->convertType(fnType.getResult(0));
    if (!resultType)
      return failure();
  }

  // Create the converted spv.func op.
  auto newFuncOp = rewriter.create<spirv::FuncOp>(
      funcOp.getLoc(), funcOp.getName(),
      rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                               resultType ? TypeRange(resultType)
                                          : TypeRange()));

  // Copy over all attributes other than the function name and type.
  for (const auto &namedAttr : funcOp->getAttrs()) {
    if (namedAttr.first != function_like_impl::getTypeAttrName() &&
        namedAttr.first != SymbolTable::getSymbolAttrName())
      newFuncOp->setAttr(namedAttr.first, namedAttr.second);
  }

  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  if (failed(rewriter.convertRegionTypes(
          &newFuncOp.getBody(), *getTypeConverter(), &signatureConverter)))
    return failure();
  rewriter.eraseOp(funcOp);
  return success();
}

void mlir::populateBuiltinFuncToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                              RewritePatternSet &patterns) {
  patterns.add<FuncOpConversion>(typeConverter, patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Builtin Variables
//===----------------------------------------------------------------------===//

static spirv::GlobalVariableOp getBuiltinVariable(Block &body,
                                                  spirv::BuiltIn builtin) {
  // Look through all global variables in the given `body` block and check if
  // there is a spv.GlobalVariable that has the same `builtin` attribute.
  for (auto varOp : body.getOps<spirv::GlobalVariableOp>()) {
    if (auto builtinAttr = varOp->getAttrOfType<StringAttr>(
            spirv::SPIRVDialect::getAttributeName(
                spirv::Decoration::BuiltIn))) {
      auto varBuiltIn = spirv::symbolizeBuiltIn(builtinAttr.getValue());
      if (varBuiltIn && varBuiltIn.getValue() == builtin) {
        return varOp;
      }
    }
  }
  return nullptr;
}

/// Gets name of global variable for a builtin.
static std::string getBuiltinVarName(spirv::BuiltIn builtin) {
  return std::string("__builtin_var_") + stringifyBuiltIn(builtin).str() + "__";
}

/// Gets or inserts a global variable for a builtin within `body` block.
static spirv::GlobalVariableOp
getOrInsertBuiltinVariable(Block &body, Location loc, spirv::BuiltIn builtin,
                           Type integerType, OpBuilder &builder) {
  if (auto varOp = getBuiltinVariable(body, builtin))
    return varOp;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&body);

  spirv::GlobalVariableOp newVarOp;
  switch (builtin) {
  case spirv::BuiltIn::NumWorkgroups:
  case spirv::BuiltIn::WorkgroupSize:
  case spirv::BuiltIn::WorkgroupId:
  case spirv::BuiltIn::LocalInvocationId:
  case spirv::BuiltIn::GlobalInvocationId: {
    auto ptrType = spirv::PointerType::get(VectorType::get({3}, integerType),
                                           spirv::StorageClass::Input);
    std::string name = getBuiltinVarName(builtin);
    newVarOp =
        builder.create<spirv::GlobalVariableOp>(loc, ptrType, name, builtin);
    break;
  }
  case spirv::BuiltIn::SubgroupId:
  case spirv::BuiltIn::NumSubgroups:
  case spirv::BuiltIn::SubgroupSize: {
    auto ptrType =
        spirv::PointerType::get(integerType, spirv::StorageClass::Input);
    std::string name = getBuiltinVarName(builtin);
    newVarOp =
        builder.create<spirv::GlobalVariableOp>(loc, ptrType, name, builtin);
    break;
  }
  default:
    emitError(loc, "unimplemented builtin variable generation for ")
        << stringifyBuiltIn(builtin);
  }
  return newVarOp;
}

Value mlir::spirv::getBuiltinVariableValue(Operation *op,
                                           spirv::BuiltIn builtin,
                                           Type integerType,
                                           OpBuilder &builder) {
  Operation *parent = SymbolTable::getNearestSymbolTable(op->getParentOp());
  if (!parent) {
    op->emitError("expected operation to be within a module-like op");
    return nullptr;
  }

  spirv::GlobalVariableOp varOp =
      getOrInsertBuiltinVariable(*parent->getRegion(0).begin(), op->getLoc(),
                                 builtin, integerType, builder);
  Value ptr = builder.create<spirv::AddressOfOp>(op->getLoc(), varOp);
  return builder.create<spirv::LoadOp>(op->getLoc(), ptr);
}

//===----------------------------------------------------------------------===//
// Push constant storage
//===----------------------------------------------------------------------===//

/// Returns the pointer type for the push constant storage containing
/// `elementCount` 32-bit integer values.
static spirv::PointerType getPushConstantStorageType(unsigned elementCount,
                                                     Builder &builder,
                                                     Type indexType) {
  auto arrayType = spirv::ArrayType::get(indexType, elementCount,
                                         /*stride=*/4);
  auto structType = spirv::StructType::get({arrayType}, /*offsetInfo=*/0);
  return spirv::PointerType::get(structType, spirv::StorageClass::PushConstant);
}

/// Returns the push constant varible containing `elementCount` 32-bit integer
/// values in `body`. Returns null op if such an op does not exit.
static spirv::GlobalVariableOp getPushConstantVariable(Block &body,
                                                       unsigned elementCount) {
  for (auto varOp : body.getOps<spirv::GlobalVariableOp>()) {
    auto ptrType = varOp.type().dyn_cast<spirv::PointerType>();
    if (!ptrType)
      continue;

    // Note that Vulkan requires "There must be no more than one push constant
    // block statically used per shader entry point." So we should always reuse
    // the existing one.
    if (ptrType.getStorageClass() == spirv::StorageClass::PushConstant) {
      auto numElements = ptrType.getPointeeType()
                             .cast<spirv::StructType>()
                             .getElementType(0)
                             .cast<spirv::ArrayType>()
                             .getNumElements();
      if (numElements == elementCount)
        return varOp;
    }
  }
  return nullptr;
}

/// Gets or inserts a global variable for push constant storage containing
/// `elementCount` 32-bit integer values in `block`.
static spirv::GlobalVariableOp
getOrInsertPushConstantVariable(Location loc, Block &block,
                                unsigned elementCount, OpBuilder &b,
                                Type indexType) {
  if (auto varOp = getPushConstantVariable(block, elementCount))
    return varOp;

  auto builder = OpBuilder::atBlockBegin(&block, b.getListener());
  auto type = getPushConstantStorageType(elementCount, builder, indexType);
  const char *name = "__push_constant_var__";
  return builder.create<spirv::GlobalVariableOp>(loc, type, name,
                                                 /*initializer=*/nullptr);
}

Value spirv::getPushConstantValue(Operation *op, unsigned elementCount,
                                  unsigned offset, Type integerType,
                                  OpBuilder &builder) {
  Location loc = op->getLoc();
  Operation *parent = SymbolTable::getNearestSymbolTable(op->getParentOp());
  if (!parent) {
    op->emitError("expected operation to be within a module-like op");
    return nullptr;
  }

  spirv::GlobalVariableOp varOp = getOrInsertPushConstantVariable(
      loc, parent->getRegion(0).front(), elementCount, builder, integerType);

  Value zeroOp = spirv::ConstantOp::getZero(integerType, loc, builder);
  Value offsetOp = builder.create<spirv::ConstantOp>(
      loc, integerType, builder.getI32IntegerAttr(offset));
  auto addrOp = builder.create<spirv::AddressOfOp>(loc, varOp);
  auto acOp = builder.create<spirv::AccessChainOp>(
      loc, addrOp, llvm::makeArrayRef({zeroOp, offsetOp}));
  return builder.create<spirv::LoadOp>(loc, acOp);
}

//===----------------------------------------------------------------------===//
// Index calculation
//===----------------------------------------------------------------------===//

Value mlir::spirv::linearizeIndex(ValueRange indices, ArrayRef<int64_t> strides,
                                  int64_t offset, Type integerType,
                                  Location loc, OpBuilder &builder) {
  assert(indices.size() == strides.size() &&
         "must provide indices for all dimensions");

  // TODO: Consider moving to use affine.apply and patterns converting
  // affine.apply to standard ops. This needs converting to SPIR-V passes to be
  // broken down into progressive small steps so we can have intermediate steps
  // using other dialects. At the moment SPIR-V is the final sink.

  Value linearizedIndex = builder.create<spirv::ConstantOp>(
      loc, integerType, IntegerAttr::get(integerType, offset));
  for (auto index : llvm::enumerate(indices)) {
    Value strideVal = builder.create<spirv::ConstantOp>(
        loc, integerType,
        IntegerAttr::get(integerType, strides[index.index()]));
    Value update = builder.create<spirv::IMulOp>(loc, strideVal, index.value());
    linearizedIndex =
        builder.create<spirv::IAddOp>(loc, linearizedIndex, update);
  }
  return linearizedIndex;
}

spirv::AccessChainOp mlir::spirv::getElementPtr(
    SPIRVTypeConverter &typeConverter, MemRefType baseType, Value basePtr,
    ValueRange indices, Location loc, OpBuilder &builder) {
  // Get base and offset of the MemRefType and verify they are static.

  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(baseType, strides, offset)) ||
      llvm::is_contained(strides, MemRefType::getDynamicStrideOrOffset()) ||
      offset == MemRefType::getDynamicStrideOrOffset()) {
    return nullptr;
  }

  auto indexType = typeConverter.getIndexType();

  SmallVector<Value, 2> linearizedIndices;
  auto zero = spirv::ConstantOp::getZero(indexType, loc, builder);

  // Add a '0' at the start to index into the struct.
  linearizedIndices.push_back(zero);

  if (baseType.getRank() == 0) {
    linearizedIndices.push_back(zero);
  } else {
    linearizedIndices.push_back(
        linearizeIndex(indices, strides, offset, indexType, loc, builder));
  }
  return builder.create<spirv::AccessChainOp>(loc, basePtr, linearizedIndices);
}

//===----------------------------------------------------------------------===//
// SPIR-V ConversionTarget
//===----------------------------------------------------------------------===//

std::unique_ptr<SPIRVConversionTarget>
SPIRVConversionTarget::get(spirv::TargetEnvAttr targetAttr) {
  std::unique_ptr<SPIRVConversionTarget> target(
      // std::make_unique does not work here because the constructor is private.
      new SPIRVConversionTarget(targetAttr));
  SPIRVConversionTarget *targetPtr = target.get();
  target->addDynamicallyLegalDialect<spirv::SPIRVDialect>(
      // We need to capture the raw pointer here because it is stable:
      // target will be destroyed once this function is returned.
      [targetPtr](Operation *op) { return targetPtr->isLegalOp(op); });
  return target;
}

SPIRVConversionTarget::SPIRVConversionTarget(spirv::TargetEnvAttr targetAttr)
    : ConversionTarget(*targetAttr.getContext()), targetEnv(targetAttr) {}

bool SPIRVConversionTarget::isLegalOp(Operation *op) {
  // Make sure this op is available at the given version. Ops not implementing
  // QueryMinVersionInterface/QueryMaxVersionInterface are available to all
  // SPIR-V versions.
  if (auto minVersion = dyn_cast<spirv::QueryMinVersionInterface>(op))
    if (minVersion.getMinVersion() > this->targetEnv.getVersion()) {
      LLVM_DEBUG(llvm::dbgs()
                 << op->getName() << " illegal: requiring min version "
                 << spirv::stringifyVersion(minVersion.getMinVersion())
                 << "\n");
      return false;
    }
  if (auto maxVersion = dyn_cast<spirv::QueryMaxVersionInterface>(op))
    if (maxVersion.getMaxVersion() < this->targetEnv.getVersion()) {
      LLVM_DEBUG(llvm::dbgs()
                 << op->getName() << " illegal: requiring max version "
                 << spirv::stringifyVersion(maxVersion.getMaxVersion())
                 << "\n");
      return false;
    }

  // Make sure this op's required extensions are allowed to use. Ops not
  // implementing QueryExtensionInterface do not require extensions to be
  // available.
  if (auto extensions = dyn_cast<spirv::QueryExtensionInterface>(op))
    if (failed(checkExtensionRequirements(op->getName(), this->targetEnv,
                                          extensions.getExtensions())))
      return false;

  // Make sure this op's required extensions are allowed to use. Ops not
  // implementing QueryCapabilityInterface do not require capabilities to be
  // available.
  if (auto capabilities = dyn_cast<spirv::QueryCapabilityInterface>(op))
    if (failed(checkCapabilityRequirements(op->getName(), this->targetEnv,
                                           capabilities.getCapabilities())))
      return false;

  SmallVector<Type, 4> valueTypes;
  valueTypes.append(op->operand_type_begin(), op->operand_type_end());
  valueTypes.append(op->result_type_begin(), op->result_type_end());

  // Special treatment for global variables, whose type requirements are
  // conveyed by type attributes.
  if (auto globalVar = dyn_cast<spirv::GlobalVariableOp>(op))
    valueTypes.push_back(globalVar.type());

  // Make sure the op's operands/results use types that are allowed by the
  // target environment.
  SmallVector<ArrayRef<spirv::Extension>, 4> typeExtensions;
  SmallVector<ArrayRef<spirv::Capability>, 8> typeCapabilities;
  for (Type valueType : valueTypes) {
    typeExtensions.clear();
    valueType.cast<spirv::SPIRVType>().getExtensions(typeExtensions);
    if (failed(checkExtensionRequirements(op->getName(), this->targetEnv,
                                          typeExtensions)))
      return false;

    typeCapabilities.clear();
    valueType.cast<spirv::SPIRVType>().getCapabilities(typeCapabilities);
    if (failed(checkCapabilityRequirements(op->getName(), this->targetEnv,
                                           typeCapabilities)))
      return false;
  }

  return true;
}
