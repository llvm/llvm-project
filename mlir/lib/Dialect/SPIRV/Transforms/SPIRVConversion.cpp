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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
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

MLIRContext *SPIRVTypeConverter::getContext() const {
  return targetEnv.getAttr().getContext();
}

bool SPIRVTypeConverter::allows(spirv::Capability capability) {
  return targetEnv.allows(capability);
}

// TODO: This is a utility function that should probably be exposed by the
// SPIR-V dialect. Keeping it local till the use case arises.
static Optional<int64_t> getTypeNumBytes(const SPIRVConversionOptions &options,
                                         Type type) {
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
    return vecType.getNumElements() * *elementSize;
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
    if (llvm::is_contained(dims, ShapedType::kDynamic) ||
        ShapedType::isDynamic(offset) ||
        llvm::is_contained(strides, ShapedType::kDynamic))
      return llvm::None;

    int64_t memrefSize = -1;
    for (const auto &shape : enumerate(dims))
      memrefSize = std::max(memrefSize, shape.value() * strides[shape.index()]);

    return (offset + memrefSize) * *elementSize;
  }

  if (auto tensorType = type.dyn_cast<TensorType>()) {
    if (!tensorType.hasStaticShape())
      return llvm::None;

    auto elementSize = getTypeNumBytes(options, tensorType.getElementType());
    if (!elementSize)
      return llvm::None;

    int64_t size = *elementSize;
    for (auto shape : tensorType.getShape())
      size *= shape;

    return size;
  }

  // TODO: Add size computation for other types.
  return llvm::None;
}

/// Converts a scalar `type` to a suitable type under the given `targetEnv`.
static Type convertScalarType(const spirv::TargetEnv &targetEnv,
                              const SPIRVConversionOptions &options,
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
  if (!options.emulateLT32BitScalarTypes)
    return nullptr;

  // We only emulate narrower scalar types here and do not truncate results.
  if (type.getIntOrFloatBitWidth() > 32) {
    LLVM_DEBUG(llvm::dbgs()
               << type
               << " not converted to 32-bit for SPIR-V to avoid truncation\n");
    return nullptr;
  }

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
                              const SPIRVConversionOptions &options,
                              VectorType type,
                              Optional<spirv::StorageClass> storageClass = {}) {
  auto scalarType = type.getElementType().cast<spirv::ScalarType>();
  if (type.getRank() <= 1 && type.getNumElements() == 1)
    return convertScalarType(targetEnv, options, scalarType, storageClass);

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

  auto elementType =
      convertScalarType(targetEnv, options, scalarType, storageClass);
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
                              const SPIRVConversionOptions &options,
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

  return spirv::ArrayType::get(arrayElemType, arrayElemCount);
}

static Type convertBoolMemrefType(const spirv::TargetEnv &targetEnv,
                                  const SPIRVConversionOptions &options,
                                  MemRefType type,
                                  spirv::StorageClass storageClass) {
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
    // For OpenCL Kernel, dynamic shaped memrefs convert into a pointer pointing
    // to the element.
    if (targetEnv.allows(spirv::Capability::Kernel))
      return spirv::PointerType::get(arrayElemType, storageClass);
    int64_t stride = needsExplicitLayout(storageClass) ? *arrayElemSize : 0;
    auto arrayType = spirv::RuntimeArrayType::get(arrayElemType, stride);
    // For Vulkan we need extra wrapping struct and array to satisfy interface
    // needs.
    return wrapInStructAndGetPointer(arrayType, storageClass);
  }

  int64_t memrefSize = (type.getNumElements() * numBoolBits + 7) / 8;
  auto arrayElemCount = llvm::divideCeil(memrefSize, *arrayElemSize);
  int64_t stride = needsExplicitLayout(storageClass) ? *arrayElemSize : 0;
  auto arrayType = spirv::ArrayType::get(arrayElemType, arrayElemCount, stride);
  if (targetEnv.allows(spirv::Capability::Kernel))
    return spirv::PointerType::get(arrayType, storageClass);
  return wrapInStructAndGetPointer(arrayType, storageClass);
}

static Type convertMemrefType(const spirv::TargetEnv &targetEnv,
                              const SPIRVConversionOptions &options,
                              MemRefType type) {
  auto attr = type.getMemorySpace().dyn_cast_or_null<spirv::StorageClassAttr>();
  if (!attr) {
    LLVM_DEBUG(
        llvm::dbgs()
        << type
        << " illegal: expected memory space to be a SPIR-V storage class "
           "attribute; please use MemorySpaceToStorageClassConverter to map "
           "numeric memory spaces beforehand\n");
    return nullptr;
  }
  spirv::StorageClass storageClass = attr.getValue();

  if (type.getElementType().isa<IntegerType>() &&
      type.getElementTypeBitWidth() == 1) {
    return convertBoolMemrefType(targetEnv, options, type, storageClass);
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

  Optional<int64_t> arrayElemSize = getTypeNumBytes(options, arrayElemType);
  if (!arrayElemSize) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot deduce converted element size\n");
    return nullptr;
  }


  if (!type.hasStaticShape()) {
    // For OpenCL Kernel, dynamic shaped memrefs convert into a pointer pointing
    // to the element.
    if (targetEnv.allows(spirv::Capability::Kernel))
      return spirv::PointerType::get(arrayElemType, storageClass);
    int64_t stride = needsExplicitLayout(storageClass) ? *arrayElemSize : 0;
    auto arrayType = spirv::RuntimeArrayType::get(arrayElemType, stride);
    // For Vulkan we need extra wrapping struct and array to satisfy interface
    // needs.
    return wrapInStructAndGetPointer(arrayType, storageClass);
  }

  Optional<int64_t> memrefSize = getTypeNumBytes(options, type);
  if (!memrefSize) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot deduce element count\n");
    return nullptr;
  }

  auto arrayElemCount = llvm::divideCeil(*memrefSize, *arrayElemSize);
  int64_t stride = needsExplicitLayout(storageClass) ? *arrayElemSize : 0;
  auto arrayType = spirv::ArrayType::get(arrayElemType, arrayElemCount, stride);
  if (targetEnv.allows(spirv::Capability::Kernel))
    return spirv::PointerType::get(arrayType, storageClass);
  return wrapInStructAndGetPointer(arrayType, storageClass);
}

SPIRVTypeConverter::SPIRVTypeConverter(spirv::TargetEnvAttr targetAttr,
                                       const SPIRVConversionOptions &options)
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
// func::FuncOp Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
/// A pattern for rewriting function signature to convert arguments of functions
/// to be of valid SPIR-V types.
class FuncOpConversion final : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
FuncOpConversion::matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  auto fnType = funcOp.getFunctionType();
  if (fnType.getNumResults() > 1)
    return failure();

  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  for (const auto &argType : enumerate(fnType.getInputs())) {
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

  // Create the converted spirv.func op.
  auto newFuncOp = rewriter.create<spirv::FuncOp>(
      funcOp.getLoc(), funcOp.getName(),
      rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                               resultType ? TypeRange(resultType)
                                          : TypeRange()));

  // Copy over all attributes other than the function name and type.
  for (const auto &namedAttr : funcOp->getAttrs()) {
    if (namedAttr.getName() != FunctionOpInterface::getTypeAttrName() &&
        namedAttr.getName() != SymbolTable::getSymbolAttrName())
      newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
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
  // there is a spirv.GlobalVariable that has the same `builtin` attribute.
  for (auto varOp : body.getOps<spirv::GlobalVariableOp>()) {
    if (auto builtinAttr = varOp->getAttrOfType<StringAttr>(
            spirv::SPIRVDialect::getAttributeName(
                spirv::Decoration::BuiltIn))) {
      auto varBuiltIn = spirv::symbolizeBuiltIn(builtinAttr.getValue());
      if (varBuiltIn && *varBuiltIn == builtin) {
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
    auto ptrType = varOp.getType().dyn_cast<spirv::PointerType>();
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
  for (const auto &index : llvm::enumerate(indices)) {
    Value strideVal = builder.create<spirv::ConstantOp>(
        loc, integerType,
        IntegerAttr::get(integerType, strides[index.index()]));
    Value update = builder.create<spirv::IMulOp>(loc, strideVal, index.value());
    linearizedIndex =
        builder.create<spirv::IAddOp>(loc, linearizedIndex, update);
  }
  return linearizedIndex;
}

Value mlir::spirv::getVulkanElementPtr(SPIRVTypeConverter &typeConverter,
                                       MemRefType baseType, Value basePtr,
                                       ValueRange indices, Location loc,
                                       OpBuilder &builder) {
  // Get base and offset of the MemRefType and verify they are static.

  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(baseType, strides, offset)) ||
      llvm::is_contained(strides, ShapedType::kDynamic) ||
      ShapedType::isDynamic(offset)) {
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

Value mlir::spirv::getOpenCLElementPtr(SPIRVTypeConverter &typeConverter,
                                       MemRefType baseType, Value basePtr,
                                       ValueRange indices, Location loc,
                                       OpBuilder &builder) {
  // Get base and offset of the MemRefType and verify they are static.

  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(baseType, strides, offset)) ||
      llvm::is_contained(strides, ShapedType::kDynamic) ||
      ShapedType::isDynamic(offset)) {
    return nullptr;
  }

  auto indexType = typeConverter.getIndexType();

  SmallVector<Value, 2> linearizedIndices;
  Value linearIndex;
  if (baseType.getRank() == 0) {
    linearIndex = spirv::ConstantOp::getZero(indexType, loc, builder);
  } else {
    linearIndex =
        linearizeIndex(indices, strides, offset, indexType, loc, builder);
  }
  Type pointeeType =
      basePtr.getType().cast<spirv::PointerType>().getPointeeType();
  if (pointeeType.isa<spirv::ArrayType>()) {
    linearizedIndices.push_back(linearIndex);
    return builder.create<spirv::AccessChainOp>(loc, basePtr,
                                                linearizedIndices);
  }
  return builder.create<spirv::PtrAccessChainOp>(loc, basePtr, linearIndex,
                                                 linearizedIndices);
}

Value mlir::spirv::getElementPtr(SPIRVTypeConverter &typeConverter,
                                 MemRefType baseType, Value basePtr,
                                 ValueRange indices, Location loc,
                                 OpBuilder &builder) {

  if (typeConverter.allows(spirv::Capability::Kernel)) {
    return getOpenCLElementPtr(typeConverter, baseType, basePtr, indices, loc,
                               builder);
  }

  return getVulkanElementPtr(typeConverter, baseType, basePtr, indices, loc,
                             builder);
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
  if (auto minVersionIfx = dyn_cast<spirv::QueryMinVersionInterface>(op)) {
    Optional<spirv::Version> minVersion = minVersionIfx.getMinVersion();
    if (minVersion && *minVersion > this->targetEnv.getVersion()) {
      LLVM_DEBUG(llvm::dbgs()
                 << op->getName() << " illegal: requiring min version "
                 << spirv::stringifyVersion(*minVersion) << "\n");
      return false;
    }
  }
  if (auto maxVersionIfx = dyn_cast<spirv::QueryMaxVersionInterface>(op)) {
    Optional<spirv::Version> maxVersion = maxVersionIfx.getMaxVersion();
    if (maxVersion && *maxVersion < this->targetEnv.getVersion()) {
      LLVM_DEBUG(llvm::dbgs()
                 << op->getName() << " illegal: requiring max version "
                 << spirv::stringifyVersion(*maxVersion) << "\n");
      return false;
    }
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

  // Ensure that all types have been converted to SPIRV types.
  if (llvm::any_of(valueTypes,
                   [](Type t) { return !t.isa<spirv::SPIRVType>(); }))
    return false;

  // Special treatment for global variables, whose type requirements are
  // conveyed by type attributes.
  if (auto globalVar = dyn_cast<spirv::GlobalVariableOp>(op))
    valueTypes.push_back(globalVar.getType());

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
