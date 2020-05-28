//===- SPIRVLowering.cpp - SPIR-V lowering utilities ----------------------===//
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

#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/LayoutUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

#include <functional>

#define DEBUG_TYPE "mlir-spirv-lowering"

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

    SmallVector<StringRef, 4> extStrings;
    for (spirv::Extension ext : ors)
      extStrings.push_back(spirv::stringifyExtension(ext));

    LLVM_DEBUG(llvm::dbgs()
               << label << " illegal: requires at least one extension in ["
               << llvm::join(extStrings, ", ")
               << "] but none allowed in target environment\n");
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

    SmallVector<StringRef, 4> capStrings;
    for (spirv::Capability cap : ors)
      capStrings.push_back(spirv::stringifyCapability(cap));

    LLVM_DEBUG(llvm::dbgs()
               << label << " illegal: requires at least one capability in ["
               << llvm::join(capStrings, ", ")
               << "] but none allowed in target environment\n");
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

Type SPIRVTypeConverter::getIndexType(MLIRContext *context) {
  // Convert to 32-bit integers for now. Might need a way to control this in
  // future.
  // TODO(ravishankarm): It is probably better to make it 64-bit integers. To
  // this some support is needed in SPIR-V dialect for Conversion
  // instructions. The Vulkan spec requires the builtins like
  // GlobalInvocationID, etc. to be 32-bit (unsigned) integers which should be
  // SExtended to 64-bit for index computations.
  return IntegerType::get(32, context);
}

/// Mapping between SPIR-V storage classes to memref memory spaces.
///
/// Note: memref does not have a defined semantics for each memory space; it
/// depends on the context where it is used. There are no particular reasons
/// behind the number assignments; we try to follow NVVM conventions and largely
/// give common storage classes a smaller number. The hope is use symbolic
/// memory space representation eventually after memref supports it.
// TODO(antiagainst): swap Generic and StorageBuffer assignment to be more akin
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

#undef STORAGE_SPACE_MAP_LIST

// TODO(ravishankarm): This is a utility function that should probably be
// exposed by the SPIR-V dialect. Keeping it local till the use case arises.
static Optional<int64_t> getTypeNumBytes(Type t) {
  if (t.isa<spirv::ScalarType>()) {
    auto bitWidth = t.getIntOrFloatBitWidth();
    // According to the SPIR-V spec:
    // "There is no physical size or bit pattern defined for values with boolean
    // type. If they are stored (in conjunction with OpVariable), they can only
    // be used with logical addressing operations, not physical, and only with
    // non-externally visible shader Storage Classes: Workgroup, CrossWorkgroup,
    // Private, Function, Input, and Output."
    if (bitWidth == 1) {
      return llvm::None;
    }
    return bitWidth / 8;
  } else if (auto memRefType = t.dyn_cast<MemRefType>()) {
    // TODO: Layout should also be controlled by the ABI attributes. For now
    // using the layout from MemRef.
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    if (!memRefType.hasStaticShape() ||
        failed(getStridesAndOffset(memRefType, strides, offset))) {
      return llvm::None;
    }
    // To get the size of the memref object in memory, the total size is the
    // max(stride * dimension-size) computed for all dimensions times the size
    // of the element.
    auto elementSize = getTypeNumBytes(memRefType.getElementType());
    if (!elementSize) {
      return llvm::None;
    }
    if (memRefType.getRank() == 0) {
      return elementSize;
    }
    auto dims = memRefType.getShape();
    if (llvm::is_contained(dims, ShapedType::kDynamicSize) ||
        offset == MemRefType::getDynamicStrideOrOffset() ||
        llvm::is_contained(strides, MemRefType::getDynamicStrideOrOffset())) {
      return llvm::None;
    }
    int64_t memrefSize = -1;
    for (auto shape : enumerate(dims)) {
      memrefSize = std::max(memrefSize, shape.value() * strides[shape.index()]);
    }
    return (offset + memrefSize) * elementSize.getValue();
  } else if (auto tensorType = t.dyn_cast<TensorType>()) {
    if (!tensorType.hasStaticShape()) {
      return llvm::None;
    }
    auto elementSize = getTypeNumBytes(tensorType.getElementType());
    if (!elementSize) {
      return llvm::None;
    }
    int64_t size = elementSize.getValue();
    for (auto shape : tensorType.getShape()) {
      size *= shape;
    }
    return size;
  }
  // TODO: Add size computation for other types.
  return llvm::None;
}

Optional<int64_t> SPIRVTypeConverter::getConvertedTypeNumBytes(Type t) {
  return getTypeNumBytes(t);
}

/// Converts a scalar `type` to a suitable type under the given `targetEnv`.
static Optional<Type>
convertScalarType(const spirv::TargetEnv &targetEnv, spirv::ScalarType type,
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
  // TODO(antiagainst): We are unconditionally converting the bitwidth here,
  // this might be okay for non-interface types (i.e., types used in
  // Private/Function storage classes), but not for interface types (i.e.,
  // types used in StorageBuffer/Uniform/PushConstant/etc. storage classes).
  // This is because the later actually affects the ABI contract with the
  // runtime. So we may want to expose a control on SPIRVTypeConverter to fail
  // conversion if we cannot change there.

  if (auto floatType = type.dyn_cast<FloatType>()) {
    LLVM_DEBUG(llvm::dbgs() << type << " converted to 32-bit for SPIR-V\n");
    return Builder(targetEnv.getContext()).getF32Type();
  }

  auto intType = type.cast<IntegerType>();
  LLVM_DEBUG(llvm::dbgs() << type << " converted to 32-bit for SPIR-V\n");
  return IntegerType::get(/*width=*/32, intType.getSignedness(),
                          targetEnv.getContext());
}

/// Converts a vector `type` to a suitable type under the given `targetEnv`.
static Optional<Type>
convertVectorType(const spirv::TargetEnv &targetEnv, VectorType type,
                  Optional<spirv::StorageClass> storageClass = {}) {
  if (!spirv::CompositeType::isValid(type)) {
    // TODO(antiagainst): One-element vector types can be translated into scalar
    // types. Vector types with more than four elements can be translated into
    // array types.
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: 1- and > 4-element unimplemented\n");
    return llvm::None;
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
      targetEnv, type.getElementType().cast<spirv::ScalarType>(), storageClass);
  if (elementType)
    return VectorType::get(type.getShape(), *elementType);
  return llvm::None;
}

/// Converts a tensor `type` to a suitable type under the given `targetEnv`.
///
/// Note that this is mainly for lowering constant tensors.In SPIR-V one can
/// create composite constants with OpConstantComposite to embed relative large
/// constant values and use OpCompositeExtract and OpCompositeInsert to
/// manipulate, like what we do for vectors.
static Optional<Type> convertTensorType(const spirv::TargetEnv &targetEnv,
                                        TensorType type) {
  // TODO(ravishankarm) : Handle dynamic shapes.
  if (!type.hasStaticShape()) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: dynamic shape unimplemented\n");
    return llvm::None;
  }

  auto scalarType = type.getElementType().dyn_cast<spirv::ScalarType>();
  if (!scalarType) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot convert non-scalar element type\n");
    return llvm::None;
  }

  Optional<int64_t> scalarSize = getTypeNumBytes(scalarType);
  Optional<int64_t> tensorSize = getTypeNumBytes(type);
  if (!scalarSize || !tensorSize) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot deduce element count\n");
    return llvm::None;
  }

  auto arrayElemCount = *tensorSize / *scalarSize;
  auto arrayElemType = convertScalarType(targetEnv, scalarType);
  if (!arrayElemType)
    return llvm::None;
  Optional<int64_t> arrayElemSize = getTypeNumBytes(*arrayElemType);
  if (!arrayElemSize) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot deduce converted element size\n");
    return llvm::None;
  }

  return spirv::ArrayType::get(*arrayElemType, arrayElemCount, *arrayElemSize);
}

static Optional<Type> convertMemrefType(const spirv::TargetEnv &targetEnv,
                                        MemRefType type) {
  Optional<spirv::StorageClass> storageClass =
      SPIRVTypeConverter::getStorageClassForMemorySpace(type.getMemorySpace());
  if (!storageClass) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot convert memory space\n");
    return llvm::None;
  }

  auto scalarType = type.getElementType().dyn_cast<spirv::ScalarType>();
  if (!scalarType) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot convert non-scalar element type\n");
    return llvm::None;
  }

  auto arrayElemType = convertScalarType(targetEnv, scalarType, storageClass);
  if (!arrayElemType)
    return llvm::None;

  Optional<int64_t> scalarSize = getTypeNumBytes(scalarType);
  if (!scalarSize) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot deduce element size\n");
    return llvm::None;
  }

  if (!type.hasStaticShape()) {
    auto arrayType = spirv::RuntimeArrayType::get(*arrayElemType, *scalarSize);
    // Wrap in a struct to satisfy Vulkan interface requirements.
    auto structType = spirv::StructType::get(arrayType, 0);
    return spirv::PointerType::get(structType, *storageClass);
  }

  Optional<int64_t> memrefSize = getTypeNumBytes(type);
  if (!memrefSize) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot deduce element count\n");
    return llvm::None;
  }

  auto arrayElemCount = *memrefSize / *scalarSize;

  Optional<int64_t> arrayElemSize = getTypeNumBytes(*arrayElemType);
  if (!arrayElemSize) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot deduce converted element size\n");
    return llvm::None;
  }

  auto arrayType =
      spirv::ArrayType::get(*arrayElemType, arrayElemCount, *arrayElemSize);

  // Wrap in a struct to satisfy Vulkan interface requirements. Memrefs with
  // workgroup storage class do not need the struct to be laid out explicitly.
  auto structType = *storageClass == spirv::StorageClass::Workgroup
                        ? spirv::StructType::get(arrayType)
                        : spirv::StructType::get(arrayType, 0);
  return spirv::PointerType::get(structType, *storageClass);
}

SPIRVTypeConverter::SPIRVTypeConverter(spirv::TargetEnvAttr targetAttr)
    : targetEnv(targetAttr) {
  // Add conversions. The order matters here: later ones will be tried earlier.

  // All other cases failed. Then we cannot convert this type.
  addConversion([](Type type) { return llvm::None; });

  // Allow all SPIR-V dialect specific types. This assumes all standard types
  // adopted in the SPIR-V dialect (i.e., IntegerType, FloatType, VectorType)
  // were tried before.
  //
  // TODO(antiagainst): this assumes that the SPIR-V types are valid to use in
  // the given target environment, which should be the case if the whole
  // pipeline is driven by the same target environment. Still, we probably still
  // want to validate and convert to be safe.
  addConversion([](spirv::SPIRVType type) { return type; });

  addConversion([](IndexType indexType) {
    return SPIRVTypeConverter::getIndexType(indexType.getContext());
  });

  addConversion([this](IntegerType intType) -> Optional<Type> {
    if (auto scalarType = intType.dyn_cast<spirv::ScalarType>())
      return convertScalarType(targetEnv, scalarType);
    return llvm::None;
  });

  addConversion([this](FloatType floatType) -> Optional<Type> {
    if (auto scalarType = floatType.dyn_cast<spirv::ScalarType>())
      return convertScalarType(targetEnv, scalarType);
    return llvm::None;
  });

  addConversion([this](VectorType vectorType) {
    return convertVectorType(targetEnv, vectorType);
  });

  addConversion([this](TensorType tensorType) {
    return convertTensorType(targetEnv, tensorType);
  });

  addConversion([this](MemRefType memRefType) {
    return convertMemrefType(targetEnv, memRefType);
  });
}

//===----------------------------------------------------------------------===//
// FuncOp Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
/// A pattern for rewriting function signature to convert arguments of functions
/// to be of valid SPIR-V types.
class FuncOpConversion final : public SPIRVOpLowering<FuncOp> {
public:
  using SPIRVOpLowering<FuncOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
FuncOpConversion::matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
  auto fnType = funcOp.getType();
  // TODO(antiagainst): support converting functions with one result.
  if (fnType.getNumResults())
    return failure();

  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  for (auto argType : enumerate(funcOp.getType().getInputs())) {
    auto convertedType = typeConverter.convertType(argType.value());
    if (!convertedType)
      return failure();
    signatureConverter.addInputs(argType.index(), convertedType);
  }

  // Create the converted spv.func op.
  auto newFuncOp = rewriter.create<spirv::FuncOp>(
      funcOp.getLoc(), funcOp.getName(),
      rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                               llvm::None));

  // Copy over all attributes other than the function name and type.
  for (const auto &namedAttr : funcOp.getAttrs()) {
    if (namedAttr.first != impl::getTypeAttrName() &&
        namedAttr.first != SymbolTable::getSymbolAttrName())
      newFuncOp.setAttr(namedAttr.first, namedAttr.second);
  }

  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  rewriter.applySignatureConversion(&newFuncOp.getBody(), signatureConverter);
  rewriter.eraseOp(funcOp);
  return success();
}

void mlir::populateBuiltinFuncToSPIRVPatterns(
    MLIRContext *context, SPIRVTypeConverter &typeConverter,
    OwningRewritePatternList &patterns) {
  patterns.insert<FuncOpConversion>(context, typeConverter);
}

//===----------------------------------------------------------------------===//
// Builtin Variables
//===----------------------------------------------------------------------===//

static spirv::GlobalVariableOp getBuiltinVariable(Block &body,
                                                  spirv::BuiltIn builtin) {
  // Look through all global variables in the given `body` block and check if
  // there is a spv.globalVariable that has the same `builtin` attribute.
  for (auto varOp : body.getOps<spirv::GlobalVariableOp>()) {
    if (auto builtinAttr = varOp.getAttrOfType<StringAttr>(
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
                           OpBuilder &builder) {
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
    auto ptrType = spirv::PointerType::get(
        VectorType::get({3}, builder.getIntegerType(32)),
        spirv::StorageClass::Input);
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
                                           OpBuilder &builder) {
  Operation *parent = SymbolTable::getNearestSymbolTable(op->getParentOp());
  if (!parent) {
    op->emitError("expected operation to be within a module-like op");
    return nullptr;
  }

  spirv::GlobalVariableOp varOp = getOrInsertBuiltinVariable(
      *parent->getRegion(0).begin(), op->getLoc(), builtin, builder);
  Value ptr = builder.create<spirv::AddressOfOp>(op->getLoc(), varOp);
  return builder.create<spirv::LoadOp>(op->getLoc(), ptr);
}

//===----------------------------------------------------------------------===//
// Index calculation
//===----------------------------------------------------------------------===//

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

  auto indexType = typeConverter.getIndexType(builder.getContext());

  SmallVector<Value, 2> linearizedIndices;
  // Add a '0' at the start to index into the struct.
  auto zero = spirv::ConstantOp::getZero(indexType, loc, builder);
  linearizedIndices.push_back(zero);

  if (baseType.getRank() == 0) {
    linearizedIndices.push_back(zero);
  } else {
    // TODO: Instead of this logic, use affine.apply and add patterns for
    // lowering affine.apply to standard ops. These will get lowered to SPIR-V
    // ops by the DialectConversion framework.
    Value ptrLoc = builder.create<spirv::ConstantOp>(
        loc, indexType, IntegerAttr::get(indexType, offset));
    assert(indices.size() == strides.size() &&
           "must provide indices for all dimensions");
    for (auto index : llvm::enumerate(indices)) {
      Value strideVal = builder.create<spirv::ConstantOp>(
          loc, indexType, IntegerAttr::get(indexType, strides[index.index()]));
      Value update =
          builder.create<spirv::IMulOp>(loc, strideVal, index.value());
      ptrLoc = builder.create<spirv::IAddOp>(loc, ptrLoc, update);
    }
    linearizedIndices.push_back(ptrLoc);
  }
  return builder.create<spirv::AccessChainOp>(loc, basePtr, linearizedIndices);
}

//===----------------------------------------------------------------------===//
// Set ABI attributes for lowering entry functions.
//===----------------------------------------------------------------------===//

LogicalResult
mlir::spirv::setABIAttrs(spirv::FuncOp funcOp,
                         spirv::EntryPointABIAttr entryPointInfo,
                         ArrayRef<spirv::InterfaceVarABIAttr> argABIInfo) {
  // Set the attributes for argument and the function.
  StringRef argABIAttrName = spirv::getInterfaceVarABIAttrName();
  for (auto argIndex : llvm::seq<unsigned>(0, funcOp.getNumArguments())) {
    funcOp.setArgAttr(argIndex, argABIAttrName, argABIInfo[argIndex]);
  }
  funcOp.setAttr(spirv::getEntryPointABIAttrName(), entryPointInfo);
  return success();
}

//===----------------------------------------------------------------------===//
// SPIR-V ConversionTarget
//===----------------------------------------------------------------------===//

std::unique_ptr<spirv::SPIRVConversionTarget>
spirv::SPIRVConversionTarget::get(spirv::TargetEnvAttr targetAttr) {
  std::unique_ptr<SPIRVConversionTarget> target(
      // std::make_unique does not work here because the constructor is private.
      new SPIRVConversionTarget(targetAttr));
  SPIRVConversionTarget *targetPtr = target.get();
  target->addDynamicallyLegalDialect<SPIRVDialect>(
      Optional<ConversionTarget::DynamicLegalityCallbackFn>(
          // We need to capture the raw pointer here because it is stable:
          // target will be destroyed once this function is returned.
          [targetPtr](Operation *op) { return targetPtr->isLegalOp(op); }));
  return target;
}

spirv::SPIRVConversionTarget::SPIRVConversionTarget(
    spirv::TargetEnvAttr targetAttr)
    : ConversionTarget(*targetAttr.getContext()), targetEnv(targetAttr) {}

bool spirv::SPIRVConversionTarget::isLegalOp(Operation *op) {
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
