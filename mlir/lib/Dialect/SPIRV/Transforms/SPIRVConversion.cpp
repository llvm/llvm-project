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
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <functional>
#include <optional>

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
    const spirv::SPIRVType::ExtensionArrayRefVector &candidates,
    ArrayRef<spirv::Extension> elidedCandidates = {}) {
  for (const auto &ors : candidates) {
    if (targetEnv.allows(ors) ||
        llvm::any_of(elidedCandidates, [&ors](spirv::Extension elidedExt) {
          return llvm::is_contained(ors, elidedExt);
        }))
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
    const spirv::SPIRVType::CapabilityArrayRefVector &candidates,
    ArrayRef<spirv::Capability> elidedCandidates = {}) {
  for (const auto &ors : candidates) {
    if (targetEnv.allows(ors) ||
        llvm::any_of(elidedCandidates, [&ors](spirv::Capability elidedCap) {
          return llvm::is_contained(ors, elidedCap);
        }))
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

/// Check capabilities and extensions requirements
/// Checks that `capCandidates`, `extCandidates`, and capability
/// (`capCandidates`) infered extension requirements are possible to be
/// satisfied with the given `targetEnv`.
/// It also provides a way to relax requirements for certain capabilities and
/// extensions (e.g., `elidedCapCandidates`, `elidedExtCandidates`), this is to
/// allow passes to relax certain requirements based on an option (e.g.,
/// relaxing bitwidth requirement, see `convertScalarType()`,
/// `ConvertVectorType()`).
template <typename LabelT>
static LogicalResult checkCapabilityAndExtensionRequirements(
    LabelT label, const spirv::TargetEnv &targetEnv,
    const spirv::SPIRVType::CapabilityArrayRefVector &capCandidates,
    const spirv::SPIRVType::ExtensionArrayRefVector &extCandidates,
    ArrayRef<spirv::Capability> elidedCapCandidates = {},
    ArrayRef<spirv::Extension> elidedExtCandidates = {}) {
  SmallVector<ArrayRef<spirv::Extension>, 8> updatedExtCandidates;
  llvm::append_range(updatedExtCandidates, extCandidates);

  if (failed(checkCapabilityRequirements(label, targetEnv, capCandidates,
                                         elidedCapCandidates)))
    return failure();
  // Add capablity infered extensions to the list of extension requirement list,
  // only considers the capabilities that already available in the `targetEnv`.

  // WARNING: Some capabilities are part of both the core SPIR-V
  // specification and an extension (e.g., 'Groups' capability is part of both
  // core specification and SPV_AMD_shader_ballot extension, hence we should
  // relax the capability inferred extension for these cases).
  static const spirv::Capability multiModalCaps[] = {spirv::Capability::Groups};
  ArrayRef<spirv::Capability> multiModalCapsArrayRef(multiModalCaps,
                                                     std::size(multiModalCaps));

  for (auto cap : targetEnv.getAttr().getCapabilities()) {
    if (llvm::any_of(multiModalCapsArrayRef,
                     [&cap](spirv::Capability mMCap) { return cap == mMCap; }))
      continue;
    std::optional<ArrayRef<spirv::Extension>> ext = getExtensions(cap);
    if (ext)
      updatedExtCandidates.push_back(*ext);
  }
  if (failed(checkExtensionRequirements(label, targetEnv, updatedExtCandidates,
                                        elidedExtCandidates)))
    return failure();
  return success();
}

/// Returns true if the given `storageClass` needs explicit layout when used
/// in Shader environments.
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

static spirv::ScalarType getIndexType(MLIRContext *ctx,
                                      const SPIRVConversionOptions &options) {
  return cast<spirv::ScalarType>(
      IntegerType::get(ctx, options.use64bitIndex ? 64 : 32));
}

Type SPIRVTypeConverter::getIndexType() const {
  return ::getIndexType(getContext(), options);
}

MLIRContext *SPIRVTypeConverter::getContext() const {
  return targetEnv.getAttr().getContext();
}

bool SPIRVTypeConverter::allows(spirv::Capability capability) const {
  return targetEnv.allows(capability);
}

// TODO: This is a utility function that should probably be exposed by the
// SPIR-V dialect. Keeping it local till the use case arises.
static std::optional<int64_t>
getTypeNumBytes(const SPIRVConversionOptions &options, Type type) {
  if (isa<spirv::ScalarType>(type)) {
    auto bitWidth = type.getIntOrFloatBitWidth();
    // According to the SPIR-V spec:
    // "There is no physical size or bit pattern defined for values with boolean
    // type. If they are stored (in conjunction with OpVariable), they can only
    // be used with logical addressing operations, not physical, and only with
    // non-externally visible shader Storage Classes: Workgroup, CrossWorkgroup,
    // Private, Function, Input, and Output."
    if (bitWidth == 1)
      return std::nullopt;
    return bitWidth / 8;
  }

  if (auto complexType = dyn_cast<ComplexType>(type)) {
    auto elementSize = getTypeNumBytes(options, complexType.getElementType());
    if (!elementSize)
      return std::nullopt;
    return 2 * *elementSize;
  }

  if (auto vecType = dyn_cast<VectorType>(type)) {
    auto elementSize = getTypeNumBytes(options, vecType.getElementType());
    if (!elementSize)
      return std::nullopt;
    return vecType.getNumElements() * *elementSize;
  }

  if (auto memRefType = dyn_cast<MemRefType>(type)) {
    // TODO: Layout should also be controlled by the ABI attributes. For now
    // using the layout from MemRef.
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    if (!memRefType.hasStaticShape() ||
        failed(getStridesAndOffset(memRefType, strides, offset)))
      return std::nullopt;

    // To get the size of the memref object in memory, the total size is the
    // max(stride * dimension-size) computed for all dimensions times the size
    // of the element.
    auto elementSize = getTypeNumBytes(options, memRefType.getElementType());
    if (!elementSize)
      return std::nullopt;

    if (memRefType.getRank() == 0)
      return elementSize;

    auto dims = memRefType.getShape();
    if (llvm::is_contained(dims, ShapedType::kDynamic) ||
        ShapedType::isDynamic(offset) ||
        llvm::is_contained(strides, ShapedType::kDynamic))
      return std::nullopt;

    int64_t memrefSize = -1;
    for (const auto &shape : enumerate(dims))
      memrefSize = std::max(memrefSize, shape.value() * strides[shape.index()]);

    return (offset + memrefSize) * *elementSize;
  }

  if (auto tensorType = dyn_cast<TensorType>(type)) {
    if (!tensorType.hasStaticShape())
      return std::nullopt;

    auto elementSize = getTypeNumBytes(options, tensorType.getElementType());
    if (!elementSize)
      return std::nullopt;

    int64_t size = *elementSize;
    for (auto shape : tensorType.getShape())
      size *= shape;

    return size;
  }

  // TODO: Add size computation for other types.
  return std::nullopt;
}

/// Converts a scalar `type` to a suitable type under the given `targetEnv`.
static Type
convertScalarType(const spirv::TargetEnv &targetEnv,
                  const SPIRVConversionOptions &options, spirv::ScalarType type,
                  std::optional<spirv::StorageClass> storageClass = {}) {
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

  if (auto floatType = dyn_cast<FloatType>(type)) {
    LLVM_DEBUG(llvm::dbgs() << type << " converted to 32-bit for SPIR-V\n");
    return Builder(targetEnv.getContext()).getF32Type();
  }

  auto intType = cast<IntegerType>(type);
  LLVM_DEBUG(llvm::dbgs() << type << " converted to 32-bit for SPIR-V\n");
  return IntegerType::get(targetEnv.getContext(), /*width=*/32,
                          intType.getSignedness());
}

/// Converts a sub-byte integer `type` to i32 regardless of target environment.
///
/// Note that we don't recognize sub-byte types in `spirv::ScalarType` and use
/// the above given that these sub-byte types are not supported at all in
/// SPIR-V; there are no compute/storage capability for them like other
/// supported integer types.
static Type convertSubByteIntegerType(const SPIRVConversionOptions &options,
                                      IntegerType type) {
  if (options.subByteTypeStorage != SPIRVSubByteTypeStorage::Packed) {
    LLVM_DEBUG(llvm::dbgs() << "unsupported sub-byte storage kind\n");
    return nullptr;
  }

  if (!llvm::isPowerOf2_32(type.getWidth())) {
    LLVM_DEBUG(llvm::dbgs()
               << "unsupported non-power-of-two bitwidth in sub-byte" << type
               << "\n");
    return nullptr;
  }

  LLVM_DEBUG(llvm::dbgs() << type << " converted to 32-bit for SPIR-V\n");
  return IntegerType::get(type.getContext(), /*width=*/32,
                          type.getSignedness());
}

/// Returns a type with the same shape but with any index element type converted
/// to the matching integer type. This is a noop when the element type is not
/// the index type.
static ShapedType
convertIndexElementType(ShapedType type,
                        const SPIRVConversionOptions &options) {
  Type indexType = dyn_cast<IndexType>(type.getElementType());
  if (!indexType)
    return type;

  return type.clone(getIndexType(type.getContext(), options));
}

/// Converts a vector `type` to a suitable type under the given `targetEnv`.
static Type
convertVectorType(const spirv::TargetEnv &targetEnv,
                  const SPIRVConversionOptions &options, VectorType type,
                  std::optional<spirv::StorageClass> storageClass = {}) {
  type = cast<VectorType>(convertIndexElementType(type, options));
  auto scalarType = dyn_cast_or_null<spirv::ScalarType>(type.getElementType());
  if (!scalarType) {
    // If this is not a spec allowed scalar type, try to handle sub-byte integer
    // types.
    auto intType = dyn_cast<IntegerType>(type.getElementType());
    if (!intType) {
      LLVM_DEBUG(llvm::dbgs()
                 << type
                 << " illegal: cannot convert non-scalar element type\n");
      return nullptr;
    }

    Type elementType = convertSubByteIntegerType(options, intType);
    if (type.getRank() <= 1 && type.getNumElements() == 1)
      return elementType;

    if (type.getNumElements() > 4) {
      LLVM_DEBUG(llvm::dbgs()
                 << type << " illegal: > 4-element unimplemented\n");
      return nullptr;
    }

    return VectorType::get(type.getShape(), elementType);
  }

  if (type.getRank() <= 1 && type.getNumElements() == 1)
    return convertScalarType(targetEnv, options, scalarType, storageClass);

  if (!spirv::CompositeType::isValid(type)) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: not a valid composite type\n");
    return nullptr;
  }

  // Get extension and capability requirements for the given type.
  SmallVector<ArrayRef<spirv::Extension>, 1> extensions;
  SmallVector<ArrayRef<spirv::Capability>, 2> capabilities;
  cast<spirv::CompositeType>(type).getExtensions(extensions, storageClass);
  cast<spirv::CompositeType>(type).getCapabilities(capabilities, storageClass);

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

static Type
convertComplexType(const spirv::TargetEnv &targetEnv,
                   const SPIRVConversionOptions &options, ComplexType type,
                   std::optional<spirv::StorageClass> storageClass = {}) {
  auto scalarType = dyn_cast_or_null<spirv::ScalarType>(type.getElementType());
  if (!scalarType) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot convert non-scalar element type\n");
    return nullptr;
  }

  auto elementType =
      convertScalarType(targetEnv, options, scalarType, storageClass);
  if (!elementType)
    return nullptr;
  if (elementType != type.getElementType()) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: complex type emulation unsupported\n");
    return nullptr;
  }

  return VectorType::get(2, elementType);
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

  type = cast<TensorType>(convertIndexElementType(type, options));
  auto scalarType = dyn_cast_or_null<spirv::ScalarType>(type.getElementType());
  if (!scalarType) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot convert non-scalar element type\n");
    return nullptr;
  }

  std::optional<int64_t> scalarSize = getTypeNumBytes(options, scalarType);
  std::optional<int64_t> tensorSize = getTypeNumBytes(options, type);
  if (!scalarSize || !tensorSize) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot deduce element count\n");
    return nullptr;
  }

  int64_t arrayElemCount = *tensorSize / *scalarSize;
  if (arrayElemCount == 0) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot handle zero-element tensors\n");
    return nullptr;
  }

  Type arrayElemType = convertScalarType(targetEnv, options, scalarType);
  if (!arrayElemType)
    return nullptr;
  std::optional<int64_t> arrayElemSize =
      getTypeNumBytes(options, arrayElemType);
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
  auto elementType = dyn_cast<spirv::ScalarType>(
      IntegerType::get(type.getContext(), numBoolBits));
  if (!elementType)
    return nullptr;
  Type arrayElemType =
      convertScalarType(targetEnv, options, elementType, storageClass);
  if (!arrayElemType)
    return nullptr;
  std::optional<int64_t> arrayElemSize =
      getTypeNumBytes(options, arrayElemType);
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

  int64_t memrefSize = llvm::divideCeil(type.getNumElements() * numBoolBits, 8);
  int64_t arrayElemCount = llvm::divideCeil(memrefSize, *arrayElemSize);
  int64_t stride = needsExplicitLayout(storageClass) ? *arrayElemSize : 0;
  auto arrayType = spirv::ArrayType::get(arrayElemType, arrayElemCount, stride);
  if (targetEnv.allows(spirv::Capability::Kernel))
    return spirv::PointerType::get(arrayType, storageClass);
  return wrapInStructAndGetPointer(arrayType, storageClass);
}

static Type convertSubByteMemrefType(const spirv::TargetEnv &targetEnv,
                                     const SPIRVConversionOptions &options,
                                     MemRefType type,
                                     spirv::StorageClass storageClass) {
  IntegerType elementType = cast<IntegerType>(type.getElementType());
  Type arrayElemType = convertSubByteIntegerType(options, elementType);
  if (!arrayElemType)
    return nullptr;
  int64_t arrayElemSize = *getTypeNumBytes(options, arrayElemType);

  if (!type.hasStaticShape()) {
    // For OpenCL Kernel, dynamic shaped memrefs convert into a pointer pointing
    // to the element.
    if (targetEnv.allows(spirv::Capability::Kernel))
      return spirv::PointerType::get(arrayElemType, storageClass);
    int64_t stride = needsExplicitLayout(storageClass) ? arrayElemSize : 0;
    auto arrayType = spirv::RuntimeArrayType::get(arrayElemType, stride);
    // For Vulkan we need extra wrapping struct and array to satisfy interface
    // needs.
    return wrapInStructAndGetPointer(arrayType, storageClass);
  }

  int64_t memrefSize =
      llvm::divideCeil(type.getNumElements() * elementType.getWidth(), 8);
  int64_t arrayElemCount = llvm::divideCeil(memrefSize, arrayElemSize);
  int64_t stride = needsExplicitLayout(storageClass) ? arrayElemSize : 0;
  auto arrayType = spirv::ArrayType::get(arrayElemType, arrayElemCount, stride);
  if (targetEnv.allows(spirv::Capability::Kernel))
    return spirv::PointerType::get(arrayType, storageClass);
  return wrapInStructAndGetPointer(arrayType, storageClass);
}

static Type convertMemrefType(const spirv::TargetEnv &targetEnv,
                              const SPIRVConversionOptions &options,
                              MemRefType type) {
  auto attr = dyn_cast_or_null<spirv::StorageClassAttr>(type.getMemorySpace());
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

  if (isa<IntegerType>(type.getElementType())) {
    if (type.getElementTypeBitWidth() == 1)
      return convertBoolMemrefType(targetEnv, options, type, storageClass);
    if (type.getElementTypeBitWidth() < 8)
      return convertSubByteMemrefType(targetEnv, options, type, storageClass);
  }

  Type arrayElemType;
  Type elementType = type.getElementType();
  if (auto vecType = dyn_cast<VectorType>(elementType)) {
    arrayElemType =
        convertVectorType(targetEnv, options, vecType, storageClass);
  } else if (auto complexType = dyn_cast<ComplexType>(elementType)) {
    arrayElemType =
        convertComplexType(targetEnv, options, complexType, storageClass);
  } else if (auto scalarType = dyn_cast<spirv::ScalarType>(elementType)) {
    arrayElemType =
        convertScalarType(targetEnv, options, scalarType, storageClass);
  } else if (auto indexType = dyn_cast<IndexType>(elementType)) {
    type = cast<MemRefType>(convertIndexElementType(type, options));
    arrayElemType = type.getElementType();
  } else {
    LLVM_DEBUG(
        llvm::dbgs()
        << type
        << " unhandled: can only convert scalar or vector element type\n");
    return nullptr;
  }
  if (!arrayElemType)
    return nullptr;

  std::optional<int64_t> arrayElemSize =
      getTypeNumBytes(options, arrayElemType);
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

  std::optional<int64_t> memrefSize = getTypeNumBytes(options, type);
  if (!memrefSize) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: cannot deduce element count\n");
    return nullptr;
  }

  if (*memrefSize == 0) {
    LLVM_DEBUG(llvm::dbgs()
               << type << " illegal: zero-element memrefs are not supported\n");
    return nullptr;
  }

  int64_t arrayElemCount = llvm::divideCeil(*memrefSize, *arrayElemSize);
  int64_t stride = needsExplicitLayout(storageClass) ? *arrayElemSize : 0;
  auto arrayType = spirv::ArrayType::get(arrayElemType, arrayElemCount, stride);
  if (targetEnv.allows(spirv::Capability::Kernel))
    return spirv::PointerType::get(arrayType, storageClass);
  return wrapInStructAndGetPointer(arrayType, storageClass);
}

//===----------------------------------------------------------------------===//
// Type casting materialization
//===----------------------------------------------------------------------===//

/// Converts the given `inputs` to the original source `type` considering the
/// `targetEnv`'s capabilities.
///
/// This function is meant to be used for source materialization in type
/// converters. When the type converter needs to materialize a cast op back
/// to some original source type, we need to check whether the original source
/// type is supported in the target environment. If so, we can insert legal
/// SPIR-V cast ops accordingly.
///
/// Note that in SPIR-V the capabilities for storage and compute are separate.
/// This function is meant to handle the **compute** side; so it does not
/// involve storage classes in its logic. The storage side is expected to be
/// handled by MemRef conversion logic.
std::optional<Value> castToSourceType(const spirv::TargetEnv &targetEnv,
                                      OpBuilder &builder, Type type,
                                      ValueRange inputs, Location loc) {
  // We can only cast one value in SPIR-V.
  if (inputs.size() != 1) {
    auto castOp = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return castOp.getResult(0);
  }
  Value input = inputs.front();

  // Only support integer types for now. Floating point types to be implemented.
  if (!isa<IntegerType>(type)) {
    auto castOp = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return castOp.getResult(0);
  }
  auto inputType = cast<IntegerType>(input.getType());

  auto scalarType = dyn_cast<spirv::ScalarType>(type);
  if (!scalarType) {
    auto castOp = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return castOp.getResult(0);
  }

  // Only support source type with a smaller bitwidth. This would mean we are
  // truncating to go back so we don't need to worry about the signedness.
  // For extension, we cannot have enough signal here to decide which op to use.
  if (inputType.getIntOrFloatBitWidth() < scalarType.getIntOrFloatBitWidth()) {
    auto castOp = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return castOp.getResult(0);
  }

  // Boolean values would need to use different ops than normal integer values.
  if (type.isInteger(1)) {
    Value one = spirv::ConstantOp::getOne(inputType, loc, builder);
    return builder.create<spirv::IEqualOp>(loc, input, one);
  }

  // Check that the source integer type is supported by the environment.
  SmallVector<ArrayRef<spirv::Extension>, 1> exts;
  SmallVector<ArrayRef<spirv::Capability>, 2> caps;
  scalarType.getExtensions(exts);
  scalarType.getCapabilities(caps);
  if (failed(checkCapabilityRequirements(type, targetEnv, caps)) ||
      failed(checkExtensionRequirements(type, targetEnv, exts))) {
    auto castOp = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return castOp.getResult(0);
  }

  // We've already made sure this is truncating previously, so we don't need to
  // care about signedness here. Still try to use a corresponding op for better
  // consistency though.
  if (type.isSignedInteger()) {
    return builder.create<spirv::SConvertOp>(loc, type, input);
  }
  return builder.create<spirv::UConvertOp>(loc, type, input);
}

//===----------------------------------------------------------------------===//
// SPIRVTypeConverter
//===----------------------------------------------------------------------===//

SPIRVTypeConverter::SPIRVTypeConverter(spirv::TargetEnvAttr targetAttr,
                                       const SPIRVConversionOptions &options)
    : targetEnv(targetAttr), options(options) {
  // Add conversions. The order matters here: later ones will be tried earlier.

  // Allow all SPIR-V dialect specific types. This assumes all builtin types
  // adopted in the SPIR-V dialect (i.e., IntegerType, FloatType, VectorType)
  // were tried before.
  //
  // TODO: This assumes that the SPIR-V types are valid to use in the given
  // target environment, which should be the case if the whole pipeline is
  // driven by the same target environment. Still, we probably still want to
  // validate and convert to be safe.
  addConversion([](spirv::SPIRVType type) { return type; });

  addConversion([this](IndexType /*indexType*/) { return getIndexType(); });

  addConversion([this](IntegerType intType) -> std::optional<Type> {
    if (auto scalarType = dyn_cast<spirv::ScalarType>(intType))
      return convertScalarType(this->targetEnv, this->options, scalarType);
    if (intType.getWidth() < 8)
      return convertSubByteIntegerType(this->options, intType);
    return Type();
  });

  addConversion([this](FloatType floatType) -> std::optional<Type> {
    if (auto scalarType = dyn_cast<spirv::ScalarType>(floatType))
      return convertScalarType(this->targetEnv, this->options, scalarType);
    return Type();
  });

  addConversion([this](ComplexType complexType) {
    return convertComplexType(this->targetEnv, this->options, complexType);
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

  // Register some last line of defense casting logic.
  addSourceMaterialization(
      [this](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
        return castToSourceType(this->targetEnv, builder, type, inputs, loc);
      });
  addTargetMaterialization([](OpBuilder &builder, Type type, ValueRange inputs,
                              Location loc) {
    auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return std::optional<Value>(cast.getResult(0));
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
    if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
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
static std::string getBuiltinVarName(spirv::BuiltIn builtin, StringRef prefix,
                                     StringRef suffix) {
  return Twine(prefix).concat(stringifyBuiltIn(builtin)).concat(suffix).str();
}

/// Gets or inserts a global variable for a builtin within `body` block.
static spirv::GlobalVariableOp
getOrInsertBuiltinVariable(Block &body, Location loc, spirv::BuiltIn builtin,
                           Type integerType, OpBuilder &builder,
                           StringRef prefix, StringRef suffix) {
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
    std::string name = getBuiltinVarName(builtin, prefix, suffix);
    newVarOp =
        builder.create<spirv::GlobalVariableOp>(loc, ptrType, name, builtin);
    break;
  }
  case spirv::BuiltIn::SubgroupId:
  case spirv::BuiltIn::NumSubgroups:
  case spirv::BuiltIn::SubgroupSize: {
    auto ptrType =
        spirv::PointerType::get(integerType, spirv::StorageClass::Input);
    std::string name = getBuiltinVarName(builtin, prefix, suffix);
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
                                           Type integerType, OpBuilder &builder,
                                           StringRef prefix, StringRef suffix) {
  Operation *parent = SymbolTable::getNearestSymbolTable(op->getParentOp());
  if (!parent) {
    op->emitError("expected operation to be within a module-like op");
    return nullptr;
  }

  spirv::GlobalVariableOp varOp =
      getOrInsertBuiltinVariable(*parent->getRegion(0).begin(), op->getLoc(),
                                 builtin, integerType, builder, prefix, suffix);
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
    auto ptrType = dyn_cast<spirv::PointerType>(varOp.getType());
    if (!ptrType)
      continue;

    // Note that Vulkan requires "There must be no more than one push constant
    // block statically used per shader entry point." So we should always reuse
    // the existing one.
    if (ptrType.getStorageClass() == spirv::StorageClass::PushConstant) {
      auto numElements = cast<spirv::ArrayType>(
                             cast<spirv::StructType>(ptrType.getPointeeType())
                                 .getElementType(0))
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
      loc, addrOp, llvm::ArrayRef({zeroOp, offsetOp}));
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

Value mlir::spirv::getVulkanElementPtr(const SPIRVTypeConverter &typeConverter,
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

Value mlir::spirv::getOpenCLElementPtr(const SPIRVTypeConverter &typeConverter,
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
      cast<spirv::PointerType>(basePtr.getType()).getPointeeType();
  if (isa<spirv::ArrayType>(pointeeType)) {
    linearizedIndices.push_back(linearIndex);
    return builder.create<spirv::AccessChainOp>(loc, basePtr,
                                                linearizedIndices);
  }
  return builder.create<spirv::PtrAccessChainOp>(loc, basePtr, linearIndex,
                                                 linearizedIndices);
}

Value mlir::spirv::getElementPtr(const SPIRVTypeConverter &typeConverter,
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
    std::optional<spirv::Version> minVersion = minVersionIfx.getMinVersion();
    if (minVersion && *minVersion > this->targetEnv.getVersion()) {
      LLVM_DEBUG(llvm::dbgs()
                 << op->getName() << " illegal: requiring min version "
                 << spirv::stringifyVersion(*minVersion) << "\n");
      return false;
    }
  }
  if (auto maxVersionIfx = dyn_cast<spirv::QueryMaxVersionInterface>(op)) {
    std::optional<spirv::Version> maxVersion = maxVersionIfx.getMaxVersion();
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
                   [](Type t) { return !isa<spirv::SPIRVType>(t); }))
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
    cast<spirv::SPIRVType>(valueType).getExtensions(typeExtensions);
    if (failed(checkExtensionRequirements(op->getName(), this->targetEnv,
                                          typeExtensions)))
      return false;

    typeCapabilities.clear();
    cast<spirv::SPIRVType>(valueType).getCapabilities(typeCapabilities);
    if (failed(checkCapabilityRequirements(op->getName(), this->targetEnv,
                                           typeCapabilities)))
      return false;
  }

  return true;
}
