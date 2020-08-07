//===- SPIRVTypes.cpp - MLIR SPIR-V Types ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::spirv;

// Pull in all enum utility function definitions
#include "mlir/Dialect/SPIRV/SPIRVEnums.cpp.inc"
// Pull in all enum type availability query function definitions
#include "mlir/Dialect/SPIRV/SPIRVEnumAvailability.cpp.inc"

//===----------------------------------------------------------------------===//
// Availability relationship
//===----------------------------------------------------------------------===//

ArrayRef<Extension> spirv::getImpliedExtensions(Version version) {
  // Note: the following lists are from "Appendix A: Changes" of the spec.

#define V_1_3_IMPLIED_EXTS                                                     \
  Extension::SPV_KHR_shader_draw_parameters, Extension::SPV_KHR_16bit_storage, \
      Extension::SPV_KHR_device_group, Extension::SPV_KHR_multiview,           \
      Extension::SPV_KHR_storage_buffer_storage_class,                         \
      Extension::SPV_KHR_variable_pointers

#define V_1_4_IMPLIED_EXTS                                                     \
  Extension::SPV_KHR_no_integer_wrap_decoration,                               \
      Extension::SPV_GOOGLE_decorate_string,                                   \
      Extension::SPV_GOOGLE_hlsl_functionality1,                               \
      Extension::SPV_KHR_float_controls

#define V_1_5_IMPLIED_EXTS                                                     \
  Extension::SPV_KHR_8bit_storage, Extension::SPV_EXT_descriptor_indexing,     \
      Extension::SPV_EXT_shader_viewport_index_layer,                          \
      Extension::SPV_EXT_physical_storage_buffer,                              \
      Extension::SPV_KHR_physical_storage_buffer,                              \
      Extension::SPV_KHR_vulkan_memory_model

  switch (version) {
  default:
    return {};
  case Version::V_1_3: {
    // The following manual ArrayRef constructor call is to satisfy GCC 5.
    static const Extension exts[] = {V_1_3_IMPLIED_EXTS};
    return ArrayRef<Extension>(exts, llvm::array_lengthof(exts));
  }
  case Version::V_1_4: {
    static const Extension exts[] = {V_1_3_IMPLIED_EXTS, V_1_4_IMPLIED_EXTS};
    return ArrayRef<Extension>(exts, llvm::array_lengthof(exts));
  }
  case Version::V_1_5: {
    static const Extension exts[] = {V_1_3_IMPLIED_EXTS, V_1_4_IMPLIED_EXTS,
                                     V_1_5_IMPLIED_EXTS};
    return ArrayRef<Extension>(exts, llvm::array_lengthof(exts));
  }
  }

#undef V_1_5_IMPLIED_EXTS
#undef V_1_4_IMPLIED_EXTS
#undef V_1_3_IMPLIED_EXTS
}

// Pull in utility function definition for implied capabilities
#include "mlir/Dialect/SPIRV/SPIRVCapabilityImplication.inc"

SmallVector<Capability, 0>
spirv::getRecursiveImpliedCapabilities(Capability cap) {
  ArrayRef<Capability> directCaps = getDirectImpliedCapabilities(cap);
  llvm::SetVector<Capability, SmallVector<Capability, 0>> allCaps(
      directCaps.begin(), directCaps.end());

  // TODO: This is insufficient; find a better way to handle this
  // (e.g., using static lists) if this turns out to be a bottleneck.
  for (unsigned i = 0; i < allCaps.size(); ++i)
    for (Capability c : getDirectImpliedCapabilities(allCaps[i]))
      allCaps.insert(c);

  return allCaps.takeVector();
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

struct spirv::detail::ArrayTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<Type, unsigned, unsigned>;

  static ArrayTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<ArrayTypeStorage>()) ArrayTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, elementCount, stride);
  }

  ArrayTypeStorage(const KeyTy &key)
      : elementType(std::get<0>(key)), elementCount(std::get<1>(key)),
        stride(std::get<2>(key)) {}

  Type elementType;
  unsigned elementCount;
  unsigned stride;
};

ArrayType ArrayType::get(Type elementType, unsigned elementCount) {
  assert(elementCount && "ArrayType needs at least one element");
  return Base::get(elementType.getContext(), TypeKind::Array, elementType,
                   elementCount, /*stride=*/0);
}

ArrayType ArrayType::get(Type elementType, unsigned elementCount,
                         unsigned stride) {
  assert(elementCount && "ArrayType needs at least one element");
  return Base::get(elementType.getContext(), TypeKind::Array, elementType,
                   elementCount, stride);
}

unsigned ArrayType::getNumElements() const { return getImpl()->elementCount; }

Type ArrayType::getElementType() const { return getImpl()->elementType; }

unsigned ArrayType::getArrayStride() const { return getImpl()->stride; }

void ArrayType::getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                              Optional<StorageClass> storage) {
  getElementType().cast<SPIRVType>().getExtensions(extensions, storage);
}

void ArrayType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    Optional<StorageClass> storage) {
  getElementType().cast<SPIRVType>().getCapabilities(capabilities, storage);
}

Optional<int64_t> ArrayType::getSizeInBytes() {
  auto elementType = getElementType().cast<SPIRVType>();
  Optional<int64_t> size = elementType.getSizeInBytes();
  if (!size)
    return llvm::None;
  return (*size + getArrayStride()) * getNumElements();
}

//===----------------------------------------------------------------------===//
// CompositeType
//===----------------------------------------------------------------------===//

bool CompositeType::classof(Type type) {
  if (auto vectorType = type.dyn_cast<VectorType>())
    return isValid(vectorType);
  return type
      .isa<spirv::ArrayType, spirv::CooperativeMatrixNVType, spirv::MatrixType,
           spirv::RuntimeArrayType, spirv::StructType>();
}

bool CompositeType::isValid(VectorType type) {
  return type.getRank() == 1 && type.getElementType().isa<ScalarType>() &&
         type.getNumElements() >= 2 && type.getNumElements() <= 4;
}

Type CompositeType::getElementType(unsigned index) const {
  return TypeSwitch<Type, Type>(*this)
      .Case<ArrayType, CooperativeMatrixNVType, RuntimeArrayType, VectorType>(
          [](auto type) { return type.getElementType(); })
      .Case<MatrixType>([](MatrixType type) { return type.getColumnType(); })
      .Case<StructType>(
          [index](StructType type) { return type.getElementType(index); })
      .Default(
          [](Type) -> Type { llvm_unreachable("invalid composite type"); });
}

unsigned CompositeType::getNumElements() const {
  switch (getKind()) {
  case spirv::TypeKind::Array:
    return cast<ArrayType>().getNumElements();
  case spirv::TypeKind::CooperativeMatrix:
    llvm_unreachable(
        "invalid to query number of elements of spirv::CooperativeMatrix type");
  case spirv::TypeKind::Matrix:
    return cast<MatrixType>().getNumColumns();
  case spirv::TypeKind::RuntimeArray:
    llvm_unreachable(
        "invalid to query number of elements of spirv::RuntimeArray type");
  case spirv::TypeKind::Struct:
    return cast<StructType>().getNumElements();
  case StandardTypes::Vector:
    return cast<VectorType>().getNumElements();
  default:
    llvm_unreachable("invalid composite type");
  }
}

bool CompositeType::hasCompileTimeKnownNumElements() const {
  switch (getKind()) {
  case TypeKind::CooperativeMatrix:
  case TypeKind::RuntimeArray:
    return false;
  default:
    return true;
  }
}

void CompositeType::getExtensions(
    SPIRVType::ExtensionArrayRefVector &extensions,
    Optional<StorageClass> storage) {
  switch (getKind()) {
  case spirv::TypeKind::Array:
    cast<ArrayType>().getExtensions(extensions, storage);
    break;
  case spirv::TypeKind::CooperativeMatrix:
    cast<CooperativeMatrixNVType>().getExtensions(extensions, storage);
    break;
  case spirv::TypeKind::Matrix:
    cast<MatrixType>().getExtensions(extensions, storage);
    break;
  case spirv::TypeKind::RuntimeArray:
    cast<RuntimeArrayType>().getExtensions(extensions, storage);
    break;
  case spirv::TypeKind::Struct:
    cast<StructType>().getExtensions(extensions, storage);
    break;
  case StandardTypes::Vector:
    cast<VectorType>().getElementType().cast<ScalarType>().getExtensions(
        extensions, storage);
    break;
  default:
    llvm_unreachable("invalid composite type");
  }
}

void CompositeType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    Optional<StorageClass> storage) {
  switch (getKind()) {
  case spirv::TypeKind::Array:
    cast<ArrayType>().getCapabilities(capabilities, storage);
    break;
  case spirv::TypeKind::CooperativeMatrix:
    cast<CooperativeMatrixNVType>().getCapabilities(capabilities, storage);
    break;
  case spirv::TypeKind::Matrix:
    cast<MatrixType>().getCapabilities(capabilities, storage);
    break;
  case spirv::TypeKind::RuntimeArray:
    cast<RuntimeArrayType>().getCapabilities(capabilities, storage);
    break;
  case spirv::TypeKind::Struct:
    cast<StructType>().getCapabilities(capabilities, storage);
    break;
  case StandardTypes::Vector:
    cast<VectorType>().getElementType().cast<ScalarType>().getCapabilities(
        capabilities, storage);
    break;
  default:
    llvm_unreachable("invalid composite type");
  }
}

Optional<int64_t> CompositeType::getSizeInBytes() {
  switch (getKind()) {
  case spirv::TypeKind::Array:
    return cast<ArrayType>().getSizeInBytes();
  case spirv::TypeKind::Struct:
    return cast<StructType>().getSizeInBytes();
  case StandardTypes::Vector: {
    auto elementSize =
        cast<VectorType>().getElementType().cast<ScalarType>().getSizeInBytes();
    if (!elementSize)
      return llvm::None;
    return *elementSize * cast<VectorType>().getNumElements();
  }
  default:
    return llvm::None;
  }
}

//===----------------------------------------------------------------------===//
// CooperativeMatrixType
//===----------------------------------------------------------------------===//

struct spirv::detail::CooperativeMatrixTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<Type, Scope, unsigned, unsigned>;

  static CooperativeMatrixTypeStorage *
  construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<CooperativeMatrixTypeStorage>())
        CooperativeMatrixTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, scope, rows, columns);
  }

  CooperativeMatrixTypeStorage(const KeyTy &key)
      : elementType(std::get<0>(key)), rows(std::get<2>(key)),
        columns(std::get<3>(key)), scope(std::get<1>(key)) {}

  Type elementType;
  unsigned rows;
  unsigned columns;
  Scope scope;
};

CooperativeMatrixNVType CooperativeMatrixNVType::get(Type elementType,
                                                     Scope scope, unsigned rows,
                                                     unsigned columns) {
  return Base::get(elementType.getContext(), TypeKind::CooperativeMatrix,
                   elementType, scope, rows, columns);
}

Type CooperativeMatrixNVType::getElementType() const {
  return getImpl()->elementType;
}

Scope CooperativeMatrixNVType::getScope() const { return getImpl()->scope; }

unsigned CooperativeMatrixNVType::getRows() const { return getImpl()->rows; }

unsigned CooperativeMatrixNVType::getColumns() const {
  return getImpl()->columns;
}

void CooperativeMatrixNVType::getExtensions(
    SPIRVType::ExtensionArrayRefVector &extensions,
    Optional<StorageClass> storage) {
  getElementType().cast<SPIRVType>().getExtensions(extensions, storage);
  static const Extension exts[] = {Extension::SPV_NV_cooperative_matrix};
  ArrayRef<Extension> ref(exts, llvm::array_lengthof(exts));
  extensions.push_back(ref);
}

void CooperativeMatrixNVType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    Optional<StorageClass> storage) {
  getElementType().cast<SPIRVType>().getCapabilities(capabilities, storage);
  static const Capability caps[] = {Capability::CooperativeMatrixNV};
  ArrayRef<Capability> ref(caps, llvm::array_lengthof(caps));
  capabilities.push_back(ref);
}

//===----------------------------------------------------------------------===//
// ImageType
//===----------------------------------------------------------------------===//

template <typename T> static constexpr unsigned getNumBits() { return 0; }
template <> constexpr unsigned getNumBits<Dim>() {
  static_assert((1 << 3) > getMaxEnumValForDim(),
                "Not enough bits to encode Dim value");
  return 3;
}
template <> constexpr unsigned getNumBits<ImageDepthInfo>() {
  static_assert((1 << 2) > getMaxEnumValForImageDepthInfo(),
                "Not enough bits to encode ImageDepthInfo value");
  return 2;
}
template <> constexpr unsigned getNumBits<ImageArrayedInfo>() {
  static_assert((1 << 1) > getMaxEnumValForImageArrayedInfo(),
                "Not enough bits to encode ImageArrayedInfo value");
  return 1;
}
template <> constexpr unsigned getNumBits<ImageSamplingInfo>() {
  static_assert((1 << 1) > getMaxEnumValForImageSamplingInfo(),
                "Not enough bits to encode ImageSamplingInfo value");
  return 1;
}
template <> constexpr unsigned getNumBits<ImageSamplerUseInfo>() {
  static_assert((1 << 2) > getMaxEnumValForImageSamplerUseInfo(),
                "Not enough bits to encode ImageSamplerUseInfo value");
  return 2;
}
template <> constexpr unsigned getNumBits<ImageFormat>() {
  static_assert((1 << 6) > getMaxEnumValForImageFormat(),
                "Not enough bits to encode ImageFormat value");
  return 6;
}

struct spirv::detail::ImageTypeStorage : public TypeStorage {
public:
  using KeyTy = std::tuple<Type, Dim, ImageDepthInfo, ImageArrayedInfo,
                           ImageSamplingInfo, ImageSamplerUseInfo, ImageFormat>;

  static ImageTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<ImageTypeStorage>()) ImageTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, dim, depthInfo, arrayedInfo, samplingInfo,
                        samplerUseInfo, format);
  }

  ImageTypeStorage(const KeyTy &key)
      : elementType(std::get<0>(key)), dim(std::get<1>(key)),
        depthInfo(std::get<2>(key)), arrayedInfo(std::get<3>(key)),
        samplingInfo(std::get<4>(key)), samplerUseInfo(std::get<5>(key)),
        format(std::get<6>(key)) {}

  Type elementType;
  Dim dim : getNumBits<Dim>();
  ImageDepthInfo depthInfo : getNumBits<ImageDepthInfo>();
  ImageArrayedInfo arrayedInfo : getNumBits<ImageArrayedInfo>();
  ImageSamplingInfo samplingInfo : getNumBits<ImageSamplingInfo>();
  ImageSamplerUseInfo samplerUseInfo : getNumBits<ImageSamplerUseInfo>();
  ImageFormat format : getNumBits<ImageFormat>();
};

ImageType
ImageType::get(std::tuple<Type, Dim, ImageDepthInfo, ImageArrayedInfo,
                          ImageSamplingInfo, ImageSamplerUseInfo, ImageFormat>
                   value) {
  return Base::get(std::get<0>(value).getContext(), TypeKind::Image, value);
}

Type ImageType::getElementType() const { return getImpl()->elementType; }

Dim ImageType::getDim() const { return getImpl()->dim; }

ImageDepthInfo ImageType::getDepthInfo() const { return getImpl()->depthInfo; }

ImageArrayedInfo ImageType::getArrayedInfo() const {
  return getImpl()->arrayedInfo;
}

ImageSamplingInfo ImageType::getSamplingInfo() const {
  return getImpl()->samplingInfo;
}

ImageSamplerUseInfo ImageType::getSamplerUseInfo() const {
  return getImpl()->samplerUseInfo;
}

ImageFormat ImageType::getImageFormat() const { return getImpl()->format; }

void ImageType::getExtensions(SPIRVType::ExtensionArrayRefVector &,
                              Optional<StorageClass>) {
  // Image types do not require extra extensions thus far.
}

void ImageType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities, Optional<StorageClass>) {
  if (auto dimCaps = spirv::getCapabilities(getDim()))
    capabilities.push_back(*dimCaps);

  if (auto fmtCaps = spirv::getCapabilities(getImageFormat()))
    capabilities.push_back(*fmtCaps);
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

struct spirv::detail::PointerTypeStorage : public TypeStorage {
  // (Type, StorageClass) as the key: Type stored in this struct, and
  // StorageClass stored as TypeStorage's subclass data.
  using KeyTy = std::pair<Type, StorageClass>;

  static PointerTypeStorage *construct(TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<PointerTypeStorage>())
        PointerTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(pointeeType, storageClass);
  }

  PointerTypeStorage(const KeyTy &key)
      : pointeeType(key.first), storageClass(key.second) {}

  Type pointeeType;
  StorageClass storageClass;
};

PointerType PointerType::get(Type pointeeType, StorageClass storageClass) {
  return Base::get(pointeeType.getContext(), TypeKind::Pointer, pointeeType,
                   storageClass);
}

Type PointerType::getPointeeType() const { return getImpl()->pointeeType; }

StorageClass PointerType::getStorageClass() const {
  return getImpl()->storageClass;
}

void PointerType::getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                                Optional<StorageClass> storage) {
  // Use this pointer type's storage class because this pointer indicates we are
  // using the pointee type in that specific storage class.
  getPointeeType().cast<SPIRVType>().getExtensions(extensions,
                                                   getStorageClass());

  if (auto scExts = spirv::getExtensions(getStorageClass()))
    extensions.push_back(*scExts);
}

void PointerType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    Optional<StorageClass> storage) {
  // Use this pointer type's storage class because this pointer indicates we are
  // using the pointee type in that specific storage class.
  getPointeeType().cast<SPIRVType>().getCapabilities(capabilities,
                                                     getStorageClass());

  if (auto scCaps = spirv::getCapabilities(getStorageClass()))
    capabilities.push_back(*scCaps);
}

//===----------------------------------------------------------------------===//
// RuntimeArrayType
//===----------------------------------------------------------------------===//

struct spirv::detail::RuntimeArrayTypeStorage : public TypeStorage {
  using KeyTy = std::pair<Type, unsigned>;

  static RuntimeArrayTypeStorage *construct(TypeStorageAllocator &allocator,
                                            const KeyTy &key) {
    return new (allocator.allocate<RuntimeArrayTypeStorage>())
        RuntimeArrayTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, stride);
  }

  RuntimeArrayTypeStorage(const KeyTy &key)
      : elementType(key.first), stride(key.second) {}

  Type elementType;
  unsigned stride;
};

RuntimeArrayType RuntimeArrayType::get(Type elementType) {
  return Base::get(elementType.getContext(), TypeKind::RuntimeArray,
                   elementType, /*stride=*/0);
}

RuntimeArrayType RuntimeArrayType::get(Type elementType, unsigned stride) {
  return Base::get(elementType.getContext(), TypeKind::RuntimeArray,
                   elementType, stride);
}

Type RuntimeArrayType::getElementType() const { return getImpl()->elementType; }

unsigned RuntimeArrayType::getArrayStride() const { return getImpl()->stride; }

void RuntimeArrayType::getExtensions(
    SPIRVType::ExtensionArrayRefVector &extensions,
    Optional<StorageClass> storage) {
  getElementType().cast<SPIRVType>().getExtensions(extensions, storage);
}

void RuntimeArrayType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    Optional<StorageClass> storage) {
  {
    static const Capability caps[] = {Capability::Shader};
    ArrayRef<Capability> ref(caps, llvm::array_lengthof(caps));
    capabilities.push_back(ref);
  }
  getElementType().cast<SPIRVType>().getCapabilities(capabilities, storage);
}

//===----------------------------------------------------------------------===//
// ScalarType
//===----------------------------------------------------------------------===//

bool ScalarType::classof(Type type) {
  if (auto floatType = type.dyn_cast<FloatType>()) {
    return isValid(floatType);
  }
  if (auto intType = type.dyn_cast<IntegerType>()) {
    return isValid(intType);
  }
  return false;
}

bool ScalarType::isValid(FloatType type) { return !type.isBF16(); }

bool ScalarType::isValid(IntegerType type) {
  switch (type.getWidth()) {
  case 1:
  case 8:
  case 16:
  case 32:
  case 64:
    return true;
  default:
    return false;
  }
}

void ScalarType::getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                               Optional<StorageClass> storage) {
  // 8- or 16-bit integer/floating-point numbers will require extra extensions
  // to appear in interface storage classes. See SPV_KHR_16bit_storage and
  // SPV_KHR_8bit_storage for more details.
  if (!storage)
    return;

  switch (*storage) {
  case StorageClass::PushConstant:
  case StorageClass::StorageBuffer:
  case StorageClass::Uniform:
    if (getIntOrFloatBitWidth() == 8) {
      static const Extension exts[] = {Extension::SPV_KHR_8bit_storage};
      ArrayRef<Extension> ref(exts, llvm::array_lengthof(exts));
      extensions.push_back(ref);
    }
    LLVM_FALLTHROUGH;
  case StorageClass::Input:
  case StorageClass::Output:
    if (getIntOrFloatBitWidth() == 16) {
      static const Extension exts[] = {Extension::SPV_KHR_16bit_storage};
      ArrayRef<Extension> ref(exts, llvm::array_lengthof(exts));
      extensions.push_back(ref);
    }
    break;
  default:
    break;
  }
}

void ScalarType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    Optional<StorageClass> storage) {
  unsigned bitwidth = getIntOrFloatBitWidth();

  // 8- or 16-bit integer/floating-point numbers will require extra capabilities
  // to appear in interface storage classes. See SPV_KHR_16bit_storage and
  // SPV_KHR_8bit_storage for more details.

#define STORAGE_CASE(storage, cap8, cap16)                                     \
  case StorageClass::storage: {                                                \
    if (bitwidth == 8) {                                                       \
      static const Capability caps[] = {Capability::cap8};                     \
      ArrayRef<Capability> ref(caps, llvm::array_lengthof(caps));              \
      capabilities.push_back(ref);                                             \
    } else if (bitwidth == 16) {                                               \
      static const Capability caps[] = {Capability::cap16};                    \
      ArrayRef<Capability> ref(caps, llvm::array_lengthof(caps));              \
      capabilities.push_back(ref);                                             \
    }                                                                          \
    /* No requirements for other bitwidths */                                  \
    return;                                                                    \
  }

  // This part only handles the cases where special bitwidths appearing in
  // interface storage classes.
  if (storage) {
    switch (*storage) {
      STORAGE_CASE(PushConstant, StoragePushConstant8, StoragePushConstant16);
      STORAGE_CASE(StorageBuffer, StorageBuffer8BitAccess,
                   StorageBuffer16BitAccess);
      STORAGE_CASE(Uniform, UniformAndStorageBuffer8BitAccess,
                   StorageUniform16);
    case StorageClass::Input:
    case StorageClass::Output: {
      if (bitwidth == 16) {
        static const Capability caps[] = {Capability::StorageInputOutput16};
        ArrayRef<Capability> ref(caps, llvm::array_lengthof(caps));
        capabilities.push_back(ref);
      }
      return;
    }
    default:
      break;
    }
  }
#undef STORAGE_CASE

  // For other non-interface storage classes, require a different set of
  // capabilities for special bitwidths.

#define WIDTH_CASE(type, width)                                                \
  case width: {                                                                \
    static const Capability caps[] = {Capability::type##width};                \
    ArrayRef<Capability> ref(caps, llvm::array_lengthof(caps));                \
    capabilities.push_back(ref);                                               \
  } break

  if (auto intType = dyn_cast<IntegerType>()) {
    switch (bitwidth) {
    case 32:
    case 1:
      break;
      WIDTH_CASE(Int, 8);
      WIDTH_CASE(Int, 16);
      WIDTH_CASE(Int, 64);
    default:
      llvm_unreachable("invalid bitwidth to getCapabilities");
    }
  } else {
    assert(isa<FloatType>());
    switch (bitwidth) {
    case 32:
      break;
      WIDTH_CASE(Float, 16);
      WIDTH_CASE(Float, 64);
    default:
      llvm_unreachable("invalid bitwidth to getCapabilities");
    }
  }

#undef WIDTH_CASE
}

Optional<int64_t> ScalarType::getSizeInBytes() {
  auto bitWidth = getIntOrFloatBitWidth();
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

//===----------------------------------------------------------------------===//
// SPIRVType
//===----------------------------------------------------------------------===//

bool SPIRVType::classof(Type type) {
  // Allow SPIR-V dialect types
  if (type.getKind() >= Type::FIRST_SPIRV_TYPE &&
      type.getKind() <= TypeKind::LAST_SPIRV_TYPE)
    return true;
  if (type.isa<ScalarType>())
    return true;
  if (auto vectorType = type.dyn_cast<VectorType>())
    return CompositeType::isValid(vectorType);
  return false;
}

bool SPIRVType::isScalarOrVector() {
  return isIntOrFloat() || isa<VectorType>();
}

void SPIRVType::getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                              Optional<StorageClass> storage) {
  if (auto scalarType = dyn_cast<ScalarType>()) {
    scalarType.getExtensions(extensions, storage);
  } else if (auto compositeType = dyn_cast<CompositeType>()) {
    compositeType.getExtensions(extensions, storage);
  } else if (auto imageType = dyn_cast<ImageType>()) {
    imageType.getExtensions(extensions, storage);
  } else if (auto matrixType = dyn_cast<MatrixType>()) {
    matrixType.getExtensions(extensions, storage);
  } else if (auto ptrType = dyn_cast<PointerType>()) {
    ptrType.getExtensions(extensions, storage);
  } else {
    llvm_unreachable("invalid SPIR-V Type to getExtensions");
  }
}

void SPIRVType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    Optional<StorageClass> storage) {
  if (auto scalarType = dyn_cast<ScalarType>()) {
    scalarType.getCapabilities(capabilities, storage);
  } else if (auto compositeType = dyn_cast<CompositeType>()) {
    compositeType.getCapabilities(capabilities, storage);
  } else if (auto imageType = dyn_cast<ImageType>()) {
    imageType.getCapabilities(capabilities, storage);
  } else if (auto matrixType = dyn_cast<MatrixType>()) {
    matrixType.getCapabilities(capabilities, storage);
  } else if (auto ptrType = dyn_cast<PointerType>()) {
    ptrType.getCapabilities(capabilities, storage);
  } else {
    llvm_unreachable("invalid SPIR-V Type to getCapabilities");
  }
}

Optional<int64_t> SPIRVType::getSizeInBytes() {
  if (auto scalarType = dyn_cast<ScalarType>())
    return scalarType.getSizeInBytes();
  if (auto compositeType = dyn_cast<CompositeType>())
    return compositeType.getSizeInBytes();
  return llvm::None;
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

struct spirv::detail::StructTypeStorage : public TypeStorage {
  StructTypeStorage(
      unsigned numMembers, Type const *memberTypes,
      StructType::OffsetInfo const *layoutInfo, unsigned numMemberDecorations,
      StructType::MemberDecorationInfo const *memberDecorationsInfo)
      : memberTypes(memberTypes), offsetInfo(layoutInfo),
        numMembers(numMembers), numMemberDecorations(numMemberDecorations),
        memberDecorationsInfo(memberDecorationsInfo) {}

  using KeyTy = std::tuple<ArrayRef<Type>, ArrayRef<StructType::OffsetInfo>,
                           ArrayRef<StructType::MemberDecorationInfo>>;
  bool operator==(const KeyTy &key) const {
    return key ==
           KeyTy(getMemberTypes(), getOffsetInfo(), getMemberDecorationsInfo());
  }

  static StructTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    ArrayRef<Type> keyTypes = std::get<0>(key);

    // Copy the member type and layout information into the bump pointer
    const Type *typesList = nullptr;
    if (!keyTypes.empty()) {
      typesList = allocator.copyInto(keyTypes).data();
    }

    const StructType::OffsetInfo *offsetInfoList = nullptr;
    if (!std::get<1>(key).empty()) {
      ArrayRef<StructType::OffsetInfo> keyOffsetInfo = std::get<1>(key);
      assert(keyOffsetInfo.size() == keyTypes.size() &&
             "size of offset information must be same as the size of number of "
             "elements");
      offsetInfoList = allocator.copyInto(keyOffsetInfo).data();
    }

    const StructType::MemberDecorationInfo *memberDecorationList = nullptr;
    unsigned numMemberDecorations = 0;
    if (!std::get<2>(key).empty()) {
      auto keyMemberDecorations = std::get<2>(key);
      numMemberDecorations = keyMemberDecorations.size();
      memberDecorationList = allocator.copyInto(keyMemberDecorations).data();
    }
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(keyTypes.size(), typesList, offsetInfoList,
                          numMemberDecorations, memberDecorationList);
  }

  ArrayRef<Type> getMemberTypes() const {
    return ArrayRef<Type>(memberTypes, numMembers);
  }

  ArrayRef<StructType::OffsetInfo> getOffsetInfo() const {
    if (offsetInfo) {
      return ArrayRef<StructType::OffsetInfo>(offsetInfo, numMembers);
    }
    return {};
  }

  ArrayRef<StructType::MemberDecorationInfo> getMemberDecorationsInfo() const {
    if (memberDecorationsInfo) {
      return ArrayRef<StructType::MemberDecorationInfo>(memberDecorationsInfo,
                                                        numMemberDecorations);
    }
    return {};
  }

  Type const *memberTypes;
  StructType::OffsetInfo const *offsetInfo;
  unsigned numMembers;
  unsigned numMemberDecorations;
  StructType::MemberDecorationInfo const *memberDecorationsInfo;
};

StructType
StructType::get(ArrayRef<Type> memberTypes,
                ArrayRef<StructType::OffsetInfo> offsetInfo,
                ArrayRef<StructType::MemberDecorationInfo> memberDecorations) {
  assert(!memberTypes.empty() && "Struct needs at least one member type");
  // Sort the decorations.
  SmallVector<StructType::MemberDecorationInfo, 4> sortedDecorations(
      memberDecorations.begin(), memberDecorations.end());
  llvm::array_pod_sort(sortedDecorations.begin(), sortedDecorations.end());
  return Base::get(memberTypes.vec().front().getContext(), TypeKind::Struct,
                   memberTypes, offsetInfo, sortedDecorations);
}

StructType StructType::getEmpty(MLIRContext *context) {
  return Base::get(context, TypeKind::Struct, ArrayRef<Type>(),
                   ArrayRef<StructType::OffsetInfo>(),
                   ArrayRef<StructType::MemberDecorationInfo>());
}

unsigned StructType::getNumElements() const { return getImpl()->numMembers; }

Type StructType::getElementType(unsigned index) const {
  assert(getNumElements() > index && "member index out of range");
  return getImpl()->memberTypes[index];
}

StructType::ElementTypeRange StructType::getElementTypes() const {
  return ElementTypeRange(getImpl()->memberTypes, getNumElements());
}

bool StructType::hasOffset() const { return getImpl()->offsetInfo; }

uint64_t StructType::getMemberOffset(unsigned index) const {
  assert(getNumElements() > index && "member index out of range");
  return getImpl()->offsetInfo[index];
}

void StructType::getMemberDecorations(
    SmallVectorImpl<StructType::MemberDecorationInfo> &memberDecorations)
    const {
  memberDecorations.clear();
  auto implMemberDecorations = getImpl()->getMemberDecorationsInfo();
  memberDecorations.append(implMemberDecorations.begin(),
                           implMemberDecorations.end());
}

void StructType::getMemberDecorations(
    unsigned index,
    SmallVectorImpl<StructType::MemberDecorationInfo> &decorationsInfo) const {
  assert(getNumElements() > index && "member index out of range");
  auto memberDecorations = getImpl()->getMemberDecorationsInfo();
  decorationsInfo.clear();
  for (const auto &memberDecoration : memberDecorations) {
    if (memberDecoration.memberIndex == index) {
      decorationsInfo.push_back(memberDecoration);
    }
    if (memberDecoration.memberIndex > index) {
      // Early exit since the decorations are stored sorted.
      return;
    }
  }
}

void StructType::getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                               Optional<StorageClass> storage) {
  for (Type elementType : getElementTypes())
    elementType.cast<SPIRVType>().getExtensions(extensions, storage);
}

void StructType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    Optional<StorageClass> storage) {
  for (Type elementType : getElementTypes())
    elementType.cast<SPIRVType>().getCapabilities(capabilities, storage);
}

llvm::hash_code spirv::hash_value(
    const StructType::MemberDecorationInfo &memberDecorationInfo) {
  return llvm::hash_combine(memberDecorationInfo.memberIndex,
                            memberDecorationInfo.decoration);
}

//===----------------------------------------------------------------------===//
// MatrixType
//===----------------------------------------------------------------------===//

struct spirv::detail::MatrixTypeStorage : public TypeStorage {
  MatrixTypeStorage(Type columnType, uint32_t columnCount)
      : TypeStorage(), columnType(columnType), columnCount(columnCount) {}

  using KeyTy = std::tuple<Type, uint32_t>;

  static MatrixTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {

    // Initialize the memory using placement new.
    return new (allocator.allocate<MatrixTypeStorage>())
        MatrixTypeStorage(std::get<0>(key), std::get<1>(key));
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(columnType, columnCount);
  }

  Type columnType;
  const uint32_t columnCount;
};

MatrixType MatrixType::get(Type columnType, uint32_t columnCount) {
  return Base::get(columnType.getContext(), TypeKind::Matrix, columnType,
                   columnCount);
}

MatrixType MatrixType::getChecked(Type columnType, uint32_t columnCount,
                                  Location location) {
  return Base::getChecked(location, TypeKind::Matrix, columnType, columnCount);
}

LogicalResult MatrixType::verifyConstructionInvariants(Location loc,
                                                       Type columnType,
                                                       uint32_t columnCount) {
  if (columnCount < 2 || columnCount > 4)
    return emitError(loc, "matrix can have 2, 3, or 4 columns only");

  if (!isValidColumnType(columnType))
    return emitError(loc, "matrix columns must be vectors of floats");

  /// The underlying vectors (columns) must be of size 2, 3, or 4
  ArrayRef<int64_t> columnShape = columnType.cast<VectorType>().getShape();
  if (columnShape.size() != 1)
    return emitError(loc, "matrix columns must be 1D vectors");

  if (columnShape[0] < 2 || columnShape[0] > 4)
    return emitError(loc, "matrix columns must be of size 2, 3, or 4");

  return success();
}

/// Returns true if the matrix elements are vectors of float elements
bool MatrixType::isValidColumnType(Type columnType) {
  if (auto vectorType = columnType.dyn_cast<VectorType>()) {
    if (vectorType.getElementType().isa<FloatType>())
      return true;
  }
  return false;
}

Type MatrixType::getColumnType() const { return getImpl()->columnType; }

Type MatrixType::getElementType() const {
  return getImpl()->columnType.cast<VectorType>().getElementType();
}

unsigned MatrixType::getNumColumns() const { return getImpl()->columnCount; }

unsigned MatrixType::getNumRows() const {
  return getImpl()->columnType.cast<VectorType>().getShape()[0];
}

unsigned MatrixType::getNumElements() const {
  return (getImpl()->columnCount) * getNumRows();
}

void MatrixType::getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                               Optional<StorageClass> storage) {
  getColumnType().cast<SPIRVType>().getExtensions(extensions, storage);
}

void MatrixType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    Optional<StorageClass> storage) {
  {
    static const Capability caps[] = {Capability::Matrix};
    ArrayRef<Capability> ref(caps, llvm::array_lengthof(caps));
    capabilities.push_back(ref);
  }
  // Add any capabilities associated with the underlying vectors (i.e., columns)
  getColumnType().cast<SPIRVType>().getCapabilities(capabilities, storage);
}
