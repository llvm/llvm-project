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

  // TODO(antiagainst): This is insufficient; find a better way to handle this
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
    return key == KeyTy(elementType, getSubclassData(), stride);
  }

  ArrayTypeStorage(const KeyTy &key)
      : TypeStorage(std::get<1>(key)), elementType(std::get<0>(key)),
        stride(std::get<2>(key)) {}

  Type elementType;
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

unsigned ArrayType::getNumElements() const {
  return getImpl()->getSubclassData();
}

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

//===----------------------------------------------------------------------===//
// CompositeType
//===----------------------------------------------------------------------===//

bool CompositeType::classof(Type type) {
  switch (type.getKind()) {
  case TypeKind::Array:
  case TypeKind::CooperativeMatrix:
  case TypeKind::Matrix:
  case TypeKind::RuntimeArray:
  case TypeKind::Struct:
    return true;
  case StandardTypes::Vector:
    return isValid(type.cast<VectorType>());
  default:
    return false;
  }
}

bool CompositeType::isValid(VectorType type) {
  return type.getRank() == 1 && type.getElementType().isa<ScalarType>() &&
         type.getNumElements() >= 2 && type.getNumElements() <= 4;
}

Type CompositeType::getElementType(unsigned index) const {
  switch (getKind()) {
  case spirv::TypeKind::Array:
    return cast<ArrayType>().getElementType();
  case spirv::TypeKind::CooperativeMatrix:
    return cast<CooperativeMatrixNVType>().getElementType();
  case spirv::TypeKind::Matrix:
    return cast<MatrixType>().getElementType();
  case spirv::TypeKind::RuntimeArray:
    return cast<RuntimeArrayType>().getElementType();
  case spirv::TypeKind::Struct:
    return cast<StructType>().getElementType(index);
  case StandardTypes::Vector:
    return cast<VectorType>().getElementType();
  default:
    llvm_unreachable("invalid composite type");
  }
}

unsigned CompositeType::getNumElements() const {
  switch (getKind()) {
  case spirv::TypeKind::Array:
    return cast<ArrayType>().getNumElements();
  case spirv::TypeKind::CooperativeMatrix:
    llvm_unreachable(
        "invalid to query number of elements of spirv::CooperativeMatrix type");
  case spirv::TypeKind::Matrix:
    return cast<MatrixType>().getNumElements();
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
    return key == KeyTy(elementType, getScope(), rows, columns);
  }

  CooperativeMatrixTypeStorage(const KeyTy &key)
      : TypeStorage(static_cast<unsigned>(std::get<1>(key))),
        elementType(std::get<0>(key)), rows(std::get<2>(key)),
        columns(std::get<3>(key)) {}

  Scope getScope() const { return static_cast<Scope>(getSubclassData()); }

  Type elementType;
  unsigned rows;
  unsigned columns;
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

Scope CooperativeMatrixNVType::getScope() const {
  return getImpl()->getScope();
}

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
private:
  /// Define a bit-field struct to pack the enum values
  union EnumPack {
    struct {
      unsigned dimEncoding : getNumBits<Dim>();
      unsigned depthInfoEncoding : getNumBits<ImageDepthInfo>();
      unsigned arrayedInfoEncoding : getNumBits<ImageArrayedInfo>();
      unsigned samplingInfoEncoding : getNumBits<ImageSamplingInfo>();
      unsigned samplerUseInfoEncoding : getNumBits<ImageSamplerUseInfo>();
      unsigned formatEncoding : getNumBits<ImageFormat>();
    } data;
    unsigned storage;
  };

public:
  using KeyTy = std::tuple<Type, Dim, ImageDepthInfo, ImageArrayedInfo,
                           ImageSamplingInfo, ImageSamplerUseInfo, ImageFormat>;

  static ImageTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<ImageTypeStorage>()) ImageTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, getDim(), getDepthInfo(), getArrayedInfo(),
                        getSamplingInfo(), getSamplerUseInfo(),
                        getImageFormat());
  }

  Dim getDim() const {
    EnumPack v;
    v.storage = getSubclassData();
    return static_cast<Dim>(v.data.dimEncoding);
  }
  void setDim(Dim dim) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.dimEncoding = static_cast<unsigned>(dim);
    setSubclassData(v.storage);
  }

  ImageDepthInfo getDepthInfo() const {
    EnumPack v;
    v.storage = getSubclassData();
    return static_cast<ImageDepthInfo>(v.data.depthInfoEncoding);
  }
  void setDepthInfo(ImageDepthInfo depthInfo) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.depthInfoEncoding = static_cast<unsigned>(depthInfo);
    setSubclassData(v.storage);
  }

  ImageArrayedInfo getArrayedInfo() const {
    EnumPack v;
    v.storage = getSubclassData();
    return static_cast<ImageArrayedInfo>(v.data.arrayedInfoEncoding);
  }
  void setArrayedInfo(ImageArrayedInfo arrayedInfo) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.arrayedInfoEncoding = static_cast<unsigned>(arrayedInfo);
    setSubclassData(v.storage);
  }

  ImageSamplingInfo getSamplingInfo() const {
    EnumPack v;
    v.storage = getSubclassData();
    return static_cast<ImageSamplingInfo>(v.data.samplingInfoEncoding);
  }
  void setSamplingInfo(ImageSamplingInfo samplingInfo) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.samplingInfoEncoding = static_cast<unsigned>(samplingInfo);
    setSubclassData(v.storage);
  }

  ImageSamplerUseInfo getSamplerUseInfo() const {
    EnumPack v;
    v.storage = getSubclassData();
    return static_cast<ImageSamplerUseInfo>(v.data.samplerUseInfoEncoding);
  }
  void setSamplerUseInfo(ImageSamplerUseInfo samplerUseInfo) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.samplerUseInfoEncoding = static_cast<unsigned>(samplerUseInfo);
    setSubclassData(v.storage);
  }

  ImageFormat getImageFormat() const {
    EnumPack v;
    v.storage = getSubclassData();
    return static_cast<ImageFormat>(v.data.formatEncoding);
  }
  void setImageFormat(ImageFormat format) {
    EnumPack v;
    v.storage = getSubclassData();
    v.data.formatEncoding = static_cast<unsigned>(format);
    setSubclassData(v.storage);
  }

  ImageTypeStorage(const KeyTy &key) : elementType(std::get<0>(key)) {
    static_assert(sizeof(EnumPack) <= sizeof(getSubclassData()),
                  "EnumPack size greater than subClassData type size");
    setDim(std::get<1>(key));
    setDepthInfo(std::get<2>(key));
    setArrayedInfo(std::get<3>(key));
    setSamplingInfo(std::get<4>(key));
    setSamplerUseInfo(std::get<5>(key));
    setImageFormat(std::get<6>(key));
  }

  Type elementType;
};

ImageType
ImageType::get(std::tuple<Type, Dim, ImageDepthInfo, ImageArrayedInfo,
                          ImageSamplingInfo, ImageSamplerUseInfo, ImageFormat>
                   value) {
  return Base::get(std::get<0>(value).getContext(), TypeKind::Image, value);
}

Type ImageType::getElementType() const { return getImpl()->elementType; }

Dim ImageType::getDim() const { return getImpl()->getDim(); }

ImageDepthInfo ImageType::getDepthInfo() const {
  return getImpl()->getDepthInfo();
}

ImageArrayedInfo ImageType::getArrayedInfo() const {
  return getImpl()->getArrayedInfo();
}

ImageSamplingInfo ImageType::getSamplingInfo() const {
  return getImpl()->getSamplingInfo();
}

ImageSamplerUseInfo ImageType::getSamplerUseInfo() const {
  return getImpl()->getSamplerUseInfo();
}

ImageFormat ImageType::getImageFormat() const {
  return getImpl()->getImageFormat();
}

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
    return key == KeyTy(pointeeType, getStorageClass());
  }

  PointerTypeStorage(const KeyTy &key)
      : TypeStorage(static_cast<unsigned>(key.second)), pointeeType(key.first) {
  }

  StorageClass getStorageClass() const {
    return static_cast<StorageClass>(getSubclassData());
  }

  Type pointeeType;
};

PointerType PointerType::get(Type pointeeType, StorageClass storageClass) {
  return Base::get(pointeeType.getContext(), TypeKind::Pointer, pointeeType,
                   storageClass);
}

Type PointerType::getPointeeType() const { return getImpl()->pointeeType; }

StorageClass PointerType::getStorageClass() const {
  return getImpl()->getStorageClass();
}

void PointerType::getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                                Optional<StorageClass> storage) {
  if (storage)
    assert(*storage == getStorageClass() && "inconsistent storage class!");

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
  if (storage)
    assert(*storage == getStorageClass() && "inconsistent storage class!");

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
    return key == KeyTy(elementType, getSubclassData());
  }

  RuntimeArrayTypeStorage(const KeyTy &key)
      : TypeStorage(key.second), elementType(key.first) {}

  Type elementType;
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

unsigned RuntimeArrayType::getArrayStride() const {
  return getImpl()->getSubclassData();
}

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
  } break

  if (storage) {
    switch (*storage) {
      STORAGE_CASE(PushConstant, StoragePushConstant8, StoragePushConstant16);
      STORAGE_CASE(StorageBuffer, StorageBuffer8BitAccess,
                   StorageBuffer16BitAccess);
      STORAGE_CASE(Uniform, UniformAndStorageBuffer8BitAccess,
                   StorageUniform16);
    case StorageClass::Input:
    case StorageClass::Output:
      if (bitwidth == 16) {
        static const Capability caps[] = {Capability::StorageInputOutput16};
        ArrayRef<Capability> ref(caps, llvm::array_lengthof(caps));
        capabilities.push_back(ref);
      }
      break;
    default:
      break;
    }
    return;
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

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

struct spirv::detail::StructTypeStorage : public TypeStorage {
  StructTypeStorage(
      unsigned numMembers, Type const *memberTypes,
      StructType::LayoutInfo const *layoutInfo, unsigned numMemberDecorations,
      StructType::MemberDecorationInfo const *memberDecorationsInfo)
      : TypeStorage(numMembers), memberTypes(memberTypes),
        layoutInfo(layoutInfo), numMemberDecorations(numMemberDecorations),
        memberDecorationsInfo(memberDecorationsInfo) {}

  using KeyTy = std::tuple<ArrayRef<Type>, ArrayRef<StructType::LayoutInfo>,
                           ArrayRef<StructType::MemberDecorationInfo>>;
  bool operator==(const KeyTy &key) const {
    return key ==
           KeyTy(getMemberTypes(), getLayoutInfo(), getMemberDecorationsInfo());
  }

  static StructTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    ArrayRef<Type> keyTypes = std::get<0>(key);

    // Copy the member type and layout information into the bump pointer
    const Type *typesList = nullptr;
    if (!keyTypes.empty()) {
      typesList = allocator.copyInto(keyTypes).data();
    }

    const StructType::LayoutInfo *layoutInfoList = nullptr;
    if (!std::get<1>(key).empty()) {
      ArrayRef<StructType::LayoutInfo> keyLayoutInfo = std::get<1>(key);
      assert(keyLayoutInfo.size() == keyTypes.size() &&
             "size of layout information must be same as the size of number of "
             "elements");
      layoutInfoList = allocator.copyInto(keyLayoutInfo).data();
    }

    const StructType::MemberDecorationInfo *memberDecorationList = nullptr;
    unsigned numMemberDecorations = 0;
    if (!std::get<2>(key).empty()) {
      auto keyMemberDecorations = std::get<2>(key);
      numMemberDecorations = keyMemberDecorations.size();
      memberDecorationList = allocator.copyInto(keyMemberDecorations).data();
    }
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(keyTypes.size(), typesList, layoutInfoList,
                          numMemberDecorations, memberDecorationList);
  }

  ArrayRef<Type> getMemberTypes() const {
    return ArrayRef<Type>(memberTypes, getSubclassData());
  }

  ArrayRef<StructType::LayoutInfo> getLayoutInfo() const {
    if (layoutInfo) {
      return ArrayRef<StructType::LayoutInfo>(layoutInfo, getSubclassData());
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
  StructType::LayoutInfo const *layoutInfo;
  unsigned numMemberDecorations;
  StructType::MemberDecorationInfo const *memberDecorationsInfo;
};

StructType
StructType::get(ArrayRef<Type> memberTypes,
                ArrayRef<StructType::LayoutInfo> layoutInfo,
                ArrayRef<StructType::MemberDecorationInfo> memberDecorations) {
  assert(!memberTypes.empty() && "Struct needs at least one member type");
  // Sort the decorations.
  SmallVector<StructType::MemberDecorationInfo, 4> sortedDecorations(
      memberDecorations.begin(), memberDecorations.end());
  llvm::array_pod_sort(sortedDecorations.begin(), sortedDecorations.end());
  return Base::get(memberTypes.vec().front().getContext(), TypeKind::Struct,
                   memberTypes, layoutInfo, sortedDecorations);
}

StructType StructType::getEmpty(MLIRContext *context) {
  return Base::get(context, TypeKind::Struct, ArrayRef<Type>(),
                   ArrayRef<StructType::LayoutInfo>(),
                   ArrayRef<StructType::MemberDecorationInfo>());
}

unsigned StructType::getNumElements() const {
  return getImpl()->getSubclassData();
}

Type StructType::getElementType(unsigned index) const {
  assert(getNumElements() > index && "member index out of range");
  return getImpl()->memberTypes[index];
}

StructType::ElementTypeRange StructType::getElementTypes() const {
  return ElementTypeRange(getImpl()->memberTypes, getNumElements());
}

bool StructType::hasLayout() const { return getImpl()->layoutInfo; }

uint64_t StructType::getOffset(unsigned index) const {
  assert(getNumElements() > index && "member index out of range");
  return getImpl()->layoutInfo[index];
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
    unsigned index, SmallVectorImpl<spirv::Decoration> &decorations) const {
  assert(getNumElements() > index && "member index out of range");
  auto memberDecorations = getImpl()->getMemberDecorationsInfo();
  decorations.clear();
  for (auto &memberDecoration : memberDecorations) {
    if (memberDecoration.first == index) {
      decorations.push_back(memberDecoration.second);
    }
    if (memberDecoration.first > index) {
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

Type MatrixType::getElementType() const { return getImpl()->columnType; }

unsigned MatrixType::getNumElements() const { return getImpl()->columnCount; }

void MatrixType::getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                               Optional<StorageClass> storage) {
  getElementType().cast<SPIRVType>().getExtensions(extensions, storage);
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
  getElementType().cast<SPIRVType>().getCapabilities(capabilities, storage);
}
