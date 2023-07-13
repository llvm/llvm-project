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

#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>
#include <iterator>

using namespace mlir;
using namespace mlir::spirv;

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
  return Base::get(elementType.getContext(), elementType, elementCount,
                   /*stride=*/0);
}

ArrayType ArrayType::get(Type elementType, unsigned elementCount,
                         unsigned stride) {
  assert(elementCount && "ArrayType needs at least one element");
  return Base::get(elementType.getContext(), elementType, elementCount, stride);
}

unsigned ArrayType::getNumElements() const { return getImpl()->elementCount; }

Type ArrayType::getElementType() const { return getImpl()->elementType; }

unsigned ArrayType::getArrayStride() const { return getImpl()->stride; }

void ArrayType::getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                              std::optional<StorageClass> storage) {
  llvm::cast<SPIRVType>(getElementType()).getExtensions(extensions, storage);
}

void ArrayType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    std::optional<StorageClass> storage) {
  llvm::cast<SPIRVType>(getElementType())
      .getCapabilities(capabilities, storage);
}

std::optional<int64_t> ArrayType::getSizeInBytes() {
  auto elementType = llvm::cast<SPIRVType>(getElementType());
  std::optional<int64_t> size = elementType.getSizeInBytes();
  if (!size)
    return std::nullopt;
  return (*size + getArrayStride()) * getNumElements();
}

//===----------------------------------------------------------------------===//
// CompositeType
//===----------------------------------------------------------------------===//

bool CompositeType::classof(Type type) {
  if (auto vectorType = llvm::dyn_cast<VectorType>(type))
    return isValid(vectorType);
  return llvm::isa<spirv::ArrayType, spirv::CooperativeMatrixType,
                   spirv::CooperativeMatrixNVType, spirv::JointMatrixINTELType,
                   spirv::MatrixType, spirv::RuntimeArrayType,
                   spirv::StructType>(type);
}

bool CompositeType::isValid(VectorType type) {
  switch (type.getNumElements()) {
  case 2:
  case 3:
  case 4:
  case 8:
  case 16:
    break;
  default:
    return false;
  }
  return type.getRank() == 1 && llvm::isa<ScalarType>(type.getElementType());
}

Type CompositeType::getElementType(unsigned index) const {
  return TypeSwitch<Type, Type>(*this)
      .Case<ArrayType, CooperativeMatrixType, CooperativeMatrixNVType,
            JointMatrixINTELType, RuntimeArrayType, VectorType>(
          [](auto type) { return type.getElementType(); })
      .Case<MatrixType>([](MatrixType type) { return type.getColumnType(); })
      .Case<StructType>(
          [index](StructType type) { return type.getElementType(index); })
      .Default(
          [](Type) -> Type { llvm_unreachable("invalid composite type"); });
}

unsigned CompositeType::getNumElements() const {
  if (auto arrayType = llvm::dyn_cast<ArrayType>(*this))
    return arrayType.getNumElements();
  if (auto matrixType = llvm::dyn_cast<MatrixType>(*this))
    return matrixType.getNumColumns();
  if (auto structType = llvm::dyn_cast<StructType>(*this))
    return structType.getNumElements();
  if (auto vectorType = llvm::dyn_cast<VectorType>(*this))
    return vectorType.getNumElements();
  if (llvm::isa<CooperativeMatrixType, CooperativeMatrixNVType>(*this)) {
    llvm_unreachable(
        "invalid to query number of elements of spirv Cooperative Matrix type");
  }
  if (llvm::isa<JointMatrixINTELType>(*this)) {
    llvm_unreachable(
        "invalid to query number of elements of spirv::JointMatrix type");
  }
  if (llvm::isa<RuntimeArrayType>(*this)) {
    llvm_unreachable(
        "invalid to query number of elements of spirv::RuntimeArray type");
  }
  llvm_unreachable("invalid composite type");
}

bool CompositeType::hasCompileTimeKnownNumElements() const {
  return !llvm::isa<CooperativeMatrixType, CooperativeMatrixNVType,
                    JointMatrixINTELType, RuntimeArrayType>(*this);
}

void CompositeType::getExtensions(
    SPIRVType::ExtensionArrayRefVector &extensions,
    std::optional<StorageClass> storage) {
  TypeSwitch<Type>(*this)
      .Case<ArrayType, CooperativeMatrixType, CooperativeMatrixNVType,
            JointMatrixINTELType, MatrixType, RuntimeArrayType, StructType>(
          [&](auto type) { type.getExtensions(extensions, storage); })
      .Case<VectorType>([&](VectorType type) {
        return llvm::cast<ScalarType>(type.getElementType())
            .getExtensions(extensions, storage);
      })
      .Default([](Type) { llvm_unreachable("invalid composite type"); });
}

void CompositeType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    std::optional<StorageClass> storage) {
  TypeSwitch<Type>(*this)
      .Case<ArrayType, CooperativeMatrixType, CooperativeMatrixNVType,
            JointMatrixINTELType, MatrixType, RuntimeArrayType, StructType>(
          [&](auto type) { type.getCapabilities(capabilities, storage); })
      .Case<VectorType>([&](VectorType type) {
        auto vecSize = getNumElements();
        if (vecSize == 8 || vecSize == 16) {
          static const Capability caps[] = {Capability::Vector16};
          ArrayRef<Capability> ref(caps, std::size(caps));
          capabilities.push_back(ref);
        }
        return llvm::cast<ScalarType>(type.getElementType())
            .getCapabilities(capabilities, storage);
      })
      .Default([](Type) { llvm_unreachable("invalid composite type"); });
}

std::optional<int64_t> CompositeType::getSizeInBytes() {
  if (auto arrayType = llvm::dyn_cast<ArrayType>(*this))
    return arrayType.getSizeInBytes();
  if (auto structType = llvm::dyn_cast<StructType>(*this))
    return structType.getSizeInBytes();
  if (auto vectorType = llvm::dyn_cast<VectorType>(*this)) {
    std::optional<int64_t> elementSize =
        llvm::cast<ScalarType>(vectorType.getElementType()).getSizeInBytes();
    if (!elementSize)
      return std::nullopt;
    return *elementSize * vectorType.getNumElements();
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// CooperativeMatrixType
//===----------------------------------------------------------------------===//

struct spirv::detail::CooperativeMatrixTypeStorage final : TypeStorage {
  using KeyTy =
      std::tuple<Type, uint32_t, uint32_t, Scope, CooperativeMatrixUseKHR>;

  static CooperativeMatrixTypeStorage *
  construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<CooperativeMatrixTypeStorage>())
        CooperativeMatrixTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, rows, columns, scope, use);
  }

  CooperativeMatrixTypeStorage(const KeyTy &key)
      : elementType(std::get<0>(key)), rows(std::get<1>(key)),
        columns(std::get<2>(key)), scope(std::get<3>(key)),
        use(std::get<4>(key)) {}

  Type elementType;
  uint32_t rows;
  uint32_t columns;
  Scope scope;
  CooperativeMatrixUseKHR use;
};

CooperativeMatrixType CooperativeMatrixType::get(Type elementType,
                                                 uint32_t rows,
                                                 uint32_t columns, Scope scope,
                                                 CooperativeMatrixUseKHR use) {
  return Base::get(elementType.getContext(), elementType, rows, columns, scope,
                   use);
}

Type CooperativeMatrixType::getElementType() const {
  return getImpl()->elementType;
}

uint32_t CooperativeMatrixType::getRows() const { return getImpl()->rows; }

uint32_t CooperativeMatrixType::getColumns() const {
  return getImpl()->columns;
}

Scope CooperativeMatrixType::getScope() const { return getImpl()->scope; }

CooperativeMatrixUseKHR CooperativeMatrixType::getUse() const {
  return getImpl()->use;
}

void CooperativeMatrixType::getExtensions(
    SPIRVType::ExtensionArrayRefVector &extensions,
    std::optional<StorageClass> storage) {
  llvm::cast<SPIRVType>(getElementType()).getExtensions(extensions, storage);
  static constexpr Extension exts[] = {Extension::SPV_KHR_cooperative_matrix};
  extensions.push_back(exts);
}

void CooperativeMatrixType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    std::optional<StorageClass> storage) {
  llvm::cast<SPIRVType>(getElementType())
      .getCapabilities(capabilities, storage);
  static constexpr Capability caps[] = {Capability::CooperativeMatrixKHR};
  capabilities.push_back(caps);
}

//===----------------------------------------------------------------------===//
// CooperativeMatrixNVType
//===----------------------------------------------------------------------===//

struct spirv::detail::CooperativeMatrixNVTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<Type, Scope, unsigned, unsigned>;

  static CooperativeMatrixNVTypeStorage *
  construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<CooperativeMatrixNVTypeStorage>())
        CooperativeMatrixNVTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, scope, rows, columns);
  }

  CooperativeMatrixNVTypeStorage(const KeyTy &key)
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
  return Base::get(elementType.getContext(), elementType, scope, rows, columns);
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
    std::optional<StorageClass> storage) {
  llvm::cast<SPIRVType>(getElementType()).getExtensions(extensions, storage);
  static const Extension exts[] = {Extension::SPV_NV_cooperative_matrix};
  ArrayRef<Extension> ref(exts, std::size(exts));
  extensions.push_back(ref);
}

void CooperativeMatrixNVType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    std::optional<StorageClass> storage) {
  llvm::cast<SPIRVType>(getElementType())
      .getCapabilities(capabilities, storage);
  static const Capability caps[] = {Capability::CooperativeMatrixNV};
  ArrayRef<Capability> ref(caps, std::size(caps));
  capabilities.push_back(ref);
}

//===----------------------------------------------------------------------===//
// JointMatrixType
//===----------------------------------------------------------------------===//

struct spirv::detail::JointMatrixTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<Type, unsigned, unsigned, MatrixLayout, Scope>;

  static JointMatrixTypeStorage *construct(TypeStorageAllocator &allocator,
                                           const KeyTy &key) {
    return new (allocator.allocate<JointMatrixTypeStorage>())
        JointMatrixTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, rows, columns, matrixLayout, scope);
  }

  JointMatrixTypeStorage(const KeyTy &key)
      : elementType(std::get<0>(key)), rows(std::get<1>(key)),
        columns(std::get<2>(key)), scope(std::get<4>(key)),
        matrixLayout(std::get<3>(key)) {}

  Type elementType;
  unsigned rows;
  unsigned columns;
  Scope scope;
  MatrixLayout matrixLayout;
};

JointMatrixINTELType JointMatrixINTELType::get(Type elementType, Scope scope,
                                               unsigned rows, unsigned columns,
                                               MatrixLayout matrixLayout) {
  return Base::get(elementType.getContext(), elementType, rows, columns,
                   matrixLayout, scope);
}

Type JointMatrixINTELType::getElementType() const {
  return getImpl()->elementType;
}

Scope JointMatrixINTELType::getScope() const { return getImpl()->scope; }

unsigned JointMatrixINTELType::getRows() const { return getImpl()->rows; }

unsigned JointMatrixINTELType::getColumns() const { return getImpl()->columns; }

MatrixLayout JointMatrixINTELType::getMatrixLayout() const {
  return getImpl()->matrixLayout;
}

void JointMatrixINTELType::getExtensions(
    SPIRVType::ExtensionArrayRefVector &extensions,
    std::optional<StorageClass> storage) {
  llvm::cast<SPIRVType>(getElementType()).getExtensions(extensions, storage);
  static const Extension exts[] = {Extension::SPV_INTEL_joint_matrix};
  ArrayRef<Extension> ref(exts, std::size(exts));
  extensions.push_back(ref);
}

void JointMatrixINTELType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    std::optional<StorageClass> storage) {
  llvm::cast<SPIRVType>(getElementType())
      .getCapabilities(capabilities, storage);
  static const Capability caps[] = {Capability::JointMatrixINTEL};
  ArrayRef<Capability> ref(caps, std::size(caps));
  capabilities.push_back(ref);
}

//===----------------------------------------------------------------------===//
// ImageType
//===----------------------------------------------------------------------===//

template <typename T>
static constexpr unsigned getNumBits() {
  return 0;
}
template <>
constexpr unsigned getNumBits<Dim>() {
  static_assert((1 << 3) > getMaxEnumValForDim(),
                "Not enough bits to encode Dim value");
  return 3;
}
template <>
constexpr unsigned getNumBits<ImageDepthInfo>() {
  static_assert((1 << 2) > getMaxEnumValForImageDepthInfo(),
                "Not enough bits to encode ImageDepthInfo value");
  return 2;
}
template <>
constexpr unsigned getNumBits<ImageArrayedInfo>() {
  static_assert((1 << 1) > getMaxEnumValForImageArrayedInfo(),
                "Not enough bits to encode ImageArrayedInfo value");
  return 1;
}
template <>
constexpr unsigned getNumBits<ImageSamplingInfo>() {
  static_assert((1 << 1) > getMaxEnumValForImageSamplingInfo(),
                "Not enough bits to encode ImageSamplingInfo value");
  return 1;
}
template <>
constexpr unsigned getNumBits<ImageSamplerUseInfo>() {
  static_assert((1 << 2) > getMaxEnumValForImageSamplerUseInfo(),
                "Not enough bits to encode ImageSamplerUseInfo value");
  return 2;
}
template <>
constexpr unsigned getNumBits<ImageFormat>() {
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
  return Base::get(std::get<0>(value).getContext(), value);
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
                              std::optional<StorageClass>) {
  // Image types do not require extra extensions thus far.
}

void ImageType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    std::optional<StorageClass>) {
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
  return Base::get(pointeeType.getContext(), pointeeType, storageClass);
}

Type PointerType::getPointeeType() const { return getImpl()->pointeeType; }

StorageClass PointerType::getStorageClass() const {
  return getImpl()->storageClass;
}

void PointerType::getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                                std::optional<StorageClass> storage) {
  // Use this pointer type's storage class because this pointer indicates we are
  // using the pointee type in that specific storage class.
  llvm::cast<SPIRVType>(getPointeeType())
      .getExtensions(extensions, getStorageClass());

  if (auto scExts = spirv::getExtensions(getStorageClass()))
    extensions.push_back(*scExts);
}

void PointerType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    std::optional<StorageClass> storage) {
  // Use this pointer type's storage class because this pointer indicates we are
  // using the pointee type in that specific storage class.
  llvm::cast<SPIRVType>(getPointeeType())
      .getCapabilities(capabilities, getStorageClass());

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
  return Base::get(elementType.getContext(), elementType, /*stride=*/0);
}

RuntimeArrayType RuntimeArrayType::get(Type elementType, unsigned stride) {
  return Base::get(elementType.getContext(), elementType, stride);
}

Type RuntimeArrayType::getElementType() const { return getImpl()->elementType; }

unsigned RuntimeArrayType::getArrayStride() const { return getImpl()->stride; }

void RuntimeArrayType::getExtensions(
    SPIRVType::ExtensionArrayRefVector &extensions,
    std::optional<StorageClass> storage) {
  llvm::cast<SPIRVType>(getElementType()).getExtensions(extensions, storage);
}

void RuntimeArrayType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    std::optional<StorageClass> storage) {
  {
    static const Capability caps[] = {Capability::Shader};
    ArrayRef<Capability> ref(caps, std::size(caps));
    capabilities.push_back(ref);
  }
  llvm::cast<SPIRVType>(getElementType())
      .getCapabilities(capabilities, storage);
}

//===----------------------------------------------------------------------===//
// ScalarType
//===----------------------------------------------------------------------===//

bool ScalarType::classof(Type type) {
  if (auto floatType = llvm::dyn_cast<FloatType>(type)) {
    return isValid(floatType);
  }
  if (auto intType = llvm::dyn_cast<IntegerType>(type)) {
    return isValid(intType);
  }
  return false;
}

bool ScalarType::isValid(FloatType type) {
  return llvm::is_contained({16u, 32u, 64u}, type.getWidth()) && !type.isBF16();
}

bool ScalarType::isValid(IntegerType type) {
  return llvm::is_contained({1u, 8u, 16u, 32u, 64u}, type.getWidth());
}

void ScalarType::getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                               std::optional<StorageClass> storage) {
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
      ArrayRef<Extension> ref(exts, std::size(exts));
      extensions.push_back(ref);
    }
    [[fallthrough]];
  case StorageClass::Input:
  case StorageClass::Output:
    if (getIntOrFloatBitWidth() == 16) {
      static const Extension exts[] = {Extension::SPV_KHR_16bit_storage};
      ArrayRef<Extension> ref(exts, std::size(exts));
      extensions.push_back(ref);
    }
    break;
  default:
    break;
  }
}

void ScalarType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    std::optional<StorageClass> storage) {
  unsigned bitwidth = getIntOrFloatBitWidth();

  // 8- or 16-bit integer/floating-point numbers will require extra capabilities
  // to appear in interface storage classes. See SPV_KHR_16bit_storage and
  // SPV_KHR_8bit_storage for more details.

#define STORAGE_CASE(storage, cap8, cap16)                                     \
  case StorageClass::storage: {                                                \
    if (bitwidth == 8) {                                                       \
      static const Capability caps[] = {Capability::cap8};                     \
      ArrayRef<Capability> ref(caps, std::size(caps));                         \
      capabilities.push_back(ref);                                             \
      return;                                                                  \
    }                                                                          \
    if (bitwidth == 16) {                                                      \
      static const Capability caps[] = {Capability::cap16};                    \
      ArrayRef<Capability> ref(caps, std::size(caps));                         \
      capabilities.push_back(ref);                                             \
      return;                                                                  \
    }                                                                          \
    /* For 64-bit integers/floats, Int64/Float64 enables support for all */    \
    /* storage classes. Fall through to the next section. */                   \
  } break

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
        ArrayRef<Capability> ref(caps, std::size(caps));
        capabilities.push_back(ref);
        return;
      }
      break;
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
    ArrayRef<Capability> ref(caps, std::size(caps));                           \
    capabilities.push_back(ref);                                               \
  } break

  if (auto intType = llvm::dyn_cast<IntegerType>(*this)) {
    switch (bitwidth) {
      WIDTH_CASE(Int, 8);
      WIDTH_CASE(Int, 16);
      WIDTH_CASE(Int, 64);
    case 1:
    case 32:
      break;
    default:
      llvm_unreachable("invalid bitwidth to getCapabilities");
    }
  } else {
    assert(llvm::isa<FloatType>(*this));
    switch (bitwidth) {
      WIDTH_CASE(Float, 16);
      WIDTH_CASE(Float, 64);
    case 32:
      break;
    default:
      llvm_unreachable("invalid bitwidth to getCapabilities");
    }
  }

#undef WIDTH_CASE
}

std::optional<int64_t> ScalarType::getSizeInBytes() {
  auto bitWidth = getIntOrFloatBitWidth();
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

//===----------------------------------------------------------------------===//
// SPIRVType
//===----------------------------------------------------------------------===//

bool SPIRVType::classof(Type type) {
  // Allow SPIR-V dialect types
  if (llvm::isa<SPIRVDialect>(type.getDialect()))
    return true;
  if (llvm::isa<ScalarType>(type))
    return true;
  if (auto vectorType = llvm::dyn_cast<VectorType>(type))
    return CompositeType::isValid(vectorType);
  return false;
}

bool SPIRVType::isScalarOrVector() {
  return isIntOrFloat() || llvm::isa<VectorType>(*this);
}

void SPIRVType::getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                              std::optional<StorageClass> storage) {
  if (auto scalarType = llvm::dyn_cast<ScalarType>(*this)) {
    scalarType.getExtensions(extensions, storage);
  } else if (auto compositeType = llvm::dyn_cast<CompositeType>(*this)) {
    compositeType.getExtensions(extensions, storage);
  } else if (auto imageType = llvm::dyn_cast<ImageType>(*this)) {
    imageType.getExtensions(extensions, storage);
  } else if (auto sampledImageType = llvm::dyn_cast<SampledImageType>(*this)) {
    sampledImageType.getExtensions(extensions, storage);
  } else if (auto matrixType = llvm::dyn_cast<MatrixType>(*this)) {
    matrixType.getExtensions(extensions, storage);
  } else if (auto ptrType = llvm::dyn_cast<PointerType>(*this)) {
    ptrType.getExtensions(extensions, storage);
  } else {
    llvm_unreachable("invalid SPIR-V Type to getExtensions");
  }
}

void SPIRVType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    std::optional<StorageClass> storage) {
  if (auto scalarType = llvm::dyn_cast<ScalarType>(*this)) {
    scalarType.getCapabilities(capabilities, storage);
  } else if (auto compositeType = llvm::dyn_cast<CompositeType>(*this)) {
    compositeType.getCapabilities(capabilities, storage);
  } else if (auto imageType = llvm::dyn_cast<ImageType>(*this)) {
    imageType.getCapabilities(capabilities, storage);
  } else if (auto sampledImageType = llvm::dyn_cast<SampledImageType>(*this)) {
    sampledImageType.getCapabilities(capabilities, storage);
  } else if (auto matrixType = llvm::dyn_cast<MatrixType>(*this)) {
    matrixType.getCapabilities(capabilities, storage);
  } else if (auto ptrType = llvm::dyn_cast<PointerType>(*this)) {
    ptrType.getCapabilities(capabilities, storage);
  } else {
    llvm_unreachable("invalid SPIR-V Type to getCapabilities");
  }
}

std::optional<int64_t> SPIRVType::getSizeInBytes() {
  if (auto scalarType = llvm::dyn_cast<ScalarType>(*this))
    return scalarType.getSizeInBytes();
  if (auto compositeType = llvm::dyn_cast<CompositeType>(*this))
    return compositeType.getSizeInBytes();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// SampledImageType
//===----------------------------------------------------------------------===//
struct spirv::detail::SampledImageTypeStorage : public TypeStorage {
  using KeyTy = Type;

  SampledImageTypeStorage(const KeyTy &key) : imageType{key} {}

  bool operator==(const KeyTy &key) const { return key == KeyTy(imageType); }

  static SampledImageTypeStorage *construct(TypeStorageAllocator &allocator,
                                            const KeyTy &key) {
    return new (allocator.allocate<SampledImageTypeStorage>())
        SampledImageTypeStorage(key);
  }

  Type imageType;
};

SampledImageType SampledImageType::get(Type imageType) {
  return Base::get(imageType.getContext(), imageType);
}

SampledImageType
SampledImageType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                             Type imageType) {
  return Base::getChecked(emitError, imageType.getContext(), imageType);
}

Type SampledImageType::getImageType() const { return getImpl()->imageType; }

LogicalResult
SampledImageType::verify(function_ref<InFlightDiagnostic()> emitError,
                         Type imageType) {
  if (!llvm::isa<ImageType>(imageType))
    return emitError() << "expected image type";

  return success();
}

void SampledImageType::getExtensions(
    SPIRVType::ExtensionArrayRefVector &extensions,
    std::optional<StorageClass> storage) {
  llvm::cast<ImageType>(getImageType()).getExtensions(extensions, storage);
}

void SampledImageType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    std::optional<StorageClass> storage) {
  llvm::cast<ImageType>(getImageType()).getCapabilities(capabilities, storage);
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

/// Type storage for SPIR-V structure types:
///
/// Structures are uniqued using:
/// - for identified structs:
///   - a string identifier;
/// - for literal structs:
///   - a list of member types;
///   - a list of member offset info;
///   - a list of member decoration info.
///
/// Identified structures only have a mutable component consisting of:
/// - a list of member types;
/// - a list of member offset info;
/// - a list of member decoration info.
struct spirv::detail::StructTypeStorage : public TypeStorage {
  /// Construct a storage object for an identified struct type. A struct type
  /// associated with such storage must call StructType::trySetBody(...) later
  /// in order to mutate the storage object providing the actual content.
  StructTypeStorage(StringRef identifier)
      : memberTypesAndIsBodySet(nullptr, false), offsetInfo(nullptr),
        numMembers(0), numMemberDecorations(0), memberDecorationsInfo(nullptr),
        identifier(identifier) {}

  /// Construct a storage object for a literal struct type. A struct type
  /// associated with such storage is immutable.
  StructTypeStorage(
      unsigned numMembers, Type const *memberTypes,
      StructType::OffsetInfo const *layoutInfo, unsigned numMemberDecorations,
      StructType::MemberDecorationInfo const *memberDecorationsInfo)
      : memberTypesAndIsBodySet(memberTypes, false), offsetInfo(layoutInfo),
        numMembers(numMembers), numMemberDecorations(numMemberDecorations),
        memberDecorationsInfo(memberDecorationsInfo) {}

  /// A storage key is divided into 2 parts:
  /// - for identified structs:
  ///   - a StringRef representing the struct identifier;
  /// - for literal structs:
  ///   - an ArrayRef<Type> for member types;
  ///   - an ArrayRef<StructType::OffsetInfo> for member offset info;
  ///   - an ArrayRef<StructType::MemberDecorationInfo> for member decoration
  ///     info.
  ///
  /// An identified struct type is uniqued only by the first part (field 0)
  /// of the key.
  ///
  /// A literal struct type is uniqued only by the second part (fields 1, 2, and
  /// 3) of the key. The identifier field (field 0) must be empty.
  using KeyTy =
      std::tuple<StringRef, ArrayRef<Type>, ArrayRef<StructType::OffsetInfo>,
                 ArrayRef<StructType::MemberDecorationInfo>>;

  /// For identified structs, return true if the given key contains the same
  /// identifier.
  ///
  /// For literal structs, return true if the given key contains a matching list
  /// of member types + offset info + decoration info.
  bool operator==(const KeyTy &key) const {
    if (isIdentified()) {
      // Identified types are uniqued by their identifier.
      return getIdentifier() == std::get<0>(key);
    }

    return key == KeyTy(StringRef(), getMemberTypes(), getOffsetInfo(),
                        getMemberDecorationsInfo());
  }

  /// If the given key contains a non-empty identifier, this method constructs
  /// an identified struct and leaves the rest of the struct type data to be set
  /// through a later call to StructType::trySetBody(...).
  ///
  /// If, on the other hand, the key contains an empty identifier, a literal
  /// struct is constructed using the other fields of the key.
  static StructTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    StringRef keyIdentifier = std::get<0>(key);

    if (!keyIdentifier.empty()) {
      StringRef identifier = allocator.copyInto(keyIdentifier);

      // Identified StructType body/members will be set through trySetBody(...)
      // later.
      return new (allocator.allocate<StructTypeStorage>())
          StructTypeStorage(identifier);
    }

    ArrayRef<Type> keyTypes = std::get<1>(key);

    // Copy the member type and layout information into the bump pointer
    const Type *typesList = nullptr;
    if (!keyTypes.empty()) {
      typesList = allocator.copyInto(keyTypes).data();
    }

    const StructType::OffsetInfo *offsetInfoList = nullptr;
    if (!std::get<2>(key).empty()) {
      ArrayRef<StructType::OffsetInfo> keyOffsetInfo = std::get<2>(key);
      assert(keyOffsetInfo.size() == keyTypes.size() &&
             "size of offset information must be same as the size of number of "
             "elements");
      offsetInfoList = allocator.copyInto(keyOffsetInfo).data();
    }

    const StructType::MemberDecorationInfo *memberDecorationList = nullptr;
    unsigned numMemberDecorations = 0;
    if (!std::get<3>(key).empty()) {
      auto keyMemberDecorations = std::get<3>(key);
      numMemberDecorations = keyMemberDecorations.size();
      memberDecorationList = allocator.copyInto(keyMemberDecorations).data();
    }

    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(keyTypes.size(), typesList, offsetInfoList,
                          numMemberDecorations, memberDecorationList);
  }

  ArrayRef<Type> getMemberTypes() const {
    return ArrayRef<Type>(memberTypesAndIsBodySet.getPointer(), numMembers);
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

  StringRef getIdentifier() const { return identifier; }

  bool isIdentified() const { return !identifier.empty(); }

  /// Sets the struct type content for identified structs. Calling this method
  /// is only valid for identified structs.
  ///
  /// Fails under the following conditions:
  /// - If called for a literal struct;
  /// - If called for an identified struct whose body was set before (through a
  /// call to this method) but with different contents from the passed
  /// arguments.
  LogicalResult mutate(
      TypeStorageAllocator &allocator, ArrayRef<Type> structMemberTypes,
      ArrayRef<StructType::OffsetInfo> structOffsetInfo,
      ArrayRef<StructType::MemberDecorationInfo> structMemberDecorationInfo) {
    if (!isIdentified())
      return failure();

    if (memberTypesAndIsBodySet.getInt() &&
        (getMemberTypes() != structMemberTypes ||
         getOffsetInfo() != structOffsetInfo ||
         getMemberDecorationsInfo() != structMemberDecorationInfo))
      return failure();

    memberTypesAndIsBodySet.setInt(true);
    numMembers = structMemberTypes.size();

    // Copy the member type and layout information into the bump pointer.
    if (!structMemberTypes.empty())
      memberTypesAndIsBodySet.setPointer(
          allocator.copyInto(structMemberTypes).data());

    if (!structOffsetInfo.empty()) {
      assert(structOffsetInfo.size() == structMemberTypes.size() &&
             "size of offset information must be same as the size of number of "
             "elements");
      offsetInfo = allocator.copyInto(structOffsetInfo).data();
    }

    if (!structMemberDecorationInfo.empty()) {
      numMemberDecorations = structMemberDecorationInfo.size();
      memberDecorationsInfo =
          allocator.copyInto(structMemberDecorationInfo).data();
    }

    return success();
  }

  llvm::PointerIntPair<Type const *, 1, bool> memberTypesAndIsBodySet;
  StructType::OffsetInfo const *offsetInfo;
  unsigned numMembers;
  unsigned numMemberDecorations;
  StructType::MemberDecorationInfo const *memberDecorationsInfo;
  StringRef identifier;
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
  return Base::get(memberTypes.vec().front().getContext(),
                   /*identifier=*/StringRef(), memberTypes, offsetInfo,
                   sortedDecorations);
}

StructType StructType::getIdentified(MLIRContext *context,
                                     StringRef identifier) {
  assert(!identifier.empty() &&
         "StructType identifier must be non-empty string");

  return Base::get(context, identifier, ArrayRef<Type>(),
                   ArrayRef<StructType::OffsetInfo>(),
                   ArrayRef<StructType::MemberDecorationInfo>());
}

StructType StructType::getEmpty(MLIRContext *context, StringRef identifier) {
  StructType newStructType = Base::get(
      context, identifier, ArrayRef<Type>(), ArrayRef<StructType::OffsetInfo>(),
      ArrayRef<StructType::MemberDecorationInfo>());
  // Set an empty body in case this is a identified struct.
  if (newStructType.isIdentified() &&
      failed(newStructType.trySetBody(
          ArrayRef<Type>(), ArrayRef<StructType::OffsetInfo>(),
          ArrayRef<StructType::MemberDecorationInfo>())))
    return StructType();

  return newStructType;
}

StringRef StructType::getIdentifier() const { return getImpl()->identifier; }

bool StructType::isIdentified() const { return getImpl()->isIdentified(); }

unsigned StructType::getNumElements() const { return getImpl()->numMembers; }

Type StructType::getElementType(unsigned index) const {
  assert(getNumElements() > index && "member index out of range");
  return getImpl()->memberTypesAndIsBodySet.getPointer()[index];
}

StructType::ElementTypeRange StructType::getElementTypes() const {
  return ElementTypeRange(getImpl()->memberTypesAndIsBodySet.getPointer(),
                          getNumElements());
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

LogicalResult
StructType::trySetBody(ArrayRef<Type> memberTypes,
                       ArrayRef<OffsetInfo> offsetInfo,
                       ArrayRef<MemberDecorationInfo> memberDecorations) {
  return Base::mutate(memberTypes, offsetInfo, memberDecorations);
}

void StructType::getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                               std::optional<StorageClass> storage) {
  for (Type elementType : getElementTypes())
    llvm::cast<SPIRVType>(elementType).getExtensions(extensions, storage);
}

void StructType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    std::optional<StorageClass> storage) {
  for (Type elementType : getElementTypes())
    llvm::cast<SPIRVType>(elementType).getCapabilities(capabilities, storage);
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
      : columnType(columnType), columnCount(columnCount) {}

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
  return Base::get(columnType.getContext(), columnType, columnCount);
}

MatrixType MatrixType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  Type columnType, uint32_t columnCount) {
  return Base::getChecked(emitError, columnType.getContext(), columnType,
                          columnCount);
}

LogicalResult MatrixType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 Type columnType, uint32_t columnCount) {
  if (columnCount < 2 || columnCount > 4)
    return emitError() << "matrix can have 2, 3, or 4 columns only";

  if (!isValidColumnType(columnType))
    return emitError() << "matrix columns must be vectors of floats";

  /// The underlying vectors (columns) must be of size 2, 3, or 4
  ArrayRef<int64_t> columnShape = llvm::cast<VectorType>(columnType).getShape();
  if (columnShape.size() != 1)
    return emitError() << "matrix columns must be 1D vectors";

  if (columnShape[0] < 2 || columnShape[0] > 4)
    return emitError() << "matrix columns must be of size 2, 3, or 4";

  return success();
}

/// Returns true if the matrix elements are vectors of float elements
bool MatrixType::isValidColumnType(Type columnType) {
  if (auto vectorType = llvm::dyn_cast<VectorType>(columnType)) {
    if (llvm::isa<FloatType>(vectorType.getElementType()))
      return true;
  }
  return false;
}

Type MatrixType::getColumnType() const { return getImpl()->columnType; }

Type MatrixType::getElementType() const {
  return llvm::cast<VectorType>(getImpl()->columnType).getElementType();
}

unsigned MatrixType::getNumColumns() const { return getImpl()->columnCount; }

unsigned MatrixType::getNumRows() const {
  return llvm::cast<VectorType>(getImpl()->columnType).getShape()[0];
}

unsigned MatrixType::getNumElements() const {
  return (getImpl()->columnCount) * getNumRows();
}

void MatrixType::getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                               std::optional<StorageClass> storage) {
  llvm::cast<SPIRVType>(getColumnType()).getExtensions(extensions, storage);
}

void MatrixType::getCapabilities(
    SPIRVType::CapabilityArrayRefVector &capabilities,
    std::optional<StorageClass> storage) {
  {
    static const Capability caps[] = {Capability::Matrix};
    ArrayRef<Capability> ref(caps, std::size(caps));
    capabilities.push_back(ref);
  }
  // Add any capabilities associated with the underlying vectors (i.e., columns)
  llvm::cast<SPIRVType>(getColumnType()).getCapabilities(capabilities, storage);
}

//===----------------------------------------------------------------------===//
// SPIR-V Dialect
//===----------------------------------------------------------------------===//

void SPIRVDialect::registerTypes() {
  addTypes<ArrayType, CooperativeMatrixType, CooperativeMatrixNVType, ImageType,
           JointMatrixINTELType, MatrixType, PointerType, RuntimeArrayType,
           SampledImageType, StructType>();
}
