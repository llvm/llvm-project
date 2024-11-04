//===- SPIRVTypes.h - MLIR SPIR-V Types -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_SPIRVTYPES_H_
#define MLIR_DIALECT_SPIRV_IR_SPIRVTYPES_H_

#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include <cstdint>
#include <tuple>

namespace mlir {
namespace spirv {

namespace detail {
struct ArrayTypeStorage;
struct CooperativeMatrixTypeStorage;
struct CooperativeMatrixNVTypeStorage;
struct ImageTypeStorage;
struct JointMatrixTypeStorage;
struct MatrixTypeStorage;
struct PointerTypeStorage;
struct RuntimeArrayTypeStorage;
struct SampledImageTypeStorage;
struct StructTypeStorage;

} // namespace detail

// Base SPIR-V type for providing availability queries.
class SPIRVType : public Type {
public:
  using Type::Type;

  static bool classof(Type type);

  bool isScalarOrVector();

  /// The extension requirements for each type are following the
  /// ((Extension::A OR Extension::B) AND (Extension::C OR Extension::D))
  /// convention.
  using ExtensionArrayRefVector = SmallVectorImpl<ArrayRef<Extension>>;

  /// Appends to `extensions` the extensions needed for this type to appear in
  /// the given `storage` class. This method does not guarantee the uniqueness
  /// of extensions; the same extension may be appended multiple times.
  void getExtensions(ExtensionArrayRefVector &extensions,
                     std::optional<StorageClass> storage = std::nullopt);

  /// The capability requirements for each type are following the
  /// ((Capability::A OR Extension::B) AND (Capability::C OR Capability::D))
  /// convention.
  using CapabilityArrayRefVector = SmallVectorImpl<ArrayRef<Capability>>;

  /// Appends to `capabilities` the capabilities needed for this type to appear
  /// in the given `storage` class. This method does not guarantee the
  /// uniqueness of capabilities; the same capability may be appended multiple
  /// times.
  void getCapabilities(CapabilityArrayRefVector &capabilities,
                       std::optional<StorageClass> storage = std::nullopt);

  /// Returns the size in bytes for each type. If no size can be calculated,
  /// returns `std::nullopt`. Note that if the type has explicit layout, it is
  /// also taken into account in calculation.
  std::optional<int64_t> getSizeInBytes();
};

// SPIR-V scalar type: bool type, integer type, floating point type.
class ScalarType : public SPIRVType {
public:
  using SPIRVType::SPIRVType;

  static bool classof(Type type);

  /// Returns true if the given integer type is valid for the SPIR-V dialect.
  static bool isValid(FloatType);
  /// Returns true if the given float type is valid for the SPIR-V dialect.
  static bool isValid(IntegerType);

  void getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                     std::optional<StorageClass> storage = std::nullopt);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       std::optional<StorageClass> storage = std::nullopt);

  std::optional<int64_t> getSizeInBytes();
};

// SPIR-V composite type: VectorType, SPIR-V ArrayType, or SPIR-V StructType.
class CompositeType : public SPIRVType {
public:
  using SPIRVType::SPIRVType;

  static bool classof(Type type);

  /// Returns true if the given vector type is valid for the SPIR-V dialect.
  static bool isValid(VectorType);

  /// Return the number of elements of the type. This should only be called if
  /// hasCompileTimeKnownNumElements is true.
  unsigned getNumElements() const;

  Type getElementType(unsigned) const;

  /// Return true if the number of elements is known at compile time and is not
  /// implementation dependent.
  bool hasCompileTimeKnownNumElements() const;

  void getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                     std::optional<StorageClass> storage = std::nullopt);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       std::optional<StorageClass> storage = std::nullopt);

  std::optional<int64_t> getSizeInBytes();
};

// SPIR-V array type
class ArrayType : public Type::TypeBase<ArrayType, CompositeType,
                                        detail::ArrayTypeStorage> {
public:
  using Base::Base;

  static ArrayType get(Type elementType, unsigned elementCount);

  /// Returns an array type with the given stride in bytes.
  static ArrayType get(Type elementType, unsigned elementCount,
                       unsigned stride);

  unsigned getNumElements() const;

  Type getElementType() const;

  /// Returns the array stride in bytes. 0 means no stride decorated on this
  /// type.
  unsigned getArrayStride() const;

  void getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                     std::optional<StorageClass> storage = std::nullopt);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       std::optional<StorageClass> storage = std::nullopt);

  /// Returns the array size in bytes. Since array type may have an explicit
  /// stride declaration (in bytes), we also include it in the calculation.
  std::optional<int64_t> getSizeInBytes();
};

// SPIR-V image type
class ImageType
    : public Type::TypeBase<ImageType, SPIRVType, detail::ImageTypeStorage> {
public:
  using Base::Base;

  static ImageType
  get(Type elementType, Dim dim,
      ImageDepthInfo depth = ImageDepthInfo::DepthUnknown,
      ImageArrayedInfo arrayed = ImageArrayedInfo::NonArrayed,
      ImageSamplingInfo samplingInfo = ImageSamplingInfo::SingleSampled,
      ImageSamplerUseInfo samplerUse = ImageSamplerUseInfo::SamplerUnknown,
      ImageFormat format = ImageFormat::Unknown) {
    return ImageType::get(
        std::tuple<Type, Dim, ImageDepthInfo, ImageArrayedInfo,
                   ImageSamplingInfo, ImageSamplerUseInfo, ImageFormat>(
            elementType, dim, depth, arrayed, samplingInfo, samplerUse,
            format));
  }

  static ImageType
      get(std::tuple<Type, Dim, ImageDepthInfo, ImageArrayedInfo,
                     ImageSamplingInfo, ImageSamplerUseInfo, ImageFormat>);

  Type getElementType() const;
  Dim getDim() const;
  ImageDepthInfo getDepthInfo() const;
  ImageArrayedInfo getArrayedInfo() const;
  ImageSamplingInfo getSamplingInfo() const;
  ImageSamplerUseInfo getSamplerUseInfo() const;
  ImageFormat getImageFormat() const;
  // TODO: Add support for Access qualifier

  void getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                     std::optional<StorageClass> storage = std::nullopt);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       std::optional<StorageClass> storage = std::nullopt);
};

// SPIR-V pointer type
class PointerType : public Type::TypeBase<PointerType, SPIRVType,
                                          detail::PointerTypeStorage> {
public:
  using Base::Base;

  static PointerType get(Type pointeeType, StorageClass storageClass);

  Type getPointeeType() const;

  StorageClass getStorageClass() const;

  void getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                     std::optional<StorageClass> storage = std::nullopt);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       std::optional<StorageClass> storage = std::nullopt);
};

// SPIR-V run-time array type
class RuntimeArrayType
    : public Type::TypeBase<RuntimeArrayType, SPIRVType,
                            detail::RuntimeArrayTypeStorage> {
public:
  using Base::Base;

  static RuntimeArrayType get(Type elementType);

  /// Returns a runtime array type with the given stride in bytes.
  static RuntimeArrayType get(Type elementType, unsigned stride);

  Type getElementType() const;

  /// Returns the array stride in bytes. 0 means no stride decorated on this
  /// type.
  unsigned getArrayStride() const;

  void getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                     std::optional<StorageClass> storage = std::nullopt);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       std::optional<StorageClass> storage = std::nullopt);
};

// SPIR-V sampled image type
class SampledImageType
    : public Type::TypeBase<SampledImageType, SPIRVType,
                            detail::SampledImageTypeStorage> {
public:
  using Base::Base;

  static SampledImageType get(Type imageType);

  static SampledImageType
  getChecked(function_ref<InFlightDiagnostic()> emitError, Type imageType);

  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              Type imageType);

  Type getImageType() const;

  void getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                     std::optional<spirv::StorageClass> storage = std::nullopt);
  void
  getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                  std::optional<spirv::StorageClass> storage = std::nullopt);
};

/// SPIR-V struct type. Two kinds of struct types are supported:
/// - Literal: a literal struct type is uniqued by its fields (types + offset
/// info + decoration info).
/// - Identified: an indentified struct type is uniqued by its string identifier
/// (name). This is useful in representing recursive structs. For example, the
/// following C struct:
///
/// struct A {
///   A* next;
/// };
///
/// would be represented in MLIR as:
///
/// !spirv.struct<A, (!spirv.ptr<!spirv.struct<A>, Generic>)>
///
/// In the above, expressing recursive struct types is accomplished by giving a
/// recursive struct a unique identified and using that identifier in the struct
/// definition for recursive references.
class StructType
    : public Type::TypeBase<StructType, CompositeType,
                            detail::StructTypeStorage, TypeTrait::IsMutable> {
public:
  using Base::Base;

  // Type for specifying the offset of the struct members
  using OffsetInfo = uint32_t;

  // Type for specifying the decoration(s) on struct members
  struct MemberDecorationInfo {
    uint32_t memberIndex : 31;
    uint32_t hasValue : 1;
    Decoration decoration;
    uint32_t decorationValue;

    MemberDecorationInfo(uint32_t index, uint32_t hasValue,
                         Decoration decoration, uint32_t decorationValue)
        : memberIndex(index), hasValue(hasValue), decoration(decoration),
          decorationValue(decorationValue) {}

    bool operator==(const MemberDecorationInfo &other) const {
      return (this->memberIndex == other.memberIndex) &&
             (this->decoration == other.decoration) &&
             (this->decorationValue == other.decorationValue);
    }

    bool operator<(const MemberDecorationInfo &other) const {
      return this->memberIndex < other.memberIndex ||
             (this->memberIndex == other.memberIndex &&
              static_cast<uint32_t>(this->decoration) <
                  static_cast<uint32_t>(other.decoration));
    }
  };

  /// Construct a literal StructType with at least one member.
  static StructType get(ArrayRef<Type> memberTypes,
                        ArrayRef<OffsetInfo> offsetInfo = {},
                        ArrayRef<MemberDecorationInfo> memberDecorations = {});

  /// Construct an identified StructType. This creates a StructType whose body
  /// (member types, offset info, and decorations) is not set yet. A call to
  /// StructType::trySetBody(...) must follow when the StructType contents are
  /// available (e.g. parsed or deserialized).
  ///
  /// Note: If another thread creates (or had already created) a struct with the
  /// same identifier, that struct will be returned as a result.
  static StructType getIdentified(MLIRContext *context, StringRef identifier);

  /// Construct a (possibly identified) StructType with no members.
  ///
  /// Note: this method might fail in a multi-threaded setup if another thread
  /// created an identified struct with the same identifier but with different
  /// contents before returning. In which case, an empty (default-constructed)
  /// StructType is returned.
  static StructType getEmpty(MLIRContext *context, StringRef identifier = "");

  /// For literal structs, return an empty string.
  /// For identified structs, return the struct's identifier.
  StringRef getIdentifier() const;

  /// Returns true if the StructType is identified.
  bool isIdentified() const;

  unsigned getNumElements() const;

  Type getElementType(unsigned) const;

  TypeRange getElementTypes() const;

  bool hasOffset() const;

  uint64_t getMemberOffset(unsigned) const;

  // Returns in `memberDecorations` the Decorations (apart from Offset)
  // associated with all members of the StructType.
  void getMemberDecorations(SmallVectorImpl<StructType::MemberDecorationInfo>
                                &memberDecorations) const;

  // Returns in `decorationsInfo` all the Decorations (apart from Offset)
  // associated with the `i`-th member of the StructType.
  void getMemberDecorations(
      unsigned i,
      SmallVectorImpl<StructType::MemberDecorationInfo> &decorationsInfo) const;

  /// Sets the contents of an incomplete identified StructType. This method must
  /// be called only for identified StructTypes and it must be called only once
  /// per instance. Otherwise, failure() is returned.
  LogicalResult
  trySetBody(ArrayRef<Type> memberTypes, ArrayRef<OffsetInfo> offsetInfo = {},
             ArrayRef<MemberDecorationInfo> memberDecorations = {});

  void getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                     std::optional<StorageClass> storage = std::nullopt);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       std::optional<StorageClass> storage = std::nullopt);
};

llvm::hash_code
hash_value(const StructType::MemberDecorationInfo &memberDecorationInfo);

// SPIR-V KHR cooperative matrix type
class CooperativeMatrixType
    : public Type::TypeBase<CooperativeMatrixType, CompositeType,
                            detail::CooperativeMatrixTypeStorage> {
public:
  using Base::Base;

  static CooperativeMatrixType get(Type elementType, uint32_t rows,
                                   uint32_t columns, Scope scope,
                                   CooperativeMatrixUseKHR use);
  Type getElementType() const;

  /// Returns the scope of the matrix.
  Scope getScope() const;
  /// Returns the number of rows of the matrix.
  uint32_t getRows() const;
  /// Returns the number of columns of the matrix.
  uint32_t getColumns() const;
  /// Returns the use parameter of the cooperative matrix.
  CooperativeMatrixUseKHR getUse() const;

  void getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                     std::optional<StorageClass> storage = std::nullopt);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       std::optional<StorageClass> storage = std::nullopt);
};

// SPIR-V NV cooperative matrix type
class CooperativeMatrixNVType
    : public Type::TypeBase<CooperativeMatrixNVType, CompositeType,
                            detail::CooperativeMatrixNVTypeStorage> {
public:
  using Base::Base;

  static CooperativeMatrixNVType get(Type elementType, Scope scope,
                                     unsigned rows, unsigned columns);
  Type getElementType() const;

  /// Returns the scope of the matrix.
  Scope getScope() const;
  /// Returns the number of rows of the matrix.
  unsigned getRows() const;
  /// Returns the number of columns of the matrix.
  unsigned getColumns() const;

  void getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                     std::optional<StorageClass> storage = std::nullopt);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       std::optional<StorageClass> storage = std::nullopt);
};

// SPIR-V joint matrix type
class JointMatrixINTELType
    : public Type::TypeBase<JointMatrixINTELType, CompositeType,
                            detail::JointMatrixTypeStorage> {
public:
  using Base::Base;

  static JointMatrixINTELType get(Type elementType, Scope scope, unsigned rows,
                                  unsigned columns, MatrixLayout matrixLayout);
  Type getElementType() const;

  /// Return the scope of the joint matrix.
  Scope getScope() const;
  /// return the number of rows of the matrix.
  unsigned getRows() const;
  /// return the number of columns of the matrix.
  unsigned getColumns() const;

  /// return the layout of the matrix
  MatrixLayout getMatrixLayout() const;

  void getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                     std::optional<StorageClass> storage = std::nullopt);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       std::optional<StorageClass> storage = std::nullopt);
};

// SPIR-V matrix type
class MatrixType : public Type::TypeBase<MatrixType, CompositeType,
                                         detail::MatrixTypeStorage> {
public:
  using Base::Base;

  static MatrixType get(Type columnType, uint32_t columnCount);

  static MatrixType getChecked(function_ref<InFlightDiagnostic()> emitError,
                               Type columnType, uint32_t columnCount);

  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              Type columnType, uint32_t columnCount);

  /// Returns true if the matrix elements are vectors of float elements.
  static bool isValidColumnType(Type columnType);

  Type getColumnType() const;

  /// Returns the number of rows.
  unsigned getNumRows() const;

  /// Returns the number of columns.
  unsigned getNumColumns() const;

  /// Returns total number of elements (rows*columns).
  unsigned getNumElements() const;

  /// Returns the elements' type (i.e, single element type).
  Type getElementType() const;

  void getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                     std::optional<StorageClass> storage = std::nullopt);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       std::optional<StorageClass> storage = std::nullopt);
};

} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_IR_SPIRVTYPES_H_
