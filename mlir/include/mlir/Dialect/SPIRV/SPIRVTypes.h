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

#ifndef MLIR_DIALECT_SPIRV_SPIRVTYPES_H_
#define MLIR_DIALECT_SPIRV_SPIRVTYPES_H_

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include <tuple>

// Forward declare enum classes related to op availability. Their definitions
// are in the TableGen'erated SPIRVEnums.h.inc and can be referenced by other
// declarations in SPIRVEnums.h.inc.
namespace mlir {
namespace spirv {
enum class Version : uint32_t;
enum class Extension;
enum class Capability : uint32_t;
} // namespace spirv
} // namespace mlir

// Pull in all enum type definitions and utility function declarations
#include "mlir/Dialect/SPIRV/SPIRVEnums.h.inc"
// Pull in all enum type availability query function declarations
#include "mlir/Dialect/SPIRV/SPIRVEnumAvailability.h.inc"

namespace mlir {
namespace spirv {
/// Returns the implied extensions for the given version. These extensions are
/// incorporated into the current version so they are implicitly declared when
/// targeting the given version.
ArrayRef<Extension> getImpliedExtensions(Version version);

/// Returns the directly implied capabilities for the given capability. These
/// capabilities are implicitly declared by the given capability.
ArrayRef<Capability> getDirectImpliedCapabilities(Capability cap);
/// Returns the recursively implied capabilities for the given capability. These
/// capabilities are implicitly declared by the given capability. Compared to
/// the above function, this function collects implied capabilities recursively:
/// if an implicitly declared capability implicitly declares a third one, the
/// third one will also be returned.
SmallVector<Capability, 0> getRecursiveImpliedCapabilities(Capability cap);

namespace detail {
struct ArrayTypeStorage;
struct CooperativeMatrixTypeStorage;
struct ImageTypeStorage;
struct MatrixTypeStorage;
struct PointerTypeStorage;
struct RuntimeArrayTypeStorage;
struct StructTypeStorage;

} // namespace detail

namespace TypeKind {
enum Kind {
  Array = Type::FIRST_SPIRV_TYPE,
  CooperativeMatrix,
  Image,
  Matrix,
  Pointer,
  RuntimeArray,
  Struct,
  LAST_SPIRV_TYPE = Struct,
};
}

// Base SPIR-V type for providing availability queries.
class SPIRVType : public Type {
public:
  using Type::Type;

  static bool classof(Type type);

  bool isScalarOrVector();

  /// The extension requirements for each type are following the
  /// ((Extension::A OR Extension::B) AND (Extension::C OR Extension::D))
  /// convention.
  using ExtensionArrayRefVector = SmallVectorImpl<ArrayRef<spirv::Extension>>;

  /// Appends to `extensions` the extensions needed for this type to appear in
  /// the given `storage` class. This method does not guarantee the uniqueness
  /// of extensions; the same extension may be appended multiple times.
  void getExtensions(ExtensionArrayRefVector &extensions,
                     Optional<spirv::StorageClass> storage = llvm::None);

  /// The capability requirements for each type are following the
  /// ((Capability::A OR Extension::B) AND (Capability::C OR Capability::D))
  /// convention.
  using CapabilityArrayRefVector = SmallVectorImpl<ArrayRef<spirv::Capability>>;

  /// Appends to `capabilities` the capabilities needed for this type to appear
  /// in the given `storage` class. This method does not guarantee the
  /// uniqueness of capabilities; the same capability may be appended multiple
  /// times.
  void getCapabilities(CapabilityArrayRefVector &capabilities,
                       Optional<spirv::StorageClass> storage = llvm::None);

  /// Returns the size in bytes for each type. If no size can be calculated,
  /// returns `llvm::None`. Note that if the type has explicit layout, it is
  /// also taken into account in calculation.
  Optional<int64_t> getSizeInBytes();
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
                     Optional<spirv::StorageClass> storage = llvm::None);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       Optional<spirv::StorageClass> storage = llvm::None);

  Optional<int64_t> getSizeInBytes();
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
                     Optional<spirv::StorageClass> storage = llvm::None);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       Optional<spirv::StorageClass> storage = llvm::None);

  Optional<int64_t> getSizeInBytes();
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
                     Optional<spirv::StorageClass> storage = llvm::None);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       Optional<spirv::StorageClass> storage = llvm::None);

  /// Returns the array size in bytes. Since array type may have an explicit
  /// stride declaration (in bytes), we also include it in the calculation.
  Optional<int64_t> getSizeInBytes();
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
                     Optional<spirv::StorageClass> storage = llvm::None);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       Optional<spirv::StorageClass> storage = llvm::None);
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
                     Optional<spirv::StorageClass> storage = llvm::None);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       Optional<spirv::StorageClass> storage = llvm::None);
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
                     Optional<spirv::StorageClass> storage = llvm::None);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       Optional<spirv::StorageClass> storage = llvm::None);
};

// SPIR-V struct type
class StructType : public Type::TypeBase<StructType, CompositeType,
                                         detail::StructTypeStorage> {
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

  /// Construct a StructType with at least one member.
  static StructType get(ArrayRef<Type> memberTypes,
                        ArrayRef<OffsetInfo> offsetInfo = {},
                        ArrayRef<MemberDecorationInfo> memberDecorations = {});

  /// Construct a struct with no members.
  static StructType getEmpty(MLIRContext *context);

  unsigned getNumElements() const;

  Type getElementType(unsigned) const;

  /// Range class for element types.
  class ElementTypeRange
      : public ::llvm::detail::indexed_accessor_range_base<
            ElementTypeRange, const Type *, Type, Type, Type> {
  private:
    using RangeBaseT::RangeBaseT;

    /// See `llvm::detail::indexed_accessor_range_base` for details.
    static const Type *offset_base(const Type *object, ptrdiff_t index) {
      return object + index;
    }
    /// See `llvm::detail::indexed_accessor_range_base` for details.
    static Type dereference_iterator(const Type *object, ptrdiff_t index) {
      return object[index];
    }

    /// Allow base class access to `offset_base` and `dereference_iterator`.
    friend RangeBaseT;
  };

  ElementTypeRange getElementTypes() const;

  bool hasOffset() const;

  uint64_t getMemberOffset(unsigned) const;

  // Returns in `memberDecorations` the spirv::Decorations (apart from
  // Offset) associated with all members of the StructType.
  void getMemberDecorations(SmallVectorImpl<StructType::MemberDecorationInfo>
                                &memberDecorations) const;

  // Returns in `decorationsInfo` all the spirv::Decorations (apart from
  // Offset) associated with the `i`-th member of the StructType.
  void getMemberDecorations(unsigned i,
                            SmallVectorImpl<StructType::MemberDecorationInfo>
                                &decorationsInfo) const;

  void getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                     Optional<spirv::StorageClass> storage = llvm::None);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       Optional<spirv::StorageClass> storage = llvm::None);
};

llvm::hash_code
hash_value(const StructType::MemberDecorationInfo &memberDecorationInfo);

// SPIR-V cooperative matrix type
class CooperativeMatrixNVType
    : public Type::TypeBase<CooperativeMatrixNVType, CompositeType,
                            detail::CooperativeMatrixTypeStorage> {
public:
  using Base::Base;

  static CooperativeMatrixNVType get(Type elementType, spirv::Scope scope,
                                     unsigned rows, unsigned columns);
  Type getElementType() const;

  /// Return the scope of the cooperative matrix.
  spirv::Scope getScope() const;
  /// return the number of rows of the matrix.
  unsigned getRows() const;
  /// return the number of columns of the matrix.
  unsigned getColumns() const;

  void getExtensions(SPIRVType::ExtensionArrayRefVector &extensions,
                     Optional<spirv::StorageClass> storage = llvm::None);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       Optional<spirv::StorageClass> storage = llvm::None);
};

// SPIR-V matrix type
class MatrixType : public Type::TypeBase<MatrixType, CompositeType,
                                         detail::MatrixTypeStorage> {
public:
  using Base::Base;

  static MatrixType get(Type columnType, uint32_t columnCount);

  static MatrixType getChecked(Type columnType, uint32_t columnCount,
                               Location location);

  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    Type columnType,
                                                    uint32_t columnCount);

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
                     Optional<spirv::StorageClass> storage = llvm::None);
  void getCapabilities(SPIRVType::CapabilityArrayRefVector &capabilities,
                       Optional<spirv::StorageClass> storage = llvm::None);
};

} // end namespace spirv
} // end namespace mlir

#endif // MLIR_DIALECT_SPIRV_SPIRVTYPES_H_
