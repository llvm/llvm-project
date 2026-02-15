//===- BuiltinAttributes.h - MLIR Builtin Attribute Classes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINATTRIBUTES_H
#define MLIR_IR_BUILTINATTRIBUTES_H

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Sequence.h"
#include <complex>
#include <optional>

namespace mlir {
class AffineMap;
class AsmResourceBlob;
class BoolAttr;
class BuiltinDialect;
class DenseIntElementsAttr;
template <typename T>
struct DialectResourceBlobHandle;
class FlatSymbolRefAttr;
class FunctionType;
class IntegerSet;
class IntegerType;
class Location;
class Operation;
class RankedTensorType;

namespace detail {
struct DenseElementsAttrStorage;
struct DenseStringElementsAttrStorage;
struct StringAttrStorage;
} // namespace detail

//===----------------------------------------------------------------------===//
// Elements Attributes
//===----------------------------------------------------------------------===//

namespace detail {
/// Pair of raw pointer and a boolean flag of whether the pointer holds a splat,
using DenseIterPtrAndSplat = std::pair<const char *, bool>;

/// Impl iterator for indexed DenseElementsAttr iterators that records a data
/// pointer and data index that is adjusted for the case of a splat attribute.
template <typename ConcreteT, typename T, typename PointerT = T *,
          typename ReferenceT = T &>
class DenseElementIndexedIteratorImpl
    : public llvm::indexed_accessor_iterator<ConcreteT, DenseIterPtrAndSplat, T,
                                             PointerT, ReferenceT> {
protected:
  DenseElementIndexedIteratorImpl(const char *data, bool isSplat,
                                  size_t dataIndex)
      : llvm::indexed_accessor_iterator<ConcreteT, DenseIterPtrAndSplat, T,
                                        PointerT, ReferenceT>({data, isSplat},
                                                              dataIndex) {}

  /// Return the current index for this iterator, adjusted for the case of a
  /// splat.
  ptrdiff_t getDataIndex() const {
    bool isSplat = this->base.second;
    return isSplat ? 0 : this->index;
  }

  /// Return the data base pointer.
  const char *getData() const { return this->base.first; }
};

/// Type trait detector that checks if a given type T is a complex type.
template <typename T>
struct is_complex_t : public std::false_type {};
template <typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};
} // namespace detail

} // namespace mlir

// DenseResourceElementsHandle is used by the generated
// DenseResourceElementsAttr.
namespace mlir {
using DenseResourceElementsHandle = DialectResourceBlobHandle<BuiltinDialect>;
} // namespace mlir

// DenseElementsAttr is defined in TableGen (see Builtin_DenseElementsAttr in
// BuiltinAttributes.td) and generated in BuiltinAttributes.h.inc below.
#define GET_ATTRDEF_CLASSES
#include "mlir/IR/BuiltinAttributes.h.inc"

// Template method definitions for DenseElementsAttr (declared in TableGen).
namespace mlir {
template <typename T, typename>
DenseElementsAttr DenseElementsAttr::get(const ShapedType &type,
                                         ArrayRef<T> values) {
  const char *data = reinterpret_cast<const char *>(values.data());
  return getRawIntOrFloat(type, ArrayRef<char>(data, values.size() * sizeof(T)),
                          sizeof(T), std::numeric_limits<T>::is_integer,
                          std::numeric_limits<T>::is_signed);
}
template <typename T, typename ElementT, typename>
DenseElementsAttr DenseElementsAttr::get(const ShapedType &type,
                                         ArrayRef<T> values) {
  const char *data = reinterpret_cast<const char *>(values.data());
  return getRawComplex(type, ArrayRef<char>(data, values.size() * sizeof(T)),
                       sizeof(T), std::numeric_limits<ElementT>::is_integer,
                       std::numeric_limits<ElementT>::is_signed);
}
template <typename T>
std::enable_if_t<!std::is_base_of<Attribute, T>::value ||
                     std::is_same<Attribute, T>::value,
                 T>
DenseElementsAttr::getSplatValue() const {
  assert(isSplat() && "expected the attribute to be a splat");
  return *value_begin<T>();
}
template <typename T>
auto DenseElementsAttr::try_value_begin() const {
  auto range = tryGetValues<T>();
  using iterator = decltype(range->begin());
  return failed(range) ? FailureOr<iterator>(failure()) : range->begin();
}
template <typename T>
auto DenseElementsAttr::try_value_end() const {
  auto range = tryGetValues<T>();
  using iterator = decltype(range->begin());
  return failed(range) ? FailureOr<iterator>(failure()) : range->end();
}
template <typename T>
auto DenseElementsAttr::getValues() const {
  auto range = tryGetValues<T>();
  assert(succeeded(range) && "element type cannot be iterated");
  return std::move(*range);
}

// tryGetValues template definitions (required for instantiation in other TUs).
template <typename T, typename>
FailureOr<DenseElementsAttr::iterator_range_impl<
    DenseElementsAttr::ElementIterator<T>>>
DenseElementsAttr::tryGetValues() const {
  if (!isValidIntOrFloat(sizeof(T), std::numeric_limits<T>::is_integer,
                         std::numeric_limits<T>::is_signed))
    return failure();
  const char *rawData = getRawData().data();
  bool splat = isSplat();
  return iterator_range_impl<ElementIterator<T>>(
      getType(), ElementIterator<T>(rawData, splat, 0),
      ElementIterator<T>(rawData, splat, getNumElements()));
}
template <typename T, typename ElementT, typename>
FailureOr<DenseElementsAttr::iterator_range_impl<
    DenseElementsAttr::ElementIterator<T>>>
DenseElementsAttr::tryGetValues() const {
  if (!isValidComplex(sizeof(T), std::numeric_limits<ElementT>::is_integer,
                      std::numeric_limits<ElementT>::is_signed))
    return failure();
  const char *rawData = getRawData().data();
  bool splat = isSplat();
  return iterator_range_impl<ElementIterator<T>>(
      getType(), ElementIterator<T>(rawData, splat, 0),
      ElementIterator<T>(rawData, splat, getNumElements()));
}
template <typename T, typename>
FailureOr<DenseElementsAttr::iterator_range_impl<
    DenseElementsAttr::AttributeElementIterator>>
DenseElementsAttr::tryGetValues() const {
  return iterator_range_impl<AttributeElementIterator>(
      getType(), AttributeElementIterator(Attribute(*this), 0),
      AttributeElementIterator(Attribute(*this), getNumElements()));
}
template <typename T, typename>
FailureOr<DenseElementsAttr::iterator_range_impl<
    DenseElementsAttr::DerivedAttributeElementIterator<T>>>
DenseElementsAttr::tryGetValues() const {
  using DerivedIterT = DerivedAttributeElementIterator<T>;
  return iterator_range_impl<DerivedIterT>(
      getType(), DerivedIterT(value_begin<Attribute>()),
      DerivedIterT(value_end<Attribute>()));
}
template <typename T, typename>
FailureOr<DenseElementsAttr::iterator_range_impl<
    DenseElementsAttr::BoolElementIterator>>
DenseElementsAttr::tryGetValues() const {
  if (!isValidBool())
    return failure();
  return iterator_range_impl<BoolElementIterator>(
      getType(), BoolElementIterator(*this, 0),
      BoolElementIterator(*this, getNumElements()));
}
template <typename T, typename>
FailureOr<DenseElementsAttr::iterator_range_impl<
    DenseElementsAttr::IntElementIterator>>
DenseElementsAttr::tryGetValues() const {
  if (!getElementType().isIntOrIndex())
    return failure();
  return iterator_range_impl<IntElementIterator>(getType(), raw_int_begin(),
                                                 raw_int_end());
}

/// An attribute that represents a reference to a splat vector or tensor
/// constant, meaning all of the elements have the same value.
class SplatElementsAttr : public DenseElementsAttr {
public:
  using DenseElementsAttr::DenseElementsAttr;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr) {
    auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(attr);
    return denseAttr && denseAttr.isSplat();
  }
};

} // namespace mlir

//===----------------------------------------------------------------------===//
// C++ Attribute Declarations
//===----------------------------------------------------------------------===//

namespace mlir {
//===----------------------------------------------------------------------===//
// DenseArrayAttr
//===----------------------------------------------------------------------===//

namespace detail {
/// Base class for DenseArrayAttr that is instantiated and specialized for each
/// supported element type below.
template <typename T>
class DenseArrayAttrImpl : public DenseArrayAttr {
public:
  using DenseArrayAttr::DenseArrayAttr;

  /// Implicit conversion to ArrayRef<T>.
  operator ArrayRef<T>() const;
  ArrayRef<T> asArrayRef() const { return ArrayRef<T>{*this}; }

  /// Random access to elements.
  T operator[](std::size_t index) const { return asArrayRef()[index]; }

  /// Builder from ArrayRef<T>.
  static DenseArrayAttrImpl get(MLIRContext *context, ArrayRef<T> content);

  /// Print the short form `[42, 100, -1]` without any type prefix.
  void print(AsmPrinter &printer) const;
  void print(raw_ostream &os) const;
  /// Print the short form `42, 100, -1` without any braces or type prefix.
  void printWithoutBraces(raw_ostream &os) const;

  /// Parse the short form `[42, 100, -1]` without any type prefix.
  static Attribute parse(AsmParser &parser, Type type);

  /// Parse the short form `42, 100, -1` without any type prefix or braces.
  static Attribute parseWithoutBraces(AsmParser &parser, Type type);

  /// Support for isa<>/cast<>.
  static bool classof(Attribute attr);
};

extern template class DenseArrayAttrImpl<bool>;
extern template class DenseArrayAttrImpl<int8_t>;
extern template class DenseArrayAttrImpl<int16_t>;
extern template class DenseArrayAttrImpl<int32_t>;
extern template class DenseArrayAttrImpl<int64_t>;
extern template class DenseArrayAttrImpl<float>;
extern template class DenseArrayAttrImpl<double>;
} // namespace detail

// Public name for all the supported DenseArrayAttr
using DenseBoolArrayAttr = detail::DenseArrayAttrImpl<bool>;
using DenseI8ArrayAttr = detail::DenseArrayAttrImpl<int8_t>;
using DenseI16ArrayAttr = detail::DenseArrayAttrImpl<int16_t>;
using DenseI32ArrayAttr = detail::DenseArrayAttrImpl<int32_t>;
using DenseI64ArrayAttr = detail::DenseArrayAttrImpl<int64_t>;
using DenseF32ArrayAttr = detail::DenseArrayAttrImpl<float>;
using DenseF64ArrayAttr = detail::DenseArrayAttrImpl<double>;

//===----------------------------------------------------------------------===//
// DenseResourceElementsAttr
//===----------------------------------------------------------------------===//

namespace detail {
/// Base class for DenseResourceElementsAttr that is instantiated and
/// specialized for each supported element type below.
template <typename T>
class DenseResourceElementsAttrBase : public DenseResourceElementsAttr {
public:
  using DenseResourceElementsAttr::DenseResourceElementsAttr;

  /// A builder that inserts a new resource using the provided blob. The handle
  /// of the inserted blob is used when building the attribute. The provided
  /// `blobName` is used as a hint for the key of the new handle for the `blob`
  /// resource, but may be changed if necessary to ensure uniqueness during
  /// insertion.
  static DenseResourceElementsAttrBase<T>
  get(ShapedType type, StringRef blobName, AsmResourceBlob blob);

  /// Return the data of this attribute as an ArrayRef<T> if it is present,
  /// returns std::nullopt otherwise.
  std::optional<ArrayRef<T>> tryGetAsArrayRef() const;

  /// Support for isa<>/cast<>.
  static bool classof(Attribute attr);
};

extern template class DenseResourceElementsAttrBase<bool>;
extern template class DenseResourceElementsAttrBase<int8_t>;
extern template class DenseResourceElementsAttrBase<int16_t>;
extern template class DenseResourceElementsAttrBase<int32_t>;
extern template class DenseResourceElementsAttrBase<int64_t>;
extern template class DenseResourceElementsAttrBase<uint8_t>;
extern template class DenseResourceElementsAttrBase<uint16_t>;
extern template class DenseResourceElementsAttrBase<uint32_t>;
extern template class DenseResourceElementsAttrBase<uint64_t>;
extern template class DenseResourceElementsAttrBase<float>;
extern template class DenseResourceElementsAttrBase<double>;
} // namespace detail

// Public names for all the supported DenseResourceElementsAttr.

using DenseBoolResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<bool>;
using DenseI8ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<int8_t>;
using DenseI16ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<int16_t>;
using DenseI32ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<int32_t>;
using DenseI64ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<int64_t>;
using DenseUI8ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<uint8_t>;
using DenseUI16ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<uint16_t>;
using DenseUI32ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<uint32_t>;
using DenseUI64ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<uint64_t>;
using DenseF32ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<float>;
using DenseF64ResourceElementsAttr =
    detail::DenseResourceElementsAttrBase<double>;

//===----------------------------------------------------------------------===//
// BoolAttr
//===----------------------------------------------------------------------===//

/// Special case of IntegerAttr to represent boolean integers, i.e., signless i1
/// integers.
class BoolAttr : public Attribute {
public:
  using Attribute::Attribute;
  using ValueType = bool;

  static BoolAttr get(MLIRContext *context, bool value);

  /// Enable conversion to IntegerAttr and its interfaces. This uses conversion
  /// vs. inheritance to avoid bringing in all of IntegerAttrs methods.
  operator IntegerAttr() const { return IntegerAttr(impl); }
  operator TypedAttr() const { return IntegerAttr(impl); }

  /// Return the boolean value of this attribute.
  bool getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// FlatSymbolRefAttr
//===----------------------------------------------------------------------===//

/// A symbol reference with a reference path containing a single element. This
/// is used to refer to an operation within the current symbol table.
class FlatSymbolRefAttr : public SymbolRefAttr {
public:
  using SymbolRefAttr::SymbolRefAttr;
  using ValueType = StringRef;

  /// Construct a symbol reference for the given value name.
  static FlatSymbolRefAttr get(StringAttr value) {
    return SymbolRefAttr::get(value);
  }
  static FlatSymbolRefAttr get(MLIRContext *ctx, StringRef value) {
    return SymbolRefAttr::get(ctx, value);
  }

  /// Convenience getter for building a SymbolRefAttr based on an operation
  /// that implements the SymbolTrait.
  static FlatSymbolRefAttr get(Operation *symbol) {
    return SymbolRefAttr::get(symbol);
  }

  /// Returns the name of the held symbol reference as a StringAttr.
  StringAttr getAttr() const { return getRootReference(); }

  /// Returns the name of the held symbol reference.
  StringRef getValue() const { return getAttr().getValue(); }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Attribute attr) {
    SymbolRefAttr refAttr = llvm::dyn_cast<SymbolRefAttr>(attr);
    return refAttr && refAttr.getNestedReferences().empty();
  }

private:
  using SymbolRefAttr::get;
  using SymbolRefAttr::getNestedReferences;
};

//===----------------------------------------------------------------------===//
// DenseFPElementsAttr
//===----------------------------------------------------------------------===//

/// An attribute that represents a reference to a dense float vector or tensor
/// object. Each element is stored as a double.
class DenseFPElementsAttr : public DenseElementsAttr {
public:
  using iterator = DenseElementsAttr::FloatElementIterator;

  using DenseElementsAttr::DenseElementsAttr;

  /// Get an instance of a DenseFPElementsAttr with the given arguments. This
  /// simply wraps the DenseElementsAttr::get calls.
  template <typename Arg>
  static DenseFPElementsAttr get(const ShapedType &type, Arg &&arg) {
    return llvm::cast<DenseFPElementsAttr>(
        DenseElementsAttr::get(type, llvm::ArrayRef(arg)));
  }
  template <typename T>
  static DenseFPElementsAttr get(const ShapedType &type,
                                 const std::initializer_list<T> &list) {
    return llvm::cast<DenseFPElementsAttr>(DenseElementsAttr::get(type, list));
  }

  /// Generates a new DenseElementsAttr by mapping each value attribute, and
  /// constructing the DenseElementsAttr given the new element type.
  DenseElementsAttr
  mapValues(Type newElementType,
            function_ref<APInt(const APFloat &)> mapping) const;

  /// Iterator access to the float element values.
  iterator begin() const { return tryGetFloatValues()->begin(); }
  iterator end() const { return tryGetFloatValues()->end(); }

  /// Method for supporting type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// DenseIntElementsAttr
//===----------------------------------------------------------------------===//

/// An attribute that represents a reference to a dense integer vector or tensor
/// object.
class DenseIntElementsAttr : public DenseElementsAttr {
public:
  /// DenseIntElementsAttr iterates on APInt, so we can use the raw element
  /// iterator directly.
  using iterator = DenseElementsAttr::IntElementIterator;

  using DenseElementsAttr::DenseElementsAttr;

  /// Get an instance of a DenseIntElementsAttr with the given arguments. This
  /// simply wraps the DenseElementsAttr::get calls.
  template <typename Arg>
  static DenseIntElementsAttr get(const ShapedType &type, Arg &&arg) {
    return llvm::cast<DenseIntElementsAttr>(
        DenseElementsAttr::get(type, llvm::ArrayRef(arg)));
  }
  template <typename T>
  static DenseIntElementsAttr get(const ShapedType &type,
                                  const std::initializer_list<T> &list) {
    return llvm::cast<DenseIntElementsAttr>(DenseElementsAttr::get(type, list));
  }

  /// Generates a new DenseElementsAttr by mapping each value attribute, and
  /// constructing the DenseElementsAttr given the new element type.
  DenseElementsAttr mapValues(Type newElementType,
                              function_ref<APInt(const APInt &)> mapping) const;

  /// Iterator access to the integer element values.
  iterator begin() const { return raw_int_begin(); }
  iterator end() const { return raw_int_end(); }

  /// Method for supporting type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// SparseElementsAttr
//===----------------------------------------------------------------------===//

template <typename T>
auto SparseElementsAttr::try_value_begin_impl(OverloadToken<T>) const
    -> FailureOr<iterator<T>> {
  auto zeroValue = getZeroValue<T>();
  auto valueIt = getValues().try_value_begin<T>();
  if (failed(valueIt))
    return failure();
  const SmallVector<ptrdiff_t> flatSparseIndices(getFlattenedSparseIndices());
  std::function<T(ptrdiff_t)> mapFn =
      [flatSparseIndices{flatSparseIndices}, valueIt{std::move(*valueIt)},
       zeroValue{std::move(zeroValue)}](ptrdiff_t index) {
        // Try to map the current index to one of the sparse indices.
        for (unsigned i = 0, e = flatSparseIndices.size(); i != e; ++i)
          if (flatSparseIndices[i] == index)
            return *std::next(valueIt, i);
        // Otherwise, return the zero value.
        return zeroValue;
      };
  return iterator<T>(llvm::seq<ptrdiff_t>(0, getNumElements()).begin(), mapFn);
}

//===----------------------------------------------------------------------===//
// DistinctAttr
//===----------------------------------------------------------------------===//

namespace detail {
struct DistinctAttrStorage;
class DistinctAttributeUniquer;
} // namespace detail

/// An attribute that associates a referenced attribute with a unique
/// identifier. Every call to the create function allocates a new distinct
/// attribute instance. The address of the attribute instance serves as a
/// temporary identifier. Similar to the names of SSA values, the final
/// identifiers are generated during pretty printing. This delayed numbering
/// ensures the printed identifiers are deterministic even if multiple distinct
/// attribute instances are created in-parallel.
///
/// Examples:
///
/// #distinct = distinct[0]<42.0 : f32>
/// #distinct1 = distinct[1]<42.0 : f32>
/// #distinct2 = distinct[2]<array<i32: 10, 42>>
///
/// NOTE: The distinct attribute cannot be defined using ODS since it uses a
/// custom distinct attribute uniquer that cannot be set from ODS.
class DistinctAttr
    : public detail::StorageUserBase<DistinctAttr, Attribute,
                                     detail::DistinctAttrStorage,
                                     detail::DistinctAttributeUniquer> {
public:
  using Base::Base;

  /// Returns the referenced attribute.
  Attribute getReferencedAttr() const;

  /// Creates a distinct attribute that associates a referenced attribute with a
  /// unique identifier.
  static DistinctAttr create(Attribute referencedAttr);

  static constexpr StringLiteral name = "builtin.distinct";
};

//===----------------------------------------------------------------------===//
// StringAttr
//===----------------------------------------------------------------------===//

/// Define comparisons for StringAttr against nullptr and itself to avoid the
/// StringRef overloads from being chosen when not desirable.
inline bool operator==(StringAttr lhs, std::nullptr_t) { return !lhs; }
inline bool operator!=(StringAttr lhs, std::nullptr_t) {
  return static_cast<bool>(lhs);
}
inline bool operator==(StringAttr lhs, StringAttr rhs) {
  return (Attribute)lhs == (Attribute)rhs;
}
inline bool operator!=(StringAttr lhs, StringAttr rhs) { return !(lhs == rhs); }

/// Allow direct comparison with StringRef.
inline bool operator==(StringAttr lhs, StringRef rhs) {
  return lhs.getValue() == rhs;
}
inline bool operator!=(StringAttr lhs, StringRef rhs) { return !(lhs == rhs); }
inline bool operator==(StringRef lhs, StringAttr rhs) {
  return rhs.getValue() == lhs;
}
inline bool operator!=(StringRef lhs, StringAttr rhs) { return !(lhs == rhs); }

} // namespace mlir

//===----------------------------------------------------------------------===//
// Attribute Utilities
//===----------------------------------------------------------------------===//

namespace mlir {

/// Given a list of strides (in which ShapedType::kDynamic
/// represents a dynamic value), return the single result AffineMap which
/// represents the linearized strided layout map. Dimensions correspond to the
/// offset followed by the strides in order. Symbols are inserted for each
/// dynamic dimension in order. A stride is always positive.
///
/// Examples:
/// =========
///
///   1. For offset: 0 strides: ?, ?, 1 return
///         (i, j, k)[M, N]->(M * i + N * j + k)
///
///   2. For offset: 3 strides: 32, ?, 16 return
///         (i, j, k)[M]->(3 + 32 * i + M * j + 16 * k)
///
///   3. For offset: ? strides: ?, ?, ? return
///         (i, j, k)[off, M, N, P]->(off + M * i + N * j + P * k)
AffineMap makeStridedLinearLayoutMap(ArrayRef<int64_t> strides, int64_t offset,
                                     MLIRContext *context);

} // namespace mlir

namespace llvm {

template <>
struct DenseMapInfo<mlir::StringAttr> : public DenseMapInfo<mlir::Attribute> {
  static mlir::StringAttr getEmptyKey() {
    const void *pointer = llvm::DenseMapInfo<const void *>::getEmptyKey();
    return mlir::StringAttr::getFromOpaquePointer(pointer);
  }
  static mlir::StringAttr getTombstoneKey() {
    const void *pointer = llvm::DenseMapInfo<const void *>::getTombstoneKey();
    return mlir::StringAttr::getFromOpaquePointer(pointer);
  }
};
template <>
struct PointerLikeTypeTraits<mlir::StringAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline mlir::StringAttr getFromVoidPointer(void *p) {
    return mlir::StringAttr::getFromOpaquePointer(p);
  }
};

template <>
struct PointerLikeTypeTraits<mlir::IntegerAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline mlir::IntegerAttr getFromVoidPointer(void *p) {
    return mlir::IntegerAttr::getFromOpaquePointer(p);
  }
};

template <>
struct PointerLikeTypeTraits<mlir::SymbolRefAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline mlir::SymbolRefAttr getFromVoidPointer(void *ptr) {
    return mlir::SymbolRefAttr::getFromOpaquePointer(ptr);
  }
};

} // namespace llvm

#endif // MLIR_IR_BUILTINATTRIBUTES_H
