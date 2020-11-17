//===- Attributes.h - MLIR Attribute Classes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_ATTRIBUTES_H
#define MLIR_IR_ATTRIBUTES_H

#include "mlir/IR/AttributeSupport.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include <complex>

namespace mlir {
class AffineMap;
class Dialect;
class FunctionType;
class Identifier;
class IntegerSet;
class Location;
class MLIRContext;
class ShapedType;
class Type;

namespace detail {

struct AffineMapAttributeStorage;
struct ArrayAttributeStorage;
struct DictionaryAttributeStorage;
struct IntegerAttributeStorage;
struct IntegerSetAttributeStorage;
struct FloatAttributeStorage;
struct OpaqueAttributeStorage;
struct StringAttributeStorage;
struct SymbolRefAttributeStorage;
struct TypeAttributeStorage;

/// Elements Attributes.
struct DenseIntOrFPElementsAttributeStorage;
struct DenseStringElementsAttributeStorage;
struct OpaqueElementsAttributeStorage;
struct SparseElementsAttributeStorage;
} // namespace detail

/// Attributes are known-constant values of operations and functions.
///
/// Instances of the Attribute class are references to immortal key-value pairs
/// with immutable, uniqued key owned by MLIRContext. As such, an Attribute is a
/// thin wrapper around an underlying storage pointer. Attributes are usually
/// passed by value.
class Attribute {
public:
  /// Utility class for implementing attributes.
  template <typename ConcreteType, typename BaseType, typename StorageType,
            template <typename T> class... Traits>
  using AttrBase = detail::StorageUserBase<ConcreteType, BaseType, StorageType,
                                           detail::AttributeUniquer, Traits...>;

  using ImplType = AttributeStorage;
  using ValueType = void;

  constexpr Attribute() : impl(nullptr) {}
  /* implicit */ Attribute(const ImplType *impl)
      : impl(const_cast<ImplType *>(impl)) {}

  Attribute(const Attribute &other) = default;
  Attribute &operator=(const Attribute &other) = default;

  bool operator==(Attribute other) const { return impl == other.impl; }
  bool operator!=(Attribute other) const { return !(*this == other); }
  explicit operator bool() const { return impl; }

  bool operator!() const { return impl == nullptr; }

  template <typename U> bool isa() const;
  template <typename First, typename Second, typename... Rest>
  bool isa() const;
  template <typename U> U dyn_cast() const;
  template <typename U> U dyn_cast_or_null() const;
  template <typename U> U cast() const;

  // Support dyn_cast'ing Attribute to itself.
  static bool classof(Attribute) { return true; }

  /// Return a unique identifier for the concrete attribute type. This is used
  /// to support dynamic type casting.
  TypeID getTypeID() { return impl->getAbstractAttribute().getTypeID(); }

  /// Return the type of this attribute.
  Type getType() const;

  /// Return the context this attribute belongs to.
  MLIRContext *getContext() const;

  /// Get the dialect this attribute is registered to.
  Dialect &getDialect() const;

  /// Print the attribute.
  void print(raw_ostream &os) const;
  void dump() const;

  /// Get an opaque pointer to the attribute.
  const void *getAsOpaquePointer() const { return impl; }
  /// Construct an attribute from the opaque pointer representation.
  static Attribute getFromOpaquePointer(const void *ptr) {
    return Attribute(reinterpret_cast<const ImplType *>(ptr));
  }

  friend ::llvm::hash_code hash_value(Attribute arg);

  /// Return the abstract descriptor for this attribute.
  const AbstractAttribute &getAbstractAttribute() const {
    return impl->getAbstractAttribute();
  }

protected:
  ImplType *impl;
};

inline raw_ostream &operator<<(raw_ostream &os, Attribute attr) {
  attr.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// AttributeTraitBase
//===----------------------------------------------------------------------===//

namespace AttributeTrait {
/// This class represents the base of an attribute trait.
template <typename ConcreteType, template <typename> class TraitType>
using TraitBase = detail::StorageUserTraitBase<ConcreteType, TraitType>;
} // namespace AttributeTrait

//===----------------------------------------------------------------------===//
// AttributeInterface
//===----------------------------------------------------------------------===//

/// This class represents the base of an attribute interface. See the definition
/// of `detail::Interface` for requirements on the `Traits` type.
template <typename ConcreteType, typename Traits>
class AttributeInterface
    : public detail::Interface<ConcreteType, Attribute, Traits, Attribute,
                               AttributeTrait::TraitBase> {
public:
  using Base = AttributeInterface<ConcreteType, Traits>;
  using InterfaceBase = detail::Interface<ConcreteType, Attribute, Traits,
                                          Attribute, AttributeTrait::TraitBase>;
  using InterfaceBase::InterfaceBase;

private:
  /// Returns the impl interface instance for the given type.
  static typename InterfaceBase::Concept *getInterfaceFor(Attribute attr) {
    return attr.getAbstractAttribute().getInterface<ConcreteType>();
  }

  /// Allow access to 'getInterfaceFor'.
  friend InterfaceBase;
};

//===----------------------------------------------------------------------===//
// AffineMapAttr
//===----------------------------------------------------------------------===//

class AffineMapAttr
    : public Attribute::AttrBase<AffineMapAttr, Attribute,
                                 detail::AffineMapAttributeStorage> {
public:
  using Base::Base;
  using ValueType = AffineMap;

  static AffineMapAttr get(AffineMap value);

  AffineMap getValue() const;
};

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//

/// Array attributes are lists of other attributes.  They are not necessarily
/// type homogenous given that attributes don't, in general, carry types.
class ArrayAttr : public Attribute::AttrBase<ArrayAttr, Attribute,
                                             detail::ArrayAttributeStorage> {
public:
  using Base::Base;
  using ValueType = ArrayRef<Attribute>;

  static ArrayAttr get(ArrayRef<Attribute> value, MLIRContext *context);

  ArrayRef<Attribute> getValue() const;
  Attribute operator[](unsigned idx) const;

  /// Support range iteration.
  using iterator = llvm::ArrayRef<Attribute>::iterator;
  iterator begin() const { return getValue().begin(); }
  iterator end() const { return getValue().end(); }
  size_t size() const { return getValue().size(); }
  bool empty() const { return size() == 0; }

private:
  /// Class for underlying value iterator support.
  template <typename AttrTy>
  class attr_value_iterator final
      : public llvm::mapped_iterator<ArrayAttr::iterator,
                                     AttrTy (*)(Attribute)> {
  public:
    explicit attr_value_iterator(ArrayAttr::iterator it)
        : llvm::mapped_iterator<ArrayAttr::iterator, AttrTy (*)(Attribute)>(
              it, [](Attribute attr) { return attr.cast<AttrTy>(); }) {}
    AttrTy operator*() const { return (*this->I).template cast<AttrTy>(); }
  };

public:
  template <typename AttrTy>
  iterator_range<attr_value_iterator<AttrTy>> getAsRange() {
    return llvm::make_range(attr_value_iterator<AttrTy>(begin()),
                            attr_value_iterator<AttrTy>(end()));
  }
  template <typename AttrTy, typename UnderlyingTy = typename AttrTy::ValueType>
  auto getAsValueRange() {
    return llvm::map_range(getAsRange<AttrTy>(), [](AttrTy attr) {
      return static_cast<UnderlyingTy>(attr.getValue());
    });
  }
};

//===----------------------------------------------------------------------===//
// DictionaryAttr
//===----------------------------------------------------------------------===//

/// NamedAttribute is used for dictionary attributes, it holds an identifier for
/// the name and a value for the attribute. The attribute pointer should always
/// be non-null.
using NamedAttribute = std::pair<Identifier, Attribute>;

bool operator<(const NamedAttribute &lhs, const NamedAttribute &rhs);
bool operator<(const NamedAttribute &lhs, StringRef rhs);

/// Dictionary attribute is an attribute that represents a sorted collection of
/// named attribute values. The elements are sorted by name, and each name must
/// be unique within the collection.
class DictionaryAttr
    : public Attribute::AttrBase<DictionaryAttr, Attribute,
                                 detail::DictionaryAttributeStorage> {
public:
  using Base::Base;
  using ValueType = ArrayRef<NamedAttribute>;

  /// Construct a dictionary attribute with the provided list of named
  /// attributes. This method assumes that the provided list is unordered. If
  /// the caller can guarantee that the attributes are ordered by name,
  /// getWithSorted should be used instead.
  static DictionaryAttr get(ArrayRef<NamedAttribute> value,
                            MLIRContext *context);

  /// Construct a dictionary with an array of values that is known to already be
  /// sorted by name and uniqued.
  static DictionaryAttr getWithSorted(ArrayRef<NamedAttribute> value,
                                      MLIRContext *context);

  ArrayRef<NamedAttribute> getValue() const;

  /// Return the specified attribute if present, null otherwise.
  Attribute get(StringRef name) const;
  Attribute get(Identifier name) const;

  /// Return the specified named attribute if present, None otherwise.
  Optional<NamedAttribute> getNamed(StringRef name) const;
  Optional<NamedAttribute> getNamed(Identifier name) const;

  /// Support range iteration.
  using iterator = llvm::ArrayRef<NamedAttribute>::iterator;
  iterator begin() const;
  iterator end() const;
  bool empty() const { return size() == 0; }
  size_t size() const;

  /// Sorts the NamedAttributes in the array ordered by name as expected by
  /// getWithSorted and returns whether the values were sorted.
  /// Requires: uniquely named attributes.
  static bool sort(ArrayRef<NamedAttribute> values,
                   SmallVectorImpl<NamedAttribute> &storage);

  /// Sorts the NamedAttributes in the array ordered by name as expected by
  /// getWithSorted in place on an array and returns whether the values needed
  /// to be sorted.
  /// Requires: uniquely named attributes.
  static bool sortInPlace(SmallVectorImpl<NamedAttribute> &array);

  /// Returns an entry with a duplicate name in `array`, if it exists, else
  /// returns llvm::None. If `isSorted` is true, the array is assumed to be
  /// sorted else it will be sorted in place before finding the duplicate entry.
  static Optional<NamedAttribute>
  findDuplicate(SmallVectorImpl<NamedAttribute> &array, bool isSorted);

private:
  /// Return empty dictionary.
  static DictionaryAttr getEmpty(MLIRContext *context);
};

//===----------------------------------------------------------------------===//
// FloatAttr
//===----------------------------------------------------------------------===//

class FloatAttr : public Attribute::AttrBase<FloatAttr, Attribute,
                                             detail::FloatAttributeStorage> {
public:
  using Base::Base;
  using ValueType = APFloat;

  /// Return a float attribute for the specified value in the specified type.
  /// These methods should only be used for simple constant values, e.g 1.0/2.0,
  /// that are known-valid both as host double and the 'type' format.
  static FloatAttr get(Type type, double value);
  static FloatAttr getChecked(Type type, double value, Location loc);

  /// Return a float attribute for the specified value in the specified type.
  static FloatAttr get(Type type, const APFloat &value);
  static FloatAttr getChecked(Type type, const APFloat &value, Location loc);

  APFloat getValue() const;

  /// This function is used to convert the value to a double, even if it loses
  /// precision.
  double getValueAsDouble() const;
  static double getValueAsDouble(APFloat val);

  /// Verify the construction invariants for a double value.
  static LogicalResult verifyConstructionInvariants(Location loc, Type type,
                                                    double value);
  static LogicalResult verifyConstructionInvariants(Location loc, Type type,
                                                    const APFloat &value);
};

//===----------------------------------------------------------------------===//
// IntegerAttr
//===----------------------------------------------------------------------===//

class IntegerAttr
    : public Attribute::AttrBase<IntegerAttr, Attribute,
                                 detail::IntegerAttributeStorage> {
public:
  using Base::Base;
  using ValueType = APInt;

  static IntegerAttr get(Type type, int64_t value);
  static IntegerAttr get(Type type, const APInt &value);

  APInt getValue() const;
  /// Return the integer value as a 64-bit int. The attribute must be a signless
  /// integer.
  // TODO: Change callers to use getValue instead.
  int64_t getInt() const;
  /// Return the integer value as a signed 64-bit int. The attribute must be
  /// a signed integer.
  int64_t getSInt() const;
  /// Return the integer value as a unsigned 64-bit int. The attribute must be
  /// an unsigned integer.
  uint64_t getUInt() const;

  static LogicalResult verifyConstructionInvariants(Location loc, Type type,
                                                    int64_t value);
  static LogicalResult verifyConstructionInvariants(Location loc, Type type,
                                                    const APInt &value);
};

//===----------------------------------------------------------------------===//
// BoolAttr

/// Special case of IntegerAttr to represent boolean integers, i.e., signless i1
/// integers.
class BoolAttr : public Attribute {
public:
  using Attribute::Attribute;
  using ValueType = bool;

  static BoolAttr get(bool value, MLIRContext *context);

  /// Enable conversion to IntegerAttr. This uses conversion vs. inheritance to
  /// avoid bringing in all of IntegerAttrs methods.
  operator IntegerAttr() const { return IntegerAttr(impl); }

  /// Return the boolean value of this attribute.
  bool getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// IntegerSetAttr
//===----------------------------------------------------------------------===//

class IntegerSetAttr
    : public Attribute::AttrBase<IntegerSetAttr, Attribute,
                                 detail::IntegerSetAttributeStorage> {
public:
  using Base::Base;
  using ValueType = IntegerSet;

  static IntegerSetAttr get(IntegerSet value);

  IntegerSet getValue() const;
};

//===----------------------------------------------------------------------===//
// OpaqueAttr
//===----------------------------------------------------------------------===//

/// Opaque attributes represent attributes of non-registered dialects. These are
/// attribute represented in their raw string form, and can only usefully be
/// tested for attribute equality.
class OpaqueAttr : public Attribute::AttrBase<OpaqueAttr, Attribute,
                                              detail::OpaqueAttributeStorage> {
public:
  using Base::Base;

  /// Get or create a new OpaqueAttr with the provided dialect and string data.
  static OpaqueAttr get(Identifier dialect, StringRef attrData, Type type,
                        MLIRContext *context);

  /// Get or create a new OpaqueAttr with the provided dialect and string data.
  /// If the given identifier is not a valid namespace for a dialect, then a
  /// null attribute is returned.
  static OpaqueAttr getChecked(Identifier dialect, StringRef attrData,
                               Type type, Location location);

  /// Returns the dialect namespace of the opaque attribute.
  Identifier getDialectNamespace() const;

  /// Returns the raw attribute data of the opaque attribute.
  StringRef getAttrData() const;

  /// Verify the construction of an opaque attribute.
  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    Identifier dialect,
                                                    StringRef attrData,
                                                    Type type);
};

//===----------------------------------------------------------------------===//
// StringAttr
//===----------------------------------------------------------------------===//

class StringAttr : public Attribute::AttrBase<StringAttr, Attribute,
                                              detail::StringAttributeStorage> {
public:
  using Base::Base;
  using ValueType = StringRef;

  /// Get an instance of a StringAttr with the given string.
  static StringAttr get(StringRef bytes, MLIRContext *context);

  /// Get an instance of a StringAttr with the given string and Type.
  static StringAttr get(StringRef bytes, Type type);

  StringRef getValue() const;
};

//===----------------------------------------------------------------------===//
// SymbolRefAttr
//===----------------------------------------------------------------------===//

class FlatSymbolRefAttr;

/// A symbol reference attribute represents a symbolic reference to another
/// operation.
class SymbolRefAttr
    : public Attribute::AttrBase<SymbolRefAttr, Attribute,
                                 detail::SymbolRefAttributeStorage> {
public:
  using Base::Base;

  /// Construct a symbol reference for the given value name.
  static FlatSymbolRefAttr get(StringRef value, MLIRContext *ctx);

  /// Construct a symbol reference for the given value name, and a set of nested
  /// references that are further resolve to a nested symbol.
  static SymbolRefAttr get(StringRef value,
                           ArrayRef<FlatSymbolRefAttr> references,
                           MLIRContext *ctx);

  /// Returns the name of the top level symbol reference, i.e. the root of the
  /// reference path.
  StringRef getRootReference() const;

  /// Returns the name of the fully resolved symbol, i.e. the leaf of the
  /// reference path.
  StringRef getLeafReference() const;

  /// Returns the set of nested references representing the path to the symbol
  /// nested under the root reference.
  ArrayRef<FlatSymbolRefAttr> getNestedReferences() const;
};

/// A symbol reference with a reference path containing a single element. This
/// is used to refer to an operation within the current symbol table.
class FlatSymbolRefAttr : public SymbolRefAttr {
public:
  using SymbolRefAttr::SymbolRefAttr;
  using ValueType = StringRef;

  /// Construct a symbol reference for the given value name.
  static FlatSymbolRefAttr get(StringRef value, MLIRContext *ctx) {
    return SymbolRefAttr::get(value, ctx);
  }

  /// Returns the name of the held symbol reference.
  StringRef getValue() const { return getRootReference(); }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Attribute attr) {
    SymbolRefAttr refAttr = attr.dyn_cast<SymbolRefAttr>();
    return refAttr && refAttr.getNestedReferences().empty();
  }

private:
  using SymbolRefAttr::get;
  using SymbolRefAttr::getNestedReferences;
};

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

class TypeAttr : public Attribute::AttrBase<TypeAttr, Attribute,
                                            detail::TypeAttributeStorage> {
public:
  using Base::Base;
  using ValueType = Type;

  static TypeAttr get(Type value);

  Type getValue() const;
};

//===----------------------------------------------------------------------===//
// UnitAttr
//===----------------------------------------------------------------------===//

/// Unit attributes are attributes that hold no specific value and are given
/// meaning by their existence.
class UnitAttr
    : public Attribute::AttrBase<UnitAttr, Attribute, AttributeStorage> {
public:
  using Base::Base;

  static UnitAttr get(MLIRContext *context);
};

//===----------------------------------------------------------------------===//
// Elements Attributes
//===----------------------------------------------------------------------===//

namespace detail {
template <typename T> class ElementsAttrIterator;
template <typename T> class ElementsAttrRange;
} // namespace detail

/// A base attribute that represents a reference to a static shaped tensor or
/// vector constant.
class ElementsAttr : public Attribute {
public:
  using Attribute::Attribute;
  template <typename T> using iterator = detail::ElementsAttrIterator<T>;
  template <typename T> using iterator_range = detail::ElementsAttrRange<T>;

  /// Return the type of this ElementsAttr, guaranteed to be a vector or tensor
  /// with static shape.
  ShapedType getType() const;

  /// Return the value at the given index. The index is expected to refer to a
  /// valid element.
  Attribute getValue(ArrayRef<uint64_t> index) const;

  /// Return the value of type 'T' at the given index, where 'T' corresponds to
  /// an Attribute type.
  template <typename T> T getValue(ArrayRef<uint64_t> index) const {
    return getValue(index).template cast<T>();
  }

  /// Return the elements of this attribute as a value of type 'T'. Note:
  /// Aborts if the subclass is OpaqueElementsAttrs, these attrs do not support
  /// iteration.
  template <typename T> iterator_range<T> getValues() const;

  /// Return if the given 'index' refers to a valid element in this attribute.
  bool isValidIndex(ArrayRef<uint64_t> index) const;

  /// Returns the number of elements held by this attribute.
  int64_t getNumElements() const;

  /// Returns the number of elements held by this attribute.
  int64_t size() const { return getNumElements(); }

  /// Generates a new ElementsAttr by mapping each int value to a new
  /// underlying APInt. The new values can represent either an integer or float.
  /// This ElementsAttr should contain integers.
  ElementsAttr mapValues(Type newElementType,
                         function_ref<APInt(const APInt &)> mapping) const;

  /// Generates a new ElementsAttr by mapping each float value to a new
  /// underlying APInt. The new values can represent either an integer or float.
  /// This ElementsAttr should contain floats.
  ElementsAttr mapValues(Type newElementType,
                         function_ref<APInt(const APFloat &)> mapping) const;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr);

protected:
  /// Returns the 1 dimensional flattened row-major index from the given
  /// multi-dimensional index.
  uint64_t getFlattenedIndex(ArrayRef<uint64_t> index) const;
};

namespace detail {
/// DenseElementsAttr data is aligned to uint64_t, so this traits class is
/// necessary to interop with PointerIntPair.
class DenseElementDataPointerTypeTraits {
public:
  static inline const void *getAsVoidPointer(const char *ptr) { return ptr; }
  static inline const char *getFromVoidPointer(const void *ptr) {
    return static_cast<const char *>(ptr);
  }

  // Note: We could steal more bits if the need arises.
  static constexpr int NumLowBitsAvailable = 1;
};

/// Pair of raw pointer and a boolean flag of whether the pointer holds a splat,
using DenseIterPtrAndSplat =
    llvm::PointerIntPair<const char *, 1, bool,
                         DenseElementDataPointerTypeTraits>;

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
    bool isSplat = this->base.getInt();
    return isSplat ? 0 : this->index;
  }

  /// Return the data base pointer.
  const char *getData() const { return this->base.getPointer(); }
};

/// Type trait detector that checks if a given type T is a complex type.
template <typename T> struct is_complex_t : public std::false_type {};
template <typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};
} // namespace detail

/// An attribute that represents a reference to a dense vector or tensor object.
///
class DenseElementsAttr : public ElementsAttr {
public:
  using ElementsAttr::ElementsAttr;

  /// Type trait used to check if the given type T is a potentially valid C++
  /// floating point type that can be used to access the underlying element
  /// types of a DenseElementsAttr.
  // TODO: Use std::disjunction when C++17 is supported.
  template <typename T> struct is_valid_cpp_fp_type {
    /// The type is a valid floating point type if it is a builtin floating
    /// point type, or is a potentially user defined floating point type. The
    /// latter allows for supporting users that have custom types defined for
    /// bfloat16/half/etc.
    static constexpr bool value = llvm::is_one_of<T, float, double>::value ||
                                  (std::numeric_limits<T>::is_specialized &&
                                   !std::numeric_limits<T>::is_integer);
  };

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr);

  /// Constructs a dense elements attribute from an array of element values.
  /// Each element attribute value is expected to be an element of 'type'.
  /// 'type' must be a vector or tensor with static shape. If the element of
  /// `type` is non-integer/index/float it is assumed to be a string type.
  static DenseElementsAttr get(ShapedType type, ArrayRef<Attribute> values);

  /// Constructs a dense integer elements attribute from an array of integer
  /// or floating-point values. Each value is expected to be the same bitwidth
  /// of the element type of 'type'. 'type' must be a vector or tensor with
  /// static shape.
  template <typename T, typename = typename std::enable_if<
                            std::numeric_limits<T>::is_integer ||
                            is_valid_cpp_fp_type<T>::value>::type>
  static DenseElementsAttr get(const ShapedType &type, ArrayRef<T> values) {
    const char *data = reinterpret_cast<const char *>(values.data());
    return getRawIntOrFloat(
        type, ArrayRef<char>(data, values.size() * sizeof(T)), sizeof(T),
        std::numeric_limits<T>::is_integer, std::numeric_limits<T>::is_signed);
  }

  /// Constructs a dense integer elements attribute from a single element.
  template <typename T, typename = typename std::enable_if<
                            std::numeric_limits<T>::is_integer ||
                            is_valid_cpp_fp_type<T>::value ||
                            detail::is_complex_t<T>::value>::type>
  static DenseElementsAttr get(const ShapedType &type, T value) {
    return get(type, llvm::makeArrayRef(value));
  }

  /// Constructs a dense complex elements attribute from an array of complex
  /// values. Each value is expected to be the same bitwidth of the element type
  /// of 'type'. 'type' must be a vector or tensor with static shape.
  template <typename T, typename ElementT = typename T::value_type,
            typename = typename std::enable_if<
                detail::is_complex_t<T>::value &&
                (std::numeric_limits<ElementT>::is_integer ||
                 is_valid_cpp_fp_type<ElementT>::value)>::type>
  static DenseElementsAttr get(const ShapedType &type, ArrayRef<T> values) {
    const char *data = reinterpret_cast<const char *>(values.data());
    return getRawComplex(type, ArrayRef<char>(data, values.size() * sizeof(T)),
                         sizeof(T), std::numeric_limits<ElementT>::is_integer,
                         std::numeric_limits<ElementT>::is_signed);
  }

  /// Overload of the above 'get' method that is specialized for boolean values.
  static DenseElementsAttr get(ShapedType type, ArrayRef<bool> values);

  /// Overload of the above 'get' method that is specialized for StringRef
  /// values.
  static DenseElementsAttr get(ShapedType type, ArrayRef<StringRef> values);

  /// Constructs a dense integer elements attribute from an array of APInt
  /// values. Each APInt value is expected to have the same bitwidth as the
  /// element type of 'type'. 'type' must be a vector or tensor with static
  /// shape.
  static DenseElementsAttr get(ShapedType type, ArrayRef<APInt> values);

  /// Constructs a dense complex elements attribute from an array of APInt
  /// values. Each APInt value is expected to have the same bitwidth as the
  /// element type of 'type'. 'type' must be a vector or tensor with static
  /// shape.
  static DenseElementsAttr get(ShapedType type,
                               ArrayRef<std::complex<APInt>> values);

  /// Constructs a dense float elements attribute from an array of APFloat
  /// values. Each APFloat value is expected to have the same bitwidth as the
  /// element type of 'type'. 'type' must be a vector or tensor with static
  /// shape.
  static DenseElementsAttr get(ShapedType type, ArrayRef<APFloat> values);

  /// Constructs a dense complex elements attribute from an array of APFloat
  /// values. Each APFloat value is expected to have the same bitwidth as the
  /// element type of 'type'. 'type' must be a vector or tensor with static
  /// shape.
  static DenseElementsAttr get(ShapedType type,
                               ArrayRef<std::complex<APFloat>> values);

  /// Construct a dense elements attribute for an initializer_list of values.
  /// Each value is expected to be the same bitwidth of the element type of
  /// 'type'. 'type' must be a vector or tensor with static shape.
  template <typename T>
  static DenseElementsAttr get(const ShapedType &type,
                               const std::initializer_list<T> &list) {
    return get(type, ArrayRef<T>(list));
  }

  /// Construct a dense elements attribute from a raw buffer representing the
  /// data for this attribute. Users should generally not use this methods as
  /// the expected buffer format may not be a form the user expects.
  static DenseElementsAttr getFromRawBuffer(ShapedType type,
                                            ArrayRef<char> rawBuffer,
                                            bool isSplatBuffer);

  /// Returns true if the given buffer is a valid raw buffer for the given type.
  /// `detectedSplat` is set if the buffer is valid and represents a splat
  /// buffer.
  static bool isValidRawBuffer(ShapedType type, ArrayRef<char> rawBuffer,
                               bool &detectedSplat);

  //===--------------------------------------------------------------------===//
  // Iterators
  //===--------------------------------------------------------------------===//

  /// A utility iterator that allows walking over the internal Attribute values
  /// of a DenseElementsAttr.
  class AttributeElementIterator
      : public llvm::indexed_accessor_iterator<AttributeElementIterator,
                                               const void *, Attribute,
                                               Attribute, Attribute> {
  public:
    /// Accesses the Attribute value at this iterator position.
    Attribute operator*() const;

  private:
    friend DenseElementsAttr;

    /// Constructs a new iterator.
    AttributeElementIterator(DenseElementsAttr attr, size_t index);
  };

  /// Iterator for walking raw element values of the specified type 'T', which
  /// may be any c++ data type matching the stored representation: int32_t,
  /// float, etc.
  template <typename T>
  class ElementIterator
      : public detail::DenseElementIndexedIteratorImpl<ElementIterator<T>,
                                                       const T> {
  public:
    /// Accesses the raw value at this iterator position.
    const T &operator*() const {
      return reinterpret_cast<const T *>(this->getData())[this->getDataIndex()];
    }

  private:
    friend DenseElementsAttr;

    /// Constructs a new iterator.
    ElementIterator(const char *data, bool isSplat, size_t dataIndex)
        : detail::DenseElementIndexedIteratorImpl<ElementIterator<T>, const T>(
              data, isSplat, dataIndex) {}
  };

  /// A utility iterator that allows walking over the internal bool values.
  class BoolElementIterator
      : public detail::DenseElementIndexedIteratorImpl<BoolElementIterator,
                                                       bool, bool, bool> {
  public:
    /// Accesses the bool value at this iterator position.
    bool operator*() const;

  private:
    friend DenseElementsAttr;

    /// Constructs a new iterator.
    BoolElementIterator(DenseElementsAttr attr, size_t dataIndex);
  };

  /// A utility iterator that allows walking over the internal raw APInt values.
  class IntElementIterator
      : public detail::DenseElementIndexedIteratorImpl<IntElementIterator,
                                                       APInt, APInt, APInt> {
  public:
    /// Accesses the raw APInt value at this iterator position.
    APInt operator*() const;

  private:
    friend DenseElementsAttr;

    /// Constructs a new iterator.
    IntElementIterator(DenseElementsAttr attr, size_t dataIndex);

    /// The bitwidth of the element type.
    size_t bitWidth;
  };

  /// A utility iterator that allows walking over the internal raw complex APInt
  /// values.
  class ComplexIntElementIterator
      : public detail::DenseElementIndexedIteratorImpl<
            ComplexIntElementIterator, std::complex<APInt>, std::complex<APInt>,
            std::complex<APInt>> {
  public:
    /// Accesses the raw std::complex<APInt> value at this iterator position.
    std::complex<APInt> operator*() const;

  private:
    friend DenseElementsAttr;

    /// Constructs a new iterator.
    ComplexIntElementIterator(DenseElementsAttr attr, size_t dataIndex);

    /// The bitwidth of the element type.
    size_t bitWidth;
  };

  /// Iterator for walking over APFloat values.
  class FloatElementIterator final
      : public llvm::mapped_iterator<IntElementIterator,
                                     std::function<APFloat(const APInt &)>> {
    friend DenseElementsAttr;

    /// Initializes the float element iterator to the specified iterator.
    FloatElementIterator(const llvm::fltSemantics &smt, IntElementIterator it);

  public:
    using reference = APFloat;
  };

  /// Iterator for walking over complex APFloat values.
  class ComplexFloatElementIterator final
      : public llvm::mapped_iterator<
            ComplexIntElementIterator,
            std::function<std::complex<APFloat>(const std::complex<APInt> &)>> {
    friend DenseElementsAttr;

    /// Initializes the float element iterator to the specified iterator.
    ComplexFloatElementIterator(const llvm::fltSemantics &smt,
                                ComplexIntElementIterator it);

  public:
    using reference = std::complex<APFloat>;
  };

  //===--------------------------------------------------------------------===//
  // Value Querying
  //===--------------------------------------------------------------------===//

  /// Returns true if this attribute corresponds to a splat, i.e. if all element
  /// values are the same.
  bool isSplat() const;

  /// Return the splat value for this attribute. This asserts that the attribute
  /// corresponds to a splat.
  Attribute getSplatValue() const { return getSplatValue<Attribute>(); }
  template <typename T>
  typename std::enable_if<!std::is_base_of<Attribute, T>::value ||
                              std::is_same<Attribute, T>::value,
                          T>::type
  getSplatValue() const {
    assert(isSplat() && "expected the attribute to be a splat");
    return *getValues<T>().begin();
  }
  /// Return the splat value for derived attribute element types.
  template <typename T>
  typename std::enable_if<std::is_base_of<Attribute, T>::value &&
                              !std::is_same<Attribute, T>::value,
                          T>::type
  getSplatValue() const {
    return getSplatValue().template cast<T>();
  }

  /// Return the value at the given index. The 'index' is expected to refer to a
  /// valid element.
  Attribute getValue(ArrayRef<uint64_t> index) const {
    return getValue<Attribute>(index);
  }
  template <typename T> T getValue(ArrayRef<uint64_t> index) const {
    // Skip to the element corresponding to the flattened index.
    return *std::next(getValues<T>().begin(), getFlattenedIndex(index));
  }

  /// Return the held element values as a range of integer or floating-point
  /// values.
  template <typename T, typename = typename std::enable_if<
                            (!std::is_same<T, bool>::value &&
                             std::numeric_limits<T>::is_integer) ||
                            is_valid_cpp_fp_type<T>::value>::type>
  llvm::iterator_range<ElementIterator<T>> getValues() const {
    assert(isValidIntOrFloat(sizeof(T), std::numeric_limits<T>::is_integer,
                             std::numeric_limits<T>::is_signed));
    const char *rawData = getRawData().data();
    bool splat = isSplat();
    return {ElementIterator<T>(rawData, splat, 0),
            ElementIterator<T>(rawData, splat, getNumElements())};
  }

  /// Return the held element values as a range of std::complex.
  template <typename T, typename ElementT = typename T::value_type,
            typename = typename std::enable_if<
                detail::is_complex_t<T>::value &&
                (std::numeric_limits<ElementT>::is_integer ||
                 is_valid_cpp_fp_type<ElementT>::value)>::type>
  llvm::iterator_range<ElementIterator<T>> getValues() const {
    assert(isValidComplex(sizeof(T), std::numeric_limits<ElementT>::is_integer,
                          std::numeric_limits<ElementT>::is_signed));
    const char *rawData = getRawData().data();
    bool splat = isSplat();
    return {ElementIterator<T>(rawData, splat, 0),
            ElementIterator<T>(rawData, splat, getNumElements())};
  }

  /// Return the held element values as a range of StringRef.
  template <typename T, typename = typename std::enable_if<
                            std::is_same<T, StringRef>::value>::type>
  llvm::iterator_range<ElementIterator<StringRef>> getValues() const {
    auto stringRefs = getRawStringData();
    const char *ptr = reinterpret_cast<const char *>(stringRefs.data());
    bool splat = isSplat();
    return {ElementIterator<StringRef>(ptr, splat, 0),
            ElementIterator<StringRef>(ptr, splat, getNumElements())};
  }

  /// Return the held element values as a range of Attributes.
  llvm::iterator_range<AttributeElementIterator> getAttributeValues() const;
  template <typename T, typename = typename std::enable_if<
                            std::is_same<T, Attribute>::value>::type>
  llvm::iterator_range<AttributeElementIterator> getValues() const {
    return getAttributeValues();
  }
  AttributeElementIterator attr_value_begin() const;
  AttributeElementIterator attr_value_end() const;

  /// Return the held element values a range of T, where T is a derived
  /// attribute type.
  template <typename T>
  using DerivedAttributeElementIterator =
      llvm::mapped_iterator<AttributeElementIterator, T (*)(Attribute)>;
  template <typename T, typename = typename std::enable_if<
                            std::is_base_of<Attribute, T>::value &&
                            !std::is_same<Attribute, T>::value>::type>
  llvm::iterator_range<DerivedAttributeElementIterator<T>> getValues() const {
    auto castFn = [](Attribute attr) { return attr.template cast<T>(); };
    return llvm::map_range(getAttributeValues(),
                           static_cast<T (*)(Attribute)>(castFn));
  }

  /// Return the held element values as a range of bool. The element type of
  /// this attribute must be of integer type of bitwidth 1.
  llvm::iterator_range<BoolElementIterator> getBoolValues() const;
  template <typename T, typename = typename std::enable_if<
                            std::is_same<T, bool>::value>::type>
  llvm::iterator_range<BoolElementIterator> getValues() const {
    return getBoolValues();
  }

  /// Return the held element values as a range of APInts. The element type of
  /// this attribute must be of integer type.
  llvm::iterator_range<IntElementIterator> getIntValues() const;
  template <typename T, typename = typename std::enable_if<
                            std::is_same<T, APInt>::value>::type>
  llvm::iterator_range<IntElementIterator> getValues() const {
    return getIntValues();
  }
  IntElementIterator int_value_begin() const;
  IntElementIterator int_value_end() const;

  /// Return the held element values as a range of complex APInts. The element
  /// type of this attribute must be a complex of integer type.
  llvm::iterator_range<ComplexIntElementIterator> getComplexIntValues() const;
  template <typename T, typename = typename std::enable_if<
                            std::is_same<T, std::complex<APInt>>::value>::type>
  llvm::iterator_range<ComplexIntElementIterator> getValues() const {
    return getComplexIntValues();
  }

  /// Return the held element values as a range of APFloat. The element type of
  /// this attribute must be of float type.
  llvm::iterator_range<FloatElementIterator> getFloatValues() const;
  template <typename T, typename = typename std::enable_if<
                            std::is_same<T, APFloat>::value>::type>
  llvm::iterator_range<FloatElementIterator> getValues() const {
    return getFloatValues();
  }
  FloatElementIterator float_value_begin() const;
  FloatElementIterator float_value_end() const;

  /// Return the held element values as a range of complex APFloat. The element
  /// type of this attribute must be a complex of float type.
  llvm::iterator_range<ComplexFloatElementIterator>
  getComplexFloatValues() const;
  template <typename T, typename = typename std::enable_if<std::is_same<
                            T, std::complex<APFloat>>::value>::type>
  llvm::iterator_range<ComplexFloatElementIterator> getValues() const {
    return getComplexFloatValues();
  }

  /// Return the raw storage data held by this attribute. Users should generally
  /// not use this directly, as the internal storage format is not always in the
  /// form the user might expect.
  ArrayRef<char> getRawData() const;

  /// Return the raw StringRef data held by this attribute.
  ArrayRef<StringRef> getRawStringData() const;

  //===--------------------------------------------------------------------===//
  // Mutation Utilities
  //===--------------------------------------------------------------------===//

  /// Return a new DenseElementsAttr that has the same data as the current
  /// attribute, but has been reshaped to 'newType'. The new type must have the
  /// same total number of elements as well as element type.
  DenseElementsAttr reshape(ShapedType newType);

  /// Generates a new DenseElementsAttr by mapping each int value to a new
  /// underlying APInt. The new values can represent either an integer or float.
  /// This underlying type must be an DenseIntElementsAttr.
  DenseElementsAttr mapValues(Type newElementType,
                              function_ref<APInt(const APInt &)> mapping) const;

  /// Generates a new DenseElementsAttr by mapping each float value to a new
  /// underlying APInt. the new values can represent either an integer or float.
  /// This underlying type must be an DenseFPElementsAttr.
  DenseElementsAttr
  mapValues(Type newElementType,
            function_ref<APInt(const APFloat &)> mapping) const;

protected:
  /// Get iterators to the raw APInt values for each element in this attribute.
  IntElementIterator raw_int_begin() const {
    return IntElementIterator(*this, 0);
  }
  IntElementIterator raw_int_end() const {
    return IntElementIterator(*this, getNumElements());
  }

  /// Overload of the raw 'get' method that asserts that the given type is of
  /// complex type. This method is used to verify type invariants that the
  /// templatized 'get' method cannot.
  static DenseElementsAttr getRawComplex(ShapedType type, ArrayRef<char> data,
                                         int64_t dataEltSize, bool isInt,
                                         bool isSigned);

  /// Overload of the raw 'get' method that asserts that the given type is of
  /// integer or floating-point type. This method is used to verify type
  /// invariants that the templatized 'get' method cannot.
  static DenseElementsAttr getRawIntOrFloat(ShapedType type,
                                            ArrayRef<char> data,
                                            int64_t dataEltSize, bool isInt,
                                            bool isSigned);

  /// Check the information for a C++ data type, check if this type is valid for
  /// the current attribute. This method is used to verify specific type
  /// invariants that the templatized 'getValues' method cannot.
  bool isValidIntOrFloat(int64_t dataEltSize, bool isInt, bool isSigned) const;

  /// Check the information for a C++ data type, check if this type is valid for
  /// the current attribute. This method is used to verify specific type
  /// invariants that the templatized 'getValues' method cannot.
  bool isValidComplex(int64_t dataEltSize, bool isInt, bool isSigned) const;
};

/// An attribute class for representing dense arrays of strings. The structure
/// storing and querying a list of densely packed strings.
class DenseStringElementsAttr
    : public Attribute::AttrBase<DenseStringElementsAttr, DenseElementsAttr,
                                 detail::DenseStringElementsAttributeStorage> {

public:
  using Base::Base;

  /// Overload of the raw 'get' method that asserts that the given type is of
  /// integer or floating-point type. This method is used to verify type
  /// invariants that the templatized 'get' method cannot.
  static DenseStringElementsAttr get(ShapedType type, ArrayRef<StringRef> data);

protected:
  friend DenseElementsAttr;
};

/// An attribute class for specializing behavior of Int and Floating-point
/// densely packed string arrays.
class DenseIntOrFPElementsAttr
    : public Attribute::AttrBase<DenseIntOrFPElementsAttr, DenseElementsAttr,
                                 detail::DenseIntOrFPElementsAttributeStorage> {

public:
  using Base::Base;

  /// Convert endianess of input ArrayRef for big-endian(BE) machines. All of
  /// the elements of `inRawData` has `type`. If `inRawData` is little endian
  /// (LE), it is converted to big endian (BE). Conversely, if `inRawData` is
  /// BE, converted to LE.
  static void
  convertEndianOfArrayRefForBEmachine(ArrayRef<char> inRawData,
                                      MutableArrayRef<char> outRawData,
                                      ShapedType type);

  /// Convert endianess of input for big-endian(BE) machines. The number of
  /// elements of `inRawData` is `numElements`, and each element has
  /// `elementBitWidth` bits. If `inRawData` is little endian (LE), it is
  /// converted to big endian (BE) and saved in `outRawData`. Conversely, if
  /// `inRawData` is BE, converted to LE.
  static void convertEndianOfCharForBEmachine(const char *inRawData,
                                              char *outRawData,
                                              size_t elementBitWidth,
                                              size_t numElements);

protected:
  friend DenseElementsAttr;

  /// Constructs a dense elements attribute from an array of raw APFloat values.
  /// Each APFloat value is expected to have the same bitwidth as the element
  /// type of 'type'. 'type' must be a vector or tensor with static shape.
  static DenseElementsAttr getRaw(ShapedType type, size_t storageWidth,
                                  ArrayRef<APFloat> values, bool isSplat);

  /// Constructs a dense elements attribute from an array of raw APInt values.
  /// Each APInt value is expected to have the same bitwidth as the element type
  /// of 'type'. 'type' must be a vector or tensor with static shape.
  static DenseElementsAttr getRaw(ShapedType type, size_t storageWidth,
                                  ArrayRef<APInt> values, bool isSplat);

  /// Get or create a new dense elements attribute instance with the given raw
  /// data buffer. 'type' must be a vector or tensor with static shape.
  static DenseElementsAttr getRaw(ShapedType type, ArrayRef<char> data,
                                  bool isSplat);

  /// Overload of the raw 'get' method that asserts that the given type is of
  /// complex type. This method is used to verify type invariants that the
  /// templatized 'get' method cannot.
  static DenseElementsAttr getRawComplex(ShapedType type, ArrayRef<char> data,
                                         int64_t dataEltSize, bool isInt,
                                         bool isSigned);

  /// Overload of the raw 'get' method that asserts that the given type is of
  /// integer or floating-point type. This method is used to verify type
  /// invariants that the templatized 'get' method cannot.
  static DenseElementsAttr getRawIntOrFloat(ShapedType type,
                                            ArrayRef<char> data,
                                            int64_t dataEltSize, bool isInt,
                                            bool isSigned);
};

/// An attribute that represents a reference to a dense float vector or tensor
/// object. Each element is stored as a double.
class DenseFPElementsAttr : public DenseIntOrFPElementsAttr {
public:
  using iterator = DenseElementsAttr::FloatElementIterator;

  using DenseIntOrFPElementsAttr::DenseIntOrFPElementsAttr;

  /// Get an instance of a DenseFPElementsAttr with the given arguments. This
  /// simply wraps the DenseElementsAttr::get calls.
  template <typename Arg>
  static DenseFPElementsAttr get(const ShapedType &type, Arg &&arg) {
    return DenseElementsAttr::get(type, llvm::makeArrayRef(arg))
        .template cast<DenseFPElementsAttr>();
  }
  template <typename T>
  static DenseFPElementsAttr get(const ShapedType &type,
                                 const std::initializer_list<T> &list) {
    return DenseElementsAttr::get(type, list)
        .template cast<DenseFPElementsAttr>();
  }

  /// Generates a new DenseElementsAttr by mapping each value attribute, and
  /// constructing the DenseElementsAttr given the new element type.
  DenseElementsAttr
  mapValues(Type newElementType,
            function_ref<APInt(const APFloat &)> mapping) const;

  /// Iterator access to the float element values.
  iterator begin() const { return float_value_begin(); }
  iterator end() const { return float_value_end(); }

  /// Method for supporting type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr);
};

/// An attribute that represents a reference to a dense integer vector or tensor
/// object.
class DenseIntElementsAttr : public DenseIntOrFPElementsAttr {
public:
  /// DenseIntElementsAttr iterates on APInt, so we can use the raw element
  /// iterator directly.
  using iterator = DenseElementsAttr::IntElementIterator;

  using DenseIntOrFPElementsAttr::DenseIntOrFPElementsAttr;

  /// Get an instance of a DenseIntElementsAttr with the given arguments. This
  /// simply wraps the DenseElementsAttr::get calls.
  template <typename Arg>
  static DenseIntElementsAttr get(const ShapedType &type, Arg &&arg) {
    return DenseElementsAttr::get(type, llvm::makeArrayRef(arg))
        .template cast<DenseIntElementsAttr>();
  }
  template <typename T>
  static DenseIntElementsAttr get(const ShapedType &type,
                                  const std::initializer_list<T> &list) {
    return DenseElementsAttr::get(type, list)
        .template cast<DenseIntElementsAttr>();
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

/// An opaque attribute that represents a reference to a vector or tensor
/// constant with opaque content. This representation is for tensor constants
/// which the compiler may not need to interpret. This attribute is always
/// associated with a particular dialect, which provides a method to convert
/// tensor representation to a non-opaque format.
class OpaqueElementsAttr
    : public Attribute::AttrBase<OpaqueElementsAttr, ElementsAttr,
                                 detail::OpaqueElementsAttributeStorage> {
public:
  using Base::Base;
  using ValueType = StringRef;

  static OpaqueElementsAttr get(Dialect *dialect, ShapedType type,
                                StringRef bytes);

  StringRef getValue() const;

  /// Return the value at the given index. The 'index' is expected to refer to a
  /// valid element.
  Attribute getValue(ArrayRef<uint64_t> index) const;

  /// Decodes the attribute value using dialect-specific decoding hook.
  /// Returns false if decoding is successful. If not, returns true and leaves
  /// 'result' argument unspecified.
  bool decode(ElementsAttr &result);

  /// Returns dialect associated with this opaque constant.
  Dialect *getDialect() const;
};

/// An attribute that represents a reference to a sparse vector or tensor
/// object.
///
/// This class uses COO (coordinate list) encoding to represent the sparse
/// elements in an element attribute. Specifically, the sparse vector/tensor
/// stores the indices and values as two separate dense elements attributes of
/// tensor type (even if the sparse attribute is of vector type, in order to
/// support empty lists). The dense elements attribute indices is a 2-D tensor
/// of 64-bit integer elements with shape [N, ndims], which specifies the
/// indices of the elements in the sparse tensor that contains nonzero values.
/// The dense elements attribute values is a 1-D tensor with shape [N], and it
/// supplies the corresponding values for the indices.
///
/// For example,
/// `sparse<tensor<3x4xi32>, [[0, 0], [1, 2]], [1, 5]>` represents tensor
/// [[1, 0, 0, 0],
///  [0, 0, 5, 0],
///  [0, 0, 0, 0]].
class SparseElementsAttr
    : public Attribute::AttrBase<SparseElementsAttr, ElementsAttr,
                                 detail::SparseElementsAttributeStorage> {
public:
  using Base::Base;

  template <typename T>
  using iterator =
      llvm::mapped_iterator<llvm::detail::value_sequence_iterator<ptrdiff_t>,
                            std::function<T(ptrdiff_t)>>;

  /// 'type' must be a vector or tensor with static shape.
  static SparseElementsAttr get(ShapedType type, DenseElementsAttr indices,
                                DenseElementsAttr values);

  DenseIntElementsAttr getIndices() const;

  DenseElementsAttr getValues() const;

  /// Return the values of this attribute in the form of the given type 'T'. 'T'
  /// may be any of Attribute, APInt, APFloat, c++ integer/float types, etc.
  template <typename T> llvm::iterator_range<iterator<T>> getValues() const {
    auto zeroValue = getZeroValue<T>();
    auto valueIt = getValues().getValues<T>().begin();
    const std::vector<ptrdiff_t> flatSparseIndices(getFlattenedSparseIndices());
    // TODO: Move-capture flatSparseIndices when c++14 is available.
    std::function<T(ptrdiff_t)> mapFn = [=](ptrdiff_t index) {
      // Try to map the current index to one of the sparse indices.
      for (unsigned i = 0, e = flatSparseIndices.size(); i != e; ++i)
        if (flatSparseIndices[i] == index)
          return *std::next(valueIt, i);
      // Otherwise, return the zero value.
      return zeroValue;
    };
    return llvm::map_range(llvm::seq<ptrdiff_t>(0, getNumElements()), mapFn);
  }

  /// Return the value of the element at the given index. The 'index' is
  /// expected to refer to a valid element.
  Attribute getValue(ArrayRef<uint64_t> index) const;

private:
  /// Get a zero APFloat for the given sparse attribute.
  APFloat getZeroAPFloat() const;

  /// Get a zero APInt for the given sparse attribute.
  APInt getZeroAPInt() const;

  /// Get a zero attribute for the given sparse attribute.
  Attribute getZeroAttr() const;

  /// Utility methods to generate a zero value of some type 'T'. This is used by
  /// the 'iterator' class.
  /// Get a zero for a given attribute type.
  template <typename T>
  typename std::enable_if<std::is_base_of<Attribute, T>::value, T>::type
  getZeroValue() const {
    return getZeroAttr().template cast<T>();
  }
  /// Get a zero for an APInt.
  template <typename T>
  typename std::enable_if<std::is_same<APInt, T>::value, T>::type
  getZeroValue() const {
    return getZeroAPInt();
  }
  template <typename T>
  typename std::enable_if<std::is_same<std::complex<APInt>, T>::value, T>::type
  getZeroValue() const {
    APInt intZero = getZeroAPInt();
    return {intZero, intZero};
  }
  /// Get a zero for an APFloat.
  template <typename T>
  typename std::enable_if<std::is_same<APFloat, T>::value, T>::type
  getZeroValue() const {
    return getZeroAPFloat();
  }
  template <typename T>
  typename std::enable_if<std::is_same<std::complex<APFloat>, T>::value,
                          T>::type
  getZeroValue() const {
    APFloat floatZero = getZeroAPFloat();
    return {floatZero, floatZero};
  }

  /// Get a zero for an C++ integer, float, StringRef, or complex type.
  template <typename T>
  typename std::enable_if<
      std::numeric_limits<T>::is_integer ||
          DenseElementsAttr::is_valid_cpp_fp_type<T>::value ||
          std::is_same<T, StringRef>::value ||
          (detail::is_complex_t<T>::value &&
           !llvm::is_one_of<T, std::complex<APInt>,
                            std::complex<APFloat>>::value),
      T>::type
  getZeroValue() const {
    return T();
  }

  /// Flatten, and return, all of the sparse indices in this attribute in
  /// row-major order.
  std::vector<ptrdiff_t> getFlattenedSparseIndices() const;
};

/// An attribute that represents a reference to a splat vector or tensor
/// constant, meaning all of the elements have the same value.
class SplatElementsAttr : public DenseElementsAttr {
public:
  using DenseElementsAttr::DenseElementsAttr;

  /// Method for support type inquiry through isa, cast and dyn_cast.
  static bool classof(Attribute attr) {
    auto denseAttr = attr.dyn_cast<DenseElementsAttr>();
    return denseAttr && denseAttr.isSplat();
  }
};

namespace detail {
/// This class represents a general iterator over the values of an ElementsAttr.
/// It supports all subclasses aside from OpaqueElementsAttr.
template <typename T>
class ElementsAttrIterator
    : public llvm::iterator_facade_base<ElementsAttrIterator<T>,
                                        std::random_access_iterator_tag, T,
                                        std::ptrdiff_t, T, T> {
  // NOTE: We use a dummy enable_if here because MSVC cannot use 'decltype'
  // inside of a conversion operator.
  using DenseIteratorT = typename std::enable_if<
      true,
      decltype(std::declval<DenseElementsAttr>().getValues<T>().begin())>::type;
  using SparseIteratorT = SparseElementsAttr::iterator<T>;

  /// A union containing the specific iterators for each derived attribute kind.
  union Iterator {
    Iterator(DenseIteratorT &&it) : denseIt(std::move(it)) {}
    Iterator(SparseIteratorT &&it) : sparseIt(std::move(it)) {}
    Iterator() {}
    ~Iterator() {}

    operator const DenseIteratorT &() const { return denseIt; }
    operator const SparseIteratorT &() const { return sparseIt; }
    operator DenseIteratorT &() { return denseIt; }
    operator SparseIteratorT &() { return sparseIt; }

    /// An instance of a dense elements iterator.
    DenseIteratorT denseIt;
    /// An instance of a sparse elements iterator.
    SparseIteratorT sparseIt;
  };

  /// Utility method to process a functor on each of the internal iterator
  /// types.
  template <typename RetT, template <typename> class ProcessFn,
            typename... Args>
  RetT process(Args &... args) const {
    if (attr.isa<DenseElementsAttr>())
      return ProcessFn<DenseIteratorT>()(args...);
    if (attr.isa<SparseElementsAttr>())
      return ProcessFn<SparseIteratorT>()(args...);
    llvm_unreachable("unexpected attribute kind");
  }

  /// Utility functors used to generically implement the iterators methods.
  template <typename ItT> struct PlusAssign {
    void operator()(ItT &it, ptrdiff_t offset) { it += offset; }
  };
  template <typename ItT> struct Minus {
    ptrdiff_t operator()(const ItT &lhs, const ItT &rhs) { return lhs - rhs; }
  };
  template <typename ItT> struct MinusAssign {
    void operator()(ItT &it, ptrdiff_t offset) { it -= offset; }
  };
  template <typename ItT> struct Dereference {
    T operator()(ItT &it) { return *it; }
  };
  template <typename ItT> struct ConstructIter {
    void operator()(ItT &dest, const ItT &it) { ::new (&dest) ItT(it); }
  };
  template <typename ItT> struct DestructIter {
    void operator()(ItT &it) { it.~ItT(); }
  };

public:
  ElementsAttrIterator(const ElementsAttrIterator<T> &rhs) : attr(rhs.attr) {
    process<void, ConstructIter>(it, rhs.it);
  }
  ~ElementsAttrIterator() { process<void, DestructIter>(it); }

  /// Methods necessary to support random access iteration.
  ptrdiff_t operator-(const ElementsAttrIterator<T> &rhs) const {
    assert(attr == rhs.attr && "incompatible iterators");
    return process<ptrdiff_t, Minus>(it, rhs.it);
  }
  bool operator==(const ElementsAttrIterator<T> &rhs) const {
    return rhs.attr == attr && process<bool, std::equal_to>(it, rhs.it);
  }
  bool operator<(const ElementsAttrIterator<T> &rhs) const {
    assert(attr == rhs.attr && "incompatible iterators");
    return process<bool, std::less>(it, rhs.it);
  }
  ElementsAttrIterator<T> &operator+=(ptrdiff_t offset) {
    process<void, PlusAssign>(it, offset);
    return *this;
  }
  ElementsAttrIterator<T> &operator-=(ptrdiff_t offset) {
    process<void, MinusAssign>(it, offset);
    return *this;
  }

  /// Dereference the iterator at the current index.
  T operator*() { return process<T, Dereference>(it); }

private:
  template <typename IteratorT>
  ElementsAttrIterator(Attribute attr, IteratorT &&it)
      : attr(attr), it(std::forward<IteratorT>(it)) {}

  /// Allow accessing the constructor.
  friend ElementsAttr;

  /// The parent elements attribute.
  Attribute attr;

  /// A union containing the specific iterators for each derived kind.
  Iterator it;
};

template <typename T>
class ElementsAttrRange : public llvm::iterator_range<ElementsAttrIterator<T>> {
  using llvm::iterator_range<ElementsAttrIterator<T>>::iterator_range;
};
} // namespace detail

/// Return the elements of this attribute as a value of type 'T'.
template <typename T>
auto ElementsAttr::getValues() const -> iterator_range<T> {
  if (DenseElementsAttr denseAttr = dyn_cast<DenseElementsAttr>()) {
    auto values = denseAttr.getValues<T>();
    return {iterator<T>(*this, values.begin()),
            iterator<T>(*this, values.end())};
  }
  if (SparseElementsAttr sparseAttr = dyn_cast<SparseElementsAttr>()) {
    auto values = sparseAttr.getValues<T>();
    return {iterator<T>(*this, values.begin()),
            iterator<T>(*this, values.end())};
  }
  llvm_unreachable("unexpected attribute kind");
}

//===----------------------------------------------------------------------===//
// Attributes Utils
//===----------------------------------------------------------------------===//

template <typename U> bool Attribute::isa() const {
  assert(impl && "isa<> used on a null attribute.");
  return U::classof(*this);
}

template <typename First, typename Second, typename... Rest>
bool Attribute::isa() const {
  return isa<First>() || isa<Second, Rest...>();
}

template <typename U> U Attribute::dyn_cast() const {
  return isa<U>() ? U(impl) : U(nullptr);
}
template <typename U> U Attribute::dyn_cast_or_null() const {
  return (impl && isa<U>()) ? U(impl) : U(nullptr);
}
template <typename U> U Attribute::cast() const {
  assert(isa<U>());
  return U(impl);
}

// Make Attribute hashable.
inline ::llvm::hash_code hash_value(Attribute arg) {
  return ::llvm::hash_value(arg.impl);
}

//===----------------------------------------------------------------------===//
// MutableDictionaryAttr
//===----------------------------------------------------------------------===//

/// A MutableDictionaryAttr is a mutable wrapper around a DictionaryAttr. It
/// provides additional interfaces for adding, removing, replacing attributes
/// within a DictionaryAttr.
///
/// We assume there will be relatively few attributes on a given operation
/// (maybe a dozen or so, but not hundreds or thousands) so we use linear
/// searches for everything.
class MutableDictionaryAttr {
public:
  MutableDictionaryAttr(DictionaryAttr attrs = nullptr)
      : attrs((attrs && !attrs.empty()) ? attrs : nullptr) {}
  MutableDictionaryAttr(ArrayRef<NamedAttribute> attributes);

  bool operator!=(const MutableDictionaryAttr &other) const {
    return !(*this == other);
  }
  bool operator==(const MutableDictionaryAttr &other) const {
    return attrs == other.attrs;
  }

  /// Return the underlying dictionary attribute.
  DictionaryAttr getDictionary(MLIRContext *context) const;

  /// Return the underlying dictionary attribute or null if there are no
  /// attributes within this dictionary.
  DictionaryAttr getDictionaryOrNull() const { return attrs; }

  /// Return all of the attributes on this operation.
  ArrayRef<NamedAttribute> getAttrs() const;

  /// Replace the held attributes with ones provided in 'newAttrs'.
  void setAttrs(ArrayRef<NamedAttribute> attributes);

  /// Return the specified attribute if present, null otherwise.
  Attribute get(StringRef name) const;
  Attribute get(Identifier name) const;

  /// Return the specified named attribute if present, None otherwise.
  Optional<NamedAttribute> getNamed(StringRef name) const;
  Optional<NamedAttribute> getNamed(Identifier name) const;

  /// If the an attribute exists with the specified name, change it to the new
  /// value.  Otherwise, add a new attribute with the specified name/value.
  void set(Identifier name, Attribute value);

  enum class RemoveResult { Removed, NotFound };

  /// Remove the attribute with the specified name if it exists.  The return
  /// value indicates whether the attribute was present or not.
  RemoveResult remove(Identifier name);

  bool empty() const { return attrs == nullptr; }

private:
  friend ::llvm::hash_code hash_value(const MutableDictionaryAttr &arg);

  DictionaryAttr attrs;
};

inline ::llvm::hash_code hash_value(const MutableDictionaryAttr &arg) {
  if (!arg.attrs)
    return ::llvm::hash_value((void *)nullptr);
  return hash_value(arg.attrs);
}

} // end namespace mlir.

namespace llvm {

// Attribute hash just like pointers.
template <> struct DenseMapInfo<mlir::Attribute> {
  static mlir::Attribute getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer));
  }
  static mlir::Attribute getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::Attribute val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::Attribute LHS, mlir::Attribute RHS) {
    return LHS == RHS;
  }
};

/// Allow LLVM to steal the low bits of Attributes.
template <> struct PointerLikeTypeTraits<mlir::Attribute> {
  static inline void *getAsVoidPointer(mlir::Attribute attr) {
    return const_cast<void *>(attr.getAsOpaquePointer());
  }
  static inline mlir::Attribute getFromVoidPointer(void *ptr) {
    return mlir::Attribute::getFromOpaquePointer(ptr);
  }
  static constexpr int NumLowBitsAvailable = 3;
};

template <>
struct PointerLikeTypeTraits<mlir::SymbolRefAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline mlir::SymbolRefAttr getFromVoidPointer(void *ptr) {
    return PointerLikeTypeTraits<mlir::Attribute>::getFromVoidPointer(ptr)
        .cast<mlir::SymbolRefAttr>();
  }
};

} // namespace llvm

#endif
