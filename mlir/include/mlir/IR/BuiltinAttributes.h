//===- BuiltinAttributes.h - MLIR Builtin Attribute Classes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINATTRIBUTES_H
#define MLIR_IR_BUILTINATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Sequence.h"
#include <complex>

namespace mlir {
class AffineMap;
class BoolAttr;
class DenseIntElementsAttr;
class FlatSymbolRefAttr;
class FunctionType;
class IntegerSet;
class IntegerType;
class Location;
class ShapedType;

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

  /// Returns if the number of elements held by this attribute is 0.
  bool empty() const { return size() == 0; }

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

} // namespace mlir

//===----------------------------------------------------------------------===//
// Tablegen Attribute Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/IR/BuiltinAttributes.h.inc"

//===----------------------------------------------------------------------===//
// C++ Attribute Declarations
//===----------------------------------------------------------------------===//

namespace mlir {
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

  /// Enable conversion to IntegerAttr. This uses conversion vs. inheritance to
  /// avoid bringing in all of IntegerAttrs methods.
  operator IntegerAttr() const { return IntegerAttr(impl); }

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
  static FlatSymbolRefAttr get(MLIRContext *ctx, StringRef value) {
    return SymbolRefAttr::get(ctx, value);
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
// DenseFPElementsAttr
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// DenseIntElementsAttr
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// SparseElementsAttr
//===----------------------------------------------------------------------===//

template <typename T>
auto SparseElementsAttr::getValues() const
    -> llvm::iterator_range<iterator<T>> {
  auto zeroValue = getZeroValue<T>();
  auto valueIt = getValues().getValues<T>().begin();
  const std::vector<ptrdiff_t> flatSparseIndices(getFlattenedSparseIndices());
  std::function<T(ptrdiff_t)> mapFn =
      [flatSparseIndices{std::move(flatSparseIndices)},
       valueIt{std::move(valueIt)},
       zeroValue{std::move(zeroValue)}](ptrdiff_t index) {
        // Try to map the current index to one of the sparse indices.
        for (unsigned i = 0, e = flatSparseIndices.size(); i != e; ++i)
          if (flatSparseIndices[i] == index)
            return *std::next(valueIt, i);
        // Otherwise, return the zero value.
        return zeroValue;
      };
  return llvm::map_range(llvm::seq<ptrdiff_t>(0, getNumElements()), mapFn);
}

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
  RetT process(Args &...args) const {
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

} // end namespace mlir.

//===----------------------------------------------------------------------===//
// Attribute Utilities
//===----------------------------------------------------------------------===//

namespace llvm {

template <>
struct PointerLikeTypeTraits<mlir::SymbolRefAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline mlir::SymbolRefAttr getFromVoidPointer(void *ptr) {
    return PointerLikeTypeTraits<mlir::Attribute>::getFromVoidPointer(ptr)
        .cast<mlir::SymbolRefAttr>();
  }
};

} // namespace llvm

#endif // MLIR_IR_BUILTINATTRIBUTES_H
