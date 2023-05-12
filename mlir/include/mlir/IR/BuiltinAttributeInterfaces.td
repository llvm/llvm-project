//===- BuiltinAttributeInterfaces.td - Attr interfaces -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of the ElementsAttr interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINATTRIBUTEINTERFACES_TD_
#define MLIR_IR_BUILTINATTRIBUTEINTERFACES_TD_

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TypedAttrInterface
//===----------------------------------------------------------------------===//

def TypedAttrInterface : AttrInterface<"TypedAttr"> {
  let cppNamespace = "::mlir";

  let description = [{
    This interface is used for attributes that have a type. The type of an
    attribute is understood to represent the type of the data contained in the
    attribute and is often used as the type of a value with this data.
  }];

  let methods = [InterfaceMethod<
    "Get the attribute's type",
    "::mlir::Type", "getType"
  >];
}

//===----------------------------------------------------------------------===//
// ElementsAttrInterface
//===----------------------------------------------------------------------===//

def ElementsAttrInterface : AttrInterface<"ElementsAttr", [TypedAttrInterface]> {
  let cppNamespace = "::mlir";
  let description = [{
    This interface is used for attributes that contain the constant elements of
    a tensor or vector type. It allows for opaquely interacting with the
    elements of the underlying attribute, and most importantly allows for
    accessing the element values (including iteration) in any of the C++ data
    types supported by the underlying attribute.

    An attribute implementing this interface can expose the supported data types
    in two steps:

    * Define the set of iterable C++ data types:

    An attribute may define the set of iterable types by providing a definition
    of tuples `ContiguousIterableTypesT` and/or `NonContiguousIterableTypesT`.

    -  `ContiguousIterableTypesT` should contain types which can be iterated
       contiguously. A contiguous range is an array-like range, such as
       ArrayRef, where all of the elements are layed out sequentially in memory.

    -  `NonContiguousIterableTypesT` should contain types which can not be
       iterated contiguously. A non-contiguous range implies no contiguity,
       whose elements may even be materialized when indexing, such as the case
       for a mapped_range.

    As an example, consider an attribute that only contains i64 elements, with
    the elements being stored within an ArrayRef. This attribute could
    potentially define the iterable types as so:

    ```c++
    using ContiguousIterableTypesT = std::tuple<uint64_t>;
    using NonContiguousIterableTypesT = std::tuple<APInt, Attribute>;
    ```

    * Provide a `FailureOr<iterator> try_value_begin_impl(OverloadToken<T>) const`
      overload for each iterable type

    These overloads should return an iterator to the start of the range for the
    respective iterable type or fail if the type cannot be iterated. Consider
    the example i64 elements attribute described in the previous section. This
    attribute may define the value_begin_impl overloads like so:

    ```c++
    /// Provide begin iterators for the various iterable types.
    /// * uint64_t
    FailureOr<const uint64_t *>
    value_begin_impl(OverloadToken<uint64_t>) const {
      return getElements().begin();
    }
    /// * APInt
    auto value_begin_impl(OverloadToken<llvm::APInt>) const {
      auto it = llvm::map_range(getElements(), [=](uint64_t value) {
        return llvm::APInt(/*numBits=*/64, value);
      }).begin();
      return FailureOr<decltype(it)>(std::move(it));
    }
    /// * Attribute
    auto value_begin_impl(OverloadToken<mlir::Attribute>) const {
      mlir::Type elementType = getShapedType().getElementType();
      auto it = llvm::map_range(getElements(), [=](uint64_t value) {
        return mlir::IntegerAttr::get(elementType,
                                      llvm::APInt(/*numBits=*/64, value));
      }).begin();
      return FailureOr<decltype(it)>(std::move(it));
    }
    ```

    After the above, ElementsAttr will now be able to iterate over elements
    using each of the registered iterable data types:

    ```c++
    ElementsAttr attr = myI64ElementsAttr;

    // We can access value ranges for the data types via `getValues<T>`.
    for (uint64_t value : attr.getValues<uint64_t>())
      ...;
    for (llvm::APInt value : attr.getValues<llvm::APInt>())
      ...;
    for (mlir::IntegerAttr value : attr.getValues<mlir::IntegerAttr>())
      ...;

    // We can also access the value iterators directly.
    auto it = attr.value_begin<uint64_t>(), e = attr.value_end<uint64_t>();
    for (; it != e; ++it) {
      uint64_t value = *it;
      ...
    }
    ```

    ElementsAttr also supports failable access to iterators and ranges. This
    allows for safely checking if the attribute supports the data type, and can
    also allow for code to have fast paths for native data types.

    ```c++
    // Using `tryGetValues<T>`, we can also safely handle when the attribute
    // doesn't support the data type.
    if (auto range = attr.tryGetValues<uint64_t>()) {
      for (uint64_t value : *range)
        ...;
      return;
    }

    // We can also access the begin iterator safely, by using `try_value_begin`.
    if (auto safeIt = attr.try_value_begin<uint64_t>()) {
      auto it = *safeIt, e = attr.value_end<uint64_t>();
      for (; it != e; ++it) {
        uint64_t value = *it;
        ...
      }
      return;
    }
    ```
  }];
  let methods = [
    InterfaceMethod<[{
      This method returns an opaque range indexer for the given elementID, which
      corresponds to a desired C++ element data type. Returns the indexer if the
      attribute supports the given data type, failure otherwise.
    }],
    "::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer>", "getValuesImpl",
    (ins "::mlir::TypeID":$elementID), [{}], /*defaultImplementation=*/[{
      auto result = getValueImpl(
        (typename ConcreteAttr::ContiguousIterableTypesT *)nullptr, elementID,
        /*isContiguous=*/std::true_type());
      if (succeeded(result))
        return std::move(result);

      return getValueImpl(
        (typename ConcreteAttr::NonContiguousIterableTypesT *)nullptr,
        elementID, /*isContiguous=*/std::false_type());
    }]>,
    InterfaceMethod<[{
      Returns true if the attribute elements correspond to a splat, i.e. that
      all elements of the attribute are the same value.
    }], "bool", "isSplat", (ins), [{}], /*defaultImplementation=*/[{
        // By default, only check for a single element splat.
        return $_attr.getNumElements() == 1;
    }]>,
    InterfaceMethod<[{
      Returns the shaped type of the elements attribute.
    }], "::mlir::ShapedType", "getShapedType", (ins), [{}], /*defaultImplementation=*/[{
        return $_attr.getType();
    }]>
  ];

  string ElementsAttrInterfaceAccessors = [{
    /// Return the number of elements held by this attribute.
    int64_t size() const { return getNumElements(); }

    /// Return if the attribute holds no elements.
    bool empty() const { return size() == 0; }
  }];

  let extraTraitClassDeclaration = [{
    // By default, no types are iterable.
    using ContiguousIterableTypesT = std::tuple<>;
    using NonContiguousIterableTypesT = std::tuple<>;

    //===------------------------------------------------------------------===//
    // Accessors
    //===------------------------------------------------------------------===//

    /// Return the element type of this ElementsAttr.
    Type getElementType() const {
      return ::mlir::ElementsAttr::getElementType($_attr);
    }

    /// Returns the number of elements held by this attribute.
    int64_t getNumElements() const {
      return ::mlir::ElementsAttr::getNumElements($_attr);
    }

    /// Return if the given 'index' refers to a valid element in this attribute.
    bool isValidIndex(ArrayRef<uint64_t> index) const {
      return ::mlir::ElementsAttr::isValidIndex($_attr, index);
    }

  protected:
    /// Returns the 1-dimensional flattened row-major index from the given
    /// multi-dimensional index.
    uint64_t getFlattenedIndex(ArrayRef<uint64_t> index) const {
      return ::mlir::ElementsAttr::getFlattenedIndex($_attr, index);
    }

    //===------------------------------------------------------------------===//
    // Value Iteration Internals
    //===------------------------------------------------------------------===//
  protected:
    /// This class is used to allow specifying function overloads for different
    /// types, without actually taking the types as parameters. This avoids the
    /// need to build complicated SFINAE to select specific overloads.
    template <typename T>
    struct OverloadToken {};

  private:
    /// This function unpacks the types within a given tuple and then forwards
    /// on to the unwrapped variant.
    template <typename... Ts, typename IsContiguousT>
    auto getValueImpl(std::tuple<Ts...> *, ::mlir::TypeID elementID,
                      IsContiguousT isContiguous) const {
      return getValueImpl<Ts...>(elementID, isContiguous);
    }
    /// Check to see if the given `elementID` matches the current type `T`. If
    /// it does, build a value result using the current type. If it doesn't,
    /// keep looking for the desired type.
    template <typename T, typename... Ts, typename IsContiguousT>
    auto getValueImpl(::mlir::TypeID elementID,
                      IsContiguousT isContiguous) const {
      if (::mlir::TypeID::get<T>() == elementID)
        return buildValueResult<T>(isContiguous);
      return getValueImpl<Ts...>(elementID, isContiguous);
    }
    /// Bottom out case for no matching type.
    template <typename IsContiguousT>
    ::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer>
    getValueImpl(::mlir::TypeID, IsContiguousT) const {
      return failure();
    }

    /// Build an indexer for the given type `T`, which is represented via a
    /// contiguous range.
    template <typename T>
    ::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> buildValueResult(
        /*isContiguous*/std::true_type) const {
      if ($_attr.empty()) {
        return ::mlir::detail::ElementsAttrIndexer::contiguous<T>(
          /*isSplat=*/false, nullptr);
      }

      auto valueIt = $_attr.try_value_begin_impl(OverloadToken<T>());
      if (::mlir::failed(valueIt))
        return ::mlir::failure();
      return ::mlir::detail::ElementsAttrIndexer::contiguous(
        $_attr.isSplat(), &**valueIt);
    }
    /// Build an indexer for the given type `T`, which is represented via a
    /// non-contiguous range.
    template <typename T>
    ::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> buildValueResult(
        /*isContiguous*/std::false_type) const {
      auto valueIt = $_attr.try_value_begin_impl(OverloadToken<T>());
      if (::mlir::failed(valueIt))
        return ::mlir::failure();
      return ::mlir::detail::ElementsAttrIndexer::nonContiguous(
        $_attr.isSplat(), *valueIt);
    }

  public:
    //===------------------------------------------------------------------===//
    // Value Iteration
    //===------------------------------------------------------------------===//

    /// The iterator for the given element type T.
    template <typename T, typename AttrT = ConcreteAttr>
    using iterator = decltype(std::declval<AttrT>().template value_begin<T>());
    /// The iterator range over the given element T.
    template <typename T, typename AttrT = ConcreteAttr>
    using iterator_range =
        decltype(std::declval<AttrT>().template getValues<T>());

    /// Return an iterator to the first element of this attribute as a value of
    /// type `T`.
    template <typename T>
    auto value_begin() const {
      return *$_attr.try_value_begin_impl(OverloadToken<T>());
    }

    /// Return the elements of this attribute as a value of type 'T'.
    template <typename T>
    auto getValues() const {
      auto beginIt = $_attr.template value_begin<T>();
      return detail::ElementsAttrRange<decltype(beginIt)>(
        $_attr.getType(), beginIt, std::next(beginIt, size()));
    }
  }] # ElementsAttrInterfaceAccessors;

  let extraClassDeclaration = [{
    template <typename T>
    using iterator = detail::ElementsAttrIterator<T>;
    template <typename T>
    using iterator_range = detail::ElementsAttrRange<iterator<T>>;

    //===------------------------------------------------------------------===//
    // Accessors
    //===------------------------------------------------------------------===//

    /// Return the element type of this ElementsAttr.
    Type getElementType() const { return getElementType(*this); }
    static Type getElementType(ElementsAttr elementsAttr);

    /// Return if the given 'index' refers to a valid element in this attribute.
    bool isValidIndex(ArrayRef<uint64_t> index) const {
      return isValidIndex(*this, index);
    }
    static bool isValidIndex(ShapedType type, ArrayRef<uint64_t> index);
    static bool isValidIndex(ElementsAttr elementsAttr,
                             ArrayRef<uint64_t> index);

    /// Return the 1 dimensional flattened row-major index from the given
    /// multi-dimensional index.
    uint64_t getFlattenedIndex(ArrayRef<uint64_t> index) const {
      return getFlattenedIndex(*this, index);
    }
    static uint64_t getFlattenedIndex(Type type,
                                      ArrayRef<uint64_t> index);
    static uint64_t getFlattenedIndex(ElementsAttr elementsAttr,
                                      ArrayRef<uint64_t> index) {
      return getFlattenedIndex(elementsAttr.getShapedType(), index);
    }

    /// Returns the number of elements held by this attribute.
    int64_t getNumElements() const { return getNumElements(*this); }
    static int64_t getNumElements(ElementsAttr elementsAttr);

    //===------------------------------------------------------------------===//
    // Value Iteration
    //===------------------------------------------------------------------===//

    template <typename T>
    using DerivedAttrValueCheckT =
        std::enable_if_t<std::is_base_of<Attribute, T>::value &&
                         !std::is_same<Attribute, T>::value>;
    template <typename T, typename ResultT>
    using DefaultValueCheckT =
        std::enable_if_t<std::is_same<Attribute, T>::value ||
                             !std::is_base_of<Attribute, T>::value,
                         ResultT>;

    /// Return the splat value for this attribute. This asserts that the
    /// attribute corresponds to a splat.
    template <typename T>
    T getSplatValue() const {
      assert(isSplat() && "expected splat attribute");
      return *value_begin<T>();
    }

    /// Return the elements of this attribute as a value of type 'T'.
    template <typename T>
    DefaultValueCheckT<T, iterator_range<T>> getValues() const {
      return {getShapedType(), value_begin<T>(), value_end<T>()};
    }
    template <typename T>
    DefaultValueCheckT<T, iterator<T>> value_begin() const;
    template <typename T>
    DefaultValueCheckT<T, iterator<T>> value_end() const {
      return iterator<T>({}, size());
    }

    /// Return the held element values a range of T, where T is a derived
    /// attribute type.
    template <typename T>
    using DerivedAttrValueIterator =
      llvm::mapped_iterator<iterator<Attribute>, T (*)(Attribute)>;
    template <typename T>
    using DerivedAttrValueIteratorRange =
      detail::ElementsAttrRange<DerivedAttrValueIterator<T>>;
    template <typename T, typename = DerivedAttrValueCheckT<T>>
    DerivedAttrValueIteratorRange<T> getValues() const {
      auto castFn = [](Attribute attr) { return ::llvm::cast<T>(attr); };
      return {getShapedType(), llvm::map_range(getValues<Attribute>(),
              static_cast<T (*)(Attribute)>(castFn))};
    }
    template <typename T, typename = DerivedAttrValueCheckT<T>>
    DerivedAttrValueIterator<T> value_begin() const {
      return getValues<T>().begin();
    }
    template <typename T, typename = DerivedAttrValueCheckT<T>>
    DerivedAttrValueIterator<T> value_end() const {
      return {value_end<Attribute>(), nullptr};
    }

    //===------------------------------------------------------------------===//
    // Failable Value Iteration

    /// If this attribute supports iterating over element values of type `T`,
    /// return the iterable range. Otherwise, return std::nullopt.
    template <typename T>
    DefaultValueCheckT<T, std::optional<iterator_range<T>>> tryGetValues() const {
      if (std::optional<iterator<T>> beginIt = try_value_begin<T>())
        return iterator_range<T>(getShapedType(), *beginIt, value_end<T>());
      return std::nullopt;
    }
    template <typename T>
    DefaultValueCheckT<T, std::optional<iterator<T>>> try_value_begin() const;

    /// If this attribute supports iterating over element values of type `T`,
    /// return the iterable range. Otherwise, return std::nullopt.
    template <typename T, typename = DerivedAttrValueCheckT<T>>
    std::optional<DerivedAttrValueIteratorRange<T>> tryGetValues() const {
      auto values = tryGetValues<Attribute>();
      if (!values)
        return std::nullopt;

      auto castFn = [](Attribute attr) { return ::llvm::cast<T>(attr); };
      return DerivedAttrValueIteratorRange<T>(
        getShapedType(),
        llvm::map_range(*values, static_cast<T (*)(Attribute)>(castFn))
      );
    }
    template <typename T, typename = DerivedAttrValueCheckT<T>>
    std::optional<DerivedAttrValueIterator<T>> try_value_begin() const {
      if (auto values = tryGetValues<T>())
        return values->begin();
      return std::nullopt;
    }
  }] # ElementsAttrInterfaceAccessors;
}

//===----------------------------------------------------------------------===//
// MemRefLayoutAttrInterface
//===----------------------------------------------------------------------===//

def MemRefLayoutAttrInterface : AttrInterface<"MemRefLayoutAttrInterface"> {
  let cppNamespace = "::mlir";

  let description = [{
    This interface is used for attributes that can represent the MemRef type's
    layout semantics, such as dimension order in the memory, strides and offsets.
    Such a layout attribute should be representable as a
    [semi-affine map](Affine.md/#semi-affine-maps).

    Note: the MemRef type's layout is assumed to represent simple strided buffer
    layout. For more complicated case, like sparse storage buffers,
    it is preferable to use separate type with more specic layout, rather then
    introducing extra complexity to the builin MemRef type.
  }];

  let methods = [
    InterfaceMethod<
      "Get the MemRef layout as an AffineMap, the method must not return NULL",
      "::mlir::AffineMap", "getAffineMap", (ins)
    >,

    InterfaceMethod<
      "Return true if this attribute represents the identity layout",
      "bool", "isIdentity", (ins),
      [{}],
      [{
        return $_attr.getAffineMap().isIdentity();
      }]
    >,

    InterfaceMethod<
      "Check if the current layout is applicable to the provided shape",
      "::mlir::LogicalResult", "verifyLayout",
      (ins "::llvm::ArrayRef<int64_t>":$shape,
           "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError),
      [{}],
      [{
        return ::mlir::detail::verifyAffineMapAsLayout($_attr.getAffineMap(),
                                                       shape, emitError);
      }]
    >
  ];
}

#endif // MLIR_IR_BUILTINATTRIBUTEINTERFACES_TD_
