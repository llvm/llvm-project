//===- BuiltinAttributes.td - Builtin attr definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the set of builtin MLIR types, or the set of types necessary for the
// validity of and defining the IR.
//
//===----------------------------------------------------------------------===//

#ifndef BUILTIN_ATTRIBUTES
#define BUILTIN_ATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SubElementInterfaces.td"

// TODO: Currently the attributes defined in this file are prefixed with
// `Builtin_`.  This is to differentiate the attributes here with the ones in
// OpBase.td. We should remove the definitions in OpBase.td, and repoint users
// to this file instead.

// Base class for Builtin dialect attributes.
class Builtin_Attr<string name, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<Builtin_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

//===----------------------------------------------------------------------===//
// AffineMapAttr
//===----------------------------------------------------------------------===//

def Builtin_AffineMapAttr : Builtin_Attr<"AffineMap", [
    MemRefLayoutAttrInterface
  ]> {
  let summary = "An Attribute containing an AffineMap object";
  let description = [{
    Syntax:

    ```
    affine-map-attribute ::= `affine_map` `<` affine-map `>`
    ```

    Examples:

    ```mlir
    affine_map<(d0) -> (d0)>
    affine_map<(d0, d1, d2) -> (d0, d1)>
    ```
  }];
  let parameters = (ins "AffineMap":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "AffineMap":$value), [{
      return $_get(value.getContext(), value);
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = AffineMap;
    AffineMap getAffineMap() const { return getValue(); }
  }];
  let skipDefaultBuilders = 1;
  let typeBuilder = "IndexType::get($_value.getContext())";
}

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//

def Builtin_ArrayAttr : Builtin_Attr<"Array", [
    DeclareAttrInterfaceMethods<SubElementAttrInterface,
        ["replaceImmediateSubAttribute"]>
  ]> {
  let summary = "A collection of other Attribute values";
  let description = [{
    Syntax:

    ```
    array-attribute ::= `[` (attribute-value (`,` attribute-value)*)? `]`
    ```

    An array attribute is an attribute that represents a collection of attribute
    values.

    Examples:

    ```mlir
    []
    [10, i32]
    [affine_map<(d0, d1, d2) -> (d0, d1)>, i32, "string attribute"]
    ```
  }];
  let parameters = (ins ArrayRefParameter<"Attribute", "">:$value);
  let extraClassDeclaration = [{
    using ValueType = ArrayRef<Attribute>;

    /// Return the element at the given index.
    Attribute operator[](unsigned idx) const {
      assert(idx < size() && "index out of bounds");
      return getValue()[idx];
    }

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
    iterator_range<attr_value_iterator<AttrTy>> getAsRange() const {
      return llvm::make_range(attr_value_iterator<AttrTy>(begin()),
                              attr_value_iterator<AttrTy>(end()));
    }
    template <typename AttrTy,
              typename UnderlyingTy = typename AttrTy::ValueType>
    auto getAsValueRange() const {
      return llvm::map_range(getAsRange<AttrTy>(), [](AttrTy attr) {
        return static_cast<UnderlyingTy>(attr.getValue());
      });
    }
  }];
}

//===----------------------------------------------------------------------===//
// DenseIntOrFPElementsAttr
//===----------------------------------------------------------------------===//

def Builtin_DenseIntOrFPElementsAttr : Builtin_Attr<
    "DenseIntOrFPElements", [ElementsAttrInterface], "DenseElementsAttr"
  > {
  let summary = "An Attribute containing a dense multi-dimensional array of "
                "integer or floating-point values";
  let description = [{
    Syntax:

    ```
    tensor-literal ::= integer-literal | float-literal | bool-literal | [] | [tensor-literal (, tensor-literal)* ]
    dense-intorfloat-elements-attribute ::= `dense` `<` tensor-literal `>` `:`
                                            ( tensor-type | vector-type )
    ```

    A dense int-or-float elements attribute is an elements attribute containing
    a densely packed vector or tensor of integer or floating-point values. The
    element type of this attribute is required to be either an `IntegerType` or
    a `FloatType`.

    Examples:

    ```
    // A splat tensor of integer values.
    dense<10> : tensor<2xi32>
    // A tensor of 2 float32 elements.
    dense<[10.0, 11.0]> : tensor<2xf32>
    ```
  }];
  let parameters = (ins AttributeSelfTypeParameter<"", "ShapedType">:$type,
                        "ArrayRef<char>":$rawData);
  let extraClassDeclaration = [{
    using DenseElementsAttr::empty;
    using DenseElementsAttr::getNumElements;
    using DenseElementsAttr::getElementType;
    using DenseElementsAttr::getValues;
    using DenseElementsAttr::isSplat;
    using DenseElementsAttr::size;
    using DenseElementsAttr::value_begin;

    /// The set of data types that can be iterated by this attribute.
    using ContiguousIterableTypesT = std::tuple<
      // Integer types.
      uint8_t, uint16_t, uint32_t, uint64_t,
      int8_t, int16_t, int32_t, int64_t,
      short, unsigned short, int, unsigned, long, unsigned long,
      std::complex<uint8_t>, std::complex<uint16_t>, std::complex<uint32_t>,
      std::complex<uint64_t>,
      std::complex<int8_t>, std::complex<int16_t>, std::complex<int32_t>,
      std::complex<int64_t>,
      // Float types.
      float, double, std::complex<float>, std::complex<double>
    >;
    using NonContiguousIterableTypesT = std::tuple<
      Attribute,
      // Integer types.
      APInt, bool, std::complex<APInt>,
      // Float types.
      APFloat, std::complex<APFloat>
    >;

    /// Provide a `value_begin_impl` to enable iteration within ElementsAttr.
    template <typename T>
    auto value_begin_impl(OverloadToken<T>) const {
      return value_begin<T>();
    }

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

    /// Constructs a dense elements attribute from an array of raw APFloat
    /// values. Each APFloat value is expected to have the same bitwidth as the
    /// element type of 'type'. 'type' must be a vector or tensor with static
    /// shape.
    ///
    /// If the `values` array only has a single element, then this constructs
    /// splat of that value.
    static DenseElementsAttr getRaw(ShapedType type, size_t storageWidth,
                                    ArrayRef<APFloat> values);

    /// Constructs a dense elements attribute from an array of raw APInt values.
    /// Each APInt value is expected to have the same bitwidth as the element
    /// type of 'type'. 'type' must be a vector or tensor with static shape.
    ///
    /// If the `values` array only has a single element, then this constructs
    /// splat of that value.
    static DenseElementsAttr getRaw(ShapedType type, size_t storageWidth,
                                    ArrayRef<APInt> values);

    /// Get or create a new dense elements attribute instance with the given raw
    /// data buffer. 'type' must be a vector or tensor with static shape.
    ///
    /// If the `values` array only has a single element, then this constructs
    /// splat of that value.
    static DenseElementsAttr getRaw(ShapedType type, ArrayRef<char> data);

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
  public:
  }];
  let genAccessors = 0;
  let genStorageClass = 0;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// DenseStringElementsAttr
//===----------------------------------------------------------------------===//

def Builtin_DenseStringElementsAttr : Builtin_Attr<
    "DenseStringElements", [ElementsAttrInterface], "DenseElementsAttr"
  > {
  let summary = "An Attribute containing a dense multi-dimensional array of "
                "strings";
  let description = [{
    Syntax:

    ```
    dense-string-elements-attribute ::= `dense` `<` attribute-value `>` `:`
                                        ( tensor-type | vector-type )
    ```

    A dense string elements attribute is an elements attribute containing a
    densely packed vector or tensor of string values. There are no restrictions
    placed on the element type of this attribute, enabling the use of dialect
    specific string types.

    Examples:

    ```
    // A splat tensor of strings.
    dense<"example"> : tensor<2x!foo.string>
    // A tensor of 2 string elements.
    dense<["example1", "example2"]> : tensor<2x!foo.string>
    ```
  }];
  let parameters = (ins AttributeSelfTypeParameter<"", "ShapedType">:$type,
                        "ArrayRef<StringRef>":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "ShapedType":$type,
                                        "ArrayRef<StringRef>":$values), [{
      return $_get(type.getContext(), type, values,
                   /* isSplat */(values.size() == 1));
    }]>,
  ];
  let extraClassDeclaration = [{
    using DenseElementsAttr::empty;
    using DenseElementsAttr::getNumElements;
    using DenseElementsAttr::getElementType;
    using DenseElementsAttr::getValues;
    using DenseElementsAttr::isSplat;
    using DenseElementsAttr::size;
    using DenseElementsAttr::value_begin;

    /// The set of data types that can be iterated by this attribute.
    using ContiguousIterableTypesT = std::tuple<StringRef>;
    using NonContiguousIterableTypesT = std::tuple<Attribute>;

    /// Provide a `value_begin_impl` to enable iteration within ElementsAttr.
    template <typename T>
    auto value_begin_impl(OverloadToken<T>) const {
      return value_begin<T>();
    }

  protected:
    friend DenseElementsAttr;

  public:
  }];
  let genAccessors = 0;
  let genStorageClass = 0;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// DictionaryAttr
//===----------------------------------------------------------------------===//

def Builtin_DictionaryAttr : Builtin_Attr<"Dictionary", [
    DeclareAttrInterfaceMethods<SubElementAttrInterface,
        ["replaceImmediateSubAttribute"]>
  ]> {
  let summary = "An dictionary of named Attribute values";
  let description = [{
    Syntax:

    ```
    dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
    ```

    A dictionary attribute is an attribute that represents a sorted collection of
    named attribute values. The elements are sorted by name, and each name must be
    unique within the collection.

    Examples:

    ```mlir
    {}
    {attr_name = "string attribute"}
    {int_attr = 10, "string attr name" = "string attribute"}
    ```
  }];
  let parameters = (ins ArrayRefParameter<"NamedAttribute", "">:$value);
  let builders = [
    AttrBuilder<(ins CArg<"ArrayRef<NamedAttribute>", "llvm::None">:$value)>
  ];
  let extraClassDeclaration = [{
    using ValueType = ArrayRef<NamedAttribute>;

    /// Construct a dictionary with an array of values that is known to already
    /// be sorted by name and uniqued.
    static DictionaryAttr getWithSorted(MLIRContext *context,
                                        ArrayRef<NamedAttribute> value);

    /// Return the specified attribute if present, null otherwise.
    Attribute get(StringRef name) const;
    Attribute get(StringAttr name) const;

    /// Return the specified named attribute if present, None otherwise.
    Optional<NamedAttribute> getNamed(StringRef name) const;
    Optional<NamedAttribute> getNamed(StringAttr name) const;

    /// Return whether the specified attribute is present.
    bool contains(StringRef name) const;
    bool contains(StringAttr name) const;

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

    /// Return the specified attribute if present and is an instance of
    /// `AttrClass`, null otherwise.
    template<typename AttrClass, typename NameClass>
    AttrClass getAs(NameClass &&name) const {
      return get(std::forward<NameClass>(name))
        .template dyn_cast_or_null<AttrClass>();
    }

  private:
    /// Return empty dictionary.
    static DictionaryAttr getEmpty(MLIRContext *context);

    /// Return empty dictionary. This is a special variant of the above method
    /// that is used by the MLIRContext to cache the empty dictionary instance.
    static DictionaryAttr getEmptyUnchecked(MLIRContext *context);

    /// Allow access to `getEmptyUnchecked`.
    friend MLIRContext;

  public:
  }];
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// FloatAttr
//===----------------------------------------------------------------------===//

def Builtin_FloatAttr : Builtin_Attr<"Float"> {
  let summary = "An Attribute containing a floating-point value";
  let description = [{
    Syntax:

    ```
    float-attribute ::= (float-literal (`:` float-type)?)
                      | (hexadecimal-literal `:` float-type)
    ```

    A float attribute is a literal attribute that represents a floating point
    value of the specified [float type](#floating-point-types). It can be
    represented in the hexadecimal form where the hexadecimal value is
    interpreted as bits of the underlying binary representation. This form is
    useful for representing infinity and NaN floating point values. To avoid
    confusion with integer attributes, hexadecimal literals _must_ be followed
    by a float type to define a float attribute.

    Examples:

    ```
    42.0         // float attribute defaults to f64 type
    42.0 : f32   // float attribute of f32 type
    0x7C00 : f16 // positive infinity
    0x7CFF : f16 // NaN (one of possible values)
    42 : f32     // Error: expected integer type
    ```
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        APFloatParameter<"">:$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const APFloat &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>,
    AttrBuilderWithInferredContext<(ins "Type":$type, "double":$value), [{
      if (type.isF64() || !type.isa<FloatType>())
        return $_get(type.getContext(), type, APFloat(value));

      // This handles, e.g., F16 because there is no APFloat constructor for it.
      bool unused;
      APFloat val(value);
      val.convert(type.cast<FloatType>().getFloatSemantics(),
                  APFloat::rmNearestTiesToEven, &unused);
      return $_get(type.getContext(), type, val);
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = APFloat;

    /// This function is used to convert the value to a double, even if it loses
    /// precision.
    double getValueAsDouble() const;
    static double getValueAsDouble(APFloat val);
  }];
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// IntegerAttr
//===----------------------------------------------------------------------===//

def Builtin_IntegerAttr : Builtin_Attr<"Integer"> {
  let summary = "An Attribute containing a integer value";
  let description = [{
    Syntax:

    ```
    integer-attribute ::= (integer-literal ( `:` (index-type | integer-type) )?)
                          | `true` | `false`
    ```

    An integer attribute is a literal attribute that represents an integral
    value of the specified integer or index type. `i1` integer attributes are
    treated as `boolean` attributes, and use a unique assembly format of either
    `true` or `false` depending on the value. The default type for non-boolean
    integer attributes, if a type is not specified, is signless 64-bit integer.

    Examples:

    ```mlir
    10 : i32
    10    // : i64 is implied here.
    true  // A bool, i.e. i1, value.
    false // A bool, i.e. i1, value.
    ```
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "APInt":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const APInt &":$value), [{
      if (type.isSignlessInteger(1))
        return BoolAttr::get(type.getContext(), value.getBoolValue());
      return $_get(type.getContext(), type, value);
    }]>,
    AttrBuilder<(ins "const APSInt &":$value), [{
      auto signedness = value.isSigned() ?
        IntegerType::Signed : IntegerType::Unsigned;
      auto type = IntegerType::get($_ctxt, value.getBitWidth(), signedness);
      return $_get(type.getContext(), type, value);
    }]>,
    AttrBuilderWithInferredContext<(ins "Type":$type, "int64_t":$value), [{
      // `index` has a defined internal storage width.
      if (type.isIndex()) {
        APInt apValue(IndexType::kInternalStorageBitWidth, value);
        return $_get(type.getContext(), type, apValue);
      }

      IntegerType intTy = type.cast<IntegerType>();
      APInt apValue(intTy.getWidth(), value, intTy.isSignedInteger());
      return $_get(type.getContext(), type, apValue);
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = APInt;

    /// Return the integer value as a 64-bit int. The attribute must be a
    /// signless integer.
    // TODO: Change callers to use getValue instead.
    int64_t getInt() const;
    /// Return the integer value as a signed 64-bit int. The attribute must be
    /// a signed integer.
    int64_t getSInt() const;
    /// Return the integer value as a unsigned 64-bit int. The attribute must be
    /// an unsigned integer.
    uint64_t getUInt() const;

    /// Return the value as an APSInt which carries the signed from the type of
    /// the attribute.  This traps on signless integers types!
    APSInt getAPSInt() const;

  private:
    /// Return a boolean attribute. This is a special variant of the `get`
    /// method that is used by the MLIRContext to cache the boolean IntegerAttr
    /// instances.
    static BoolAttr getBoolAttrUnchecked(IntegerType type, bool value);

    /// Allow access to `getBoolAttrUnchecked`.
    friend MLIRContext;

  public:
  }];
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// IntegerSetAttr
//===----------------------------------------------------------------------===//

def Builtin_IntegerSetAttr : Builtin_Attr<"IntegerSet"> {
  let summary = "An Attribute containing an IntegerSet object";
  let description = [{
    Syntax:

    ```
    integer-set-attribute ::= `affine_set` `<` integer-set `>`
    ```

    Examples:

    ```mlir
    affine_set<(d0) : (d0 - 2 >= 0)>
    ```
  }];
  let parameters = (ins "IntegerSet":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "IntegerSet":$value), [{
      return $_get(value.getContext(), value);
    }]>
  ];
  let extraClassDeclaration = "using ValueType = IntegerSet;";
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// OpaqueAttr
//===----------------------------------------------------------------------===//

def Builtin_OpaqueAttr : Builtin_Attr<"Opaque"> {
  let summary = "An opaque representation of another Attribute";
  let description = [{
    Syntax:

    ```
    opaque-attribute ::= dialect-namespace `<` attr-data `>`
    ```

    Opaque attributes represent attributes of non-registered dialects. These are
    attribute represented in their raw string form, and can only usefully be
    tested for attribute equality.

    Examples:

    ```mlir
    #dialect<"opaque attribute data">
    ```
  }];
  let parameters = (ins "StringAttr":$dialectNamespace,
                        StringRefParameter<"">:$attrData,
                        AttributeSelfTypeParameter<"">:$type);
  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$dialect,
                                        "StringRef":$attrData,
                                        "Type":$type), [{
      return $_get(dialect.getContext(), dialect, attrData, type);
    }]>
  ];
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// OpaqueElementsAttr
//===----------------------------------------------------------------------===//

def Builtin_OpaqueElementsAttr : Builtin_Attr<
    "OpaqueElements", [ElementsAttrInterface]
  > {
  let summary = "An opaque representation of a multi-dimensional array";
  let description = [{
    Syntax:

    ```
    opaque-elements-attribute ::= `opaque` `<` dialect-namespace  `,`
                                  hex-string-literal `>` `:`
                                  ( tensor-type | vector-type )
    ```

    An opaque elements attribute is an elements attribute where the content of
    the value is opaque. The representation of the constant stored by this
    elements attribute is only understood, and thus decodable, by the dialect
    that created it.

    Note: The parsed string literal must be in hexadecimal form.

    Examples:

    ```mlir
    opaque<"foo_dialect", "0xDEADBEEF"> : tensor<10xi32>
    ```
  }];

  // TODO: Provide a way to avoid copying content of large opaque
  // tensors This will likely require a new reference attribute kind.
  let parameters = (ins "StringAttr":$dialect,
                        StringRefParameter<"">:$value,
                        AttributeSelfTypeParameter<"", "ShapedType">:$type);
  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$dialect,
                                        "ShapedType":$type,
                                        "StringRef":$value), [{
      return $_get(dialect.getContext(), dialect, value, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "Dialect *":$dialect,
                                        "ShapedType":$type,
                                        "StringRef":$value), [{
      MLIRContext *ctxt = dialect->getContext();
      StringAttr dialectName = StringAttr::get(ctxt, dialect->getNamespace());
      return $_get(ctxt, dialectName, value, type);
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = StringRef;

    /// Decodes the attribute value using dialect-specific decoding hook.
    /// Returns false if decoding is successful. If not, returns true and leaves
    /// 'result' argument unspecified.
    bool decode(ElementsAttr &result);
  }];
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// SparseElementsAttr
//===----------------------------------------------------------------------===//

def Builtin_SparseElementsAttr : Builtin_Attr<
    "SparseElements", [ElementsAttrInterface]
  > {
  let summary = "An opaque representation of a multi-dimensional array";
  let description = [{
    Syntax:

    ```
    sparse-elements-attribute ::= `sparse` `<` attribute-value `,`
                                  attribute-value `>` `:`
                                  ( tensor-type | vector-type )
    ```

    A sparse elements attribute is an elements attribute that represents a
    sparse vector or tensor object. This is where very few of the elements are
    non-zero.

    The attribute uses COO (coordinate list) encoding to represent the sparse
    elements of the elements attribute. The indices are stored via a 2-D tensor
    of 64-bit integer elements with shape [N, ndims], which specifies the
    indices of the elements in the sparse tensor that contains non-zero values.
    The element values are stored via a 1-D tensor with shape [N], that supplies
    the corresponding values for the indices.

    Example:

    ```mlir
    sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>

    // This represents the following tensor:
    ///  [[1, 0, 0, 0],
    ///   [0, 0, 5, 0],
    ///   [0, 0, 0, 0]]
    ```
  }];

  let parameters = (ins AttributeSelfTypeParameter<"", "ShapedType">:$type,
                        "DenseIntElementsAttr":$indices,
                        "DenseElementsAttr":$values);
  let builders = [
    AttrBuilderWithInferredContext<(ins "ShapedType":$type,
                                        "DenseElementsAttr":$indices,
                                        "DenseElementsAttr":$values), [{
      assert(indices.getType().getElementType().isInteger(64) &&
             "expected sparse indices to be 64-bit integer values");
      assert((type.isa<RankedTensorType, VectorType>()) &&
             "type must be ranked tensor or vector");
      assert(type.hasStaticShape() && "type must have static shape");
      return $_get(type.getContext(), type,
                   indices.cast<DenseIntElementsAttr>(), values);
    }]>,
  ];
  let extraClassDeclaration = [{
    /// The set of data types that can be iterated by this attribute.
    // FIXME: Realistically, SparseElementsAttr could use ElementsAttr for the
    // value storage. This would mean dispatching to `values` when accessing
    // values. For now, we just add the types that can be iterated by
    // DenseElementsAttr.
    using NonContiguousIterableTypesT = std::tuple<
      Attribute,
      // Integer types.
      APInt, bool, uint8_t, uint16_t, uint32_t, uint64_t,
      int8_t, int16_t, int32_t, int64_t,
      short, unsigned short, int, unsigned, long, unsigned long,
      std::complex<APInt>, std::complex<uint8_t>, std::complex<uint16_t>,
      std::complex<uint32_t>, std::complex<uint64_t>, std::complex<int8_t>,
      std::complex<int16_t>, std::complex<int32_t>, std::complex<int64_t>,
      // Float types.
      APFloat, float, double,
      std::complex<APFloat>, std::complex<float>, std::complex<double>,
      // String types.
      StringRef
    >;
    using ElementsAttr::Trait<SparseElementsAttr>::getValues;

    /// Provide a `value_begin_impl` to enable iteration within ElementsAttr.
    template <typename T>
    auto value_begin_impl(OverloadToken<T>) const {
      return value_begin<T>();
    }

    template <typename T>
    using iterator =
        llvm::mapped_iterator<typename decltype(llvm::seq<ptrdiff_t>(0, 0))::iterator,
                              std::function<T(ptrdiff_t)>>;

    /// Return the values of this attribute in the form of the given type 'T'.
    /// 'T'  may be any of Attribute, APInt, APFloat, c++ integer/float types,
    /// etc.
    template <typename T> iterator<T> value_begin() const;

  private:
    /// Get a zero APFloat for the given sparse attribute.
    APFloat getZeroAPFloat() const;

    /// Get a zero APInt for the given sparse attribute.
    APInt getZeroAPInt() const;

    /// Get a zero attribute for the given sparse attribute.
    Attribute getZeroAttr() const;

    /// Utility methods to generate a zero value of some type 'T'. This is used
    /// by the 'iterator' class.
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
    typename std::enable_if<std::is_same<std::complex<APInt>, T>::value,
                            T>::type
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

  public:
  }];
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// StringAttr
//===----------------------------------------------------------------------===//

def Builtin_StringAttr : Builtin_Attr<"String"> {
  let summary = "An Attribute containing a string";
  let description = [{
    Syntax:

    ```
    string-attribute ::= string-literal (`:` type)?
    ```

    A string attribute is an attribute that represents a string literal value.

    Examples:

    ```mlir
    "An important string"
    "string with a type" : !dialect.string
    ```
  }];
  let parameters = (ins StringRefParameter<"">:$value,
                        AttributeSelfTypeParameter<"">:$type);
  let builders = [
    AttrBuilderWithInferredContext<(ins "const Twine &":$bytes, "Type":$type)>,
    /// Build an string attr with NoneType.
    AttrBuilder<(ins "const Twine &":$bytes)>,
    /// Build an empty string attr with NoneType.
    AttrBuilder<(ins)>
  ];
  let extraClassDeclaration = [{
    using ValueType = StringRef;

    /// If the value of this string is prefixed with a dialect namespace,
    /// returns the dialect corresponding to that namespace if it is loaded,
    /// nullptr otherwise. For example, the string `llvm.fastmathflags` would
    /// return the LLVM dialect, assuming it is loaded in the context.
    Dialect *getReferencedDialect() const;

    /// Enable conversion to StringRef.
    operator StringRef() const { return getValue(); }

    /// Returns the underlying string value
    StringRef strref() const { return getValue(); }

    /// Convert the underling value to an std::string.
    std::string str() const { return getValue().str(); }

    /// Return a pointer to the start of the string data.
    const char *data() const { return getValue().data(); }

    /// Return the number of bytes in this string.
    size_t size() const { return getValue().size(); }

    /// Iterate over the underlying string data.
    StringRef::iterator begin() const { return getValue().begin(); }
    StringRef::iterator end() const { return getValue().end(); }

    /// Compare the underlying string value to the one in `rhs`.
    int compare(StringAttr rhs) const {
      if (*this == rhs)
        return 0;
      return getValue().compare(rhs.getValue());
    }

  private:
    /// Return an empty StringAttr with NoneType type. This is a special variant
    /// of the `get` method that is used by the MLIRContext to cache the
    /// instance.
    static StringAttr getEmptyStringAttrUnchecked(MLIRContext *context);
    friend MLIRContext;
  public:
  }];
  let genStorageClass = 0;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// SymbolRefAttr
//===----------------------------------------------------------------------===//

def Builtin_SymbolRefAttr : Builtin_Attr<"SymbolRef"> {
  let summary = "An Attribute containing a symbolic reference to an Operation";
  let description = [{
    Syntax:

    ```
    symbol-ref-attribute ::= symbol-ref-id (`::` symbol-ref-id)*
    ```

    A symbol reference attribute is a literal attribute that represents a named
    reference to an operation that is nested within an operation with the
    `OpTrait::SymbolTable` trait. As such, this reference is given meaning by
    the nearest parent operation containing the `OpTrait::SymbolTable` trait. It
    may optionally contain a set of nested references that further resolve to a
    symbol nested within a different symbol table.

    This attribute can only be held internally by
    [array attributes](#array-attribute),
    [dictionary attributes](#dictionary-attribute)(including the top-level
    operation attribute dictionary) as well as attributes exposing it via
    the `SubElementAttrInterface` interface. Symbol reference attributes
    nested in types are currently not supported.

    **Rationale:** Identifying accesses to global data is critical to
    enabling efficient multi-threaded compilation. Restricting global
    data access to occur through symbols and limiting the places that can
    legally hold a symbol reference simplifies reasoning about these data
    accesses.

    See [`Symbols And SymbolTables`](../SymbolsAndSymbolTables.md) for more
    information.

    Examples:

    ```mlir
    @flat_reference
    @parent_reference::@nested_reference
    ```
  }];
  let parameters =
     (ins "StringAttr":$rootReference,
          ArrayRefParameter<"FlatSymbolRefAttr", "">:$nestedReferences);

  let builders = [
    AttrBuilderWithInferredContext<
      (ins "StringAttr":$rootReference,
           "ArrayRef<FlatSymbolRefAttr>":$nestedReferences), [{
      return $_get(rootReference.getContext(), rootReference, nestedReferences);
    }]>,
  ];
  let extraClassDeclaration = [{
    static SymbolRefAttr get(MLIRContext *ctx, StringRef value,
                             ArrayRef<FlatSymbolRefAttr> nestedRefs);
    /// Convenience getters for building a SymbolRefAttr with no path, which is
    /// known to produce a FlatSymbolRefAttr.
    static FlatSymbolRefAttr get(StringAttr value);
    static FlatSymbolRefAttr get(MLIRContext *ctx, StringRef value);

    /// Convenience getter for buliding a SymbolRefAttr based on an operation
    /// that implements the SymbolTrait.
    static FlatSymbolRefAttr get(Operation *symbol);

    /// Returns the name of the fully resolved symbol, i.e. the leaf of the
    /// reference path.
    StringAttr getLeafReference() const;
  }];
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// TypeAttr
//===----------------------------------------------------------------------===//

def Builtin_TypeAttr : Builtin_Attr<"Type", [
    DeclareAttrInterfaceMethods<SubElementAttrInterface>
  ]> {
  let summary = "An Attribute containing a Type";
  let description = [{
    Syntax:

    ```
    type-attribute ::= type
    ```

    A type attribute is an attribute that represents a
    [type object](#type-system).

    Examples:

    ```mlir
    i32
    !dialect.type
    ```
  }];
  let parameters = (ins "Type":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type), [{
      return $_get(type.getContext(), type);
    }]>,
  ];
  let extraClassDeclaration = "using ValueType = Type;";
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// UnitAttr
//===----------------------------------------------------------------------===//

def Builtin_UnitAttr : Builtin_Attr<"Unit"> {
  let summary = "An Attribute value of `unit` type";
  let description = [{
    Syntax:

    ```
    unit-attribute ::= `unit`
    ```

    A unit attribute is an attribute that represents a value of `unit` type. The
    `unit` type allows only one value forming a singleton set. This attribute
    value is used to represent attributes that only have meaning from their
    existence.

    One example of such an attribute could be the `swift.self` attribute. This
    attribute indicates that a function parameter is the self/context parameter.
    It could be represented as a [boolean attribute](#boolean-attribute)(true or
    false), but a value of false doesn't really bring any value. The parameter
    either is the self/context or it isn't.


    Examples:

    ```mlir
    // A unit attribute defined with the `unit` value specifier.
    func.func @verbose_form() attributes {dialectName.unitAttr = unit}

    // A unit attribute in an attribute dictionary can also be defined without
    // the value specifier.
    func.func @simple_form() attributes {dialectName.unitAttr}
    ```
  }];
  let extraClassDeclaration = [{
    static UnitAttr get(MLIRContext *context);
  }];
}

#endif // BUILTIN_ATTRIBUTES
