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

include "mlir/IR/BuiltinDialect.td"

// TODO: Currently the attributes defined in this file are prefixed with
// `Builtin_`.  This is to differentiate the attributes here with the ones in
// OpBase.td. We should remove the definitions in OpBase.td, and repoint users
// to this file instead.

// Base class for Builtin dialect attributes.
class Builtin_Attr<string name> : AttrDef<Builtin_Dialect, name> {
  let mnemonic = ?;
}

//===----------------------------------------------------------------------===//
// AffineMapAttr
//===----------------------------------------------------------------------===//

def Builtin_AffineMapAttr : Builtin_Attr<"AffineMap"> {
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
  let extraClassDeclaration = "using ValueType = AffineMap;";
  let skipDefaultBuilders = 1;
  let typeBuilder = "IndexType::get($_value.getContext())";
}

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//

def Builtin_ArrayAttr : Builtin_Attr<"Array"> {
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
    iterator_range<attr_value_iterator<AttrTy>> getAsRange() {
      return llvm::make_range(attr_value_iterator<AttrTy>(begin()),
                              attr_value_iterator<AttrTy>(end()));
    }
    template <typename AttrTy,
              typename UnderlyingTy = typename AttrTy::ValueType>
    auto getAsValueRange() {
      return llvm::map_range(getAsRange<AttrTy>(), [](AttrTy attr) {
        return static_cast<UnderlyingTy>(attr.getValue());
      });
    }
  }];
}

//===----------------------------------------------------------------------===//
// DictionaryAttr
//===----------------------------------------------------------------------===//

def Builtin_DictionaryAttr : Builtin_Attr<"Dictionary"> {
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
    AttrBuilder<(ins "ArrayRef<NamedAttribute>":$value)>
  ];
  let extraClassDeclaration = [{
    using ValueType = ArrayRef<NamedAttribute>;

    /// Construct a dictionary with an array of values that is known to already
    /// be sorted by name and uniqued.
    static DictionaryAttr getWithSorted(MLIRContext *context,
                                        ArrayRef<NamedAttribute> value);

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
  let parameters = (ins "Identifier":$dialectNamespace,
                        StringRefParameter<"">:$attrData,
                        AttributeSelfTypeParameter<"">:$type);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Identifier":$dialect,
                                        "StringRef":$attrData,
                                        "Type":$type), [{
      return $_get(dialect.getContext(), dialect, attrData, type);
    }]>
  ];
  bit genVerifyDecl = 1;
  // let skipDefaultBuilders = 1;
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
    AttrBuilderWithInferredContext<(ins "StringRef":$bytes,
                                        "Type":$type), [{
      return $_get(type.getContext(), bytes, type);
    }]>,
    AttrBuilder<(ins "StringRef":$bytes), [{
      return $_get($_ctxt, bytes, NoneType::get($_ctxt));
    }]>
  ];
  let extraClassDeclaration = "using ValueType = StringRef;";
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
    [array attributes](#array-attribute) and
    [dictionary attributes](#dictionary-attribute)(including the top-level
    operation attribute dictionary), i.e. no other attribute kinds such as
    Locations or extended attribute kinds.

    **Rationale:** Identifying accesses to global data is critical to
    enabling efficient multi-threaded compilation. Restricting global
    data access to occur through symbols and limiting the places that can
    legally hold a symbol reference simplifies reasoning about these data
    accesses.

    See [`Symbols And SymbolTables`](SymbolsAndSymbolTables.md) for more
    information.

    Examples:

    ```mlir
    @flat_reference
    @parent_reference::@nested_reference
    ```
  }];
  let parameters = (ins
    StringRefParameter<"">:$rootReference,
    ArrayRefParameter<"FlatSymbolRefAttr", "">:$nestedReferences
  );
  let extraClassDeclaration = [{
    static FlatSymbolRefAttr get(MLIRContext *ctx, StringRef value);

    /// Returns the name of the fully resolved symbol, i.e. the leaf of the
    /// reference path.
    StringRef getLeafReference() const;
  }];
}

//===----------------------------------------------------------------------===//
// TypeAttr
//===----------------------------------------------------------------------===//

def Builtin_TypeAttr : Builtin_Attr<"Type"> {
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
    func @verbose_form() attributes {dialectName.unitAttr = unit}

    // A unit attribute in an attribute dictionary can also be defined without
    // the value specifier.
    func @simple_form() attributes {dialectName.unitAttr}
    ```
  }];
  let extraClassDeclaration = [{
    static UnitAttr get(MLIRContext *context);
  }];
}

#endif // BUILTIN_ATTRIBUTES
