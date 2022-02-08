# Defining Dialect Attributes and Types

This document is a quickstart to defining dialect specific extensions to the
[attribute](../LangRef.md/#attributes) and [type](../LangRef.md/#type-system)
systems in MLIR. The main part of this tutorial focuses on defining types, but
the instructions are nearly identical for defining attributes.

See [MLIR specification](../LangRef.md) for more information about MLIR, the
structure of the IR, operations, etc.

## Types

Types in MLIR (like attributes, locations, and many other things) are
value-typed. This means that instances of `Type` are passed around by-value, as
opposed to by-pointer or by-reference. The `Type` class in itself acts as a
wrapper around an internal storage object that is uniqued within an instance of
an `MLIRContext`.

### Defining the type class

As described above, `Type` objects in MLIR are value-typed and rely on having an
implicit internal storage object that holds the actual data for the type. When
defining a new `Type` it isn't always necessary to define a new storage class.
So before defining the derived `Type`, it's important to know which of the two
classes of `Type` we are defining:

Some types are *singleton* in nature, meaning they have no parameters and only
ever have one instance, like the
[`index` type](../Dialects/Builtin.md/#indextype).

Other types are *parametric*, and contain additional information that
differentiates different instances of the same `Type`. For example the
[`integer` type](../Dialects/Builtin.md/#integertype) contains a bitwidth, with
`i8` and `i16` representing different instances of
[`integer` type](../Dialects/Builtin.md/#integertype). *Parametric* may also
contain a mutable component, which can be used, for example, to construct
self-referring recursive types. The mutable component *cannot* be used to
differentiate instances of a type class, so usually such types contain other
parametric components that serve to identify them.

#### Singleton types

For singleton types, we can jump straight into defining the derived type class.
Given that only one instance of such types may exist, there is no need to
provide our own storage class.

```c++
/// This class defines a simple parameterless singleton type. All derived types
/// must inherit from the CRTP class 'Type::TypeBase'. It takes as template
/// parameters the concrete type (SimpleType), the base class to use (Type),
/// the internal storage class (the default TypeStorage here), and an optional
/// set of type traits and interfaces(detailed below).
class SimpleType : public Type::TypeBase<SimpleType, Type, TypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// The `TypeBase` class provides the following utility methods for
  /// constructing instances of this type:
  /// static SimpleType get(MLIRContext *ctx);
};
```

#### Parametric types

Parametric types are those with additional construction or uniquing constraints,
that allow for representing multiple different instances of a single class. As
such, these types require defining a type storage class to contain the
parametric data.

##### Defining a type storage

Type storage objects contain all of the data necessary to construct and unique a
parametric type instance. The storage classes must obey the following:

*   Inherit from the base type storage class `TypeStorage`.
*   Define a type alias, `KeyTy`, that maps to a type that uniquely identifies
    an instance of the derived type.
*   Provide a construction method that is used to allocate a new instance of the
    storage class.
    -   `static Storage *construct(TypeStorageAllocator &, const KeyTy &key)`
*   Provide a comparison method between the storage and `KeyTy`.
    -   `bool operator==(const KeyTy &) const`
*   Provide a method to generate the `KeyTy` from a list of arguments passed to
    the uniquer. (Note: This is only necessary if the `KeyTy` cannot be default
    constructed from these arguments).
    -   `static KeyTy getKey(Args...&& args)`
*   Provide a method to hash an instance of the `KeyTy`. (Note: This is not
    necessary if an `llvm::DenseMapInfo<KeyTy>` specialization exists)
    -   `static llvm::hash_code hashKey(const KeyTy &)`

Let's look at an example:

```c++
/// Here we define a storage class for a ComplexType, that holds a non-zero
/// integer and an integer type.
struct ComplexTypeStorage : public TypeStorage {
  ComplexTypeStorage(unsigned nonZeroParam, Type integerType)
      : nonZeroParam(nonZeroParam), integerType(integerType) {}

  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy = std::pair<unsigned, Type>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(nonZeroParam, integerType);
  }

  /// Define a hash function for the key type.
  /// Note: This isn't necessary because std::pair, unsigned, and Type all have
  /// hash functions already available.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  /// Define a construction function for the key type.
  /// Note: This isn't necessary because KeyTy can be directly constructed with
  /// the given parameters.
  static KeyTy getKey(unsigned nonZeroParam, Type integerType) {
    return KeyTy(nonZeroParam, integerType);
  }

  /// Define a construction method for creating a new instance of this storage.
  static ComplexTypeStorage *construct(TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<ComplexTypeStorage>())
        ComplexTypeStorage(key.first, key.second);
  }

  /// The parametric data held by the storage class.
  unsigned nonZeroParam;
  Type integerType;
};
```

##### Type class definition

Now that the storage class has been created, the derived type class can be
defined. This structure is similar to [singleton types](#singleton-types),
except that a bit more of the functionality provided by `Type::TypeBase` is put
to use.

```c++
/// This class defines a parametric type. All derived types must inherit from
/// the CRTP class 'Type::TypeBase'. It takes as template parameters the
/// concrete type (ComplexType), the base class to use (Type), the storage
/// class (ComplexTypeStorage), and an optional set of traits and
/// interfaces(detailed below).
class ComplexType : public Type::TypeBase<ComplexType, Type,
                                          ComplexTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This method is used to get an instance of the 'ComplexType'. This method
  /// asserts that all of the construction invariants were satisfied. To
  /// gracefully handle failed construction, getChecked should be used instead.
  static ComplexType get(unsigned param, Type type) {
    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. All parameters to the storage class are passed after the
    // context.
    return Base::get(type.getContext(), param, type);
  }

  /// This method is used to get an instance of the 'ComplexType'. If any of the
  /// construction invariants are invalid, errors are emitted with the provided
  /// `emitError` function and a null type is returned.
  /// Note: This method is completely optional.
  static ComplexType getChecked(function_ref<InFlightDiagnostic()> emitError,
                                unsigned param, Type type) {
    // Call into a helper 'getChecked' method in 'TypeBase' to get a uniqued
    // instance of this type. All parameters to the storage class are passed
    // after the context.
    return Base::getChecked(emitError, type.getContext(), param, type);
  }

  /// This method is used to verify the construction invariants passed into the
  /// 'get' and 'getChecked' methods. Note: This method is completely optional.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              unsigned param, Type type) {
    // Our type only allows non-zero parameters.
    if (param == 0)
      return emitError() << "non-zero parameter passed to 'ComplexType'";
    // Our type also expects an integer type.
    if (!type.isa<IntegerType>())
      return emitError() << "non integer-type passed to 'ComplexType'";
    return success();
  }

  /// Return the parameter value.
  unsigned getParameter() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->nonZeroParam;
  }

  /// Return the integer parameter type.
  IntegerType getParameterType() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->integerType;
  }
};
```

#### Mutable types

Types with a mutable component are special instances of parametric types that
allow for mutating certain parameters after construction.

##### Defining a type storage

In addition to the requirements for the type storage class for parametric types,
the storage class for types with a mutable component must additionally obey the
following.

*   The mutable component must not participate in the storage `KeyTy`.
*   Provide a mutation method that is used to modify an existing instance of the
    storage. This method modifies the mutable component based on arguments,
    using `allocator` for any newly dynamically-allocated storage, and indicates
    whether the modification was successful.
    -   `LogicalResult mutate(StorageAllocator &allocator, Args ...&& args)`

Let's define a simple storage for recursive types, where a type is identified by
its name and may contain another type including itself.

```c++
/// Here we define a storage class for a RecursiveType that is identified by its
/// name and contains another type.
struct RecursiveTypeStorage : public TypeStorage {
  /// The type is uniquely identified by its name. Note that the contained type
  /// is _not_ a part of the key.
  using KeyTy = StringRef;

  /// Construct the storage from the type name. Explicitly initialize the
  /// containedType to nullptr, which is used as marker for the mutable
  /// component being not yet initialized.
  RecursiveTypeStorage(StringRef name) : name(name), containedType(nullptr) {}

  /// Define the comparison function.
  bool operator==(const KeyTy &key) const { return key == name; }

  /// Define a construction method for creating a new instance of the storage.
  static RecursiveTypeStorage *construct(StorageAllocator &allocator,
                                         const KeyTy &key) {
    // Note that the key string is copied into the allocator to ensure it
    // remains live as long as the storage itself.
    return new (allocator.allocate<RecursiveTypeStorage>())
        RecursiveTypeStorage(allocator.copyInto(key));
  }

  /// Define a mutation method for changing the type after it is created. In
  /// many cases, we only want to set the mutable component once and reject
  /// any further modification, which can be achieved by returning failure from
  /// this function.
  LogicalResult mutate(StorageAllocator &, Type body) {
    // If the contained type has been initialized already, and the call tries
    // to change it, reject the change.
    if (containedType && containedType != body)
      return failure();

    // Change the body successfully.
    containedType = body;
    return success();
  }

  StringRef name;
  Type containedType;
};
```

##### Type class definition

Having defined the storage class, we can define the type class itself.
`Type::TypeBase` provides a `mutate` method that forwards its arguments to the
`mutate` method of the storage and ensures the mutation happens safely.

```c++
class RecursiveType : public Type::TypeBase<RecursiveType, Type,
                                            RecursiveTypeStorage> {
public:
  /// Inherit parent constructors.
  using Base::Base;

  /// Creates an instance of the Recursive type. This only takes the type name
  /// and returns the type with uninitialized body.
  static RecursiveType get(MLIRContext *ctx, StringRef name) {
    // Call into the base to get a uniqued instance of this type. The parameter
    // (name) is passed after the context.
    return Base::get(ctx, name);
  }

  /// Now we can change the mutable component of the type. This is an instance
  /// method callable on an already existing RecursiveType.
  void setBody(Type body) {
    // Call into the base to mutate the type.
    LogicalResult result = Base::mutate(body);

    // Most types expect the mutation to always succeed, but types can implement
    // custom logic for handling mutation failures.
    assert(succeeded(result) &&
           "attempting to change the body of an already-initialized type");

    // Avoid unused-variable warning when building without assertions.
    (void) result;
  }

  /// Returns the contained type, which may be null if it has not been
  /// initialized yet.
  Type getBody() {
    return getImpl()->containedType;
  }

  /// Returns the name.
  StringRef getName() {
    return getImpl()->name;
  }
};
```

### Registering types with a Dialect

Once the dialect types have been defined, they must then be registered with a
`Dialect`. This is done via a similar mechanism to
[operations](../LangRef.md/#operations), with the `addTypes` method. The one
distinct difference with operations, is that when a type is registered the
definition of its storage class must be visible.

```c++
struct MyDialect : public Dialect {
  MyDialect(MLIRContext *context) : Dialect(/*name=*/"mydialect", context) {
    /// Add these defined types to the dialect.
    addTypes<SimpleType, ComplexType, RecursiveType>();
  }
};
```

### Parsing and Printing

As a final step after registration, a dialect must override the `printType` and
`parseType` hooks. These enable native support for round-tripping the type in
the textual `.mlir`.

```c++
class MyDialect : public Dialect {
public:
  /// Parse an instance of a type registered to the dialect.
  Type parseType(DialectAsmParser &parser) const override;

  /// Print an instance of a type registered to the dialect.
  void printType(Type type, DialectAsmPrinter &printer) const override;
};
```

These methods take an instance of a high-level parser or printer that allows for
easily implementing the necessary functionality. As described in the
[MLIR language reference](../LangRef.md/#dialect-types), dialect types are
generally represented as: `! dialect-namespace < type-data >`, with a pretty
form available under certain circumstances. The responsibility of our parser and
printer is to provide the `type-data` bits.

### Traits

Similarly to operations, `Type` classes may attach `Traits` that provide
additional mixin methods and other data. `Trait` classes may be specified via
the trailing template argument of the `Type::TypeBase` class. See the main
[`Trait`](../Traits.md) documentation for more information on defining and using
traits.

### Interfaces

Similarly to operations, `Type` classes may attach `Interfaces` to provide an
abstract interface into the type. See the main [`Interface`](../Interfaces.md)
documentation for more information on defining and using interfaces.

## Attributes

As stated in the introduction, the process for defining dialect attributes is
nearly identical to that of defining dialect types. That key difference is that
the things named `*Type` are generally now named `*Attr`.

*   `Type::TypeBase` -> `Attribute::AttrBase`
*   `TypeStorageAllocator` -> `AttributeStorageAllocator`
*   `addTypes` -> `addAttributes`

Aside from that, all of the interfaces for uniquing and storage construction are
all the same.

## Defining Custom Parsers and Printers using Assembly Formats

Attributes and types defined in ODS with a mnemonic can define an
`assemblyFormat` to declaratively describe custom parsers and printers. The
assembly format consists of literals, variables, and directives.

*   A literal is a keyword or valid punctuation enclosed in backticks, e.g. ``
    `keyword` `` or `` `<` ``.
*   A variable is a parameter name preceeded by a dollar sign, e.g. `$param0`,
    which captures one attribute or type parameter.
*   A directive is a keyword followed by an optional argument list that defines
    special parser and printer behaviour.

```tablegen
// An example type with an assembly format.
def MyType : TypeDef<My_Dialect, "MyType"> {
  // Define a mnemonic to allow the dialect's parser hook to call into the
  // generated parser.
  let mnemonic = "my_type";

  // Define two parameters whose C++ types are indicated in string literals.
  let parameters = (ins "int":$count, "AffineMap":$map);

  // Define the assembly format. Surround the format with less `<` and greater
  // `>` so that MLIR's printers use the pretty format.
  let assemblyFormat = "`<` $count `,` `map` `=` $map `>`";
}
```

The declarative assembly format for `MyType` results in the following format in
the IR:

```mlir
!my_dialect.my_type<42, map = affine_map<(i, j) -> (j, i)>
```

### Parameter Parsing and Printing

For many basic parameter types, no additional work is needed to define how these
parameters are parsed or printed.

*   The default printer for any parameter is `$_printer << $_self`, where
    `$_self` is the C++ value of the parameter and `$_printer` is an
    `AsmPrinter`.
*   The default parser for a parameter is
    `FieldParser<$cppClass>::parse($_parser)`, where `$cppClass` is the C++ type
    of the parameter and `$_parser` is an `AsmParser`.

Printing and parsing behaviour can be added to additional C++ types by
overloading these functions or by defining a `parser` and `printer` in an ODS
parameter class.

Example of overloading:

```c++
using MyParameter = std::pair<int, int>;

AsmPrinter &operator<<(AsmPrinter &printer, MyParameter param) {
  printer << param.first << " * " << param.second;
}

template <> struct FieldParser<MyParameter> {
  static FailureOr<MyParameter> parse(AsmParser &parser) {
    int a, b;
    if (parser.parseInteger(a) || parser.parseStar() ||
        parser.parseInteger(b))
      return failure();
    return MyParameter(a, b);
  }
};
```

Example of using ODS parameter classes:

```
def MyParameter : TypeParameter<"std::pair<int, int>", "pair of ints"> {
  let printer = [{ $_printer << $_self.first << " * " << $_self.second }];
  let parser = [{ [&] -> FailureOr<std::pair<int, int>> {
    int a, b;
    if ($_parser.parseInteger(a) || $_parser.parseStar() ||
        $_parser.parseInteger(b))
      return failure();
    return std::make_pair(a, b);
  }() }];
}
```

A type using this parameter with the assembly format `` `<` $myParam `>` `` will
look as follows in the IR:

```mlir
!my_dialect.my_type<42 * 24>
```

#### Non-POD Parameters

Parameters that aren't plain-old-data (e.g. references) may need to define a
`cppStorageType` to contain the data until it is copied into the allocator. For
example, `StringRefParameter` uses `std::string` as its storage type, whereas
`ArrayRefParameter` uses `SmallVector` as its storage type. The parsers for
these parameters are expected to return `FailureOr<$cppStorageType>`.

#### Optional Parameters

Optional parameters in the assembly format can be indicated by setting
`isOptional`. The C++ type of an optional parameter is required to satisfy the
following requirements:

*   is default-constructible
*   is contextually convertible to `bool`
*   only the default-constructed value is `false`

The parameter parser should return the default-constructed value to indicate "no
value present". The printer will guard on the presence of a value to print the
parameter.

If a value was not parsed for an optional parameter, then the parameter will be
set to its default-constructed C++ value. For example, `Optional<int>` will be
set to `llvm::None` and `Attribute` will be set to `nullptr`.

Only optional parameters or directives that only capture optional parameters can
be used in optional groups. An optional group is a set of elements optionally
printed based on the presence of an anchor. Suppose parameter `a` is an
`IntegerAttr`.

```
( `(` $a^ `)` ) : (`x`)?
```

In the above assembly format, if `a` is present (non-null), then it will be
printed as `(5 : i32)`. If it is not present, it will be `x`. Directives that
are used inside optional groups are allowed only if all captured parameters are
also optional.

### Assembly Format Directives

Attribute and type assembly formats have the following directives:

*   `params`: capture all parameters of an attribute or type.
*   `qualified`: mark a parameter to be printed with its leading dialect and
    mnemonic.
*   `struct`: generate a "struct-like" parser and printer for a list of
    key-value pairs.

#### `params` Directive

This directive is used to refer to all parameters of an attribute or type. When
used as a top-level directive, `params` generates a parser and printer for a
comma-separated list of the parameters. For example:

```tablegen
def MyPairType : TypeDef<My_Dialect, "MyPairType"> {
  let parameters = (ins "int":$a, "int":$b);
  let mnemonic = "pair";
  let assemblyFormat = "`<` params `>`";
}
```

In the IR, this type will appear as:

```mlir
!my_dialect.pair<42, 24>
```

The `params` directive can also be passed to other directives, such as `struct`,
as an argument that refers to all parameters in place of explicitly listing all
parameters as variables.

#### `qualified` Directive

This directive can be used to wrap attribute or type parameters such that they
are printed in a fully qualified form, i.e., they include the dialect name and
mnemonic prefix.

For example:

```tablegen
def OuterType : TypeDef<My_Dialect, "MyOuterType"> {
  let parameters = (ins MyPairType:$inner);
  let mnemonic = "outer";
  let assemblyFormat = "`<` pair `:` $inner `>`";
}
def OuterQualifiedType : TypeDef<My_Dialect, "MyOuterQualifiedType"> {
  let parameters = (ins MyPairType:$inner);
  let mnemonic = "outer_qual";
  let assemblyFormat = "`<` pair `:` qualified($inner) `>`";
}
```

In the IR, the types will appear as:

```mlir
!my_dialect.outer<pair : <42, 24>>
!my_dialect.outer_qual<pair : !mydialect.pair<42, 24>>
```

If optional parameters are present, they are not printed in the parameter list
if they are not present.

#### `struct` Directive

The `struct` directive accepts a list of variables to capture and will generate
a parser and printer for a comma-separated list of key-value pairs. If an
optional parameter is included in the `struct`, it can be elided. The variables
are printed in the order they are specified in the argument list **but can be
parsed in any order**. For example:

```tablegen
def MyStructType : TypeDef<My_Dialect, "MyStructType"> {
  let parameters = (ins StringRefParameter<>:$sym_name,
                        "int":$a, "int":$b, "int":$c);
  let mnemonic = "struct";
  let assemblyFormat = "`<` $sym_name `->` struct($a, $b, $c) `>`";
}
```

In the IR, this type can appear with any permutation of the order of the
parameters captured in the directive.

```mlir
!my_dialect.struct<"foo" -> a = 1, b = 2, c = 3>
!my_dialect.struct<"foo" -> b = 2, c = 3, a = 1>
```

Passing `params` as the only argument to `struct` makes the directive capture
all the parameters of the attribute or type. For the same type above, an
assembly format of `` `<` struct(params) `>` `` will result in:

```mlir
!my_dialect.struct<b = 2, sym_name = "foo", c = 3, a = 1>
```

The order in which the parameters are printed is the order in which they are
declared in the attribute's or type's `parameter` list.
