# Defining Dialect Attributes and Types

This document describes how to define dialect
[attributes](LangRef.md/#attributes) and [types](LangRef.md/#type-system).

[TOC]

## LangRef Refresher

Before diving into how to define these constructs, below is a quick refresher
from the [MLIR LangRef](LangRef.md).

### Attributes

Attributes are the mechanism for specifying constant data on operations in
places where a variable is never allowed - e.g. the comparison predicate of a
[`arith.cmpi` operation](Dialects/ArithmeticOps.md#arithcmpi-mlirarithcmpiop), or
the underlying value of a [`arith.constant` operation](Dialects/ArithmeticOps.md#arithconstant-mlirarithconstantop).
Each operation has an attribute dictionary, which associates a set of attribute
names to attribute values.

### Types

Every SSA value, such as operation results or block arguments, in MLIR has a type
defined by the type system. MLIR has an open type system with no fixed list of types,
and there are no restrictions on the abstractions they represent. For example, take
the following [Arithmetic AddI operation](Dialects/ArithmeticOps.md#arithaddi-mlirarithaddiop):

```mlir
  %result = arith.addi %lhs, %rhs : i64
```

It takes two input SSA values (`%lhs` and `%rhs`), and returns a single SSA
value (`%result`). The inputs and outputs of this operation are of type `i64`,
which is an instance of the [Builtin IntegerType](Dialects/Builtin.md#integertype).

## Attributes and Types

The C++ Attribute and Type classes in MLIR (like Ops, and many other things) are
value-typed. This means that instances of `Attribute` or `Type` are passed
around by-value, as opposed to by-pointer or by-reference. The `Attribute` and
`Type` classes act as wrappers around internal storage objects that are uniqued
within an instance of an `MLIRContext`.

The structure for defining Attributes and Types is nearly identical, with only a
few differences depending on the context. As such, a majority of this document
describes the process for defining both Attributes and Types side-by-side with
examples for both. If necessary, a section will explicitly call out any
distinct differences.

### Adding a new Attribute or Type definition

As described above, C++ Attribute and Type objects in MLIR are value-typed and
essentially function as helpful wrappers around an internal storage object that
holds the actual data for the type. Similarly to Operations, Attributes and Types
are defined declaratively via [TableGen](https://llvm.org/docs/TableGen/index.html);
a generic language with tooling to maintain records of domain-specific information.
It is highly recommended that users review the
[TableGen Programmer's Reference](https://llvm.org/docs/TableGen/ProgRef.html)
for an introduction to its syntax and constructs.

Starting the definition of a new attribute or type simply requires adding a
specialization for either the `AttrDef` or `TypeDef` class respectively. Instances
of the classes correspond to unqiue Attribute or Type classes.

Below show cases an example Attribute and Type definition. We generally recommend
defining Attribute and Type classes in different `.td` files to better encapsulate
the different constructs, and define a proper layering between them. This
recommendation extends to all of the MLIR constructs, including [Interfaces](Interfaces.md),
Operations, etc.

```tablegen
// Include the definition of the necessary tablegen constructs for defining
// our types.
include "mlir/IR/AttrTypeBase.td"

// It's common to define a base classes for types in the same dialect. This
// removes the need to pass in the dialect for each type, and can also be used
// to define a few fields ahead of time.
class MyDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<My_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Here is a simple definition of an "integer" type, with a width parameter.
def My_IntegerType : MyDialect_Type<"Integer", "int"> {
  let summary = "Integer type with arbitrary precision up to a fixed limit";
  let description = [{
    Integer types have a designated bit width.
  }];
  /// Here we defined a single parameter for the type, which is the bitwidth.
  let parameters = (ins "unsigned":$width);

  /// Here we define the textual format of the type declaratively, which will
  /// automatically generate parser and printer logic. This will allow for
  /// instances of the type to be output as, for example:
  ///
  ///    !my.int<10> // a 10-bit integer.
  ///
  let assemblyFormat = "`<` $width `>`";

  /// Indicate that our type will add additional verification to the parameters.
  let genVerifyDecl = 1;
}
```

Below is an example of an Attribute:

```tablegen
// Include the definition of the necessary tablegen constructs for defining
// our attributes.
include "mlir/IR/AttrTypeBase.td"

// It's common to define a base classes for attributes in the same dialect. This
// removes the need to pass in the dialect for each attribute, and can also be used
// to define a few fields ahead of time.
class MyDialect_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<My_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

// Here is a simple definition of an "integer" attribute, with a type and value parameter.
def My_IntegerAttr : MyDialect_Attr<"Integer", "int"> {
  let summary = "An Attribute containing a integer value";
  let description = [{
    An integer attribute is a literal attribute that represents an integral
    value of the specified integer type.
  }];
  /// Here we've defined two parameters, one is the `self` type of the attribute
  /// (i.e. the type of the Attribute itself), and the other is the integer value
  /// of the attribute.
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "APInt":$value);

  /// Here we've defined a custom builder for the type, that removes the need to pass
  /// in an MLIRContext instance; as it can be infered from the `type`.
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const APInt &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  /// Here we define the textual format of the attribute declaratively, which will
  /// automatically generate parser and printer logic. This will allow for
  /// instances of the attribute to be output as, for example:
  ///
  ///    #my.int<50> : !my.int<32> // a 32-bit integer of value 50.
  ///
  let assemblyFormat = "`<` $value `>`";

  /// Indicate that our attribute will add additional verification to the parameters.
  let genVerifyDecl = 1;

  /// Indicate to the ODS generator that we do not want the default builders,
  /// as we have defined our own simpler ones.
  let skipDefaultBuilders = 1;
}
```

### Class Name

The name of the C++ class which gets generated defaults to
`<classParamName>Attr` or `<classParamName>Type` for attributes and types
respectively. In the examples above, this was the `name` template parameter that
was provided to `MyDialect_Attr` and `MyDialect_Type`. For the definitions we
added above, we would get C++ classes named `IntegerType` and `IntegerAttr`
respectively. This can be explicitly overridden via the `cppClassName` field.

### Documentation

The `summary` and `description` fields allow for providing user documentation
for the attribute or type. The `summary` field expects a simple single-line
string, with the `description` field used for long and extensive documentation.
This documentation can be used to generate markdown documentation for the
dialect and is used by upstream
[MLIR dialects](https://mlir.llvm.org/docs/Dialects/).

### Mnemonic

The `mnemonic` field, i.e. the template parameters `attrMnemonic` and
`typeMnemonic` we specified above, are used to specify a name for use during
parsing. This allows for more easily dispatching to the current attribute or
type class when parsing IR. This field is generally optional, and custom
parsing/printing logic can be added without defining it, though most classes
will want to take advantage of the convenience it provides. This is why we
added it as a template parameter in the examples above.

### Parameters

The `parameters` field is a variable length list containing the attribute or
type's parameters. If no parameters are specified (the default), this type is
considered a singleton type (meaning there is only one possible instance).
Parameters in this list take the form: `"c++Type":$paramName`. Parameter types
with a C++ type that requires allocation when constructing the storage instance
in the context require one of the following:

- Utilize the `AttrParameter` or `TypeParameter` classes instead of the raw
  "c++Type" string. This allows for providing custom allocation code when using
  that parameter. `StringRefParameter` and `ArrayRefParameter` are examples of
  common parameter types that require allocation.
- Set the `genAccessors` field to 1 (the default) to generate accessor methods
  for each parameter (e.g. `int getWidth() const` in the Type example above).
- Set the `hasCustomStorageConstructor` field to `1` to generate a storage class
  that only declares the constructor, allowing for you to specialize it with
  whatever allocation code necessary.

#### AttrParameter, TypeParameter, and AttrOrTypeParameter

As hinted at above, these classes allow for specifying parameter types with
additional functionality. This is generally useful for complex parameters, or those
with additional invariants that prevent using the raw C++ class. Examples
include documentation (e.g. the `summary` and `syntax` field), the C++ type, a
custom allocator to use in the storage constructor method, a custom comparator
to decide if two instances of the parameter type are equal, etc. As the names
may suggest, `AttrParameter` is intended for parameters on Attributes,
`TypeParameter` for Type parameters, and `AttrOrTypeParameters` for either.

Below is an easy parameter pitfall, and highlights when to use these parameter
classes.

```tablegen
let parameters = (ins "ArrayRef<int>":$dims);
```

The above seems innocuous, but it is often a bug! The default storage
constructor blindly copies parameters by value. It does not know anything about
the types, meaning that the data of this ArrayRef will be copied as-is and is
likely to lead to use-after-free errors when using the created Attribute or
Type if the underlying does not have a lifetime exceeding that of the MLIRContext.
If the lifetime of the data can't be guaranteed, the `ArrayRef<int>` requires
allocation to ensure that its elements reside within the MLIRContext, e.g. with
`dims = allocator.copyInto(dims)`.

Here is a simple example for the exact situation above:

```tablegen
def ArrayRefIntParam : TypeParameter<"::llvm::ArrayRef<int>", "Array of int"> {
  let allocator = "$_dst = $_allocator.copyInto($_self);";
}

The parameter can then be used as so:

...
let parameters = (ins ArrayRefIntParam:$dims);
```

Below contains descriptions for other various available fields:

The `allocator` code block has the following substitutions:

- `$_allocator` is the TypeStorageAllocator in which to allocate objects.
- `$_dst` is the variable in which to place the allocated data.

The `comparator` code block has the following substitutions:

- `$_lhs` is an instance of the parameter type.
- `$_rhs` is an instance of the parameter type.

MLIR includes several specialized classes for common situations:

- `APFloatParameter` for APFloats.

- `StringRefParameter<descriptionOfParam>` for StringRefs.

- `ArrayRefParameter<arrayOf, descriptionOfParam>` for ArrayRefs of value types.

- `SelfAllocationParameter<descriptionOfParam>` for C++ classes which contain a
  method called `allocateInto(StorageAllocator &allocator)` to allocate itself
  into `allocator`.

- `ArrayRefOfSelfAllocationParameter<arrayOf, descriptionOfParam>` for arrays of
  objects which self-allocate as per the last specialization.

- `AttributeSelfTypeParameter` is a special AttrParameter that corresponds to
  the `Type` of the attribute. Only one parameter of the attribute may be of
  this parameter type.

### Traits

Similarly to operations, Attribute and Type classes may attach `Traits` that
provide additional mixin methods and other data. `Trait`s may be attached via
the trailing template argument, i.e. the `traits` list parameter in the example
above. See the main [`Trait`](Traits.md) documentation for more information
on defining and using traits.

### Interfaces

Attribute and Type classes may attach `Interfaces` to provide an virtual
interface into the Attribute or Type. `Interfaces` are added in the same way as
[Traits](#Traits), by using the `traits` list template parameter of the
`AttrDef` or `TypeDef`. See the main [`Interface`](Interfaces.md)
documentation for more information on defining and using interfaces.

### Builders

For each attribute or type, there are a few builders(`get`/`getChecked`)
automatically generated based on the parameters of the type. These are used to
construct instances of the corresponding attribute or type. For example, given
the following definition:

```tablegen
def MyAttrOrType : ... {
  let parameters = (ins "int":$intParam);
}
```

The following builders are generated:

```c++
// Builders are named `get`, and return a new instance for a given set of parameters.
static MyAttrOrType get(MLIRContext *context, int intParam);

// If `genVerifyDecl` is set to 1, the following method is also generated. This method
// is similar to `get`, but is failable and on error will return nullptr.
static MyAttrOrType getChecked(function_ref<InFlightDiagnostic()> emitError,
                               MLIRContext *context, int intParam);
```

If these autogenerated methods are not desired, such as when they conflict with
a custom builder method, the `skipDefaultBuilders` field may be set to 1 to
signal that the default builders should not be generated.

#### Custom builder methods

The default builder methods may cover a majority of the simple cases related to
construction, but when they cannot satisfy all of an attribute or type's needs,
additional builders may be defined via the `builders` field. The `builders`
field is a list of custom builders, either using `TypeBuilder` for types or
`AttrBuilder` for attributes, that are added to the attribute or type class. The
following will showcase several examples for defining builders for a custom type
`MyType`, the process is the same for attributes except that attributes use
`AttrBuilder` instead of `TypeBuilder`.

```tablegen
def MyType : ... {
  let parameters = (ins "int":$intParam);

  let builders = [
    TypeBuilder<(ins "int":$intParam)>,
    TypeBuilder<(ins CArg<"int", "0">:$intParam)>,
    TypeBuilder<(ins CArg<"int", "0">:$intParam), [{
      // Write the body of the `get` builder inline here.
      return Base::get($_ctxt, intParam);
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$typeParam), [{
      // This builder states that it can infer an MLIRContext instance from
      // its arguments.
      return Base::get(typeParam.getContext(), ...);
    }]>,
  ];
}
```

In this example, we provide several different convenience builders that are
useful in different scenarios. The `ins` prefix is common to many function
declarations in ODS, which use a TableGen [`dag`](#tablegen-syntax). What
follows is a comma-separated list of types (quoted string or `CArg`) and names
prefixed with the `$` sign. The use of `CArg` allows for providing a default
value to that argument. Let's take a look at each of these builders individually

The first builder will generate the declaration of a builder method that looks
like:

```tablegen
  let builders = [
    TypeBuilder<(ins "int":$intParam)>,
  ];
```

```c++
class MyType : /*...*/ {
  /*...*/
  static MyType get(::mlir::MLIRContext *context, int intParam);
};
```

This builder is identical to the one that will be automatically generated for
`MyType`. The `context` parameter is implicitly added by the generator, and is
used when building the Type instance (with `Base::get`). The distinction here is
that we can provide the implementation of this `get` method. With this style of
builder definition only the declaration is generated, the implementor of
`MyType` will need to provide a definition of `MyType::get`.

The second builder will generate the declaration of a builder method that looks
like:

```tablegen
  let builders = [
    TypeBuilder<(ins CArg<"int", "0">:$intParam)>,
  ];
```

```c++
class MyType : /*...*/ {
  /*...*/
  static MyType get(::mlir::MLIRContext *context, int intParam = 0);
};
```

The constraints here are identical to the first builder example except for the
fact that `intParam` now has a default value attached.

The third builder will generate the declaration of a builder method that looks
like:

```tablegen
  let builders = [
    TypeBuilder<(ins CArg<"int", "0">:$intParam), [{
      // Write the body of the `get` builder inline here.
      return Base::get($_ctxt, intParam);
    }]>,
  ];
```

```c++
class MyType : /*...*/ {
  /*...*/
  static MyType get(::mlir::MLIRContext *context, int intParam = 0);
};

MyType MyType::get(::mlir::MLIRContext *context, int intParam) {
  // Write the body of the `get` builder inline here.
  return Base::get(context, intParam);
}
```

This is identical to the second builder example. The difference is that now, a
definition for the builder method will be generated automatically using the
provided code block as the body. When specifying the body inline, `$_ctxt` may
be used to access the `MLIRContext *` parameter.

The fourth builder will generate the declaration of a builder method that looks
like:

```tablegen
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$typeParam), [{
      // This builder states that it can infer an MLIRContext instance from
      // its arguments.
      return Base::get(typeParam.getContext(), ...);
    }]>,
  ];
```

```c++
class MyType : /*...*/ {
  /*...*/
  static MyType get(Type typeParam);
};

MyType MyType::get(Type typeParam) {
  // This builder states that it can infer an MLIRContext instance from its
  // arguments.
  return Base::get(typeParam.getContext(), ...);
}
```

In this builder example, the main difference from the third builder example
there is that the `MLIRContext` parameter is no longer added. This is because
the builder used `TypeBuilderWithInferredContext` implies that the context
parameter is not necessary as it can be inferred from the arguments to the
builder.

### Parsing and Printing

If a mnemonic was specified, the `hasCustomAssemblyFormat` and `assemblyFormat`
fields may be used to specify the assembly format of an attribute or type. Attributes
and Types with no parameters need not use either of these fields, in which case
the syntax for the Attribute or Type is simply the mnemonic.

For each dialect, two "dispatch" functions will be created: one for parsing and
one for printing. These static functions placed alongside the class definitions
and have the following function signatures:

```c++
static ParseResult generatedAttributeParser(DialectAsmParser& parser, StringRef *mnemonic, Type attrType, Attribute &result);
static LogicalResult generatedAttributePrinter(Attribute attr, DialectAsmPrinter& printer);

static ParseResult generatedTypeParser(DialectAsmParser& parser, StringRef *mnemonic, Type &result);
static LogicalResult generatedTypePrinter(Type type, DialectAsmPrinter& printer);
```

The above functions should be added to the respective in your
`Dialect::printType` and `Dialect::parseType` methods, or consider using the
`useDefaultAttributePrinterParser` and `useDefaultTypePrinterParser` ODS Dialect
options if all attributes or types define a mnemonic.

The mnemonic, hasCustomAssemblyFormat, and assemblyFormat fields are optional.
If none are defined, the generated code will not include any parsing or printing
code and omit the attribute or type from the dispatch functions above. In this
case, the dialect author is responsible for parsing/printing in the respective
`Dialect::parseAttribute`/`Dialect::printAttribute` and
`Dialect::parseType`/`Dialect::printType` methods.

#### Using `hasCustomAssemblyFormat`

Attributes and types defined in ODS with a mnemonic can define an
`hasCustomAssemblyFormat` to specify custom parsers and printers defined in C++.
When set to `1` a corresponding `parse` and `print` method will be declared on
the Attribute or Type class to be defined by the user.

For Types, these methods will have the form:

- `static Type MyType::parse(AsmParser &parser)`

- `Type MyType::print(AsmPrinter &p) const`

For Attributes, these methods will have the form:

- `static Attribute MyAttr::parse(AsmParser &parser, Type attrType)`

- `Attribute MyAttr::print(AsmPrinter &p) const`

#### Using `assemblyFormat`

Attributes and types defined in ODS with a mnemonic can define an
`assemblyFormat` to declaratively describe custom parsers and printers. The
assembly format consists of literals, variables, and directives.

- A literal is a keyword or valid punctuation enclosed in backticks, e.g.
  `` `keyword` `` or `` `<` ``.
- A variable is a parameter name preceded by a dollar sign, e.g. `$param0`,
  which captures one attribute or type parameter.
- A directive is a keyword followed by an optional argument list that defines
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
  // `>` so that MLIR's printer uses the pretty format.
  let assemblyFormat = "`<` $count `,` `map` `=` $map `>`";
}
```

The declarative assembly format for `MyType` results in the following format in
the IR:

```mlir
!my_dialect.my_type<42, map = affine_map<(i, j) -> (j, i)>>
```

##### Parameter Parsing and Printing

For many basic parameter types, no additional work is needed to define how these
parameters are parsed or printed.

- The default printer for any parameter is `$_printer << $_self`, where `$_self`
  is the C++ value of the parameter and `$_printer` is an `AsmPrinter`.
- The default parser for a parameter is
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

```tablegen
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

###### Non-POD Parameters

Parameters that aren't plain-old-data (e.g. references) may need to define a
`cppStorageType` to contain the data until it is copied into the allocator. For
example, `StringRefParameter` uses `std::string` as its storage type, whereas
`ArrayRefParameter` uses `SmallVector` as its storage type. The parsers for
these parameters are expected to return `FailureOr<$cppStorageType>`.

To add a custom conversion between the `cppStorageType` and the C++ type of the
parameter, parameters can override `convertFromStorage`, which by default is
`"$_self"` (i.e., it attempts an implicit conversion from `cppStorageType`).

###### Optional Parameters

Optional parameters in the assembly format can be indicated by setting
`isOptional`. The C++ type of an optional parameter is required to satisfy the
following requirements:

- is default-constructible
- is contextually convertible to `bool`
- only the default-constructed value is `false`

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

###### Default-Valued Parameters

Optional parameters can be given default values by setting `defaultValue`, a
string of the C++ default value, or by using `DefaultValuedParameter`. If a
value for the parameter was not encountered during parsing, it is set to this
default value. If a parameter is equal to its default value, it is not printed.
The `comparator` field of the parameter is used, but if one is not specified,
the equality operator is used.

For example:

```tablegen
let parameters = (ins DefaultValuedParameter<"Optional<int>", "5">:$a)
let mnemonic = "default_valued";
let assemblyFormat = "(`<` $a^ `>`)?";
```

Which will look like:

```mlir
!test.default_valued     // a = 5
!test.default_valued<10> // a = 10
```

For optional `Attribute` or `Type` parameters, the current MLIR context is
available through `$_ctxt`. E.g.

```tablegen
DefaultValuedParameter<"IntegerType", "IntegerType::get($_ctxt, 32)">
```

##### Assembly Format Directives

Attribute and type assembly formats have the following directives:

- `params`: capture all parameters of an attribute or type.
- `qualified`: mark a parameter to be printed with its leading dialect and
  mnemonic.
- `struct`: generate a "struct-like" parser and printer for a list of key-value
  pairs.
- `custom`: dispatch a call to user-define parser and printer functions
- `ref`: in a custom directive, references a previously bound variable

###### `params` Directive

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

###### `qualified` Directive

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

###### `struct` Directive

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

###### `custom` and `ref` directive

The `custom` directive is used to dispatch calls to user-defined printer and
parser functions. For example, suppose we had the following type:

```tablegen
let parameters = (ins "int":$foo, "int":$bar);
let assemblyFormat = "custom<Foo>($foo) custom<Bar>($bar, ref($foo))";
```

The `custom` directive `custom<Foo>($foo)` will in the parser and printer
respectively generate calls to:

```c++
LogicalResult parseFoo(AsmParser &parser, FailureOr<int> &foo);
void printFoo(AsmPrinter &printer, int foo);
```

A previously bound variable can be passed as a parameter to a `custom` directive
by wrapping it in a `ref` directive. In the previous example, `$foo` is bound by
the first directive. The second directive references it and expects the
following printer and parser signatures:

```c++
LogicalResult parseBar(AsmParser &parser, FailureOr<int> &bar, int foo);
void printBar(AsmPrinter &printer, int bar, int foo);
```

More complex C++ types can be used with the `custom` directive. The only caveat
is that the parameter for the parser must use the storage type of the parameter.
For example, `StringRefParameter` expects the parser and printer signatures as:

```c++
LogicalResult parseStringParam(AsmParser &parser,
                               FailureOr<std::string> &value);
void printStringParam(AsmPrinter &printer, StringRef value);
```

The custom parser is considered to have failed if it returns failure or if any
bound parameters have failure values afterwards.

### Verification

If the `genVerifyDecl` field is set, additional verification methods are
generated on the class.

- `static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError, parameters...)`

These methods are used to verify the parameters provided to the attribute or
type class on construction, and emit any necessary diagnostics. This method is
automatically invoked from the builders of the attribute or type class.

- `AttrOrType getChecked(function_ref<InFlightDiagnostic()> emitError, parameters...)`

As noted in the [Builders](#Builders) section, these methods are companions to
`get` builders that are failable. If the `verify` invocation fails when these
methods are called, they return nullptr instead of asserting.

### Storage Classes

Somewhat alluded to in the sections above is the concept of a "storage class"
(often abbreviated to "storage"). Storage classes contain all of the data
necessary to construct and unique a attribute or type instance. These classes
are the "immortal" objects that get uniqued within an MLIRContext and get
wrapped by the `Attribute` and `Type` classes. Every Attribute or Type class has
a corresponding storage class, that can be accessed via the protected
`getImpl()` method.

In most cases the storage class is auto generated, but if necessary it can be
manually defined by setting the `genStorageClass` field to 0. The name and
namespace (defaults to `detail`) can additionally be controlled via the The
`storageClass` and `storageNamespace` fields.

#### Defining a storage class

User defined storage classes must adhere to the following:

- Inherit from the base type storage class of `AttributeStorage` or
  `TypeStorage` respectively.
- Define a type alias, `KeyTy`, that maps to a type that uniquely identifies an
  instance of the derived type. For example, this could be a `std::tuple` of all
  of the storage parameters.
- Provide a construction method that is used to allocate a new instance of the
  storage class.
  - `static Storage *construct(StorageAllocator &allocator, const KeyTy &key)`
- Provide a comparison method between an instance of the storage and the
  `KeyTy`.
  - `bool operator==(const KeyTy &) const`
- Provide a method to generate the `KeyTy` from a list of arguments passed to
  the uniquer when building an Attribute or Type. (Note: This is only necessary
  if the `KeyTy` cannot be default constructed from these arguments).
  - `static KeyTy getKey(Args...&& args)`
- Provide a method to hash an instance of the `KeyTy`. (Note: This is not
  necessary if an `llvm::DenseMapInfo<KeyTy>` specialization exists)
  - `static llvm::hash_code hashKey(const KeyTy &)`

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
  static ComplexTypeStorage *construct(StorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<ComplexTypeStorage>())
        ComplexTypeStorage(key.first, key.second);
  }

  /// The parametric data held by the storage class.
  unsigned nonZeroParam;
  Type integerType;
};
```

### Mutable attributes and types

Attributes and Types are immutable objects uniqued within an MLIRContext. That
being said, some parameters may be treated as "mutable" and modified after
construction. Mutable parameters should be reserved for parameters that can not
be reasonably initialized during construction time. Given the mutable component,
these parameters do not take part in the uniquing of the Attribute or Type.

TODO: Mutable parameters are currently not supported in the declarative
specification of attributes and types, and thus requires defining the Attribute
or Type class in C++.

#### Defining a mutable storage

In addition to the base requirements for a storage class, instances with a
mutable component must additionally adhere to the following:

- The mutable component must not participate in the storage `KeyTy`.
- Provide a mutation method that is used to modify an existing instance of the
  storage. This method modifies the mutable component based on arguments, using
  `allocator` for any newly dynamically-allocated storage, and indicates whether
  the modification was successful.
  - `LogicalResult mutate(StorageAllocator &allocator, Args ...&& args)`

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

#### Type class definition

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
  Type getBody() { return getImpl()->containedType; }

  /// Returns the name.
  StringRef getName() { return getImpl()->name; }
};
```

### Extra declarations

The declarative Attribute and Type definitions try to auto-generate as much
logic and methods as possible. With that said, there will always be long-tail
cases that won't be covered. For such cases, `extraClassDeclaration` can be
used. Code within the `extraClassDeclaration` field will be copied literally to
the generated C++ Attribute or Type class.

Note that `extraClassDeclaration` is a mechanism intended for long-tail cases by
power users; for not-yet-implemented widely-applicable cases, improving the
infrastructure is preferable.

### Registering with the Dialect

Once the attributes and types have been defined, they must then be registered
with the parent `Dialect`. This is done via the `addAttributes` and `addTypes`
methods. Note that when registering, the full definition of the storage classes
must be visible.

```c++
void MyDialect::initialize() {
    /// Add the defined attributes to the dialect.
  addAttributes<
#define GET_ATTRDEF_LIST
#include "MyDialect/Attributes.cpp.inc"
  >();

    /// Add the defined types to the dialect.
  addTypes<
#define GET_TYPEDEF_LIST
#include "MyDialect/Types.cpp.inc"
  >();
}
```
