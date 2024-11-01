# Defining Dialects

This document describes how to define [Dialects](../LangRef.md/#dialects).

[TOC]

## LangRef Refresher

Before diving into how to define these constructs, below is a quick refresher
from the [MLIR LangRef](../LangRef.md).

Dialects are the mechanism by which to engage with and extend the MLIR
ecosystem. They allow for defining new [attributes](../LangRef.md#attributes),
[operations](../LangRef.md#operations), and [types](../LangRef.md#type-system).
Dialects are used to model a variety of different abstractions; from traditional
[arithmetic](../Dialects/ArithOps.md) to
[pattern rewrites](../Dialects/PDLOps.md); and is one of the most fundamental
aspects of MLIR.

## Defining a Dialect

At the most fundamental level, defining a dialect in MLIR is as simple as
specializing the
[C++ `Dialect` class](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Dialect.h).
That being said, MLIR provides a powerful declaratively specification mechanism via
[TableGen](https://llvm.org/docs/TableGen/index.html); a generic language with
tooling to maintain records of domain-specific information; that simplifies the
definition process by automatically generating all of the necessary boilerplate
C++ code, significantly reduces maintainence burden when changing aspects of dialect
definitions, and also provides additional tools on top (such as
documentation generation). Given the above, the declarative specification is the
expected mechanism for defining new dialects, and is the method detailed within
this document. Before continuing, it is highly recommended that users review the
[TableGen Programmer's Reference](https://llvm.org/docs/TableGen/ProgRef.html)
for an introduction to its syntax and constructs.

Below showcases an example simple Dialect definition. We generally recommend defining
the Dialect class in a different `.td` file from the attributes, operations, types,
and other sub-components of the dialect to establish a proper layering between
the various different dialect components. It also prevents situations where you may
inadvertantly generate multiple definitions for some constructs. This recommendation
extends to all of the MLIR constructs, including [Interfaces](../Interfaces.md) for example.

```tablegen
// Include the definition of the necessary tablegen constructs for defining
// our dialect. 
include "mlir/IR/DialectBase.td"

// Here is a simple definition of a dialect.
def MyDialect : Dialect {
  let summary = "A short one line description of my dialect.";
  let description = [{
    My dialect is a very important dialect. This section contains a much more
    detailed description that documents all of the important pieces of information
    to know about the document.
  }];

  /// This is the namespace of the dialect. It is used to encapsulate the sub-components
  /// of the dialect, such as operations ("my_dialect.foo").
  let name = "my_dialect";

  /// The C++ namespace that the dialect, and its sub-components, get placed in.
  let cppNamespace = "::my_dialect";
}
```

The above showcases a very simple description of a dialect, but dialects have lots
of other capabilities that you may or may not need to utilize.

### Initialization

Every dialect must implement an initialization hook to add attributes, operations, types,
attach any desired interfaces, or perform any other necessary initialization for the
dialect that should happen on construction. This hook is declared for every dialect to
define, and has the form:

```c++
void MyDialect::initialize() {
  // Dialect initialization logic should be defined in here.
}
```

### Documentation

The `summary` and `description` fields allow for providing user documentation
for the dialect. The `summary` field expects a simple single-line string, with the
`description` field used for long and extensive documentation. This documentation can be 
used to generate markdown documentation for the dialect and is used by upstream
[MLIR dialects](https://mlir.llvm.org/docs/Dialects/).

### Class Name

The name of the C++ class which gets generated is the same as the name of our TableGen
dialect definition, but with any `_` characters stripped out. This means that if you name
your dialect `Foo_Dialect`, the generated C++ class would be `FooDialect`. In the example
above, we would get a C++ dialect named `MyDialect`.

### C++ Namespace

The namespace that the C++ class for our dialect, and all of its sub-components, is placed
under is specified by the `cppNamespace` field. By default, uses the name of the dialect as
the only namespace. To avoid placing in any namespace, use `""`. To specify nested namespaces,
use `"::"` as the delimiter between namespace, e.g., given `"A::B"`, C++ classes will be placed
within: `namespace A { namespace B { <classes> } }`.

Note that this works in conjunction with the dialect's C++ code. Depending on how the generated files
are included, you may want to specify a full namespace path or a partial one. In general, it's best
to use full namespaces whenever you can. This makes it easier for dialects within different namespaces,
and projects, to interact with each other.

### C++ Accessor Generation

When generating accessors for dialects and their components (attributes, operations, types, etc.),
we prefix the name with `get` and `set` respectively, and transform `snake_style` names to camel
case (`UpperCamel` when prefixed, and `lowerCamel` for individual variable names). For example, if an
operation were defined as:

```tablegen
def MyOp : MyDialect<"op"> {
  let arguments = (ins StrAttr:$value, StrAttr:$other_value);
}
```

It would have accessors generated for the `value` and `other_value` attributes as follows:

```c++
StringAttr MyOp::getValue();
void MyOp::setValue(StringAttr newValue);

StringAttr MyOp::getOtherValue();
void MyOp::setOtherValue(StringAttr newValue);
```

### Dependent Dialects

MLIR has a very large ecosystem, and contains dialects that server many different purposes. It
is quite common, given the above, that dialects may want to reuse certain components from other
dialects. This may mean generating operations from those dialects during canonicalization, reusing
attributes or types, etc. When a dialect has a dependency on another, i.e. when it constructs and/or
generally relies on the components of another dialect, a dialect dependency should be explicitly
recorded. An explicitly dependency ensures that dependent dialects are loaded alongside the
dialect. Dialect dependencies can be recorded using the `dependentDialects` dialects field:

```tablegen
def MyDialect : Dialect {
  // Here we register the Arithmetic and Func dialect as dependencies of our `MyDialect`.
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect"
  ];
}
```

### Extra declarations

The declarative Dialect definitions try to auto-generate as much logic and methods
as possible. With that said, there will always be long-tail cases that won't be covered.
For such cases, `extraClassDeclaration` can be used. Code within the `extraClassDeclaration`
field will be copied literally to the generated C++ Dialect class.

Note that `extraClassDeclaration` is a mechanism intended for long-tail cases by
power users; for not-yet-implemented widely-applicable cases, improving the
infrastructure is preferable.

### `hasConstantMaterializer`: Materializing Constants from Attributes

This field is utilized to materialize a constant operation from an `Attribute` value and
a `Type`. This is generally used when an operation within this dialect has been folded,
and a constant operation should be generated. `hasConstantMaterializer` is used to enable
materialization, and the `materializeConstant` hook is declared on the dialect. This
hook takes in an `Attribute` value, generally returned by `fold`, and produces a
"constant-like" operation that materializes that value. See the
[documentation for canonicalization](../Canonicalization.md) for a more in-depth
introduction to `folding` in MLIR.

Constant materialization logic can then be defined in the source file:

```c++
/// Hook to materialize a single constant operation from a given attribute value
/// with the desired resultant type. This method should use the provided builder
/// to create the operation without changing the insertion position. The
/// generated operation is expected to be constant-like. On success, this hook
/// should return the operation generated to represent the constant value.
/// Otherwise, it should return nullptr on failure.
Operation *MyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  ...
}
```

### `hasNonDefaultDestructor`: Providing a custom destructor

This field should be used when the Dialect class has a custom destructor, i.e.
when the dialect has some special logic to be run in the `~MyDialect`. In this case,
only the declaration of the destructor is generated for the Dialect class.

### Discardable Attribute Verification

As described by the [MLIR Language Reference](../LangRef.md#attributes),
*discardable attribute* are a type of attribute that has its semantics defined
by the dialect whose name prefixes that of the attribute. For example, if an
operation has an attribute named `gpu.contained_module`, the `gpu` dialect
defines the semantics and invariants, such as when and where it is valid to use,
of that attribute. To hook into this verification for attributes that are prefixed
by our dialect, several hooks on the Dialect may be used:

#### `hasOperationAttrVerify`

This field generates the hook for verifying when a discardable attribute of this dialect
has been used within the attribute dictionary of an operation. This hook has the form:

```c++
/// Verify the use of the given attribute, whose name is prefixed by the namespace of this
/// dialect, that was used in `op`s dictionary.
LogicalResult MyDialect::verifyOperationAttribute(Operation *op, NamedAttribute attribute);
```

#### `hasRegionArgAttrVerify`

This field generates the hook for verifying when a discardable attribute of this dialect
has been used within the attribute dictionary of a region entry block argument. Note that
the block arguments of a region entry block do not themselves have attribute dictionaries,
but some operations may provide special dictionary attributes that correspond to the arguments
of a region. For example, operations that implement `FunctionOpInterface` may have attribute
dictionaries on the operation that correspond to the arguments of entry block of the function.
In these cases, those operations will invoke this hook on the dialect to ensure the attribute
is verified. The hook necessary for the dialect to implement has the form:

```c++
/// Verify the use of the given attribute, whose name is prefixed by the namespace of this
/// dialect, that was used on the attribute dictionary of a region entry block argument.
/// Note: As described above, when a region entry block has a dictionary is up to the individual
/// operation to define. 
LogicalResult MyDialect::verifyRegionArgAttribute(Operation *op, unsigned regionIndex,
                                                  unsigned argIndex, NamedAttribute attribute);
```

#### `hasRegionResultAttrVerify`

This field generates the hook for verifying when a discardable attribute of this dialect
has been used within the attribute dictionary of a region result. Note that the results of a
region do not themselves have attribute dictionaries, but some operations may provide special
dictionary attributes that correspond to the results of a region. For example, operations that
implement `FunctionOpInterface` may have attribute dictionaries on the operation that correspond
to the results of the function. In these cases, those operations will invoke this hook on the
dialect to ensure the attribute is verified. The hook necessary for the dialect to implement
has the form:

```c++
/// Generate verification for the given attribute, whose name is prefixed by the namespace
/// of this dialect, that was used on the attribute dictionary of a region result.
/// Note: As described above, when a region entry block has a dictionary is up to the individual
/// operation to define. 
LogicalResult MyDialect::verifyRegionResultAttribute(Operation *op, unsigned regionIndex,
                                                     unsigned argIndex, NamedAttribute attribute);
```

### Operation Interface Fallback

Some dialects have an open ecosystem and don't register all of the possible operations. In such
cases it is still possible to provide support for implementing an `OpInterface` for these 
operations. When an operation isn't registered or does not provide an implementation for an 
interface, the query will fallback to the dialect itself. The `hasOperationInterfaceFallback`
field may be used to declare this fallback for operations:

```c++
/// Return an interface model for the interface with the given `typeId` for the operation
/// with the given name.
void *MyDialect::getRegisteredInterfaceForOp(TypeID typeID, StringAttr opName);
```

For a more detail description of the expected usages of this hook, view the detailed 
[interface documentation](../Interfaces.md#dialect-fallback-for-opinterface).

### Default Attribute/Type Parsers and Printers 

When a dialect registers an Attribute or Type, it must also override the respective
`Dialect::parseAttribute`/`Dialect::printAttribute` or
`Dialect::parseType`/`Dialect::printType` methods. In these cases, the dialect must
explicitly handle the parsing and printing of each individual attribute or type within
the dialect. If all of the attributes and types of the dialect provide a mnemonic,
however, these methods may be autogenerated by using the
`useDefaultAttributePrinterParser` and `useDefaultTypePrinterParser` fields. By default,
these fields are set to `1`(enabled), meaning that if a dialect needs to explicitly handle the
parser and printer of its Attributes and Types it should set these to `0` as necessary.

### Dialect-wide Canonicalization Patterns

Generally, [canonicalization](../Canonicalization.md) patterns are specific to individual 
operations within a dialect. There are some cases, however, that prompt canonicalization
patterns to be added to the dialect-level. For example, if a dialect defines a canonicalization
pattern that operates on an interface or trait, it can be beneficial to only add this pattern
once, instead of duplicating per-operation that implements that interface. To enable the
generation of this hook, the `hasCanonicalizer` field may be used. This will declare
the `getCanonicalizationPatterns` method on the dialect, which has the form:

```c++
/// Return the canonicalization patterns for this dialect:
void MyDialect::getCanonicalizationPatterns(RewritePatternSet &results) const;
```

See the documentation for [Canonicalization in MLIR](../Canonicalization.md) for a much more 
detailed description about canonicalization patterns.

## Defining an Extensible dialect

This section documents the design and API of the extensible dialects. Extensible
dialects are dialects that can be extended with new operations and types defined
at runtime. This allows for users to define dialects via with meta-programming,
or from another language, without having to recompile C++ code.

### Defining an extensible dialect

Dialects defined in C++ can be extended with new operations, types, etc., at
runtime by inheriting from `mlir::ExtensibleDialect` instead of `mlir::Dialect`
(note that `ExtensibleDialect` inherits from `Dialect`). The `ExtensibleDialect`
class contains the necessary fields and methods to extend the dialect at
runtime.

```c++
class MyDialect : public mlir::ExtensibleDialect {
    ...
}
```

For dialects defined in TableGen, this is done by setting the `isExtensible`
flag to `1`.

```tablegen
def Test_Dialect : Dialect {
  let isExtensible = 1;
  ...
}
```

An extensible `Dialect` can be casted back to `ExtensibleDialect` using
`llvm::dyn_cast`, or `llvm::cast`:

```c++
if (auto extensibleDialect = llvm::dyn_cast<ExtensibleDialect>(dialect)) {
    ...
}
```

### Defining a dynamic dialect

Dynamic dialects are extensible dialects that can be defined at runtime. They
are only populated with dynamic operations, types, and attributes. They can be
registered in a `DialectRegistry` with `insertDynamic`.

```c++
auto populateDialect = [](MLIRContext *ctx, DynamicDialect* dialect) {
  // Code that will be ran when the dynamic dialect is created and loaded.
  // For instance, this is where we register the dynamic operations, types, and
  // attributes of the dialect.
  ...
}

registry.insertDynamic("dialectName", populateDialect);
```

Once a dynamic dialect is registered in the `MLIRContext`, it can be retrieved
with `getOrLoadDialect`.

```c++
Dialect *dialect = ctx->getOrLoadDialect("dialectName");
```

### Defining an operation at runtime

The `DynamicOpDefinition` class represents the definition of an operation
defined at runtime. It is created using the `DynamicOpDefinition::get`
functions. An operation defined at runtime must provide a name, a dialect in
which the operation will be registered in, an operation verifier. It may also
optionally define a custom parser and a printer, fold hook, and more.

```c++
// The operation name, without the dialect name prefix.
StringRef name = "my_operation_name";

// The dialect defining the operation.
Dialect* dialect = ctx->getOrLoadDialect<MyDialect>();

// Operation verifier definition.
AbstractOperation::VerifyInvariantsFn verifyFn = [](Operation* op) {
    // Logic for the operation verification.
    ...
}

// Parser function definition.
AbstractOperation::ParseAssemblyFn parseFn =
    [](OpAsmParser &parser, OperationState &state) {
        // Parse the operation, given that the name is already parsed.
        ...    
};

// Printer function
auto printFn = [](Operation *op, OpAsmPrinter &printer) {
        printer << op->getName();
        // Print the operation, given that the name is already printed.
        ...
};

// General folder implementation, see AbstractOperation::foldHook for more
// information.
auto foldHookFn = [](Operation * op, ArrayRef<Attribute> operands, 
                                   SmallVectorImpl<OpFoldResult> &result) {
    ...
};

// Returns any canonicalization pattern rewrites that the operation
// supports, for use by the canonicalization pass.
auto getCanonicalizationPatterns = 
        [](RewritePatternSet &results, MLIRContext *context) {
    ...
}

// Definition of the operation.
std::unique_ptr<DynamicOpDefinition> opDef =
    DynamicOpDefinition::get(name, dialect, std::move(verifyFn),
        std::move(parseFn), std::move(printFn), std::move(foldHookFn),
        std::move(getCanonicalizationPatterns));
```

Once the operation is defined, it can be registered by an `ExtensibleDialect`:

```c++
extensibleDialect->registerDynamicOperation(std::move(opDef));
```

Note that the `Dialect` given to the operation should be the one registering
the operation.

### Using an operation defined at runtime

It is possible to match on an operation defined at runtime using their names:

```c++
if (op->getName().getStringRef() == "my_dialect.my_dynamic_op") {
    ...
}
```

An operation defined at runtime can be created by instantiating an
`OperationState` with the operation name, and using it with a rewriter
(for instance a `PatternRewriter`) to create the operation.

```c++
OperationState state(location, "my_dialect.my_dynamic_op",
                     operands, resultTypes, attributes);

rewriter.createOperation(state);
```


### Defining a type at runtime

Contrary to types defined in C++ or in TableGen, types defined at runtime can
only have as argument a list of `Attribute`.

Similarily to operations, a type is defined at runtime using the class
`DynamicTypeDefinition`, which is created using the `DynamicTypeDefinition::get`
functions. A type definition requires a name, the dialect that will register the
type, and a parameter verifier. It can also define optionally a custom parser
and printer for the arguments (the type name is assumed to be already
parsed/printed).

```c++
// The type name, without the dialect name prefix.
StringRef name = "my_type_name";

// The dialect defining the type.
Dialect* dialect = ctx->getOrLoadDialect<MyDialect>();

// The type verifier.
// A type defined at runtime has a list of attributes as parameters.
auto verifier = [](function_ref<InFlightDiagnostic()> emitError,
                   ArrayRef<Attribute> args) {
    ...
};

// The type parameters parser.
auto parser = [](DialectAsmParser &parser,
                 llvm::SmallVectorImpl<Attribute> &parsedParams) {
    ...
};

// The type parameters printer.
auto printer =[](DialectAsmPrinter &printer, ArrayRef<Attribute> params) {
    ...
};

std::unique_ptr<DynamicTypeDefinition> typeDef =
    DynamicTypeDefinition::get(std::move(name), std::move(dialect),
                               std::move(verifier), std::move(printer),
                               std::move(parser));
```

If the printer and the parser are ommited, a default parser and printer is
generated with the format `!dialect.typename<arg1, arg2, ..., argN>`.

The type can then be registered by the `ExtensibleDialect`:

```c++
dialect->registerDynamicType(std::move(typeDef));
```

### Parsing types defined at runtime in an extensible dialect

`parseType` methods generated by TableGen can parse types defined at runtime,
though overriden `parseType` methods need to add the necessary support for them.

```c++
Type MyDialect::parseType(DialectAsmParser &parser) const {
    ...
    
    // The type name.
    StringRef typeTag;
    if (failed(parser.parseKeyword(&typeTag)))
        return Type();

    // Try to parse a dynamic type with 'typeTag' name.
    Type dynType;
    auto parseResult = parseOptionalDynamicType(typeTag, parser, dynType);
    if (parseResult.has_value()) {
        if (succeeded(parseResult.getValue()))
            return dynType;
         return Type();
    }
    
    ...
}
```

### Using a type defined at runtime

Dynamic types are instances of `DynamicType`. It is possible to get a dynamic
type with `DynamicType::get` and `ExtensibleDialect::lookupTypeDefinition`.

```c++
auto typeDef = extensibleDialect->lookupTypeDefinition("my_dynamic_type");
ArrayRef<Attribute> params = ...;
auto type = DynamicType::get(typeDef, params);
```

It is also possible to cast a `Type` known to be defined at runtime to a
`DynamicType`.

```c++
auto dynType = type.cast<DynamicType>();
auto typeDef = dynType.getTypeDef();
auto args = dynType.getParams();
```

### Defining an attribute at runtime

Similar to types defined at runtime, attributes defined at runtime can only have
as argument a list of `Attribute`.

Similarily to types, an attribute is defined at runtime using the class
`DynamicAttrDefinition`, which is created using the `DynamicAttrDefinition::get`
functions. An attribute definition requires a name, the dialect that will
register the attribute, and a parameter verifier. It can also define optionally
a custom parser and printer for the arguments (the attribute name is assumed to
be already parsed/printed).

```c++
// The attribute name, without the dialect name prefix.
StringRef name = "my_attribute_name";

// The dialect defining the attribute.
Dialect* dialect = ctx->getOrLoadDialect<MyDialect>();

// The attribute verifier.
// An attribute defined at runtime has a list of attributes as parameters.
auto verifier = [](function_ref<InFlightDiagnostic()> emitError,
                   ArrayRef<Attribute> args) {
    ...
};

// The attribute parameters parser.
auto parser = [](DialectAsmParser &parser,
                 llvm::SmallVectorImpl<Attribute> &parsedParams) {
    ...
};

// The attribute parameters printer.
auto printer =[](DialectAsmPrinter &printer, ArrayRef<Attribute> params) {
    ...
};

std::unique_ptr<DynamicAttrDefinition> attrDef =
    DynamicAttrDefinition::get(std::move(name), std::move(dialect),
                               std::move(verifier), std::move(printer),
                               std::move(parser));
```

If the printer and the parser are ommited, a default parser and printer is
generated with the format `!dialect.attrname<arg1, arg2, ..., argN>`.

The attribute can then be registered by the `ExtensibleDialect`:

```c++
dialect->registerDynamicAttr(std::move(typeDef));
```

### Parsing attributes defined at runtime in an extensible dialect

`parseAttribute` methods generated by TableGen can parse attributes defined at
runtime, though overriden `parseAttribute` methods need to add the necessary
support for them.

```c++
Attribute MyDialect::parseAttribute(DialectAsmParser &parser,
                                    Type type) const override {
    ...
    // The attribute name.
    StringRef attrTag;
    if (failed(parser.parseKeyword(&attrTag)))
        return Attribute();

    // Try to parse a dynamic attribute with 'attrTag' name.
    Attribute dynAttr;
    auto parseResult = parseOptionalDynamicAttr(attrTag, parser, dynAttr);
    if (parseResult.has_value()) {
        if (succeeded(*parseResult))
            return dynAttr;
         return Attribute();
    }
```

### Using an attribute defined at runtime

Similar to types, attributes defined at runtime are instances of `DynamicAttr`.
It is possible to get a dynamic attribute with `DynamicAttr::get` and
`ExtensibleDialect::lookupAttrDefinition`.

```c++
auto attrDef = extensibleDialect->lookupAttrDefinition("my_dynamic_attr");
ArrayRef<Attribute> params = ...;
auto attr = DynamicAttr::get(attrDef, params);
```

It is also possible to cast an `Attribute` known to be defined at runtime to a
`DynamicAttr`.

```c++
auto dynAttr = attr.cast<DynamicAttr>();
auto attrDef = dynAttr.getAttrDef();
auto args = dynAttr.getParams();
```

### Implementation Details of Extensible Dialects

#### Extensible dialect

The role of extensible dialects is to own the necessary data for defined
operations and types. They also contain the necessary accessors to easily
access them.

In order to cast a `Dialect` back to an `ExtensibleDialect`, we implement the
`IsExtensibleDialect` interface to all `ExtensibleDialect`. The casting is done
by checking if the `Dialect` implements `IsExtensibleDialect` or not.

#### Operation representation and registration

Operations are represented in mlir using the `AbstractOperation` class. They are
registered in dialects the same way operations defined in C++ are registered,
which is by calling `AbstractOperation::insert`.

The only difference is that a new `TypeID` needs to be created for each
operation, since operations are not represented by a C++ class. This is done
using a `TypeIDAllocator`, which can allocate a new unique `TypeID` at runtime.

#### Type representation and registration

Unlike operations, types need to define a C++ storage class that takes care of
type parameters. They also need to define another C++ class to access that
storage. `DynamicTypeStorage` defines the storage of types defined at runtime,
and `DynamicType` gives access to the storage, as well as defining useful
functions. A `DynamicTypeStorage` contains a list of `Attribute` type
parameters, as well as a pointer to the type definition.

Types are registered using the `Dialect::addType` method, which expect a
`TypeID` that is generated using a `TypeIDAllocator`. The type uniquer also
register the type with the given `TypeID`. This mean that we can reuse our
single `DynamicType` with different `TypeID` to represent the different types
defined at runtime.

Since the different types defined at runtime have different `TypeID`, it is not
possible to use `TypeID` to cast a `Type` into a `DynamicType`. Thus, similar to
`Dialect`, all `DynamicType` define a `IsDynamicTypeTrait`, so casting a `Type`
to a `DynamicType` boils down to querying the `IsDynamicTypeTrait` trait.
