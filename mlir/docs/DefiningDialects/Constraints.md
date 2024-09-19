# Constraints

[TOC]

## Attribute / Type Constraints

When defining the arguments of an operation in TableGen, users can specify
either plain attributes/types or use attribute/type constraints to levy
additional requirements on the attribute value or operand type.

```tablegen
def My_Type1 : MyDialect_Type<"Type1", "type1"> { ... }
def My_Type2 : MyDialect_Type<"Type2", "type2"> { ... }

// Plain type
let arguments = (ins MyType1:$val);
// Type constraint
let arguments = (ins AnyTypeOf<[MyType1, MyType2]>:$val);
```

`AnyTypeOf` is an example for a type constraints. Many useful type constraints
can be found in `mlir/IR/CommonTypeConstraints.td`. Additional verification
code is generated for type/attribute constraints. Type constraints can not only
be used when defining operation arguments, but also when defining type
parameters.

Optionally, C++ functions can be generated, so that type constraints can be
checked from C++. The name of the C++ function must be specified in the
`cppFunctionName` field. If no function name is specified, no C++ function is
emitted.

```tablegen
// Example: Element type constraint for VectorType
def Builtin_VectorTypeElementType : AnyTypeOf<[AnyInteger, Index, AnyFloat]> {
  let cppFunctionName = "isValidVectorTypeElementType";
}
```

The above example tranlates into the following C++ code:
```c++
bool isValidVectorTypeElementType(::mlir::Type type) {
  return (((::llvm::isa<::mlir::IntegerType>(type))) || ((::llvm::isa<::mlir::IndexType>(type))) || ((::llvm::isa<::mlir::FloatType>(type))));
}
```

An extra TableGen rule is needed to emit C++ code for type constraints. This
will generate only the declarations/definitions of the type constaraints that
are defined in the specified `.td` file, but not those that are in included
`.td` files.

```cmake
mlir_tablegen(<Your Dialect>TypeConstraints.h.inc -gen-type-constraint-decls)
mlir_tablegen(<Your Dialect>TypeConstraints.cpp.inc -gen-type-constraint-defs)
```

The generated `<Your Dialect>TypeConstraints.h.inc` will need to be included
whereever you are referencing the type constraint in C++. Note that no C++
namespace will be emitted by the code generator. The `#include` statements of
the `.h.inc`/`.cpp.inc` files should be wrapped in C++ namespaces by the user.
