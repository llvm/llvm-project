# Customizing Assembly Behavior

[TOC]

## Generating Aliases

`AsmPrinter` can generate aliases for frequently used types and attributes when not printing them in generic form. For example, `!my_dialect.type<a=3,b=4,c=5,d=tuple,e=another_type>` and `#my_dialect.attr<a=3>` can be aliased to `!my_dialect_type` and `#my_dialect_attr`.

There are mainly two ways to hook into the `AsmPrinter`. One is the attribute/type interface and the other is the dialect interface.

The attribute/type interface is the first hook to check. If no such hook is found, or the hook returns `OverridableAlias` (see definition below), then dialect interfaces are involved.

The dialect interface for one specific dialect could generate alias for all types/attributes, even when it does not "own" them. The `AsmPrinter` checks all dialect interfaces based on their order of registration. For example, the default alias `map` for `builtin` attribute `AffineMapAttr` could be overriden by the dialect interface for `my_dialect` as custom dialect is often registered after the `builtin` dialect.

```cpp
/// Holds the result of `OpAsm{Dialect,Attr,Type}Interface::getAlias` hook call.
enum class OpAsmAliasResult {
  /// The object (type or attribute) is not supported by the hook
  /// and an alias was not provided.
  NoAlias,
  /// An alias was provided, but it might be overriden by other hook.
  OverridableAlias,
  /// An alias was provided and it should be used
  /// (no other hooks will be checked).
  FinalAlias
};
```

If multiple types/attributes have the same alias from `getAlias` hooks, a number is appended to the alias to avoid conflicts.

### `OpAsmDialectInterface`

```cpp
#include "mlir/IR/OpImplementation.h"

struct MyDialectOpAsmDialectInterface : public OpAsmDialectInterface {
 public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream& os) const override {
    if (mlir::isa<MyType>(type)) {
      os << "my_dialect_type";
      return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }

  AliasResult getAlias(Attribute attr, raw_ostream& os) const override {
    if (mlir::isa<MyAttribute>(attr)) {
      os << "my_dialect_attr";
      return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }
};

void MyDialect::initialize() {
  // register the interface to the dialect
  addInterface<MyDialectOpAsmDialectInterface>();
}
```

### `OpAsmAttrInterface` and `OpAsmTypeInterface`

The easiest way to use these interfaces is toggling `genMnemonicAlias` in the tablegen file of the attribute/alias. It directly uses the mnemonic as alias. See [Defining Dialect Attributes and Types](/docs/DefiningDialects/AttributesAndTypes) for details.

If a more custom behavior is wanted, the following modification to the attribute/type should be made

1. Add `OpAsmAttrInterface` or `OpAsmTypeInterface` into its trait list.
2. Implement the `getAlias` method, either in tablegen or its cpp file.

```tablegen
include "mlir/IR/OpAsmInterface.td"

// Add OpAsmAttrInterface trait
def MyAttr : MyDialect_Attr<"MyAttr",
         [ OpAsmAttrInterface ] > {

  // This method could be put in the cpp file.
  let extraClassDeclaration = [{
    ::mlir::OpAsmAliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "alias_name";
      return ::mlir::OpAsmAliasResult::OverridableAlias;
    }
  }];
}
```

## Suggesting SSA/Block Names

An `Operation` can suggest the SSA name prefix using `OpAsmOpInterface`.

For example, `arith.constant` will suggest a name like `%c42_i32` for its result:

```tablegen
include "mlir/IR/OpAsmInterface.td"

def Arith_ConstantOp : Op<Arith_Dialect, "constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
...
}
```

And the corresponding method:

```cpp
// from https://github.com/llvm/llvm-project/blob/5ce271ef74dd3325993c827f496e460ced41af11/mlir/lib/Dialect/Arith/IR/ArithOps.cpp#L184
void arith::ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto type = getType();
  if (auto intCst = llvm::dyn_cast<IntegerAttr>(getValue())) {
    auto intType = llvm::dyn_cast<IntegerType>(type);

    // Sugar i1 constants with 'true' and 'false'.
    if (intType && intType.getWidth() == 1)
      return setNameFn(getResult(), (intCst.getInt() ? "true" : "false"));

    // Otherwise, build a complex name with the value and type.
    SmallString<32> specialNameBuffer;
    llvm::raw_svector_ostream specialName(specialNameBuffer);
    specialName << 'c' << intCst.getValue();
    if (intType)
      specialName << '_' << type;
    setNameFn(getResult(), specialName.str());
  } else {
    setNameFn(getResult(), "cst");
  }
}
```

Similarly, an `Operation` can suggest the name for its block arguments using `getAsmBlockArgumentNames` method in `OpAsmOpInterface`.

For custom block names, `OpAsmOpInterface` has a method `getAsmBlockNames` so that
the operation can suggest a custom prefix instead of a generic `^bb0`.

Alternatively, `OpAsmTypeInterface` provides a `getAsmName` method for scenarios where the name could be inferred from its type.

## Defining Default Dialect

An `Operation` can indicate that the nested region in it has a default dialect prefix, and the operations in the region could elide the dialect prefix.

For example, in a `func.func` op all `func` prefix could be omitted:

```tablegen
include "mlir/IR/OpAsmInterface.td"

def FuncOp : Func_Op<"func", [
  OpAsmOpInterface
  ...
]> {
  let extraClassDeclaration = [{
    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "func"; }
  }];
}
```

```mlir
func.func @main() {
  // actually func.call
  call @another()
}
```
