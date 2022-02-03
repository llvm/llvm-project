//===- BuiltinOps.td - Builtin operation definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the set of builtin MLIR operations, or the set of operations
// necessary for the validity of and defining the IR.
//
//===----------------------------------------------------------------------===//

#ifndef BUILTIN_OPS
#define BUILTIN_OPS

include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for Builtin dialect ops.
class Builtin_Op<string mnemonic, list<Trait> traits = []> :
    Op<Builtin_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : Builtin_Op<"func", [
  AffineScope, AutomaticAllocationScope, CallableOpInterface,
  FunctionOpInterface, IsolatedFromAbove, Symbol
]> {
  let summary = "An operation with a name containing a single `SSACFG` region";
  let description = [{
    Operations within the function cannot implicitly capture values defined
    outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    external references must use function arguments or attributes that establish
    a symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr). An external function declaration (used when
    referring to a function declared in some other module) has no body. While
    the MLIR textual form provides a nice inline syntax for function arguments,
    they are internally represented as “block arguments” to the first block in
    the region.

    Only dialect attribute names may be specified in the attribute dictionaries
    for function arguments, results, or the function itself.

    Example:

    ```mlir
    // External function definitions.
    func @abort()
    func @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64

    // A function that returns its argument twice:
    func @count(%x: i64) -> (i64, i64)
      attributes {fruit: "banana"} {
      return %x, %x: i64, i64
    }

    // A function with an argument attribute
    func @example_fn_arg(%x: i32 {swift.self = unit})

    // A function with a result attribute
    func @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})

    // A function with an attribute
    func @example_fn_attr() attributes {dialectName.attrName = false}
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$type,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];
  let extraClassDeclaration = [{
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         Operation::dialect_attr_range attrs);
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);

    /// Create a deep copy of this function and all of its blocks, remapping any
    /// operands that use values outside of the function using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for function arguments, these arguments are not
    /// included in the new function. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    FuncOp clone(BlockAndValueMapping &mapper);
    FuncOp clone();

    /// Clone the internal blocks and attributes from this function into dest.
    /// Any cloned blocks are appended to the back of dest. This function
    /// asserts that the attributes of the current function and dest are
    /// compatible.
    void cloneInto(FuncOp dest, BlockAndValueMapping &mapper);

    /// Returns the type of this function.
    /// FIXME: We should drive this via the ODS `type` param.
    FunctionType getType() { 
      return getTypeAttr().getValue().cast<FunctionType>();
    }

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getType().getResults(); }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getType().getResults(); }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];
  let parser = [{ return ::parseFuncOp(parser, result); }];
  let printer = [{ return ::print(*this, p); }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

def ModuleOp : Builtin_Op<"module", [
    AffineScope, IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol,
    OpAsmOpInterface
  ] # GraphRegionNoTerminator.traits> {
  let summary = "A top level container operation";
  let description = [{
    A `module` represents a top-level container operation. It contains a single
    [graph region](../LangRef.md#control-flow-and-ssacfg-regions) containing a single block
    which can contain any operations and does not have a terminator. Operations
    within this region cannot implicitly capture values defined outside the module,
    i.e. Modules are [IsolatedFromAbove](../Traits.md#isolatedfromabove). Modules have
    an optional [symbol name](../SymbolsAndSymbolTables.md) which can be used to refer
    to them in operations.

    Example:

    ```mlir
    module {
      func @foo()
    }
    ```
  }];

  let arguments = (ins OptionalAttr<SymbolNameAttr>:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "($sym_name^)? attr-dict-with-keyword $body";
  let builders = [OpBuilder<(ins CArg<"Optional<StringRef>", "{}">:$name)>];
  let extraClassDeclaration = [{
    /// Construct a module from the given location with an optional name.
    static ModuleOp create(Location loc, Optional<StringRef> name = llvm::None);

    /// Return the name of this module if present.
    Optional<StringRef> getName() { return sym_name(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// A ModuleOp may optionally define a symbol.
    bool isOptionalSymbol() { return true; }

    //===------------------------------------------------------------------===//
    // DataLayoutOpInterface Methods
    //===------------------------------------------------------------------===//

    DataLayoutSpecInterface getDataLayoutSpec();

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    static ::llvm::StringRef getDefaultDialect() {
      return "builtin";
    }
  }];
  let hasVerifier = 1;

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// UnrealizedConversionCastOp
//===----------------------------------------------------------------------===//

def UnrealizedConversionCastOp : Builtin_Op<"unrealized_conversion_cast", [
    DeclareOpInterfaceMethods<CastOpInterface>, NoSideEffect
  ]> {
  let summary = "An unrealized conversion from one set of types to another";
  let description = [{
    An `unrealized_conversion_cast` operation represents an unrealized
    conversion from one set of types to another, that is used to enable the
    inter-mixing of different type systems. This operation should not be
    attributed any special representational or execution semantics, and is
    generally only intended to be used to satisfy the temporary intermixing of
    type systems during the conversion of one type system to another.

    This operation may produce results of arity 1-N, and accept as input
    operands of arity 0-N.

    Example:

    ```mlir
    // An unrealized 0-1 conversion. These types of conversions are useful in
    // cases where a type is removed from the type system, but not all uses have
    // been converted. For example, imagine we have a tuple type that is
    // expanded to its element types. If only some uses of an empty tuple type
    // instance are converted we still need an instance of the tuple type, but
    // have no inputs to the unrealized conversion.
    %result = unrealized_conversion_cast to !bar.tuple_type<>

    // An unrealized 1-1 conversion.
    %result1 = unrealized_conversion_cast %operand : !foo.type to !bar.lowered_type

    // An unrealized 1-N conversion.
    %results2:2 = unrealized_conversion_cast %tuple_operand : !foo.tuple_type<!foo.type, !foo.type> to !foo.type, !foo.type

    // An unrealized N-1 conversion.
    %result3 = unrealized_conversion_cast %operand, %operand : !foo.type, !foo.type to !bar.tuple_type<!foo.type, !foo.type>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    ($inputs^ `:` type($inputs))? `to` type($outputs) attr-dict
  }];
  let hasFolder = 1;
}

#endif // BUILTIN_OPS
