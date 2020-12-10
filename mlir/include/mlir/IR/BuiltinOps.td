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

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

def Builtin_Dialect : Dialect {
  let summary =
    "A dialect containing the builtin Attributes, Operations, and Types";

  let name = "";
  let cppNamespace = "::mlir";
}

// Base class for Builtin dialect ops.
class Builtin_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Builtin_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : Builtin_Op<"func", [
  AffineScope, AutomaticAllocationScope, CallableOpInterface, FunctionLike,
  IsolatedFromAbove, Symbol
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

  let builders = [OpBuilderDAG<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<MutableDictionaryAttr>", "{}">:$argAttrs)
  >];
  let extraClassDeclaration = [{
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         iterator_range<dialect_attr_iterator> attrs);
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<MutableDictionaryAttr> argAttrs);

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

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getType().getResults(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }

  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<FuncOp>;

    /// Returns the number of arguments. This is a hook for
    /// OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }

    /// Returns the number of results. This is a hook for OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type. Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    /// safely.
    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  }];
  let parser = [{ return ::parseFuncOp(parser, result); }];
  let printer = [{ return ::print(*this, p); }];
  let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

def ModuleOp : Builtin_Op<"module", [
  AffineScope, IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol,
  SingleBlockImplicitTerminator<"ModuleTerminatorOp">
]> {
  let summary = "A top level container operation";
  let description = [{
    A `module` represents a top-level container operation. It contains a single
    SSACFG region containing a single block which can contain any
    operations. Operations within this region cannot implicitly capture values
    defined outside the module, i.e. Modules are `IsolatedFromAbove`. Modules
    have an optional symbol name which can be used to refer to them in
    operations.

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
  let builders = [OpBuilderDAG<(ins CArg<"Optional<StringRef>", "{}">:$name)>];
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
  }];
  let verifier = [{ return ::verify(*this); }];

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// ModuleTerminatorOp
//===----------------------------------------------------------------------===//

def ModuleTerminatorOp : Builtin_Op<"module_terminator", [
  Terminator, HasParent<"ModuleOp">
]> {
  let summary = "A pseudo op that marks the end of a module";
  let description = [{
    `module_terminator` is a special terminator operation for the body of a
    `module`, it has no semantic meaning beyond keeping the body of a `module`
    well-formed.
  }];
  let assemblyFormat = "attr-dict";
}

#endif // BUILTIN_OPS
