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
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for Builtin dialect ops.
class Builtin_Op<string mnemonic, list<Trait> traits = []> :
    Op<Builtin_Dialect, mnemonic, traits>;

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
      func.func @foo()
    }
    ```
  }];

  let arguments = (ins OptionalAttr<SymbolNameAttr>:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "($sym_name^)? attr-dict-with-keyword $bodyRegion";
  let builders = [OpBuilder<(ins CArg<"std::optional<StringRef>", "{}">:$name)>];
  let extraClassDeclaration = [{
    /// Construct a module from the given location with an optional name.
    static ModuleOp create(Location loc, std::optional<StringRef> name = std::nullopt);

    /// Return the name of this module if present.
    std::optional<StringRef> getName() { return getSymName(); }

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
    Pure
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
  let hasVerifier = 1;
}

#endif // BUILTIN_OPS
