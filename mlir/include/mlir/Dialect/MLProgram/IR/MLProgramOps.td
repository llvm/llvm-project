//===- MLProgramOps.td - Structural ML Program Ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLPROGRAM_OPS
#define MLPROGRAM_OPS

include "mlir/Dialect/MLProgram/IR/MLProgramBase.td"
include "mlir/Dialect/MLProgram/IR/MLProgramAttributes.td"
include "mlir/Dialect/MLProgram/IR/MLProgramTypes.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class MLProgram_Op<string mnemonic, list<Trait> traits = []> :
    Op<MLProgram_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def MLProgram_FuncOp : MLProgram_Op<"func", [
    CallableOpInterface, FunctionOpInterface, IsolatedFromAbove,
    RegionKindInterface, Symbol
  ]> {
  let summary = "Function containing a single `SSACFG` region";
  let description = [{
    This simple function container represents callables in an ML program where
    the body is an `SSACFG` region. It must be terminated by a `return` op which
    yields values with the same arity and types as the `FunctionType` results
    of the containing `func`.

    This op is a `Symbol` but does not introduce a new `SymbolTable`. As such,
    it cannot represent nested symbols.

    Example:

    ```mlir
    ml_program.func private @some_extern(i32) -> i32
    ml_program.func @compute(%arg0 : i32) -> i32 {
      ml_program.return %arg0 : i32
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // RegionKindInterface Methods
    //===------------------------------------------------------------------===//
    static ::mlir::RegionKind getRegionKind(unsigned index) {
      return ::mlir::RegionKind::SSACFG;
    }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

def MLProgram_GlobalOp : MLProgram_Op<"global", [
    Symbol
  ]> {
  let summary = "Module level declaration of a global variable";
  let description = [{
    Declares a named global variable (or constant).

    A global contains a value of a specified type which can be accessed at
    runtime via appropriate load/store operations. It can be mutable or
    constant, optionally taking an initial value or declared as
    extern (in which case, the initial value is found in external storage
    by symbol name).

    Generally, the type of the global and the type of the initial value
    will be the same. However, for type hierarchies which can have a more
    generalized bounding type that can be assigned from a narrow type, this
    is allowed (but not verified).

    Examples:

    ```mlir
    // Constant global.
    ml_program.global @foobar(dense<4> : tensor<4xi32>) : tensor<?xi32>

    // Constant with external linkage.
    ml_program.global mutable @foobar(#ml_program.extern<tensor<4xi32>>)
      : tensor<?xi32>

    // Mutable global with an undefined initial value.
    ml_program.global mutable @foobar : tensor<?xi32>
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    UnitAttr:$is_mutable,
    OptionalAttr<AnyAttr>:$value,
    OptionalAttr<StrAttr>:$sym_visibility
  );

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    (`mutable` $is_mutable^)?
    $sym_name ``
    custom<TypedInitialValue>($type, $value)
    attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GlobalLoadOp
//===----------------------------------------------------------------------===//

def MLProgram_GlobalLoadOp : MLProgram_Op<"global_load", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Direct load of a mutable value from a global";
  let description = [{
    Performs a non-atomic, non-volatile, non-synchronized load from a global
    that may be mutable.

    It is fully expected that these constraints are not suitable for
    all situations, and alternative ops should be defined and used for more
    advanced cases.

    This op is side effecting and may not be valid to use in graph regions
    without additional consideration to evaluation order constraints. See
    `global_load_graph` for op which allows for explicit ordering constraints.

    Example:

    ```mlir
    %0 = ml_program.global_load @foobar : tensor<?xi32>
    ```
  }];

  let arguments = (ins
    Arg<SymbolRefAttr, "", [MemRead]>:$global
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $global `:` type($result) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Gets the corresponding GlobalOp (or nullptr).
    GlobalOp getGlobalOp(SymbolTableCollection &symbolTable);
  }];
}

//===----------------------------------------------------------------------===//
// GlobalLoadConstOp
//===----------------------------------------------------------------------===//

def MLProgram_GlobalLoadConstOp : MLProgram_Op<"global_load_const", [
    Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Direct load a constant value from a global";
  let description = [{
    Loads a constant (immutable) value from a global directly by symbol.

    This op is only legal for globals that are not mutable and exists because
    such a load can be considered to have no side effects.

    Example:

    ```mlir
    %0 = ml_program.global_load_const @foobar : tensor<?xi32>
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$global
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $global `:` type($result) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Gets the corresponding GlobalOp (or nullptr).
    GlobalOp getGlobalOp(SymbolTableCollection &symbolTable);
  }];
}

//===----------------------------------------------------------------------===//
// GlobalLoadGraphOp
//===----------------------------------------------------------------------===//

def MLProgram_GlobalLoadGraphOp : MLProgram_Op<"global_load_graph", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Direct load of a mutable value from a global in Graph region";
  let description = [{
    Performs a non-atomic, non-volatile, non-synchronized load from a global
    that may be mutable.

    It is fully expected that these constraints are not suitable for all
    situations, and alternative ops should be defined and used for more advanced
    cases.

    This op is side effecting and may not be valid to use in graph regions
    without additional consideration to evaluation order constraints.

    Example:

    ```mlir
    %0, %cstr = ml_program.global_load_graph @foobar
      ordering (%token -> !ml_program.token) : tensor<?xi32>
    ```
  }];

  let arguments = (ins
    Arg<SymbolRefAttr, "", [MemRead]>:$global,
    Variadic<MLProgram_TokenType>:$consumeTokens
  );
  let results = (outs
    AnyType:$result,
    MLProgram_TokenType:$produceToken
  );

  let assemblyFormat = [{
    $global `` custom<TokenOrdering>($consumeTokens, type($produceToken)) `:` type($result) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Gets the corresponding GlobalOp (or nullptr).
    GlobalOp getGlobalOp(SymbolTableCollection &symbolTable);
  }];
}

//===----------------------------------------------------------------------===//
// GlobalStoreOp
//===----------------------------------------------------------------------===//

def MLProgram_GlobalStoreOp : MLProgram_Op<"global_store", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Direct store of a value into a mutable global";
  let description = [{
    Performs a non-atomic, non-volatile, non-synchronized store to a mutable
    global.

    It is fully expected that these constraints are not suitable for
    all situations, and alternative ops should be defined and used for more
    advanced cases.

    This op is side effecting and may not be valid to use in graph regions
    without additional consideration to evaluation order constraints. See
    `global_store_graph` for op which allows for explicit ordering constraints.

    Example:

    ```mlir
    ml_program.global_store @foobar = %0 : tensor<?xi32>
    ```
  }];

  let arguments = (ins
    Arg<SymbolRefAttr, "", [MemWrite]>:$global,
    AnyType:$value
  );

  let assemblyFormat = [{
    $global `=` $value `:` type($value) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Gets the corresponding GlobalOp (or nullptr).
    GlobalOp getGlobalOp(SymbolTableCollection &symbolTable);
  }];
}

//===----------------------------------------------------------------------===//
// GlobalStoreGraphOp
//===----------------------------------------------------------------------===//

def MLProgram_GlobalStoreGraphOp : MLProgram_Op<"global_store_graph", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Direct store of a value into a mutable global";
  let description = [{
    Performs a non-atomic, non-volatile, non-synchronized store to a mutable
    global.

    It is fully expected that these constraints are not suitable for
    all situations, and alternative ops should be defined and used for more
    advanced cases.

    This op is side effecting and may not be valid to use in graph regions
    without additional consideration to evaluation order constraints.

    Example:

    ```mlir
    %token = ml_program.global_store @foobar = %0 : tensor<?xi32>
      ordering (%in_token -> !ml_program.token) : tensor<?xi32>
    ```
  }];

  let arguments = (ins
    Arg<SymbolRefAttr, "", [MemRead]>:$global,
    AnyType:$value,
    Variadic<MLProgram_TokenType>:$consumeTokens
  );
  let results = (outs
    MLProgram_TokenType:$produceToken
  );

  let assemblyFormat = [{
    $global `=` $value `` custom<TokenOrdering>($consumeTokens, type($produceToken)) `:` type($value) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Gets the corresponding GlobalOp (or nullptr).
    GlobalOp getGlobalOp(SymbolTableCollection &symbolTable);
  }];
}

//===----------------------------------------------------------------------===//
// SubgraphOp
//===----------------------------------------------------------------------===//

def MLProgram_SubgraphOp : MLProgram_Op<"subgraph", [
    CallableOpInterface, FunctionOpInterface, HasOnlyGraphRegion,
    IsolatedFromAbove, RegionKindInterface, SingleBlock, Symbol
  ]> {
  let summary = "An function containing a single `Graph` region";
  let description = [{
    This simple function container represents callables in an ML program where
    the body is a `Graph` region containing a single block. It must be
    terminated by an `output` op which yields values with the same arity and
    types as the `FunctionType` results of the containing `subgraph`.

    This op is a `Symbol` but does not introduce a new `SymbolTable`. As such,
    it cannot represented nested symbols.

    Example:

    ```mlir
    ml_program.subgraph private @some_extern(i32) -> i32
    ml_program.subgraph @compute(%arg0 : i32) -> i32 {
      ml_program.output %arg0 : i32
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def MLProgram_OutputOp : MLProgram_Op<"output", [
    Pure, HasParent<"SubgraphOp">, ReturnLike, Terminator
  ]> {
  let summary = "Outputs values from a subgraph function";
  let description = [{
    The `output` operation terminates a subgraph by yielding values
    to the caller.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def MLProgram_ReturnOp : MLProgram_Op<"return", [
    Pure, HasParent<"FuncOp">, ReturnLike, Terminator
  ]> {
  let summary = "Returns values from a `func` function";
  let description = [{
    The `return` operation terminates a `func` function by yielding values
    to the caller.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TokenOp
//===----------------------------------------------------------------------===//

def MLProgram_TokenOp : MLProgram_Op<"token", [
    Pure
  ]> {
  let summary = "Produces a new token value";
  let description = [{
    Token values are used to chain side effecting ops in a graph so as to
    establish an execution order. This op produces a token.
  }];

  let results = (outs
    MLProgram_TokenType:$token
  );

  let assemblyFormat = "attr-dict";
}

#endif // MLPROGRAM_OPS
