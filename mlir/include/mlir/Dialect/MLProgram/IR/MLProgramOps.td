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
// GlobalLoadConstOp
//===----------------------------------------------------------------------===//

def MLProgram_GlobalLoadConstOp : MLProgram_Op<"global_load_const", [
    NoSideEffect,
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
    FlatSymbolRefAttr:$global
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $global attr-dict `:` type($result)
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
    NoSideEffect, HasParent<"SubgraphOp">, ReturnLike, Terminator
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
    build($_builder, $_state, llvm::None);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def MLProgram_ReturnOp : MLProgram_Op<"return", [
    NoSideEffect, HasParent<"FuncOp">, ReturnLike, Terminator
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
    build($_builder, $_state, llvm::None);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

#endif // MLPROGRAM_OPS
