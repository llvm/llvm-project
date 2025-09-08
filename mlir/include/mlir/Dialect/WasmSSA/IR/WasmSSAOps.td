//===- WasmSSAOps.td - WasmSSA op definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WasmSSA_OPS
#define WasmSSA_OPS


include "mlir/Dialect/WasmSSA/IR/WasmSSABase.td"
include "mlir/Dialect/WasmSSA/IR/WasmSSATypes.td"
include "mlir/Dialect/WasmSSA/IR/WasmSSAInterfaces.td"

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

// Base class for WasmSSA operations.
// Most operations are made to match 1:1, only ignoring the stack-based approach of Wasm
// for an SSA based approach. In cases where operations match 1:1 the Wasm spec,
// no description is provided.
class WasmSSA_Op<string mnemonic, list<Trait> traits = []> :
    Op<WasmSSA_Dialect, mnemonic, traits>;

class WasmSSA_BlockLikeOp<string mnemonic, string summaryStr> :
  WasmSSA_Op<mnemonic, [Terminator, DeclareOpInterfaceMethods<LabelLevelOpInterface>]> {
  let summary = summaryStr;
  let arguments = (ins Variadic<WasmSSA_ValType>: $inputs);
  let regions = (region AnyRegion: $body);
  let successors = (successor AnySuccessor: $target);
  let extraClassDeclaration = [{
    ::mlir::Block* createBlock() {
      auto &block = getBody().emplaceBlock();
      for (auto input : getInputs())
        block.addArgument(input.getType(), input.getLoc());
      return &block;
    }
  }];
  let assemblyFormat = "(`(`$inputs^`)` `:` type($inputs))? attr-dict  `:` $body `>` $target";
}

def WasmSSA_BlockOp : WasmSSA_BlockLikeOp<"block", "Create a nesting level"> {}

def WasmSSA_LoopOp : WasmSSA_BlockLikeOp<"loop", "Create a nesting level similar to Block Op, except that it has itself as a successor."> {}

def WasmSSA_BlockReturnOp : WasmSSA_Op<"block_return", [Terminator,
    DeclareOpInterfaceMethods<LabelBranchingOpInterface>]> {
  let summary = "Return from the current block";
  let arguments = (ins Variadic<WasmSSA_ValType>: $inputs);
  let extraClassDeclaration = [{
    ::mlir::Block* getTarget();
  }];
  let description = [{
     Marks a return from the current block.

     Example:

    ```mlir
      wasmssa.block_return
    ```
  }];
  let assemblyFormat = "($inputs^ `:` type($inputs))? attr-dict";
}

def WasmSSA_BranchIfOp : WasmSSA_Op<"branch_if", [
    Terminator,
    DeclareOpInterfaceMethods<LabelBranchingOpInterface>]> {
  let summary = "Jump to target level if condition has non-zero value";
  let arguments = (ins I32: $condition,
                       UI32Attr: $exitLevel,
                       Variadic<WasmSSA_ValType>: $inputs);
  let description = [{
     Jump to target level if the condition is has a non-zero value.

     Example:

     ```mlir
     wasmssa.branch_if %a to level 0 with args(%b : i32) else ^bb1
     ```
    }];
  let successors = (successor AnySuccessor: $elseSuccessor);
  let assemblyFormat = "$condition `to` `level` $exitLevel (`with` `args`  `(`$inputs^ `:` type($inputs)`)`)?  `else` $elseSuccessor  attr-dict";
}

def WasmSSA_ConstOp : WasmSSA_Op<"const", [
    AllTypesMatch<["value", "result"]>,
    ConstantExprOpTrait]> {
  let summary = "Operator that represents a constant value";
  let description = [{
     Defines a constant value.

     Example:

     ```mlir
     // Example of integer constant
     %a = wasmssa.const 1 : i32

     // Example of floating point constant
     %b = wasmssa.const 9.000000e+00 : f64
     ```
    }];
  let arguments = (ins TypedAttrInterface: $value);
  let results = (outs WasmSSA_NumericType: $result);
  let assemblyFormat = "$value attr-dict";
}

def WasmSSA_FuncOp : WasmSSA_Op<"func", [
    AffineScope, AutomaticAllocationScope,
    DeclareOpInterfaceMethods<FunctionOpInterface, ["verifyBody"]>,
    IsolatedFromAbove,
    Symbol]> {
  let description = [{
    Represents a Wasm function definition.

    In Wasm function, locals and function arguments are interchangeable.
    They are for instance both accessed using `local.get` instruction.

    On the other hand, a function type is defined as a pair of tuples of Wasm value types.
    To model this, the wasm.func operation has:

    - A function type that represents the corresponding Wasm type (tuples of value types)

    - Arguments of the entry block of type `!wasm<local T>`, with T the corresponding type
     in the function type.

     Example:

     ```mlir
     // A simple function with no arguments that returns a float32
     wasmssa.func @my_f32_func() -> f32

     // A function that takes a local ref argument
     wasmssa.func @i64_wrap(%a: !wasmssa<local ref to i64>) -> i32
     ```
  }];
  let arguments = (ins SymbolNameAttr: $sym_name,
                     WasmSSA_FuncTypeAttr: $functionType,
                     OptionalAttr<DictArrayAttr>:$arg_attrs,
                     OptionalAttr<DictArrayAttr>:$res_attrs,
                     DefaultValuedAttr<StrAttr, "\"nested\"">:$sym_visibility);
  let regions = (region AnyRegion: $body);
  let extraClassDeclaration = [{

    /// Create the entry block for the function with parameters wrapped in local ref.
    ::mlir::Block* addEntryBlock();

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$symbol, "FunctionType":$funcType )>
  ];
  let hasCustomAssemblyFormat = 1;
}

def WasmSSA_FuncCallOp : WasmSSA_Op<"call"> {
  let summary = "Calling a Wasm function";
  let description = [{
     Emits a call to a defined function

     Example:

     ```mlir
     %a = wasmssa.call @func_0 : () -> i32
     ```
    }];
  let arguments = (ins FlatSymbolRefAttr: $callee,
                       Variadic<WasmSSA_ValType>:  $operands);
  let results = (outs Variadic<WasmSSA_ValType>: $results);
  let assemblyFormat = "$callee (`(`$operands^`)`)? attr-dict `:` functional-type($operands, $results)";
}

def WasmSSA_FuncImportOp : WasmSSA_Op<"import_func", [
    Symbol,
    CallableOpInterface,
    ImportOpInterface]> {
  let summary = "Importing a function variable";
  let description = [{
     Imports a function from another module

     Example:

     ```mlir
     // Imports foo(i32) -> () from the module my_module
     wasmssa.import_func "foo" from "my_module" as @func_0 {sym_visibility = "nested", type = (i32) -> ()}
     ```
    }];
  let arguments = (ins SymbolNameAttr: $sym_name,
                     StrAttr: $moduleName,
                     StrAttr: $importName,
                     WasmSSA_FuncTypeAttr: $type,
                     OptionalAttr<DictArrayAttr>:$arg_attrs,
                     OptionalAttr<DictArrayAttr>:$res_attrs,
                     OptionalAttr<StrAttr>:$sym_visibility);
  let extraClassDeclaration = [{
    bool isDeclaration() const { return true; }

    Region *getCallableRegion() { return nullptr; }

    ::llvm::ArrayRef<Type> getArgumentTypes() {
      return getType().getInputs();
    }

    ::llvm::ArrayRef<Type> getResultTypes() {
      return getType().getResults();
    }
  }];
  let builders = [
    OpBuilder<(ins "StringRef":$symbol,
                   "StringRef":$moduleName,
                   "StringRef":$importName,
                   "FunctionType": $type)>
  ];
  let assemblyFormat = "$importName `from` $moduleName `as` $sym_name attr-dict";
}

def WasmSSA_GlobalOp : WasmSSA_Op<"global", [
  AffineScope, AutomaticAllocationScope,
  IsolatedFromAbove, Symbol, ConstantExpressionInitializerOpTrait]> {
  let summary= "WebAssembly global value";
  let arguments = (ins SymbolNameAttr: $sym_name,
                     WasmSSA_ValTypeAttr: $type,
                     UnitAttr: $isMutable,
                     OptionalAttr<StrAttr>:$sym_visibility);
  let description = [{
    WebAssembly global variable.
    Body contains the initialization instructions for the variable value.
    The body must contain only instructions considered `const` in a webassembly context,
    such as `wasmssa.const` or `global.get`.

    Example:

    ```mlir
    // Define a global_var, a mutable i32 global variable equal to 10.
    wasmssa.global @global_var i32 mutable nested : {
          %[[VAL_0:.*]] = wasmssa.const 10 : i32
          wasmssa.return %[[VAL_0]] : i32
    }
    ```
  }];
  let regions = (region AnyRegion: $initializer);

  let builders = [
    OpBuilder<(ins "StringRef":$symbol,
                   "Type": $type,
                   "bool": $isMutable)>
  ];
  let hasCustomAssemblyFormat = 1;
}

def WasmSSA_GlobalImportOp : WasmSSA_Op<"import_global", [
    Symbol,
    ImportOpInterface]> {
  let summary = "Importing a global variable";
  let description = [{
     Imports a global from another module

     Example:

     ```mlir
     // Imports the "glob" i32 global from the module my_module as "global_0"
     wasmssa.import_global "glob" from "my_module" as @global_0 nested : i32
     ```
    }];
  let arguments = (ins SymbolNameAttr: $sym_name,
                     StrAttr: $moduleName,
                     StrAttr: $importName,
                     WasmSSA_ValTypeAttr: $type,
                     UnitAttr: $isMutable,
                     OptionalAttr<StrAttr>:$sym_visibility);
  let extraClassDeclaration = [{
    bool isDeclaration() const { return true; }
  }];
  let builders = [
    OpBuilder<(ins "StringRef":$symbol,
                   "StringRef":$moduleName,
                   "StringRef":$importName,
                   "Type": $type,
                   "bool": $isMutable)>
  ];
  let hasCustomAssemblyFormat = 1;
}

def WasmSSA_GlobalGetOp : WasmSSA_Op<"global_get", [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  ConstantExprOpTrait]> {
  let summary = "Returns the value of the global passed as argument.";
  let description = [{
     Retrieves the value of the global passed as argument and stores it in a
     variable

     Example:

     ```mlir
     // Gets the value of `@global_0` and stores its value in %a
     %a = wasmssa.global_get @global_0 : i32
     ```
    }];
  let arguments = (ins FlatSymbolRefAttr: $global);
  let results = (outs WasmSSA_ValType: $global_val);
  let assemblyFormat = "$global attr-dict `:` type($global_val)";
}

def WasmSSA_IfOp : WasmSSA_Op<"if", [Terminator,
    DeclareOpInterfaceMethods<LabelLevelOpInterface>]> {
  let summary = "Execute the if region if condition value is non-zero, the else region otherwise.";
  let description = [{
    Execute the if region if the condition is non-zero. Otherwise the else region is executed.
    The else region can be empty but must return the same datatype as the if region.
    If clauses can be nested.

     Example:

     ```mlir
     // Runs the if clause is %a is non-zero
     wasmssa.if %a {
        // Execute if %a is non-zero
     } else {
        // else clause
     }
     ```
    }];
  let arguments = (ins I32:$condition, Variadic<WasmSSA_ValType>: $inputs);
  let regions = (region AnyRegion: $if, AnyRegion: $else);
  let successors = (successor AnySuccessor: $target);
  let extraClassDeclaration = [{
    private:
    inline ::mlir::Block* createBlock(::mlir::Region& region) {
      assert(region.empty() && "Creating entry block on non empty region");
      assert(region.getParentOp() == this->getOperation() &&
        "Creating block for region that isn't part of the current op");
      auto &block = region.emplaceBlock();
      for (auto input : getInputs())
        block.addArgument(input.getType(), input.getLoc());
      return &block;
    }

    public:
    ::mlir::Block* createIfBlock() {
      return createBlock(getIf());
    }
    ::mlir::Block* createElseBlock() {
      return createBlock(getElse());
    }
  }];
  let assemblyFormat = "$condition (`(`$inputs^`)` `:` type($inputs))? attr-dict  `:` $if custom<ElseRegion>($else) `>` $target";
}

def WasmSSA_LocalOp : WasmSSA_Op<"local", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Declaration of local variable";
  let description = [{
    Declares a local variable

     Example:

     ```mlir
     // Declares `%a`, a float32 local
     %a = wasmssa.local of type f32
     ```
    }];
  let arguments = (ins WasmSSA_ValTypeAttr: $type);
  let results = (outs WasmSSA_LocalRef: $result);
  let assemblyFormat = "`of` `type` $type attr-dict";
}

def WasmSSA_LocalGetOp : WasmSSA_Op<"local_get", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Set local to value and return the operand.";
  let description = [{
    Gets the value of a local variable and returns a reference to it.

     Example:

     ```mlir
     // Retrieves a reference to `%a`, a float32 local
     %b = wasmssa.local_get %a : ref to f32
     ```
    }];
  let arguments = (ins WasmSSA_LocalRef: $localVar);
  let results = (outs WasmSSA_ValType: $result);
  let assemblyFormat = "$localVar `:` type($localVar) attr-dict";
}

def WasmSSA_LocalSetOp : WasmSSA_Op<"local_set"> {
  let summary = "Set local to given value";
  let description = [{
    Sets the value of a local variable.

     Example:

     ```mlir
     // Sets `%d`, to the value of `%c`
     wasmssa.local_set %d :  ref to i32 to %c : i32
     ```
    }];
  let arguments = (ins WasmSSA_LocalRef: $localVar,
                       WasmSSA_ValType: $value);
  let hasVerifier = 1;
  let assemblyFormat = "$localVar `:` type($localVar) `to` $value `:` type($value) attr-dict";
}

def WasmSSA_LocalTeeOp : WasmSSA_Op<"local_tee", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Set local to value and return the operand.";
  let description = [{
    Sets the value of a local variable and returns it.

     Example:

     ```mlir
     // Sets `%b`, to the value of `%c` and returns it in %a
      %a = wasmssa.local_tee %b :  ref to i32 to %c : i32
     ```
    }];
  let arguments = (ins WasmSSA_LocalRef: $localVar,
                       WasmSSA_ValType: $value);
  let results = (outs WasmSSA_ValType: $result);
  let hasVerifier = 1;
  let assemblyFormat = "$localVar `:` type($localVar) `to` $value `:` type($value) attr-dict";
}

def WasmSSA_MemOp : WasmSSA_Op<"memory", [Symbol]> {
  let summary= "WebAssembly memory definition";
  let description = [{
    Define a memory to be used by the program.
    Multiple memories can be defined in the same module.

     Example:

     ```mlir
     // Define the `mem_0` memory with defined bounds of 0 -> 65536
     wasmssa.memory @mem_0 !wasmssa<limit[0:65536]>
     ```
    }];
  let arguments = (ins SymbolNameAttr: $sym_name,
                     WasmSSA_LimitTypeAttr: $limits,
                     OptionalAttr<StrAttr>:$sym_visibility);
  let builders = [
    OpBuilder<(ins
    "::llvm::StringRef":$symbol,
    "wasmssa::LimitType":$limit)>
  ];

  let assemblyFormat = "$sym_name custom<WasmVisibility>($sym_visibility) $limits attr-dict";
}

def WasmSSA_MemImportOp : WasmSSA_Op<"import_mem", [Symbol, ImportOpInterface]> {
  let summary = "Importing a memory";
  let description = [{
    Import a memory from another module.

     Example:

     ```mlir
     // Import the memory `mem` from `my_module` as @mem_0
     wasmssa.import_mem "mem" from "my_module" as @mem_0 {limits = !wasmssa<limit[2:]>}
    ```
    }];
  let arguments = (ins SymbolNameAttr: $sym_name,
                     StrAttr: $moduleName,
                     StrAttr: $importName,
                     WasmSSA_LimitTypeAttr: $limits,
                     OptionalAttr<StrAttr>:$sym_visibility);
  let extraClassDeclaration = [{
     bool isDeclaration() const { return true; }
   }];
  let builders = [OpBuilder<(ins
    "::llvm::StringRef":$symbol,
    "::llvm::StringRef":$moduleName,
    "::llvm::StringRef":$importName,
    "wasmssa::LimitType":$limits)>];
  let assemblyFormat = "$importName `from` $moduleName `as` $sym_name attr-dict";
}

def WasmSSA_TableOp : WasmSSA_Op<"table", [Symbol]> {
  let summary= "WebAssembly table value";
  let arguments = (ins SymbolNameAttr: $sym_name,
                     WasmSSA_TableTypeAttr: $type,
                     OptionalAttr<StrAttr>:$sym_visibility);
  let builders = [OpBuilder<(ins
      "::llvm::StringRef":$symbol,
      "wasmssa::TableType":$type)>];
  let assemblyFormat = "$sym_name custom<WasmVisibility>($sym_visibility) $type attr-dict";
}

def WasmSSA_TableImportOp : WasmSSA_Op<"import_table", [Symbol, ImportOpInterface]> {
  let summary = "Importing a table";
  let description = [{
    Import a table from another module.

     Example:

     ```mlir
     // Import the table `table` from `my_module` as @table_0
     wasmssa.import_table "table" from "my_module" as @table_0 {type = !wasmssa<tabletype !wasmssa.funcref [2:]>}
    ```
    }];
  let arguments = (ins SymbolNameAttr: $sym_name,
                     StrAttr: $moduleName,
                     StrAttr: $importName,
                     WasmSSA_TableTypeAttr: $type,
                     OptionalAttr<StrAttr>:$sym_visibility);
  let extraClassDeclaration = [{
    bool isDeclaration() const { return true; }
  }];
  let assemblyFormat = "$importName `from` $moduleName `as` $sym_name attr-dict";
  let builders = [OpBuilder<(ins
      "::llvm::StringRef":$symbol,
      "::llvm::StringRef":$moduleName,
      "::llvm::StringRef":$importName,
      "wasmssa::TableType":$type)>];
}

def WasmSSA_ReturnOp : WasmSSA_Op<"return", [Terminator]> {
  let summary = "Return from the current function frame";
  let arguments = (ins Variadic<WasmSSA_ValType>:  $operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let builders = [
    OpBuilder<(ins)>
  ];
}

// ---- Numeric ops

class WasmSSA_BinaryNumericalOp<string mnemonic, string summaryStr, string descStr,
                       list<Type> validOpTypes> :
  WasmSSA_Op<mnemonic, [AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = summaryStr;
  let description = descStr;
  let arguments = (ins AnyTypeOf<validOpTypes>:$lhs, AnyTypeOf<validOpTypes>:$rhs);
  let results = (outs AnyTypeOf<validOpTypes>:$result);
  let assemblyFormat = "$lhs $rhs `:` type($lhs) attr-dict";
}

def WasmSSA_AddOp : WasmSSA_BinaryNumericalOp<"add",
    "Sum two values",
    [{Example:

     ```mlir
     %a = wasmssa.add %b %c : i32
    ```
    }],
    [WasmSSA_NumericType]>{}

def WasmSSA_AndOp : WasmSSA_BinaryNumericalOp<"and",
    "Compute the bitwise AND between two values",
    [{Example:

     ```mlir
     %a = wasmssa.and %b %c : i32
    ```
    }],
    [WasmSSA_NumericType]>{}

def WasmSSA_DivOp : WasmSSA_BinaryNumericalOp<"div",
    "Division between floating point values",
    [{Example:

     ```mlir
     %a = wasmssa.div %b %c : f32
    ```
    }],
    [WasmSSA_FPType]>{}

def WasmSSA_DivUIOp : WasmSSA_BinaryNumericalOp<"div_ui",
    "Divide values interpreted as unsigned int",
    [{Example:

     ```mlir
     %a = wasmssa.div_ui %b %c : i32
    ```
    }],
    [WasmSSA_IntegerType]>{}

def WasmSSA_DivSIOp : WasmSSA_BinaryNumericalOp<"div_si",
    "Divide values interpreted as signed int",
    [{Example:

     ```mlir
     %a = wasmssa.div_si %b %c : i32
    ```
    }],
    [WasmSSA_IntegerType]>{}

def WasmSSA_MulOp : WasmSSA_BinaryNumericalOp<"mul",
    "Multiply two values",
    [{Example:

     ```mlir
     %a = wasmssa.mul %b %c : i32
    ```
    }],
    [WasmSSA_NumericType]>{}

def WasmSSA_OrOp : WasmSSA_BinaryNumericalOp<"or",
    "Compute the bitwise OR of two values",
    [{Example:

     ```mlir
     %a = wasmssa.or %b %c : i32
    ```
    }],
    [WasmSSA_NumericType]>{}

def WasmSSA_SubOp : WasmSSA_BinaryNumericalOp<"sub",
    "Subtract two values",
    [{Example:

     ```mlir
     %a = wasmssa.sub %b %c : i32
    ```
    }],
    [WasmSSA_NumericType]>{}

def WasmSSA_RemUIOp : WasmSSA_BinaryNumericalOp<"rem_ui",
    "Calculate the remainder of dividing two integer values as an unsigned integer",
    [{Example:

     ```mlir
     %a = wasmssa.rem_ui %b %c : i32
    ```
    }],
    [WasmSSA_IntegerType]>{}

def WasmSSA_RemSIOp : WasmSSA_BinaryNumericalOp<"rem_si",
    "Calculate the remainder of dividing two integer values as signed integer",
    [{Example:

     ```mlir
     %a = wasmssa.rem_si %b %c : i32
    ```
    }],
    [WasmSSA_IntegerType]>{}

def WasmSSA_XOrOp : WasmSSA_BinaryNumericalOp<"xor",
    "Compute the bitwise XOR of two values",
    [{Example:

     ```mlir
     %a = wasmssa.xor %b %c : i32
    ```
    }],
    [WasmSSA_NumericType]>{}

def WasmSSA_MinOp : WasmSSA_BinaryNumericalOp<"min",
    "Compute the minimum of two floating point values.",
    [{Example:

     ```mlir
     %a = wasmssa.min %b %c : f32
    ```
    }],
    [WasmSSA_FPType]>{}

def WasmSSA_MaxOp : WasmSSA_BinaryNumericalOp<"max",
    "Compute the minimum of two floating point values.",
    [{Example:

     ```mlir
     %a = wasmssa.max %b %c : f32
    ```
    }],
    [WasmSSA_FPType]>{}

def WasmSSA_CopySignOp : WasmSSA_BinaryNumericalOp<"copysign",
    "Copy sign from one floating point value to the other.",
    [{Example:

     ```mlir
     %a = wasmssa.copysign %b %c : f32
    ```
    }],
    [WasmSSA_FPType]>{}

class WasmSSA_BinaryComparisonOp<string mnemonic, string summaryStr, string descStr,
                       list<Type> validOpTypes> :
  WasmSSA_Op<mnemonic, [AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = summaryStr;
  let description = descStr;
  let arguments = (ins AnyTypeOf<validOpTypes>:$lhs, AnyTypeOf<validOpTypes>:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs $rhs `:` type($lhs) `->` type($result) attr-dict";
}

def WasmSSA_EqOp : WasmSSA_BinaryComparisonOp<"eq",
    "Check if two values are equal",
    [{Example:

     ```mlir
     %a = wasmssa.eq %b %c : i32
    ```
    }],
    [WasmSSA_NumericType]>{}

def WasmSSA_NeOp : WasmSSA_BinaryComparisonOp<"ne",
    "Check if two values are different",
    [{Example:

     ```mlir
     %a = wasmssa.ne %b %c : i32
    ```
    }],
    [WasmSSA_NumericType]>{}

def WasmSSA_LtSIOp : WasmSSA_BinaryComparisonOp<"lt_si",
    "Check if a signed integer value is less than another",
    [{Example:

     ```mlir
     %a = wasmssa.lt_si %b %c : i32
    ```
    }],
    [WasmSSA_IntegerType]>{}

def WasmSSA_LtUIOp : WasmSSA_BinaryComparisonOp<"lt_ui",
    "Check if an unsigned integer value is less than another",
    [{Example:

     ```mlir
     %a = wasmssa.lt_ui %b %c : i32
    ```
    }],
    [WasmSSA_IntegerType]>{}

def WasmSSA_LeSIOp : WasmSSA_BinaryComparisonOp<"le_si",
    "Check if a signed integer value is less or equal to another",
    [{Example:

     ```mlir
     %a = wasmssa.le_si %b %c : i32
    ```
    }],
    [WasmSSA_IntegerType]>{}

def WasmSSA_LeUIOp : WasmSSA_BinaryComparisonOp<"le_ui",
    "Check if an unsigned integer value is less or equal to another",
    [{Example:

     ```mlir
     %a = wasmssa.le_ui %b %c : i32
    ```
    }],
    [WasmSSA_IntegerType]>{}

def WasmSSA_GtSIOp : WasmSSA_BinaryComparisonOp<"gt_si",
    "Check if a signed integer value is greater than another",
    [{Example:

     ```mlir
     %a = wasmssa.gt_si %b %c : i32
    ```
    }],
    [WasmSSA_IntegerType]>{}

def WasmSSA_GtUIOp : WasmSSA_BinaryComparisonOp<"gt_ui",
    "Check if an unsigned integer value is greater than another",
    [{Example:

     ```mlir
     %a = wasmssa.gt_ui %b %c : i32
    ```
    }],
    [WasmSSA_IntegerType]>{}

def WasmSSA_GeSIOp : WasmSSA_BinaryComparisonOp<"ge_si",
    "Check if a signed integer value is greater or equal to another",
    [{Example:

     ```mlir
     %a = wasmssa.ge_si %b %c : i32
    ```
    }],
    [WasmSSA_IntegerType]>{}

def WasmSSA_GeUIOp : WasmSSA_BinaryComparisonOp<"ge_ui",
    "Check if an unsigned integer value is greater or equal to another",
    [{Example:

     ```mlir
     %a = wasmssa.ge_ui %b %c : i32
    ```
    }],
    [WasmSSA_IntegerType]>{}

def WasmSSA_LtOp : WasmSSA_BinaryComparisonOp<"lt",
    "Check if a float value is less than another",
    [{Example:

     ```mlir
     %a = wasmssa.lt %b %c : i32
    ```
    }],
    [WasmSSA_FPType]>{}

def WasmSSA_LeOp : WasmSSA_BinaryComparisonOp<"le",
    "Check if a float value is less or equal to another",
    [{Example:

     ```mlir
     %a = wasmssa.le %b %c : i32
    ```
    }],
    [WasmSSA_FPType]>{}

def WasmSSA_GtOp : WasmSSA_BinaryComparisonOp<"gt",
    "Check if a float value is greater than another",
    [{Example:

     ```mlir
     %a = wasmssa.gt %b %c : i32
    ```
    }],
    [WasmSSA_FPType]>{}

def WasmSSA_GeOp : WasmSSA_BinaryComparisonOp<"ge",
    "Check if a float value is greater or equal to another",
    [{Example:

     ```mlir
     %a = wasmssa.ge %b %c : i32
    ```
    }],
    [WasmSSA_FPType]>{}

// Integer shift and rotate operations.
class WasmSSA_ShiftRotateOp<string mnemonic, string summaryStr, string descStr> :
  WasmSSA_Op<mnemonic, [AllTypesMatch<["val", "bits", "result"]>]> {
  let summary = summaryStr;
  let description = descStr;
  let arguments = (ins WasmSSA_IntegerType:$val, WasmSSA_IntegerType:$bits);
  let results = (outs WasmSSA_IntegerType:$result);
  let assemblyFormat = "$val `by` $bits `bits` `:` type($val) attr-dict";
}

def WasmSSA_ShLOp : WasmSSA_ShiftRotateOp<"shl",
    [{Consume an integer and an integer shift amount. The first
    integer shall be shifted left by N bits, where N is the value of the second
    integer.}],
    [{Example:

     ```mlir
     %a = wasmssa.shl %b by %c bits : i64
    ```
    }]
    >{}

def WasmSSA_ShRSOp : WasmSSA_ShiftRotateOp<"shr_s",
    [{Arithmetic right shift.

    Consume an integer and an integer shift amount. The first
    integer shall be shifted right by N bits, where N is the value of the
    second integer.

    Vacated bits on the left shall be filled with the sign bit.}],
    [{Example:

     ```mlir
     %a = wasmssa.shr_s %b by %c bits : i64
    ```
    }]
    >{}

def WasmSSA_ShRUOp : WasmSSA_ShiftRotateOp<"shr_u",
    [{Logical right shift.

    Consume an integer, and an integer shift amount. The first
    integer shall be shifted right by N bits, where N is the value of the
    second integer.

    Vacated bits on the left shall be filled with zeroes.}],
    [{Example:

     ```mlir
     %a = wasmssa.shr_u %b by %c bits : i64
    ```
    }]
    >{}

def WasmSSA_RotlOp : WasmSSA_ShiftRotateOp<"rotl",
    [{Rotate left.

    Consume an integer and an integer rotate. The first
    integer shall be rotated left by N bits, where N is the value of the
    second integer.}],
    [{Example:

     ```mlir
     %a = wasmssa.rotl %b by %c bits : i64
    ```
    }]
    >{}

def WasmSSA_RotrOp : WasmSSA_ShiftRotateOp<"rotr",
    [{Rotate right.

    Consume an integer, and an integer rotate. The first
    integer shall be rotated right by N bits, where N is the value of the
    second integer.}],
    [{Example:

     ```mlir
     %a = wasmssa.rotr %b by %c bits : i64
    ```
    }]
    >{}

class WasmSSA_ConversionOp<string mnemonic, string summaryStr, string descStr,
                       list<Type> ValidInputTypes,
                       list<Type> ValidOutputTypes> :
    WasmSSA_Op<mnemonic> {
  let summary = summaryStr;
  let description = descStr;
  let arguments = (ins AnyTypeOf<ValidInputTypes>:$input);
  let results = (outs AnyTypeOf<ValidOutputTypes>:$result);
  let assemblyFormat = "$input `:` type($input) `to` type($result)  attr-dict";
}

def WasmSSA_ConvertUOp : WasmSSA_ConversionOp<"convert_u",
    [{Convert integer, interpreted as binary encoded positive value, to floating-point value.

    Consume an integer and produces a floating point value containing the rounded value of the original operand. Rounding is round to nearest, tie to even.}],
    [{Example:

     ```mlir
     %a = wasmssa.convert_u %b : i32 to f64
    ```
    }],
    [WasmSSA_IntegerType],
    [WasmSSA_FPType]>{}

def WasmSSA_ConvertSOp : WasmSSA_ConversionOp<"convert_s",
    [{Convert integer interpreted as 2's complement signed value to floating-point value.

    Consume an integer and produces a floating point value containing the rounded value of the original operand. Rounding is round to nearest, tie to even.}],
    [{Example:

     ```mlir
     %a = wasmssa.convert_s %b : i32 to f64
    ```
    }],
    [WasmSSA_IntegerType],
    [WasmSSA_FPType]>{}

def WasmSSA_DemoteOp : WasmSSA_ConversionOp<"demote",
    "Convert a f64 value to f32",
    [{Example:

     ```mlir
     %a = wasmssa.demote %b : f64 to f32
    ```
    }],
    [F64],
    [F32]>{}

def WasmSSA_ExtendSI32Op : WasmSSA_Op<"extend_i32_s">{
  let summary = [{Sign extend i32 to i64.}];
  let description =  [{Example:

     ```mlir
     %a = wasmssa.extend_i32_s %b to i64
    ```
    }];
  let arguments = (ins I32:$input);
  let results = (outs I64:$result);
  let assemblyFormat = "$input `to` type($result)  attr-dict";
}

def WasmSSA_ExtendUI32Op : WasmSSA_Op<"extend_i32_u">{
  let summary = [{Zero extend i32 to i64.}];
  let description = [{Example:

     ```mlir
     %a = wasmssa.extend_i32_s %b to i64
    ```
    }];
  let arguments = (ins I32:$input);
  let results = (outs I64:$result);
  let assemblyFormat = "$input `to` type($result)  attr-dict";
}

def WasmSSA_ExtendLowBitsSOp : WasmSSA_Op<"extend", [AllTypesMatch<["input", "result"]>]> {
  let summary = "";
  let description = [{
  Extend low bytes of a value to fit a given width.
  For instance, signed extension from 8 low bits of the 32-bits integer value
  254 (0x000000FE) would produce the value -2 (0xFFFFFFFE).

  This corresponds to the `extendnn` instruction of Wasm, which shouldn't be
  confused with the `extend_inn` Wasm instruction, for which all input bits
  are used and widened to wider output type.
  In this operation, input and output types are the same.

  Example:

  ```mlir
  %a = wasmssa.extend 16 low bits from %[[VAL_0]]: i64
  ```
  }];
  let arguments = (ins WasmSSA_IntegerType:$input, Builtin_IntegerAttr: $bitsToTake);
  let results = (outs WasmSSA_IntegerType: $result);
  let hasVerifier = 1;
  let assemblyFormat = "$bitsToTake `low` `bits` `from` $input `:` type($input) attr-dict";
}

def WasmSSA_PromoteOp : WasmSSA_ConversionOp<"promote",
    "Get f64 representation of a f32 value.",
    [{Example:

     ```mlir
     %a = wasmssa.promote %b : f32 to f64
     ```
    }],
    [Builtin_Float32],
    [Builtin_Float64]>{}

def WasmSSA_WrapOp : WasmSSA_ConversionOp<"wrap",
    "Cast an i64 to i32 by using a wrapping mechanism: y = x mod 2^32",
    [{Example:

     ```mlir
     %a = wasmssa.wrap %b : i64 to i32
     ```
    }],
    [I64],
    [I32]>{}

// Reinterpret ops are basically all one-offs. They all have an unique,
// type-postfixed opcode, and support exactly one input and output type.
def WasmSSA_ReinterpretOp : WasmSSA_ConversionOp<"reinterpret",
    [{Reinterpret the value represented by a bit vector by
      bit-casting it to another type of same representation width.}],
    [{Example:

     ```mlir
     %a = wasmssa.reinterpret %b : f32 as i32
     ```
    }],
    [WasmSSA_NumericType], [WasmSSA_NumericType]>{
    let assemblyFormat = "$input `:` type($input) `as` type($result) attr-dict";
    let hasVerifier = 1;
}

class WasmSSA_UnaryNumericalOp<string mnemonic,
                            string summaryStr,
                            string descStr,
                            list<Type> validOpTypes> :
  WasmSSA_Op<mnemonic, [AllTypesMatch<["src", "result"]>]> {
    let summary = summaryStr;
    let description = descStr;
    let arguments = (ins AnyTypeOf<validOpTypes>:$src);
    let results = (outs AnyTypeOf<validOpTypes>:$result);
    let assemblyFormat = "$src`:` type($src) attr-dict";
}

def WasmSSA_AbsOp : WasmSSA_UnaryNumericalOp<"abs",
                                       "Floating point absolute value",
                                        [{Example:

                                         ```mlir
                                         %a = wasmssa.abs %b : f32
                                         ```
                                        }],
                                       [WasmSSA_FPType]>{}

def WasmSSA_CeilOp : WasmSSA_UnaryNumericalOp<"ceil",
                                       "Ceil rounding of floating point value",
                                        [{Example:

                                         ```mlir
                                         %a = wasmssa.ceil %b : f32
                                         ```
                                        }],
                                       [WasmSSA_FPType]>{}

def WasmSSA_FloorOp : WasmSSA_UnaryNumericalOp<"floor",
                                       "Floor rounding of floating point value",
                                        [{Example:

                                         ```mlir
                                         %a = wasmssa.floor %b : f32
                                         ```
                                        }],
                                       [WasmSSA_FPType]>{}

def WasmSSA_NegOp : WasmSSA_UnaryNumericalOp<"neg",
                                       "Floating point negation",
                                        [{Example:

                                         ```mlir
                                         %a = wasmssa.neg %b : f32
                                         ```
                                        }],
                                       [WasmSSA_FPType]>{}

def WasmSSA_SqrtOp : WasmSSA_UnaryNumericalOp<"sqrt",
                                       "Floating point square root",
                                        [{Example:

                                         ```mlir
                                         %a = wasmssa.sqrt %b : f32
                                         ```
                                        }],
                                       [WasmSSA_FPType]>{}

def WasmSSA_TruncOp : WasmSSA_UnaryNumericalOp<"trunc",
                                       "Trunc of floating point value",
                                        [{Example:

                                         ```mlir
                                         %a = wasmssa.trunc %b : f32
                                         ```
                                        }],
                                       [WasmSSA_FPType]>{}

def WasmSSA_CtzOp : WasmSSA_UnaryNumericalOp<"ctz",
                                       "Count trailing zeroes of an integer",
                                        [{Example:

                                         ```mlir
                                         %a = wasmssa.ctz %b : i32
                                         ```
                                        }],
                                       [WasmSSA_IntegerType]>{}

def WasmSSA_ClzOp : WasmSSA_UnaryNumericalOp<"clz",
                                       "Count leading zeroes of an integer",
                                        [{Example:

                                         ```mlir
                                         %a = wasmssa.clz %b : i32
                                         ```
                                        }],
                                       [WasmSSA_IntegerType]>{}

def WasmSSA_EqzOp : WasmSSA_Op<"eqz", []> {
  let summary = "Check if the given value is equal to zero";
  let description =
    [{Example:

     ```mlir
     %a = wasmssa.eqz %b : i64 -> i32
     ```
    }];
  let arguments = (ins WasmSSA_IntegerType: $input);
  let results = (outs I32: $result);
  let assemblyFormat = "$input`:` type($input) `->` type($result) attr-dict";
}


def WasmSSA_PopCntOp : WasmSSA_UnaryNumericalOp<"popcnt",
                                       "Population count of an integer.",
                                        [{Example:

                                         ```mlir
                                         %a = wasmssa.popcnt %b : i32
                                         ```
                                        }],
                                       [WasmSSA_IntegerType]>{}


#endif // WasmSSA_OPS
