//===- WebAssemblySSAOps.td - WebAssemblySSA op definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WEBASSEMBLYSSA_OPS
#define WEBASSEMBLYSSA_OPS


include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSABase.td"
include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSATypes.td"
include "mlir/Dialect/WebAssemblySSA/IR/WebAssemblySSAInterfaces.td"

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
  WasmSSA_Op<mnemonic, [Terminator, DeclareOpInterfaceMethods<WasmSSALabelLevelOpInterface>]> {
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
    DeclareOpInterfaceMethods<WasmSSALabelBranchingOpInterface>]> {
  let summary = "Return from the current block";
  let arguments = (ins Variadic<WasmSSA_ValType>: $inputs);
  let extraClassDeclaration = [{
    ::mlir::Block* getTarget();
  }];
  let assemblyFormat = "($inputs^ `:` type($inputs))? attr-dict";
}

def WasmSSA_BranchIfOp : WasmSSA_Op<"branch_if", [
    Terminator,
    DeclareOpInterfaceMethods<WasmSSALabelBranchingOpInterface>]> {
  let summary = "Jump to target level if condition has non-zero value";
  let arguments = (ins I32: $condition,
                       UI32Attr: $exitLevel,
                       Variadic<WasmSSA_ValType>: $inputs);
  let successors = (successor AnySuccessor: $elseSuccessor);
  let assemblyFormat = "$condition `to` `level` $exitLevel (`with` `args`  `(`$inputs^ `:` type($inputs)`)`)?  `else` $elseSuccessor  attr-dict";
}

def WasmSSA_ConstOp : WasmSSA_Op<"const", [
    AllTypesMatch<["value", "result"]>,
    ConstantExprCheckInterface,
    AlwaysValidConstantExprTrait]> {
  let summary = "Operator that represents a constant value";
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

    - A function type that represents the corresponding wasm type (tuples of value types)

    - Arguments of the entry block of type `!wasm<local T>`, with T the corresponding type
     in the function type.
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
  let summary = "Calling a wasm function";
  let arguments = (ins FlatSymbolRefAttr: $callee,
                       Variadic<WasmSSA_ValType>:  $operands);
  let results = (outs Variadic<WasmSSA_ValType>: $results);
  let assemblyFormat = "$callee (`(`$operands^`)`)? attr-dict `:` functional-type($operands, $results)";
  let description = [{
    Emits a call to callee.
  }];
}

def WasmSSA_FuncImportOp : WasmSSA_Op<"import_func", [
    Symbol,
    CallableOpInterface,
    WasmSSAImportOpInterface]> {
  let summary = "Importing a function variable";
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
  IsolatedFromAbove, Symbol, ConstantExpressionInitializerTrait]> {
  let summary= "WebAssembly global value";
  let arguments = (ins SymbolNameAttr: $sym_name,
                     WasmSSA_ValTypeAttr: $type,
                     UnitAttr: $isMutable,
                     OptionalAttr<StrAttr>:$sym_visibility);
  let description = [{
    WebAssembly global variable.
    Body contains the initialization instructions for the variable value.
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
    WasmSSAImportOpInterface]> {
  let summary = "Importing a global variable";
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

def WasmSSA_GlobalGetOp : WasmSSA_Op<"global_get", [DeclareOpInterfaceMethods<ConstantExprCheckInterface>]> {
  let summary = "Returns the value of the global passed as argument.";
  let arguments = (ins FlatSymbolRefAttr: $global);
  let results = (outs WasmSSA_ValType: $global_val);
  let assemblyFormat = "$global attr-dict `:` type($global_val)";
}

def WasmSSA_IfOp : WasmSSA_Op<"if", [Terminator,
    DeclareOpInterfaceMethods<WasmSSALabelLevelOpInterface>]> {
  let summary = "Execute the if region if condition value is nonzero, the else region otherwise.";
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
}

def WasmSSA_LocalOp : WasmSSA_Op<"local", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Declaration of local variable";
  let arguments = (ins WasmSSA_ValTypeAttr: $type);
  let results = (outs WasmSSA_LocalRef: $result);
  let assemblyFormat = "`of` `type` $type attr-dict";
}

def WasmSSA_LocalGetOp : WasmSSA_Op<"local_get", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Set local to value and return the operand.";
  let arguments = (ins WasmSSA_LocalRef: $localVar);
  let results = (outs WasmSSA_ValType: $result);
  let assemblyFormat = "$localVar `:` type($localVar) attr-dict";
  let hasVerifier = 1;
}

def WasmSSA_LocalSetOp : WasmSSA_Op<"local_set"> {
  let summary = "Set local to given value";
  let arguments = (ins WasmSSA_LocalRef: $localVar,
                       WasmSSA_ValType: $value);
  let hasVerifier = 1;
  let assemblyFormat = "$localVar `:` type($localVar) `to` $value `:` type($value) attr-dict";
}

def WasmSSA_LocalTeeOp : WasmSSA_Op<"local_tee", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Set local to value and return the operand.";
  let arguments = (ins WasmSSA_LocalRef: $localVar,
                       WasmSSA_ValType: $value);
  let results = (outs WasmSSA_ValType: $result);
  let hasVerifier = 1;
  let assemblyFormat = "$localVar `:` type($localVar) `to` $value `:` type($value) attr-dict";
}

def WasmSSA_MemOp : WasmSSA_Op<"memory", [Symbol]> {
  let summary= "WebAssembly memory definition";
  let arguments = (ins SymbolNameAttr: $sym_name,
                     WasmSSA_LimitTypeAttr: $limits,
                     OptionalAttr<StrAttr>:$sym_visibility);
  let builders = [
    OpBuilder<(ins
    "::llvm::StringRef":$symbol,
    "wasmssa::LimitType":$limit)>
  ];
}

def WasmSSA_MemImportOp : WasmSSA_Op<"import_mem", [Symbol, WasmSSAImportOpInterface]> {
  let summary = "Importing a memory";
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
}

def WasmSSA_TableImportOp : WasmSSA_Op<"import_table", [Symbol, WasmSSAImportOpInterface]> {
  let summary = "Importing a table";
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

class WasmSSA_BinaryNumericalOp<string mnemonic, string summaryStr,
                       list<Type> validOpTypes> :
  WasmSSA_Op<mnemonic, [AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = summaryStr;
  let arguments = (ins AnyTypeOf<validOpTypes>:$lhs, AnyTypeOf<validOpTypes>:$rhs);
  let results = (outs AnyTypeOf<validOpTypes>:$result);
  let assemblyFormat = "$lhs $rhs `:` type($lhs) attr-dict";
}

def WasmSSA_AddOp : WasmSSA_BinaryNumericalOp<"add",
    "Sum two values",
    [WasmSSA_NumericType]>{}

def WasmSSA_AndOp : WasmSSA_BinaryNumericalOp<"and",
    "Compute the bitwise AND between two values",
    [WasmSSA_NumericType]>{}

def WasmSSA_DivOp : WasmSSA_BinaryNumericalOp<"div",
    "Division between floating point values",
    [WasmSSA_FPType]>{}

def WasmSSA_DivUIOp : WasmSSA_BinaryNumericalOp<"div_ui",
    "Divide values interpreted as unsigned int",
    [WasmSSA_IntegerType]>{}

def WasmSSA_DivSIOp : WasmSSA_BinaryNumericalOp<"div_si",
    "Divide values interpreted as signed int",
    [WasmSSA_IntegerType]>{}

def WasmSSA_MulOp : WasmSSA_BinaryNumericalOp<"mul",
    "Multiply two values",
    [WasmSSA_NumericType]>{}

def WasmSSA_OrOp : WasmSSA_BinaryNumericalOp<"or",
    "Compute the bitwise OR of two values",
    [WasmSSA_NumericType]>{}

def WasmSSA_SubOp : WasmSSA_BinaryNumericalOp<"sub",
    "Subtract two values",
    [WasmSSA_NumericType]>{}

def WasmSSA_RemUIOp : WasmSSA_BinaryNumericalOp<"rem_ui",
    "Calculate the remainder of dividing two integer values as an unsigned integer",
    [WasmSSA_IntegerType]>{}

def WasmSSA_RemSIOp : WasmSSA_BinaryNumericalOp<"rem_si",
    "Calculate the remainder of dividing two integer values as signed integer",
    [WasmSSA_IntegerType]>{}

def WasmSSA_XOrOp : WasmSSA_BinaryNumericalOp<"xor",
    "Compute the bitwise XOR of two values",
    [WasmSSA_NumericType]>{}

def WasmSSA_MinOp : WasmSSA_BinaryNumericalOp<"min",
    "Compute the minimum of two floating point values.",
    [WasmSSA_FPType]>{}

def WasmSSA_MaxOp : WasmSSA_BinaryNumericalOp<"max",
    "Compute the minimum of two floating point values.",
    [WasmSSA_FPType]>{}

def WasmSSA_CopySignOp : WasmSSA_BinaryNumericalOp<"copysign",
    "Copy sign from one floating point vaue to the other.",
    [WasmSSA_FPType]>{}

class WasmSSA_BinaryComparisonOp<string mnemonic, string summaryStr,
                       list<Type> validOpTypes> :
  WasmSSA_Op<mnemonic, [AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = summaryStr;
  let arguments = (ins AnyTypeOf<validOpTypes>:$lhs, AnyTypeOf<validOpTypes>:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs $rhs `:` type($lhs) `->` type($result) attr-dict";
}

def WasmSSA_EqOp : WasmSSA_BinaryComparisonOp<"eq",
    "Check if two values are equal",
    [WasmSSA_NumericType]>{}

def WasmSSA_NeOp : WasmSSA_BinaryComparisonOp<"ne",
    "Check if two values are different",
    [WasmSSA_NumericType]>{}

def WasmSSA_LtSIOp : WasmSSA_BinaryComparisonOp<"lt_si",
    "Check if a signed integer value is less than another",
    [WasmSSA_IntegerType]>{}

def WasmSSA_LtUIOp : WasmSSA_BinaryComparisonOp<"lt_ui",
    "Check if an unsigned integer value is less than another",
    [WasmSSA_IntegerType]>{}

def WasmSSA_LeSIOp : WasmSSA_BinaryComparisonOp<"le_si",
    "Check if a signed integer value is less or equal to another",
    [WasmSSA_IntegerType]>{}

def WasmSSA_LeUIOp : WasmSSA_BinaryComparisonOp<"le_ui",
    "Check if an unsigned integer value is less or equal to another",
    [WasmSSA_IntegerType]>{}

def WasmSSA_GtSIOp : WasmSSA_BinaryComparisonOp<"gt_si",
    "Check if a signed integer value is greater than another",
    [WasmSSA_IntegerType]>{}

def WasmSSA_GtUIOp : WasmSSA_BinaryComparisonOp<"gt_ui",
    "Check if an unsigned integer value is greater than another",
    [WasmSSA_IntegerType]>{}

def WasmSSA_GeSIOp : WasmSSA_BinaryComparisonOp<"ge_si",
    "Check if a signed integer value is greater or equal to another",
    [WasmSSA_IntegerType]>{}

def WasmSSA_GeUIOp : WasmSSA_BinaryComparisonOp<"ge_ui",
    "Check if an unsigned integer value is greater or equal to another",
    [WasmSSA_IntegerType]>{}

def WasmSSA_LtOp : WasmSSA_BinaryComparisonOp<"lt",
    "Check if a float value is less than another",
    [WasmSSA_FPType]>{}

def WasmSSA_LeOp : WasmSSA_BinaryComparisonOp<"le",
    "Check if a float value is less or equal to another",
    [WasmSSA_FPType]>{}

def WasmSSA_GtOp : WasmSSA_BinaryComparisonOp<"gt",
    "Check if a float value is greater than another",
    [WasmSSA_FPType]>{}

def WasmSSA_GeOp : WasmSSA_BinaryComparisonOp<"ge",
    "Check if a float value is greater or equal to another",
    [WasmSSA_FPType]>{}

// Integer shift and rotate operations.
class WasmSSA_ShiftRotateOp<string mnemonic, string summaryStr> :
  WasmSSA_Op<mnemonic, [AllTypesMatch<["val", "bits", "result"]>]> {
  let summary = summaryStr;
  let arguments = (ins WasmSSA_IntegerType:$val, WasmSSA_IntegerType:$bits);
  let results = (outs WasmSSA_IntegerType:$result);
  let assemblyFormat = "$val `by` $bits `bits` `:` type($val) attr-dict";
}

def WasmSSA_ShLOp : WasmSSA_ShiftRotateOp<"shl",
    [{Consume an integer and an integer shift amount. The first
    integer shall be shifted left by N bits, where N is the value of the second
    integer.}]
    >{}

def WasmSSA_ShRSOp : WasmSSA_ShiftRotateOp<"shr_s",
    [{Arithmetic right shift.

    Consume an integer and an integer shift amount. The first
    integer shall be shifted right by N bits, where N is the value of the
    second integer.

    Vacated bits on the left shall be filled with the sign bit.}]
    >{}

def WasmSSA_ShRUOp : WasmSSA_ShiftRotateOp<"shr_u",
    [{Logical right shift.

    Consume an integer, and an integer shift amount. The first
    integer shall be shifted right by N bits, where N is the value of the
    second integer.

    Vacated bits on the left shall be filled with zeroes.}]
    >{}

def WasmSSA_RotlOp : WasmSSA_ShiftRotateOp<"rotl",
    [{Rotate left.

    Consume an integer and an integer rotate. The first
    integer shall be rotated left by N bits, where N is the value of the
    second integer.}]
    >{}

def WasmSSA_RotrOp : WasmSSA_ShiftRotateOp<"rotr",
    [{Rotate right.

    Consume an integer, and an integer rotate. The first
    integer shall be rotated right by N bits, where N is the value of the
    second integer.}]
    >{}

class WasmSSA_ConversionOp<string mnemonic, string summaryStr,
                       list<Type> ValidInputTypes,
                       list<Type> ValidOutputTypes> :
    WasmSSA_Op<mnemonic> {
  let summary = summaryStr;
  let arguments = (ins AnyTypeOf<ValidInputTypes>:$input);
  let results = (outs AnyTypeOf<ValidOutputTypes>:$result);
  let assemblyFormat = "$input `:` type($input) `to` type($result)  attr-dict";
}

def WasmSSA_ConvertUOp : WasmSSA_ConversionOp<"convert_u",
    [{Convert integer, interpreted as binary encoded positive value, to floating-point value.

    Consume an integer and produces a floating point value containing the rounded value of the original operand. Rounding is round to nearest, tie to even.}],
    [WasmSSA_IntegerType],
    [WasmSSA_FPType]>{}

def WasmSSA_ConvertSOp : WasmSSA_ConversionOp<"convert_s",
    [{Convert integer interpreted as 2's complement signed value to floating-point value.

    Consume an integer and produces a floating point value containing the rounded value of the original operand. Rounding is round to nearest, tie to even.}],
    [WasmSSA_IntegerType],
    [WasmSSA_FPType]>{}

def WasmSSA_DemoteOp : WasmSSA_ConversionOp<"demote",
    "Convert a f64 value to f32",
    [F64],
    [F32]>{}

def WasmSSA_ExtendSI32Op : WasmSSA_Op<"extend_i32_s">{
  let summary = [{Sign extend i32 to i64.}];
  let arguments = (ins I32:$input);
  let results = (outs I64:$result);
  let assemblyFormat = "$input `to` type($result)  attr-dict";
}

def WasmSSA_ExtendUI32Op : WasmSSA_Op<"extend_i32_u">{
  let summary = [{Zero extend i32 to i64.}];
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
  }];
  let arguments = (ins WasmSSA_IntegerType:$input, Builtin_IntegerAttr: $bitsToTake);
  let results = (outs WasmSSA_IntegerType: $result);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def WasmSSA_PromoteOp : WasmSSA_ConversionOp<"promote",
    "Get f64 representation of a f32 value.",
    [Builtin_Float32],
    [Builtin_Float64]>{}

def WasmSSA_WrapOp : WasmSSA_ConversionOp<"wrap",
    "Cast an i64 to i32 by using a wrapping mechanism: y = x mod 2^32",
    [I64],
    [I32]>{}
// Reinterpret ops are basically all one-offs. They all have an unique,
// type-postfixed opcode, and support exactly one input and output type.
def WasmSSA_ReinterpretOp : WasmSSA_ConversionOp<"reinterpret",
    [{Reinterpret the value represented by a bit vector by
      bit-casting it to another type of same representation width.}],
    [WasmSSA_NumericType], [WasmSSA_NumericType]>{
    let assemblyFormat = "$input `:` type($input) `as` type($result) attr-dict";
    let hasVerifier = 1;
}

class WasmSSA_UnaryNumericalOp<string mnemonic,
                            string summaryStr,
                            list<Type> validOpTypes> :
  WasmSSA_Op<mnemonic, [AllTypesMatch<["src", "result"]>]> {
    let summary = summaryStr;
    let arguments = (ins AnyTypeOf<validOpTypes>:$src);
    let results = (outs AnyTypeOf<validOpTypes>:$result);
    let assemblyFormat = "$src`:` type($src) attr-dict";
}

def WasmSSA_AbsOp : WasmSSA_UnaryNumericalOp<"abs",
                                       "Floating point absolute value",
                                       [WasmSSA_FPType]>{}

def WasmSSA_CeilOp : WasmSSA_UnaryNumericalOp<"ceil",
                                       "Ceil rounding of floating point value",
                                       [WasmSSA_FPType]>{}

def WasmSSA_FloorOp : WasmSSA_UnaryNumericalOp<"floor",
                                       "Floor rounding of floating point value",
                                       [WasmSSA_FPType]>{}

def WasmSSA_NegOp : WasmSSA_UnaryNumericalOp<"neg",
                                       "Floating point negation",
                                       [WasmSSA_FPType]>{}

def WasmSSA_SqrtOp : WasmSSA_UnaryNumericalOp<"sqrt",
                                       "Floating point square root",
                                       [WasmSSA_FPType]>{}

def WasmSSA_TruncOp : WasmSSA_UnaryNumericalOp<"trunc",
                                       "Trunc of floating point value",
                                       [WasmSSA_FPType]>{}

def WasmSSA_CtzOp : WasmSSA_UnaryNumericalOp<"ctz",
                                       "Count trailing zeroes of an integer",
                                       [WasmSSA_IntegerType]>{}

def WasmSSA_ClzOp : WasmSSA_UnaryNumericalOp<"clz",
                                       "Count leading zeroes of an integer",
                                       [WasmSSA_IntegerType]>{}

def WasmSSA_EqzOp : WasmSSA_Op<"eqz", []> {
  let summary = "Check if the given value is equal to zero";
  let arguments = (ins WasmSSA_IntegerType: $input);
  let results = (outs I32: $result);
  let assemblyFormat = "$input`:` type($input) `->` type($result) attr-dict";
}


def WasmSSA_PopCntOp : WasmSSA_UnaryNumericalOp<"popcnt",
                                       "Population count of an integer.",
                                       [WasmSSA_IntegerType]>{}


#endif // WEBASSEMBLYSSA_OPS
