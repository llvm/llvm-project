//===- FuncOps.td - Func operation definitions -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_FUNC_IR_FUNCOPS_TD
#define MLIR_DIALECT_FUNC_IR_FUNCOPS_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Func_Dialect : Dialect {
  let name = "func";
  let cppNamespace = "::mlir::func";
  let hasConstantMaterializer = 1;
}

// Base class for Func dialect ops.
class Func_Op<string mnemonic, list<Trait> traits = []> :
    Op<Func_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp : Func_Op<"call",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    The `func.call` operation represents a direct call to a function that is
    within the same symbol scope as the call. The operands and result types of
    the call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = func.call @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

def CallIndirectOp : Func_Op<"call_indirect", [
      CallOpInterface,
      TypesMatchWith<"callee input types match argument types",
                     "callee", "callee_operands",
                     "::llvm::cast<FunctionType>($_self).getInputs()">,
      TypesMatchWith<"callee result types match result types",
                     "callee", "results",
                     "::llvm::cast<FunctionType>($_self).getResults()">
    ]> {
  let summary = "indirect call operation";
  let description = [{
    The `func.call_indirect` operation represents an indirect call to a value
    of function type. The operands and result types of the call must match the
    specified function type.

    Function values can be created with the
    [`func.constant` operation](#funcconstant-constantop).

    Example:

    ```mlir
    %func = func.constant @my_func : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %result = func.call_indirect %func(%0, %1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    ```
  }];

  let arguments = (ins FunctionType:$callee,
                       Variadic<AnyType>:$callee_operands);
  let results = (outs Variadic<AnyType>:$results);

  let builders = [
    OpBuilder<(ins "Value":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.operands.push_back(callee);
      $_state.addOperands(operands);
      $_state.addTypes(::llvm::cast<FunctionType>(callee.getType()).getResults());
    }]>];

  let extraClassDeclaration = [{
    // TODO: Remove once migrated callers.
    ValueRange operands() { return getCalleeOperands(); }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getCalleeOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      setOperand(0, callee.get<Value>());
    }
  }];

  let hasCanonicalizeMethod = 1;
  let assemblyFormat = [{
    $callee `(` $callee_operands `)` attr-dict `:` type($callee)
  }];
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : Func_Op<"constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "constant";
  let description = [{
    The `func.constant` operation produces an SSA value from a symbol reference
    to a `func.func` operation

    Example:

    ```mlir
    // Reference to function @myfn.
    %2 = func.constant @myfn : (tensor<16xf32>, f32) -> tensor<16xf32>

    // Equivalent generic forms
    %2 = "func.constant"() { value = @myfn } : () -> ((tensor<16xf32>, f32) -> tensor<16xf32>)
    ```

    MLIR does not allow direct references to functions in SSA operands because
    the compiler is multithreaded, and disallowing SSA values to directly
    reference a function simplifies this
    ([rationale](../Rationale/Rationale.md#multithreading-the-compiler)).
  }];

  let arguments = (ins FlatSymbolRefAttr:$value);
  let results = (outs AnyType);
  let assemblyFormat = "attr-dict $value `:` type(results)";

  let extraClassDeclaration = [{
    /// Returns true if a constant operation can be built with the given value
    /// and result type.
    static bool isBuildableWith(Attribute value, Type type);
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : Func_Op<"func", [
  AffineScope, AutomaticAllocationScope,
  FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
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
    func.func private @abort()
    func.func private @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64

    // A function that returns its argument twice:
    func.func @count(%x: i64) -> (i64, i64)
      attributes {fruit: "banana"} {
      return %x, %x: i64, i64
    }

    // A function with an argument attribute
    func.func private @example_fn_arg(%x: i32 {swift.self = unit})

    // A function with a result attribute
    func.func private @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})

    // A function with an attribute
    func.func private @example_fn_attr() attributes {dialectName.attrName = false}
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
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
    FuncOp clone(IRMapping &mapper);
    FuncOp clone();

    /// Clone the internal blocks and attributes from this function into dest.
    /// Any cloned blocks are appended to the back of dest. This function
    /// asserts that the attributes of the current function and dest are
    /// compatible.
    void cloneInto(FuncOp dest, IRMapping &mapper);

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

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "func"; }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Func_Op<"return", [Pure, HasParent<"FuncOp">,
                                MemRefsNormalizable, ReturnLike, Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `func.return` operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.

    Example:

    ```mlir
    func.func @foo() : (i32, f8) {
      ...
      return %0, %1 : i32, f8
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_FUNC_IR_FUNCOPS_TD
