//===-- CIROps.td - CIR dialect definition -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the CIR dialect
///
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_DIALECT_CIR_OPS
#define MLIR_CIR_DIALECT_CIR_OPS

include "mlir/Dialect/CIR/IR/CIRDialect.td"
include "mlir/Dialect/CIR/IR/CIRTypes.td"
include "mlir/Dialect/CIR/IR/CIRAttrs.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// CIR Ops
//===----------------------------------------------------------------------===//

class CIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<CIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

// The enumaration value isn't in sync with clang.
def CK_IntegralToBoolean : I32EnumAttrCase<"int_to_bool", 1>;

def CastKind : I32EnumAttr<
    "CastKind",
    "cast kind",
    [CK_IntegralToBoolean]> {
  let cppNamespace = "::mlir::cir";
}

def CastOp : CIR_Op<"cast", [Pure]> {
  // FIXME: not all conversions are free of side effects.
  let summary = "cast";
  let description = [{
    Apply C/C++ usual conversions rules between types. The full list of those
    can be seen in clang/include/clang/AST/OperationKinds.def, but note that some
    of the conversions aren't implemented in terms of cir.cast, lvalue-to-rvalue
    for instance is modeled as a load.

    ```mlir
    %4 = cir.cast (int_to_bool, %3 : i32), i1
    ```
  }];

  let arguments = (ins CastKind:$kind, AnyType:$src);
  let results = (outs AnyType:$res);

  let assemblyFormat = "`(` $kind `,` $src `:` type($src) `)` `,` type($res) attr-dict";

  // The input and output types should match the cast kind.
  //let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : CIR_Op<"cst",
    [ConstantLike, Pure]> {

  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = cir.cst(42 : i32)
      %1 = cir.cst(4.2 : f32)
      %2 = cir.cst(nullptr : !cir.ptr<i32>)
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins TypedAttrInterface:$value);

  // The constant operation returns a single value of AnyType.
  let results = (outs AnyType:$res);

  let assemblyFormat = "`(` custom<ConstantValue>($value) `)` attr-dict `:` type($res)";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
     bool isNullPtr() {
       return getValue().isa<UnitAttr>();
     }
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

def InitStyle_None      : I32EnumAttrCase<"uninitialized", 1>;
def InitStyle_ParamInit : I32EnumAttrCase<"paraminit", 2>;
def InitStyle_CInit     : I32EnumAttrCase<"cinit", 3>;
def InitStyle_CallInit  : I32EnumAttrCase<"callinit", 4>;
def InitStyle_ListInit  : I32EnumAttrCase<"listinit", 5>;

def InitStyle : I32EnumAttr<
    "InitStyle",
    "variable initialization style",
    [InitStyle_None, InitStyle_ParamInit,
     InitStyle_CInit, InitStyle_CallInit,
     InitStyle_ListInit]> {
  let cppNamespace = "::mlir::cir";
}

class AllocaTypesMatchWith<string summary, string lhsArg, string rhsArg,
                     string transform, string comparator = "std::equal_to<>()">
  : PredOpTrait<summary, CPred<
      comparator # "(" #
      !subst("$_self", "$" # lhsArg # ".getType()", transform) #
             ", $" # rhsArg # ")">> {
  string lhs = lhsArg;
  string rhs = rhsArg;
  string transformer = transform;
}

def AllocaOp : CIR_Op<"alloca", [
  AllocaTypesMatchWith<"'allocaType' matches pointee type of 'addr'",
                 "addr", "allocaType",
                 "$_self.cast<PointerType>().getPointee()">]> {
  let summary = "local variable";
  let description = [{
    The `cir.alloca` operation defines a local variable.

    Possible initialization styles are: uninitialized, paraminit,
    callinit, cinit and listinit.

    The result is a pointer type for the original input type.

    Example:

    ```mlir
    // Local variable with uninitialized value.
    %0 = cir.alloca i32, !cir.ptr<i32>, [cinit]
    ```
  }];

  let arguments = (ins
    TypeAttr:$allocaType,
    // FIXME: add "uninitialzed" as default mode
    Arg<InitStyle, "initialization style">:$init,
    ConfinedAttr<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$alignment
  );

  let results = (outs Res<CIR_PointerType, "",
                      [MemAlloc<AutomaticAllocationScopeResource>]>:$addr);

  let assemblyFormat = [{
    $allocaType `,` `cir.ptr` type($addr) `,` `[` $init `]`  attr-dict
  }];

  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def LoadOp : CIR_Op<"load", [
  TypesMatchWith<"type of 'result' matches pointee type of 'addr'",
                 "addr", "result",
                 "$_self.cast<PointerType>().getPointee()">]> {

  let summary = "load operation";
  let description = [{
    `cir.load` reads a variable using a pointer type.

    Example:

    ```mlir

    // Read from local variable, address in %0.
    %1 = cir.load %0 : !cir.ptr<i32>, i32
    ```
  }];

  let arguments = (ins Arg<CIR_PointerType, "the address to load from",
                           [MemRead]>:$addr,
                       UnitAttr:$conv);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $addr (`lvalue_to_rvalue` $conv^)? attr-dict `:` `cir.ptr` type($addr) `,` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def StoreOp : CIR_Op<"store", [
  TypesMatchWith<"type of 'value' matches pointee type of 'addr'",
                 "addr", "value",
                 "$_self.cast<PointerType>().getPointee()">]> {

  let summary = "store operation";
  let description = [{
    `cir.load` reads a variable using a pointer type.

    Example:

    ```mlir

    // Store to local variable, address in %0.
    cir.store %arg0, %0 : i32, !cir.ptr<i32>
  }];

  let arguments = (ins AnyType:$value,
                       Arg<CIR_PointerType, "the address to store the value",
                           [MemWrite]>:$addr);

  let assemblyFormat =
      "$value `,` $addr attr-dict `:` type($value) `,` `cir.ptr` type($addr)";
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : CIR_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      func @foo() -> AnyType {
        ...
        cir.return %0 : AnyType
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<AnyType>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def IfOp : CIR_Op<"if",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       SingleBlockImplicitTerminator<"cir::YieldOp">, RecursivelySpeculatable,
       AutomaticAllocationScope, NoRegionArguments]> {
  let summary = "if-then-else operation";
  let description = [{
    The `scf.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. For example:

    ```mlir
    cir.if %b  {
      ...
    } else {
      ...
    }
    ```

    "cir.if" defines no values and the 'else' can be omitted.

    Example:

    ```mlir
    cir.if %b  {
      ...
    }
    ```
  }];
  let arguments = (ins CIR_BoolType:$condition);

  // FIXME: for now the "then" region only has one block, that should change
  // soon as building CIR becomes more complex.
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  // FIXME: unify these within CIR_Ops.
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$cond, "bool":$withElseRegion,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "buildTerminatedBody">:$thenBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$elseBuilder)>
  ];

  let extraClassDeclaration = [{
    OpBuilder getThenBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(0);
      return OpBuilder::atBlockTerminator(body, listener);
    }
    OpBuilder getElseBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(1);
      return OpBuilder::atBlockTerminator(body, listener);
    }
    Block* thenBlock();
    Block* elseBlock();
  }];

  // TODO: let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : CIR_Op<"yield", [Pure, ReturnLike, Terminator,
                               ParentOneOf<["IfOp", "ScopeOp"]>]> {
  let summary = "termination operation for regions inside if, for, scope, etc";
  let description = [{
    "cir.yield" yields an SSA value from a CIR dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.

    Currently, there are not parents where "cir.yield" has any operands,
    but it will be useful to represent lifetime extension in the future. In
    that case the operands must match the parent operation's results.

    If the parent operation defines no values, then the "cir.yield" may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

def ScopeOp : CIR_Op<"scope", [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       SingleBlockImplicitTerminator<"cir::YieldOp">, RecursivelySpeculatable,
       AutomaticAllocationScope, NoRegionArguments]> {
  let summary = "";
  let description = [{
    "cir.scope" contains one region and defines a strict "scope" for all new
    values produced within its blocks.

    "cir.yield" is required as a terminator and can have results, in which case
    it can be omitted. Not used anywhere just yet but might be used to explicitly
    model lifetime extension.
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$scopeRegion);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, 
               "function_ref<void(OpBuilder &, Location)>":$scopeBuilder)>
  ];

  let extraClassDeclaration = [{
    Block* scopeBlock();
  }];
}

#endif // MLIR_CIR_DIALECT_CIR_OPS
