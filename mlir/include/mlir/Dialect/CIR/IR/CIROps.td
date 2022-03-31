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
include "mlir/Interfaces/InferTypeOpInterface.td"
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
       return getValue().isa<mlir::cir::NullAttr>();
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
    // int count = ...
    %0 = cir.alloca i32, !cir.ptr<i32>, ["count", cinit]
    ```
  }];

  let arguments = (ins
    TypeAttr:$allocaType,
    StrAttr:$name,
    // FIXME: add "uninitialzed" as default mode
    Arg<InitStyle, "initialization style">:$init,
    ConfinedAttr<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$alignment
  );

  let results = (outs Res<CIR_PointerType, "",
                      [MemAlloc<AutomaticAllocationScopeResource>]>:$addr);

  let extraClassDeclaration = [{
    // Whether the alloca input type is a pointer.
    bool isPointerType() { return getAllocaType().isa<::mlir::cir::PointerType>(); }
  }];

  let assemblyFormat = [{
    $allocaType `,` `cir.ptr` type($addr) `,` `[` $name `,` $init `]` attr-dict
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
    `cir.load` reads a variable (lvalue to rvalue conversion) given an address
    backed up by a `cir.ptr` type.

    Example:

    ```mlir

    // Read from local variable, address in %0.
    %1 = cir.load %0 : !cir.ptr<i32>, i32

    // Load address from memory at address %0. %3 provides
    // the address used while dereferecing a pointer.
    %3 = cir.load deref %0 : cir.ptr <!cir.ptr<i32>>
    ```
  }];

  let arguments = (ins Arg<CIR_PointerType, "the address to load from",
                           [MemRead]>:$addr, UnitAttr:$isDeref);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    (`deref` $isDeref^)? $addr `:` `cir.ptr` type($addr) `,`
                                   type($result) attr-dict
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

def ReturnOp : CIR_Op<"return", [HasParent<"FuncOp, ScopeOp, IfOp, SwitchOp">,
                                 Terminator]> {
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
       RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments]> {
  let summary = "if-then-else operation";
  let description = [{
    The `cir.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value.

    Each region can contain an arbitrary number of blocks but there usually be
    only one block in each region unless the presence of return and goto.

    Examples:

    ```mlir
    cir.if %b  {
      ...
    } else {
      ...
    }

    cir.if %c  {
      ...
    }

    cir.if %c  {
      ...
      br ^a
    ^a:
      cir.yield
    }
    ```

    `cir.if` defines no values and the 'else' can be omitted. `cir.yield` must
    explicitly terminate the region if it has more than one block.
  }];
  let arguments = (ins CIR_BoolType:$condition);
  let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);

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
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : CIR_Op<"yield", [ReturnLike, Terminator,
                               ParentOneOf<["IfOp", "ScopeOp", "SwitchOp"]>]> {
  let summary = "termination operation for regions inside if, for, scope, etc";
  let description = [{
    "cir.yield" yields an SSA value from a CIR dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.

    Currently, there are not parents where `cir.yield` has any operands,
    but it will be useful to represent lifetime extension in the future.

    When used to leave `cir.switch` regions there are two possible meanings:
    1. Plain `cir.yield` has "breaking out of a switch" semantics.
    2. `cir.yield fallthrough` means the next region in the case list should
    be executed.

    The `cir.yield` must be explicitly used whenever a region has more than
    one block, or within `cir.switch` regions not `cir.return` terminated.

    Example:
    ```
    cir.if %4 {
      ...
      cir.yield
    }

    cir.switch (%5) [
      case (equal, 3) {
        ...
        cir.yield fallthrough
      }, ...
    ]
    ```
  }];

  let arguments = (ins UnitAttr:$fallthrough, Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = [{
    (`fallthrough` $fallthrough^)? ($results^ `:` type($results))?
    attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

def ScopeOp : CIR_Op<"scope", [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments]> {
  let summary = "";
  let description = [{
    "cir.scope" contains one region and defines a strict "scope" for all new
    values produced within its blocks.

    The region can contain an arbitrary number of blocks but should usually be
    only one block if return and goto are not present.

    "cir.yield" is required as a terminator and can have results, in which case
    it can be omitted. Not used anywhere just yet but might be used to explicitly
    model lifetime extension.
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$scopeRegion);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, 
               "function_ref<void(OpBuilder &, Location)>":$scopeBuilder)>
  ];
}

//===----------------------------------------------------------------------===//
// BinOp
//===----------------------------------------------------------------------===//

// FIXME: represent Commutative, Idempotent traits for appropriate binops
def BinOpKind_Mul : I32EnumAttrCase<"Mul", 1>;
def BinOpKind_Div : I32EnumAttrCase<"Div", 2>;
def BinOpKind_Rem : I32EnumAttrCase<"Rem", 3>;
def BinOpKind_Add : I32EnumAttrCase<"Add", 4>;
def BinOpKind_Sub : I32EnumAttrCase<"Sub", 5>;
def BinOpKind_Shl : I32EnumAttrCase<"Shl", 6>;
def BinOpKind_Shr : I32EnumAttrCase<"Shr", 7>;
def BinOpKind_And : I32EnumAttrCase<"And", 8>;
def BinOpKind_Xor : I32EnumAttrCase<"Xor", 9>;
def BinOpKind_Or  : I32EnumAttrCase<"Or", 10>;

def BinOpKind : I32EnumAttr<
    "BinOpKind",
    "binary operation (arith and logic) kind",
    [BinOpKind_Mul, BinOpKind_Div, BinOpKind_Rem,
     BinOpKind_Add, BinOpKind_Sub, BinOpKind_Shl,
     BinOpKind_Shr, BinOpKind_And, BinOpKind_Xor,
     BinOpKind_Or]> {
  let cppNamespace = "::mlir::cir";
}

// FIXME: Pure won't work when we add overloading.
def BinOp : CIR_Op<"binop", [Pure,
  SameTypeOperands, SameOperandsAndResultType]> {

  let summary = "binary operations (arith and logic)";
  let description = [{
    "cir.binop performs the binary operation according to
     the specified kind/opcode: [mul, div, rem, add, sub, shl,
     shr, and, xor, or]. It accepts to input operands and the
     result type must match both types.

     Example
     ```
     %7 = binop(add, %1, %2) : i32
     %7 = binop(mul, %1, %2) : i8
     ```
  }];

  // TODO: get more accurate than AnyType
  let results = (outs AnyType:$result);
  let arguments = (ins Arg<BinOpKind, "binop kind">:$kind,
                       AnyType:$lhs, AnyType:$rhs);

  let assemblyFormat = [{
    `(` custom<BinOpKind>($kind) `,` $lhs `,` $rhs  `)` `:` type($lhs) attr-dict
  }];

  // Already covered by the traits
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

def CmpOpKind_LT : I32EnumAttrCase<"lt", 1>;
def CmpOpKind_LE : I32EnumAttrCase<"le", 2>;
def CmpOpKind_GT : I32EnumAttrCase<"gt", 3>;
def CmpOpKind_GE : I32EnumAttrCase<"ge", 4>;
def CmpOpKind_EQ : I32EnumAttrCase<"eq", 5>;
def CmpOpKind_NE : I32EnumAttrCase<"ne", 6>;

def CmpOpKind : I32EnumAttr<
    "CmpOpKind",
    "compare operation kind",
    [CmpOpKind_LT, CmpOpKind_LE, CmpOpKind_GT,
     CmpOpKind_GE, CmpOpKind_EQ, CmpOpKind_NE]> {
  let cppNamespace = "::mlir::cir";
}

// FIXME: Pure might not work when we add overloading.
def CmpOp : CIR_Op<"cmp", [Pure, SameTypeOperands]> {

  let summary = "compare operation";
  let description = [{
    "cir.cmp compares two input operands and produces a bool result. The input
     operands must have the same type. The kinds of comparison available are:
     [lt,gt,ge,eq,ne]

     Example
     ```
     %7 = cir.cmp(gt, %1, %2) : i32, !cir.bool
     ```
  }];

  // TODO: get more accurate than AnyType
  let results = (outs AnyType:$result);
  let arguments = (ins Arg<CmpOpKind, "cmp kind">:$kind,
                       AnyType:$lhs, AnyType:$rhs);

  let assemblyFormat = [{
    `(` $kind `,` $lhs `,` $rhs  `)` `:` type($lhs) `,` type($result) attr-dict
  }];

  // Already covered by the traits
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

// FIXME: even though printed/parsed names assume lowercase, we capitalize here
// because "default" is a C++ reserved keyword and can't show up in a enum.
def CaseOpKind_DT : I32EnumAttrCase<"Default", 1>;
def CaseOpKind_EQ : I32EnumAttrCase<"Equal", 2>;

def CaseOpKind : I32EnumAttr<
    "CaseOpKind",
    "case kind",
    [CaseOpKind_DT, CaseOpKind_EQ]> {
  let cppNamespace = "::mlir::cir";
}

def CaseEltValueListAttr :
  TypedArrayAttrBase<AnyAttr, "cir.switch case value condition"> {
  let constBuilderCall = ?;
}

def CaseAttr : AttrDef<CIR_Dialect, "Case"> {
  // FIXME: value should probably be optional for more clear "default"
  // representation.
  let parameters = (ins "ArrayAttr":$value, "CaseOpKindAttr":$kind);
  let mnemonic = "case";
  let assemblyFormat = "`<` struct(params) `>`";
}

def CaseArrayAttr :
  TypedArrayAttrBase<CaseAttr, "cir.switch case array attribute"> {
  let constBuilderCall = ?;
}

def SwitchOp : CIR_Op<"switch",
      [SameVariadicOperandSize,
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments]> {
  let summary = "a switch operation";
  let description = [{
    The `cir.switch` operation represents a C/C++ switch stmt for conditionally
    executing multiple regions of code. The operand to an switch is an integral
    value.

    Each region contains only one block and must be explicitly terminated with
    a cir.yield operation.

    Examples:

    ```mlir
    cir.switch (%b : i32) [
      case (#equal, 20) {
        ...
        cir.yield // break semantics
      },
      case (#default) {
        ...
        cir.yield #fallthrough
      }
    ]
    ```
  }];

  let arguments = (ins AnyInteger:$condition,
                       OptionalAttr<CaseArrayAttr>:$cases);
  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);

  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$condition,
               "function_ref<void(OpBuilder &, Location, OperationState &)>":$switchBuilder)>
  ];

  let assemblyFormat = [{
    custom<SwitchOp>(
      $regions, $cases, $condition, type($condition)
    )
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// BrOp
//===----------------------------------------------------------------------===//

def BrOp : CIR_Op<"br",
      [DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
     Pure, Terminator]> {
  let summary = "branch operation";
  let description = [{
    The `cir.br` branches unconditionally to a block.

    Example:

    ```mlir
      ...
        cir.br ^bb3
      ^bb3:  // pred: ^bb2
        cir.return
    ```
  }];

  let builders = [
    OpBuilder<(ins "Block *":$dest,
              CArg<"ValueRange", "{}">:$destOperands), [{
      $_state.addSuccessors(dest);
      $_state.addOperands(destOperands);
    }]>
  ];

  let arguments = (ins Variadic<AnyType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
}


#endif // MLIR_CIR_DIALECT_CIR_OPS
