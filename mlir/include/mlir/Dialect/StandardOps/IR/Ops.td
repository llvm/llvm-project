//===- Ops.td - Standard operation definitions -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines some MLIR standard operations.
//
//===----------------------------------------------------------------------===//

#ifndef STANDARD_OPS
#define STANDARD_OPS

include "mlir/Dialect/StandardOps/IR/StandardOpsBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def StandardOps_Dialect : Dialect {
  let name = "std";
  let cppNamespace = "";
  let hasConstantMaterializer = 1;
}

// Base class for Standard dialect ops.
class Std_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<StandardOps_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

// Base class for arithmetic cast operations. Requires single operand and
// result, but does not constrain them to specific types.
class ArithmeticCastOp<string mnemonic, list<OpTrait> traits = []> :
    Std_Op<mnemonic, traits # ElementwiseMappable.traits # [
      DeclareOpInterfaceMethods<VectorUnrollOpInterface>,
      NoSideEffect, SameOperandsAndResultShape,
      DeclareOpInterfaceMethods<CastOpInterface>]> {
  let results = (outs AnyType);

  let builders = [
    OpBuilder<(ins "Value":$source, "Type":$destType), [{
       impl::buildCastOp($_builder, $_state, source, destType);
    }]>
  ];

  let parser = [{
    return impl::parseCastOp(parser, result);
  }];
  let printer = [{
    return printStandardCastOp(this->getOperation(), p);
  }];

  // Cast operations are fully verified by its traits.
  let verifier = ?;
}

// Base class for unary ops. Requires single operand and result. Individual
// classes will have `operand` accessor.
class UnaryOp<string mnemonic, list<OpTrait> traits = []> :
    Op<StandardOps_Dialect, mnemonic, !listconcat(traits, [NoSideEffect])> {
  let results = (outs AnyType);
  let printer = [{
    return printStandardUnaryOp(this->getOperation(), p);
  }];
}

class UnaryOpSameOperandAndResultType<string mnemonic,
                                      list<OpTrait> traits = []> :
    UnaryOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])> {
  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];
}

class FloatUnaryOp<string mnemonic, list<OpTrait> traits = []> :
    UnaryOpSameOperandAndResultType<mnemonic, traits # [
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
    ElementwiseMappable.traits>, Arguments<(ins FloatLike:$operand)>;

// Base class for standard arithmetic operations.  Requires operands and
// results to be of the same type, but does not constrain them to specific
// types.
class ArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    Op<StandardOps_Dialect, mnemonic, traits # [NoSideEffect,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
    ElementwiseMappable.traits> {

  let results = (outs AnyType:$result);

  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];

  let printer = [{
    return printStandardBinaryOp(this->getOperation(), p);
  }];
}

// Base class for standard binary arithmetic operations.
class ArithmeticBinaryOp<string mnemonic, list<OpTrait> traits = []> :
    ArithmeticOp<mnemonic, traits> {

  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];

  let printer = [{
    return printStandardBinaryOp(this->getOperation(), p);
  }];
}

// Base class for standard ternary arithmetic operations.
class ArithmeticTernaryOp<string mnemonic, list<OpTrait> traits = []> :
    ArithmeticOp<mnemonic, traits> {

  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];

  let printer = [{
    return printStandardTernaryOp(this->getOperation(), p);
  }];
}

// Base class for standard arithmetic operations on integers, vectors and
// tensors thereof.  This operation takes two operands and returns one result,
// each of these is required to be of the same type.  This type may be an
// integer scalar type, a vector whose element type is an integer type, or an
// integer tensor.  The custom assembly form of the operation is as follows
//
//     <op>i %0, %1 : i32
//
class IntBinaryOp<string mnemonic, list<OpTrait> traits = []> :
    ArithmeticBinaryOp<mnemonic,
      !listconcat(traits,
                  [DeclareOpInterfaceMethods<VectorUnrollOpInterface>])>,
    Arguments<(ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs)>;

// Base class for standard arithmetic binary operations on floats, vectors and
// tensors thereof.  This operation has two operands and returns one result,
// each of these is required to be of the same type.  This type may be a
// floating point scalar type, a vector whose element type is a floating point
// type, or a floating point tensor.  The custom assembly form of the operation
// is as follows
//
//     <op>f %0, %1 : f32
//
class FloatBinaryOp<string mnemonic, list<OpTrait> traits = []> :
    ArithmeticBinaryOp<mnemonic,
      !listconcat(traits,
                  [DeclareOpInterfaceMethods<VectorUnrollOpInterface>])>,
    Arguments<(ins FloatLike:$lhs, FloatLike:$rhs)>;

// Base class for standard arithmetic ternary operations on floats, vectors and
// tensors thereof.  This operation has three operands and returns one result,
// each of these is required to be of the same type.  This type may be a
// floating point scalar type, a vector whose element type is a floating point
// type, or a floating point tensor.  The custom assembly form of the operation
// is as follows
//
//     <op> %0, %1, %2 : f32
//
class FloatTernaryOp<string mnemonic, list<OpTrait> traits = []> :
    ArithmeticTernaryOp<mnemonic,
      !listconcat(traits,
                  [DeclareOpInterfaceMethods<VectorUnrollOpInterface>])>,
    Arguments<(ins FloatLike:$a, FloatLike:$b, FloatLike:$c)>;

//===----------------------------------------------------------------------===//
// AbsFOp
//===----------------------------------------------------------------------===//

def AbsFOp : FloatUnaryOp<"absf"> {
  let summary = "floating point absolute-value operation";
  let description = [{
    The `absf` operation computes the absolute value. It takes one operand and
    returns one result of the same type. This type may be a float scalar type,
    a vector whose element type is float, or a tensor of floats.

    Example:

    ```mlir
    // Scalar absolute value.
    %a = absf %b : f64

    // SIMD vector element-wise absolute value.
    %f = absf %g : vector<4xf32>

    // Tensor element-wise absolute value.
    %x = absf %y : tensor<4x?xf8>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

def AddFOp : FloatBinaryOp<"addf"> {
  let summary = "floating point addition operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.addf` ssa-use `,` ssa-use `:` type
    ```

    The `addf` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be a floating point
    scalar type, a vector whose element type is a floating point type, or a
    floating point tensor.

    Example:

    ```mlir
    // Scalar addition.
    %a = addf %b, %c : f64

    // SIMD vector addition, e.g. for Intel SSE.
    %f = addf %g, %h : vector<4xf32>

    // Tensor addition.
    %x = addf %y, %z : tensor<4x?xbf16>
    ```

    TODO: In the distant future, this will accept optional attributes for fast
    math, contraction, rounding mode, and other controls.
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

def AddIOp : IntBinaryOp<"addi", [Commutative]> {
  let summary = "integer addition operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.addi` ssa-use `,` ssa-use `:` type
    ```

    The `addi` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar
    type, a vector whose element type is integer, or a tensor of integers. It
    has no standard attributes.

    Example:

    ```mlir
    // Scalar addition.
    %a = addi %b, %c : i64

    // SIMD vector element-wise addition, e.g. for Intel SSE.
    %f = addi %g, %h : vector<4xi32>

    // Tensor element-wise addition.
    %x = addi %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

def AndOp : IntBinaryOp<"and", [Commutative]> {
  let summary = "integer binary and";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.and` ssa-use `,` ssa-use `:` type
    ```

    The `and` operation takes two operands and returns one result, each of these
    is required to be the same type. This type may be an integer scalar type, a
    vector whose element type is integer, or a tensor of integers. It has no
    standard attributes.

    Example:

    ```mlir
    // Scalar integer bitwise and.
    %a = and %b, %c : i64

    // SIMD vector element-wise bitwise integer and.
    %f = and %g, %h : vector<4xi32>

    // Tensor element-wise bitwise integer and.
    %x = and %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

def AssertOp : Std_Op<"assert"> {
  let summary = "Assert operation with message attribute";
  let description = [{
    Assert operation with single boolean operand and an error message attribute.
    If the argument is `true` this operation has no effect. Otherwise, the
    program execution will abort. The provided error message may be used by a
    runtime to propagate the error to the user.

    Example:

    ```mlir
    assert %b, "Expected ... to be true"
    ```
  }];

  let arguments = (ins I1:$arg, StrAttr:$msg);

  let assemblyFormat = "$arg `,` $msg attr-dict";

  // AssertOp is fully verified by its traits.
  let verifier = ?;

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// AtomicRMWOp
//===----------------------------------------------------------------------===//

def AtomicRMWOp : Std_Op<"atomic_rmw", [
      AllTypesMatch<["value", "result"]>,
      TypesMatchWith<"value type matches element type of memref",
                     "memref", "value",
                     "$_self.cast<MemRefType>().getElementType()">
    ]> {
  let summary = "atomic read-modify-write operation";
  let description = [{
    The `atomic_rmw` operation provides a way to perform a read-modify-write
    sequence that is free from data races. The kind enumeration specifies the
    modification to perform. The value operand represents the new value to be
    applied during the modification. The memref operand represents the buffer
    that the read and write will be performed against, as accessed by the
    specified indices. The arity of the indices is the rank of the memref. The
    result represents the latest value that was stored.

    Example:

    ```mlir
    %x = atomic_rmw "addf" %value, %I[%i] : (f32, memref<10xf32>) -> f32
    ```
  }];

  let arguments = (ins
      AtomicRMWKindAttr:$kind,
      AnyTypeOf<[AnySignlessInteger, AnyFloat]>:$value,
      MemRefOf<[AnySignlessInteger, AnyFloat]>:$memref,
      Variadic<Index>:$indices);
  let results = (outs AnyTypeOf<[AnySignlessInteger, AnyFloat]>:$result);

  let assemblyFormat = [{
    $kind $value `,` $memref `[` $indices `]` attr-dict `:` `(` type($value) `,`
    type($memref) `)` `->` type($result)
  }];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return memref().getType().cast<MemRefType>();
    }
  }];
}

def GenericAtomicRMWOp : Std_Op<"generic_atomic_rmw", [
      SingleBlockImplicitTerminator<"AtomicYieldOp">,
      TypesMatchWith<"result type matches element type of memref",
                     "memref", "result",
                     "$_self.cast<MemRefType>().getElementType()">
    ]> {
  let summary = "atomic read-modify-write operation with a region";
  let description = [{
    The `generic_atomic_rmw` operation provides a way to perform a read-modify-write
    sequence that is free from data races. The memref operand represents the
    buffer that the read and write will be performed against, as accessed by
    the specified indices. The arity of the indices is the rank of the memref.
    The result represents the latest value that was stored. The region contains
    the code for the modification itself. The entry block has a single argument
    that represents the value stored in `memref[indices]` before the write is
    performed. No side-effecting ops are allowed in the body of
    `GenericAtomicRMWOp`.

    Example:

    ```mlir
    %x = generic_atomic_rmw %I[%i] : memref<10xf32> {
      ^bb0(%current_value : f32):
        %c1 = constant 1.0 : f32
        %inc = addf %c1, %current_value : f32
        atomic_yield %inc : f32
    }
    ```
  }];

  let arguments = (ins
      MemRefOf<[AnySignlessInteger, AnyFloat]>:$memref,
      Variadic<Index>:$indices);

  let results = (outs
      AnyTypeOf<[AnySignlessInteger, AnyFloat]>:$result);

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$memref, "ValueRange":$ivs)>];

  let extraClassDeclaration = [{
    // The value stored in memref[ivs].
    Value getCurrentValue() {
      return body().getArgument(0);
    }
    MemRefType getMemRefType() {
      return memref().getType().cast<MemRefType>();
    }
  }];
}

def AtomicYieldOp : Std_Op<"atomic_yield", [
      HasParent<"GenericAtomicRMWOp">,
      NoSideEffect,
      Terminator
    ]> {
  let summary = "yield operation for GenericAtomicRMWOp";
  let description = [{
    "atomic_yield" yields an SSA value from a GenericAtomicRMWOp region.
  }];

  let arguments = (ins AnyType:$result);
  let assemblyFormat = "$result attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// BranchOp
//===----------------------------------------------------------------------===//

def BranchOp : Std_Op<"br",
    [DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
     NoSideEffect, Terminator]> {
  let summary = "branch operation";
  let description = [{
    The `br` operation represents a branch operation in a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types for each successor must match the arguments of
    the block successor.

    Example:

    ```mlir
    ^bb2:
      %2 = call @someFn()
      br ^bb3(%2 : tensor<*xf32>)
    ^bb3(%3: tensor<*xf32>):
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);

  let builders = [
    OpBuilder<(ins "Block *":$dest,
                  CArg<"ValueRange", "{}">:$destOperands), [{
      $_state.addSuccessors(dest);
      $_state.addOperands(destOperands);
    }]>];

  // BranchOp is fully verified by traits.
  let verifier = ?;

  let extraClassDeclaration = [{
    Block *getDest();
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];

  let hasCanonicalizer = 1;
  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp : Std_Op<"call",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    The `call` operation represents a direct call to a function that is within
    the same symbol scope as the call. The operands and result types of the
    call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = call @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee",$_builder.getSymbolRefAttr(callee));
      $_state.addTypes(callee.getType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, $_builder.getSymbolRefAttr(callee), results,
            operands);
    }]>];

  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
  let verifier = ?;
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

def CallIndirectOp : Std_Op<"call_indirect", [
      CallOpInterface,
      TypesMatchWith<"callee input types match argument types",
                     "callee", "operands",
                     "$_self.cast<FunctionType>().getInputs()">,
      TypesMatchWith<"callee result types match result types",
                     "callee", "results",
                     "$_self.cast<FunctionType>().getResults()">
    ]> {
  let summary = "indirect call operation";
  let description = [{
    The `call_indirect` operation represents an indirect call to a value of
    function type. Functions are first class types in MLIR, and may be passed as
    arguments and merged together with block arguments. The operands and result
    types of the call must match the specified function type.

    Function values can be created with the
    [`constant` operation](#stdconstant-constantop).

    Example:

    ```mlir
    %31 = call_indirect %15(%0, %1)
            : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    ```
  }];

  let arguments = (ins FunctionType:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);

  let builders = [
    OpBuilder<(ins "Value":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.operands.push_back(callee);
      $_state.addOperands(operands);
      $_state.addTypes(callee.getType().cast<FunctionType>().getResults());
    }]>];

  let extraClassDeclaration = [{
    Value getCallee() { return getOperand(0); }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }
  }];

  let verifier = ?;
  let hasCanonicalizer = 1;

  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` type($callee)";
}

//===----------------------------------------------------------------------===//
// CeilFOp
//===----------------------------------------------------------------------===//

def CeilFOp : FloatUnaryOp<"ceilf"> {
  let summary = "ceiling of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.ceilf` ssa-use `:` type
    ```

    The `ceilf` operation computes the ceiling of a given value. It takes one
    operand and returns one result of the same type. This type may be a float
    scalar type, a vector whose element type is float, or a tensor of floats.
    It has no standard attributes.

    Example:

    ```mlir
    // Scalar ceiling value.
    %a = ceilf %b : f64

    // SIMD vector element-wise ceiling value.
    %f = ceilf %g : vector<4xf32>

    // Tensor element-wise ceiling value.
    %x = ceilf %y : tensor<4x?xf8>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// FloorFOp
//===----------------------------------------------------------------------===//

def FloorFOp : FloatUnaryOp<"floorf"> {
  let summary = "floor of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.floorf` ssa-use `:` type
    ```

    The `floorf` operation computes the floor of a given value. It takes one
    operand and returns one result of the same type. This type may be a float
    scalar type, a vector whose element type is float, or a tensor of floats.
    It has no standard attributes.

    Example:

    ```mlir
    // Scalar floor value.
    %a = floorf %b : f64

    // SIMD vector element-wise floor value.
    %f = floorf %g : vector<4xf32>

    // Tensor element-wise floor value.
    %x = floorf %y : tensor<4x?xf8>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//

// The predicate indicates the type of the comparison to perform:
// (un)orderedness, (in)equality and less/greater than (or equal to) as
// well as predicates that are always true or false.
def CMPF_P_FALSE   : I64EnumAttrCase<"AlwaysFalse", 0, "false">;
def CMPF_P_OEQ     : I64EnumAttrCase<"OEQ", 1, "oeq">;
def CMPF_P_OGT     : I64EnumAttrCase<"OGT", 2, "ogt">;
def CMPF_P_OGE     : I64EnumAttrCase<"OGE", 3, "oge">;
def CMPF_P_OLT     : I64EnumAttrCase<"OLT", 4, "olt">;
def CMPF_P_OLE     : I64EnumAttrCase<"OLE", 5, "ole">;
def CMPF_P_ONE     : I64EnumAttrCase<"ONE", 6, "one">;
def CMPF_P_ORD     : I64EnumAttrCase<"ORD", 7, "ord">;
def CMPF_P_UEQ     : I64EnumAttrCase<"UEQ", 8, "ueq">;
def CMPF_P_UGT     : I64EnumAttrCase<"UGT", 9, "ugt">;
def CMPF_P_UGE     : I64EnumAttrCase<"UGE", 10, "uge">;
def CMPF_P_ULT     : I64EnumAttrCase<"ULT", 11, "ult">;
def CMPF_P_ULE     : I64EnumAttrCase<"ULE", 12, "ule">;
def CMPF_P_UNE     : I64EnumAttrCase<"UNE", 13, "une">;
def CMPF_P_UNO     : I64EnumAttrCase<"UNO", 14, "uno">;
def CMPF_P_TRUE    : I64EnumAttrCase<"AlwaysTrue", 15, "true">;

def CmpFPredicateAttr : I64EnumAttr<
    "CmpFPredicate", "",
    [CMPF_P_FALSE, CMPF_P_OEQ, CMPF_P_OGT, CMPF_P_OGE, CMPF_P_OLT, CMPF_P_OLE,
     CMPF_P_ONE, CMPF_P_ORD, CMPF_P_UEQ, CMPF_P_UGT, CMPF_P_UGE, CMPF_P_ULT,
     CMPF_P_ULE, CMPF_P_UNE, CMPF_P_UNO, CMPF_P_TRUE]> {
  let cppNamespace = "::mlir";
}

def CmpFOp : Std_Op<"cmpf", [NoSideEffect, SameTypeOperands,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>, TypesMatchWith<
    "result type has i1 element type and same shape as operands",
    "lhs", "result", "getI1SameShape($_self)">] # ElementwiseMappable.traits> {
  let summary = "floating-point comparison operation";
  let description = [{
    The `cmpf` operation compares its two operands according to the float
    comparison rules and the predicate specified by the respective attribute.
    The predicate defines the type of comparison: (un)orderedness, (in)equality
    and signed less/greater than (or equal to) as well as predicates that are
    always true or false.  The operands must have the same type, and this type
    must be a float type, or a vector or tensor thereof.  The result is an i1,
    or a vector/tensor thereof having the same shape as the inputs. Unlike cmpi,
    the operands are always treated as signed. The u prefix indicates
    *unordered* comparison, not unsigned comparison, so "une" means unordered or
    not equal. For the sake of readability by humans, custom assembly form for
    the operation uses a string-typed attribute for the predicate.  The value of
    this attribute corresponds to lower-cased name of the predicate constant,
    e.g., "one" means "ordered not equal".  The string representation of the
    attribute is merely a syntactic sugar and is converted to an integer
    attribute by the parser.

    Example:

    ```mlir
    %r1 = cmpf "oeq" %0, %1 : f32
    %r2 = cmpf "ult" %0, %1 : tensor<42x42xf64>
    %r3 = "std.cmpf"(%0, %1) {predicate: 0} : (f8, f8) -> i1
    ```
  }];

  let arguments = (ins
    CmpFPredicateAttr:$predicate,
    FloatLike:$lhs,
    FloatLike:$rhs
  );
  let results = (outs BoolLike:$result);

  let builders = [
    OpBuilder<(ins "CmpFPredicate":$predicate, "Value":$lhs,
                  "Value":$rhs), [{
      ::buildCmpFOp($_builder, $_state, predicate, lhs, rhs);
    }]>];

  let extraClassDeclaration = [{
    static StringRef getPredicateAttrName() { return "predicate"; }
    static CmpFPredicate getPredicateByName(StringRef name);

    CmpFPredicate getPredicate() {
      return (CmpFPredicate)(*this)->getAttrOfType<IntegerAttr>(
          getPredicateAttrName()).getInt();
    }
  }];

  let verifier = [{ return success(); }];

  let hasFolder = 1;

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

def CMPI_P_EQ  : I64EnumAttrCase<"eq", 0>;
def CMPI_P_NE  : I64EnumAttrCase<"ne", 1>;
def CMPI_P_SLT : I64EnumAttrCase<"slt", 2>;
def CMPI_P_SLE : I64EnumAttrCase<"sle", 3>;
def CMPI_P_SGT : I64EnumAttrCase<"sgt", 4>;
def CMPI_P_SGE : I64EnumAttrCase<"sge", 5>;
def CMPI_P_ULT : I64EnumAttrCase<"ult", 6>;
def CMPI_P_ULE : I64EnumAttrCase<"ule", 7>;
def CMPI_P_UGT : I64EnumAttrCase<"ugt", 8>;
def CMPI_P_UGE : I64EnumAttrCase<"uge", 9>;

def CmpIPredicateAttr : I64EnumAttr<
    "CmpIPredicate", "",
    [CMPI_P_EQ, CMPI_P_NE, CMPI_P_SLT, CMPI_P_SLE, CMPI_P_SGT,
     CMPI_P_SGE, CMPI_P_ULT, CMPI_P_ULE, CMPI_P_UGT, CMPI_P_UGE]> {
  let cppNamespace = "::mlir";
}

def CmpIOp : Std_Op<"cmpi", [NoSideEffect, SameTypeOperands,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>, TypesMatchWith<
    "result type has i1 element type and same shape as operands",
    "lhs", "result", "getI1SameShape($_self)">] # ElementwiseMappable.traits> {
  let summary = "integer comparison operation";
  let description = [{
    The `cmpi` operation is a generic comparison for integer-like types. Its two
    arguments can be integers, vectors or tensors thereof as long as their types
    match. The operation produces an i1 for the former case, a vector or a
    tensor of i1 with the same shape as inputs in the other cases.

    Its first argument is an attribute that defines which type of comparison is
    performed. The following comparisons are supported:

    -   equal (mnemonic: `"eq"`; integer value: `0`)
    -   not equal (mnemonic: `"ne"`; integer value: `1`)
    -   signed less than (mnemonic: `"slt"`; integer value: `2`)
    -   signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
    -   signed greater than (mnemonic: `"sgt"`; integer value: `4`)
    -   signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
    -   unsigned less than (mnemonic: `"ult"`; integer value: `6`)
    -   unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
    -   unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
    -   unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)

    The result is `1` if the comparison is true and `0` otherwise. For vector or
    tensor operands, the comparison is performed elementwise and the element of
    the result indicates whether the comparison is true for the operand elements
    with the same indices as those of the result.

    Note: while the custom assembly form uses strings, the actual underlying
    attribute has integer type (or rather enum class in C++ code) as seen from
    the generic assembly form. String literals are used to improve readability
    of the IR by humans.

    This operation only applies to integer-like operands, but not floats. The
    main reason being that comparison operations have diverging sets of
    attributes: integers require sign specification while floats require various
    floating point-related particularities, e.g., `-ffast-math` behavior,
    IEEE754 compliance, etc
    ([rationale](../Rationale/Rationale.md#splitting-floating-point-vs-integer-operations)).
    The type of comparison is specified as attribute to avoid introducing ten
    similar operations, taking into account that they are often implemented
    using the same operation downstream
    ([rationale](../Rationale/Rationale.md#specifying-comparison-kind-as-attribute)). The
    separation between signed and unsigned order comparisons is necessary
    because of integers being signless. The comparison operation must know how
    to interpret values with the foremost bit being set: negatives in two's
    complement or large positives
    ([rationale](../Rationale/Rationale.md#specifying-sign-in-integer-comparison-operations)).

    Example:

    ```mlir
    // Custom form of scalar "signed less than" comparison.
    %x = cmpi "slt", %lhs, %rhs : i32

    // Generic form of the same operation.
    %x = "std.cmpi"(%lhs, %rhs) {predicate = 2 : i64} : (i32, i32) -> i1

    // Custom form of vector equality comparison.
    %x = cmpi "eq", %lhs, %rhs : vector<4xi64>

    // Generic form of the same operation.
    %x = "std.cmpi"(%lhs, %rhs) {predicate = 0 : i64}
        : (vector<4xi64>, vector<4xi64>) -> vector<4xi1>
    ```
  }];

  let arguments = (ins
      CmpIPredicateAttr:$predicate,
      SignlessIntegerLike:$lhs,
      SignlessIntegerLike:$rhs
  );
  let results = (outs BoolLike:$result);

  let builders = [
    OpBuilder<(ins "CmpIPredicate":$predicate, "Value":$lhs,
                 "Value":$rhs), [{
      ::buildCmpIOp($_builder, $_state, predicate, lhs, rhs);
    }]>];

  let extraClassDeclaration = [{
    static StringRef getPredicateAttrName() { return "predicate"; }
    static CmpIPredicate getPredicateByName(StringRef name);

    CmpIPredicate getPredicate() {
      return (CmpIPredicate)(*this)->getAttrOfType<IntegerAttr>(
          getPredicateAttrName()).getInt();
    }
  }];

  let verifier = [{ return success(); }];

  let hasFolder = 1;

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

//===----------------------------------------------------------------------===//
// CondBranchOp
//===----------------------------------------------------------------------===//

def CondBranchOp : Std_Op<"cond_br",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
     NoSideEffect, Terminator]> {
  let summary = "conditional branch operation";
  let description = [{
    The `cond_br` terminator operation represents a conditional branch on a
    boolean (1-bit integer) value. If the bit is set, then the first destination
    is jumped to; if it is false, the second destination is chosen. The count
    and types of operands must align with the arguments in the corresponding
    target blocks.

    The MLIR conditional branch operation is not allowed to target the entry
    block for a region. The two destinations of the conditional branch operation
    are allowed to be the same.

    The following example illustrates a function with a conditional branch
    operation that targets the same block.

    Example:

    ```mlir
    func @select(%a: i32, %b: i32, %flag: i1) -> i32 {
      // Both targets are the same, operands differ
      cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)

    ^bb1(%x : i32) :
      return %x : i32
    }
    ```
  }];

  let arguments = (ins I1:$condition,
                       Variadic<AnyType>:$trueDestOperands,
                       Variadic<AnyType>:$falseDestOperands);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

  let builders = [
    OpBuilder<(ins "Value":$condition, "Block *":$trueDest,
      "ValueRange":$trueOperands, "Block *":$falseDest,
      "ValueRange":$falseOperands), [{
      build($_builder, $_state, condition, trueOperands, falseOperands, trueDest,
            falseDest);
    }]>,
    OpBuilder<(ins "Value":$condition, "Block *":$trueDest,
      "Block *":$falseDest, CArg<"ValueRange", "{}">:$falseOperands), [{
      build($_builder, $_state, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
    }]>];

  // CondBranchOp is fully verified by traits.
  let verifier = ?;

  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };

    // The condition operand is the first operand in the list.
    Value getCondition() { return getOperand(0); }

    /// Return the destination if the condition is true.
    Block *getTrueDest() {
      return getSuccessor(trueIndex);
    }

    /// Return the destination if the condition is false.
    Block *getFalseDest() {
      return getSuccessor(falseIndex);
    }

    // Accessors for operands to the 'true' destination.
    Value getTrueOperand(unsigned idx) {
      assert(idx < getNumTrueOperands());
      return getOperand(getTrueDestOperandIndex() + idx);
    }

    void setTrueOperand(unsigned idx, Value value) {
      assert(idx < getNumTrueOperands());
      setOperand(getTrueDestOperandIndex() + idx, value);
    }

    operand_range getTrueOperands() { return trueDestOperands(); }

    unsigned getNumTrueOperands()  { return getTrueOperands().size(); }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index)  {
      trueDestOperandsMutable().erase(index);
    }

    // Accessors for operands to the 'false' destination.
    Value getFalseOperand(unsigned idx) {
      assert(idx < getNumFalseOperands());
      return getOperand(getFalseDestOperandIndex() + idx);
    }
    void setFalseOperand(unsigned idx, Value value) {
      assert(idx < getNumFalseOperands());
      setOperand(getFalseDestOperandIndex() + idx, value);
    }

    operand_range getFalseOperands() { return falseDestOperands(); }

    unsigned getNumFalseOperands() { return getFalseOperands().size(); }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      falseDestOperandsMutable().erase(index);
    }

  private:
    /// Get the index of the first true destination operand.
    unsigned getTrueDestOperandIndex() { return 1; }

    /// Get the index of the first false destination operand.
    unsigned getFalseDestOperandIndex() {
      return getTrueDestOperandIndex() + getNumTrueOperands();
    }
  }];

  let hasCanonicalizer = 1;
  let assemblyFormat = [{
    $condition `,`
    $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
    $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : Std_Op<"constant",
    [ConstantLike, NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
  let summary = "constant";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.constant` attribute-value `:` type
    ```

    The `constant` operation produces an SSA value equal to some constant
    specified by an attribute. This is the way that MLIR uses to form simple
    integer and floating point constants, as well as more exotic things like
    references to functions and tensor/vector constants.

    Example:

    ```mlir
    // Integer constant
    %1 = constant 42 : i32

    // Reference to function @myfn.
    %3 = constant @myfn : (tensor<16xf32>, f32) -> tensor<16xf32>

    // Equivalent generic forms
    %1 = "std.constant"() {value = 42 : i32} : () -> i32
    %3 = "std.constant"() {value = @myfn}
       : () -> ((tensor<16xf32>, f32) -> tensor<16xf32>)
    ```

    MLIR does not allow direct references to functions in SSA operands because
    the compiler is multithreaded, and disallowing SSA values to directly
    reference a function simplifies this
    ([rationale](../Rationale/Rationale.md#multithreading-the-compiler)).
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let builders = [
    OpBuilder<(ins "Attribute":$value),
    [{ build($_builder, $_state, value.getType(), value); }]>];

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("value"); }

    /// Returns true if a constant operation can be built with the given value
    /// and result type.
    static bool isBuildableWith(Attribute value, Type type);
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CopySignOp
//===----------------------------------------------------------------------===//

def CopySignOp : FloatBinaryOp<"copysign"> {
  let summary = "A copysign operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.copysign` ssa-use `,` ssa-use `:` type
    ```

    The `copysign` returns a value with the magnitude of the first operand and
    the sign of the second operand. It takes two operands and returns one
    result of the same type. This type may be a float scalar type, a vector
    whose element type is float, or a tensor of floats. It has no standard
    attributes.

    Example:

    ```mlir
    // Scalar copysign value.
    %a = copysign %b, %c : f64

    // SIMD vector element-wise copysign value.
    %f = copysign %g, %h : vector<4xf32>

    // Tensor element-wise copysign value.
    %x = copysign %y, %z : tensor<4x?xf8>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DivFOp
//===----------------------------------------------------------------------===//

def DivFOp : FloatBinaryOp<"divf"> {
  let summary = "floating point division operation";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FmaFOp
//===----------------------------------------------------------------------===//

def FmaFOp : FloatTernaryOp<"fmaf"> {
  let summary = "floating point fused multipy-add operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.fmaf` ssa-use `,` ssa-use `,` ssa-use `:` type
    ```

    The `fmaf` operation takes three operands and returns one result, each of
    these is required to be the same type. This type may be a floating point
    scalar type, a vector whose element type is a floating point type, or a
    floating point tensor.

    Example:

    ```mlir
    // Scalar fused multiply-add: d = a*b + c
    %d = fmaf %a, %b, %c : f64

    // SIMD vector fused multiply-add, e.g. for Intel SSE.
    %i = fmaf %f, %g, %h : vector<4xf32>

    // Tensor fused multiply-add.
    %w = fmaf %x, %y, %z : tensor<4x?xbf16>
    ```

    The semantics of the operation correspond to those of the `llvm.fma`
    [intrinsic](https://llvm.org/docs/LangRef.html#llvm-fma-intrinsic). In the
    particular case of lowering to LLVM, this is guaranteed to lower
    to the `llvm.fma.*` intrinsic.
  }];
}

//===----------------------------------------------------------------------===//
// FPExtOp
//===----------------------------------------------------------------------===//

def FPExtOp : ArithmeticCastOp<"fpext">, Arguments<(ins AnyType:$in)> {
  let summary = "cast from floating-point to wider floating-point";
  let description = [{
    Cast a floating-point value to a larger floating-point-typed value.
    The destination type must to be strictly wider than the source type.
    Only scalars are currently supported.
  }];
}

//===----------------------------------------------------------------------===//
// FPToSIOp
//===----------------------------------------------------------------------===//

def FPToSIOp : ArithmeticCastOp<"fptosi">, Arguments<(ins AnyType:$in)> {
  let summary = "cast from floating-point type to integer type";
  let description = [{
    Cast from a value interpreted as floating-point to the nearest (rounding
    towards zero) signed integer value.
  }];
}

//===----------------------------------------------------------------------===//
// FPToUIOp
//===----------------------------------------------------------------------===//

def FPToUIOp : ArithmeticCastOp<"fptoui">, Arguments<(ins AnyType:$in)> {
  let summary = "cast from floating-point type to integer type";
  let description = [{
    Cast from a value interpreted as floating-point to the nearest (rounding
    towards zero) unsigned integer value.
  }];
}

//===----------------------------------------------------------------------===//
// FPTruncOp
//===----------------------------------------------------------------------===//

def FPTruncOp : ArithmeticCastOp<"fptrunc">, Arguments<(ins AnyType:$in)> {
  let summary = "cast from floating-point to narrower floating-point";
  let description = [{
    Truncate a floating-point value to a smaller floating-point-typed value.
    The destination type must be strictly narrower than the source type.
    If the value cannot be exactly represented, it is rounded using the default
    rounding mode. Only scalars are currently supported.
  }];
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

def IndexCastOp : ArithmeticCastOp<"index_cast">, Arguments<(ins AnyType:$in)> {
  let summary = "cast between index and integer types";
  let description = [{
    Casts between integer scalars and 'index' scalars. Index is an integer of
    platform-specific bit width. If casting to a wider integer, the value is
    sign-extended. If casting to a narrower integer, the value is truncated.
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

def MulFOp : FloatBinaryOp<"mulf"> {
  let summary = "floating point multiplication operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.mulf` ssa-use `,` ssa-use `:` type
    ```

    The `mulf` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be a floating point
    scalar type, a vector whose element type is a floating point type, or a
    floating point tensor.

    Example:

    ```mlir
    // Scalar multiplication.
    %a = mulf %b, %c : f64

    // SIMD pointwise vector multiplication, e.g. for Intel SSE.
    %f = mulf %g, %h : vector<4xf32>

    // Tensor pointwise multiplication.
    %x = mulf %y, %z : tensor<4x?xbf16>
    ```

    TODO: In the distant future, this will accept optional attributes for fast
    math, contraction, rounding mode, and other controls.
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

def MulIOp : IntBinaryOp<"muli", [Commutative]> {
  let summary = "integer multiplication operation";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// NegFOp
//===----------------------------------------------------------------------===//

def NegFOp : FloatUnaryOp<"negf"> {
  let summary = "floating point negation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `negf` ssa-use `:` type
    ```

    The `negf` operation computes the negation of a given value. It takes one
    operand and returns one result of the same type. This type may be a float
    scalar type, a vector whose element type is float, or a tensor of floats.
    It has no standard attributes.

    Example:

    ```mlir
    // Scalar negation value.
    %a = negf %b : f64

    // SIMD vector element-wise negation value.
    %f = negf %g : vector<4xf32>

    // Tensor element-wise negation value.
    %x = negf %y : tensor<4x?xf8>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

def OrOp : IntBinaryOp<"or", [Commutative]> {
  let summary = "integer binary or";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `or` ssa-use `,` ssa-use `:` type
    ```

    The `or` operation takes two operands and returns one result, each of these
    is required to be the same type. This type may be an integer scalar type, a
    vector whose element type is integer, or a tensor of integers. It has no
    standard attributes.

    Example:

    ```mlir
    // Scalar integer bitwise or.
    %a = or %b, %c : i64

    // SIMD vector element-wise bitwise integer or.
    %f = or %g, %h : vector<4xi32>

    // Tensor element-wise bitwise integer or.
    %x = or %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// RankOp
//===----------------------------------------------------------------------===//

def RankOp : Std_Op<"rank", [NoSideEffect]> {
  let summary = "rank operation";
  let description = [{
    The `rank` operation takes a memref/tensor operand and returns its rank.

    Example:

    ```mlir
    %1 = rank %arg0 : tensor<*xf32>
    %2 = rank %arg1 : memref<*xf32>
    ```
  }];

  let arguments = (ins AnyTypeOf<[AnyRankedOrUnrankedMemRef, AnyTensor],
                                 "any memref or tensor type">:$memrefOrTensor);
  let results = (outs Index);
  let verifier = ?;

  let builders = [
    OpBuilder<(ins "Value":$tensor), [{
      auto indexType = $_builder.getIndexType();
      build($_builder, $_state, indexType, tensor);
    }]>];

  let hasFolder = 1;
  let assemblyFormat = "$memrefOrTensor attr-dict `:` type($memrefOrTensor)";
}

//===----------------------------------------------------------------------===//
// RemFOp
//===----------------------------------------------------------------------===//

def RemFOp : FloatBinaryOp<"remf"> {
  let summary = "floating point division remainder operation";
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Std_Op<"return", [NoSideEffect, HasParent<"FuncOp">,
                                MemRefsNormalizable, ReturnLike, Terminator]> {
  let summary = "return operation";
  let description = [{
    The `return` operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.

    Example:

    ```mlir
    func @foo() : (i32, f8) {
      ...
      return %0, %1 : i32, f8
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins),
    [{ build($_builder, $_state, llvm::None); }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def SelectOp : Std_Op<"select", [NoSideEffect,
    AllTypesMatch<["true_value", "false_value", "result"]>,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
    ElementwiseMappable.traits> {
  let summary = "select operation";
  let description = [{
    The `select` operation chooses one value based on a binary condition
    supplied as its first operand. If the value of the first operand is `1`,
    the second operand is chosen, otherwise the third operand is chosen.
    The second and the third operand must have the same type.

    The operation applies to vectors and tensors elementwise given the _shape_
    of all operands is identical. The choice is made for each element
    individually based on the value at the same position as the element in the
    condition operand. If an i1 is provided as the condition, the entire vector
    or tensor is chosen.

    The `select` operation combined with [`cmpi`](#stdcmpi-cmpiop) can be used
    to implement `min` and `max` with signed or unsigned comparison semantics.

    Example:

    ```mlir
    // Custom form of scalar selection.
    %x = select %cond, %true, %false : i32

    // Generic form of the same operation.
    %x = "std.select"(%cond, %true, %false) : (i1, i32, i32) -> i32

    // Element-wise vector selection.
    %vx = std.select %vcond, %vtrue, %vfalse : vector<42xi1>, vector<42xf32>

    // Full vector selection.
    %vx = std.select %cond, %vtrue, %vfalse : vector<42xf32>
    ```
  }];

  let arguments = (ins BoolLike:$condition,
                       AnyType:$true_value,
                       AnyType:$false_value);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "Value":$condition, "Value":$trueValue,
      "Value":$falseValue), [{
      $_state.addOperands({condition, trueValue, falseValue});
      $_state.addTypes(trueValue.getType());
    }]>];

  let extraClassDeclaration = [{
      Value getCondition() { return condition(); }
      Value getTrueValue() { return true_value(); }
      Value getFalseValue() { return false_value(); }
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ShiftLeftOp
//===----------------------------------------------------------------------===//

def ShiftLeftOp : IntBinaryOp<"shift_left"> {
  let summary = "integer left-shift";
  let description = [{
    The shift_left operation shifts an integer value to the left by a variable
    amount. The low order bits are filled with zeros.

    Example:

    ```mlir
    %1 = constant 5 : i8                       // %1 is 0b00000101
    %2 = constant 3 : i8
    %3 = shift_left %1, %2 : (i8, i8) -> i8    // %3 is 0b00101000
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SignedDivIOp
//===----------------------------------------------------------------------===//

def SignedDivIOp : IntBinaryOp<"divi_signed"> {
  let summary = "signed integer division operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `divi_signed` ssa-use `,` ssa-use `:` type
    ```

    Signed integer division. Rounds towards zero. Treats the leading bit as
    sign, i.e. `6 / -2 = -3`.

    Note: the semantics of division by zero or signed division overflow (minimum
    value divided by -1) is TBD; do NOT assume any specific behavior.

    Example:

    ```mlir
    // Scalar signed integer division.
    %a = divis %b, %c : i64

    // SIMD vector element-wise division.
    %f = divis %g, %h : vector<4xi32>

    // Tensor element-wise integer division.
    %x = divis %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SignedFloorDivIOp
//===----------------------------------------------------------------------===//

def SignedFloorDivIOp : IntBinaryOp<"floordivi_signed"> {
  let summary = "signed floor integer division operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `floordivi_signed` ssa-use `,` ssa-use `:` type
    ```

    Signed integer division. Rounds towards negative infinity, i.e. `5 / -2 = -3`.

    Note: the semantics of division by zero or signed division overflow (minimum
    value divided by -1) is TBD; do NOT assume any specific behavior.

    Example:

    ```mlir
    // Scalar signed integer division.
    %a = floordivi_signed %b, %c : i64

    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SignedCeilDivIOp
//===----------------------------------------------------------------------===//

def SignedCeilDivIOp : IntBinaryOp<"ceildivi_signed"> {
  let summary = "signed ceil integer division operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `ceildivi_signed` ssa-use `,` ssa-use `:` type
    ```

    Signed integer division. Rounds towards positive infinity, i.e. `7 / -2 = -3`.

    Note: the semantics of division by zero or signed division overflow (minimum
    value divided by -1) is TBD; do NOT assume any specific behavior.

    Example:

    ```mlir
    // Scalar signed integer division.
    %a = ceildivi_signed %b, %c : i64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SignedRemIOp
//===----------------------------------------------------------------------===//

def SignedRemIOp : IntBinaryOp<"remi_signed"> {
  let summary = "signed integer division remainder operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.remi_signed` ssa-use `,` ssa-use `:` type
    ```

    Signed integer division remainder. Treats the leading bit as sign, i.e. `6 %
    -2 = 0`.

    Note: the semantics of division by zero is TBD; do NOT assume any specific
    behavior.

    Example:

    ```mlir
    // Scalar signed integer division remainder.
    %a = remis %b, %c : i64

    // SIMD vector element-wise division remainder.
    %f = remis %g, %h : vector<4xi32>

    // Tensor element-wise integer division remainder.
    %x = remis %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SignedShiftRightOp
//===----------------------------------------------------------------------===//

def SignedShiftRightOp : IntBinaryOp<"shift_right_signed"> {
  let summary = "signed integer right-shift";
  let description = [{
    The shift_right_signed operation shifts an integer value to the right by
    a variable amount. The integer is interpreted as signed. The high order
    bits in the output are filled with copies of the most-significant bit
    of the shifted value (which means that the sign of the value is preserved).

    Example:

    ```mlir
    %1 = constant 160 : i8                             // %1 is 0b10100000
    %2 = constant 3 : i8
    %3 = shift_right_signed %1, %2 : (i8, i8) -> i8    // %3 is 0b11110100
    %4 = constant 96 : i8                              // %4 is 0b01100000
    %5 = shift_right_signed %4, %2 : (i8, i8) -> i8    // %5 is 0b00001100
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SignExtendIOp
//===----------------------------------------------------------------------===//

def SignExtendIOp : Std_Op<"sexti", [NoSideEffect,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
    ElementwiseMappable.traits> {
  let summary = "integer sign extension operation";
  let description = [{
    The integer sign extension operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be larger than the input bit-width (N > M).
    The top-most (N - M) bits of the output are filled with copies
    of the most-significant bit of the input.

    Example:

    ```mlir
    %1 = constant 5 : i3            // %1 is 0b101
    %2 = sexti %1 : i3 to i6        // %2 is 0b111101
    %3 = constant 2 : i3            // %3 is 0b010
    %4 = sexti %3 : i3 to i6        // %4 is 0b000010

    %5 = sexti %0 : vector<2 x i32> to vector<2 x i64>
    ```
  }];

  let arguments = (ins SignlessIntegerLike:$value);
  let results = (outs SignlessIntegerLike);

  let builders = [
    OpBuilder<(ins "Value":$value, "Type":$destType), [{
      $_state.addOperands(value);
      $_state.addTypes(destType);
    }]>];

  let parser = [{
    return impl::parseCastOp(parser, result);
  }];
  let printer = [{
    return printStandardCastOp(this->getOperation(), p);
  }];
}

//===----------------------------------------------------------------------===//
// SIToFPOp
//===----------------------------------------------------------------------===//

def SIToFPOp : ArithmeticCastOp<"sitofp">, Arguments<(ins AnyType:$in)> {
  let summary = "cast from integer type to floating-point";
  let description = [{
    Cast from a value interpreted as signed or vector of signed integers to the
    corresponding floating-point scalar or vector value. If the value cannot be
    exactly represented, it is rounded using the default rounding mode. Scalars
    and vector types are currently supported.
  }];
}

//===----------------------------------------------------------------------===//
// SplatOp
//===----------------------------------------------------------------------===//

def SplatOp : Std_Op<"splat", [NoSideEffect,
     TypesMatchWith<"operand type matches element type of result",
                    "aggregate", "input",
                    "$_self.cast<ShapedType>().getElementType()">]> {
  let summary = "splat or broadcast operation";
  let description = [{
    Broadcast the operand to all elements of the result vector or tensor. The
    operand has to be of either integer or float type. When the result is a
    tensor, it has to be statically shaped.

    Example:

    ```mlir
    %s = load %A[%i] : memref<128xf32>
    %v = splat %s : vector<4xf32>
    %t = splat %s : tensor<8x16xi32>
    ```

    TODO: This operation is easy to extend to broadcast to dynamically shaped
    tensors in the same way dynamically shaped memrefs are handled.

    ```mlir
    // Broadcasts %s to a 2-d dynamically shaped tensor, with %m, %n binding
    // to the sizes of the two dynamic dimensions.
    %m = "foo"() : () -> (index)
    %n = "bar"() : () -> (index)
    %t = splat %s [%m, %n] : tensor<?x?xi32>
    ```
  }];

  let arguments = (ins AnyTypeOf<[AnySignlessInteger, AnyFloat],
                                 "integer or float type">:$input);
  let results = (outs AnyTypeOf<[AnyVector, AnyStaticShapeTensor]>:$aggregate);

  let builders = [
    OpBuilder<(ins "Value":$element, "Type":$aggregateType),
    [{ build($_builder, $_state, aggregateType, element); }]>];

  let hasFolder = 1;

  let assemblyFormat = "$input attr-dict `:` type($aggregate)";
}

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

def SubFOp : FloatBinaryOp<"subf"> {
  let summary = "floating point subtraction operation";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

def SubIOp : IntBinaryOp<"subi"> {
  let summary = "integer subtraction operation";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SubTensorOp
//===----------------------------------------------------------------------===//

def SubTensorOp : BaseOpWithOffsetSizesAndStrides<
    StandardOps_Dialect, "subtensor", [NoSideEffect, AttrSizedOperandSegments,
                                       OffsetSizeAndStrideOpInterface]> {
  let summary = "subtensor operation";
  let description = [{
    The "subtensor" operation extract a tensor from another tensor as
    specified by the operation's offsets, sizes and strides arguments.

    The subtensor operation supports the following arguments:

    * source: the "base" tensor from which to extract a subtensor.
    * offsets: tensor-rank number of offsets into the "base" tensor from which
               to extract the subtensor.
    * sizes: tensor-rank number of sizes which specify the sizes of the result
             tensor type.
    * strides: tensor-rank number of strides specifying subsampling in each
               dimension.

    The representation based on offsets, sizes and strides support a
    partially-static specification via attributes specified through the
    `static_offsets`, `static_sizes` and `static_strides` arguments. A special
    sentinel value ShapedType::kDynamicSize and
    ShapedType::kDynamicStrideOrOffset encodes that the corresponding entry has
    a dynamic value.

    After buffer-allocation, the "subtensor" op is expected to lower into a
    "subview" op.

    A subtensor operation may additionally reduce the rank of the resulting
    tensor by removing dimensions that are statically known to be of size 1.

    Example:

    ```
    // Rank-reducing subtensor.
    %1 = subtensor %0[0, 0, 0][1, 16, 4][1, 1, 1] :
      tensor<8x16x4xf32> to tensor<16x4xf32>
    %3 = subtensor %2[3, 4, 2][1, 6, 3][1, 1, 1] :
      tensor<8x16x4xf32> to tensor<6x3xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    I64ArrayAttr:$static_offsets,
    I64ArrayAttr:$static_sizes,
    I64ArrayAttr:$static_strides
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source ``
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    custom<OperandsOrIntegersSizesList>($sizes, $static_sizes)
    custom<OperandsOrIntegersOffsetsOrStridesList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
  }];

  let builders = [
    // Build a SubTensorOp with mixed static and dynamic entries and inferred
    // result type.
    OpBuilder<(ins "Value":$source, "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$sizes, "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubTensorOp with mixed static and dynamic entries and custom
    // result type. If the type passed is nullptr, it is inferred.
    OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubTensorOp with dynamic entries and custom result type. If the
    // type passed is nullptr, it is inferred.
    OpBuilder<(ins "Value":$source, "ValueRange":$offsets,
      "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubTensorOp with dynamic entries and inferred result type.
    OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Returns the type of the base tensor operand.
    RankedTensorType getSourceType() {
      return source().getType().cast<RankedTensorType>();
    }

    /// The result of a subtensor is always a tensor.
    RankedTensorType getType() {
      return getResult().getType().cast<RankedTensorType>();
    }

    /// A subtensor result type can be fully inferred from the source type and
    /// the static representation of offsets, sizes and strides. Special
    /// sentinels encode the dynamic case.
    static Type inferResultType(RankedTensorType sourceRankedTensorType,
                                ArrayRef<int64_t> staticOffsets,
                                ArrayRef<int64_t> staticSizes,
                                ArrayRef<int64_t> staticStrides);
    static Type inferResultType(RankedTensorType sourceRankedTensorType,
                                ArrayRef<OpFoldResult> staticOffsets,
                                ArrayRef<OpFoldResult> staticSizes,
                                ArrayRef<OpFoldResult> staticStrides);
    static Type inferRankReducedResultType(unsigned resultRank,
                                           RankedTensorType sourceRankedTensorType,
                                           ArrayRef<int64_t> staticOffsets,
                                           ArrayRef<int64_t> staticSizes,
                                           ArrayRef<int64_t> staticStrides);
    static Type inferRankReducedResultType(unsigned resultRank,
                                           RankedTensorType sourceRankedTensorType,
                                           ArrayRef<OpFoldResult> staticOffsets,
                                           ArrayRef<OpFoldResult> staticSizes,
                                           ArrayRef<OpFoldResult> staticStrides);

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SubTensorInsertOp
//===----------------------------------------------------------------------===//

def SubTensorInsertOp : BaseOpWithOffsetSizesAndStrides<
    StandardOps_Dialect, "subtensor_insert",
    [NoSideEffect, AttrSizedOperandSegments, OffsetSizeAndStrideOpInterface,
     TypesMatchWith<"expected result type to match dest type",
                    "dest", "result", "$_self">]> {
  let summary = "subtensor_insert operation";
  let description = [{
    The "subtensor_insert" operation insert a tensor `source` into another
    tensor `dest` as specified by the operation's offsets, sizes and strides
    arguments.

    It returns a copy of `dest` with the proper subtensor updated with the value
    of `source`.

    The subtensor_insert operation has the encodes the following information:

    * source: the tensor that is inserted.
    * dest: the tensor into which the source tensor is inserted.
    * offsets: tensor-rank number of offsets into the "base" tensor from which
               to extract the subtensor.
    * sizes: tensor-rank number of sizes which specify the sizes of the result
             tensor type.
    * strides: tensor-rank number of strides that specify subsampling in each
               dimension.

    The representation based on offsets, sizes and strides support a
    partially-static specification via attributes specified through the
    `static_offsets`, `static_sizes` and `static_strides` arguments. A special
    sentinel value ShapedType::kDynamicSize and
    ShapedType::kDynamicStrideOrOffset encodes that the corresponding entry has
    a dynamic value.

    After buffer-allocation, the "subtensor_insert" op is expected to become
    an in-place buffer update.
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    AnyRankedTensor:$dest,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    I64ArrayAttr:$static_offsets,
    I64ArrayAttr:$static_sizes,
    I64ArrayAttr:$static_strides
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source `into` $dest ``
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    custom<OperandsOrIntegersSizesList>($sizes, $static_sizes)
    custom<OperandsOrIntegersOffsetsOrStridesList>($strides, $static_strides)
    attr-dict `:` type($source) `into` type($dest)
  }];

  let verifier = ?;

  let builders = [
    // Build a SubTensorInsertOp with mixed static and dynamic entries.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a SubTensorInsertOp with dynamic entries.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Returns the type of the base tensor operand.
    RankedTensorType getSourceType() {
      return source().getType().cast<RankedTensorType>();
    }

    /// The result of a subtensor_insert is always a tensor.
    RankedTensorType getType() {
      return getResult().getType().cast<RankedTensorType>();
    }

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TruncateIOp
//===----------------------------------------------------------------------===//

def TruncateIOp : Std_Op<"trunci", [NoSideEffect,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
    ElementwiseMappable.traits> {
  let summary = "integer truncation operation";
  let description = [{
    The integer truncation operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be smaller than the input bit-width (N < M).
    The top-most (N - M) bits of the input are discarded.

    Example:

    ```mlir
      %1 = constant 21 : i5           // %1 is 0b10101
      %2 = trunci %1 : i5 to i4       // %2 is 0b0101
      %3 = trunci %1 : i5 to i3       // %3 is 0b101

      %5 = trunci %0 : vector<2 x i32> to vector<2 x i16>
    ```
  }];

  let arguments = (ins SignlessIntegerLike:$value);
  let results = (outs SignlessIntegerLike);

  let builders = [
    OpBuilder<(ins "Value":$value, "Type":$destType), [{
      $_state.addOperands(value);
      $_state.addTypes(destType);
    }]>];

  let parser = [{
    return impl::parseCastOp(parser, result);
  }];
  let printer = [{
    return printStandardCastOp(this->getOperation(), p);
  }];
}

//===----------------------------------------------------------------------===//
// UIToFPOp
//===----------------------------------------------------------------------===//

def UIToFPOp : ArithmeticCastOp<"uitofp">, Arguments<(ins AnyType:$in)> {
  let summary = "cast from unsigned integer type to floating-point";
  let description = [{
    Cast from a value interpreted as unsigned integer or vector of unsigned
    integers to the corresponding scalar or vector floating-point value. If the
    value cannot be exactly represented, it is rounded using the default
    rounding mode. Scalars and vector types are currently supported.
  }];
}

//===----------------------------------------------------------------------===//
// UnsignedDivIOp
//===----------------------------------------------------------------------===//

def UnsignedDivIOp : IntBinaryOp<"divi_unsigned"> {
  let summary = "unsigned integer division operation";
  let description = [{
    Syntax:
    ```
    operation ::= ssa-id `=` `std.divi_unsigned` ssa-use `,` ssa-use `:` type
    ```

    Unsigned integer division. Rounds towards zero. Treats the leading bit as
    the most significant, i.e. for `i16` given two's complement representation,
    `6 / -2 = 6 / (2^16 - 2) = 0`.

    Note: the semantics of division by zero is TBD; do NOT assume any specific
    behavior.

    Example:

    ```mlir
    // Scalar unsigned integer division.
    %a = diviu %b, %c : i64

    // SIMD vector element-wise division.
    %f = diviu %g, %h : vector<4xi32>

    // Tensor element-wise integer division.
    %x = diviu %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// UnsignedRemIOp
//===----------------------------------------------------------------------===//

def UnsignedRemIOp : IntBinaryOp<"remi_unsigned"> {
  let summary = "unsigned integer division remainder operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.remi_unsigned` ssa-use `,` ssa-use `:` type
    ```

    Unsigned integer division remainder. Treats the leading bit as the most
    significant, i.e. for `i16`, `6 % -2 = 6 % (2^16 - 2) = 6`.

    Note: the semantics of division by zero is TBD; do NOT assume any specific
    behavior.

    Example:

    ```mlir
    // Scalar unsigned integer division remainder.
    %a = remiu %b, %c : i64

    // SIMD vector element-wise division remainder.
    %f = remiu %g, %h : vector<4xi32>

    // Tensor element-wise integer division remainder.
    %x = remiu %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// UnsignedShiftRightOp
//===----------------------------------------------------------------------===//

def UnsignedShiftRightOp : IntBinaryOp<"shift_right_unsigned"> {
  let summary = "unsigned integer right-shift";
  let description = [{
    The shift_right_unsigned operation shifts an integer value to the right by
    a variable amount. The integer is interpreted as unsigned. The high order
    bits are always filled with zeros.

    Example:

    ```mlir
    %1 = constant 160 : i8                               // %1 is 0b10100000
    %2 = constant 3 : i8
    %3 = shift_right_unsigned %1, %2 : (i8, i8) -> i8    // %3 is 0b00010100
    ```
  }];
}

//===----------------------------------------------------------------------===//
// XOrOp
//===----------------------------------------------------------------------===//

def XOrOp : IntBinaryOp<"xor", [Commutative]> {
  let summary = "integer binary xor";
  let description = [{
    The `xor` operation takes two operands and returns one result, each of these
    is required to be the same type. This type may be an integer scalar type, a
    vector whose element type is integer, or a tensor of integers. It has no
    standard attributes.

    Example:

    ```mlir
    // Scalar integer bitwise xor.
    %a = xor %b, %c : i64

    // SIMD vector element-wise bitwise integer xor.
    %f = xor %g, %h : vector<4xi32>

    // Tensor element-wise bitwise integer xor.
    %x = xor %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ZeroExtendIOp
//===----------------------------------------------------------------------===//

def ZeroExtendIOp : Std_Op<"zexti", [NoSideEffect,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
    ElementwiseMappable.traits> {
  let summary = "integer zero extension operation";
  let description = [{
    The integer zero extension operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be larger than the input bit-width (N > M).
    The top-most (N - M) bits of the output are filled with zeros.

    Example:

    ```mlir
      %1 = constant 5 : i3            // %1 is 0b101
      %2 = zexti %1 : i3 to i6        // %2 is 0b000101
      %3 = constant 2 : i3            // %3 is 0b010
      %4 = zexti %3 : i3 to i6        // %4 is 0b000010

      %5 = zexti %0 : vector<2 x i32> to vector<2 x i64>
    ```
  }];

  let arguments = (ins SignlessIntegerLike:$value);
  let results = (outs SignlessIntegerLike);

  let builders = [
    OpBuilder<(ins "Value":$value, "Type":$destType), [{
      $_state.addOperands(value);
      $_state.addTypes(destType);
    }]>];

  let parser = [{
    return impl::parseCastOp(parser, result);
  }];
  let printer = [{
    return printStandardCastOp(this->getOperation(), p);
  }];
}

#endif // STANDARD_OPS
