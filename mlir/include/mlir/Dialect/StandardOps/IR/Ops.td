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

include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
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

// Base class for standard cast operations. Requires single operand and result,
// but does not constrain them to specific types.
class CastOp<string mnemonic, list<OpTrait> traits = []> :
    Std_Op<mnemonic,
           !listconcat(traits, [NoSideEffect, SameOperandsAndResultShape])> {

  let results = (outs AnyType);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value source, Type destType", [{
       impl::buildCastOp(builder, result, source, destType);
  }]>];

  let parser = [{
    return impl::parseCastOp(parser, result);
  }];
  let printer = [{
    return printStandardCastOp(this->getOperation(), p);
  }];
  let verifier = [{ return ::verifyCastOp(*this); }];

  let hasFolder = 1;
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
    UnaryOpSameOperandAndResultType<mnemonic, traits>,
    Arguments<(ins FloatLike:$operand)>;

// Base class for standard arithmetic operations.  Requires operands and
// results to be of the same type, but does not constrain them to specific
// types.  Individual classes will have `lhs` and `rhs` accessor to operands.
class ArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    Op<StandardOps_Dialect, mnemonic,
       !listconcat(traits, [NoSideEffect, SameOperandsAndResultType])> {

  let results = (outs AnyType);

  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];

  let printer = [{
    return printStandardBinaryOp(this->getOperation(), p);
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
class IntArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    ArithmeticOp<mnemonic, traits>,
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
class FloatArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    ArithmeticOp<mnemonic, traits>,
    Arguments<(ins FloatLike:$lhs, FloatLike:$rhs)>;

// Base class for standard arithmetic operations on complex numbers with a
// floating-point element type.
// These operations take two operands and return one result, all of which must
// be complex numbers of the same type.
// The assembly format is as follows
//
//     <op>cf %0, %1 : complex<f32>
//
class ComplexFloatArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    ArithmeticOp<mnemonic, traits>,
    Arguments<(ins Complex<AnyFloat>:$lhs, Complex<AnyFloat>:$rhs)>;

// Base class for memref allocating ops: alloca and alloc.
//
//   %0 = alloclike(%m)[%s] : memref<8x?xf32, (d0, d1)[s0] -> ((d0 + s0), d1)>
//
class AllocLikeOp<string mnemonic,
                  Resource resource,
                  list<OpTrait> traits = []> :
    Std_Op<mnemonic, !listconcat([MemoryEffects<[MemAlloc<resource>]>], traits)> {

  let arguments = (ins Variadic<Index>:$value,
                   Confined<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$alignment);
  let results = (outs Res<AnyMemRef, "", [MemAlloc<resource>]>);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, MemRefType memrefType", [{
       result.types.push_back(memrefType);
     }]>,
    OpBuilder<
    "OpBuilder &builder, OperationState &result, MemRefType memrefType, " #
    "ValueRange operands, IntegerAttr alignment = IntegerAttr()", [{
       result.addOperands(operands);
       result.types.push_back(memrefType);
       if (alignment)
         result.addAttribute(getAlignmentAttrName(), alignment);
     }]>];

  let extraClassDeclaration = [{
    static StringRef getAlignmentAttrName() { return "alignment"; }

    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

    /// Returns the number of symbolic operands (the ones in square brackets),
    /// which bind to the symbols of the memref's layout map.
    unsigned getNumSymbolicOperands() {
      return getNumOperands() - getType().getNumDynamicDims();
    }

    /// Returns the symbolic operands (the ones in square brackets), which bind
    /// to the symbols of the memref's layout map.
    operand_range getSymbolicOperands() {
      return {operand_begin() + getType().getNumDynamicDims(), operand_end()};
    }

    /// Returns the dynamic sizes for this alloc operation if specified.
    operand_range getDynamicSizes() { return getOperands(); }
  }];

  let parser = [{ return ::parseAllocLikeOp(parser, result); }];

  let hasCanonicalizer = 1;
}

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
// AddCFOp
//===----------------------------------------------------------------------===//

def AddCFOp : ComplexFloatArithmeticOp<"addcf"> {
  let summary = "complex number addition";
  let description = [{
    The `addcf` operation takes two complex number operands and returns their
    sum, a single complex number.
    All operands and result must be of the same type, a complex number with a
    floating-point element type.

    Example:

    ```mlir
    %a = addcf %b, %c : complex<f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

def AddFOp : FloatArithmeticOp<"addf"> {
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

def AddIOp : IntArithmeticOp<"addi", [Commutative]> {
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
// AllocOp
//===----------------------------------------------------------------------===//

def AllocOp : AllocLikeOp<"alloc", DefaultResource> {
  let summary = "memory allocation operation";
  let description = [{
    The `alloc` operation allocates a region of memory, as specified by its
    memref type.

    Example:

    ```mlir
    %0 = alloc() : memref<8x64xf32, 1>
    ```

    The optional list of dimension operands are bound to the dynamic dimensions
    specified in its memref type. In the example below, the ssa value '%d' is
    bound to the second dimension of the memref (which is dynamic).

    ```mlir
    %0 = alloc(%d) : memref<8x?xf32, 1>
    ```

    The optional list of symbol operands are bound to the symbols of the
    memrefs affine map. In the example below, the ssa value '%s' is bound to
    the symbol 's0' in the affine map specified in the allocs memref type.

    ```mlir
    %0 = alloc()[%s] : memref<8x64xf32,
                              affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
    ```

    This operation returns a single ssa value of memref type, which can be used
    by subsequent load and store operations.

    The optional `alignment` attribute may be specified to ensure that the
    region of memory that will be indexed is aligned at the specified byte
    boundary.

    ```mlir
    %0 = alloc()[%s] {alignment = 8} :
      memref<8x64xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

def AllocaOp : AllocLikeOp<"alloca", AutomaticAllocationScopeResource> {
  let summary = "stack memory allocation operation";
  let description = [{
    The `alloca` operation allocates memory on the stack, to be automatically
    released when control transfers back from the region of its closest
    surrounding operation with an
    [`AutomaticAllocationScope`](../Traits.md#automaticallocationscope) trait.
    The amount of memory allocated is specified by its memref and additional
    operands. For example:

    ```mlir
		%0 = alloca() : memref<8x64xf32>
    ```

    The optional list of dimension operands are bound to the dynamic dimensions
    specified in its memref type. In the example below, the SSA value '%d' is
    bound to the second dimension of the memref (which is dynamic).

    ```mlir
		%0 = alloca(%d) : memref<8x?xf32>
    ```

    The optional list of symbol operands are bound to the symbols of the
    memref's affine map. In the example below, the SSA value '%s' is bound to
    the symbol 's0' in the affine map specified in the allocs memref type.

    ```mlir
		%0 = alloca()[%s] : memref<8x64xf32,
                               affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>>
    ```

    This operation returns a single SSA value of memref type, which can be used
    by subsequent load and store operations. An optional alignment attribute, if
    specified, guarantees alignment at least to that boundary. If not specified,
    an alignment on any convenient boundary compatible with the type will be
    chosen.
  }];
}

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

def AndOp : IntArithmeticOp<"and", [Commutative]> {
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
// AssumeAlignmentOp
//===----------------------------------------------------------------------===//

def AssumeAlignmentOp : Std_Op<"assume_alignment"> {
  let summary =
      "assertion that gives alignment information to the input memref";
  let description = [{
    The `assume_alignment` operation takes a memref and an integer of alignment
    value, and internally annotates the buffer with the given alignment. If
    the buffer isn't aligned to the given alignment, the behavior is undefined.

    This operation doesn't affect the semantics of a correct program. It's for
    optimization only, and the optimization is best-effort.
  }];
  let arguments = (ins AnyMemRef:$memref,
                       Confined<I32Attr, [IntPositive]>:$alignment);
  let results = (outs);

  let assemblyFormat = "$memref `,` $alignment attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// AtomicRMWOp
//===----------------------------------------------------------------------===//

def ATOMIC_RMW_KIND_ADDF    : I64EnumAttrCase<"addf", 0>;
def ATOMIC_RMW_KIND_ADDI    : I64EnumAttrCase<"addi", 1>;
def ATOMIC_RMW_KIND_ASSIGN  : I64EnumAttrCase<"assign", 2>;
def ATOMIC_RMW_KIND_MAXF    : I64EnumAttrCase<"maxf", 3>;
def ATOMIC_RMW_KIND_MAXS    : I64EnumAttrCase<"maxs", 4>;
def ATOMIC_RMW_KIND_MAXU    : I64EnumAttrCase<"maxu", 5>;
def ATOMIC_RMW_KIND_MINF    : I64EnumAttrCase<"minf", 6>;
def ATOMIC_RMW_KIND_MINS    : I64EnumAttrCase<"mins", 7>;
def ATOMIC_RMW_KIND_MINU    : I64EnumAttrCase<"minu", 8>;
def ATOMIC_RMW_KIND_MULF    : I64EnumAttrCase<"mulf", 9>;
def ATOMIC_RMW_KIND_MULI    : I64EnumAttrCase<"muli", 10>;

def AtomicRMWKindAttr : I64EnumAttr<
    "AtomicRMWKind", "",
    [ATOMIC_RMW_KIND_ADDF, ATOMIC_RMW_KIND_ADDI, ATOMIC_RMW_KIND_ASSIGN,
     ATOMIC_RMW_KIND_MAXF, ATOMIC_RMW_KIND_MAXS, ATOMIC_RMW_KIND_MAXU,
     ATOMIC_RMW_KIND_MINF, ATOMIC_RMW_KIND_MINS, ATOMIC_RMW_KIND_MINU,
     ATOMIC_RMW_KIND_MULF, ATOMIC_RMW_KIND_MULI]> {
  let cppNamespace = "::mlir";
}

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
    The `atomic_rmw` operation provides a way to perform a read-modify-write
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
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "Value memref, ValueRange ivs">
  ];

  let extraClassDeclaration = [{
    // The value stored in memref[ivs].
    Value getCurrentValue() {
      return body().front().getArgument(0);
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

  let builders = [OpBuilder<"OpBuilder &, OperationState &result, Block *dest, "
                            "ValueRange destOperands = {}", [{
    result.addSuccessors(dest);
    result.addOperands(destOperands);
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

def CallOp : Std_Op<"call", [CallOpInterface]> {
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

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, FuncOp callee,"
    "ValueRange operands = {}", [{
      result.addOperands(operands);
      result.addAttribute("callee", builder.getSymbolRefAttr(callee));
      result.addTypes(callee.getType().getResults());
  }]>, OpBuilder<
    "OpBuilder &builder, OperationState &result, SymbolRefAttr callee,"
    "ArrayRef<Type> results, ValueRange operands = {}", [{
      result.addOperands(operands);
      result.addAttribute("callee", callee);
      result.addTypes(results);
  }]>, OpBuilder<
    "OpBuilder &builder, OperationState &result, StringRef callee,"
    "ArrayRef<Type> results, ValueRange operands = {}", [{
      build(builder, result, builder.getSymbolRefAttr(callee), results,
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
      return getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
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

  let builders = [OpBuilder<
    "OpBuilder &, OperationState &result, Value callee,"
    "ValueRange operands = {}", [{
      result.operands.push_back(callee);
      result.addOperands(operands);
      result.addTypes(callee.getType().cast<FunctionType>().getResults());
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

def CmpFOp : Std_Op<"cmpf",
    [NoSideEffect, SameTypeOperands, SameOperandsAndResultShape,
     TypesMatchWith<
       "result type has i1 element type and same shape as operands",
       "lhs", "result", "getI1SameShape($_self)">]> {
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

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, CmpFPredicate predicate,"
    "Value lhs, Value rhs", [{
      ::buildCmpFOp(builder, result, predicate, lhs, rhs);
  }]>];

  let extraClassDeclaration = [{
    static StringRef getPredicateAttrName() { return "predicate"; }
    static CmpFPredicate getPredicateByName(StringRef name);

    CmpFPredicate getPredicate() {
      return (CmpFPredicate)getAttrOfType<IntegerAttr>(getPredicateAttrName())
          .getInt();
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

def CmpIOp : Std_Op<"cmpi",
    [NoSideEffect, SameTypeOperands, SameOperandsAndResultShape,
     TypesMatchWith<
       "result type has i1 element type and same shape as operands",
       "lhs", "result", "getI1SameShape($_self)">]> {
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

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, CmpIPredicate predicate,"
    "Value lhs, Value rhs", [{
      ::buildCmpIOp(builder, result, predicate, lhs, rhs);
  }]>];

  let extraClassDeclaration = [{
    static StringRef getPredicateAttrName() { return "predicate"; }
    static CmpIPredicate getPredicateByName(StringRef name);

    CmpIPredicate getPredicate() {
      return (CmpIPredicate)getAttrOfType<IntegerAttr>(getPredicateAttrName())
          .getInt();
    }
  }];

  let verifier = [{ return success(); }];

  let hasFolder = 1;

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

//===----------------------------------------------------------------------===//
// CreateComplexOp
//===----------------------------------------------------------------------===//

def CreateComplexOp : Std_Op<"create_complex",
    [NoSideEffect,
     AllTypesMatch<["real", "imaginary"]>,
     TypesMatchWith<"complex element type matches real operand type",
                    "complex", "real",
                    "$_self.cast<ComplexType>().getElementType()">,
     TypesMatchWith<"complex element type matches imaginary operand type",
                    "complex", "imaginary",
                    "$_self.cast<ComplexType>().getElementType()">]> {
  let summary = "creates a complex number";
  let description = [{
    The `create_complex` operation creates a complex number from two
    floating-point operands, the real and the imaginary part.

    Example:

    ```mlir
    %a = create_complex %b, %c : complex<f32>
    ```
  }];

  let arguments = (ins AnyFloat:$real, AnyFloat:$imaginary);
  let results = (outs Complex<AnyFloat>:$complex);

  let assemblyFormat = "$real `,` $imaginary attr-dict `:` type($complex)";

  // `CreateComplexOp` is fully verified by its traits.
  let verifier = ?;
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

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value condition,"
    "Block *trueDest, ValueRange trueOperands,"
    "Block *falseDest, ValueRange falseOperands", [{
      build(builder, result, condition, trueOperands, falseOperands, trueDest,
            falseDest);
  }]>, OpBuilder<
    "OpBuilder &builder, OperationState &result, Value condition,"
    "Block *trueDest, Block *falseDest, ValueRange falseOperands = {}", [{
      build(builder, result, condition, trueDest, ValueRange(), falseDest,
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

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Attribute value",
    [{ build(builder, result, value.getType(), value); }]>];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }

    /// Returns true if a constant operation can be built with the given value
    /// and result type.
    static bool isBuildableWith(Attribute value, Type type);
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CopySignOp
//===----------------------------------------------------------------------===//

def CopySignOp : FloatArithmeticOp<"copysign"> {
  let summary = "A copysign operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.copysign` ssa-use `:` type
    ```

    The `copysign` returns a value with the magnitude of the first operand and
    the sign of the second operand. It takes two operands and returns one
    result of the same type. This type may be a float scalar type, a vector
    whose element type is float, or a tensor of floats. It has no standard
    attributes.

    Example:

    ```mlir
    // Scalar copysign value.
    %a = copysign %b %c : f64

    // SIMD vector element-wise copysign value.
    %f = copysign %g %h : vector<4xf32>

    // Tensor element-wise copysign value.
    %x = copysign %y %z : tensor<4x?xf8>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// CosOp
//===----------------------------------------------------------------------===//

def CosOp : FloatUnaryOp<"cos"> {
  let summary = "cosine of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.cos` ssa-use `:` type
    ```

    The `cos` operation computes the cosine of a given value. It takes one
    operand and returns one result of the same type. This type may be a float
    scalar type, a vector whose element type is float, or a tensor of floats.
    It has no standard attributes.

    Example:

    ```mlir
    // Scalar cosine value.
    %a = cos %b : f64

    // SIMD vector element-wise cosine value.
    %f = cos %g : vector<4xf32>

    // Tensor element-wise cosine value.
    %x = cos %y : tensor<4x?xf8>
    ```
  }];
}

def SinOp : FloatUnaryOp<"sin"> {
  let summary = "sine of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.sin` ssa-use `:` type
    ```

    The `sin` operation computes the sine of a given value. It takes one
    operand and returns one result of the same type. This type may be a float
    scalar type, a vector whose element type is float, or a tensor of floats.
    It has no standard attributes.

    Example:

    ```mlir
    // Scalar sine value.
    %a = sin %b : f64

    // SIMD vector element-wise sine value.
    %f = sin %g : vector<4xf32>

    // Tensor element-wise sine value.
    %x = sin %y : tensor<4x?xf8>
    ```
  }];
}


//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

def DeallocOp : Std_Op<"dealloc", [MemoryEffects<[MemFree]>]> {
  let summary = "memory deallocation operation";
  let description = [{
    The `dealloc` operation frees the region of memory referenced by a memref
    which was originally created by the `alloc` operation.
    The `dealloc` operation should not be called on memrefs which alias an
    alloc'd memref (e.g. memrefs returned by `view` operations).

    Example:

    ```mlir
    %0 = alloc() : memref<8x64xf32, (d0, d1) -> (d0, d1), 1>
    dealloc %0 : memref<8x64xf32, (d0, d1) -> (d0, d1), 1>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "", [MemFree]>:$memref);

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

def DimOp : Std_Op<"dim", [NoSideEffect]> {
  let summary = "dimension index operation";
  let description = [{
    The `dim` operation takes a memref/tensor and a dimension operand of type
    `index`.
    It returns the size of the requested dimension of the given memref/tensor.

    The specified memref or tensor type is that of the first operand.

    Example:

    ```mlir
    // Always returns 4, can be constant folded:
    %c0 = constant 0 : index
    %x = = dim %A, %c0 : tensor<4 x ? x f32>

    // Returns the dynamic dimension of %A.
    %c1 = constant 1 : index
    %y = dim %A, %c1 : tensor<4 x ? x f32>

    // Equivalent generic form:
    %x = "std.dim"(%A, %c0) : (tensor<4 x ? x f32>, index) -> index
    %y = "std.dim"(%A, %c1) : (tensor<4 x ? x f32>, index) -> index
    ```
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyTensor],
                                 "any tensor or memref type">:$memrefOrTensor,
                       Index:$index);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    attr-dict $memrefOrTensor `,` $index `:` type($memrefOrTensor)
  }];

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "Value memrefOrTensor, int64_t index">
  ];

  let extraClassDeclaration = [{
    /// Helper function to get the index as a simple integer if it is constant.
    Optional<int64_t> getConstantIndex();
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// DivFOp
//===----------------------------------------------------------------------===//

def DivFOp : FloatArithmeticOp<"divf"> {
  let summary = "floating point division operation";
}

//===----------------------------------------------------------------------===//
// ExpOp
//===----------------------------------------------------------------------===//

def ExpOp : FloatUnaryOp<"exp"> {
  let summary = "base-e exponential of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.exp` ssa-use `:` type
    ```

    The `exp` operation takes one operand and returns one result of the same
    type. This type may be a float scalar type, a vector whose element type is
    float, or a tensor of floats. It has no standard attributes.

    Example:

    ```mlir
    // Scalar natural exponential.
    %a = exp %b : f64

    // SIMD vector element-wise natural exponential.
    %f = exp %g : vector<4xf32>

    // Tensor element-wise natural exponential.
    %x = exp %y : tensor<4x?xf8>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ExpOp
//===----------------------------------------------------------------------===//

def Exp2Op : FloatUnaryOp<"exp2"> {
  let summary = "base-2 exponential of the specified value";
}

//===----------------------------------------------------------------------===//
// ExtractElementOp
//===----------------------------------------------------------------------===//

def ExtractElementOp : Std_Op<"extract_element",
    [NoSideEffect,
     TypesMatchWith<"result type matches element type of aggregate",
                    "aggregate", "result",
                    "$_self.cast<ShapedType>().getElementType()">]> {
  let summary = "element extract operation";
  let description = [{
    The `extract_element` op reads a tensor or vector and returns one element
    from it specified by an index list. The output of the 'extract_element' is a
    new value with the same type as the elements of the tensor or vector. The
    arity of indices matches the rank of the accessed value (i.e., if a tensor
    is of rank 3, then 3 indices are required for the extract. The indices
    should all be of `index` type.

    Example:

    ```mlir
    %3 = extract_element %v[%1, %2] : vector<4x4xi32>
    %4 = extract_element %t[%1, %2] : tensor<4x4xi32>
    %5 = extract_element %ut[%1, %2] : tensor<*xi32>
    ```
  }];

  let arguments = (ins AnyTypeOf<[AnyVector, AnyTensor]>:$aggregate,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$result);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value aggregate,"
    "ValueRange indices = {}", [{
      auto resType = aggregate.getType().cast<ShapedType>()
                                         .getElementType();
      build(builder, result, resType, aggregate, indices);
    }]>];

  let extraClassDeclaration = [{
    Value getAggregate() { return getOperand(0); }

    operand_range getIndices() {
      return {operand_begin() + 1, operand_end()};
    }
  }];

  let hasFolder = 1;

  let assemblyFormat = [{
    $aggregate `[` $indices `]` attr-dict `:` type($aggregate)
  }];
}

//===----------------------------------------------------------------------===//
// TensorFromElementsOp
//===----------------------------------------------------------------------===//

def TensorFromElementsOp : Std_Op<"tensor_from_elements",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "tensor from elements operation.";
  string description = [{
    Create a 1D tensor from a range of same-type arguments.

    Example:

    ```mlir
    tensor_from_elements(i_1, ..., i_N) :  tensor<Nxindex>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs AnyTensor:$result);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, ValueRange elements", [{
      assert(!elements.empty() && "expected at least one element");
      result.addOperands(elements);
      result.addTypes(
          RankedTensorType::get({static_cast<int64_t>(elements.size())},
                                *elements.getTypes().begin()));
    }]>];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// FPExtOp
//===----------------------------------------------------------------------===//

def FPExtOp : CastOp<"fpext">, Arguments<(ins AnyType:$in)> {
  let summary = "cast from floating-point to wider floating-point";
  let description = [{
    Cast a floating-point value to a larger floating-point-typed value.
    The destination type must to be strictly wider than the source type.
    Only scalars are currently supported.
  }];

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);
  }];

  let hasFolder = 0;
}

//===----------------------------------------------------------------------===//
// FPToSIOp
//===----------------------------------------------------------------------===//

def FPToSIOp : CastOp<"fptosi">, Arguments<(ins AnyType:$in)> {
  let summary = "cast from floating-point type to integer type";
  let description = [{
    Cast from a value interpreted as floating-point to the nearest (rounding
    towards zero) signed integer value.
  }];

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);
  }];

  let hasFolder = 0;
}

//===----------------------------------------------------------------------===//
// FPTruncOp
//===----------------------------------------------------------------------===//

def FPTruncOp : CastOp<"fptrunc">, Arguments<(ins AnyType:$in)> {
  let summary = "cast from floating-point to narrower floating-point";
  let description = [{
    Truncate a floating-point value to a smaller floating-point-typed value.
    The destination type must be strictly narrower than the source type.
    If the value cannot be exactly represented, it is rounded using the default
    rounding mode. Only scalars are currently supported.
  }];

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);
  }];

  let hasFolder = 0;
}

//===----------------------------------------------------------------------===//
// ImOp
//===----------------------------------------------------------------------===//

def ImOp : Std_Op<"im",
    [NoSideEffect,
     TypesMatchWith<"complex element type matches result type",
                    "complex", "imaginary",
                    "$_self.cast<ComplexType>().getElementType()">]> {
  let summary = "extracts the imaginary part of a complex number";
  let description = [{
    The `im` operation takes a single complex number as its operand and extracts
    the imaginary part as a floating-point value.

    Example:

    ```mlir
    %a = im %b : complex<f32>
    ```
  }];

  let arguments = (ins Complex<AnyFloat>:$complex);
  let results = (outs AnyFloat:$imaginary);

  let assemblyFormat = "$complex attr-dict `:` type($complex)";

  // `ImOp` is fully verified by its traits.
  let verifier = ?;
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

def IndexCastOp : CastOp<"index_cast">, Arguments<(ins AnyType:$in)> {
  let summary = "cast between index and integer types";
  let description = [{
    Casts between integer scalars and 'index' scalars. Index is an integer of
    platform-specific bit width. If casting to a wider integer, the value is
    sign-extended. If casting to a narrower integer, the value is truncated.
  }];

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def LoadOp : Std_Op<"load",
     [TypesMatchWith<"result type matches element type of 'memref'",
                     "memref", "result",
                     "$_self.cast<MemRefType>().getElementType()">]> {
  let summary = "load operation";
  let description = [{
    The `load` op reads an element from a memref specified by an index list. The
    output of load is a new value with the same type as the elements of the
    memref. The arity of indices is the rank of the memref (i.e., if the memref
    loaded from is of rank 3, then 3 indices are required for the load following
    the memref identifier).

    In an `affine.if` or `affine.for` body, the indices of a load are restricted
    to SSA values bound to surrounding loop induction variables,
    [symbols](../Affine.md#dimensions-and-symbols), results of a
    [`constant` operation](#stdconstant-constantop), or the result of an
    `affine.apply` operation that can in turn take as arguments all of the
    aforementioned SSA values or the recursively result of such an
    `affine.apply` operation.

    Example:

    ```mlir
    %1 = affine.apply affine_map<(d0, d1) -> (3*d0)> (%i, %j)
    %2 = affine.apply affine_map<(d0, d1) -> (d1+1)> (%i, %j)
    %12 = load %A[%1, %2] : memref<8x?xi32, #layout, memspace0>

    // Example of an indirect load (treated as non-affine)
    %3 = affine.apply affine_map<(d0) -> (2*d0 + 1)>(%12)
    %13 = load %A[%3, %2] : memref<4x?xi32, #layout, memspace0>
    ```

    **Context:** The `load` and `store` operations are specifically crafted to
    fully resolve a reference to an element of a memref, and (in affine
    `affine.if` and `affine.for` operations) the compiler can follow use-def
    chains (e.g. through [`affine.apply`](Affine.md#affineapply-affineapplyop)
    operations) to precisely analyze references at compile-time using polyhedral
    techniques. This is possible because of the
    [restrictions on dimensions and symbols](Affine.md#restrictions-on-dimensions-and-symbols)
    in these contexts.
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$result);

  let builders = [OpBuilder<
    "OpBuilder &, OperationState &result, Value memref,"
    "ValueRange indices = {}", [{
      auto memrefType = memref.getType().cast<MemRefType>();
      result.addOperands(memref);
      result.addOperands(indices);
      result.types.push_back(memrefType.getElementType());
  }]>];

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return getMemRef().getType().cast<MemRefType>();
    }

    operand_range getIndices() { return {operand_begin() + 1, operand_end()}; }
  }];

  let hasFolder = 1;

  let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// LogOp
//===----------------------------------------------------------------------===//

def LogOp : FloatUnaryOp<"log"> {
  let summary = "base-e logarithm of the specified value";
}

def Log10Op : FloatUnaryOp<"log10"> {
  let summary = "base-10 logarithm of the specified value";
}

def Log2Op : FloatUnaryOp<"log2"> {
  let summary = "base-2 logarithm of the specified value";
}

//===----------------------------------------------------------------------===//
// MemRefCastOp
//===----------------------------------------------------------------------===//

def MemRefCastOp : CastOp<"memref_cast"> {
  let summary = "memref cast operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.memref_cast` ssa-use `:` type `to` type
    ```

    The `memref_cast` operation converts a memref from one type to an equivalent
    type with a compatible shape. The source and destination types are
    compatible if:

    a. Both are ranked memref types with the same element type, address space,
    and rank and:
      1. Both have the same layout or both have compatible strided layouts.
      2. The individual sizes (resp. offset and strides in the case of strided
         memrefs) may convert constant dimensions to dynamic dimensions and
         vice-versa.

    If the cast converts any dimensions from an unknown to a known size, then it
    acts as an assertion that fails at runtime if the dynamic dimensions
    disagree with resultant destination size.

    Example:

    ```mlir
    // Assert that the input dynamic shape matches the destination static shape.
    %2 = memref_cast %1 : memref<?x?xf32> to memref<4x4xf32>
    // Erase static shape information, replacing it with dynamic information.
    %3 = memref_cast %1 : memref<4xf32> to memref<?xf32>

    // The same holds true for offsets and strides.

    // Assert that the input dynamic shape matches the destination static stride.
    %4 = memref_cast %1 : memref<12x4xf32, offset:?, strides: [?, ?]> to
                          memref<12x4xf32, offset:5, strides: [4, 1]>
    // Erase static offset and stride information, replacing it with
    // dynamic information.
    %5 = memref_cast %1 : memref<12x4xf32, offset:5, strides: [4, 1]> to
                          memref<12x4xf32, offset:?, strides: [?, ?]>
    ```

    b. Either or both memref types are unranked with the same element type, and
    address space.

    Example:

    ```mlir
    Cast to concrete shape.
        %4 = memref_cast %1 : memref<*xf32> to memref<4x?xf32>

    Erase rank information.
        %5 = memref_cast %1 : memref<4x?xf32> to memref<*xf32>
    ```
  }];

  let arguments = (ins AnyRankedOrUnrankedMemRef:$source);
  let results = (outs AnyRankedOrUnrankedMemRef);

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);

    /// The result of a memref_cast is always a memref.
    Type getType() { return getResult().getType(); }
  }];
}

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

def MulFOp : FloatArithmeticOp<"mulf"> {
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

def MulIOp : IntArithmeticOp<"muli", [Commutative]> {
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

def OrOp : IntArithmeticOp<"or", [Commutative]> {
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
// PrefetchOp
//===----------------------------------------------------------------------===//

def PrefetchOp : Std_Op<"prefetch"> {
  let summary = "prefetch operation";
  let description = [{
    The "prefetch" op prefetches data from a memref location described with
    subscript indices similar to std.load, and with three attributes: a
    read/write specifier, a locality hint, and a cache type specifier as shown
    below:

    ```mlir
    prefetch %0[%i, %j], read, locality<3>, data : memref<400x400xi32>
    ```

    The read/write specifier is either 'read' or 'write', the locality hint
    ranges from locality<0> (no locality) to locality<3> (extremely local keep
    in cache). The cache type specifier is either 'data' or 'instr'
    and specifies whether the prefetch is performed on data cache or on
    instruction cache.
  }];

  let arguments = (ins AnyMemRef:$memref, Variadic<Index>:$indices,
                   BoolAttr:$isWrite,
                   Confined<I32Attr, [IntMinValue<0>,
                     IntMaxValue<3>]>:$localityHint,
                   BoolAttr:$isDataCache);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value memref,"
    "ArrayRef<Value> indices, bool isWrite, unsigned hint, bool isData",
    [{
      auto hintAttr = builder.getI32IntegerAttr(hint);
      auto isWriteAttr = builder.getBoolAttr(isWrite);
      auto isDataCacheAttr = builder.getBoolAttr(isData);
      result.addOperands(memref);
      result.addOperands(indices);
      result.addAttribute("localityHint", hintAttr);
      result.addAttribute("isWrite", isWriteAttr);
      result.addAttribute("isDataCache", isDataCacheAttr);
    }]>];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return memref().getType().cast<MemRefType>();
    }
    static StringRef getLocalityHintAttrName() { return "localityHint"; }
    static StringRef getIsWriteAttrName() { return "isWrite"; }
    static StringRef getIsDataCacheAttrName() { return "isDataCache"; }
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// RankOp
//===----------------------------------------------------------------------===//

def RankOp : Std_Op<"rank", [NoSideEffect]> {
  let summary = "rank operation";
  let description = [{
    The `rank` operation takes a tensor operand and returns its rank.

    Example:

    ```mlir
    %1 = rank %0 : tensor<*xf32>
    ```
  }];

  let arguments = (ins AnyTensor);
  let results = (outs Index);
  let verifier = ?;

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value tensor", [{
      auto indexType = builder.getIndexType();
      build(builder, result, indexType, tensor);
    }]>];

  let hasFolder = 1;
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//
// ReOp
//===----------------------------------------------------------------------===//

def ReOp : Std_Op<"re",
    [NoSideEffect,
     TypesMatchWith<"complex element type matches result type",
                    "complex", "real",
                    "$_self.cast<ComplexType>().getElementType()">]> {
  let summary = "extracts the real part of a complex number";
  let description = [{
    The `re` operation takes a single complex number as its operand and extracts
    the real part as a floating-point value.

    Example:

    ```mlir
    %a = re %b : complex<f32>
    ```
  }];

  let arguments = (ins Complex<AnyFloat>:$complex);
  let results = (outs AnyFloat:$real);

  let assemblyFormat = "$complex attr-dict `:` type($complex)";

  // `ReOp` is fully verified by its traits.
  let verifier = ?;
}

//===----------------------------------------------------------------------===//
// RemFOp
//===----------------------------------------------------------------------===//

def RemFOp : FloatArithmeticOp<"remf"> {
  let summary = "floating point division remainder operation";
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Std_Op<"return", [NoSideEffect, HasParent<"FuncOp">, ReturnLike,
                                 Terminator]> {
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

  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// RsqrtOp
//===----------------------------------------------------------------------===//

def RsqrtOp : FloatUnaryOp<"rsqrt"> {
  let summary = "reciprocal of sqrt (1 / sqrt of the specified value)";
  let description = [{
    The `rsqrt` operation computes the reciprocal of the square root. It takes
    one operand and returns one result of the same type. This type may be a
    float scalar type, a vector whose element type is float, or a tensor of
    floats. It has no standard attributes.
  }];
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def SelectOp : Std_Op<"select", [NoSideEffect,
     AllTypesMatch<["true_value", "false_value", "result"]>]> {
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

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value condition,"
    "Value trueValue, Value falseValue", [{
      result.addOperands({condition, trueValue, falseValue});
      result.addTypes(trueValue.getType());
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

def ShiftLeftOp : IntArithmeticOp<"shift_left"> {
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

def SignedDivIOp : IntArithmeticOp<"divi_signed"> {
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
// SignedRemIOp
//===----------------------------------------------------------------------===//

def SignedRemIOp : IntArithmeticOp<"remi_signed"> {
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

def SignedShiftRightOp : IntArithmeticOp<"shift_right_signed"> {
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

def SignExtendIOp : Std_Op<"sexti",
    [NoSideEffect, SameOperandsAndResultShape]> {
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

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value value, Type destType", [{
      result.addOperands(value);
      result.addTypes(destType);
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

def SIToFPOp : CastOp<"sitofp">, Arguments<(ins AnyType:$in)> {
  let summary = "cast from integer type to floating-point";
  let description = [{
    Cast from a value interpreted as signed integer to the corresponding
    floating-point value. If the value cannot be exactly represented, it is
    rounded using the default rounding mode. Only scalars are currently
    supported.
  }];

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);
  }];

  let hasFolder = 0;
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

  let builders =
      [OpBuilder<"OpBuilder &builder, OperationState &result, Value element, "
                  "Type aggregateType",
                  [{ build(builder, result, aggregateType, element); }]>];

  let hasFolder = 1;

  let assemblyFormat = "$input attr-dict `:` type($aggregate)";
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

def SqrtOp : FloatUnaryOp<"sqrt"> {
  let summary = "sqrt of the specified value";
  let description = [{
    The `sqrt` operation computes the square root. It takes one operand and
    returns one result of the same type. This type may be a float scalar type, a
    vector whose element type is float, or a tensor of floats. It has no standard
    attributes.

    Example:

    ```mlir
    // Scalar square root value.
    %a = sqrt %b : f64
    // SIMD vector element-wise square root value.
    %f = sqrt %g : vector<4xf32>
    // Tensor element-wise square root value.
    %x = sqrt %y : tensor<4x?xf32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def StoreOp : Std_Op<"store",
     [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                     "memref", "value",
                     "$_self.cast<MemRefType>().getElementType()">]> {
  let summary = "store operation";
  let description = [{
    Store a value to a memref location given by indices. The value stored should
    have the same type as the elemental type of the memref. The number of
    arguments provided within brackets need to match the rank of the memref.

    In an affine context, the indices of a store are restricted to SSA values
    bound to surrounding loop induction variables,
    [symbols](Affine.md#restrictions-on-dimensions-and-symbols), results of a
    [`constant` operation](#stdconstant-constantop), or the result of an
    [`affine.apply`](Affine.md#affineapply-affineapplyop) operation that can in turn
    take as arguments all of the aforementioned SSA values or the recursively
    result of such an `affine.apply` operation.

    Example:

    ```mlir
    store %100, %A[%1, 1023] : memref<4x?xf32, #layout, memspace0>
    ```

    **Context:** The `load` and `store` operations are specifically crafted to
    fully resolve a reference to an element of a memref, and (in polyhedral
    `affine.if` and `affine.for` operations) the compiler can follow use-def
    chains (e.g. through [`affine.apply`](Affine.md#affineapply-affineapplyop)
    operations) to precisely analyze references at compile-time using polyhedral
    techniques. This is possible because of the
    [restrictions on dimensions and symbols](Affine.md#restrictions-on-dimensions-and-symbols)
    in these contexts.
  }];

  let arguments = (ins AnyType:$value,
                       Arg<AnyMemRef, "the reference to store to",
                           [MemWrite]>:$memref,
                       Variadic<Index>:$indices);

  let builders = [OpBuilder<
    "OpBuilder &, OperationState &result, Value valueToStore, Value memref", [{
      result.addOperands(valueToStore);
      result.addOperands(memref);
  }]>];

  let extraClassDeclaration = [{
      Value getValueToStore() { return getOperand(0); }

      Value getMemRef() { return getOperand(1); }
      void setMemRef(Value value) { setOperand(1, value); }
      MemRefType getMemRefType() {
        return getMemRef().getType().cast<MemRefType>();
      }

      operand_range getIndices() {
        return {operand_begin() + 2, operand_end()};
      }
  }];

  let hasFolder = 1;

  let assemblyFormat = [{
    $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// SubCFOp
//===----------------------------------------------------------------------===//

def SubCFOp : ComplexFloatArithmeticOp<"subcf"> {
  let summary = "complex number subtraction";
  let description = [{
    The `subcf` operation takes two complex number operands and returns their
    difference, a single complex number.
    All operands and result must be of the same type, a complex number with a
    floating-point element type.

    Example:

    ```mlir
    %a = subcf %b, %c : complex<f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

def SubFOp : FloatArithmeticOp<"subf"> {
  let summary = "floating point subtraction operation";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

def SubIOp : IntArithmeticOp<"subi"> {
  let summary = "integer subtraction operation";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SubViewOp
//===----------------------------------------------------------------------===//

def SubViewOp : Std_Op<"subview", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<ViewLikeOpInterface>,
    NoSideEffect,
  ]> {
  let summary = "memref subview operation";
  let description = [{
    The "subview" operation converts a memref type to another memref type
    which represents a reduced-size view of the original memref as specified by
    the operation's offsets, sizes and strides arguments.

    The SubView operation supports the following arguments:
    *) Memref: the "base" memref on which to create a "view" memref.
    *) Offsets: memref-rank number of dynamic offsets or static integer
                attributes into the "base" memref at which to create the "view"
                memref.
    *) Sizes: memref-rank number of dynamic sizes or static integer attributes
              which specify the sizes of the result "view" memref type.
    *) Strides: memref-rank number of dynamic strides or static integer
                attributes multiplicatively to the base memref strides in each
                dimension.

    Example 1:

    ```mlir
    %0 = alloc() : memref<64x4xf32, (d0, d1) -> (d0 * 4 + d1)>

    // Create a sub-view of "base" memref '%0' with offset arguments '%c0',
    // dynamic sizes for each dimension, and stride arguments '%c1'.
    %1 = subview %0[%c0, %c0][%size0, %size1][%c1, %c1]
      : memref<64x4xf32, (d0, d1) -> (d0 * 4 + d1) > to
        memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + d1 + s0)>
    ```

    Example 2:

    ```mlir
    %0 = alloc() : memref<8x16x4xf32, (d0, d1, d1) -> (d0 * 64 + d1 * 4 + d2)>

    // Create a sub-view of "base" memref '%0' with dynamic offsets, sizes,
    // and strides.
    // Note that dynamic offsets are represented by the linearized dynamic
    // offset symbol 's0' in the subview memref layout map, and that the
    // dynamic strides operands, after being applied to the base memref
    // strides in each dimension, are represented in the view memref layout
    // map as symbols 's1', 's2' and 's3'.
    %1 = subview %0[%i, %j, %k][%size0, %size1, %size2][%x, %y, %z]
      : memref<8x16x4xf32, (d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)> to
        memref<?x?x?xf32,
          (d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + d1 * s2 + d2 * s3 + s0)>
    ```

    Example 3:

    ```mlir
    %0 = alloc() : memref<8x16x4xf32, (d0, d1, d1) -> (d0 * 64 + d1 * 4 + d2)>

    // Subview with constant offsets, sizes and strides.
    %1 = subview %0[0, 2, 0][4, 4, 4][64, 4, 1]
      : memref<8x16x4xf32, (d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)> to
        memref<4x4x4xf32, (d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2 + 8)>
    ```

    Example 4:

    ```mlir
    %0 = alloc(%arg0, %arg1) : memref<?x?xf32>

    // Subview with constant size, but dynamic offsets and
    // strides. The resulting memref has a static shape, but if the
    // base memref has an affine map to describe the layout, the result
    // memref also uses an affine map to describe the layout. The
    // strides of the result memref is computed as follows:
    //
    // Let #map1 represents the layout of the base memref, and #map2
    // represents the layout of the result memref. A #mapsubview can be
    // constructed to map an index from the result memref to the base
    // memref (note that the description below uses more convenient
    // naming for symbols, while in affine maps, symbols are
    // represented as unsigned numbers that identify that symbol in the
    // given affine map.
    //
    // #mapsubview = (d0, d1)[o0, o1, t0, t1] -> (d0 * t0 + o0, d1 * t1 + o1)
    //
    // where, o0, o1, ... are offsets, and t0, t1, ... are strides. Then,
    //
    // #map2 = #map1.compose(#mapsubview)
    //
    // If the layout map is represented as
    //
    // #map1 = (d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)
    //
    // then,
    //
    // #map2 = (d0, d1)[s0, s1, s2, o0, o1, t0, t1] ->
    //              (d0 * s1 * t0 + d1 * s2 * t1 + o0 * s1 + o1 * s2 + s0)
    //
    // Representing this canonically
    //
    // #map2 = (d0, d1)[r0, r1, r2] -> (d0 * r1 + d1 * r2 + r0)
    //
    // where, r0 = o0 * s1 + o1 * s2 + s0, r1 = s1 * t0, r2 = s2 * t1.
    %1 = subview %0[%i, %j][4, 4][%x, %y] :
      : memref<?x?xf32, (d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)> to
        memref<4x4xf32, (d0, d1)[r0, r1, r2] -> (d0 * r1 + d1 * r2 + r0)>

    // Note that the subview op does not guarantee that the result
    // memref is "inbounds" w.r.t to base memref. It is upto the client
    // to ensure that the subview is accessed in a manner that is
    // in-bounds.
    ```
    }
  }];

  let arguments = (ins
    AnyMemRef:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    I64ArrayAttr:$static_offsets,
    I64ArrayAttr:$static_sizes,
    I64ArrayAttr:$static_strides
  );
  let results = (outs AnyMemRef:$result);

  let builders = [
    // Build a SubViewOp with mized static and dynamic entries.
    OpBuilder<
      "OpBuilder &b, OperationState &result, Value source, "
      "ArrayRef<int64_t> staticOffsets, ArrayRef<int64_t> staticSizes,"
      "ArrayRef<int64_t> staticStrides, ValueRange offsets, ValueRange sizes, "
      "ValueRange strides, ArrayRef<NamedAttribute> attrs = {}">,
    // Build a SubViewOp with all dynamic entries.
    OpBuilder<
      "OpBuilder &b, OperationState &result, Value source, "
      "ValueRange offsets, ValueRange sizes, ValueRange strides, "
      "ArrayRef<NamedAttribute> attrs = {}">
  ];

  let extraClassDeclaration = [{
    /// Returns the type of the base memref operand.
    MemRefType getBaseMemRefType() {
      return source().getType().cast<MemRefType>();
    }

    /// The result of a subview is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

    /// Returns as integer value the number of offset operands.
    int64_t getNumOffsets() { return llvm::size(offsets()); }

    /// Returns as integer value the number of size operands.
    int64_t getNumSizes() { return llvm::size(sizes()); }

    /// Returns as integer value the number of stride operands.
    int64_t getNumStrides() { return llvm::size(strides()); }

    /// Returns the dynamic sizes for this subview operation if specified.
    operand_range getDynamicSizes() { return sizes(); }

    /// Returns in `staticStrides` the static value of the stride
    /// operands. Returns failure() if the static value of the stride
    /// operands could not be retrieved.
    LogicalResult getStaticStrides(SmallVectorImpl<int64_t> &staticStrides);

    /// Auxiliary range data structure and helper function that unpacks the
    /// offset, size and stride operands of the SubViewOp into a list of triples.
    /// Such a list of triple is sometimes more convenient to manipulate.
    struct Range {
      Value offset, size, stride;
    };
    /// Return the list of SubViewOp::Range (i.e. offset, size, stride). Each
    /// Range entry contains either the dynamic value or a ConstantIndexOp
    /// constructed with `b` at location `loc`.
    SmallVector<Range, 8> getOrCreateRanges(OpBuilder &b, Location loc);

    /// Return the offsets as Values. Each Value is either the dynamic
    /// value specified in the op or a ConstantIndexOp constructed
    /// with `b` at location `loc`
    SmallVector<Value, 4> getOrCreateOffsets(OpBuilder &b, Location loc);

    /// Return the sizes as Values. Each Value is either the dynamic
    /// value specified in the op or a ConstantIndexOp constructed
    /// with `b` at location `loc`
    SmallVector<Value, 4> getOrCreateSizes(OpBuilder &b, Location loc);

    /// Return the strides as Values. Each Value is either the dynamic
    /// value specified in the op or a ConstantIndexOp constructed with
    /// `b` at location `loc`
    SmallVector<Value, 4> getOrCreateStrides(OpBuilder &b, Location loc);

    /// A subview result type can be fully inferred from the source type and the
    /// static representation of offsets, sizes and strides. Special sentinels
    /// encode the dynamic case.
    static Type inferSubViewResultType(MemRefType sourceMemRefType,
                                       ArrayRef<int64_t> staticOffsets,
                                       ArrayRef<int64_t> staticSizes,
                                       ArrayRef<int64_t> staticStrides);

    /// Return the rank of the result MemRefType.
    unsigned getRank() { return getType().getRank(); }

    /// Return true if the offset `idx` is a static constant.
    bool isDynamicOffset(unsigned idx);
    /// Return true if the size `idx` is a static constant.
    bool isDynamicSize(unsigned idx);
    /// Return true if the stride `idx` is a static constant.
    bool isDynamicStride(unsigned idx);

    /// Assert the offset `idx` is a static constant and return its value.
    int64_t getStaticOffset(unsigned idx) {
      assert(!isDynamicOffset(idx) && "expected static offset");
      return
        static_offsets().cast<ArrayAttr>()[idx].cast<IntegerAttr>().getInt();
    }
    /// Assert the size `idx` is a static constant and return its value.
    int64_t getStaticSize(unsigned idx) {
      assert(!isDynamicSize(idx) && "expected static size");
      return static_sizes().cast<ArrayAttr>()[idx].cast<IntegerAttr>().getInt();
    }
    /// Assert the stride `idx` is a static constant and return its value.
    int64_t getStaticStride(unsigned idx) {
      assert(!isDynamicStride(idx) && "expected static stride");
      return
        static_strides().cast<ArrayAttr>()[idx].cast<IntegerAttr>().getInt();
    }

    /// Assert the offset `idx` is dynamic and return the position of the
    /// corresponding operand.
    unsigned getIndexOfDynamicOffset(unsigned idx);
    /// Assert the size `idx` is dynamic and return the position of the
    /// corresponding operand.
    unsigned getIndexOfDynamicSize(unsigned idx);
    /// Assert the stride `idx` is dynamic and return the position of the
    /// corresponding operand.
    unsigned getIndexOfDynamicStride(unsigned idx);

    /// Assert the offset `idx` is dynamic and return its value.
    Value getDynamicOffset(unsigned idx) {
      return getOperand(getIndexOfDynamicOffset(idx));
    }
    /// Assert the size `idx` is dynamic and return its value.
    Value getDynamicSize(unsigned idx) {
      return getOperand(getIndexOfDynamicSize(idx));
    }
    /// Assert the stride `idx` is dynamic and return its value.
    Value getDynamicStride(unsigned idx) {
      return getOperand(getIndexOfDynamicStride(idx));
    }

    static StringRef getStaticOffsetsAttrName() {
      return "static_offsets";
    }
    static StringRef getStaticSizesAttrName() {
      return "static_sizes";
    }
    static StringRef getStaticStridesAttrName() {
      return "static_strides";
    }
    static ArrayRef<StringRef> getSpecialAttrNames() {
      static SmallVector<StringRef, 4> names{
        getStaticOffsetsAttrName(),
        getStaticSizesAttrName(),
        getStaticStridesAttrName(),
        getOperandSegmentSizeAttr()};
      return names;
   }
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// TanhOp
//===----------------------------------------------------------------------===//

def TanhOp : FloatUnaryOp<"tanh"> {
  let summary = "hyperbolic tangent of the specified value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.tanh` ssa-use `:` type
    ```

    The `tanh` operation computes the hyperbolic tangent. It takes one operand
    and returns one result of the same type. This type may be a float scalar
    type, a vector whose element type is float, or a tensor of floats. It has
    no standard attributes.

    Example:

    ```mlir
    // Scalar hyperbolic tangent value.
    %a = tanh %b : f64

    // SIMD vector element-wise hyperbolic tangent value.
    %f = tanh %g : vector<4xf32>

    // Tensor element-wise hyperbolic tangent value.
    %x = tanh %y : tensor<4x?xf8>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// TensorCastOp
//===----------------------------------------------------------------------===//

def TensorCastOp : CastOp<"tensor_cast"> {
  let summary = "tensor cast operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `std.tensor_cast` ssa-use `:` type `to` type
    ```

    Convert a tensor from one type to an equivalent type without changing any
    data elements. The source and destination types must both be tensor types
    with the same element type. If both are ranked, then the rank should be the
    same and static dimensions should match. The operation is invalid if
    converting to a mismatching constant dimension.

    Example:

    ```mlir
    // Convert from unknown rank to rank 2 with unknown dimension sizes.
    %2 = "std.tensor_cast"(%1) : (tensor<*xf32>) -> tensor<?x?xf32>
    %2 = tensor_cast %1 : tensor<*xf32> to tensor<?x?xf32>

    // Convert to a type with more known dimensions.
    %3 = "std.tensor_cast"(%2) : (tensor<?x?xf32>) -> tensor<4x?xf32>

    // Discard static dimension and rank information.
    %4 = "std.tensor_cast"(%3) : (tensor<4x?xf32>) -> tensor<?x?xf32>
    %5 = "std.tensor_cast"(%4) : (tensor<?x?xf32>) -> tensor<*xf32>
    ```
  }];

  let arguments = (ins AnyTensor);
  let results = (outs AnyTensor);

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);

    /// The result of a tensor_cast is always a tensor.
    TensorType getType() { return getResult().getType().cast<TensorType>(); }
  }];
}

//===----------------------------------------------------------------------===//
// TensorLoadOp
//===----------------------------------------------------------------------===//

def TensorLoadOp : Std_Op<"tensor_load",
    [SameOperandsAndResultShape, SameOperandsAndResultElementType,
     TypesMatchWith<"result type matches tensor equivalent of 'memref'",
                    "memref", "result",
                    "getTensorTypeFromMemRefType($_self)">]> {
  let summary = "tensor load operation";
  let description = [{
    Create a tensor from a memref, making an independent copy of the element
    data. The result value is a tensor whose shape and element type match the
    memref operand.

    Example:

    ```mlir
    // Produces a value of tensor<4x?xf32> type.
    %12 = tensor_load %10 : memref<4x?xf32, #layout, memspace0>
    ```
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref);
  let results = (outs AnyTensor:$result);
  // TensorLoadOp is fully verified by traits.
  let verifier = ?;

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value memref", [{
      auto memrefType = memref.getType().cast<MemRefType>();
      auto resultType = RankedTensorType::get(memrefType.getShape(),
                                              memrefType.getElementType());
      result.addOperands(memref);
      result.addTypes(resultType);
  }]>];

  let extraClassDeclaration = [{
    /// The result of a tensor_load is always a tensor.
    TensorType getType() { return getResult().getType().cast<TensorType>(); }
  }];

  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// TensorStoreOp
//===----------------------------------------------------------------------===//

def TensorStoreOp : Std_Op<"tensor_store",
    [SameOperandsShape, SameOperandsElementType,
     TypesMatchWith<"type of 'value' matches tensor equivalent of 'memref'",
                    "memref", "tensor",
                    "getTensorTypeFromMemRefType($_self)">]> {
  let summary = "tensor store operation";
  let description = [{
    Stores the contents of a tensor into a memref. The first operand is a value
    of tensor type, the second operand is a value of memref type. The shapes and
    element types of these must match, and are specified by the memref type.

    Example:

    ```mlir
    %9 = dim %8, 1 : tensor<4x?xf32>
    %10 = alloc(%9) : memref<4x?xf32, #layout, memspace0>
    tensor_store %8, %10 : memref<4x?xf32, #layout, memspace0>
    ```
  }];

  let arguments = (ins AnyTensor:$tensor,
                       Arg<AnyMemRef, "the reference to store to",
                           [MemWrite]>:$memref);
  // TensorStoreOp is fully verified by traits.
  let verifier = ?;

  let assemblyFormat = "$tensor `,` $memref attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// TruncateIOp
//===----------------------------------------------------------------------===//

def TruncateIOp : Std_Op<"trunci", [NoSideEffect, SameOperandsAndResultShape]> {
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

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value value, Type destType", [{
      result.addOperands(value);
      result.addTypes(destType);
  }]>];

  let parser = [{
    return impl::parseCastOp(parser, result);
  }];
  let printer = [{
    return printStandardCastOp(this->getOperation(), p);
  }];
}

//===----------------------------------------------------------------------===//
// UnsignedDivIOp
//===----------------------------------------------------------------------===//

def UnsignedDivIOp : IntArithmeticOp<"divi_unsigned"> {
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

def UnsignedRemIOp : IntArithmeticOp<"remi_unsigned"> {
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

def UnsignedShiftRightOp : IntArithmeticOp<"shift_right_unsigned"> {
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
// ViewOp
//===----------------------------------------------------------------------===//

def ViewOp : Std_Op<"view", [
    DeclareOpInterfaceMethods<ViewLikeOpInterface>, NoSideEffect]> {
  let summary = "memref view operation";
  let description = [{
    The "view" operation extracts an N-D contiguous memref with empty layout map
    with arbitrary element type from a 1-D contiguous memref with empty layout
    map of i8 element  type. The ViewOp supports the following arguments:
    *) A single dynamic byte-shift operand must be specified which represents a
       a shift of the base 1-D memref pointer from which to create the resulting
       contiguous memref view with identity layout.
    *) A dynamic size operand that must be specified for each dynamic dimension
       in the resulting view memref type.

    The "view" operation gives a structured indexing form to a flat 1-D buffer.
    Unlike "subview" it can perform a type change. The type change behavior
    requires the op to have special semantics because, e.g. a byte shift of 3
    cannot be represented as an offset on f64.
    For now, a "view" op:
      1) Only takes a contiguous source memref with 0 offset and empty layout.
      2) Must specify a byte_shift operand (in the future, a special integer
      attribute may be added to support the folded case).
      3) Returns a contiguous memref with 0 offset and empty layout.

    Example:

    ```mlir
    // Allocate a flat 1D/i8 memref.
    %0 = alloc() : memref<2048xi8>

    // ViewOp with dynamic offset and static sizes.
    %1 = view %0[%offset_1024][] : memref<2048xi8> to memref<64x4xf32>

    // ViewOp with dynamic offset and two dynamic size.
    %2 = view %0[%offset_1024][%size0, %size1] :
      memref<2048xi8> to memref<?x4x?xf32>
    ```
  }];

  let arguments = (ins MemRefRankOf<[I8], [1]>:$source,
                   Index:$byte_shift,
                   Variadic<Index>:$sizes);
  let results = (outs AnyMemRef);

  let extraClassDeclaration = [{
    /// The result of a view is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

    /// Returns the dynamic sizes for this view operation. This is redundant
    /// with `sizes` but needed in template implementations. More specifically:
    /// ```
    /// template <typename AnyMemRefDefOp>
    /// bool isMemRefSizeValidSymbol(AnyMemRefDefOp memrefDefOp, unsigned index,
    ///                              Region *region)
    /// ```
    operand_range getDynamicSizes() {
      return {sizes().begin(), sizes().end()};
    }
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// XOrOp
//===----------------------------------------------------------------------===//

def XOrOp : IntArithmeticOp<"xor", [Commutative]> {
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

def ZeroExtendIOp : Std_Op<"zexti", [NoSideEffect, SameOperandsAndResultShape]> {
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

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value value, Type destType", [{
      result.addOperands(value);
      result.addTypes(destType);
  }]>];

  let parser = [{
    return impl::parseCastOp(parser, result);
  }];
  let printer = [{
    return printStandardCastOp(this->getOperation(), p);
  }];
}

#endif // STANDARD_OPS
