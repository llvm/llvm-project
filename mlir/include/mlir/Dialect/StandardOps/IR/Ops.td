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

def StandardOps_Dialect : Dialect {
  let name = "std";
  let cppNamespace = "::mlir";
  let dependentDialects = ["arith::ArithmeticDialect"];
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
  let arguments = (ins AnyType:$in);
  let results = (outs AnyType:$out);

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

  let hasCanonicalizeMethod = 1;
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
        %c1 = arith.constant 1.0 : f32
        %inc = arith.addf %c1, %current_value : f32
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

  let hasCanonicalizeMethod = 1;
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
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getType().getResults());
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
    StringRef getCallee() { return callee(); }
    StringAttr getCalleeAttr() { return calleeAttr().getAttr(); }
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
  let hasCanonicalizeMethod = 1;

  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` type($callee)";
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
    [ConstantLike, NoSideEffect,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
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
    // Complex constant
    %1 = constant [1.0 : f32, 1.0 : f32] : complex<f32>

    // Reference to function @myfn.
    %2 = constant @myfn : (tensor<16xf32>, f32) -> tensor<16xf32>

    // Equivalent generic forms
    %1 = "std.constant"() {value = [1.0 : f32, 1.0 : f32] : complex<f32>}
       : () -> complex<f32>
    %2 = "std.constant"() {value = @myfn}
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
    [{ build($_builder, $_state, value.getType(), value); }]>,
    OpBuilder<(ins "Attribute":$value, "Type":$type),
    [{ build($_builder, $_state, type, value); }]>,
  ];

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("value"); }

    /// Returns true if a constant operation can be built with the given value
    /// and result type.
    static bool isBuildableWith(Attribute value, Type type);
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MaxFOp
//===----------------------------------------------------------------------===//

def MaxFOp : FloatBinaryOp<"maxf"> {
  let summary = "floating-point maximum operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `maxf` ssa-use `,` ssa-use `:` type
    ```

    Returns the maximum of the two arguments, treating -0.0 as less than +0.0.
    If one of the arguments is NaN, then the result is also NaN.

    Example:

    ```mlir
    // Scalar floating-point maximum.
    %a = maxf %b, %c : f64
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MaxSIOp
//===----------------------------------------------------------------------===//

def MaxSIOp : IntBinaryOp<"maxsi"> {
  let summary = "signed integer maximum operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `maxsi` ssa-use `,` ssa-use `:` type
    ```

    Returns the larger of %a and %b comparing the values as signed integers.

    Example:

    ```mlir
    // Scalar signed integer maximum.
    %a = maxsi %b, %c : i64
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MaxUIOp
//===----------------------------------------------------------------------===//

def MaxUIOp : IntBinaryOp<"maxui"> {
  let summary = "unsigned integer maximum operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `maxui` ssa-use `,` ssa-use `:` type
    ```

    Returns the larger of %a and %b comparing the values as unsigned integers.

    Example:

    ```mlir
    // Scalar unsigned integer maximum.
    %a = maxui %b, %c : i64
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MinFOp
//===----------------------------------------------------------------------===//

def MinFOp : FloatBinaryOp<"minf"> {
  let summary = "floating-point minimum operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `minf` ssa-use `,` ssa-use `:` type
    ```

    Returns the minimum of the two arguments, treating -0.0 as less than +0.0.
    If one of the arguments is NaN, then the result is also NaN.

    Example:

    ```mlir
    // Scalar floating-point minimum.
    %a = minf %b, %c : f64
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MinSIOp
//===----------------------------------------------------------------------===//

def MinSIOp : IntBinaryOp<"minsi"> {
  let summary = "signed integer minimum operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `minsi` ssa-use `,` ssa-use `:` type
    ```

    Returns the smaller of %a and %b comparing the values as signed integers.

    Example:

    ```mlir
    // Scalar signed integer minimum.
    %a = minsi %b, %c : i64
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MinUIOp
//===----------------------------------------------------------------------===//

def MinUIOp : IntBinaryOp<"minui"> {
  let summary = "unsigned integer minimum operation";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `minui` ssa-use `,` ssa-use `:` type
    ```

    Returns the smaller of %a and %b comparing the values as unsigned integers.

    Example:

    ```mlir
    // Scalar unsigned integer minimum.
    %a = minui %b, %c : i64
    ```
  }];
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

  let hasCanonicalizer = 1;
  let hasFolder = 1;
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
    operand has to be of integer/index/float type. When the result is a tensor,
    it has to be statically shaped.

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

  let arguments = (ins AnyTypeOf<[AnySignlessInteger, Index, AnyFloat],
                                 "integer/index/float type">:$input);
  let results = (outs AnyTypeOf<[AnyVector, AnyStaticShapeTensor]>:$aggregate);

  let builders = [
    OpBuilder<(ins "Value":$element, "Type":$aggregateType),
    [{ build($_builder, $_state, aggregateType, element); }]>];

  let hasFolder = 1;

  let assemblyFormat = "$input attr-dict `:` type($aggregate)";
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

def SwitchOp : Std_Op<"switch",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
     NoSideEffect, Terminator]> {
  let summary = "switch operation";
  let description = [{
    The `switch` terminator operation represents a switch on a signless integer
    value. If the flag matches one of the specified cases, then the
    corresponding destination is jumped to. If the flag does not match any of
    the cases, the default destination is jumped to. The count and types of
    operands must align with the arguments in the corresponding target blocks.

    Example:

    ```mlir
    switch %flag : i32, [
      default: ^bb1(%a : i32),
      42: ^bb1(%b : i32),
      43: ^bb3(%c : i32)
    ]
    ```
  }];

  let arguments = (ins
    AnyInteger:$flag,
    Variadic<AnyType>:$defaultOperands,
    VariadicOfVariadic<AnyType, "case_operand_segments">:$caseOperands,
    OptionalAttr<AnyIntElementsAttr>:$case_values,
    I32ElementsAttr:$case_operand_segments
  );
  let successors = (successor
    AnySuccessor:$defaultDestination,
    VariadicSuccessor<AnySuccessor>:$caseDestinations
  );
  let builders = [
    OpBuilder<(ins "Value":$flag,
      "Block *":$defaultDestination,
      "ValueRange":$defaultOperands,
      CArg<"ArrayRef<APInt>", "{}">:$caseValues,
      CArg<"BlockRange", "{}">:$caseDestinations,
      CArg<"ArrayRef<ValueRange>", "{}">:$caseOperands)>,
    OpBuilder<(ins "Value":$flag,
      "Block *":$defaultDestination,
      "ValueRange":$defaultOperands,
      CArg<"ArrayRef<int32_t>", "{}">:$caseValues,
      CArg<"BlockRange", "{}">:$caseDestinations,
      CArg<"ArrayRef<ValueRange>", "{}">:$caseOperands)>,
    OpBuilder<(ins "Value":$flag,
      "Block *":$defaultDestination,
      "ValueRange":$defaultOperands,
      CArg<"DenseIntElementsAttr", "{}">:$caseValues,
      CArg<"BlockRange", "{}">:$caseDestinations,
      CArg<"ArrayRef<ValueRange>", "{}">:$caseOperands)>
  ];

  let assemblyFormat = [{
    $flag `:` type($flag) `,` `[` `\n`
      custom<SwitchOpCases>(ref(type($flag)),$defaultDestination,
                            $defaultOperands,
                            type($defaultOperands),
                            $case_values,
                            $caseDestinations,
                            $caseOperands,
                            type($caseOperands))
   `]`
    attr-dict
  }];

  let extraClassDeclaration = [{
    /// Return the operands for the case destination block at the given index.
    OperandRange getCaseOperands(unsigned index) {
      return caseOperands()[index];
    }

    /// Return a mutable range of operands for the case destination block at the
    /// given index.
    MutableOperandRange getCaseOperandsMutable(unsigned index) {
      return caseOperandsMutable()[index];
    }
  }];

  let hasCanonicalizer = 1;
}

#endif // STANDARD_OPS
