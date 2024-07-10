//===- ArithOps.td - Arith op definitions ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARITH_OPS
#define ARITH_OPS

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/EnumAttr.td"

// Base class for Arith dialect ops. Ops in this dialect have no memory
// effects and can be applied element-wise to vectors and tensors.
class Arith_Op<string mnemonic, list<Trait> traits = []> :
    Op<Arith_Dialect, mnemonic,
       traits #
       [DeclareOpInterfaceMethods<VectorUnrollOpInterface>, NoMemoryEffect] #
       ElementwiseMappable.traits>;

// Base class for integer and floating point arithmetic ops. All ops have one
// result, require operands and results to be of the same type, and can accept
// tensors or vectors of integers or floats.
class Arith_ArithOp<string mnemonic, list<Trait> traits = []> :
    Arith_Op<mnemonic, traits # [SameOperandsAndResultType]>;

// Base class for unary arithmetic operations.
class Arith_UnaryOp<string mnemonic, list<Trait> traits = []> :
    Arith_ArithOp<mnemonic, traits # [Pure]> {
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

// Base class for binary arithmetic operations.
class Arith_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Arith_ArithOp<mnemonic, traits> {
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// Base class for integer binary operations.
class Arith_IntBinaryOp<string mnemonic, list<Trait> traits = []> :
    Arith_BinaryOp<mnemonic, traits #
      [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]>,
    Arguments<(ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs)>,
    Results<(outs SignlessIntegerLike:$result)>;

// Base class for integer binary operations without undefined behavior.
class Arith_TotalIntBinaryOp<string mnemonic, list<Trait> traits = []> :
    Arith_IntBinaryOp<mnemonic, traits # [Pure]>;

// Base class for floating point unary operations.
class Arith_FloatUnaryOp<string mnemonic, list<Trait> traits = []> :
    Arith_UnaryOp<mnemonic,
      !listconcat([DeclareOpInterfaceMethods<ArithFastMathInterface>],
                  traits)>,
    Arguments<(ins FloatLike:$operand,
      DefaultValuedAttr<
        Arith_FastMathAttr, "::mlir::arith::FastMathFlags::none">:$fastmath)>,
    Results<(outs FloatLike:$result)> {
  let assemblyFormat = [{ $operand (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

// Base class for floating point binary operations.
class Arith_FloatBinaryOp<string mnemonic, list<Trait> traits = []> :
    Arith_BinaryOp<mnemonic,
      !listconcat([Pure, DeclareOpInterfaceMethods<ArithFastMathInterface>],
                  traits)>,
    Arguments<(ins FloatLike:$lhs, FloatLike:$rhs,
      DefaultValuedAttr<
        Arith_FastMathAttr, "::mlir::arith::FastMathFlags::none">:$fastmath)>,
    Results<(outs FloatLike:$result)> {
  let assemblyFormat = [{ $lhs `,` $rhs (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

// Checks that tensor input and outputs have identical shapes. This is stricker
// than the verification done in `SameOperandsAndResultShape` that allows for
// tensor dimensions to be 'compatible' (e.g., dynamic dimensions being
// compatible with static ones).
def SameInputOutputTensorDims : PredOpTrait<
    "input and output have the same tensor dimensions",
    AllMatchSameOperatorPred<["in", "out"],
      "(::llvm::isa<::mlir::TensorType>($_self.getType()) ?"
      " ::llvm::cast<::mlir::TensorType>($_self.getType()).getShape() :"
      " ::llvm::ArrayRef<int64_t>{})">>;

// Base class for arithmetic cast operations. Requires a single operand and
// result. If either is a shaped type, then the other must be of the same
// shape.  In the case of tensor types, this also includes the corresponding
// operand/result dimensions being equal.
class Arith_CastOp<string mnemonic, TypeConstraint From, TypeConstraint To,
                   list<Trait> traits = []> :
    Arith_Op<mnemonic, traits # [Pure, SameOperandsAndResultShape,
      SameInputOutputTensorDims, DeclareOpInterfaceMethods<CastOpInterface>]>,
    Arguments<(ins From:$in)>,
    Results<(outs To:$out)> {
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";
}

// Casts do not accept indices. Type constraint for signless-integer-like types
// excluding indices: signless integers, vectors or tensors thereof.
def SignlessFixedWidthIntegerLike : TypeConstraint<Or<[
        AnySignlessInteger.predicate,
        VectorOfAnyRankOf<[AnySignlessInteger]>.predicate,
        TensorOf<[AnySignlessInteger]>.predicate]>,
    "signless-fixed-width-integer-like">;

// Cast from an integer type to another integer type.
class Arith_IToICastOp<string mnemonic, list<Trait> traits = []> :
    Arith_CastOp<mnemonic, SignlessFixedWidthIntegerLike,
                           SignlessFixedWidthIntegerLike,
                           traits #
                           [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]>;
// Cast from an integer type to a floating point type.
class Arith_IToFCastOp<string mnemonic, list<Trait> traits = []> :
    Arith_CastOp<mnemonic, SignlessFixedWidthIntegerLike, FloatLike, traits>;
// Cast from a floating point type to an integer type.
class Arith_FToICastOp<string mnemonic, list<Trait> traits = []> :
    Arith_CastOp<mnemonic, FloatLike, SignlessFixedWidthIntegerLike, traits>;
// Cast from a floating point type to another floating point type.
class Arith_FToFCastOp<string mnemonic, list<Trait> traits = []> :
    Arith_CastOp<mnemonic, FloatLike, FloatLike, traits>;

// Base class for compare operations. Requires two operands of the same type
// and returns a single `BoolLike` result. If the operand type is a vector or
// tensor, then the result will be one of `i1` of the same shape.
class Arith_CompareOp<string mnemonic, list<Trait> traits = []> :
    Arith_Op<mnemonic, traits # [Pure, SameTypeOperands, TypesMatchWith<
    "result type has i1 element type and same shape as operands",
    "lhs", "result", "::getI1SameShape($_self)">]> {
  let results = (outs BoolLike:$result);

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

// Just like `Arith_CompareOp` but also admits 0-D vectors. Introduced
// temporarily to allow gradual transition to 0-D vectors.
class Arith_CompareOpOfAnyRank<string mnemonic, list<Trait> traits = []> :
    Arith_CompareOp<mnemonic, traits> {
  let results = (outs BoolLikeOfAnyRank:$result);
}

class Arith_IntBinaryOpWithOverflowFlags<string mnemonic, list<Trait> traits = []> :
    Arith_BinaryOp<mnemonic, traits #
      [Pure, DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>,
       DeclareOpInterfaceMethods<ArithIntegerOverflowFlagsInterface>]>,
    Arguments<(ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs,
      DefaultValuedAttr<
        Arith_IntegerOverflowAttr,
        "::mlir::arith::IntegerOverflowFlags::none">:$overflowFlags)>,
    Results<(outs SignlessIntegerLike:$result)> {

  let assemblyFormat = [{ $lhs `,` $rhs (`overflow` `` $overflowFlags^)?
                          attr-dict `:` type($result) }];
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def Arith_ConstantOp : Op<Arith_Dialect, "constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     AllTypesMatch<["value", "result"]>,
     DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let summary = "integer or floating point constant";
  let description = [{
    The `constant` operation produces an SSA value equal to some integer or
    floating-point constant specified by an attribute. This is the way MLIR
    forms simple integer and floating point constants.

    Example:

    ```
    // Integer constant
    %1 = arith.constant 42 : i32

    // Equivalent generic form
    %1 = "arith.constant"() {value = 42 : i32} : () -> i32
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  // TODO: Disallow arith.constant to return anything other than a signless
  // integer or float like. Downstream users of Arith should only be
  // working with signless integers, floats, or vectors/tensors thereof.
  // However, it is necessary to allow arith.constant to return vectors/tensors
  // of strings and signed/unsigned integers (for now) as an artefact of
  // splitting the Standard dialect.
  let results = (outs /*SignlessIntegerOrFloatLike*/AnyType:$result);

  let extraClassDeclaration = [{
    /// Whether the constant op can be constructed with a particular value and
    /// type.
    static bool isBuildableWith(Attribute value, Type type);

    /// Build the constant op with `value` and `type` if possible, otherwise
    /// returns null.
    static ConstantOp materialize(OpBuilder &builder, Attribute value,
                                  Type type, Location loc);
  }];

  let hasFolder = 1;
  let assemblyFormat = "attr-dict $value";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

def Arith_AddIOp : Arith_IntBinaryOpWithOverflowFlags<"addi", [Commutative]> {
  let summary = "integer addition operation";
  let description = [{
    Performs N-bit addition on the operands. The operands are interpreted as 
    unsigned bitvectors. The result is represented by a bitvector containing the 
    mathematical value of the addition modulo 2^n, where `n` is the bitwidth. 
    Because `arith` integers use a two's complement representation, this operation 
    is applicable on both signed and unsigned integer operands.

    The `addi` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar type, 
    a vector whose element type is integer, or a tensor of integers.

    This op supports `nuw`/`nsw` overflow flags which stands stand for
    "No Unsigned Wrap" and "No Signed Wrap", respectively. If the `nuw` and/or
    `nsw` flags are present, and an unsigned/signed overflow occurs
    (respectively), the result is poison.

    Example:

    ```mlir
    // Scalar addition.
    %a = arith.addi %b, %c : i64

    // Scalar addition with overflow flags.
    %a = arith.addi %b, %c overflow<nsw, nuw> : i64

    // SIMD vector element-wise addition.
    %f = arith.addi %g, %h : vector<4xi32>

    // Tensor element-wise addition.
    %x = arith.addi %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// AddUIExtendedOp
//===----------------------------------------------------------------------===//

def Arith_AddUIExtendedOp : Arith_Op<"addui_extended", [Pure, Commutative,
    AllTypesMatch<["lhs", "rhs", "sum"]>]> {
  let summary = [{
    extended unsigned integer addition operation returning sum and overflow bit
  }];

  let description = [{
    Performs (N+1)-bit addition on zero-extended operands. Returns two results:
    the N-bit sum (same type as both operands), and the overflow bit
    (boolean-like), where `1` indicates unsigned addition overflow, while `0`
    indicates no overflow.

    Example:

    ```mlir
    // Scalar addition.
    %sum, %overflow = arith.addui_extended %b, %c : i64, i1

    // Vector element-wise addition.
    %d:2 = arith.addui_extended %e, %f : vector<4xi32>, vector<4xi1>

    // Tensor element-wise addition.
    %x:2 = arith.addui_extended %y, %z : tensor<4x?xi8>, tensor<4x?xi1>
    ```
  }];

  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
  let results = (outs SignlessIntegerLike:$sum, BoolLike:$overflow);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($sum) `,` type($overflow)
  }];

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      build($_builder, $_state, lhs.getType(), ::getI1SameShape(lhs.getType()),
            lhs, rhs);
    }]>
  ];

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    std::optional<SmallVector<int64_t, 4>> getShapeForUnroll();
  }];
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

def Arith_SubIOp : Arith_IntBinaryOpWithOverflowFlags<"subi"> {
  let summary = [{
    Integer subtraction operation.
  }];
  let description = [{
    Performs N-bit subtraction on the operands. The operands are interpreted as unsigned
    bitvectors. The result is represented by a bitvector containing the mathematical
    value of the subtraction modulo 2^n, where `n` is the bitwidth. Because `arith`
    integers use a two's complement representation, this operation is applicable on
    both signed and unsigned integer operands.

    The `subi` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar type,
    a vector whose element type is integer, or a tensor of integers.

    This op supports `nuw`/`nsw` overflow flags which stands stand for
    "No Unsigned Wrap" and "No Signed Wrap", respectively. If the `nuw` and/or
    `nsw` flags are present, and an unsigned/signed overflow occurs
    (respectively), the result is poison.

    Example:

    ```mlir
    // Scalar subtraction.
    %a = arith.subi %b, %c : i64

    // Scalar subtraction with overflow flags.
    %a = arith.subi %b, %c overflow<nsw, nuw> : i64

    // SIMD vector element-wise subtraction.
    %f = arith.subi %g, %h : vector<4xi32>

    // Tensor element-wise subtraction.
    %x = arith.subi %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

def Arith_MulIOp : Arith_IntBinaryOpWithOverflowFlags<"muli",
  [Commutative, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]
> {
  let summary = [{
    Integer multiplication operation.
  }];
  let description = [{
    Performs N-bit multiplication on the operands. The operands are interpreted as
    unsigned bitvectors. The result is represented by a bitvector containing the
    mathematical value of the multiplication modulo 2^n, where `n` is the bitwidth.
    Because `arith` integers use a two's complement representation, this operation is
    applicable on both signed and unsigned integer operands.

    The `muli` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar type,
    a vector whose element type is integer, or a tensor of integers.

    This op supports `nuw`/`nsw` overflow flags which stands stand for
    "No Unsigned Wrap" and "No Signed Wrap", respectively. If the `nuw` and/or
    `nsw` flags are present, and an unsigned/signed overflow occurs
    (respectively), the result is poison.

    Example:

    ```mlir
    // Scalar multiplication.
    %a = arith.muli %b, %c : i64

    // Scalar multiplication with overflow flags.
    %a = arith.muli %b, %c overflow<nsw, nuw> : i64

    // SIMD vector element-wise multiplication.
    %f = arith.muli %g, %h : vector<4xi32>

    // Tensor element-wise multiplication.
    %x = arith.muli %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// MulSIExtendedOp
//===----------------------------------------------------------------------===//

def Arith_MulSIExtendedOp : Arith_Op<"mulsi_extended", [Pure, Commutative,
    AllTypesMatch<["lhs", "rhs", "low", "high"]>]> {
  let summary = [{
    extended signed integer multiplication operation
  }];

  let description = [{
    Performs (2*N)-bit multiplication on sign-extended operands. Returns two
    N-bit results: the low and the high halves of the product. The low half has
    the same value as the result of regular multiplication `arith.muli` with
    the same operands.

    Example:

    ```mlir
    // Scalar multiplication.
    %low, %high = arith.mulsi_extended %a, %b : i32

    // Vector element-wise multiplication.
    %c:2 = arith.mulsi_extended %d, %e : vector<4xi32>

    // Tensor element-wise multiplication.
    %x:2 = arith.mulsi_extended %y, %z : tensor<4x?xi8>
    ```
  }];

  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
  let results = (outs SignlessIntegerLike:$low, SignlessIntegerLike:$high);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    std::optional<SmallVector<int64_t, 4>> getShapeForUnroll();
  }];
}

//===----------------------------------------------------------------------===//
// MulUIExtendedOp
//===----------------------------------------------------------------------===//

def Arith_MulUIExtendedOp : Arith_Op<"mului_extended", [Pure, Commutative,
    AllTypesMatch<["lhs", "rhs", "low", "high"]>]> {
  let summary = [{
    extended unsigned integer multiplication operation
  }];

  let description = [{
    Performs (2*N)-bit multiplication on zero-extended operands. Returns two
    N-bit results: the low and the high halves of the product. The low half has
    the same value as the result of regular multiplication `arith.muli` with
    the same operands.

    Example:

    ```mlir
    // Scalar multiplication.
    %low, %high = arith.mului_extended %a, %b : i32

    // Vector element-wise multiplication.
    %c:2 = arith.mului_extended %d, %e : vector<4xi32>

    // Tensor element-wise multiplication.
    %x:2 = arith.mului_extended %y, %z : tensor<4x?xi8>
    ```
  }];

  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
  let results = (outs SignlessIntegerLike:$low, SignlessIntegerLike:$high);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    std::optional<SmallVector<int64_t, 4>> getShapeForUnroll();
  }];
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

def Arith_DivUIOp : Arith_IntBinaryOp<"divui", [ConditionallySpeculatable]> {
  let summary = "unsigned integer division operation";
  let description = [{
    Unsigned integer division. Rounds towards zero. Treats the leading bit as
    the most significant, i.e. for `i16` given two's complement representation,
    `6 / -2 = 6 / (2^16 - 2) = 0`.

    Division by zero is undefined behavior. When applied to `vector` and 
    `tensor` values, the behavior is undefined if _any_ elements are divided by 
    zero.

    Example:

    ```mlir
    // Scalar unsigned integer division.
    %a = arith.divui %b, %c : i64

    // SIMD vector element-wise division.
    %f = arith.divui %g, %h : vector<4xi32>

    // Tensor element-wise integer division.
    %x = arith.divui %y, %z : tensor<4x?xi8>
    ```
  }];

  let extraClassDeclaration = [{
    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability getSpeculatability();
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

def Arith_DivSIOp : Arith_IntBinaryOp<"divsi", [ConditionallySpeculatable]> {
  let summary = "signed integer division operation";
  let description = [{
    Signed integer division. Rounds towards zero. Treats the leading bit as
    sign, i.e. `6 / -2 = -3`.

    Divison by zero, or signed division overflow (minimum value divided by -1) 
    is undefined behavior. When applied to `vector` and `tensor` values, the 
    behavior is undefined if _any_ of its elements are divided by zero or has a 
    signed division overflow.

    Example:

    ```mlir
    // Scalar signed integer division.
    %a = arith.divsi %b, %c : i64

    // SIMD vector element-wise division.
    %f = arith.divsi %g, %h : vector<4xi32>

    // Tensor element-wise integer division.
    %x = arith.divsi %y, %z : tensor<4x?xi8>
    ```
  }];

  let extraClassDeclaration = [{
    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability getSpeculatability();
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CeilDivUIOp
//===----------------------------------------------------------------------===//

def Arith_CeilDivUIOp : Arith_IntBinaryOp<"ceildivui",
                                          [ConditionallySpeculatable]> {
  let summary = "unsigned ceil integer division operation";
  let description = [{
    Unsigned integer division. Rounds towards positive infinity. Treats the
    leading bit as the most significant, i.e. for `i16` given two's complement
    representation, `6 / -2 = 6 / (2^16 - 2) = 1`. 

    Division by zero is undefined behavior. When applied to `vector` and 
    `tensor` values, the behavior is undefined if _any_ elements are divided by 
    zero.

    Example:

    ```mlir
    // Scalar unsigned integer division.
    %a = arith.ceildivui %b, %c : i64
    ```
  }];

  let extraClassDeclaration = [{
    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability getSpeculatability();
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CeilDivSIOp
//===----------------------------------------------------------------------===//

def Arith_CeilDivSIOp : Arith_IntBinaryOp<"ceildivsi",
                                          [ConditionallySpeculatable]> {
  let summary = "signed ceil integer division operation";
  let description = [{
    Signed integer division. Rounds towards positive infinity, i.e. `7 / -2 = -3`.

    Divison by zero, or signed division overflow (minimum value divided by -1) 
    is undefined behavior. When applied to `vector` and `tensor` values, the 
    behavior is undefined if _any_ of its elements are divided by zero or has a 
    signed division overflow.

    Example:

    ```mlir
    // Scalar signed integer division.
    %a = arith.ceildivsi %b, %c : i64
    ```
  }];

  let extraClassDeclaration = [{
    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability getSpeculatability();
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FloorDivSIOp
//===----------------------------------------------------------------------===//

def Arith_FloorDivSIOp : Arith_TotalIntBinaryOp<"floordivsi"> {
  let summary = "signed floor integer division operation";
  let description = [{
    Signed integer division. Rounds towards negative infinity, i.e. `5 / -2 = -3`.

    Divison by zero, or signed division overflow (minimum value divided by -1) 
    is undefined behavior. When applied to `vector` and `tensor` values, the 
    behavior is undefined if _any_ of its elements are divided by zero or has a 
    signed division overflow.

    Example:

    ```mlir
    // Scalar signed integer division.
    %a = arith.floordivsi %b, %c : i64

    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// RemUIOp
//===----------------------------------------------------------------------===//

def Arith_RemUIOp : Arith_TotalIntBinaryOp<"remui"> {
  let summary = "unsigned integer division remainder operation";
  let description = [{
    Unsigned integer division remainder. Treats the leading bit as the most
    significant, i.e. for `i16`, `6 % -2 = 6 % (2^16 - 2) = 6`.

    Division by zero is undefined behavior. When applied to `vector` and 
    `tensor` values, the behavior is undefined if _any_ elements are divided by 
    zero.

    Example:

    ```mlir
    // Scalar unsigned integer division remainder.
    %a = arith.remui %b, %c : i64

    // SIMD vector element-wise division remainder.
    %f = arith.remui %g, %h : vector<4xi32>

    // Tensor element-wise integer division remainder.
    %x = arith.remui %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// RemSIOp
//===----------------------------------------------------------------------===//

def Arith_RemSIOp : Arith_TotalIntBinaryOp<"remsi"> {
  let summary = "signed integer division remainder operation";
  let description = [{
    Signed integer division remainder. Treats the leading bit as sign, i.e. `6 %
    -2 = 0`.

    Division by zero is undefined behavior. When applied to `vector` and 
    `tensor` values, the behavior is undefined if _any_ elements are divided by 
    zero.

    Example:

    ```mlir
    // Scalar signed integer division remainder.
    %a = arith.remsi %b, %c : i64

    // SIMD vector element-wise division remainder.
    %f = arith.remsi %g, %h : vector<4xi32>

    // Tensor element-wise integer division remainder.
    %x = arith.remsi %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

def Arith_AndIOp : Arith_TotalIntBinaryOp<"andi", [Commutative, Idempotent]> {
  let summary = "integer binary and";
  let description = [{
    The `andi` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar
    type, a vector whose element type is integer, or a tensor of integers. It
    has no standard attributes.

    Example:

    ```mlir
    // Scalar integer bitwise and.
    %a = arith.andi %b, %c : i64

    // SIMD vector element-wise bitwise integer and.
    %f = arith.andi %g, %h : vector<4xi32>

    // Tensor element-wise bitwise integer and.
    %x = arith.andi %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

def Arith_OrIOp : Arith_TotalIntBinaryOp<"ori", [Commutative, Idempotent]> {
  let summary = "integer binary or";
  let description = [{
    The `ori` operation takes two operands and returns one result, each of these
    is required to be the same type. This type may be an integer scalar type, a
    vector whose element type is integer, or a tensor of integers. It has no
    standard attributes.

    Example:

    ```mlir
    // Scalar integer bitwise or.
    %a = arith.ori %b, %c : i64

    // SIMD vector element-wise bitwise integer or.
    %f = arith.ori %g, %h : vector<4xi32>

    // Tensor element-wise bitwise integer or.
    %x = arith.ori %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

def Arith_XOrIOp : Arith_TotalIntBinaryOp<"xori", [Commutative]> {
  let summary = "integer binary xor";
  let description = [{
    The `xori` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar
    type, a vector whose element type is integer, or a tensor of integers. It
    has no standard attributes.

    Example:

    ```mlir
    // Scalar integer bitwise xor.
    %a = arith.xori %b, %c : i64

    // SIMD vector element-wise bitwise integer xor.
    %f = arith.xori %g, %h : vector<4xi32>

    // Tensor element-wise bitwise integer xor.
    %x = arith.xori %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// ShLIOp
//===----------------------------------------------------------------------===//

def Arith_ShLIOp : Arith_IntBinaryOpWithOverflowFlags<"shli"> {
  let summary = "integer left-shift";
  let description = [{
    The `shli` operation shifts the integer value of the first operand to the left 
    by the integer value of the second operand. The second operand is interpreted as 
    unsigned. The low order bits are filled with zeros. If the value of the second 
    operand is greater or equal than the bitwidth of the first operand, then the
    operation returns poison.

    This op supports `nuw`/`nsw` overflow flags which stands stand for
    "No Unsigned Wrap" and "No Signed Wrap", respectively. If the `nuw` and/or
    `nsw` flags are present, and an unsigned/signed overflow occurs
    (respectively), the result is poison.

    Example:

    ```mlir
    %1 = arith.constant 5 : i8  // %1 is 0b00000101
    %2 = arith.constant 3 : i8
    %3 = arith.shli %1, %2 : i8 // %3 is 0b00101000
    %4 = arith.shli %1, %2 overflow<nsw, nuw> : i8  
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ShRUIOp
//===----------------------------------------------------------------------===//

def Arith_ShRUIOp : Arith_TotalIntBinaryOp<"shrui"> {
  let summary = "unsigned integer right-shift";
  let description = [{
    The `shrui` operation shifts an integer value of the first operand to the right 
    by the value of the second operand. The first operand is interpreted as unsigned,
    and the second operand is interpreted as unsigned. The high order bits are always 
    filled with zeros. If the value of the second operand is greater or equal than the
    bitwidth of the first operand, then the operation returns poison.

    Example:

    ```mlir
    %1 = arith.constant 160 : i8               // %1 is 0b10100000
    %2 = arith.constant 3 : i8
    %3 = arith.shrui %1, %2 : (i8, i8) -> i8   // %3 is 0b00010100
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ShRSIOp
//===----------------------------------------------------------------------===//

def Arith_ShRSIOp : Arith_TotalIntBinaryOp<"shrsi"> {
  let summary = "signed integer right-shift";
  let description = [{
    The `shrsi` operation shifts an integer value of the first operand to the right 
    by the value of the second operand. The first operand is interpreted as signed, 
    and the second operand is interpreter as unsigned. The high order bits in the 
    output are filled with copies of the most-significant bit of the shifted value 
    (which means that the sign of the value is preserved). If the value of the second 
    operand is greater or equal than bitwidth of the first operand, then the operation
    returns poison.

    Example:

    ```mlir
    %1 = arith.constant 160 : i8               // %1 is 0b10100000
    %2 = arith.constant 3 : i8
    %3 = arith.shrsi %1, %2 : (i8, i8) -> i8   // %3 is 0b11110100
    %4 = arith.constant 96 : i8                   // %4 is 0b01100000
    %5 = arith.shrsi %4, %2 : (i8, i8) -> i8   // %5 is 0b00001100
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// NegFOp
//===----------------------------------------------------------------------===//

def Arith_NegFOp : Arith_FloatUnaryOp<"negf"> {
  let summary = "floating point negation";
  let description = [{
    The `negf` operation computes the negation of a given value. It takes one
    operand and returns one result of the same type. This type may be a float
    scalar type, a vector whose element type is float, or a tensor of floats.
    It has no standard attributes.

    Example:

    ```mlir
    // Scalar negation value.
    %a = arith.negf %b : f64

    // SIMD vector element-wise negation value.
    %f = arith.negf %g : vector<4xf32>

    // Tensor element-wise negation value.
    %x = arith.negf %y : tensor<4x?xf8>
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

def Arith_AddFOp : Arith_FloatBinaryOp<"addf", [Commutative]> {
  let summary = "floating point addition operation";
  let description = [{
    The `addf` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be a floating point
    scalar type, a vector whose element type is a floating point type, or a
    floating point tensor.

    Example:

    ```mlir
    // Scalar addition.
    %a = arith.addf %b, %c : f64

    // SIMD vector addition, e.g. for Intel SSE.
    %f = arith.addf %g, %h : vector<4xf32>

    // Tensor addition.
    %x = arith.addf %y, %z : tensor<4x?xbf16>
    ```

    TODO: In the distant future, this will accept optional attributes for fast
    math, contraction, rounding mode, and other controls.
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

def Arith_SubFOp : Arith_FloatBinaryOp<"subf"> {
  let summary = "floating point subtraction operation";
  let description = [{
    The `subf` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be a floating point
    scalar type, a vector whose element type is a floating point type, or a
    floating point tensor.

    Example:

    ```mlir
    // Scalar subtraction.
    %a = arith.subf %b, %c : f64

    // SIMD vector subtraction, e.g. for Intel SSE.
    %f = arith.subf %g, %h : vector<4xf32>

    // Tensor subtraction.
    %x = arith.subf %y, %z : tensor<4x?xbf16>
    ```

    TODO: In the distant future, this will accept optional attributes for fast
    math, contraction, rounding mode, and other controls.
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MaximumFOp
//===----------------------------------------------------------------------===//

def Arith_MaximumFOp : Arith_FloatBinaryOp<"maximumf", [Commutative]> {
  let summary = "floating-point maximum operation";
  let description = [{
    Returns the maximum of the two arguments, treating -0.0 as less than +0.0.
    If one of the arguments is NaN, then the result is also NaN.

    Example:

    ```mlir
    // Scalar floating-point maximum.
    %a = arith.maximumf %b, %c : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MaxNumFOp
//===----------------------------------------------------------------------===//

def Arith_MaxNumFOp : Arith_FloatBinaryOp<"maxnumf", [Commutative]> {
  let summary = "floating-point maximum operation";
  let description = [{
    Returns the maximum of the two arguments.
    If the arguments are -0.0 and +0.0, then the result is either of them.
    If one of the arguments is NaN, then the result is the other argument.

    Example:

    ```mlir
    // Scalar floating-point maximum.
    %a = arith.maxnumf %b, %c : f64
    ```
  }];
  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// MaxSIOp
//===----------------------------------------------------------------------===//

def Arith_MaxSIOp : Arith_TotalIntBinaryOp<"maxsi", [Commutative]> {
  let summary = "signed integer maximum operation";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MaxUIOp
//===----------------------------------------------------------------------===//

def Arith_MaxUIOp : Arith_TotalIntBinaryOp<"maxui", [Commutative]> {
  let summary = "unsigned integer maximum operation";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MinimumFOp
//===----------------------------------------------------------------------===//

def Arith_MinimumFOp : Arith_FloatBinaryOp<"minimumf", [Commutative]> {
  let summary = "floating-point minimum operation";
  let description = [{
    Returns the minimum of the two arguments, treating -0.0 as less than +0.0.
    If one of the arguments is NaN, then the result is also NaN.

    Example:

    ```mlir
    // Scalar floating-point minimum.
    %a = arith.minimumf %b, %c : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MinNumFOp
//===----------------------------------------------------------------------===//

def Arith_MinNumFOp : Arith_FloatBinaryOp<"minnumf", [Commutative]> {
  let summary = "floating-point minimum operation";
  let description = [{
    Returns the minimum of the two arguments.
    If the arguments are -0.0 and +0.0, then the result is either of them.
    If one of the arguments is NaN, then the result is the other argument.

    Example:

    ```mlir
    // Scalar floating-point minimum.
    %a = arith.minnumf %b, %c : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MinSIOp
//===----------------------------------------------------------------------===//

def Arith_MinSIOp : Arith_TotalIntBinaryOp<"minsi", [Commutative]> {
  let summary = "signed integer minimum operation";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MinUIOp
//===----------------------------------------------------------------------===//

def Arith_MinUIOp : Arith_TotalIntBinaryOp<"minui", [Commutative]> {
  let summary = "unsigned integer minimum operation";
  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

def Arith_MulFOp : Arith_FloatBinaryOp<"mulf", [Commutative]> {
  let summary = "floating point multiplication operation";
  let description = [{
    The `mulf` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be a floating point
    scalar type, a vector whose element type is a floating point type, or a
    floating point tensor.

    Example:

    ```mlir
    // Scalar multiplication.
    %a = arith.mulf %b, %c : f64

    // SIMD pointwise vector multiplication, e.g. for Intel SSE.
    %f = arith.mulf %g, %h : vector<4xf32>

    // Tensor pointwise multiplication.
    %x = arith.mulf %y, %z : tensor<4x?xbf16>
    ```

    TODO: In the distant future, this will accept optional attributes for fast
    math, contraction, rounding mode, and other controls.
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// DivFOp
//===----------------------------------------------------------------------===//

def Arith_DivFOp : Arith_FloatBinaryOp<"divf"> {
  let summary = "floating point division operation";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// RemFOp
//===----------------------------------------------------------------------===//

def Arith_RemFOp : Arith_FloatBinaryOp<"remf"> {
  let summary = "floating point division remainder operation";
  let description = [{
    Returns the floating point division remainder.
    The remainder has the same sign as the dividend (lhs operand).
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

def Arith_ExtUIOp : Arith_IToICastOp<"extui"> {
  let summary = "integer zero extension operation";
  let description = [{
    The integer zero extension operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be larger than the input bit-width (N > M).
    The top-most (N - M) bits of the output are filled with zeros.

    Example:

    ```mlir
      %1 = arith.constant 5 : i3      // %1 is 0b101
      %2 = arith.extui %1 : i3 to i6  // %2 is 0b000101
      %3 = arith.constant 2 : i3      // %3 is 0b010
      %4 = arith.extui %3 : i3 to i6  // %4 is 0b000010

      %5 = arith.extui %0 : vector<2 x i32> to vector<2 x i64>
    ```
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

def Arith_ExtSIOp : Arith_IToICastOp<"extsi"> {
  let summary = "integer sign extension operation";

  let description = [{
    The integer sign extension operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be larger than the input bit-width (N > M).
    The top-most (N - M) bits of the output are filled with copies
    of the most-significant bit of the input.

    Example:

    ```mlir
    %1 = arith.constant 5 : i3      // %1 is 0b101
    %2 = arith.extsi %1 : i3 to i6  // %2 is 0b111101
    %3 = arith.constant 2 : i3      // %3 is 0b010
    %4 = arith.extsi %3 : i3 to i6  // %4 is 0b000010

    %5 = arith.extsi %0 : vector<2 x i32> to vector<2 x i64>
    ```
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExtFOp
//===----------------------------------------------------------------------===//

def Arith_ExtFOp : Arith_FToFCastOp<"extf", [DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let summary = "cast from floating-point to wider floating-point";
  let description = [{
    Cast a floating-point value to a larger floating-point-typed value.
    The destination type must to be strictly wider than the source type.
    When operating on vectors, casts elementwise.
  }];
  let hasVerifier = 1;
  let hasFolder = 1;

  let arguments = (ins FloatLike:$in,
                       OptionalAttr<Arith_FastMathAttr>:$fastmath);
  let results = (outs FloatLike:$out);

  let assemblyFormat = [{ $in (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($in) `to` type($out) }];
}

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

def Arith_TruncIOp : Arith_IToICastOp<"trunci"> {
  let summary = "integer truncation operation";
  let description = [{
    The integer truncation operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be smaller than the input bit-width (N < M).
    The top-most (N - M) bits of the input are discarded.

    Example:

    ```mlir
      %1 = arith.constant 21 : i5     // %1 is 0b10101
      %2 = arith.trunci %1 : i5 to i4 // %2 is 0b0101
      %3 = arith.trunci %1 : i5 to i3 // %3 is 0b101

      %5 = arith.trunci %0 : vector<2 x i32> to vector<2 x i16>
    ```
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TruncFOp
//===----------------------------------------------------------------------===//

def Arith_TruncFOp :
    Arith_Op<"truncf",
      [Pure, SameOperandsAndResultShape, SameInputOutputTensorDims,
       DeclareOpInterfaceMethods<ArithRoundingModeInterface>,
       DeclareOpInterfaceMethods<ArithFastMathInterface>,
       DeclareOpInterfaceMethods<CastOpInterface>]>,
    Arguments<(ins FloatLike:$in,
                   OptionalAttr<Arith_RoundingModeAttr>:$roundingmode,
                   OptionalAttr<Arith_FastMathAttr>:$fastmath)>,
    Results<(outs FloatLike:$out)> {
  let summary = "cast from floating-point to narrower floating-point";
  let description = [{
    Truncate a floating-point value to a smaller floating-point-typed value.
    The destination type must be strictly narrower than the source type.
    If the value cannot be exactly represented, it is rounded using the
    provided rounding mode or the default one if no rounding mode is provided.
    When operating on vectors, casts elementwise.
  }];
  let builders = [
    OpBuilder<(ins "Type":$out, "Value":$in), [{
      $_state.addOperands(in);
      $_state.addTypes(out);
    }]>
  ];

  let hasFolder = 1;
  let hasVerifier = 1;
  let assemblyFormat = [{ $in ($roundingmode^)?
                          (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($in) `to` type($out) }];
}

//===----------------------------------------------------------------------===//
// UIToFPOp
//===----------------------------------------------------------------------===//

def Arith_UIToFPOp : Arith_IToFCastOp<"uitofp"> {
  let summary = "cast from unsigned integer type to floating-point";
  let description = [{
    Cast from a value interpreted as unsigned integer to the corresponding
    floating-point value. If the value cannot be exactly represented, it is
    rounded using the default rounding mode. When operating on vectors, casts
    elementwise.
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SIToFPOp
//===----------------------------------------------------------------------===//

def Arith_SIToFPOp : Arith_IToFCastOp<"sitofp"> {
  let summary = "cast from integer type to floating-point";
  let description = [{
    Cast from a value interpreted as a signed integer to the corresponding
    floating-point value. If the value cannot be exactly represented, it is
    rounded using the default rounding mode. When operating on vectors, casts
    elementwise.
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FPToUIOp
//===----------------------------------------------------------------------===//

def Arith_FPToUIOp : Arith_FToICastOp<"fptoui"> {
  let summary = "cast from floating-point type to integer type";
  let description = [{
    Cast from a value interpreted as floating-point to the nearest (rounding
    towards zero) unsigned integer value. When operating on vectors, casts
    elementwise.
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FPToSIOp
//===----------------------------------------------------------------------===//

def Arith_FPToSIOp : Arith_FToICastOp<"fptosi"> {
  let summary = "cast from floating-point type to integer type";
  let description = [{
    Cast from a value interpreted as floating-point to the nearest (rounding
    towards zero) signed integer value. When operating on vectors, casts
    elementwise.
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

// Index cast can convert between memrefs of signless integers and indices too.
def IndexCastTypeConstraint : TypeConstraint<Or<[
        SignlessIntegerLike.predicate,
        MemRefOf<[AnySignlessInteger, Index]>.predicate]>,
    "signless-integer-like or memref of signless-integer">;

def Arith_IndexCastOp
  : Arith_CastOp<"index_cast", IndexCastTypeConstraint, IndexCastTypeConstraint,
                 [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let summary = "cast between index and integer types";
  let description = [{
    Casts between scalar or vector integers and corresponding 'index' scalar or
    vectors. Index is an integer of platform-specific bit width. If casting to
    a wider integer, the value is sign-extended. If casting to a narrower
    integer, the value is truncated.
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// IndexCastUIOp
//===----------------------------------------------------------------------===//

def Arith_IndexCastUIOp
  : Arith_CastOp<"index_castui", IndexCastTypeConstraint, IndexCastTypeConstraint,
                 [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let summary = "unsigned cast between index and integer types";
  let description = [{
    Casts between scalar or vector integers and corresponding 'index' scalar or
    vectors. Index is an integer of platform-specific bit width. If casting to
    a wider integer, the value is zero-extended. If casting to a narrower
    integer, the value is truncated.
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

// Bitcast can convert between memrefs of signless integers, indices, and
// floats too.
def BitcastTypeConstraint : TypeConstraint<Or<[
        SignlessIntegerOrFloatLike.predicate,
        MemRefOf<[AnySignlessInteger, Index, AnyFloat]>.predicate]>,
    "signless-integer-or-float-like or memref of signless-integer or float">;

def Arith_BitcastOp : Arith_CastOp<"bitcast", BitcastTypeConstraint,
                                              BitcastTypeConstraint> {
  let summary = "bitcast between values of equal bit width";
  let description = [{
    Bitcast an integer or floating point value to an integer or floating point
    value of equal bit width. When operating on vectors, casts elementwise.

    Note that this implements a logical bitcast independent of target
    endianness. This allows constant folding without target information and is
    consitent with the bitcast constant folders in LLVM (see
    https://github.com/llvm/llvm-project/blob/18c19414eb/llvm/lib/IR/ConstantFold.cpp#L168)
    For targets where the source and target type have the same endianness (which
    is the standard), this cast will also change no bits at runtime, but it may
    still require an operation, for example if the machine has different
    floating point and integer register files. For targets that have a different
    endianness for the source and target types (e.g. float is big-endian and
    integer is little-endian) a proper lowering would add operations to swap the
    order of words in addition to the bitcast.
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

def Arith_CmpIOp
  : Arith_CompareOpOfAnyRank<"cmpi",
                             [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
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
    %x = arith.cmpi slt, %lhs, %rhs : i32

    // Generic form of the same operation.
    %x = "arith.cmpi"(%lhs, %rhs) {predicate = 2 : i64} : (i32, i32) -> i1

    // Custom form of vector equality comparison.
    %x = arith.cmpi eq, %lhs, %rhs : vector<4xi64>

    // Generic form of the same operation.
    %x = "arith.cmpi"(%lhs, %rhs) {predicate = 0 : i64}
        : (vector<4xi64>, vector<4xi64>) -> vector<4xi1>
    ```
  }];

  let arguments = (ins Arith_CmpIPredicateAttr:$predicate,
                       SignlessIntegerLikeOfAnyRank:$lhs,
                       SignlessIntegerLikeOfAnyRank:$rhs);

  let extraClassDeclaration = [{
    static arith::CmpIPredicate getPredicateByName(StringRef name);
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//

def Arith_CmpFOp : Arith_CompareOp<"cmpf",
    [DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
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
    %r1 = arith.cmpf oeq, %0, %1 : f32
    %r2 = arith.cmpf ult, %0, %1 : tensor<42x42xf64>
    %r3 = "arith.cmpf"(%0, %1) {predicate: 0} : (f8, f8) -> i1
    ```
  }];

  let arguments = (ins Arith_CmpFPredicateAttr:$predicate,
                       FloatLike:$lhs,
                       FloatLike:$rhs,
                       DefaultValuedAttr<
                         Arith_FastMathAttr, "::mlir::arith::FastMathFlags::none">:$fastmath);

  let extraClassDeclaration = [{
    static arith::CmpFPredicate getPredicateByName(StringRef name);
  }];

  let builders = [
    OpBuilder<(ins "::mlir::arith::CmpFPredicateAttr":$predicate,
                   "Value":$lhs, "Value":$rhs), [{
      build($_builder, $_state, predicate, lhs, rhs,
          mlir::arith::FastMathFlagsAttr::get($_builder.getContext(),
              mlir::arith::FastMathFlags::none));
    }]>
  ];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let assemblyFormat = [{ $predicate `,` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($lhs)}];
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

class BooleanConditionOrMatchingShape<string condition, string result> :
    PredOpTrait<
      condition # " is signless i1 or has matching shape",
      Or<[TypeIsPred<condition, I1>,
          AllShapesMatch<[condition, result]>.predicate]>>;

def SelectOp : Arith_Op<"select", [Pure,
    AllTypesMatch<["true_value", "false_value", "result"]>,
    BooleanConditionOrMatchingShape<"condition", "result">,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRangesFromOptional"]>,
  ] # ElementwiseMappable.traits> {
  let summary = "select operation";
  let description = [{
    The `arith.select` operation chooses one value based on a binary condition
    supplied as its first operand.

    If the value of the first operand (the condition) is `1`, then the second
    operand is returned, and the third operand is ignored, even if it was poison.

    If the value of the first operand (the condition) is `0`, then the third
    operand is returned, and the second operand is ignored, even if it was poison.

    If the value of the first operand (the condition) is poison, then the
    operation returns poison.

    The operation applies to vectors and tensors elementwise given the _shape_
    of all operands is identical. The choice is made for each element
    individually based on the value at the same position as the element in the
    condition operand. If an i1 is provided as the condition, the entire vector
    or tensor is chosen.

    Example:

    ```mlir
    // Custom form of scalar selection.
    %x = arith.select %cond, %true, %false : i32

    // Generic form of the same operation.
    %x = "arith.select"(%cond, %true, %false) : (i1, i32, i32) -> i32

    // Element-wise vector selection.
    %vx = arith.select %vcond, %vtrue, %vfalse : vector<42xi1>, vector<42xf32>

    // Full vector selection.
    %vx = arith.select %cond, %vtrue, %vfalse : vector<42xf32>
    ```
  }];

  let arguments = (ins BoolLike:$condition,
                       AnyType:$true_value,
                       AnyType:$false_value);
  let results = (outs AnyType:$result);

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  // FIXME: Switch this to use the declarative assembly format.
  let hasCustomAssemblyFormat = 1;
}

#endif // ARITH_OPS
