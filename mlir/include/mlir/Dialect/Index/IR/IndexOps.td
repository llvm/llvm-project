//===- IndexOps.td - Index operation definitions -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INDEX_OPS
#define INDEX_OPS

include "mlir/Dialect/Index/IR/IndexDialect.td"
include "mlir/Dialect/Index/IR/IndexEnums.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IndexOp
//===----------------------------------------------------------------------===//

/// Base class for Index dialect operations.
class IndexOp<string mnemonic, list<Trait> traits = []>
    : Op<IndexDialect, mnemonic,
      [Pure, DeclareOpInterfaceMethods<InferIntRangeInterface>] # traits>;

//===----------------------------------------------------------------------===//
// IndexBinaryOp
//===----------------------------------------------------------------------===//

/// Base class for binary Index dialect operations.
class IndexBinaryOp<string mnemonic, list<Trait> traits = []>
    : IndexOp<mnemonic, traits> {
  let arguments = (ins Index:$lhs, Index:$rhs);
  let results = (outs Index:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def Index_AddOp : IndexBinaryOp<"add"> {
  let summary = "index addition";
  let description = [{
    The `index.add` operation takes two index values and computes their sum.

    Example:

    ```mlir
    // c = a + b
    %c = index.add %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def Index_SubOp : IndexBinaryOp<"sub"> {
  let summary = "index subtraction";
  let description = [{
    The `index.sub` operation takes two index values and computes the difference
    of the first from the second operand.

    Example:

    ```mlir
    // c = a - b
    %c = index.sub %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def Index_MulOp : IndexBinaryOp<"mul"> {
  let summary = "index multiplication";
  let description = [{
    The `index.mul` operation takes two index values and computes their product.

    Example:

    ```mlir
    // c = a * b
    %c = index.mul %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DivSOp
//===----------------------------------------------------------------------===//

def Index_DivSOp : IndexBinaryOp<"divs"> {
  let summary = "index signed division";
  let description = [{
    The `index.divs` operation takes two index values and computes their signed
    quotient. Treats the leading bit as the sign and rounds towards zero, i.e.
    `6 / -2 = -3`.

    Note: division by zero and signed division overflow are undefined behaviour.

    Example:

    ```mlir
    // c = a / b
    %c = index.divs %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DivUOp
//===----------------------------------------------------------------------===//

def Index_DivUOp : IndexBinaryOp<"divu"> {
  let summary = "index unsigned division";
  let description = [{
    The `index.divu` operation takes two index values and computes their
    unsigned quotient. Treats the leading bit as the most significant and rounds
    towards zero, i.e. `6 / -2 = 0`.

    Note: division by zero is undefined behaviour.

    Example:

    ```mlir
    // c = a / b
    %c = index.divu %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// CeilDivSOp
//===----------------------------------------------------------------------===//

def Index_CeilDivSOp : IndexBinaryOp<"ceildivs"> {
  let summary = "index signed ceil division";
  let description = [{
    The `index.ceildivs` operation takes two index values and computes their
    signed quotient. Treats the leading bit as the sign and rounds towards
    positive infinity, i.e. `7 / -2 = -3`.

    Note: division by zero and signed division overflow are undefined behaviour.

    Example:

    ```mlir
    // c = ceil(a / b)
    %c = index.ceildivs %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// CeilDivUOp
//===----------------------------------------------------------------------===//

def Index_CeilDivUOp : IndexBinaryOp<"ceildivu"> {
  let summary = "index unsigned ceil division";
  let description = [{
    The `index.ceildivu` operation takes two index values and computes their
    unsigned quotient. Treats the leading bit as the most significant and rounds
    towards positive infinity, i.e. `6 / -2 = 1`.

    Note: division by zero is undefined behaviour.

    Example:

    ```mlir
    // c = ceil(a / b)
    %c = index.ceildivu %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// FloorDivSOp
//===----------------------------------------------------------------------===//

def Index_FloorDivSOp : IndexBinaryOp<"floordivs"> {
  let summary = "index signed floor division";
  let description = [{
    The `index.floordivs` operation takes two index values and computes their
    signed quotient. Treats the leading bit as the sign and rounds towards
    negative infinity, i.e. `5 / -2 = -3`.

    Note: division by zero and signed division overflow are undefined behaviour.

    Example:

    ```mlir
    // c = floor(a / b)
    %c = index.floordivs %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RemSOp
//===----------------------------------------------------------------------===//

def Index_RemSOp : IndexBinaryOp<"rems"> {
  let summary = "index signed remainder";
  let description = [{
    The `index.rems` operation takes two index values and computes their signed
    remainder. Treats the leading bit as the sign, i.e. `6 % -2 = 0`.

    Example:

    ```mlir
    // c = a % b
    %c = index.rems %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RemUOp
//===----------------------------------------------------------------------===//

def Index_RemUOp : IndexBinaryOp<"remu"> {
  let summary = "index unsigned remainder";
  let description = [{
    The `index.remu` operation takes two index values and computes their
    unsigned remainder. Treats the leading bit as the most significant, i.e.
    `6 % -2 = 6`.

    Example:

    ```mlir
    // c = a % b
    %c = index.remu %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MaxSOp
//===----------------------------------------------------------------------===//

def Index_MaxSOp : IndexBinaryOp<"maxs"> {
  let summary = "index signed maximum";
  let description = [{
    The `index.maxs` operation takes two index values and computes their signed
    maximum value. Treats the leading bit as the sign, i.e. `max(-2, 6) = 6`.

    Example:

    ```mlir
    // c = max(a, b)
    %c = index.maxs %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MaxUOp
//===----------------------------------------------------------------------===//

def Index_MaxUOp : IndexBinaryOp<"maxu"> {
  let summary = "index unsigned maximum";
  let description = [{
    The `index.maxu` operation takes two index values and computes their
    unsigned maximum value. Treats the leading bit as the most significant, i.e.
    `max(15, 6) = 15` or `max(-2, 6) = -2`.

    Example:

    ```mlir
    // c = max(a, b)
    %c = index.maxu %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MinSOp
//===----------------------------------------------------------------------===//

def Index_MinSOp : IndexBinaryOp<"mins"> {
  let summary = "index signed minimum";
  let description = [{
    The `index.mins` operation takes two index values and computes their signed
    minimum value. Treats the leading bit as the sign, i.e. `min(-2, 6) = -2`.

    Example:

    ```mlir
    // c = min(a, b)
    %c = index.mins %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MinUOp
//===----------------------------------------------------------------------===//

def Index_MinUOp : IndexBinaryOp<"minu"> {
  let summary = "index unsigned minimum";
  let description = [{
    The `index.minu` operation takes two index values and computes their
    unsigned minimum value. Treats the leading bit as the most significant, i.e.
    `min(15, 6) = 6` or `min(-2, 6) = 6`.

    Example:

    ```mlir
    // c = min(a, b)
    %c = index.minu %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ShlOp
//===----------------------------------------------------------------------===//

def Index_ShlOp : IndexBinaryOp<"shl"> {
  let summary = "index shift left";
  let description = [{
    The `index.shl` operation shifts an index value to the left by a variable
    amount. The low order bits are filled with zeroes. The RHS operand is always
    treated as unsigned. If the RHS operand is equal to or greater than the
    index bitwidth, the operation is undefined.

    Example:

    ```mlir
    // c = a << b
    %c = index.shl %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ShrSOp
//===----------------------------------------------------------------------===//

def Index_ShrSOp : IndexBinaryOp<"shrs"> {
  let summary = "signed index shift right";
  let description = [{
    The `index.shrs` operation shifts an index value to the right by a variable
    amount. The LHS operand is treated as signed. The high order bits are filled
    with copies of the most significant bit. If the RHS operand is equal to or
    greater than the index bitwidth, the operation is undefined.

    Example:

    ```mlir
    // c = a >> b
    %c = index.shrs %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ShrUOp
//===----------------------------------------------------------------------===//

def Index_ShrUOp : IndexBinaryOp<"shru"> {
  let summary = "unsigned index shift right";
  let description = [{
    The `index.shru` operation shifts an index value to the right by a variable
    amount. The LHS operand is treated as unsigned. The high order bits are
    filled with zeroes. If the RHS operand is equal to or greater than the index
    bitwidth, the operation is undefined.

    Example:

    ```mlir
    // c = a >> b
    %c = index.shru %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

def Index_AndOp : IndexBinaryOp<"and"> {
  let summary = "index bitwise and";
  let description = [{
    The `index.and` operation takes two index values and computes their bitwise
    and.

    Example:

    ```mlir
    // c = a & b
    %c = index.and %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

def Index_OrOp : IndexBinaryOp<"or"> {
  let summary = "index bitwise or";
  let description = [{
    The `index.or` operation takes two index values and computes their bitwise
    or.

    Example:

    ```mlir
    // c = a | b
    %c = index.or %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// XorOp
//===----------------------------------------------------------------------===//

def Index_XOrOp : IndexBinaryOp<"xor"> {
  let summary = "index bitwise xor";
  let description = [{
    The `index.xor` operation takes two index values and computes their bitwise
    xor.

    Example:

    ```mlir
    // c = a ^ b
    %c = index.xor %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// CastSOp
//===----------------------------------------------------------------------===//

def Index_CastSOp : IndexOp<"casts",
                           [DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "index signed cast";
  let description = [{
    The `index.casts` operation enables conversions between values of index type
    and concrete fixed-width integer types. If casting to a wider integer, the
    value is sign-extended. If casting to a narrower integer, the value is
    truncated.

    Example:

    ```mlir
    // Cast to i32
    %0 = index.casts %a : index to i32

    // Cast from i64
    %1 = index.casts %b : i64 to index
    ```
  }];

  let arguments = (ins AnyTypeOf<[AnyInteger, Index]>:$input);
  let results = (outs AnyTypeOf<[AnyInteger, Index]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

//===----------------------------------------------------------------------===//
// CastUOp
//===----------------------------------------------------------------------===//

def Index_CastUOp : IndexOp<"castu",
                           [DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "index unsigned cast";
  let description = [{
    The `index.castu` operation enables conversions between values of index type
    and concrete fixed-width integer types. If casting to a wider integer, the
    value is zero-extended. If casting to a narrower integer, the value is
    truncated.

    Example:

    ```mlir
    // Cast to i32
    %0 = index.castu %a : index to i32

    // Cast from i64
    %1 = index.castu %b : i64 to index
    ```
  }];

  let arguments = (ins AnyTypeOf<[AnyInteger, Index]>:$input);
  let results = (outs AnyTypeOf<[AnyInteger, Index]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

def Index_CmpOp : IndexOp<"cmp"> {
  let summary = "index compare";
  let description = [{
    The `index.cmp` operation takes two index values and compares them according
    to the comparison predicate and returns an `i1`. The following comparisons
    are supported:

    -   `eq`:  equal
    -   `ne`:  not equal
    -   `slt`: signed less than
    -   `sle`: signed less than or equal
    -   `sgt`: signed greater than
    -   `sge`: signed greater than or equal
    -   `ult`: unsigned less than
    -   `ule`: unsigned less than or equal
    -   `ugt`: unsigned greater than
    -   `uge`: unsigned greater than or equal

    The result is `1` if the comparison is true and `0` otherwise.

    Example:

    ```mlir
    // Signed less than comparison.
    %0 = index.cmp slt(%a, %b)

    // Unsigned greater than or equal comparison.
    %1 = index.cmp uge(%a, %b)

    // Not equal comparison.
    %2 = index.cmp ne(%a, %b)
    ```
  }];

  let arguments = (ins IndexCmpPredicateAttr:$pred, Index:$lhs, Index:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "`` $pred `(` $lhs `,` $rhs `)` attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SizeOfOp
//===----------------------------------------------------------------------===//

def Index_SizeOfOp : IndexOp<"sizeof"> {
  let summary = "size in bits of the index type";
  let description = [{
    The `index.sizeof` operation produces an index-typed SSA value equal to the
    size in bits of the `index` type. For example, on 32-bit systems, the result
    is `32 : index`, and on 64-bit systems, the result is `64 : index`.

    Example:

    ```mlir
    %0 = index.sizeof
    ```
  }];

  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def Index_ConstantOp : IndexOp<"constant", [
    ConstantLike,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "index constant";
  let description = [{
    The `index.constant` operation produces an index-typed SSA value equal to
    some index-typed integer constant.

    Example:

    ```mlir
    %0 = index.constant 42
    ```
  }];

  let arguments = (ins IndexAttr:$value);
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;

  let builders = [OpBuilder<(ins "int64_t":$value)>];
}

//===----------------------------------------------------------------------===//
// BoolConstantOp
//===----------------------------------------------------------------------===//

def Index_BoolConstantOp : IndexOp<"bool.constant", [
    ConstantLike,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "boolean constant";
  let description = [{
    The `index.bool.constant` operation produces an bool-typed SSA value equal
    to either `true` or `false`.

    This operation is used to materialize bool constants that arise when folding
    `index.cmp`.

    Example:

    ```mlir
    %0 = index.bool.constant true
    ```
  }];

  let arguments = (ins BoolAttr:$value);
  let results = (outs I1:$result);
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

#endif // INDEX_OPS
