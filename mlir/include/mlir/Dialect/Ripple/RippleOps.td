//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//
// This file contains operation definitions for the Ripple dialect.
//
//==============================================================================

#ifndef RIPPLE_OPS
#define RIPPLE_OPS

include "mlir/Dialect/Ripple/RippleBase.td"

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def Ripple_SetShapeOp : Op<Ripple_Dialect, "setshape", []> {
  let summary =
      "The '``ripple.setshape``' defines the SPMD block dimensions "
      "for a particular processing element. The operation returns  "
      "an opaque pointer corresponding to a '``ripple_block_t``' struct.";

  let arguments = (ins AnySignlessInteger:$peid,
      Variadic<AnySignlessInteger>:$size);
  let results = (outs PtrLikeTypeInterface:$result);

  let assemblyFormat = "$peid `[` $size attr-dict `:` type($size) `]` `:` "
                       "type($peid) `->` type($result)";
}

def Ripple_GetSizeOp : Op<Ripple_Dialect, "getsize", []> {
  let summary =
      "The '``ripple.getsize``' intrinsic returns the block size along the "
      "queried dimension for a particular '``ripple_block_t``'.";

  let arguments = (ins PtrLikeTypeInterface:$bs, AnySignlessInteger:$dim);
  let results = (outs AnySignlessInteger:$result);

  let assemblyFormat = "`(` $bs `:` type($bs) `)` `[` $dim attr-dict `:` "
                       "type($dim) `]` `->` type($result)";
}

def Ripple_IndexOp : Op<Ripple_Dialect, "index", []> {
  let summary = "The '``ripple.index``' intrinsic returns the index of an SPMD "
                "processing element executing the task within an SPMD block.";

  let arguments = (ins PtrLikeTypeInterface:$bs, AnySignlessInteger:$dim);
  let results = (outs AnySignlessInteger:$result);

  let assemblyFormat = "`(` $bs `:` type($bs) `)` `[` $dim attr-dict `:` "
                       "type($dim) `]` `->` type($result)";
}

def Ripple_BroadcastOp : Op<Ripple_Dialect, "broadcast", []> {
  let summary = "The '``ripple.broadcast``' intrinsic offers an option to "
                "explicitly splat a scalar value into a tensor whose shape "
                "is determined by the bitmask provided.";

  let arguments = (ins PtrLikeTypeInterface:$bs, I64:$dim,
      Ripple_ScalarType:$to_bcast);
  let results = (outs Ripple_ScalarType:$result);

  let assemblyFormat = "`(` $bs `:` type($bs) `)` `[` $dim `,` $to_bcast  "
                       "attr-dict `:` type($to_bcast) `]` `->` type($result)";
}

def Ripple_BroadcastPtrOp : Op<Ripple_Dialect, "broadcast.ptr", []> {
  let summary = "The '``ripple.broadcast.ptr``' intrinsic provides "
                "an explicit pointer broadcast of a pointer to a scalar "
                "value along dimensions defined by a bitmask.";

  let arguments = (ins PtrLikeTypeInterface:$bs, I64:$dim,
      PtrLikeTypeInterface:$to_bcast);
  let results = (outs PtrLikeTypeInterface:$result);

  let assemblyFormat = "`(` $bs `:` type($bs) `)` `[` $dim `,` $to_bcast  "
                       "attr-dict `:` type($to_bcast) `]` `->` type($result)";
}

def Ripple_SliceOp : Op<Ripple_Dialect, "slice", []> {
  let summary =
      "The '``ripple.slice``' intrinsic returns a copy of a slice of the input "
      "vector along the ranks queried.";

  let arguments = (ins Ripple_ScalarType:$src, Variadic<I64>:$shape);
  let results = (outs Ripple_ScalarType:$result);

  let assemblyFormat =
      "`[` $src attr-dict `:` type($src) `,` $shape `]` `->` type($result)";
}

def Ripple_ShuffleOp : Op<Ripple_Dialect, "shuffle", []> {
  let summary = "The '``ripple.shuffle``' intrinsic modifies the ordering of "
                "the elements of a one-dimensional block. The resulting "
                "order is determined by a provided shuffle function.";

  let arguments = (ins Ripple_ScalarType:$src, FunctionType:$callee);
  let results = (outs Ripple_ScalarType:$result);

  let assemblyFormat = "`[` $src  attr-dict `:` type($src) `,` $callee `:` "
                       "type($callee) `]` `->` type($result)";
}

def Ripple_ShufflePairOp : Op<Ripple_Dialect, "shuffle.pair", []> {
  let summary =
      "The '``ripple.shuffle.pair``' intrinsic extends the "
      "'``ripple.shuffle``' intrinsic to operate on the concatenation of two "
      "input blocks. The resulting order is determined by the provided shuffle "
      "function.";

  let arguments = (ins Ripple_ScalarType:$src1, Ripple_ScalarType:$src2,
      FunctionType:$callee);
  let results = (outs Ripple_ScalarType:$result);

  let assemblyFormat =
      "`[` $src1 attr-dict `:` type($src1) `,` $src2 `:` type($src2) `,` "
      "$callee `:` type($callee) `]` `->` type($result)";
}

class Ripple_ReduceOp<string mnemonic, list<Trait> traits = []>
    : Op<Ripple_Dialect, mnemonic, traits> {
  let arguments = (ins I64:$ind, Ripple_ScalarType:$partial);
  let results = (outs Ripple_ScalarType:$result);

  let assemblyFormat =
      "$ind `[` $partial attr-dict `:` type($partial) `]` `->` type($result)";
}

def Ripple_ReduceAddOp : Ripple_ReduceOp<"reduce.add", []> {
    let summary = "The ``ripple.reduce.add`` intrinsic performs a reduction "
                  "along the dimension specified by the bitfield. The result "
                  "is the scalar sum of all elements along that dimension.";
}

def Ripple_ReduceMulOp : Ripple_ReduceOp<"reduce.mul", []> {
    let summary = "The ``ripple.reduce.mul`` intrinsic performs a reduction "
                  "along the dimension specified by the bitfield. The result "
                  "is the scalar product of all elements along that dimension.";
}

def Ripple_ReduceAndOp : Ripple_ReduceOp<"reduce.and", []> {
    let summary = "The ``ripple.reduce.and`` intrinsic performs a reduction "
                  "along the dimension specified by the bitfield. The result "
                  "is the logical and of all elements along that dimension.";
}

def Ripple_ReduceOrOp : Ripple_ReduceOp<"reduce.or", []> {
    let summary = "The ``ripple.reduce.or`` intrinsic performs a reduction "
                  "along the dimension specified by the bitfield. The result "
                  "is the logical or of all elements along that dimension.";
}

def Ripple_ReduceXorOp : Ripple_ReduceOp<"reduce.xor", []> {
    let summary = "The ``ripple.reduce.xor`` intrinsic performs a reduction "
                  "along the dimension specified by the bitfield. The result "
                  "is the logical exclusive or of all elements along that "
                  "dimension.";
}

def Ripple_ReduceMaxOp : Ripple_ReduceOp<"reduce.max", []> {
    let summary = "The ``ripple.reduce.max`` intrinsic performs a reduction "
                  "along the dimension specified by the bitfield. The result "
                  "is the maximum value of all elements along that dimension.";
}

def Ripple_ReduceMinOp : Ripple_ReduceOp<"reduce.min", []> {
    let summary = "The ``ripple.reduce.min`` intrinsic performs a reduction "
                  "along the dimension specified by the bitfield. The result "
                  "is the minimum value of all elements along that dimension.";
}

#endif // RIPPLE_OPS
