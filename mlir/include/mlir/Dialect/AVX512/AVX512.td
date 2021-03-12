//===-- AVX512Ops.td - AVX512 dialect operation definitions *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the AVX512 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AVX512_OPS
#define AVX512_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AVX512 dialect definition
//===----------------------------------------------------------------------===//

def AVX512_Dialect : Dialect {
  let name = "avx512";
  let cppNamespace = "::mlir::avx512";
}

//===----------------------------------------------------------------------===//
// AVX512 op definitions
//===----------------------------------------------------------------------===//

class AVX512_Op<string mnemonic, list<OpTrait> traits = []> :
  Op<AVX512_Dialect, mnemonic, traits> {}

def MaskCompressOp : AVX512_Op<"mask.compress", [NoSideEffect,
  // TODO: Support optional arguments in `AllTypesMatch`. "type($src)" could
  // then be removed from assemblyFormat.
  AllTypesMatch<["a", "dst"]>,
  TypesMatchWith<"`k` has the same number of bits as elements in `dst`",
                 "dst", "k",
                 "VectorType::get({$_self.cast<VectorType>().getShape()[0]}, "
                 "IntegerType::get($_self.getContext(), 1))">]> {
  let summary = "Masked compress op";
  let description = [{
  The mask.compress op is an AVX512 specific op that can lower to the
  `llvm.mask.compress` instruction. Instead of `src`, a constant vector
  vector attribute `constant_src` may be specified. If neither `src` nor
  `constant_src` is specified, the remaining elements in the result vector are
  set to zero.

  #### From the Intel Intrinsics Guide:

  Contiguously store the active integer/floating-point elements in `a` (those
  with their respective bit set in writemask `k`) to `dst`, and pass through the
  remaining elements from `src`.
  }];
  let verifier = [{ return ::verify(*this); }];
  let arguments = (ins VectorOfLengthAndType<[16, 8],
                                             [I1]>:$k,
                   VectorOfLengthAndType<[16, 8],
                                         [F32, I32, F64, I64]>:$a,
                   Optional<VectorOfLengthAndType<[16, 8],
                                                  [F32, I32, F64, I64]>>:$src,
                   OptionalAttr<ElementsAttr>:$constant_src);
  let results = (outs VectorOfLengthAndType<[16, 8],
                                            [F32, I32, F64, I64]>:$dst);
  let assemblyFormat = "$k `,` $a (`,` $src^)? attr-dict"
                       " `:` type($dst) (`,` type($src)^)?";
}

def MaskRndScaleOp : AVX512_Op<"mask.rndscale", [NoSideEffect,
  AllTypesMatch<["src", "a", "dst"]>,
  TypesMatchWith<"imm has the same number of bits as elements in dst",
                 "dst", "imm",
                 "IntegerType::get($_self.getContext(), "
                 "($_self.cast<VectorType>().getShape()[0]))">]> {
  let summary = "Masked roundscale op";
  let description = [{
    The mask.rndscale op is an AVX512 specific op that can lower to the proper
    LLVMAVX512 operation: `llvm.mask.rndscale.ps.512` or
    `llvm.mask.rndscale.pd.512` instruction depending on the type of vectors it
    is applied to.

    #### From the Intel Intrinsics Guide:

    Round packed floating-point elements in `a` to the number of fraction bits
    specified by `imm`, and store the results in `dst` using writemask `k`
    (elements are copied from src when the corresponding mask bit is not set).
  }];
  // Supports vector<16xf32> and vector<8xf64>.
  let arguments = (ins VectorOfLengthAndType<[16, 8], [F32, F64]>:$src,
                   I32:$k,
                   VectorOfLengthAndType<[16, 8], [F32, F64]>:$a,
                   AnyTypeOf<[I16, I8]>:$imm,
                   // TODO: figure rounding out (optional operand?).
                   I32:$rounding
            );
  let results = (outs VectorOfLengthAndType<[16, 8], [F32, F64]>:$dst);
  let assemblyFormat =
    "$src `,` $k `,` $a `,` $imm `,` $rounding attr-dict `:` type($dst)";
}

def MaskScaleFOp : AVX512_Op<"mask.scalef", [NoSideEffect,
  AllTypesMatch<["src", "a", "b", "dst"]>,
  TypesMatchWith<"k has the same number of bits as elements in dst",
                 "dst", "k",
                 "IntegerType::get($_self.getContext(), "
                 "($_self.cast<VectorType>().getShape()[0]))">]> {
  let summary = "ScaleF op";
  let description = [{
    The `mask.scalef` op is an AVX512 specific op that can lower to the proper
    LLVMAVX512 operation: `llvm.mask.scalef.ps.512` or
    `llvm.mask.scalef.pd.512` depending on the type of MLIR vectors it is
    applied to.

    #### From the Intel Intrinsics Guide:

    Scale the packed floating-point elements in `a` using values from `b`, and
    store the results in `dst` using writemask `k` (elements are copied from src
    when the corresponding mask bit is not set).
  }];
  // Supports vector<16xf32> and vector<8xf64>.
  let arguments = (ins VectorOfLengthAndType<[16, 8], [F32, F64]>:$src,
                   VectorOfLengthAndType<[16, 8], [F32, F64]>:$a,
                   VectorOfLengthAndType<[16, 8], [F32, F64]>:$b,
                   AnyTypeOf<[I16, I8]>:$k,
                   // TODO: figure rounding out (optional operand?).
                   I32:$rounding
            );
  let results = (outs VectorOfLengthAndType<[16, 8], [F32, F64]>:$dst);
  // Fully specified by traits.
  let assemblyFormat =
    "$src `,` $a `,` $b `,` $k `,` $rounding attr-dict `:` type($dst)";
}

def Vp2IntersectOp : AVX512_Op<"vp2intersect", [NoSideEffect,
  AllTypesMatch<["a", "b"]>,
  TypesMatchWith<"k1 has the same number of bits as elements in a",
                 "a", "k1",
                 "VectorType::get({$_self.cast<VectorType>().getShape()[0]}, "
                 "IntegerType::get($_self.getContext(), 1))">,
  TypesMatchWith<"k2 has the same number of bits as elements in b",
                 // Should use `b` instead of `a`, but that would require
                 // adding `type($b)` to assemblyFormat.
                 "a", "k2",
                 "VectorType::get({$_self.cast<VectorType>().getShape()[0]}, "
                 "IntegerType::get($_self.getContext(), 1))">]> {
  let summary = "Vp2Intersect op";
  let description = [{
    The `vp2intersect` op is an AVX512 specific op that can lower to the proper
    LLVMAVX512 operation: `llvm.vp2intersect.d.512` or
    `llvm.vp2intersect.q.512` depending on the type of MLIR vectors it is
    applied to.

    #### From the Intel Intrinsics Guide:

    Compute intersection of packed integer vectors `a` and `b`, and store
    indication of match in the corresponding bit of two mask registers
    specified by `k1` and `k2`. A match in corresponding elements of `a` and
    `b` is indicated by a set bit in the corresponding bit of the mask
    registers.
  }];
  let arguments = (ins VectorOfLengthAndType<[16, 8], [I32, I64]>:$a,
                   VectorOfLengthAndType<[16, 8], [I32, I64]>:$b
                   );
  let results = (outs VectorOfLengthAndType<[16, 8], [I1]>:$k1,
                 VectorOfLengthAndType<[16, 8], [I1]>:$k2
                 );
  let assemblyFormat =
    "$a `,` $b attr-dict `:` type($a)";
}

#endif // AVX512_OPS
