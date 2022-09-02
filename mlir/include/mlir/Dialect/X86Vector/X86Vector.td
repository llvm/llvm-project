//===-- X86VectorOps.td - X86Vector dialect operation defs -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the X86Vector dialect.
//
//===----------------------------------------------------------------------===//

#ifndef X86VECTOR_OPS
#define X86VECTOR_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// X86Vector dialect definition
//===----------------------------------------------------------------------===//

def X86Vector_Dialect : Dialect {
  let name = "x86vector";
  let cppNamespace = "::mlir::x86vector";
}

//===----------------------------------------------------------------------===//
// AVX512 op definitions
//===----------------------------------------------------------------------===//

// Operation that is part of the input dialect.
class AVX512_Op<string mnemonic, list<Trait> traits = []> :
  Op<X86Vector_Dialect, "avx512." # mnemonic, traits> {}

// Intrinsic operation used during lowering to LLVM IR.
class AVX512_IntrOp<string mnemonic, int numResults, list<Trait> traits = []> :
  LLVM_IntrOpBase<X86Vector_Dialect, "avx512.intr." # mnemonic,
                  "x86_avx512_" # !subst(".", "_", mnemonic),
                  [], [], traits, numResults>;

// Defined by first result overload. May have to be extended for other
// instructions in the future.
class AVX512_IntrOverloadedOp<string mnemonic,
                              list<Trait> traits = []> :
  LLVM_IntrOpBase<X86Vector_Dialect, "avx512.intr." # mnemonic,
                  "x86_avx512_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0],
                  /*list<int> overloadedOperands=*/[],
                  traits, /*numResults=*/1>;

//----------------------------------------------------------------------------//
// MaskCompressOp
//----------------------------------------------------------------------------//

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
  let hasVerifier = 1;
}

def MaskCompressIntrOp : AVX512_IntrOverloadedOp<"mask.compress", [
  NoSideEffect,
  AllTypesMatch<["a", "src", "res"]>,
  TypesMatchWith<"`k` has the same number of bits as elements in `res`",
                 "res", "k",
                 "VectorType::get({$_self.cast<VectorType>().getShape()[0]}, "
                 "IntegerType::get($_self.getContext(), 1))">]> {
  let arguments = (ins VectorOfLengthAndType<[16, 8],
                                             [F32, I32, F64, I64]>:$a,
                   VectorOfLengthAndType<[16, 8],
                                         [F32, I32, F64, I64]>:$src,
                   VectorOfLengthAndType<[16, 8],
                                         [I1]>:$k);
}

//----------------------------------------------------------------------------//
// MaskRndScaleOp
//----------------------------------------------------------------------------//

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

def MaskRndScalePSIntrOp : AVX512_IntrOp<"mask.rndscale.ps.512", 1, [
  NoSideEffect,
  AllTypesMatch<["src", "a", "res"]>]> {
  let arguments = (ins VectorOfLengthAndType<[16], [F32]>:$src,
                   I32:$k,
                   VectorOfLengthAndType<[16], [F32]>:$a,
                   I16:$imm,
                   LLVM_Type:$rounding);
}

def MaskRndScalePDIntrOp : AVX512_IntrOp<"mask.rndscale.pd.512", 1, [
  NoSideEffect,
  AllTypesMatch<["src", "a", "res"]>]> {
  let arguments = (ins VectorOfLengthAndType<[8], [F64]>:$src,
                   I32:$k,
                   VectorOfLengthAndType<[8], [F64]>:$a,
                   I8:$imm,
                   LLVM_Type:$rounding);
}

//----------------------------------------------------------------------------//
// MaskScaleFOp
//----------------------------------------------------------------------------//

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

def MaskScaleFPSIntrOp : AVX512_IntrOp<"mask.scalef.ps.512", 1, [
  NoSideEffect,
  AllTypesMatch<["src", "a", "b", "res"]>]> {
  let arguments = (ins VectorOfLengthAndType<[16], [F32]>:$src,
                   VectorOfLengthAndType<[16], [F32]>:$a,
                   VectorOfLengthAndType<[16], [F32]>:$b,
                   I16:$k,
                   LLVM_Type:$rounding);
}

def MaskScaleFPDIntrOp : AVX512_IntrOp<"mask.scalef.pd.512", 1, [
  NoSideEffect,
  AllTypesMatch<["src", "a", "b", "res"]>]> {
  let arguments = (ins VectorOfLengthAndType<[8], [F64]>:$src,
                   VectorOfLengthAndType<[8], [F64]>:$a,
                   VectorOfLengthAndType<[8], [F64]>:$b,
                   I8:$k,
                   LLVM_Type:$rounding);
}

//----------------------------------------------------------------------------//
// Vp2IntersectOp
//----------------------------------------------------------------------------//

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

def Vp2IntersectDIntrOp : AVX512_IntrOp<"vp2intersect.d.512", 2, [
  NoSideEffect]> {
  let arguments = (ins VectorOfLengthAndType<[16], [I32]>:$a,
                   VectorOfLengthAndType<[16], [I32]>:$b);
}

def Vp2IntersectQIntrOp : AVX512_IntrOp<"vp2intersect.q.512", 2, [
  NoSideEffect]> {
  let arguments = (ins VectorOfLengthAndType<[8], [I64]>:$a,
                   VectorOfLengthAndType<[8], [I64]>:$b);
}

//===----------------------------------------------------------------------===//
// AVX op definitions
//===----------------------------------------------------------------------===//

// Operation that is part of the input dialect.
class AVX_Op<string mnemonic, list<Trait> traits = []> :
  Op<X86Vector_Dialect, "avx." # mnemonic, traits> {}

// Operation that may be part of the input dialect, but whose
// form is somewhere between the user view of the operation
// and the actual lower level intrinsic in LLVM IR.
class AVX_LowOp<string mnemonic, list<Trait> traits = []> :
  Op<X86Vector_Dialect, "avx.intr." # mnemonic, traits> {}

// Intrinsic operation used during lowering to LLVM IR.
class AVX_IntrOp<string mnemonic, int numResults, list<Trait> traits = []> :
  LLVM_IntrOpBase<X86Vector_Dialect, "avx.intr." # mnemonic,
                  "x86_avx_" # !subst(".", "_", mnemonic),
                  [], [], traits, numResults>;

//----------------------------------------------------------------------------//
// AVX Rsqrt
//----------------------------------------------------------------------------//

def RsqrtOp : AVX_Op<"rsqrt", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Rsqrt";
  let arguments = (ins VectorOfLengthAndType<[8], [F32]>:$a);
  let results = (outs VectorOfLengthAndType<[8], [F32]>:$b);
  let assemblyFormat = "$a attr-dict `:` type($a)";
}

def RsqrtIntrOp : AVX_IntrOp<"rsqrt.ps.256", 1, [NoSideEffect,
  SameOperandsAndResultType]> {
  let arguments = (ins VectorOfLengthAndType<[8], [F32]>:$a);
}

//----------------------------------------------------------------------------//
// AVX Dot
//----------------------------------------------------------------------------//

def DotOp : AVX_LowOp<"dot", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Dot";
  let description = [{
    Computes the 4-way dot products of the lower and higher parts of the source
    vectors and broadcasts the two results to the lower and higher elements of
    the destination vector, respectively. Adding one element of the lower part
    to one element of the higher part in the destination vector yields the full
    dot product of the two source vectors.

    Example:

    ```mlir
    %0 = x86vector.avx.intr.dot %a, %b : vector<8xf32>
    %1 = vector.extractelement %0[%i0 : i32]: vector<8xf32>
    %2 = vector.extractelement %0[%i4 : i32]: vector<8xf32>
    %d = arith.addf %1, %2 : f32
    ```
  }];
  let arguments = (ins VectorOfLengthAndType<[8], [F32]>:$a,
                       VectorOfLengthAndType<[8], [F32]>:$b);
  let results = (outs VectorOfLengthAndType<[8], [F32]>:$res);
  let assemblyFormat = "$a `,` $b attr-dict `:` type($res)";
}

def DotIntrOp : AVX_IntrOp<"dp.ps.256", 1, [NoSideEffect,
    AllTypesMatch<["a", "b", "res"]>]> {
  let arguments = (ins VectorOfLengthAndType<[8], [F32]>:$a,
                       VectorOfLengthAndType<[8], [F32]>:$b, I8:$c);
  let results = (outs VectorOfLengthAndType<[8], [F32]>:$res);
}

#endif // X86VECTOR_OPS
