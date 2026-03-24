//===-- X86Ops.td - X86 dialect operation defs -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the X86 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef X86_OPS
#define X86_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/X86/X86Interfaces.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// X86 dialect definition
//===----------------------------------------------------------------------===//

def X86_Dialect : Dialect {
  let name = "x86";
  let cppNamespace = "::mlir::x86";

  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// AVX512 op definitions
//===----------------------------------------------------------------------===//

// Operation that is part of the input dialect.
class AVX512_Op<string mnemonic, list<Trait> traits = []>
    : Op<X86_Dialect, "avx512." # mnemonic, traits> {
  let cppNamespace = X86_Dialect.cppNamespace # "::avx512";
}

//----------------------------------------------------------------------------//
// MaskCompressOp
//----------------------------------------------------------------------------//

def MaskCompressOp : AVX512_Op<"mask.compress", [Pure,
    X86IntrinsicOpInterface,
    // TODO: Support optional arguments in `AllTypesMatch`. "type($src)" could
    // then be removed from assemblyFormat.
    AllTypesMatch<["a", "dst"]>,
    TypesMatchWith<"`k` has the same number of bits as elements in `dst`",
                   "dst", "k",
                   "VectorType::get({::llvm::cast<VectorType>($_self).getShape()[0]}, "
                   "IntegerType::get($_self.getContext(), 1))">
  ]> {
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

  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Call the baseline overloaded intrisic.
      // Final overload name mangling is resolved by the created function call.
      return "llvm.x86.avx512.mask.compress";
    }

    SmallVector<Value> getIntrinsicOperands(
        ::mlir::ArrayRef<Value> operands,
        const ::mlir::LLVMTypeConverter &typeConverter,
        ::mlir::RewriterBase &rewriter);
  }];
}

//----------------------------------------------------------------------------//
// MaskRndScaleOp
//----------------------------------------------------------------------------//

def MaskRndScaleOp : AVX512_Op<"mask.rndscale", [Pure,
    X86IntrinsicOpInterface,
    AllTypesMatch<["src", "a", "dst"]>,
    TypesMatchWith<"imm has the same number of bits as elements in dst",
                   "dst", "imm",
                   "IntegerType::get($_self.getContext(), "
                   "(::llvm::cast<VectorType>($_self).getShape()[0]))">
  ]> {
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

  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      std::string intr = "llvm.x86.avx512.mask.rndscale";
      VectorType vecType = getSrc().getType();
      Type elemType = vecType.getElementType();
      intr += ".";
      intr += elemType.isF32() ? "ps" : "pd";
      unsigned elemBitWidth = vecType.getElementTypeBitWidth();
      unsigned opBitWidth = vecType.getShape()[0] * elemBitWidth;
      intr += "." + std::to_string(opBitWidth);
      return intr;
    }
  }];
}

//----------------------------------------------------------------------------//
// MaskScaleFOp
//----------------------------------------------------------------------------//

def MaskScaleFOp : AVX512_Op<"mask.scalef", [Pure,
    X86IntrinsicOpInterface,
    AllTypesMatch<["src", "a", "b", "dst"]>,
    TypesMatchWith<"k has the same number of bits as elements in dst",
                   "dst", "k",
                   "IntegerType::get($_self.getContext(), "
                   "(::llvm::cast<VectorType>($_self).getShape()[0]))">
  ]> {
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

  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      std::string intr = "llvm.x86.avx512.mask.scalef";
      VectorType vecType = getSrc().getType();
      Type elemType = vecType.getElementType();
      intr += ".";
      intr += elemType.isF32() ? "ps" : "pd";
      unsigned elemBitWidth = vecType.getElementTypeBitWidth();
      unsigned opBitWidth = vecType.getShape()[0] * elemBitWidth;
      intr += "." + std::to_string(opBitWidth);
      return intr;
    }
  }];
}

//----------------------------------------------------------------------------//
// Vp2IntersectOp
//----------------------------------------------------------------------------//

def Vp2IntersectOp : AVX512_Op<"vp2intersect", [Pure,
    X86IntrinsicOpInterface,
    AllTypesMatch<["a", "b"]>,
    TypesMatchWith<"k1 has the same number of bits as elements in a",
                   "a", "k1",
                   "VectorType::get({::llvm::cast<VectorType>($_self).getShape()[0]}, "
                   "IntegerType::get($_self.getContext(), 1))">,
    TypesMatchWith<"k2 has the same number of bits as elements in b",
                   // Should use `b` instead of `a`, but that would require
                   // adding `type($b)` to assemblyFormat.
                   "a", "k2",
                   "VectorType::get({::llvm::cast<VectorType>($_self).getShape()[0]}, "
                   "IntegerType::get($_self.getContext(), 1))">
  ]> {
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

  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      std::string intr = "llvm.x86.avx512.vp2intersect";
      VectorType vecType = getA().getType();
      Type elemType = vecType.getElementType();
      intr += ".";
      intr += elemType.isInteger(32) ? "d" : "q";
      unsigned elemBitWidth = vecType.getElementTypeBitWidth();
      unsigned opBitWidth = vecType.getShape()[0] * elemBitWidth;
      intr += "." + std::to_string(opBitWidth);
      return intr;
    }
  }];
}

//----------------------------------------------------------------------------//
// Dot BF16
//----------------------------------------------------------------------------//

def DotBF16Op : AVX512_Op<"dot", [Pure,
    X86IntrinsicOpInterface,
    AllTypesMatch<["a", "b"]>,
    AllTypesMatch<["src", "dst"]>,
    TypesMatchWith<"`a` has twice an many elements as `src`",
                   "src", "a",
                   "VectorType::get({::llvm::cast<VectorType>($_self).getShape()[0] * 2}, "
                   "BFloat16Type::get($_self.getContext()))">
  ]> {
  let summary = "Dot BF16 op";
  let description = [{
    The `dot` op is an AVX512-BF16 specific op that can lower to the proper
    LLVMAVX512BF16 operation `llvm.dpbf16ps` depending on the width of MLIR
    vectors it is applied to.

    #### From the Intel Intrinsics Guide:

    Compute dot-product of BF16 (16-bit) floating-point pairs in `a` and `b`,
    accumulating the intermediate single-precision (32-bit) floating-point
    elements with elements in `src`, and store the results in `dst`.

    Example:
    ```mlir
    %dst = x86.avx512.dot %src, %a, %b : vector<32xbf16> -> vector<16xf32>
    ```
  }];
  let arguments = (ins VectorOfLengthAndType<[4, 8, 16], [F32]>:$src,
                   VectorOfLengthAndType<[8, 16, 32], [BF16]>:$a,
                   VectorOfLengthAndType<[8, 16, 32], [BF16]>:$b
                   );
  let results = (outs VectorOfLengthAndType<[4, 8, 16], [F32]>:$dst);
  let assemblyFormat =
    "$src `,` $a `,` $b attr-dict `:` type($a) `->` type($src)";

  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      std::string intr = "llvm.x86.avx512bf16.dpbf16ps";
      VectorType vecType = getSrc().getType();
      unsigned elemBitWidth = vecType.getElementTypeBitWidth();
      unsigned opBitWidth = vecType.getShape()[0] * elemBitWidth;
      intr += "." + std::to_string(opBitWidth);
      return intr;
    }
  }];
}

//----------------------------------------------------------------------------//
// Convert packed F32 to packed BF16
//----------------------------------------------------------------------------//

def CvtPackedF32ToBF16Op : AVX512_Op<"cvt.packed.f32_to_bf16", [Pure,
    X86IntrinsicOpInterface,
    AllElementCountsMatch<["a", "dst"]>
  ]> {
  let summary = "Convert packed F32 to packed BF16 Data.";
  let description = [{
    The `convert_f32_to_bf16` op is an AVX512-BF16 specific op that can lower
    to the proper LLVMAVX512BF16 operation `llvm.cvtneps2bf16` depending on
    the width of MLIR vectors it is applied to.

    #### From the Intel Intrinsics Guide:

    Convert packed single-precision (32-bit) floating-point elements in `a` to
    packed BF16 (16-bit) floating-point elements, and store the results in `dst`.

    Example:
    ```mlir
    %dst = x86.avx512.cvt.packed.f32_to_bf16 %a : vector<8xf32> -> vector<8xbf16>
    ```
  }];
  let arguments = (ins VectorOfLengthAndType<[8, 16], [F32]>:$a);
  let results = (outs VectorOfLengthAndType<[8, 16], [BF16]>:$dst);
  let assemblyFormat =
    "$a attr-dict `:` type($a) `->` type($dst)";

  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      std::string intr = "llvm.x86.avx512bf16.cvtneps2bf16";
      VectorType vecType = getA().getType();
      unsigned elemBitWidth = vecType.getElementTypeBitWidth();
      unsigned opBitWidth = vecType.getShape()[0] * elemBitWidth;
      intr += "." + std::to_string(opBitWidth);
      return intr;
    }
  }];
}

//===----------------------------------------------------------------------===//
// AVX10 op definitions
//===----------------------------------------------------------------------===//

// Operation that is part of the input dialect.
class AVX10_Op<string mnemonic, list<Trait> traits = []>
    : Op<X86_Dialect, "avx10." # mnemonic, traits> {
  let cppNamespace = X86_Dialect.cppNamespace # "::avx10";
}

//----------------------------------------------------------------------------//
// AVX10 Int8 Dot
//----------------------------------------------------------------------------//

def AVX10DotInt8Op : AVX10_Op<"dot.i8", [Pure,
    X86IntrinsicOpInterface,
    AllTypesMatch<["a", "b"]>,
    AllTypesMatch<["w", "dst"]>,
    TypesMatchWith<"`a` has four times elements as `w`",
                   "w", "a",
                   "VectorType::get({::llvm::cast<VectorType>($_self).getShape()[0] * 4}, "
                   "IntegerType::get($_self.getContext(), 8))">
  ]> {
  let summary = "AVX10 Dot Int8 op";
  let description = [{
    The `dot` op is an AVX10-Int8 specific op that can lower to the proper
    LLVMAVX10-INT8 operation `llvm.vpdpbssd.512`.

    Multiply groups of 4 adjacent pairs of signed 8-bit integers in `a` with
    corresponding signed 8-bit integers in `b`, producing 4 intermediate signed 16-bit
    results. Sum these 4 results with the corresponding 32-bit integer in `w`, and
    store the packed 32-bit results in `dst`.

    Example:
    ```mlir
    %dst = x86.avx10.dot.i8 %w, %a, %b : vector<64xi8> -> vector<16xi32>
    ```
  }];
  let arguments = (ins VectorOfLengthAndType<[16], [I32]>:$w,
                   VectorOfLengthAndType<[64], [I8]>:$a,
                   VectorOfLengthAndType<[64], [I8]>:$b
                   );
  let results = (outs VectorOfLengthAndType<[16], [I32]>:$dst);
  let assemblyFormat =
    "$w `,` $a `,` $b attr-dict `:` type($a) `->` type($w)";

  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      return "llvm.x86.avx10.vpdpbssd.512";
    }
  }];
}


//===----------------------------------------------------------------------===//
// AVX op definitions
//===----------------------------------------------------------------------===//

// Operation that is part of the input dialect.
class AVX_Op<string mnemonic, list<Trait> traits = []>
  : Op<X86_Dialect, "avx." # mnemonic, traits> {
  let cppNamespace = X86_Dialect.cppNamespace # "::avx";
}

// Operation that may be part of the input dialect, but whose
// form is somewhere between the user view of the operation
// and the actual lower level intrinsic in LLVM IR.
class AVX_LowOp<string mnemonic, list<Trait> traits = []>
  : Op<X86_Dialect, "avx.intr." # mnemonic, traits> {
  let cppNamespace = X86_Dialect.cppNamespace # "::avx";
}

//----------------------------------------------------------------------------//
// AVX Rsqrt
//----------------------------------------------------------------------------//

def RsqrtOp : AVX_Op<"rsqrt", [Pure,
    X86IntrinsicOpInterface,
    SameOperandsAndResultType
  ]> {
  let summary = "Rsqrt";
  let arguments = (ins VectorOfLengthAndType<[8], [F32]>:$a);
  let results = (outs VectorOfLengthAndType<[8], [F32]>:$b);
  let assemblyFormat = "$a attr-dict `:` type($a)";

  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      return "llvm.x86.avx.rsqrt.ps.256";
    }
  }];
}

//----------------------------------------------------------------------------//
// AVX Dot
//----------------------------------------------------------------------------//

def DotOp : AVX_LowOp<"dot", [Pure,
    X86IntrinsicOpInterface,
    SameOperandsAndResultType
  ]> {
  let summary = "Dot";
  let description = [{
    Computes the 4-way dot products of the lower and higher parts of the source
    vectors and broadcasts the two results to the lower and higher elements of
    the destination vector, respectively. Adding one element of the lower part
    to one element of the higher part in the destination vector yields the full
    dot product of the two source vectors.

    Example:

    ```mlir
    %0 = x86.avx.intr.dot %a, %b : vector<8xf32>
    %1 = vector.extract %0[%i0] : f32 from vector<8xf32>
    %2 = vector.extract %0[%i4] : f32 from vector<8xf32>
    %d = arith.addf %1, %2 : f32
    ```
  }];
  let arguments = (ins VectorOfLengthAndType<[8], [F32]>:$a,
                       VectorOfLengthAndType<[8], [F32]>:$b);
  let results = (outs VectorOfLengthAndType<[8], [F32]>:$res);
  let assemblyFormat = "$a `,` $b attr-dict `:` type($res)";

  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Only one variant is supported right now - no extra mangling.
      return "llvm.x86.avx.dp.ps.256";
    }

    SmallVector<Value> getIntrinsicOperands(
        ::mlir::ArrayRef<Value> operands,
        const ::mlir::LLVMTypeConverter &typeConverter,
        ::mlir::RewriterBase &rewriter);
  }];
}

//----------------------------------------------------------------------------//
// AVX Int8 Dot
//----------------------------------------------------------------------------//

def DotInt8Op : AVX_Op<"dot.i8", [Pure,
    X86IntrinsicOpInterface,
    AllTypesMatch<["a", "b"]>,
    AllTypesMatch<["w", "dst"]>,
    TypesMatchWith<"`a` has four times elements as `w`",
                   "w", "a",
                   "VectorType::get({::llvm::cast<VectorType>($_self).getShape()[0] * 4}, "
                   "IntegerType::get($_self.getContext(), 8))">
  ]> {
  let summary = "Dot Int8 op";
  let description = [{
    The `dot` op is an AVX2-Int8 specific op that can lower to the proper
    LLVMAVX2-INT8 operation `llvm.vpdpbssd` depending on the width of MLIR
    vectors it is applied to.

    #### From the Intel Intrinsics Guide:

    Multiply groups of 4 adjacent pairs of signed 8-bit integers in `a` with 
    corresponding signed 8-bit integers in `b`, producing 4 intermediate signed 16-bit 
    results. Sum these 4 results with the corresponding 32-bit integer in `w`, and 
    store the packed 32-bit results in `dst`.

    Example:
    ```mlir
    %dst = x86.avx.dot.i8 %w, %a, %b : vector<32xi8> -> vector<8xi32>
    ```
  }];
  let arguments = (ins VectorOfLengthAndType<[4, 8], [I32]>:$w,
                   VectorOfLengthAndType<[16, 32], [I8]>:$a,
                   VectorOfLengthAndType<[16, 32], [I8]>:$b
                   );
  let results = (outs VectorOfLengthAndType<[4, 8], [I32]>:$dst);
  let assemblyFormat =
    "$w `,` $a `,` $b attr-dict `:` type($a) `->` type($w)";

  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      std::string intr = "llvm.x86.avx2.vpdpbssd";
      VectorType vecType = getW().getType();
      unsigned elemBitWidth = vecType.getElementTypeBitWidth();
      unsigned opBitWidth = vecType.getShape()[0] * elemBitWidth;
      intr += "." + std::to_string(opBitWidth);
      return intr;
    }
  }];
}

//----------------------------------------------------------------------------//
// AVX: Convert BF16/F16 to F32 and broadcast into packed F32
//----------------------------------------------------------------------------//

def BcstToPackedF32Op
  : AVX_Op<"bcst_to_f32.packed", [
    MemoryEffects<[MemRead]>,
    X86IntrinsicOpInterface
  ]> {
  let summary = "AVX: Broadcasts BF16/F16 into packed F32 Data.";
  let description = [{
    #### From the Intel Intrinsics Guide:

    Convert scalar BF16 or F16 (16-bit) floating-point element stored at memory locations
    starting at location `__A` to a single-precision (32-bit) floating-point,
    broadcast it to packed single-precision (32-bit) floating-point elements,
    and store the results in `dst`.

    Example:
    ```mlir
    %dst = x86.avx.bcst_to_f32.packed %a : memref<1xbf16> -> vector<8xf32>
    %dst = x86.avx.bcst_to_f32.packed %a : memref<1xf16> -> vector<8xf32>
    ```
  }];
  let arguments = (ins MemRefOf<[BF16, F16]>:$a);
  let results = (outs VectorOfLengthAndType<[4, 8], [F32]>:$dst);
  let assemblyFormat =
    "$a  attr-dict`:` type($a)`->` type($dst)";

  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      auto elementType =
        getA().getType().getElementType();
      std::string intr = "llvm.x86.";
      if (elementType.isBF16())
        intr += "vbcstnebf162ps";
      if (elementType.isF16())
        intr += "vbcstnesh2ps";
      VectorType vecType = getDst().getType();
      unsigned elemBitWidth = vecType.getElementTypeBitWidth();
      unsigned opBitWidth = vecType.getShape()[0] * elemBitWidth;
      intr += std::to_string(opBitWidth);
      return intr;
    }

    SmallVector<Value> getIntrinsicOperands(
        ::mlir::ArrayRef<Value> operands,
        const ::mlir::LLVMTypeConverter &typeConverter,
        ::mlir::RewriterBase &rewriter);
  }];

}

//------------------------------------------------------------------------------//
// AVX: Convert packed BF16/F16 even-indexed/odd-indexed elements into packed F32
//------------------------------------------------------------------------------//

def CvtPackedEvenIndexedToF32Op
  : AVX_Op<"cvt.packed.even.indexed_to_f32", [
    MemoryEffects<[MemRead]>,
    X86IntrinsicOpInterface
  ]> {
  let summary = "AVX: Convert packed BF16/F16 even-indexed elements into packed F32 Data.";
  let description = [{
    #### From the Intel Intrinsics Guide:

    Convert packed BF16 or F16 (16-bit) floating-point even-indexed elements stored at
    memory locations starting at location `__A` to packed single-precision
    (32-bit) floating-point elements, and store the results in `dst`.

    Example:
    ```mlir
    %dst = x86.avx.cvt.packed.even.indexed_to_f32 %a : memref<16xbf16> -> vector<8xf32>
    %dst = x86.avx.cvt.packed.even.indexed_to_f32 %a : memref<16xf16> -> vector<8xf32>
    ```
  }];
  let arguments = (ins MemRefOf<[BF16, F16]>:$a);
  let results = (outs VectorOfLengthAndType<[4, 8], [F32]>:$dst);
  let assemblyFormat =
    "$a  attr-dict`:` type($a)`->` type($dst)";

  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      auto elementType =
        getA().getType().getElementType();
      std::string intr = "llvm.x86.";
      if (elementType.isBF16())
        intr += "vcvtneebf162ps";
      if (elementType.isF16())
        intr += "vcvtneeph2ps";
      VectorType vecType = getDst().getType();
      unsigned elemBitWidth = vecType.getElementTypeBitWidth();
      unsigned opBitWidth = vecType.getShape()[0] * elemBitWidth;
      intr += std::to_string(opBitWidth);
      return intr;
    }

    SmallVector<Value> getIntrinsicOperands(
        ::mlir::ArrayRef<Value> operands,
        const ::mlir::LLVMTypeConverter &typeConverter,
        ::mlir::RewriterBase &rewriter);
  }];
}

def CvtPackedOddIndexedToF32Op
  : AVX_Op<"cvt.packed.odd.indexed_to_f32", [
    MemoryEffects<[MemRead]>,
    X86IntrinsicOpInterface
  ]> {
  let summary = "AVX: Convert packed BF16/F16 odd-indexed elements into packed F32 Data.";
  let description = [{
    #### From the Intel Intrinsics Guide:

    Convert packed BF16 or F16 (16-bit) floating-point odd-indexed elements stored at
    memory locations starting at location `__A` to packed single-precision
    (32-bit) floating-point elements, and store the results in `dst`.

    Example:
    ```mlir
    %dst = x86.avx.cvt.packed.odd.indexed_to_f32 %a : memref<16xbf16> -> vector<8xf32>
    %dst = x86.avx.cvt.packed.odd.indexed_to_f32 %a : memref<16xf16> -> vector<8xf32>
    ```
  }];
  let arguments = (ins MemRefOf<[BF16, F16]>:$a);
  let results = (outs VectorOfLengthAndType<[4, 8], [F32]>:$dst);
  let assemblyFormat =
    "$a  attr-dict`:` type($a)`->` type($dst)";

  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      auto elementType =
        getA().getType().getElementType();
      std::string intr = "llvm.x86.";
      if (elementType.isBF16())
        intr += "vcvtneobf162ps";
      if (elementType.isF16())
        intr += "vcvtneoph2ps";
      VectorType vecType = getDst().getType();
      unsigned elemBitWidth = vecType.getElementTypeBitWidth();
      unsigned opBitWidth = vecType.getShape()[0] * elemBitWidth;
      intr += std::to_string(opBitWidth);
      return intr;
    }

    SmallVector<Value> getIntrinsicOperands(
        ::mlir::ArrayRef<Value> operands,
        const ::mlir::LLVMTypeConverter &typeConverter,
        ::mlir::RewriterBase &rewriter);
  }];
}

//===----------------------------------------------------------------------===//
// AMX Tile definition
//===----------------------------------------------------------------------===//

class AMX_Type<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<X86_Dialect, "AMX" # typeName, traits> {
  let mnemonic = "amx." # typeMnemonic;
}

def AMX_TileTypeElementType : AnyTypeOf<[F32, F16, BF16, I32, I8]> {
  let cppFunctionName = "isValidTileTypeElementType";
}

def AMX_TileType : AMX_Type<"Tile", "tile", [ShapedTypeInterface, ValueSemantics]> {
  let summary = "AMX 2D tile to be used by AMX opertaions.";

  let description = [{
    This type is used to represent values in AMX tile registers. All AMX operations
    work on AMX tiles and these tiles cannot be used in other operations directly.
    LLVM IR type for AMX tile is a primitive type, but in MLIR we provide shape and
    element type for IR verification and lowering to LLVMIR dialect.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    AMX_TileTypeElementType:$elementType
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape, "Type":$elementType), [{
      return $_get(elementType.getContext(), shape, elementType);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns if this type is ranked (always true).
    bool hasRank() const { return true; }

    /// Clone this tile type with the given shape and element type. If the
    /// provided shape is `std::nullopt`, the current shape of the type is used.
    AMXTileType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                       Type elementType) const {
      return get(shape.value_or(getShape()), elementType);
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def IsAMXTilePred : And<[CPred<"::llvm::isa<::mlir::x86::AMXTileType>($_self)">,
  CPred<[{::llvm::cast<::mlir::x86::AMXTileType>($_self).getRank() == 2}]>]>;

class AMXTileOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsAMXTilePred, "tile",
                      "::mlir::x86::AMXTileType">;

def AnyAMXTile : AMXTileOf<[F32, F16, BF16, I32, I8]>;

def AMXTileF32 : AMXTileOf<[F32]>;

def AMXTileF16OrBF16 : AMXTileOf<[F16, BF16]>;

def AMXTileI32 : AMXTileOf<[I32]>;

def AMXTileI8 : AMXTileOf<[I8]>;

//===----------------------------------------------------------------------===//
// AMX Op definitions
//===----------------------------------------------------------------------===//

class AMX_Op<string mnemonic, list<Trait> traits = []>
    : Op<X86_Dialect, "amx." # mnemonic, traits> {
  let cppNamespace = X86_Dialect.cppNamespace # "::amx";
}

//===----------------------------------------------------------------------===//
// AMX Tile Zero
//===----------------------------------------------------------------------===//

def TileZeroOp : AMX_Op<"tile_zero", [
    X86IntrinsicOpInterface,
    MemoryEffects<[MemWrite]>
  ]> {
  let summary = "tile zero operation";
  let description = [{
    Zeroes the destination tile, with the shape defined by the 2-dim
    vector type of the result.
    
    The operation is eventually lowered into the "tilezero" instruction
    with the corresponding tile configuration.
    
    With the write memory effect, each `x86.amx.tile_zero` operation serves as
    a compilation hint to use a separate tile register.

    Example:

    ```mlir
      %0 = x86.amx.tile_zero : !x86.amx.tile<16x16xbf16>
    ```
  }];
  let results = (outs AnyAMXTile:$res);
  let extraClassDeclaration = [{
    AMXTileType getTileType() {
      return ::llvm::cast<AMXTileType>(getRes().getType());
    }

    std::string getIntrinsicName() {
      return "llvm.x86.tilezero.internal";
    }
    SmallVector<Value> getIntrinsicOperands(
        ::mlir::ArrayRef<Value> operands,
        const ::mlir::LLVMTypeConverter &typeConverter,
        ::mlir::RewriterBase &rewriter);
  }];
  let assemblyFormat = "attr-dict `:` qualified(type($res))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AMX Tile Load
//===----------------------------------------------------------------------===//

def TileLoadOp : AMX_Op<"tile_load", [
    X86IntrinsicOpInterface,
    MemoryEffects<[MemWrite]>,
    AttrSizedOperandSegments
  ]> {
  let summary = "tile load operation";
  let description = [{
    Loads a tile from memory defined by a `base` and `indices`, with the
    shape defined by the 2-dim vector type of the result.
    The tile's rows are populated by reading contiguous elements starting
    at the `base`. For each tile row, the `base` is incremented by `stride`
    number of elements.

    The tile is loaded using the following indexing scheme:

    ```
    for row in enumerate(tile_rows):
      mem_row = base[i0, i1, ..., iN + row * stride]
      for col in enumerate(tile_cols):
        tile[row, col] = mem_row[col]
    ```

    If the `stride` is not provided, then the `base` buffer must be at least
    2-dimensional, and the `stride` is automatically inferred and corresponds
    to the stride of the buffer's second innermost dimension.

    The operation is eventually lowered into the "tileloadd" instruction
    with the corresponding tile configuration.

    With the write memory effect, each `x86.amx.tile_load` operation serves as
    a compilation hint to use a separate tile register.

    Example:

    ```mlir
      // Tile load from a 2-D memref with implicit stride.
      %0 = x86.amx.tile_load %arg0[%c0, %c0] : memref<?x?xi8> into !x86.amx.tile<16x64xi8>

      // Tile load from a 1-D memref with explicit stride.
      %0 = x86.amx.tile_load %arg0[%c0], %stride : memref<?xi8> into !x86.amx.tile<16x64xi8>
    ```
  }];
  let arguments = (ins Arg<AnyMemRef, "load base", [MemRead]>:$base,
                   Variadic<Index>:$indices,
                   Optional<Index>:$stride);
  let results = (outs AnyAMXTile:$res);
  let builders = [
    OpBuilder<(ins "Type":$res, "Value":$base, "ValueRange":$indices)>
  ];
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    AMXTileType getTileType() {
      return ::llvm::cast<AMXTileType>(getRes().getType());
    }

    std::string getIntrinsicName() {
      return "llvm.x86.tileloadd64.internal";
    }
    SmallVector<Value> getIntrinsicOperands(
        ::mlir::ArrayRef<Value> operands,
        const ::mlir::LLVMTypeConverter &typeConverter,
        ::mlir::RewriterBase &rewriter);
  }];
  let assemblyFormat = "$base `[` $indices `]` (`,` $stride^ )? attr-dict"
                       "`:` type($base) `into` qualified(type($res))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AMX Tile Store
//===----------------------------------------------------------------------===//

def TileStoreOp : AMX_Op<"tile_store", [
    X86IntrinsicOpInterface,
    AttrSizedOperandSegments
  ]> {
  let summary = "tile store operation";
  let description = [{
    Stores a tile to memory defined by a `base` and `indices`, with the
    shape defined by the 2-dim vector type of the value.
    The tile's rows are written contiguously to the buffer starting at
    the `base`. For each tile row, the `base` is incremented by `stride`
    number of elements.

    The tile is stored using the following indexing scheme:

    ```
    for row in enumerate(tile_rows):
      mem_row = base[i0, i1, ..., iN + row * stride]
      for col in enumerate(tile_cols):
        mem_row[col] = tile[row, col]
    ```

    If the `stride` is not provided, then the `base` buffer must be at least
    2-dimensional, and the `stride` is automatically inferred and corresponds
    to the stride of the buffer's second innermost dimension.

    The operation is eventually lowered into the "tilestored" instruction
    with the corresponding tile configuration.

    Example:

    ```mlir
      // Tile store to a 2-D memref with implicit stride.
      x86.amx.tile_store %arg1[%c0, %c0], %0 : memref<?x?xi8>, !x86.amx.tile<16x64xi8>

      // Tile store to a 1-D memref with explicit stride.
      x86.amx.tile_store %arg1[%c0], %0, %stride : memref<?xi8>, !x86.amx.tile<16x64xi8>
    ```
  }];
  let arguments = (ins Arg<AnyMemRef, "store base", [MemWrite]>:$base,
                   Variadic<Index>:$indices,
                   AnyAMXTile:$val,
                   Optional<Index>:$stride);
  let builders = [
    OpBuilder<(ins "Value":$base, "ValueRange":$indices, "Value":$val)>
  ];
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    AMXTileType getTileType() {
      return ::llvm::cast<AMXTileType>(getVal().getType());
    }

    std::string getIntrinsicName() {
      return "llvm.x86.tilestored64.internal";
    }
    SmallVector<Value> getIntrinsicOperands(
        ::mlir::ArrayRef<Value> operands,
        const ::mlir::LLVMTypeConverter &typeConverter,
        ::mlir::RewriterBase &rewriter);
  }];
  let assemblyFormat = "$base `[` $indices `]` `,` $val (`,` $stride^ )?"
                       "attr-dict `:` type($base) `,` qualified(type($val))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AMX Tile Multiply
//===----------------------------------------------------------------------===//

def TileMulFOp : AMX_Op<"tile_mulf", [Pure,
    X86IntrinsicOpInterface,
    AllTypesMatch<["acc", "res"]>
  ]> {
  let summary = "tile multiplication operation (floating-point)";
  let description = [{
    Multiplies a "m x k" tile with a "k x n" tile and accumulates the results
    into a "m x n" destination tile. Supports "f32 <- bf16 x bf16" (with
    pairs of "bf16").
    
    The operation is eventually lowered into the "tdpbf16ps" instruction with
    the corresponding tile configuration.

    Example:

    ```mlir
      %0 = x86.amx.tile_mulf %a, %b, %c
        : !x86.amx.tile<16x32xbf16>, !x86.amx.tile<16x32xbf16>, !x86.amx.tile<16x16xf32>
    ```
  }];
  let arguments = (ins AMXTileF16OrBF16:$lhs,
                       AMXTileF16OrBF16:$rhs,
                       AMXTileF32:$acc);
  let results = (outs AMXTileF32:$res);
  let extraClassDeclaration = [{
    AMXTileType getLhsTileType() {
      return ::llvm::cast<AMXTileType>(getLhs().getType());
    }
    AMXTileType getRhsTileType() {
      return ::llvm::cast<AMXTileType>(getRhs().getType());
    }
    AMXTileType getTileType() {
      return ::llvm::cast<AMXTileType>(getRes().getType());
    }

    std::string getIntrinsicName() {
      std::string intr = "llvm.x86.tdp";
      auto elementType =
        getLhsTileType().getElementType();
      intr += elementType.isF16() ? "fp16" : "bf16";
      intr += "ps.internal";
      return intr;
    }
    SmallVector<Value> getIntrinsicOperands(
        ::mlir::ArrayRef<Value> operands,
        const ::mlir::LLVMTypeConverter &typeConverter,
        ::mlir::RewriterBase &rewriter);
  }];
  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` "
                       "qualified(type($lhs)) `,` qualified(type($rhs))"
                       " `,` qualified(type($acc)) ";
  let hasVerifier = 1;
}

def TileMulIOp : AMX_Op<"tile_muli", [Pure,
    X86IntrinsicOpInterface,
    AllTypesMatch<["acc", "res"]>
  ]> {
  let summary = "tile multiplication operation (integer)";
  let description = [{
    Multiplies a "m x k" tile with a "k x n" tile and accumulates the results
    into a "m x n" destination tile. Supports all "si32 <- s/ui8 x s/ui8"
    combinations (4 bytes packed into dwords in the columns of both the
    source operand tiles; the zero or sign extension is specified with
    the attributes and default to sign extended).
    
    The operation is eventually lowered into one of the "tdpbssd",
    "tdpbsud", "tdpbusd", or "tdpbuud" instructions with the corresponding
    tile configuration.

    Example:

    ```mlir
      %0 = x86.amx.tile_muli %a zext, %b zext, %c
        : !x86.amx.tile<16x64xi8>, !x86.amx.tile<16x64xi8>, !x86.amx.tile<16x16xi32>
    ```
  }];
  let arguments = (ins AMXTileI8:$lhs,
                       AMXTileI8:$rhs,
                       AMXTileI32:$acc,
                       UnitAttr:$isZextLhs,
                       UnitAttr:$isZextRhs
                       );
  let results = (outs AMXTileI32:$res);
  let extraClassDeclaration = [{
    AMXTileType getLhsTileType() {
      return ::llvm::cast<AMXTileType>(getLhs().getType());
    }
    AMXTileType getRhsTileType() {
      return ::llvm::cast<AMXTileType>(getRhs().getType());
    }
    AMXTileType getTileType() {
      return ::llvm::cast<AMXTileType>(getRes().getType());
    }

    std::string getIntrinsicName() {
      std::string intr = "llvm.x86.tdpb";
      intr += getIsZextLhs() ? "u" : "s";
      intr += getIsZextRhs() ? "u" : "s";
      intr += "d.internal";
      return intr;
    }
    SmallVector<Value> getIntrinsicOperands(
        ::mlir::ArrayRef<Value> operands,
        const ::mlir::LLVMTypeConverter &typeConverter,
        ::mlir::RewriterBase &rewriter);
  }];
  let assemblyFormat = "$lhs (`zext` $isZextLhs^)? `,` $rhs (`zext` $isZextRhs^)? `,` $acc attr-dict `:` "
                       "qualified(type($lhs)) `,` qualified(type($rhs)) `,` qualified(type($acc)) ";
  let hasVerifier = 1;
}

#endif // X86_OPS
