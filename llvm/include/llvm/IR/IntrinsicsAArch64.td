//===- IntrinsicsAARCH64.td - Defines AARCH64 intrinsics ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the AARCH64-specific intrinsics.
// SVE/SME TargetGuards describe legal overloads and feature requirements of
// the intrinsics below.
//
//===----------------------------------------------------------------------===//

// Legal overloads and non-streaming and streaming feature guards.
// Non-streaming guards explicitly require SVE when needed; streaming mode
// supplies SME. An empty guard makes that mode unavailable.
// NonStreamingVariant and StreamingVariant take one guard for intrinsics
// available in only one mode.
// Each inner type list is an independent overload in intrinsic overload-index
// order. Dependent and fixed result and operand types are inferred by TableGen.
defvar Invalid = "";

class Variant<string nonStreamingGuard, string streamingGuard,
              list<list<LLVMType>> overloads>
    : GuardedVariant<[nonStreamingGuard, streamingGuard], overloads>;

class NonStreamingVariant<string nonStreamingGuard,
                          list<list<LLVMType>> overloads>
    : Variant<nonStreamingGuard, Invalid, overloads>;

class StreamingVariant<string streamingGuard,
                       list<list<LLVMType>> overloads>
    : Variant<Invalid, streamingGuard, overloads>;

let TargetPrefix = "aarch64" in {

def int_aarch64_ldxr : Intrinsic<[llvm_i64_ty], [llvm_anyptr_ty],
                                 [IntrNoFree, IntrWillReturn]>;
def int_aarch64_ldaxr : Intrinsic<[llvm_i64_ty], [llvm_anyptr_ty],
                                  [IntrNoFree, IntrWillReturn]>;
def int_aarch64_stxr : Intrinsic<[llvm_i32_ty], [llvm_i64_ty, llvm_anyptr_ty],
                                 [IntrNoFree, IntrWillReturn]>;
def int_aarch64_stlxr : Intrinsic<[llvm_i32_ty], [llvm_i64_ty, llvm_anyptr_ty],
                                  [IntrNoFree, IntrWillReturn]>;

def int_aarch64_ldxp : Intrinsic<[llvm_i64_ty, llvm_i64_ty], [llvm_ptr_ty],
                                 [IntrNoFree, IntrWillReturn]>;
def int_aarch64_ldaxp : Intrinsic<[llvm_i64_ty, llvm_i64_ty], [llvm_ptr_ty],
                                  [IntrNoFree, IntrWillReturn]>;
def int_aarch64_stxp : Intrinsic<[llvm_i32_ty],
                               [llvm_i64_ty, llvm_i64_ty, llvm_ptr_ty],
                               [IntrNoFree, IntrWillReturn]>;
def int_aarch64_stlxp : Intrinsic<[llvm_i32_ty],
                                  [llvm_i64_ty, llvm_i64_ty, llvm_ptr_ty],
                                  [IntrNoFree, IntrWillReturn]>;

def int_aarch64_clrex : Intrinsic<[]>;

def int_aarch64_sdiv : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>,
                                LLVMMatchType<0>], [IntrNoMem]>;
def int_aarch64_udiv : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>,
                                LLVMMatchType<0>], [IntrNoMem]>;

def int_aarch64_fjcvtzs : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_double_ty], [IntrNoMem]>;

def int_aarch64_cls: DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;
def int_aarch64_cls64: DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem]>;

def int_aarch64_frint32z
    : DefaultAttrsIntrinsic<[ llvm_anyfloat_ty ], [ LLVMMatchType<0> ],
                            [ IntrNoMem ]>;
def int_aarch64_frint64z
    : DefaultAttrsIntrinsic<[ llvm_anyfloat_ty ], [ LLVMMatchType<0> ],
                            [ IntrNoMem ]>;
def int_aarch64_frint32x
    : DefaultAttrsIntrinsic<[ llvm_anyfloat_ty ], [ LLVMMatchType<0> ],
                            [ IntrNoMem ]>;
def int_aarch64_frint64x
    : DefaultAttrsIntrinsic<[ llvm_anyfloat_ty ], [ LLVMMatchType<0> ],
                            [ IntrNoMem ]>;

def int_aarch64_sys  : DefaultAttrsIntrinsic<[], [llvm_i32_ty, llvm_i32_ty,
                       llvm_i32_ty, llvm_i32_ty, llvm_i64_ty],
                       [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>,
                       ImmArg<ArgIndex<3>>]>;
//===----------------------------------------------------------------------===//
// HINT

def int_aarch64_hint : DefaultAttrsIntrinsic<[], [llvm_i32_ty]>;

def int_aarch64_break : Intrinsic<[], [llvm_i32_ty],
    [IntrNoMem, IntrHasSideEffects, IntrNoReturn, IntrCold, ImmArg<ArgIndex<0>>]>;

def int_aarch64_hlt : Intrinsic<[], [llvm_i32_ty],
    [IntrNoMem, IntrHasSideEffects, IntrNoReturn, IntrCold, ImmArg<ArgIndex<0>>]>;

// The first operand is the 16-bit instruction immediate.  The remaining four
// operands are the values passed in X0-X3 following the Microsoft __hvc calling
// convention (this is a Microsoft software convention, not an architectural
// requirement); an unused argument is passed as poison.  The result is the
// value left in X0.
def int_aarch64_hvc : Intrinsic<[llvm_i64_ty],
    [llvm_i32_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
    [IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

def int_aarch64_svc : Intrinsic<[llvm_i64_ty],
    [llvm_i32_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
    [IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

def int_aarch64_prefetch : Intrinsic<[],
    [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrInaccessibleMemOrArgMemOnly, IntrWillReturn, ReadOnly<ArgIndex<0>>,
     ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>
     ]>,
    ClangBuiltin<"__builtin_arm_prefetch">;

def int_aarch64_prefetch_ir : Intrinsic<[], [llvm_ptr_ty],
    [IntrHasSideEffects, IntrInaccessibleMemOrArgMemOnly,
     IntrWillReturn, ReadOnly<ArgIndex<0>>]>,
    ClangBuiltin<"__builtin_arm_prefetch_ir">;

def int_aarch64_range_prefetch : Intrinsic<[],
    [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i64_ty],
    [IntrInaccessibleMemOrArgMemOnly, IntrWillReturn, ReadOnly<ArgIndex<0>>,
     ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>,
    ClangBuiltin<"__builtin_arm_range_prefetch">;

//===----------------------------------------------------------------------===//
// Data Barrier Instructions

def int_aarch64_dmb : ClangBuiltin<"__builtin_arm_dmb">, MSBuiltin<"__dmb">,
                      Intrinsic<[], [llvm_i32_ty], [IntrNoFree, IntrWillReturn]>;
def int_aarch64_dsb : ClangBuiltin<"__builtin_arm_dsb">, MSBuiltin<"__dsb">,
                      Intrinsic<[], [llvm_i32_ty], [IntrNoFree, IntrWillReturn]>;
def int_aarch64_isb : ClangBuiltin<"__builtin_arm_isb">, MSBuiltin<"__isb">,
                      Intrinsic<[], [llvm_i32_ty], [IntrNoFree, IntrWillReturn]>;

// A space-consuming intrinsic primarily for testing block and jump table
// placements. The first argument is the number of bytes this "instruction"
// takes up, the second and return value are essentially chains, used to force
// ordering during ISel.
def int_aarch64_space : DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_i32_ty, llvm_i64_ty], []>;

//===----------------------------------------------------------------------===//
// Guarded Control Stack

def int_aarch64_chkfeat : ClangBuiltin<"__builtin_arm_chkfeat">,
                          DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_i64_ty],
                                                [IntrNoMem]>;

// FIXME: This should be marked as [IntrReadMem, IntrHasSideEffects], as it has
// the side-effect of updating gcspr, but this combination doesn't work
// correctly.
def int_aarch64_gcspopm : ClangBuiltin<"__builtin_arm_gcspopm">,
                          DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_i64_ty],
                                                []>;

def int_aarch64_gcsss : ClangBuiltin<"__builtin_arm_gcsss">,
                        DefaultAttrsIntrinsic<[llvm_ptr_ty], [llvm_ptr_ty], []>;
}

//===----------------------------------------------------------------------===//
// Advanced SIMD (NEON)

let TargetPrefix = "aarch64" in {  // All intrinsics start with "llvm.aarch64.".
  class AdvSIMD_2Scalar_Float_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, LLVMMatchType<0>],
                [IntrNoMem]>;

  class AdvSIMD_FPToIntRounding_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyint_ty], [llvm_anyfloat_ty], [IntrNoMem]>;

  class AdvSIMD_1IntArg_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>], [IntrNoMem]>;
  class AdvSIMD_1FloatArg_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
  class AdvSIMD_1VectorArg_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty], [LLVMMatchType<0>], !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_1VectorArg_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty], [LLVMMatchType<0>], !listconcat(Attrs, [IntrNoMem])>;
  class AdvSIMD_1VectorArg_Expand_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;
  class AdvSIMD_1IntArg_Narrow_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_any_ty], [llvm_any_ty], [IntrNoMem]>;
  class AdvSIMD_1VectorArg_Narrow_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMExtendedType<0>], [IntrNoMem]>;
  class AdvSIMD_1VectorArg_Int_Across_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyint_ty], [llvm_anyvector_ty], [IntrNoMem]>;
  class AdvSIMD_1VectorArg_Float_Across_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [llvm_anyvector_ty], [IntrNoMem]>;

  class AdvSIMD_2IntArg_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>],
                [IntrNoMem]>;
  class AdvSIMD_2FloatArg_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, LLVMMatchType<0>],
                [IntrNoMem]>;
  class AdvSIMD_2VectorArg_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty], [LLVMMatchType<0>, LLVMMatchType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_2VectorArg_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty], [LLVMMatchType<0>, LLVMMatchType<0>],
                !listconcat(Attrs, [IntrNoMem])>;
  class AdvSIMD_2Arg_FloatCompare_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyint_ty], [llvm_anyfloat_ty, LLVMMatchType<1>],
                [IntrNoMem]>;

  class Guarded_AdvSIMD_2Arg_FloatCompare_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyint_ty], [llvm_anyfloat_ty, LLVMMatchType<1>],
                [IntrNoMem]>;
  class AdvSIMD_2VectorArg_Long_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                [LLVMTruncatedType<0>, LLVMTruncatedType<0>],
                [IntrNoMem]>;
  class AdvSIMD_2VectorArg_Narrow_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                [LLVMExtendedType<0>, LLVMExtendedType<0>],
                [IntrNoMem]>;
  class AdvSIMD_2Arg_Scalar_Narrow_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                [LLVMExtendedType<0>, llvm_i32_ty],
                [IntrNoMem, ImmArg<ArgIndex<1>>]>;
  class AdvSIMD_2VectorArg_Scalar_Wide_BySize_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                [LLVMTruncatedType<0>],
                [IntrNoMem]>;
  class AdvSIMD_2VectorArg_Scalar_Wide_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                [LLVMTruncatedType<0>, llvm_i32_ty],
                [IntrNoMem]>;
  class AdvSIMD_2VectorArg_Lane_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                [LLVMMatchType<0>, llvm_anyint_ty, llvm_i32_ty],
                [IntrNoMem]>;
  class AdvSIMD_3IntArg_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                [IntrNoMem]>;
  class AdvSIMD_3VectorArg_Intrinsic<list<IntrinsicProperty> Attrs = []>
      : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
               [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
               !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_3VectorArg_Intrinsic<list<IntrinsicProperty> Attrs = []>
      : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
               [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
               !listconcat(Attrs, [IntrNoMem])>;
  class AdvSIMD_3VectorArg_Scalar_Intrinsic
      : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
               [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty],
               [IntrNoMem, ImmArg<ArgIndex<2>>]>;
  class AdvSIMD_CvtFxToFP_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [llvm_anyint_ty, llvm_i32_ty],
                [IntrNoMem, ImmArg<ArgIndex<1>>]>;
  class AdvSIMD_CvtFPToFx_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyint_ty], [llvm_anyfloat_ty, llvm_i32_ty],
                [IntrNoMem, ImmArg<ArgIndex<1>>]>;

  class AdvSIMD_1Arg_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_any_ty], [LLVMMatchType<0>], [IntrNoMem]>;

  class AdvSIMD_Dot_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<1>],
                [IntrNoMem]>;

  class AdvSIMD_FP16FML_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<1>],
                [IntrNoMem]>;

  class AdvSIMD_MatMul_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<1>],
                [IntrNoMem]>;

  class AdvSIMD_BF16FML_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_v4f32_ty],
                [llvm_v4f32_ty, llvm_v8bf16_ty, llvm_v8bf16_ty],
                [IntrNoMem]>;
}

// Arithmetic ops

let TargetPrefix = "aarch64" in {
  // Vector Add Across Lanes
  def int_aarch64_neon_saddv : AdvSIMD_1VectorArg_Int_Across_Intrinsic;
  def int_aarch64_neon_uaddv : AdvSIMD_1VectorArg_Int_Across_Intrinsic;
  def int_aarch64_neon_faddv : AdvSIMD_1VectorArg_Float_Across_Intrinsic;

  // Vector Long Add Across Lanes
  def int_aarch64_neon_saddlv : AdvSIMD_1VectorArg_Int_Across_Intrinsic;
  def int_aarch64_neon_uaddlv : AdvSIMD_1VectorArg_Int_Across_Intrinsic;

  // Vector Halving Add
  def int_aarch64_neon_shadd : AdvSIMD_2VectorArg_Intrinsic;
  def int_aarch64_neon_uhadd : AdvSIMD_2VectorArg_Intrinsic;

  // Vector Rounding Halving Add
  def int_aarch64_neon_srhadd : AdvSIMD_2VectorArg_Intrinsic;
  def int_aarch64_neon_urhadd : AdvSIMD_2VectorArg_Intrinsic;

  // Vector Saturating Add
  def int_aarch64_neon_sqadd : AdvSIMD_2IntArg_Intrinsic;
  def int_aarch64_neon_suqadd : AdvSIMD_2IntArg_Intrinsic;
  def int_aarch64_neon_usqadd : AdvSIMD_2IntArg_Intrinsic;
  def int_aarch64_neon_uqadd : AdvSIMD_2IntArg_Intrinsic;

  // Vector Add High-Half
  // FIXME: this is a legacy intrinsic for aarch64_simd.h. Remove it when that
  // header is no longer supported.
  def int_aarch64_neon_addhn : AdvSIMD_2VectorArg_Narrow_Intrinsic;

  // Vector Rounding Add High-Half
  def int_aarch64_neon_raddhn : AdvSIMD_2VectorArg_Narrow_Intrinsic;

  // Vector Saturating Doubling Multiply High
  def int_aarch64_neon_sqdmulh : AdvSIMD_2IntArg_Intrinsic;
  def int_aarch64_neon_sqdmulh_lane : AdvSIMD_2VectorArg_Lane_Intrinsic;
  def int_aarch64_neon_sqdmulh_laneq : AdvSIMD_2VectorArg_Lane_Intrinsic;

  // Vector Saturating Rounding Doubling Multiply High
  def int_aarch64_neon_sqrdmulh : AdvSIMD_2IntArg_Intrinsic;
  def int_aarch64_neon_sqrdmulh_lane : AdvSIMD_2VectorArg_Lane_Intrinsic;
  def int_aarch64_neon_sqrdmulh_laneq : AdvSIMD_2VectorArg_Lane_Intrinsic;

  def int_aarch64_neon_sqrdmlah : AdvSIMD_3IntArg_Intrinsic;
  def int_aarch64_neon_sqrdmlsh : AdvSIMD_3IntArg_Intrinsic;

  let IntrProperties = [IntrNoMem, Commutative] in {
    // Vector Polynominal Multiply
    def int_aarch64_neon_pmul : AdvSIMD_2VectorArg_Intrinsic;

    // Vector Long Multiply
    def int_aarch64_neon_smull : AdvSIMD_2VectorArg_Long_Intrinsic;
    def int_aarch64_neon_umull : AdvSIMD_2VectorArg_Long_Intrinsic;
    def int_aarch64_neon_pmull : AdvSIMD_2VectorArg_Long_Intrinsic;

    // 64-bit polynomial multiply really returns an i128, which is not legal.
    // Fake it with a v16i8.
    def int_aarch64_neon_pmull64
      : DefaultAttrsIntrinsic<[llvm_v16i8_ty], [llvm_i64_ty, llvm_i64_ty]>;

    // Vector Extending Multiply
    def int_aarch64_neon_fmulx : AdvSIMD_2FloatArg_Intrinsic;
  }

  // Vector Saturating Doubling Long Multiply
  def int_aarch64_neon_sqdmull : AdvSIMD_2VectorArg_Long_Intrinsic;
  def int_aarch64_neon_sqdmulls_scalar
    : DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

  // Vector Halving Subtract
  def int_aarch64_neon_shsub : AdvSIMD_2VectorArg_Intrinsic;
  def int_aarch64_neon_uhsub : AdvSIMD_2VectorArg_Intrinsic;

  // Vector Saturating Subtract
  def int_aarch64_neon_sqsub : AdvSIMD_2IntArg_Intrinsic;
  def int_aarch64_neon_uqsub : AdvSIMD_2IntArg_Intrinsic;

  // Vector Subtract High-Half
  // FIXME: this is a legacy intrinsic for aarch64_simd.h. Remove it when that
  // header is no longer supported.
  def int_aarch64_neon_subhn : AdvSIMD_2VectorArg_Narrow_Intrinsic;

  // Vector Rounding Subtract High-Half
  def int_aarch64_neon_rsubhn : AdvSIMD_2VectorArg_Narrow_Intrinsic;

  // Vector Compare Absolute Greater-than-or-equal
  def int_aarch64_neon_facge : AdvSIMD_2Arg_FloatCompare_Intrinsic;

  // Vector Compare Absolute Greater-than
  def int_aarch64_neon_facgt : AdvSIMD_2Arg_FloatCompare_Intrinsic;

  // Vector Absolute Difference
  def int_aarch64_neon_sabd : AdvSIMD_2VectorArg_Intrinsic;
  def int_aarch64_neon_uabd : AdvSIMD_2VectorArg_Intrinsic;
  def int_aarch64_neon_fabd : AdvSIMD_2VectorArg_Intrinsic;

  // Scalar Absolute Difference
  def int_aarch64_sisd_fabd : AdvSIMD_2Scalar_Float_Intrinsic;

  // Vector Max
  def int_aarch64_neon_fmax : AdvSIMD_2FloatArg_Intrinsic;
  def int_aarch64_neon_fmaxnmp : AdvSIMD_2VectorArg_Intrinsic;

  // Vector Max Across Lanes
  def int_aarch64_neon_smaxv : AdvSIMD_1VectorArg_Int_Across_Intrinsic;
  def int_aarch64_neon_umaxv : AdvSIMD_1VectorArg_Int_Across_Intrinsic;
  def int_aarch64_neon_fmaxv : AdvSIMD_1VectorArg_Float_Across_Intrinsic;
  def int_aarch64_neon_fmaxnmv : AdvSIMD_1VectorArg_Float_Across_Intrinsic;

  // Vector Min
  def int_aarch64_neon_fmin : AdvSIMD_2FloatArg_Intrinsic;
  def int_aarch64_neon_fminnmp : AdvSIMD_2VectorArg_Intrinsic;

  // Vector Min/Max Number
  def int_aarch64_neon_fminnm : AdvSIMD_2FloatArg_Intrinsic;
  def int_aarch64_neon_fmaxnm : AdvSIMD_2FloatArg_Intrinsic;

  // Vector Min Across Lanes
  def int_aarch64_neon_sminv : AdvSIMD_1VectorArg_Int_Across_Intrinsic;
  def int_aarch64_neon_uminv : AdvSIMD_1VectorArg_Int_Across_Intrinsic;
  def int_aarch64_neon_fminv : AdvSIMD_1VectorArg_Float_Across_Intrinsic;
  def int_aarch64_neon_fminnmv : AdvSIMD_1VectorArg_Float_Across_Intrinsic;

  // Pairwise Add
  def int_aarch64_neon_addp : AdvSIMD_2VectorArg_Intrinsic;
  def int_aarch64_neon_faddp : AdvSIMD_2VectorArg_Intrinsic;

  // Long Pairwise Add
  // FIXME: In theory, we shouldn't need intrinsics for saddlp or
  // uaddlp, but tblgen's type inference currently can't handle the
  // pattern fragments this ends up generating.
  def int_aarch64_neon_saddlp : AdvSIMD_1VectorArg_Expand_Intrinsic;
  def int_aarch64_neon_uaddlp : AdvSIMD_1VectorArg_Expand_Intrinsic;

  // Folding Maximum
  def int_aarch64_neon_smaxp : AdvSIMD_2VectorArg_Intrinsic;
  def int_aarch64_neon_umaxp : AdvSIMD_2VectorArg_Intrinsic;
  def int_aarch64_neon_fmaxp : AdvSIMD_2VectorArg_Intrinsic;

  // Folding Minimum
  def int_aarch64_neon_sminp : AdvSIMD_2VectorArg_Intrinsic;
  def int_aarch64_neon_uminp : AdvSIMD_2VectorArg_Intrinsic;
  def int_aarch64_neon_fminp : AdvSIMD_2VectorArg_Intrinsic;

  // Reciprocal Estimate/Step
  def int_aarch64_neon_frecps : AdvSIMD_2FloatArg_Intrinsic;
  def int_aarch64_neon_frsqrts : AdvSIMD_2FloatArg_Intrinsic;

  // Reciprocal Exponent
  def int_aarch64_neon_frecpx : AdvSIMD_1FloatArg_Intrinsic;

  // Vector Saturating Shift Left
  def int_aarch64_neon_sqshl : AdvSIMD_2IntArg_Intrinsic;
  def int_aarch64_neon_uqshl : AdvSIMD_2IntArg_Intrinsic;

  // Vector Rounding Shift Left
  def int_aarch64_neon_srshl : AdvSIMD_2IntArg_Intrinsic;
  def int_aarch64_neon_urshl : AdvSIMD_2IntArg_Intrinsic;

  // Vector Saturating Rounding Shift Left
  def int_aarch64_neon_sqrshl : AdvSIMD_2IntArg_Intrinsic;
  def int_aarch64_neon_uqrshl : AdvSIMD_2IntArg_Intrinsic;

  // Vector Signed->Unsigned Shift Left by Constant
  def int_aarch64_neon_sqshlu : AdvSIMD_2IntArg_Intrinsic;

  // Vector Signed->Unsigned Narrowing Saturating Shift Right by Constant
  def int_aarch64_neon_sqshrun : AdvSIMD_2Arg_Scalar_Narrow_Intrinsic;

  // Vector Signed->Unsigned Rounding Narrowing Saturating Shift Right by Const
  def int_aarch64_neon_sqrshrun : AdvSIMD_2Arg_Scalar_Narrow_Intrinsic;

  // Vector Narrowing Shift Right by Constant
  def int_aarch64_neon_sqshrn : AdvSIMD_2Arg_Scalar_Narrow_Intrinsic;
  def int_aarch64_neon_uqshrn : AdvSIMD_2Arg_Scalar_Narrow_Intrinsic;

  // Vector Rounding Narrowing Shift Right by Constant
  def int_aarch64_neon_rshrn : AdvSIMD_2Arg_Scalar_Narrow_Intrinsic;

  // Vector Rounding Narrowing Saturating Shift Right by Constant
  def int_aarch64_neon_sqrshrn : AdvSIMD_2Arg_Scalar_Narrow_Intrinsic;
  def int_aarch64_neon_uqrshrn : AdvSIMD_2Arg_Scalar_Narrow_Intrinsic;

  // Vector Shift Left
  def int_aarch64_neon_sshl : AdvSIMD_2IntArg_Intrinsic;
  def int_aarch64_neon_ushl : AdvSIMD_2IntArg_Intrinsic;

  // Vector Widening Shift Left by Constant
  def int_aarch64_neon_shll : AdvSIMD_2VectorArg_Scalar_Wide_BySize_Intrinsic;
  def int_aarch64_neon_sshll : AdvSIMD_2VectorArg_Scalar_Wide_Intrinsic;
  def int_aarch64_neon_ushll : AdvSIMD_2VectorArg_Scalar_Wide_Intrinsic;

  // Vector Shift Right by Constant and Insert
  def int_aarch64_neon_vsri : AdvSIMD_3VectorArg_Scalar_Intrinsic;

  // Vector Shift Left by Constant and Insert
  def int_aarch64_neon_vsli : AdvSIMD_3VectorArg_Scalar_Intrinsic;

  // Vector Saturating Narrow
  def int_aarch64_neon_scalar_sqxtn: AdvSIMD_1IntArg_Narrow_Intrinsic;
  def int_aarch64_neon_scalar_uqxtn : AdvSIMD_1IntArg_Narrow_Intrinsic;
  def int_aarch64_neon_sqxtn : AdvSIMD_1VectorArg_Narrow_Intrinsic;
  def int_aarch64_neon_uqxtn : AdvSIMD_1VectorArg_Narrow_Intrinsic;

  // Vector Saturating Extract and Unsigned Narrow
  def int_aarch64_neon_scalar_sqxtun : AdvSIMD_1IntArg_Narrow_Intrinsic;
  def int_aarch64_neon_sqxtun : AdvSIMD_1VectorArg_Narrow_Intrinsic;

  // Vector Absolute Value
  def int_aarch64_neon_abs : AdvSIMD_1Arg_Intrinsic;

  // Vector Saturating Absolute Value
  def int_aarch64_neon_sqabs : AdvSIMD_1IntArg_Intrinsic;

  // Vector Saturating Negation
  def int_aarch64_neon_sqneg : AdvSIMD_1IntArg_Intrinsic;

  // Vector Count Leading Sign Bits
  def int_aarch64_neon_cls : AdvSIMD_1VectorArg_Intrinsic;

  // Vector Reciprocal Estimate
  def int_aarch64_neon_urecpe : AdvSIMD_1VectorArg_Intrinsic;
  def int_aarch64_neon_frecpe : AdvSIMD_1FloatArg_Intrinsic;

  // Vector Square Root Estimate
  def int_aarch64_neon_ursqrte : AdvSIMD_1VectorArg_Intrinsic;
  def int_aarch64_neon_frsqrte : AdvSIMD_1FloatArg_Intrinsic;

  // Vector Conversions Between Floating-point and Fixed-point.
  def int_aarch64_neon_vcvtfp2fxs : AdvSIMD_CvtFPToFx_Intrinsic;
  def int_aarch64_neon_vcvtfp2fxu : AdvSIMD_CvtFPToFx_Intrinsic;
  def int_aarch64_neon_vcvtfxs2fp : AdvSIMD_CvtFxToFP_Intrinsic;
  def int_aarch64_neon_vcvtfxu2fp : AdvSIMD_CvtFxToFP_Intrinsic;

  // Vector FP->Int Conversions
  def int_aarch64_neon_fcvtas : AdvSIMD_FPToIntRounding_Intrinsic;
  def int_aarch64_neon_fcvtau : AdvSIMD_FPToIntRounding_Intrinsic;
  def int_aarch64_neon_fcvtms : AdvSIMD_FPToIntRounding_Intrinsic;
  def int_aarch64_neon_fcvtmu : AdvSIMD_FPToIntRounding_Intrinsic;
  def int_aarch64_neon_fcvtns : AdvSIMD_FPToIntRounding_Intrinsic;
  def int_aarch64_neon_fcvtnu : AdvSIMD_FPToIntRounding_Intrinsic;
  def int_aarch64_neon_fcvtps : AdvSIMD_FPToIntRounding_Intrinsic;
  def int_aarch64_neon_fcvtpu : AdvSIMD_FPToIntRounding_Intrinsic;
  def int_aarch64_neon_fcvtzs : AdvSIMD_FPToIntRounding_Intrinsic;
  def int_aarch64_neon_fcvtzu : AdvSIMD_FPToIntRounding_Intrinsic;

  // v8.5-A Vector FP Rounding
  def int_aarch64_neon_frint32x : AdvSIMD_1FloatArg_Intrinsic;
  def int_aarch64_neon_frint32z : AdvSIMD_1FloatArg_Intrinsic;
  def int_aarch64_neon_frint64x : AdvSIMD_1FloatArg_Intrinsic;
  def int_aarch64_neon_frint64z : AdvSIMD_1FloatArg_Intrinsic;

  // Scalar FP->Int conversions

  // Vector FP Inexact Narrowing
  def int_aarch64_neon_fcvtxn : AdvSIMD_1VectorArg_Expand_Intrinsic;

  // Scalar FP Inexact Narrowing
  def int_aarch64_sisd_fcvtxn : DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_double_ty],
                                        [IntrNoMem]>;

  // v8.2-A Dot Product
  def int_aarch64_neon_udot : AdvSIMD_Dot_Intrinsic;
  def int_aarch64_neon_sdot : AdvSIMD_Dot_Intrinsic;

  // v8.6-A Matrix Multiply Intrinsics
  def int_aarch64_neon_ummla : AdvSIMD_MatMul_Intrinsic;
  def int_aarch64_neon_smmla : AdvSIMD_MatMul_Intrinsic;
  def int_aarch64_neon_usmmla : AdvSIMD_MatMul_Intrinsic;
  def int_aarch64_neon_fmmla : AdvSIMD_MatMul_Intrinsic;
  def int_aarch64_neon_usdot : AdvSIMD_Dot_Intrinsic;
  def int_aarch64_neon_bfdot : AdvSIMD_Dot_Intrinsic;
  def int_aarch64_neon_fdot : AdvSIMD_Dot_Intrinsic;
  def int_aarch64_neon_bfmmla
    : DefaultAttrsIntrinsic<[llvm_v4f32_ty],
                [llvm_v4f32_ty, llvm_v8bf16_ty, llvm_v8bf16_ty],
                [IntrNoMem]>;
  def int_aarch64_neon_bfmlalb : AdvSIMD_BF16FML_Intrinsic;
  def int_aarch64_neon_bfmlalt : AdvSIMD_BF16FML_Intrinsic;

  // v8.2-A FP16 Fused Multiply-Add Long
  def int_aarch64_neon_fmlal : AdvSIMD_FP16FML_Intrinsic;
  def int_aarch64_neon_fmlsl : AdvSIMD_FP16FML_Intrinsic;
  def int_aarch64_neon_fmlal2 : AdvSIMD_FP16FML_Intrinsic;
  def int_aarch64_neon_fmlsl2 : AdvSIMD_FP16FML_Intrinsic;

  // v8.3-A Floating-point complex add
  def int_aarch64_neon_vcadd_rot90  : AdvSIMD_2VectorArg_Intrinsic;
  def int_aarch64_neon_vcadd_rot270 : AdvSIMD_2VectorArg_Intrinsic;

  def int_aarch64_neon_vcmla_rot0   : AdvSIMD_3VectorArg_Intrinsic;
  def int_aarch64_neon_vcmla_rot90  : AdvSIMD_3VectorArg_Intrinsic;
  def int_aarch64_neon_vcmla_rot180 : AdvSIMD_3VectorArg_Intrinsic;
  def int_aarch64_neon_vcmla_rot270 : AdvSIMD_3VectorArg_Intrinsic;

  // FP8 fscale
  def int_aarch64_neon_fp8_fscale : DefaultAttrsIntrinsic<
                                    [llvm_anyvector_ty],
                                    [LLVMMatchType<0>,
                                    LLVMVectorOfBitcastsToInt<0>],
                                    [IntrNoMem]>;
}

let TargetPrefix = "aarch64" in {
def int_aarch64_neon_vluti2_lane : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                       [llvm_anyvector_ty, llvm_v8i8_ty,
                                        llvm_i32_ty],
                                       [IntrNoMem, ImmArg<ArgIndex<2>>]>;

def int_aarch64_neon_vluti2_laneq : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                       [llvm_anyvector_ty, llvm_v16i8_ty,
                                        llvm_i32_ty],
                                       [IntrNoMem, ImmArg<ArgIndex<2>>]>;

def int_aarch64_neon_vluti4q_lane: DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                       [LLVMMatchType<0>, llvm_v8i8_ty,
                                        llvm_i32_ty],
                                       [IntrNoMem, ImmArg<ArgIndex<2>>]>;

def int_aarch64_neon_vluti4q_laneq: DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                       [LLVMMatchType<0>, llvm_v16i8_ty,
                                        llvm_i32_ty],
                                       [IntrNoMem, ImmArg<ArgIndex<2>>]>;


def int_aarch64_neon_vluti4q_lane_x2:
    DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                          [LLVMMatchType<0>, LLVMMatchType<0>,
                           llvm_v8i8_ty, llvm_i32_ty],
                          [IntrNoMem, ImmArg<ArgIndex<3>>]>;

def int_aarch64_neon_vluti4q_laneq_x2:
    DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                          [LLVMMatchType<0>, LLVMMatchType<0>,
                           llvm_v16i8_ty, llvm_i32_ty],
                          [IntrNoMem, ImmArg<ArgIndex<3>>]>;
}

let TargetPrefix = "aarch64" in {  // All intrinsics start with "llvm.aarch64.".
  class AdvSIMD_2Vector2Index_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                [llvm_anyvector_ty, llvm_i64_ty, LLVMMatchType<0>, llvm_i64_ty],
                [IntrNoMem]>;
}

// Vector element to element moves
def int_aarch64_neon_vcopy_lane: AdvSIMD_2Vector2Index_Intrinsic;

let TargetPrefix = "aarch64" in {  // All intrinsics start with "llvm.aarch64.".
  class AdvSIMD_1Vec_Load_Intrinsic
      : DefaultAttrsIntrinsic<[llvm_anyvector_ty], [llvm_anyptr_ty],
                  [IntrReadMem, IntrArgMemOnly]>;
  class AdvSIMD_1Vec_Store_Lane_Intrinsic
    : DefaultAttrsIntrinsic<[], [llvm_anyvector_ty, llvm_i64_ty, llvm_anyptr_ty],
                [IntrWriteMem, IntrArgMemOnly, NoCapture<ArgIndex<2>>]>;

  class AdvSIMD_2Vec_Load_Intrinsic
    : DefaultAttrsIntrinsic<[LLVMMatchType<0>, llvm_anyvector_ty],
                [llvm_anyptr_ty],
                [IntrReadMem, IntrArgMemOnly]>;
  class AdvSIMD_2Vec_Load_Lane_Intrinsic
    : DefaultAttrsIntrinsic<[LLVMMatchType<0>, LLVMMatchType<0>],
                [LLVMMatchType<0>, llvm_anyvector_ty,
                 llvm_i64_ty, llvm_anyptr_ty],
                [IntrReadMem, IntrArgMemOnly]>;
  class AdvSIMD_2Vec_Store_Intrinsic
    : DefaultAttrsIntrinsic<[], [llvm_anyvector_ty, LLVMMatchType<0>,
                     llvm_anyptr_ty],
                [IntrWriteMem, IntrArgMemOnly, NoCapture<ArgIndex<2>>]>;
  class AdvSIMD_2Vec_Store_Lane_Intrinsic
    : DefaultAttrsIntrinsic<[], [llvm_anyvector_ty, LLVMMatchType<0>,
                 llvm_i64_ty, llvm_anyptr_ty],
                [IntrWriteMem, IntrArgMemOnly, NoCapture<ArgIndex<3>>]>;

  class AdvSIMD_3Vec_Load_Intrinsic
    : DefaultAttrsIntrinsic<[LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty],
                [llvm_anyptr_ty],
                [IntrReadMem, IntrArgMemOnly]>;
  class AdvSIMD_3Vec_Load_Lane_Intrinsic
    : DefaultAttrsIntrinsic<[LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyvector_ty,
                 llvm_i64_ty, llvm_anyptr_ty],
                [IntrReadMem, IntrArgMemOnly]>;
  class AdvSIMD_3Vec_Store_Intrinsic
    : DefaultAttrsIntrinsic<[], [llvm_anyvector_ty, LLVMMatchType<0>,
                     LLVMMatchType<0>, llvm_anyptr_ty],
                [IntrWriteMem, IntrArgMemOnly, NoCapture<ArgIndex<3>>]>;
  class AdvSIMD_3Vec_Store_Lane_Intrinsic
    : DefaultAttrsIntrinsic<[], [llvm_anyvector_ty,
                 LLVMMatchType<0>, LLVMMatchType<0>,
                 llvm_i64_ty, llvm_anyptr_ty],
                [IntrWriteMem, IntrArgMemOnly, NoCapture<ArgIndex<4>>]>;

  class AdvSIMD_4Vec_Load_Intrinsic
    : DefaultAttrsIntrinsic<[LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>, llvm_anyvector_ty],
                [llvm_anyptr_ty],
                [IntrReadMem, IntrArgMemOnly]>;
  class AdvSIMD_4Vec_Load_Lane_Intrinsic
    : DefaultAttrsIntrinsic<[LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>, LLVMMatchType<0>],
                [LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>, llvm_anyvector_ty,
                 llvm_i64_ty, llvm_anyptr_ty],
                [IntrReadMem, IntrArgMemOnly]>;
  class AdvSIMD_4Vec_Store_Intrinsic
    : DefaultAttrsIntrinsic<[], [llvm_anyvector_ty, LLVMMatchType<0>,
                 LLVMMatchType<0>, LLVMMatchType<0>,
                 llvm_anyptr_ty],
                [IntrWriteMem, IntrArgMemOnly, NoCapture<ArgIndex<4>>]>;
  class AdvSIMD_4Vec_Store_Lane_Intrinsic
    : DefaultAttrsIntrinsic<[], [llvm_anyvector_ty, LLVMMatchType<0>,
                 LLVMMatchType<0>, LLVMMatchType<0>,
                 llvm_i64_ty, llvm_anyptr_ty],
                [IntrWriteMem, IntrArgMemOnly, NoCapture<ArgIndex<5>>]>;
}

// Memory ops

def int_aarch64_neon_ld1x2 : AdvSIMD_2Vec_Load_Intrinsic;
def int_aarch64_neon_ld1x3 : AdvSIMD_3Vec_Load_Intrinsic;
def int_aarch64_neon_ld1x4 : AdvSIMD_4Vec_Load_Intrinsic;

def int_aarch64_neon_st1x2 : AdvSIMD_2Vec_Store_Intrinsic;
def int_aarch64_neon_st1x3 : AdvSIMD_3Vec_Store_Intrinsic;
def int_aarch64_neon_st1x4 : AdvSIMD_4Vec_Store_Intrinsic;

def int_aarch64_neon_ld2 : AdvSIMD_2Vec_Load_Intrinsic;
def int_aarch64_neon_ld3 : AdvSIMD_3Vec_Load_Intrinsic;
def int_aarch64_neon_ld4 : AdvSIMD_4Vec_Load_Intrinsic;

def int_aarch64_neon_ld2lane : AdvSIMD_2Vec_Load_Lane_Intrinsic;
def int_aarch64_neon_ld3lane : AdvSIMD_3Vec_Load_Lane_Intrinsic;
def int_aarch64_neon_ld4lane : AdvSIMD_4Vec_Load_Lane_Intrinsic;

def int_aarch64_neon_ld2r : AdvSIMD_2Vec_Load_Intrinsic;
def int_aarch64_neon_ld3r : AdvSIMD_3Vec_Load_Intrinsic;
def int_aarch64_neon_ld4r : AdvSIMD_4Vec_Load_Intrinsic;

def int_aarch64_neon_st2  : AdvSIMD_2Vec_Store_Intrinsic;
def int_aarch64_neon_st3  : AdvSIMD_3Vec_Store_Intrinsic;
def int_aarch64_neon_st4  : AdvSIMD_4Vec_Store_Intrinsic;

def int_aarch64_neon_st2lane  : AdvSIMD_2Vec_Store_Lane_Intrinsic;
def int_aarch64_neon_st3lane  : AdvSIMD_3Vec_Store_Lane_Intrinsic;
def int_aarch64_neon_st4lane  : AdvSIMD_4Vec_Store_Lane_Intrinsic;

let TargetPrefix = "aarch64" in {  // All intrinsics start with "llvm.aarch64.".
  class AdvSIMD_Tbl1_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty], [llvm_v16i8_ty, LLVMMatchType<0>],
                [IntrNoMem]>;
  class AdvSIMD_Tbl2_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                [llvm_v16i8_ty, llvm_v16i8_ty, LLVMMatchType<0>], [IntrNoMem]>;
  class AdvSIMD_Tbl3_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,
                 LLVMMatchType<0>],
                [IntrNoMem]>;
  class AdvSIMD_Tbl4_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,
                 LLVMMatchType<0>],
                [IntrNoMem]>;

  class AdvSIMD_Tbx1_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>, llvm_v16i8_ty, LLVMMatchType<0>],
                [IntrNoMem]>;
  class AdvSIMD_Tbx2_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>, llvm_v16i8_ty, llvm_v16i8_ty,
                 LLVMMatchType<0>],
                [IntrNoMem]>;
  class AdvSIMD_Tbx3_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>, llvm_v16i8_ty, llvm_v16i8_ty,
                 llvm_v16i8_ty, LLVMMatchType<0>],
                [IntrNoMem]>;
  class AdvSIMD_Tbx4_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>, llvm_v16i8_ty, llvm_v16i8_ty,
                 llvm_v16i8_ty, llvm_v16i8_ty, LLVMMatchType<0>],
                [IntrNoMem]>;
}
def int_aarch64_neon_tbl1 : AdvSIMD_Tbl1_Intrinsic;
def int_aarch64_neon_tbl2 : AdvSIMD_Tbl2_Intrinsic;
def int_aarch64_neon_tbl3 : AdvSIMD_Tbl3_Intrinsic;
def int_aarch64_neon_tbl4 : AdvSIMD_Tbl4_Intrinsic;

def int_aarch64_neon_tbx1 : AdvSIMD_Tbx1_Intrinsic;
def int_aarch64_neon_tbx2 : AdvSIMD_Tbx2_Intrinsic;
def int_aarch64_neon_tbx3 : AdvSIMD_Tbx3_Intrinsic;
def int_aarch64_neon_tbx4 : AdvSIMD_Tbx4_Intrinsic;

// Maps Memory locations to registers.
defvar FFR = InaccessibleMem;
defvar FPMR = InaccessibleMem;
defvar ZT0 = TargetMem0;
defvar ZA = TargetMem1;

let TargetPrefix = "aarch64" in {
  class FPENV_Get_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_i64_ty], [], [IntrNoMem, IntrHasSideEffects]>;
  class FPENV_Set_Intrinsic
    : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrNoMem, IntrHasSideEffects]>;
  class RNDR_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_i64_ty, llvm_i1_ty], [], [IntrNoMem, IntrHasSideEffects]>;
  class FPMR_Set_Intrinsic
    : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrWrite<[FPMR]>, IntrWriteOnly]>;
}

// FP environment registers.
def int_aarch64_get_fpcr : FPENV_Get_Intrinsic;
def int_aarch64_set_fpcr : FPENV_Set_Intrinsic;
def int_aarch64_get_fpsr : FPENV_Get_Intrinsic;
def int_aarch64_set_fpsr : FPENV_Set_Intrinsic;
def int_aarch64_set_fpmr : FPMR_Set_Intrinsic;

// Armv8.5-A Random number generation intrinsics
def int_aarch64_rndr : RNDR_Intrinsic;
def int_aarch64_rndrrs : RNDR_Intrinsic;

let TargetPrefix = "aarch64" in {
  class Crypto_AES_DataKey_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty], [IntrNoMem]>;

  class Crypto_AES_Data_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty], [IntrNoMem]>;

  // SHA intrinsic taking 5 words of the hash (v4i32, i32) and 4 of the schedule
  // (v4i32).
  class Crypto_SHA_5Hash4Schedule_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_i32_ty, llvm_v4i32_ty],
                [IntrNoMem]>;

  // SHA intrinsic taking 5 words of the hash (v4i32, i32) and 4 of the schedule
  // (v4i32).
  class Crypto_SHA_1Hash_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;

  // SHA intrinsic taking 8 words of the schedule
  class Crypto_SHA_8Schedule_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem]>;

  // SHA intrinsic taking 12 words of the schedule
  class Crypto_SHA_12Schedule_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                [IntrNoMem]>;

  // SHA intrinsic taking 8 words of the hash and 4 of the schedule.
  class Crypto_SHA_8Hash4Schedule_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                [IntrNoMem]>;

  // SHA512 intrinsic taking 2 arguments
  class Crypto_SHA512_2Arg_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty], [IntrNoMem]>;

  // SHA512 intrinsic taking 3 Arguments
  class Crypto_SHA512_3Arg_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty, llvm_v2i64_ty],
                [IntrNoMem]>;

  // SHA3 Intrinsics taking 3 arguments
  class Crypto_SHA3_3Arg_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
               [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
               [IntrNoMem]>;

  // SHA3 Intrinsic taking 2 arguments
  class Crypto_SHA3_2Arg_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
               [IntrNoMem]>;

  // SHA3 Intrinsic taking 3 Arguments 1 immediate
  class Crypto_SHA3_2ArgImm_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty, llvm_i64_ty],
               [IntrNoMem, ImmArg<ArgIndex<2>>]>;

  class Crypto_SM3_3Vector_Intrinsic
    : Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                [IntrNoMem]>;

  class Crypto_SM3_3VectorIndexed_Intrinsic
    : Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty, llvm_i64_ty],
                [IntrNoMem, ImmArg<ArgIndex<3>>]>;

  class Crypto_SM4_2Vector_Intrinsic
    : Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem]>;
}

// AES
def int_aarch64_crypto_aese   : Crypto_AES_DataKey_Intrinsic;
def int_aarch64_crypto_aesd   : Crypto_AES_DataKey_Intrinsic;
def int_aarch64_crypto_aesmc  : Crypto_AES_Data_Intrinsic;
def int_aarch64_crypto_aesimc : Crypto_AES_Data_Intrinsic;

// SHA1
def int_aarch64_crypto_sha1c  : Crypto_SHA_5Hash4Schedule_Intrinsic;
def int_aarch64_crypto_sha1p  : Crypto_SHA_5Hash4Schedule_Intrinsic;
def int_aarch64_crypto_sha1m  : Crypto_SHA_5Hash4Schedule_Intrinsic;
def int_aarch64_crypto_sha1h  : Crypto_SHA_1Hash_Intrinsic;

def int_aarch64_crypto_sha1su0 : Crypto_SHA_12Schedule_Intrinsic;
def int_aarch64_crypto_sha1su1 : Crypto_SHA_8Schedule_Intrinsic;

// SHA256
def int_aarch64_crypto_sha256h   : Crypto_SHA_8Hash4Schedule_Intrinsic;
def int_aarch64_crypto_sha256h2  : Crypto_SHA_8Hash4Schedule_Intrinsic;
def int_aarch64_crypto_sha256su0 : Crypto_SHA_8Schedule_Intrinsic;
def int_aarch64_crypto_sha256su1 : Crypto_SHA_12Schedule_Intrinsic;

//SHA3
def int_aarch64_crypto_eor3s : Crypto_SHA3_3Arg_Intrinsic;
def int_aarch64_crypto_eor3u : Crypto_SHA3_3Arg_Intrinsic;
def int_aarch64_crypto_bcaxs : Crypto_SHA3_3Arg_Intrinsic;
def int_aarch64_crypto_bcaxu : Crypto_SHA3_3Arg_Intrinsic;
def int_aarch64_crypto_rax1 : Crypto_SHA3_2Arg_Intrinsic;
def int_aarch64_crypto_xar : Crypto_SHA3_2ArgImm_Intrinsic;

// SHA512
def int_aarch64_crypto_sha512h : Crypto_SHA512_3Arg_Intrinsic;
def int_aarch64_crypto_sha512h2 : Crypto_SHA512_3Arg_Intrinsic;
def int_aarch64_crypto_sha512su0 : Crypto_SHA512_2Arg_Intrinsic;
def int_aarch64_crypto_sha512su1 : Crypto_SHA512_3Arg_Intrinsic;

//SM3 & SM4
def int_aarch64_crypto_sm3partw1 : Crypto_SM3_3Vector_Intrinsic;
def int_aarch64_crypto_sm3partw2 : Crypto_SM3_3Vector_Intrinsic;
def int_aarch64_crypto_sm3ss1    : Crypto_SM3_3Vector_Intrinsic;
def int_aarch64_crypto_sm3tt1a   : Crypto_SM3_3VectorIndexed_Intrinsic;
def int_aarch64_crypto_sm3tt1b   : Crypto_SM3_3VectorIndexed_Intrinsic;
def int_aarch64_crypto_sm3tt2a   : Crypto_SM3_3VectorIndexed_Intrinsic;
def int_aarch64_crypto_sm3tt2b   : Crypto_SM3_3VectorIndexed_Intrinsic;
def int_aarch64_crypto_sm4e      : Crypto_SM4_2Vector_Intrinsic;
def int_aarch64_crypto_sm4ekey   : Crypto_SM4_2Vector_Intrinsic;

//===----------------------------------------------------------------------===//
// CRC32

let TargetPrefix = "aarch64" in {

def int_aarch64_crc32b  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_aarch64_crc32cb : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_aarch64_crc32h  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_aarch64_crc32ch : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_aarch64_crc32w  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_aarch64_crc32cw : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem]>;
def int_aarch64_crc32x  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i64_ty],
    [IntrNoMem]>;
def int_aarch64_crc32cx : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i64_ty],
    [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// Memory Tagging Extensions (MTE) Intrinsics
let TargetPrefix = "aarch64" in {
def int_aarch64_irg   : DefaultAttrsIntrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i64_ty],
    [IntrNoMem, IntrHasSideEffects]>;
def int_aarch64_addg  : DefaultAttrsIntrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i64_ty],
    [IntrNoMem]>;
def int_aarch64_gmi   : DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_ptr_ty, llvm_i64_ty],
    [IntrNoMem]>;
def int_aarch64_ldg   : DefaultAttrsIntrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_ptr_ty],
    [IntrReadMem]>;
def int_aarch64_stg   : DefaultAttrsIntrinsic<[], [llvm_ptr_ty, llvm_ptr_ty],
    [IntrWriteMem]>;
def int_aarch64_subp :  DefaultAttrsIntrinsic<[llvm_i64_ty], [llvm_ptr_ty, llvm_ptr_ty],
    [IntrNoMem]>;

// The following are codegen-only intrinsics for stack instrumentation.

// Generate a randomly tagged stack base pointer.
def int_aarch64_irg_sp   : DefaultAttrsIntrinsic<[llvm_ptr_ty], [llvm_i64_ty],
    [IntrNoMem, IntrHasSideEffects]>;

// Transfer pointer tag with offset.
// ptr1 = tagp(ptr0, baseptr, tag_offset) returns a pointer where
// * address is the address in ptr0
// * tag is a function of (tag in baseptr, tag_offset).
// ** Beware, this is not the same function as implemented by the ADDG instruction!
//    Backend optimizations may change tag_offset; the only guarantee is that calls
//    to tagp with the same pair of (baseptr, tag_offset) will produce pointers
//    with the same tag value, assuming the set of excluded tags has not changed.
// Address bits in baseptr and tag bits in ptr0 are ignored.
// When offset between ptr0 and baseptr is a compile time constant, this can be emitted as
//   ADDG ptr1, baseptr, (ptr0 - baseptr), tag_offset
// It is intended that ptr0 is an alloca address, and baseptr is the direct output of llvm.aarch64.irg.sp.
def int_aarch64_tagp : DefaultAttrsIntrinsic<[llvm_anyptr_ty], [LLVMMatchType<0>, llvm_ptr_ty, llvm_i64_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// Update allocation tags for the memory range to match the tag in the pointer argument.
def int_aarch64_settag  : DefaultAttrsIntrinsic<[], [llvm_ptr_ty, llvm_i64_ty],
    [IntrWriteMem, IntrArgMemOnly, NoCapture<ArgIndex<0>>, WriteOnly<ArgIndex<0>>]>;

// Update allocation tags for the memory range to match the tag in the pointer argument,
// and set memory contents to zero.
def int_aarch64_settag_zero  : DefaultAttrsIntrinsic<[], [llvm_ptr_ty, llvm_i64_ty],
    [IntrWriteMem, IntrArgMemOnly, NoCapture<ArgIndex<0>>, WriteOnly<ArgIndex<0>>]>;

// Update allocation tags for 16-aligned, 16-sized memory region, and store a pair 8-byte values.
def int_aarch64_stgp  : DefaultAttrsIntrinsic<[], [llvm_ptr_ty, llvm_i64_ty, llvm_i64_ty],
    [IntrWriteMem, IntrArgMemOnly, NoCapture<ArgIndex<0>>, WriteOnly<ArgIndex<0>>]>;
}

//===----------------------------------------------------------------------===//
// Memory Operations (MOPS) Intrinsics
let TargetPrefix = "aarch64" in {
  // Sizes are chosen to correspond to the llvm.memset intrinsic: ptr, i8, i64
  def int_aarch64_mops_memset_tag : DefaultAttrsIntrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i8_ty, llvm_i64_ty],
      [IntrWriteMem, IntrArgMemOnly, NoCapture<ArgIndex<0>>, WriteOnly<ArgIndex<0>>]>;
}

let TargetPrefix = "aarch64" in {
// Armv8.7-A load/store 64-byte intrinsics
defvar data512 = !listsplat(llvm_i64_ty, 8);
def int_aarch64_ld64b: Intrinsic<data512, [llvm_ptr_ty]>;
def int_aarch64_st64b: Intrinsic<[], !listconcat([llvm_ptr_ty], data512)>;
def int_aarch64_st64bv: Intrinsic<[llvm_i64_ty], !listconcat([llvm_ptr_ty], data512)>;
def int_aarch64_st64bv0: Intrinsic<[llvm_i64_ty], !listconcat([llvm_ptr_ty], data512)>;

  //
  // Neon FP8 intrinsics
  //

  // Conversions
  class AdvSIMD_FP8_1VectorArg_Long_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty],
                            [IntrRead<[FPMR]>, IntrReadOnly]>;

  def int_aarch64_neon_fp8_cvtl1   : AdvSIMD_FP8_1VectorArg_Long_Intrinsic;
  def int_aarch64_neon_fp8_cvtl2   : AdvSIMD_FP8_1VectorArg_Long_Intrinsic;

  def int_aarch64_neon_fp8_fcvtn
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                            [llvm_anyvector_ty,
                             LLVMMatchType<1>],
                            [IntrRead<[FPMR]>, IntrReadOnly]>;
  def int_aarch64_neon_fp8_fcvtn2
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                            [LLVMMatchType<0>,
                             llvm_anyvector_ty,
                             LLVMMatchType<1>],
                            [IntrRead<[FPMR]>, IntrReadOnly]>;

  // Dot-product
  class AdvSIMD_FP8_DOT_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                            [LLVMMatchType<0>,
                             llvm_anyvector_ty,
                             LLVMMatchType<1>],
                            [IntrRead<[FPMR]>, IntrReadOnly]>;
  class AdvSIMD_FP8_DOT_LANE_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                            [LLVMMatchType<0>,
                             llvm_anyvector_ty,
                             llvm_v16i8_ty,
                             llvm_i32_ty],
                             [IntrRead<[FPMR]>, IntrReadOnly, ImmArg<ArgIndex<3>>]>;

  def int_aarch64_neon_fp8_fdot2 : AdvSIMD_FP8_DOT_Intrinsic;
  def int_aarch64_neon_fp8_fdot2_lane : AdvSIMD_FP8_DOT_LANE_Intrinsic;

  def int_aarch64_neon_fp8_fdot4 : AdvSIMD_FP8_DOT_Intrinsic;
  def int_aarch64_neon_fp8_fdot4_lane : AdvSIMD_FP8_DOT_LANE_Intrinsic;


// Fused multiply-add
  class AdvSIMD_FP8_FMLA_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                            [LLVMMatchType<0>,
                             llvm_v16i8_ty,
                             llvm_v16i8_ty],
                            [IntrRead<[FPMR]>, IntrReadOnly]>;

  class AdvSIMD_FP8_FMLA_LANE_Intrinsic
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                            [LLVMMatchType<0>,
                             llvm_v16i8_ty,
                             llvm_v16i8_ty,
                             llvm_i32_ty],
                             [IntrRead<[FPMR]>, IntrReadOnly, ImmArg<ArgIndex<3>>]>;

  def int_aarch64_neon_fp8_fmlalb : AdvSIMD_FP8_FMLA_Intrinsic;
  def int_aarch64_neon_fp8_fmlalt : AdvSIMD_FP8_FMLA_Intrinsic;

  def int_aarch64_neon_fp8_fmlallbb : AdvSIMD_FP8_FMLA_Intrinsic;
  def int_aarch64_neon_fp8_fmlallbt : AdvSIMD_FP8_FMLA_Intrinsic;
  def int_aarch64_neon_fp8_fmlalltb : AdvSIMD_FP8_FMLA_Intrinsic;
  def int_aarch64_neon_fp8_fmlalltt : AdvSIMD_FP8_FMLA_Intrinsic;

  def int_aarch64_neon_fp8_fmlalb_lane : AdvSIMD_FP8_FMLA_LANE_Intrinsic;
  def int_aarch64_neon_fp8_fmlalt_lane : AdvSIMD_FP8_FMLA_LANE_Intrinsic;

  def int_aarch64_neon_fp8_fmlallbb_lane : AdvSIMD_FP8_FMLA_LANE_Intrinsic;
  def int_aarch64_neon_fp8_fmlallbt_lane : AdvSIMD_FP8_FMLA_LANE_Intrinsic;
  def int_aarch64_neon_fp8_fmlalltb_lane : AdvSIMD_FP8_FMLA_LANE_Intrinsic;
  def int_aarch64_neon_fp8_fmlalltt_lane : AdvSIMD_FP8_FMLA_LANE_Intrinsic;
}

def llvm_nxv1i1_ty  : LLVMType<nxv1i1>;
def llvm_nxv2i1_ty  : LLVMType<nxv2i1>;
def llvm_nxv4i1_ty  : LLVMType<nxv4i1>;
def llvm_nxv8i1_ty  : LLVMType<nxv8i1>;
def llvm_nxv16i1_ty : LLVMType<nxv16i1>;
def llvm_nxv16i8_ty : LLVMType<nxv16i8>;
def llvm_nxv8i16_ty : LLVMType<nxv8i16>;
def llvm_nxv4i32_ty : LLVMType<nxv4i32>;
def llvm_nxv2i64_ty : LLVMType<nxv2i64>;
def llvm_nxv8f16_ty : LLVMType<nxv8f16>;
def llvm_nxv8bf16_ty : LLVMType<nxv8bf16>;
def llvm_nxv4f32_ty : LLVMType<nxv4f32>;
def llvm_nxv2f64_ty : LLVMType<nxv2f64>;

// Type overloads
defvar EmptyOverload = [[]<LLVMType>];

// Integer type overloads
defvar I8_Overloads = [[llvm_nxv16i8_ty]];
defvar I16_Overloads = [[llvm_nxv8i16_ty]];
defvar I32_Overloads = [[llvm_nxv4i32_ty]];
defvar I64_Overloads = [[llvm_nxv2i64_ty]];

defvar I8_I16_Overloads = [[llvm_nxv16i8_ty], [llvm_nxv8i16_ty]];
defvar I16_I32_Overloads = [[llvm_nxv4i32_ty], [llvm_nxv8i16_ty]];
defvar I32_I64_Overloads = [[llvm_nxv2i64_ty], [llvm_nxv4i32_ty]];

defvar I8_I16_I32_Overloads = [[llvm_nxv16i8_ty], [llvm_nxv4i32_ty], [llvm_nxv8i16_ty]];
defvar I16_I32_I64_Overloads = [[llvm_nxv2i64_ty], [llvm_nxv4i32_ty], [llvm_nxv8i16_ty]];

defvar I8_I16_I32_I64_Overloads = [[llvm_nxv16i8_ty], [llvm_nxv8i16_ty], [llvm_nxv4i32_ty], [llvm_nxv2i64_ty]];

// Floating point type overloads
defvar BF16_Overloads = [[llvm_nxv8bf16_ty]];
defvar F16_Overloads = [[llvm_nxv8f16_ty]];
defvar F32_Overloads = [[llvm_nxv4f32_ty]];
defvar F64_Overloads = [[llvm_nxv2f64_ty]];

defvar BF16_F16_Overloads = [[llvm_nxv8bf16_ty], [llvm_nxv8f16_ty]];
defvar F16_F32_Overloads = [[llvm_nxv4f32_ty], [llvm_nxv8f16_ty]];
defvar F32_F64_Overloads = [[llvm_nxv2f64_ty], [llvm_nxv4f32_ty]];

defvar F16_F32_F64_Overloads = [[llvm_nxv8f16_ty], [llvm_nxv4f32_ty], [llvm_nxv2f64_ty]];
defvar BF16_F16_F32_Overloads = [[llvm_nxv4f32_ty], [llvm_nxv8bf16_ty], [llvm_nxv8f16_ty]];
defvar BF16_F16_I16_Overloads = [[llvm_nxv8bf16_ty], [llvm_nxv8f16_ty], [llvm_nxv8i16_ty]];

defvar I8_I16_BF16_F16_Overloads = [[llvm_nxv16i8_ty], [llvm_nxv8bf16_ty], [llvm_nxv8f16_ty], [llvm_nxv8i16_ty]];
defvar I32_I64_F32_F64_Overloads = [[llvm_nxv2f64_ty], [llvm_nxv2i64_ty], [llvm_nxv4f32_ty], [llvm_nxv4i32_ty]];

defvar F64F64_OverloadPair = [[llvm_nxv2f64_ty, llvm_nxv2f64_ty]];
defvar F32F32_OverloadPair = [[llvm_nxv4f32_ty, llvm_nxv4f32_ty]];
defvar F32BF16_OverloadPair = [[llvm_nxv4f32_ty, llvm_nxv8bf16_ty]];
defvar F32F16_OverloadPair = [[llvm_nxv4f32_ty, llvm_nxv8f16_ty]];
defvar BF16BF16_OverloadPair = [[llvm_nxv8bf16_ty, llvm_nxv8bf16_ty]];
defvar F16F16_OverloadPair = [[llvm_nxv8f16_ty, llvm_nxv8f16_ty]];

defvar UnpackedIntegerDataVectorTypes = [
  LLVMType<nxv2i8>, LLVMType<nxv4i8>, LLVMType<nxv8i8>,
  LLVMType<nxv2i16>, LLVMType<nxv4i16>,
  LLVMType<nxv2i32>
];

defvar PackedIntegerVectorTypes = [
  llvm_nxv16i8_ty, llvm_nxv2i64_ty, llvm_nxv4i32_ty, llvm_nxv8i16_ty
];

defvar PackedFPVectorTypes = [
  llvm_nxv8f16_ty, llvm_nxv8bf16_ty, llvm_nxv4f32_ty, llvm_nxv2f64_ty
];

defvar PackedDataVectorTypes = !listconcat(PackedIntegerVectorTypes, PackedFPVectorTypes);

defvar AllPackedData_Overloads = !foreach(T, PackedDataVectorTypes, [T]);

// Predicate type overloads
defvar Predicate_Overloads = [
  [llvm_nxv16i1_ty],
  [llvm_nxv8i1_ty],
  [llvm_nxv4i1_ty],
  [llvm_nxv2i1_ty]
];

// Pointer related
defvar PTR_Overloads = [[llvm_ptr_ty]];

defvar F64PTR_I64PTR_OverloadPair = [
  [llvm_nxv2f64_ty, llvm_ptr_ty],
  [llvm_nxv2i64_ty, llvm_ptr_ty],
];

defvar F32PTR_I32PTR_OverloadPair = [
  [llvm_nxv4f32_ty, llvm_ptr_ty],
  [llvm_nxv4i32_ty, llvm_ptr_ty],
];

defvar PredicatePTR_OverloadPairs = [
  [llvm_nxv16i1_ty, llvm_ptr_ty],
  [llvm_nxv8i1_ty, llvm_ptr_ty],
  [llvm_nxv4i1_ty, llvm_ptr_ty],
  [llvm_nxv2i1_ty, llvm_ptr_ty],
];

defvar DataPTR_OverloadPairs =
  !foreach(T, !listconcat(PackedDataVectorTypes, UnpackedIntegerDataVectorTypes), [T, llvm_ptr_ty]);

defvar PackedDataPTR_OverloadPairs =
  !foreach(T, PackedDataVectorTypes, [T, llvm_ptr_ty]);

defvar PackedDataI64_OverloadPairs =
  !foreach(T, PackedDataVectorTypes, [T, llvm_nxv2i64_ty]);

defvar DataI32_DataI64_OverloadPairs =
  !listconcat(
    !foreach(T, !listconcat(PackedDataVectorTypes, UnpackedIntegerDataVectorTypes), [T, llvm_nxv4i32_ty]),
    !foreach(T, !listconcat(PackedDataVectorTypes, UnpackedIntegerDataVectorTypes), [T, llvm_nxv2i64_ty]));

let TargetPrefix = "aarch64" in {  // All intrinsics start with "llvm.aarch64.".

  class Guarded_AdvSIMD_1Vec_PredLoad_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyptr_ty],
                [IntrReadMem, IntrArgMemOnly]>;

  class Guarded_AdvSIMD_2Vec_PredLoad_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyptr_ty],
                [IntrReadMem, IntrArgMemOnly]>;

  class Guarded_AdvSIMD_3Vec_PredLoad_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyptr_ty],
                [IntrReadMem, IntrArgMemOnly]>;

  class Guarded_AdvSIMD_4Vec_PredLoad_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyptr_ty],
                [IntrReadMem, IntrArgMemOnly]>;

  class Guarded_AdvSIMD_1Vec_PredLoad_WriteFFR_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyptr_ty],
                [IntrRead<[ArgMem, FFR]>, IntrWrite<[FFR]>]>;

  class Guarded_AdvSIMD_1Vec_PredStore_Intrinsic
    : GuardedDefaultIntrinsic<[],
                [llvm_anyvector_ty,
                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyptr_ty],
                [IntrArgMemOnly, NoCapture<ArgIndex<2>>]>;

  class Guarded_AdvSIMD_2Vec_PredStore_Intrinsic
      : GuardedDefaultIntrinsic<[],
                  [llvm_anyvector_ty, LLVMMatchType<0>,
                   LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyptr_ty],
                  [IntrArgMemOnly, NoCapture<ArgIndex<3>>]>;

  class Guarded_AdvSIMD_3Vec_PredStore_Intrinsic
      : GuardedDefaultIntrinsic<[],
                  [llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>,
                   LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyptr_ty],
                  [IntrArgMemOnly, NoCapture<ArgIndex<4>>]>;

  class Guarded_AdvSIMD_4Vec_PredStore_Intrinsic
      : GuardedDefaultIntrinsic<[],
                  [llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>,
                   LLVMMatchType<0>,
                   LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyptr_ty],
                  [IntrArgMemOnly, NoCapture<ArgIndex<5>>]>;

  class Guarded_AdvSIMD_SVE_Index_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMVectorElementType<0>,
                 LLVMVectorElementType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 LLVMMatchType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMMatchType<0>,
                 llvm_i32_ty],
                !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<2>>])>;

  class Guarded_AdvSIMD_3VectorArgIndexed_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMMatchType<0>,
                 LLVMMatchType<0>,
                 llvm_i32_ty],
                !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<3>>])>;

  class Guarded_AdvSIMD_Pred1VectorArg_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 LLVMMatchType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 LLVMMatchType<0>,
                 LLVMMatchType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_Pred3VectorArg_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 LLVMMatchType<0>,
                 LLVMMatchType<0>,
                 LLVMMatchType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_Compare_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 llvm_anyvector_ty,
                 LLVMMatchType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_CompareWide_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 llvm_anyvector_ty,
                 llvm_nxv2i64_ty],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_Saturating_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_SaturatingWithPattern_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 llvm_i32_ty,
                 llvm_i32_ty],
                !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>])>;

  class Guarded_AdvSIMD_SVE_Saturating_N_Intrinsic<LLVMType T, list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[T],
                [T, llvm_anyvector_ty],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<LLVMType T, list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[T],
                [T, llvm_i32_ty, llvm_i32_ty],
                !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>])>;

  class Guarded_AdvSIMD_SVE_CNT_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[LLVMVectorOfBitcastsToInt<0>],
                [LLVMVectorOfBitcastsToInt<0>,
                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 llvm_anyvector_ty],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_ReduceWithInit_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[LLVMVectorElementType<0>],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 LLVMVectorElementType<0>,
                 llvm_anyvector_ty],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_ShiftByImm_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 LLVMMatchType<0>,
                 llvm_i32_ty],
                !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<2>>])>;

  class Guarded_AdvSIMD_SVE_ShiftWide_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 LLVMMatchType<0>,
                 llvm_nxv2i64_ty],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_Unpack_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
               [LLVMSubdivide2VectorType<0>],
               !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_CADD_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 LLVMMatchType<0>,
                 LLVMMatchType<0>,
                 llvm_i32_ty],
                [IntrNoMem, ImmArg<ArgIndex<3>>]>;

  class Guarded_AdvSIMD_SVE_CMLA_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 LLVMMatchType<0>,
                 LLVMMatchType<0>,
                 LLVMMatchType<0>,
                 llvm_i32_ty],
                [IntrNoMem, ImmArg<ArgIndex<4>>]>;

  class Guarded_AdvSIMD_SVE_CMLA_LANE_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMMatchType<0>,
                 LLVMMatchType<0>,
                 llvm_i32_ty,
                 llvm_i32_ty],
                !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>])>;

  class Guarded_AdvSIMD_SVE_DUP_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 LLVMVectorElementType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_DUP_Unpred_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty], [LLVMVectorElementType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_DUPQ_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 llvm_i64_ty],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_EXPA_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMVectorOfBitcastsToInt<0>],
                [IntrNoMem]>;

  class Guarded_AdvSIMD_SVE_FCVTZS_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMVectorOfBitcastsToInt<0>,
                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 llvm_anyvector_ty],
                [IntrNoMem]>;

  class Guarded_AdvSIMD_SVE_INSR_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMVectorElementType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_PTRUE_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [llvm_i32_ty],
                [IntrNoMem, ImmArg<ArgIndex<0>>]>;

  class Guarded_AdvSIMD_SVE_PUNPKHI_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[LLVMOneNthElementsVectorType<0, 2>],
                [llvm_anyvector_ty],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_SCALE_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 LLVMMatchType<0>,
                 LLVMVectorOfBitcastsToInt<0>],
                [IntrNoMem]>;

  class Guarded_AdvSIMD_SVE_SCVTF_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 llvm_anyvector_ty],
                [IntrNoMem]>;

  class Guarded_AdvSIMD_SVE_TSMUL_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMVectorOfBitcastsToInt<0>],
                [IntrNoMem]>;

  class Guarded_AdvSIMD_SVE_CNTB_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_i64_ty],
                [llvm_i32_ty],
                !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<0>>])>;

  class Guarded_AdvSIMD_SVE_CNTP_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_i64_ty],
                [llvm_anyvector_ty, LLVMMatchType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_DOT_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMSubdivide4VectorType<0>,
                 LLVMSubdivide4VectorType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_DOT_Indexed_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMSubdivide4VectorType<0>,
                 LLVMSubdivide4VectorType<0>,
                 llvm_i32_ty],
                !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<3>>])>;

  class Guarded_AdvSIMD_SVE_PTEST_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_i1_ty],
                [llvm_anyvector_ty,
                 LLVMMatchType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE_TBL_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMVectorOfBitcastsToInt<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_AdvSIMD_SVE2_TBX_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMMatchType<0>,
                 LLVMVectorOfBitcastsToInt<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_SVE2_LUTI_Intrinsic<list<IntrinsicProperty> Attrs = []>
    :  GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 llvm_nxv16i8_ty,
                 llvm_i32_ty],
                 !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<2>>])>;

  class Guarded_SVE2_LUTI_X2_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMMatchType<0>,
                 llvm_nxv16i8_ty,
                 llvm_i32_ty],
                !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<3>>])>;

  class Guarded_SVE2_LUTI6_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_nxv16i8_ty],
                [llvm_nxv16i8_ty,
                 llvm_nxv16i8_ty,
                 llvm_nxv16i8_ty],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_SVE2_1VectorArg_Long_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMSubdivide2VectorType<0>,
                 llvm_i32_ty],
                !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<1>>])>;

  class Guarded_SVE2_2VectorArg_Long_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMSubdivide2VectorType<0>,
                 LLVMSubdivide2VectorType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_SVE2_2VectorArgIndexed_Long_Intrinsic<list<IntrinsicProperty> Attrs = []>
  : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
              [LLVMSubdivide2VectorType<0>,
               LLVMSubdivide2VectorType<0>,
               llvm_i32_ty],
              !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<2>>])>;

  class Guarded_SVE2_2VectorArg_Wide_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMSubdivide2VectorType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_SVE2_2VectorArg_Pred_Long_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 LLVMMatchType<0>,
                 LLVMSubdivide2VectorType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_SVE2_3VectorArg_Long_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMSubdivide2VectorType<0>,
                 LLVMSubdivide2VectorType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_SVE2_3VectorArgIndexed_Long_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMSubdivide2VectorType<0>,
                 LLVMSubdivide2VectorType<0>,
                 llvm_i32_ty],
                !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<3>>])>;

  class Guarded_SVE2_1VectorArg_Narrowing_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[LLVMSubdivide2VectorType<0>],
                [llvm_anyvector_ty],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_SVE2_Merged1VectorArg_Narrowing_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[LLVMSubdivide2VectorType<0>],
                [LLVMSubdivide2VectorType<0>,
                 llvm_anyvector_ty],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_SVE2_2VectorArg_Narrowing_Intrinsic<list<IntrinsicProperty> Attrs = []>
      : GuardedDefaultIntrinsic<
            [LLVMSubdivide2VectorType<0>],
            [llvm_anyvector_ty, LLVMMatchType<0>],
            !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_SVE2_Merged2VectorArg_Narrowing_Intrinsic<list<IntrinsicProperty> Attrs = []>
      : GuardedDefaultIntrinsic<
            [LLVMSubdivide2VectorType<0>],
            [LLVMSubdivide2VectorType<0>, llvm_anyvector_ty, LLVMMatchType<0>],
            !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_SVE2_1VectorArg_Imm_Narrowing_Intrinsic<list<IntrinsicProperty> Attrs = []>
      : GuardedDefaultIntrinsic<[LLVMSubdivide2VectorType<0>],
                  [llvm_anyvector_ty, llvm_i32_ty],
                  !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<1>>])>;

  class Guarded_SVE2_2VectorArg_Imm_Narrowing_Intrinsic<list<IntrinsicProperty> Attrs = []>
      : GuardedDefaultIntrinsic<[LLVMSubdivide2VectorType<0>],
                  [LLVMSubdivide2VectorType<0>, llvm_anyvector_ty,
                   llvm_i32_ty],
                  !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<2>>])>;

  class Guarded_SVE2_CONFLICT_DETECT_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [llvm_anyptr_ty, LLVMMatchType<1>],
                !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_SVE2_3VectorArg_Indexed_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMSubdivide2VectorType<0>,
                 LLVMSubdivide2VectorType<0>,
                 llvm_i32_ty],
                !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<3>>])>;

  class Guarded_SVE2_1VectorArgIndexed_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 llvm_i32_ty],
                !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<1>>])>;

  class Guarded_AdvSIMD_SVE_CDOT_LANE_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>,
                 LLVMSubdivide4VectorType<0>,
                 LLVMSubdivide4VectorType<0>,
                 llvm_i32_ty,
                 llvm_i32_ty],
                !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>])>;

  class Guarded_SVE2_1VectorArg_Pred_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                            [llvm_anyvector_ty],
                            !listconcat(Attrs, [IntrNoMem])>;

  class Guarded_SVE2_1VectorArgIndexed_Pred_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                            [llvm_anyvector_ty, llvm_i32_ty],
                            !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<1>>])>;

  class Guarded_SVE2_Pred_1VectorArgIndexed_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                            [LLVMMatchType<0>,
                             LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_i32_ty],
                            !listconcat(Attrs, [IntrNoMem, ImmArg<ArgIndex<2>>])>;

  class Guarded_SVE2_Pred_1VectorArg_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                            [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                            !listconcat(Attrs, [IntrNoMem])>;

  // NOTE: There is no relationship between these intrinsics beyond an attempt
  // to reuse currently identical class definitions.
  class AdvSIMD_SVE_LOGB_Intrinsic : Guarded_AdvSIMD_SVE_CNT_Intrinsic;
  class Guarded_AdvSIMD_SVE2_CADD_Intrinsic<list<IntrinsicProperty> Attrs = []> : Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic<Attrs>;
  class Guarded_AdvSIMD_SVE2_CMLA_Intrinsic<list<IntrinsicProperty> Attrs = []> : Guarded_AdvSIMD_3VectorArgIndexed_Intrinsic<Attrs>;

  // This class of intrinsics are not intended to be useful within LLVM IR but
  // are instead here to support some of the more regid parts of the ACLE.
  class Guarded_Builtin_SVCVT<LLVMType OUT, LLVMType PRED, LLVMType IN>
      : GuardedDefaultIntrinsic<[OUT], [OUT, PRED, IN], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// SVE

let TargetPrefix = "aarch64" in {  // All intrinsics start with "llvm.aarch64.".

class Guarded_AdvSIMD_SVE_2SVBoolArg_Intrinsic<list<IntrinsicProperty> Attrs = []>
  : GuardedDefaultIntrinsic<[llvm_nxv16i1_ty],
                          [llvm_nxv16i1_ty],
                          !listconcat(Attrs, [IntrNoMem])>;

class Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<list<IntrinsicProperty> Attrs = []>
  : GuardedDefaultIntrinsic<[llvm_nxv16i1_ty],
                          [llvm_nxv16i1_ty, llvm_nxv16i1_ty],
                          !listconcat(Attrs, [IntrNoMem])>;

class Guarded_AdvSIMD_SVE_Reduce_Intrinsic<list<IntrinsicProperty> Attrs = []>
  : GuardedDefaultIntrinsic<[LLVMVectorElementType<0>],
              [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
               llvm_anyvector_ty],
              !listconcat(Attrs, [IntrNoMem])>;

class Guarded_AdvSIMD_SVE_V128_Reduce_Intrinsic<list<IntrinsicProperty> Attrs = []>
  : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
              [LLVMScalarOrSameVectorWidth<1, llvm_i1_ty>,
               llvm_anyvector_ty],
               !listconcat(Attrs, [IntrNoMem])>;

class Guarded_AdvSIMD_SVE_SADDV_Reduce_Intrinsic<list<IntrinsicProperty> Attrs = []>
  : GuardedDefaultIntrinsic<[llvm_i64_ty],
              [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
               llvm_anyvector_ty],
              !listconcat(Attrs, [IntrNoMem])>;

class Guarded_AdvSIMD_SVE_WHILE_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [llvm_anyint_ty, LLVMMatchType<1>],
                !listconcat(Attrs, [IntrNoMem])>;

class Guarded_AdvSIMD_GatherLoad_SV_64b_Offsets_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [
                  LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                  llvm_anyptr_ty,
                  LLVMScalarOrSameVectorWidth<0, llvm_i64_ty>
                ],
                [IntrReadMem, IntrArgMemOnly]>;

class Guarded_AdvSIMD_GatherLoad_SV_64b_Offsets_WriteFFR_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [
                  LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                  llvm_anyptr_ty,
                  LLVMScalarOrSameVectorWidth<0, llvm_i64_ty>
                ],
                [IntrRead<[ArgMem, FFR]>, IntrWrite<[FFR]>]>;

class Guarded_AdvSIMD_GatherLoad_SV_32b_Offsets_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [
                  LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                  llvm_anyptr_ty,
                  LLVMScalarOrSameVectorWidth<0, llvm_i32_ty>
                ],
                [IntrReadMem, IntrArgMemOnly]>;

class Guarded_AdvSIMD_GatherLoad_SV_32b_Offsets_WriteFFR_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [
                  LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                  llvm_anyptr_ty,
                  LLVMScalarOrSameVectorWidth<0, llvm_i32_ty>
                ],
                [IntrRead<[ArgMem, FFR]>, IntrWrite<[FFR]>]>;

class Guarded_AdvSIMD_GatherLoad_VS_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [
                  LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                  llvm_anyvector_ty,
                  llvm_i64_ty
                ],
                [IntrReadMem]>;

class Guarded_AdvSIMD_GatherLoadQ_VS_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [
                  llvm_nxv1i1_ty,
                  llvm_anyvector_ty,
                  llvm_i64_ty
                ],
                [IntrReadMem]>;

class Guarded_AdvSIMD_GatherLoadQ_SV_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [
                  llvm_nxv1i1_ty,
                  llvm_anyptr_ty,
                  llvm_nxv2i64_ty
                ],
                [IntrReadMem, IntrArgMemOnly]>;

class Guarded_AdvSIMD_GatherLoad_VS_WriteFFR_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [
                  LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                  llvm_anyvector_ty,
                  llvm_i64_ty
                ],
                [IntrRead<[ArgMem, FFR]>, IntrWrite<[FFR]>]>;

class Guarded_AdvSIMD_ScatterStore_SV_64b_Offsets_Intrinsic
    : GuardedDefaultIntrinsic<[],
               [
                 llvm_anyvector_ty,
                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 llvm_anyptr_ty,
                 LLVMScalarOrSameVectorWidth<0, llvm_i64_ty>
               ],
               [IntrWriteMem, IntrArgMemOnly]>;

class Guarded_AdvSIMD_ScatterStore_SV_32b_Offsets_Intrinsic
    : GuardedDefaultIntrinsic<[],
               [
                 llvm_anyvector_ty,
                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 llvm_anyptr_ty,
                 LLVMScalarOrSameVectorWidth<0, llvm_i32_ty>
               ],
               [IntrWriteMem, IntrArgMemOnly]>;

class Guarded_AdvSIMD_ScatterStore_VS_Intrinsic
    : GuardedDefaultIntrinsic<[],
               [
                 llvm_anyvector_ty,
                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                 llvm_anyvector_ty, llvm_i64_ty
               ],
               [IntrWriteMem]>;

class Guarded_AdvSIMD_ScatterStoreQ_VS_Intrinsic
    : GuardedDefaultIntrinsic<[],
               [
                 llvm_anyvector_ty,
                 llvm_nxv1i1_ty,
                 llvm_anyvector_ty,
                 llvm_i64_ty
               ],
               [IntrWriteMem]>;

class Guarded_AdvSIMD_ScatterStoreQ_SV_Intrinsic
    : GuardedDefaultIntrinsic<[],
               [
                 llvm_anyvector_ty,
                 llvm_nxv1i1_ty,
                 llvm_anyptr_ty,
                 llvm_nxv2i64_ty
               ],
               [IntrWriteMem, IntrArgMemOnly]>;

class Guarded_SVE_gather_prf_SV
    : GuardedDefaultIntrinsic<[],
                [
                  LLVMScalarOrSameVectorWidth<1, llvm_i1_ty>, // Predicate
                  llvm_anyptr_ty, // Base address
                  llvm_anyvector_ty, // Offsets
                  llvm_i32_ty // Prfop
                ],
                [IntrInaccessibleMemOrArgMemOnly, NoCapture<ArgIndex<1>>, ImmArg<ArgIndex<3>>]>;

class Guarded_SVE_gather_prf_VS
    : GuardedDefaultIntrinsic<[],
                [
                  LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, // Predicate
                  llvm_anyvector_ty, // Base addresses
                  llvm_i64_ty, // Scalar offset
                  llvm_i32_ty // Prfop
                ],
                [IntrInaccessibleMemOrArgMemOnly, ImmArg<ArgIndex<3>>]>;

class Guarded_SVE_MatMul_Intrinsic<list<IntrinsicProperty> Attrs = []>
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                [LLVMMatchType<0>, LLVMSubdivide4VectorType<0>, LLVMSubdivide4VectorType<0>],
                !listconcat(Attrs, [IntrNoMem])>;

class Guarded_SVE_4Vec_BF16
    : GuardedDefaultIntrinsic<[llvm_nxv4f32_ty],
                [llvm_nxv4f32_ty, llvm_nxv8bf16_ty, llvm_nxv8bf16_ty],
                [IntrNoMem]>;

class Guarded_SVE_4Vec_BF16_Indexed
    : GuardedDefaultIntrinsic<[llvm_nxv4f32_ty],
                [llvm_nxv4f32_ty, llvm_nxv8bf16_ty, llvm_nxv8bf16_ty, llvm_i32_ty],
                [IntrNoMem, ImmArg<ArgIndex<3>>]>;

//
// Loads
//

let TargetGuards = [Variant<"sve", "sme", DataPTR_OverloadPairs>] in {
  def int_aarch64_sve_ld1   : Guarded_AdvSIMD_1Vec_PredLoad_Intrinsic;
}

let TargetGuards = [Variant<"sve", "sme", PackedDataPTR_OverloadPairs>] in {
  def int_aarch64_sve_ldnt1 : Guarded_AdvSIMD_1Vec_PredLoad_Intrinsic;

  def int_aarch64_sve_ld2_sret : Guarded_AdvSIMD_2Vec_PredLoad_Intrinsic;
  def int_aarch64_sve_ld3_sret : Guarded_AdvSIMD_3Vec_PredLoad_Intrinsic;
  def int_aarch64_sve_ld4_sret : Guarded_AdvSIMD_4Vec_PredLoad_Intrinsic;
  def int_aarch64_sve_ld1rq : Guarded_AdvSIMD_1Vec_PredLoad_Intrinsic;
}

let TargetGuards = [NonStreamingVariant<"sve", DataPTR_OverloadPairs>] in {
  def int_aarch64_sve_ldnf1 : Guarded_AdvSIMD_1Vec_PredLoad_WriteFFR_Intrinsic;
  def int_aarch64_sve_ldff1 : Guarded_AdvSIMD_1Vec_PredLoad_WriteFFR_Intrinsic;
}

let TargetGuards = [NonStreamingVariant<"f64mm", PackedDataPTR_OverloadPairs>] in
def int_aarch64_sve_ld1ro : Guarded_AdvSIMD_1Vec_PredLoad_Intrinsic;

//
// Stores
//

let TargetGuards = [Variant<"sve", "sme", DataPTR_OverloadPairs>] in {
  def int_aarch64_sve_st1  : Guarded_AdvSIMD_1Vec_PredStore_Intrinsic;
}

let TargetGuards = [Variant<"sve", "sme", PackedDataPTR_OverloadPairs>] in {
  def int_aarch64_sve_stnt1 : Guarded_AdvSIMD_1Vec_PredStore_Intrinsic;

  def int_aarch64_sve_st2  : Guarded_AdvSIMD_2Vec_PredStore_Intrinsic;
  def int_aarch64_sve_st3  : Guarded_AdvSIMD_3Vec_PredStore_Intrinsic;
  def int_aarch64_sve_st4  : Guarded_AdvSIMD_4Vec_PredStore_Intrinsic;
}

//
// Prefetches
//

let TargetGuards = [Variant<"sve", "sme", PredicatePTR_OverloadPairs>] in
def int_aarch64_sve_prf
  : GuardedDefaultIntrinsic<[], [llvm_anyvector_ty, llvm_anyptr_ty, llvm_i32_ty],
                  [IntrArgMemOnly, ImmArg<ArgIndex<2>>]>;

defvar PointerAndI32_Overloads = [
  [llvm_ptr_ty, llvm_nxv4i32_ty],
  [llvm_ptr_ty, LLVMType<nxv2i32>]
];

// Scalar + 32-bit scaled offset vector, zero extend, packed and
// unpacked.
let TargetGuards = [NonStreamingVariant<"sve", PointerAndI32_Overloads>] in {
  def int_aarch64_sve_prfb_gather_uxtw_index : Guarded_SVE_gather_prf_SV;
  def int_aarch64_sve_prfh_gather_uxtw_index : Guarded_SVE_gather_prf_SV;
  def int_aarch64_sve_prfw_gather_uxtw_index : Guarded_SVE_gather_prf_SV;
  def int_aarch64_sve_prfd_gather_uxtw_index : Guarded_SVE_gather_prf_SV;
}

// Scalar + 32-bit scaled offset vector, sign extend, packed and
// unpacked.
let TargetGuards = [NonStreamingVariant<"sve", PointerAndI32_Overloads>] in {
  def int_aarch64_sve_prfb_gather_sxtw_index : Guarded_SVE_gather_prf_SV;
  def int_aarch64_sve_prfw_gather_sxtw_index : Guarded_SVE_gather_prf_SV;
  def int_aarch64_sve_prfh_gather_sxtw_index : Guarded_SVE_gather_prf_SV;
  def int_aarch64_sve_prfd_gather_sxtw_index : Guarded_SVE_gather_prf_SV;
}

defvar PointerAndI64_Overloads = [
  [llvm_ptr_ty, llvm_nxv2i64_ty]
];

// Scalar + 64-bit scaled offset vector.
let TargetGuards = [NonStreamingVariant<"sve", PointerAndI64_Overloads>] in {
  def int_aarch64_sve_prfb_gather_index : Guarded_SVE_gather_prf_SV;
  def int_aarch64_sve_prfh_gather_index : Guarded_SVE_gather_prf_SV;
  def int_aarch64_sve_prfw_gather_index : Guarded_SVE_gather_prf_SV;
  def int_aarch64_sve_prfd_gather_index : Guarded_SVE_gather_prf_SV;
}

// Vector + scalar.
let TargetGuards = [NonStreamingVariant<"sve", I32_I64_Overloads>] in {
  def int_aarch64_sve_prfb_gather_scalar_offset : Guarded_SVE_gather_prf_VS;
  def int_aarch64_sve_prfh_gather_scalar_offset : Guarded_SVE_gather_prf_VS;
  def int_aarch64_sve_prfw_gather_scalar_offset : Guarded_SVE_gather_prf_VS;
  def int_aarch64_sve_prfd_gather_scalar_offset : Guarded_SVE_gather_prf_VS;
}

//
// Scalar to vector operations
//

let TargetGuards = [Variant<"sve", "sme", AllPackedData_Overloads>] in {
  def int_aarch64_sve_dup : Guarded_AdvSIMD_SVE_DUP_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_dup_x : Guarded_AdvSIMD_SVE_DUP_Unpred_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve", "sme", I8_I16_I32_I64_Overloads>] in
def int_aarch64_sve_index : Guarded_AdvSIMD_SVE_Index_Intrinsic<[IntrSpeculatable]>;

//
// Address calculation
//

let TargetGuards = [NonStreamingVariant<"sve", I32_I64_Overloads>] in {
  def int_aarch64_sve_adrb : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_adrh : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_adrw : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_adrd : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
}

//
// Integer arithmetic
//

let TargetGuards = [Variant<"sve", "sme", I8_I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_add   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_add_u : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sub   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sub_u : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_subr  : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_mul        : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_mul_u      : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smulh      : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smulh_u    : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umulh      : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umulh_u    : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smax       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smax_u     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umax       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umax_u     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smin       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smin_u     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umin       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umin_u     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sabd       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sabd_u     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uabd       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uabd_u     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_mad        : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_msb        : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_mla        : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_mla_u      : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_mls        : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_mls_u      : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_saddv      : Guarded_AdvSIMD_SVE_SADDV_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uaddv      : Guarded_AdvSIMD_SVE_SADDV_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smaxv      : Guarded_AdvSIMD_SVE_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umaxv      : Guarded_AdvSIMD_SVE_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sminv      : Guarded_AdvSIMD_SVE_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uminv      : Guarded_AdvSIMD_SVE_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_orv        : Guarded_AdvSIMD_SVE_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_eorv       : Guarded_AdvSIMD_SVE_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_andv       : Guarded_AdvSIMD_SVE_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_abs : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_neg : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqadd_x   : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqsub_x   : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqadd_x   : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqsub_x   : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I8_Overloads>] in
def int_aarch64_sve_pmul       : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_mul_lane   : Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_mla_lane   : Guarded_AdvSIMD_3VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_mls_lane   : Guarded_AdvSIMD_3VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve", "sme", I32_I64_Overloads>] in {
  def int_aarch64_sve_sdiv       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sdiv_u     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_udiv       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_udiv_u     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sdivr      : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_udivr      : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sdot      : Guarded_AdvSIMD_SVE_DOT_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sdot_lane : Guarded_AdvSIMD_SVE_DOT_Indexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_udot      : Guarded_AdvSIMD_SVE_DOT_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_udot_lane : Guarded_AdvSIMD_SVE_DOT_Indexed_Intrinsic<[IntrSpeculatable]>;
}

defvar SveOrqvTypes = [
  [llvm_v16i8_ty, llvm_nxv16i8_ty],
  [llvm_v2i64_ty, llvm_nxv2i64_ty],
  [llvm_v4i32_ty, llvm_nxv4i32_ty],
  [llvm_v8i16_ty, llvm_nxv8i16_ty],
];
let TargetGuards = [Variant<"sve,(sve2p1|sme2p1)", "sve2p1|sme2p1", SveOrqvTypes>] in {
  def int_aarch64_sve_orqv      : Guarded_AdvSIMD_SVE_V128_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_eorqv     : Guarded_AdvSIMD_SVE_V128_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_andqv     : Guarded_AdvSIMD_SVE_V128_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_addqv     : Guarded_AdvSIMD_SVE_V128_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smaxqv    : Guarded_AdvSIMD_SVE_V128_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umaxqv    : Guarded_AdvSIMD_SVE_V128_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sminqv    : Guarded_AdvSIMD_SVE_V128_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uminqv    : Guarded_AdvSIMD_SVE_V128_Reduce_Intrinsic<[IntrSpeculatable]>;
}

// Shifts

let TargetGuards = [Variant<"sve", "sme", I8_I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_asr      : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_asr_u    : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_asrd     : Guarded_AdvSIMD_SVE_ShiftByImm_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_lsl      : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_lsl_u    : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_lsr      : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_lsr_u    : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve", "sme", I8_I16_I32_Overloads>] in {
  def int_aarch64_sve_asr_wide : Guarded_AdvSIMD_SVE_ShiftWide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_lsl_wide : Guarded_AdvSIMD_SVE_ShiftWide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_lsr_wide : Guarded_AdvSIMD_SVE_ShiftWide_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve", "sme", AllPackedData_Overloads>] in
def int_aarch64_sve_insr     : Guarded_AdvSIMD_SVE_INSR_Intrinsic<[IntrSpeculatable]>;

//
// Integer comparisons
//

let TargetGuards = [Variant<"sve", "sme", I8_I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_cmpeq : Guarded_AdvSIMD_SVE_Compare_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cmpge : Guarded_AdvSIMD_SVE_Compare_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cmpgt : Guarded_AdvSIMD_SVE_Compare_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cmphi : Guarded_AdvSIMD_SVE_Compare_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cmphs : Guarded_AdvSIMD_SVE_Compare_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cmpne : Guarded_AdvSIMD_SVE_Compare_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve", "sme", I8_I16_I32_Overloads>] in {
  def int_aarch64_sve_cmpeq_wide : Guarded_AdvSIMD_SVE_CompareWide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cmpge_wide : Guarded_AdvSIMD_SVE_CompareWide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cmpgt_wide : Guarded_AdvSIMD_SVE_CompareWide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cmphi_wide : Guarded_AdvSIMD_SVE_CompareWide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cmphs_wide : Guarded_AdvSIMD_SVE_CompareWide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cmple_wide : Guarded_AdvSIMD_SVE_CompareWide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cmplo_wide : Guarded_AdvSIMD_SVE_CompareWide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cmpls_wide : Guarded_AdvSIMD_SVE_CompareWide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cmplt_wide : Guarded_AdvSIMD_SVE_CompareWide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cmpne_wide : Guarded_AdvSIMD_SVE_CompareWide_Intrinsic<[IntrSpeculatable]>;
}

//
// Counting bits
//

let TargetGuards = [Variant<"sve", "sme", I8_I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_cls : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_clz : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
}
let TargetGuards = [Variant<"sve", "sme", AllPackedData_Overloads>] in
def int_aarch64_sve_cnt : Guarded_AdvSIMD_SVE_CNT_Intrinsic<[IntrSpeculatable]>;

//
// Counting elements
//

let TargetGuards = [Variant<"sve", "sme", EmptyOverload>] in {
  def int_aarch64_sve_cntb : Guarded_AdvSIMD_SVE_CNTB_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cnth : Guarded_AdvSIMD_SVE_CNTB_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cntw : Guarded_AdvSIMD_SVE_CNTB_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cntd : Guarded_AdvSIMD_SVE_CNTB_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve,(sve2p2|sme2p2)", "sve2p2|sme2p2", Predicate_Overloads>] in {
  def int_aarch64_sve_firstp : Guarded_AdvSIMD_SVE_CNTP_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_lastp  : Guarded_AdvSIMD_SVE_CNTP_Intrinsic<[IntrSpeculatable]>;
}

//
// FFR manipulation
//

// FIXME: Add target guards
def int_aarch64_sve_rdffr   : ClangBuiltin<"__builtin_sve_svrdffr">,   DefaultAttrsIntrinsic<[llvm_nxv16i1_ty], [], [IntrRead<[FFR]>, IntrReadOnly]>;
def int_aarch64_sve_rdffr_z : ClangBuiltin<"__builtin_sve_svrdffr_z">, DefaultAttrsIntrinsic<[llvm_nxv16i1_ty], [llvm_nxv16i1_ty], [IntrRead<[FFR]>, IntrReadOnly]>;
def int_aarch64_sve_setffr  : ClangBuiltin<"__builtin_sve_svsetffr">,  DefaultAttrsIntrinsic<[], [], [IntrWrite<[FFR]>, IntrWriteOnly]>;
def int_aarch64_sve_wrffr   : ClangBuiltin<"__builtin_sve_svwrffr">,   DefaultAttrsIntrinsic<[], [llvm_nxv16i1_ty], [IntrWrite<[FFR]>, IntrWriteOnly]>;

//
// Saturating scalar arithmetic
//

let TargetGuards = [Variant<"sve", "sme", I16_Overloads>] in {
  def int_aarch64_sve_sqdech : Guarded_AdvSIMD_SVE_SaturatingWithPattern_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqinch : Guarded_AdvSIMD_SVE_SaturatingWithPattern_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqdech : Guarded_AdvSIMD_SVE_SaturatingWithPattern_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqinch : Guarded_AdvSIMD_SVE_SaturatingWithPattern_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve", "sme", I32_Overloads>] in {
  def int_aarch64_sve_sqdecw : Guarded_AdvSIMD_SVE_SaturatingWithPattern_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqincw : Guarded_AdvSIMD_SVE_SaturatingWithPattern_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqdecw : Guarded_AdvSIMD_SVE_SaturatingWithPattern_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqincw : Guarded_AdvSIMD_SVE_SaturatingWithPattern_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve", "sme", I64_Overloads>] in {
  def int_aarch64_sve_sqdecd : Guarded_AdvSIMD_SVE_SaturatingWithPattern_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqincd : Guarded_AdvSIMD_SVE_SaturatingWithPattern_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqdecd : Guarded_AdvSIMD_SVE_SaturatingWithPattern_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqincd : Guarded_AdvSIMD_SVE_SaturatingWithPattern_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve", "sme", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_sqdecp : Guarded_AdvSIMD_SVE_Saturating_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqincp : Guarded_AdvSIMD_SVE_Saturating_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqdecp : Guarded_AdvSIMD_SVE_Saturating_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqincp : Guarded_AdvSIMD_SVE_Saturating_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve", "sme", EmptyOverload>] in {
  def int_aarch64_sve_sqdecb_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqdecb_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqdech_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqdech_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqdecw_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqdecw_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqdecd_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqdecd_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqincb_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqincb_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqinch_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqinch_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqincw_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqincw_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqincd_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqincd_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqdecb_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqdecb_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqdech_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqdech_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqdecw_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqdecw_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqdecd_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqdecd_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqincb_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqincb_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqinch_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqinch_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqincw_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqincw_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqincd_n32 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqincd_n64 : Guarded_AdvSIMD_SVE_SaturatingWithPattern_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
}

//
// Reversal
//

let TargetGuards = [Variant<"sve", "sme", I8_I16_I32_I64_Overloads>] in
def int_aarch64_sve_rbit : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
let TargetGuards = [Variant<"sve", "sme", I16_I32_I64_Overloads>] in
def int_aarch64_sve_revb : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
let TargetGuards = [Variant<"sve", "sme", I32_I64_Overloads>] in
def int_aarch64_sve_revh : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
let TargetGuards = [Variant<"sve", "sme", I64_Overloads>] in
def int_aarch64_sve_revw : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;

//
// Permutations and selection
//

let TargetGuards = [Variant<"sve", "sme", AllPackedData_Overloads>] in {
  def int_aarch64_sve_clasta    : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_clasta_n  : Guarded_AdvSIMD_SVE_ReduceWithInit_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_clastb    : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_clastb_n  : Guarded_AdvSIMD_SVE_ReduceWithInit_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_dupq_lane : Guarded_AdvSIMD_SVE_DUPQ_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_ext       : Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_lasta     : Guarded_AdvSIMD_SVE_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_lastb     : Guarded_AdvSIMD_SVE_Reduce_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_splice    : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_tbl       : Guarded_AdvSIMD_SVE_TBL_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [
  Variant<"sve", "sme2p2", I32_I64_F32_F64_Overloads>,
  Variant<"sve,(sve2p2|sme2p2)", "sme2p2", I8_I16_BF16_F16_Overloads>,
] in
def int_aarch64_sve_compact   : Guarded_AdvSIMD_Pred1VectorArg_Intrinsic<[IntrSpeculatable]>;

let TargetGuards = [Variant<"sve,(sve2p2|sme2p2)", "sme2p2", AllPackedData_Overloads>] in
def int_aarch64_sve_expand    : Guarded_AdvSIMD_Pred1VectorArg_Intrinsic<[IntrSpeculatable]>;

let TargetGuards = [Variant<"sve,(sve2p1|sme2p1)", "sve2p1|sme2p1", AllPackedData_Overloads>] in
def int_aarch64_sve_dup_laneq : Guarded_SVE2_1VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;

let TargetGuards = [Variant<"sve", "sme", !listconcat([[llvm_nxv16i1_ty]], AllPackedData_Overloads)>] in {
  def int_aarch64_sve_sel       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_trn1      : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_trn2      : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uzp1      : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uzp2      : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_zip1      : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_zip2      : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve", "sme", EmptyOverload>] in {
  def int_aarch64_sve_rev_b16   : Guarded_AdvSIMD_SVE_2SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_rev_b32   : Guarded_AdvSIMD_SVE_2SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_rev_b64   : Guarded_AdvSIMD_SVE_2SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_trn1_b16  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_trn1_b32  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_trn1_b64  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_trn2_b16  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_trn2_b32  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_trn2_b64  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uzp1_b16  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uzp1_b32  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uzp1_b64  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uzp2_b16  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uzp2_b32  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uzp2_b64  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_zip1_b16  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_zip1_b32  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_zip1_b64  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_zip2_b16  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_zip2_b32  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_zip2_b64  : Guarded_AdvSIMD_SVE_3SVBoolArg_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve", "sme", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_sunpkhi   : Guarded_AdvSIMD_SVE_Unpack_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sunpklo   : Guarded_AdvSIMD_SVE_Unpack_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uunpkhi   : Guarded_AdvSIMD_SVE_Unpack_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uunpklo   : Guarded_AdvSIMD_SVE_Unpack_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [NonStreamingVariant<"f64mm", AllPackedData_Overloads>] in {
  def int_aarch64_sve_trn1q     : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_trn2q     : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uzp1q     : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uzp2q     : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_zip1q     : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_zip2q     : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
}

//
// Logical operations
//

let TargetGuards = [Variant<"sve", "sme", I8_I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_and  : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_and_u: Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_bic  : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_bic_u: Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cnot : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_eor  : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_eor_u: Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_not  : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_orr  : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_orr_u: Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
}

//
// Conversion
//

let TargetGuards = [Variant<"sve", "sme", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_sxtb : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uxtb : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve", "sme", I32_I64_Overloads>] in {
  def int_aarch64_sve_sxth : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uxth : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve", "sme", I64_Overloads>] in {
  def int_aarch64_sve_sxtw : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uxtw : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
}

//
// While comparisons
//

defvar PredicateAndIntegerScalarTypes = [
  [llvm_nxv2i1_ty, llvm_i32_ty],
  [llvm_nxv4i1_ty, llvm_i32_ty],
  [llvm_nxv8i1_ty, llvm_i32_ty],
  [llvm_nxv16i1_ty, llvm_i32_ty],
  [llvm_nxv2i1_ty, llvm_i64_ty],
  [llvm_nxv4i1_ty, llvm_i64_ty],
  [llvm_nxv8i1_ty, llvm_i64_ty],
  [llvm_nxv16i1_ty, llvm_i64_ty]
];

let TargetGuards = [Variant<"sve", "sme", PredicateAndIntegerScalarTypes>] in {
  def int_aarch64_sve_whilele : Guarded_AdvSIMD_SVE_WHILE_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_whilelo : Guarded_AdvSIMD_SVE_WHILE_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_whilels : Guarded_AdvSIMD_SVE_WHILE_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_whilelt : Guarded_AdvSIMD_SVE_WHILE_Intrinsic<[IntrSpeculatable]>;
}
let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", PredicateAndIntegerScalarTypes>] in {
  def int_aarch64_sve_whilege : Guarded_AdvSIMD_SVE_WHILE_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_whilegt : Guarded_AdvSIMD_SVE_WHILE_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_whilehs : Guarded_AdvSIMD_SVE_WHILE_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_whilehi : Guarded_AdvSIMD_SVE_WHILE_Intrinsic<[IntrSpeculatable]>;
}

//
// Floating-point arithmetic
//

let TargetGuards = [Variant<"sve", "sme", !listconcat(BF16_Overloads, F16_F32_F64_Overloads)>] in {
  // The following intrinsics may equally work on bfloat types,
  // hence no '+bf16' required on the tests.
  //
  // FIXME: Lock this list down further, as not all intrinsics
  // support bfloat types.
  def int_aarch64_sve_fabd       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fabd_u     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fabs       : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic;
  def int_aarch64_sve_fcadd      : Guarded_AdvSIMD_SVE_CADD_Intrinsic;
  def int_aarch64_sve_fcmla      : Guarded_AdvSIMD_SVE_CMLA_Intrinsic;
  def int_aarch64_sve_fdiv       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fdiv_u     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fdivr      : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fmad       : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic;
  def int_aarch64_sve_fmsb       : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic;
  def int_aarch64_sve_fmulx      : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fmulx_u    : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fneg       : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic;
  def int_aarch64_sve_fnmad      : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic;
  def int_aarch64_sve_fnmla      : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic;
  def int_aarch64_sve_fnmla_u    : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic;
  def int_aarch64_sve_fnmls      : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic;
  def int_aarch64_sve_fnmls_u    : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic;
  def int_aarch64_sve_fnmsb      : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic;
  def int_aarch64_sve_frecpe_x   : Guarded_AdvSIMD_1VectorArg_Intrinsic;
  def int_aarch64_sve_frecps_x   : Guarded_AdvSIMD_2VectorArg_Intrinsic;
  def int_aarch64_sve_frecpx     : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic;
  def int_aarch64_sve_frinta     : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic;
  def int_aarch64_sve_frinti     : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic;
  def int_aarch64_sve_frintm     : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic;
  def int_aarch64_sve_frintn     : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic;
  def int_aarch64_sve_frintp     : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic;
  def int_aarch64_sve_frintx     : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic;
  def int_aarch64_sve_frintz     : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic;
  def int_aarch64_sve_frsqrte_x  : Guarded_AdvSIMD_1VectorArg_Intrinsic;
  def int_aarch64_sve_frsqrts_x  : Guarded_AdvSIMD_2VectorArg_Intrinsic;
  def int_aarch64_sve_fsqrt      : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic;
  def int_aarch64_sve_fsubr      : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
}

let TargetGuards = [
  Variant<"sve,sve-b16b16", "sme2,sve-b16b16", BF16_Overloads>,
  Variant<"sve", "sme", F16_F32_F64_Overloads>,
] in
def int_aarch64_sve_fadd       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;

let TargetGuards = [
  Variant<"sve", "sme", F32_F64_Overloads>,
  Variant<"sve,sve-b16b16", "sme2,sve-b16b16", BF16_Overloads>,
  Variant<"sve", "sme", F16_Overloads>,
] in {
  def int_aarch64_sve_fadd_u     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fmax       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fmax_u     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fmaxnm     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fmaxnm_u   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fmin       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fmin_u     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fminnm     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fminnm_u   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fmla       : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic;
  def int_aarch64_sve_fmla_lane  : Guarded_AdvSIMD_3VectorArgIndexed_Intrinsic;
  def int_aarch64_sve_fmla_u     : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic;
  def int_aarch64_sve_fmls       : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic;
  def int_aarch64_sve_fmls_lane  : Guarded_AdvSIMD_3VectorArgIndexed_Intrinsic;
  def int_aarch64_sve_fmls_u     : Guarded_AdvSIMD_Pred3VectorArg_Intrinsic;
  def int_aarch64_sve_fmul       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fmul_lane  : Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic;
  def int_aarch64_sve_fmul_u     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fsub       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fsub_u     : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
}

let TargetGuards = [Variant<"sve", "sme", F16_F32_Overloads>] in
def int_aarch64_sve_fcmla_lane : Guarded_AdvSIMD_SVE_CMLA_LANE_Intrinsic;

let TargetGuards = [Variant<"sve", "sme2,ssve-fexpa", F16_F32_F64_Overloads>] in
def int_aarch64_sve_fexpa_x    : Guarded_AdvSIMD_SVE_EXPA_Intrinsic;

let TargetGuards = [Variant<"sve,(sve2p2|sme2p2)", "sve2p2|sme2p2", F32_F64_Overloads>] in {
  def int_aarch64_sve_frint32x   : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic;
  def int_aarch64_sve_frint32z   : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic;
  def int_aarch64_sve_frint64x   : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic;
  def int_aarch64_sve_frint64z   : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic;
}

let TargetGuards = [
  Variant<"sve", "sme", F32_F64_Overloads>,
  Variant<"sve,sve-bfscale", "sve-bfscale,sme2", BF16_Overloads>,
  Variant<"sve", "sme", F16_Overloads>,
] in
def int_aarch64_sve_fscale     : Guarded_AdvSIMD_SVE_SCALE_Intrinsic;

let TargetGuards = [NonStreamingVariant<"sve", F16_F32_F64_Overloads>] in {
  def int_aarch64_sve_ftmad_x    : Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic;
  def int_aarch64_sve_ftsmul_x   : Guarded_AdvSIMD_SVE_TSMUL_Intrinsic;
  def int_aarch64_sve_ftssel_x   : Guarded_AdvSIMD_SVE_TSMUL_Intrinsic;
}

//
// Floating-point reductions
//

let TargetGuards = [NonStreamingVariant<"sve", F16_F32_F64_Overloads>] in
def int_aarch64_sve_fadda   : Guarded_AdvSIMD_SVE_ReduceWithInit_Intrinsic;
let TargetGuards = [Variant<"sve", "sme", F16_F32_F64_Overloads>] in {
  def int_aarch64_sve_faddv   : Guarded_AdvSIMD_SVE_Reduce_Intrinsic;
  def int_aarch64_sve_fmaxv   : Guarded_AdvSIMD_SVE_Reduce_Intrinsic;
  def int_aarch64_sve_fmaxnmv : Guarded_AdvSIMD_SVE_Reduce_Intrinsic;
  def int_aarch64_sve_fminv   : Guarded_AdvSIMD_SVE_Reduce_Intrinsic;
  def int_aarch64_sve_fminnmv : Guarded_AdvSIMD_SVE_Reduce_Intrinsic;
}

defvar SveFaddqvTypes = [
  [llvm_v2f64_ty, llvm_nxv2f64_ty],
  [llvm_v4f32_ty, llvm_nxv4f32_ty],
  [llvm_v8f16_ty, llvm_nxv8f16_ty],
];
let TargetGuards = [Variant<"sve,(sve2p1|sme2p1)", "sve2p1|sme2p1", SveFaddqvTypes>] in {
  def int_aarch64_sve_faddqv   : Guarded_AdvSIMD_SVE_V128_Reduce_Intrinsic;
  def int_aarch64_sve_fmaxnmqv : Guarded_AdvSIMD_SVE_V128_Reduce_Intrinsic;
  def int_aarch64_sve_fminnmqv : Guarded_AdvSIMD_SVE_V128_Reduce_Intrinsic;
  def int_aarch64_sve_fmaxqv   : Guarded_AdvSIMD_SVE_V128_Reduce_Intrinsic;
  def int_aarch64_sve_fminqv   : Guarded_AdvSIMD_SVE_V128_Reduce_Intrinsic;
}

//
// Floating-point conversions
//

defvar SveFcvtzsTypes = [
  [llvm_nxv2i64_ty, llvm_nxv2f64_ty],
  [llvm_nxv4i32_ty, llvm_nxv4f32_ty],
  [llvm_nxv8i16_ty, llvm_nxv8f16_ty],
];

let TargetGuards = [Variant<"sve", "sme", SveFcvtzsTypes>] in {
  def int_aarch64_sve_fcvtzs : Guarded_AdvSIMD_SVE_FCVTZS_Intrinsic;
  def int_aarch64_sve_fcvtzu : Guarded_AdvSIMD_SVE_FCVTZS_Intrinsic;
}

defvar SveScvtfTypes = [
  [llvm_nxv2f64_ty, llvm_nxv2i64_ty],
  [llvm_nxv4f32_ty, llvm_nxv4i32_ty],
  [llvm_nxv8f16_ty, llvm_nxv8i16_ty],
];

let TargetGuards = [Variant<"sve", "sme", SveScvtfTypes>] in {
  def int_aarch64_sve_scvtf  : Guarded_AdvSIMD_SVE_SCVTF_Intrinsic;
  def int_aarch64_sve_ucvtf  : Guarded_AdvSIMD_SVE_SCVTF_Intrinsic;
}

//
// Floating-point comparisons
//

let TargetGuards = [Variant<"sve", "sme", F16_F32_F64_Overloads>] in {
  def int_aarch64_sve_facge : Guarded_AdvSIMD_SVE_Compare_Intrinsic;
  def int_aarch64_sve_facgt : Guarded_AdvSIMD_SVE_Compare_Intrinsic;
  def int_aarch64_sve_fcmpeq : Guarded_AdvSIMD_SVE_Compare_Intrinsic;
  def int_aarch64_sve_fcmpge : Guarded_AdvSIMD_SVE_Compare_Intrinsic;
  def int_aarch64_sve_fcmpgt : Guarded_AdvSIMD_SVE_Compare_Intrinsic;
  def int_aarch64_sve_fcmpne : Guarded_AdvSIMD_SVE_Compare_Intrinsic;
  def int_aarch64_sve_fcmpuo : Guarded_AdvSIMD_SVE_Compare_Intrinsic;
}

let TargetGuards = [Variant<"sve", "sme", EmptyOverload>] in {
  def int_aarch64_sve_fcvtzs_i32f16   : Guarded_Builtin_SVCVT<llvm_nxv4i32_ty, llvm_nxv4i1_ty, llvm_nxv8f16_ty>;
  def int_aarch64_sve_fcvtzs_i32f64   : Guarded_Builtin_SVCVT<llvm_nxv4i32_ty, llvm_nxv2i1_ty, llvm_nxv2f64_ty>;
  def int_aarch64_sve_fcvtzs_i64f16   : Guarded_Builtin_SVCVT<llvm_nxv2i64_ty, llvm_nxv2i1_ty, llvm_nxv8f16_ty>;
  def int_aarch64_sve_fcvtzs_i64f32   : Guarded_Builtin_SVCVT<llvm_nxv2i64_ty, llvm_nxv2i1_ty, llvm_nxv4f32_ty>;
  def int_aarch64_sve_fcvtzu_i32f16   : Guarded_Builtin_SVCVT<llvm_nxv4i32_ty, llvm_nxv4i1_ty, llvm_nxv8f16_ty>;
  def int_aarch64_sve_fcvtzu_i32f64   : Guarded_Builtin_SVCVT<llvm_nxv4i32_ty, llvm_nxv2i1_ty, llvm_nxv2f64_ty>;
  def int_aarch64_sve_fcvtzu_i64f16   : Guarded_Builtin_SVCVT<llvm_nxv2i64_ty, llvm_nxv2i1_ty, llvm_nxv8f16_ty>;
  def int_aarch64_sve_fcvtzu_i64f32   : Guarded_Builtin_SVCVT<llvm_nxv2i64_ty, llvm_nxv2i1_ty, llvm_nxv4f32_ty>;
  def int_aarch64_sve_fcvt_f16f32     : Guarded_Builtin_SVCVT<llvm_nxv8f16_ty, llvm_nxv4i1_ty, llvm_nxv4f32_ty>;
  def int_aarch64_sve_fcvt_f16f64     : Guarded_Builtin_SVCVT<llvm_nxv8f16_ty, llvm_nxv2i1_ty, llvm_nxv2f64_ty>;
  def int_aarch64_sve_fcvt_f32f64     : Guarded_Builtin_SVCVT<llvm_nxv4f32_ty, llvm_nxv2i1_ty, llvm_nxv2f64_ty>;
  def int_aarch64_sve_fcvt_f32f16     : Guarded_Builtin_SVCVT<llvm_nxv4f32_ty, llvm_nxv4i1_ty, llvm_nxv8f16_ty>;
  def int_aarch64_sve_fcvt_f64f16     : Guarded_Builtin_SVCVT<llvm_nxv2f64_ty, llvm_nxv2i1_ty, llvm_nxv8f16_ty>;
  def int_aarch64_sve_fcvt_f64f32     : Guarded_Builtin_SVCVT<llvm_nxv2f64_ty, llvm_nxv2i1_ty, llvm_nxv4f32_ty>;
  def int_aarch64_sve_scvtf_f16i32    : Guarded_Builtin_SVCVT<llvm_nxv8f16_ty, llvm_nxv4i1_ty, llvm_nxv4i32_ty>;
  def int_aarch64_sve_scvtf_f16i64    : Guarded_Builtin_SVCVT<llvm_nxv8f16_ty, llvm_nxv2i1_ty, llvm_nxv2i64_ty>;
  def int_aarch64_sve_scvtf_f32i64    : Guarded_Builtin_SVCVT<llvm_nxv4f32_ty, llvm_nxv2i1_ty, llvm_nxv2i64_ty>;
  def int_aarch64_sve_scvtf_f64i32    : Guarded_Builtin_SVCVT<llvm_nxv2f64_ty, llvm_nxv2i1_ty, llvm_nxv4i32_ty>;
  def int_aarch64_sve_ucvtf_f16i32    : Guarded_Builtin_SVCVT<llvm_nxv8f16_ty, llvm_nxv4i1_ty, llvm_nxv4i32_ty>;
  def int_aarch64_sve_ucvtf_f16i64    : Guarded_Builtin_SVCVT<llvm_nxv8f16_ty, llvm_nxv2i1_ty, llvm_nxv2i64_ty>;
  def int_aarch64_sve_ucvtf_f32i64    : Guarded_Builtin_SVCVT<llvm_nxv4f32_ty, llvm_nxv2i1_ty, llvm_nxv2i64_ty>;
  def int_aarch64_sve_ucvtf_f64i32    : Guarded_Builtin_SVCVT<llvm_nxv2f64_ty, llvm_nxv2i1_ty, llvm_nxv4i32_ty>;
}

let TargetGuards = [Variant<"sve,bf16", "sme,bf16", EmptyOverload>] in {
  def int_aarch64_sve_fcvt_bf16f32_v2   : Guarded_Builtin_SVCVT<llvm_nxv8bf16_ty, llvm_nxv4i1_ty, llvm_nxv4f32_ty>;
  def int_aarch64_sve_fcvtnt_bf16f32_v2 : Guarded_Builtin_SVCVT<llvm_nxv8bf16_ty, llvm_nxv4i1_ty, llvm_nxv4f32_ty>;
}

let TargetGuards = [Variant<"sve,(sve2p2|sme2p2)", "sve2p2|sme2p2", EmptyOverload>] in {
  def int_aarch64_sve_fcvtnt_z_bf16f32  : Guarded_Builtin_SVCVT<llvm_nxv8bf16_ty, llvm_nxv4i1_ty, llvm_nxv4f32_ty>;
  def int_aarch64_sve_fcvtnt_z_f16f32 : Guarded_Builtin_SVCVT<llvm_nxv8f16_ty, llvm_nxv4i1_ty, llvm_nxv4f32_ty>;
  def int_aarch64_sve_fcvtnt_z_f32f64 : Guarded_Builtin_SVCVT<llvm_nxv4f32_ty, llvm_nxv2i1_ty, llvm_nxv2f64_ty>;
  def int_aarch64_sve_fcvtxnt_z_f32f64 : Guarded_Builtin_SVCVT<llvm_nxv4f32_ty, llvm_nxv2i1_ty, llvm_nxv2f64_ty>;
}

let TargetGuards = [
  Variant<"sve,(sve2|sme)", "sme", EmptyOverload>,
] in {
  def int_aarch64_sve_fcvtlt_f32f16   : Guarded_Builtin_SVCVT<llvm_nxv4f32_ty, llvm_nxv4i1_ty, llvm_nxv8f16_ty>;
  def int_aarch64_sve_fcvtlt_f64f32   : Guarded_Builtin_SVCVT<llvm_nxv2f64_ty, llvm_nxv2i1_ty, llvm_nxv4f32_ty>;
}

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", EmptyOverload>] in {
  def int_aarch64_sve_fcvtnt_f16f32   : Guarded_Builtin_SVCVT<llvm_nxv8f16_ty, llvm_nxv4i1_ty, llvm_nxv4f32_ty>;
  def int_aarch64_sve_fcvtnt_f32f64   : Guarded_Builtin_SVCVT<llvm_nxv4f32_ty, llvm_nxv2i1_ty, llvm_nxv2f64_ty>;
  def int_aarch64_sve_fcvtx_f32f64     : Guarded_Builtin_SVCVT<llvm_nxv4f32_ty, llvm_nxv2i1_ty, llvm_nxv2f64_ty>;
  def int_aarch64_sve_fcvtxnt_f32f64   : Guarded_Builtin_SVCVT<llvm_nxv4f32_ty, llvm_nxv2i1_ty, llvm_nxv2f64_ty>;
}

let TargetGuards = [Variant<"sve", "sme", [[llvm_nxv16i1_ty]]>] in {
  //
  // Predicate operations
  //
  def int_aarch64_sve_and_z   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_bic_z   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_brka    : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_brka_z  : Guarded_AdvSIMD_Pred1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_brkb    : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_brkb_z  : Guarded_AdvSIMD_Pred1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_brkn_z  : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_brkpa_z : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_brkpb_z : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_eor_z   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_nand_z  : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_nor_z   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_orn_z   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_orr_z   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_pfirst  : Guarded_AdvSIMD_Pred1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_punpkhi : Guarded_AdvSIMD_SVE_PUNPKHI_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_punpklo : Guarded_AdvSIMD_SVE_PUNPKHI_Intrinsic<[IntrSpeculatable]>;

  //
  // Testing predicates
  //
  def int_aarch64_sve_ptest_any   : Guarded_AdvSIMD_SVE_PTEST_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_ptest_first : Guarded_AdvSIMD_SVE_PTEST_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_ptest_last  : Guarded_AdvSIMD_SVE_PTEST_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve", "sme", Predicate_Overloads>] in {
  //
  // Predicate creation
  //
  def int_aarch64_sve_ptrue : Guarded_AdvSIMD_SVE_PTRUE_Intrinsic;

  def int_aarch64_sve_pnext   : Guarded_AdvSIMD_Pred1VectorArg_Intrinsic<[IntrSpeculatable]>;

  //
  // Counting active elements
  //
  def int_aarch64_sve_cntp : Guarded_AdvSIMD_SVE_CNTP_Intrinsic<[IntrSpeculatable]>;

  //
  // Saturating inc/dec of active elements
  //
  def int_aarch64_sve_sqdecp_n32 : Guarded_AdvSIMD_SVE_Saturating_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqdecp_n64 : Guarded_AdvSIMD_SVE_Saturating_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqincp_n32 : Guarded_AdvSIMD_SVE_Saturating_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_sqincp_n64 : Guarded_AdvSIMD_SVE_Saturating_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqdecp_n32 : Guarded_AdvSIMD_SVE_Saturating_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqdecp_n64 : Guarded_AdvSIMD_SVE_Saturating_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqincp_n32 : Guarded_AdvSIMD_SVE_Saturating_N_Intrinsic<llvm_i32_ty, [IntrSpeculatable]>;
  def int_aarch64_sve_uqincp_n64 : Guarded_AdvSIMD_SVE_Saturating_N_Intrinsic<llvm_i64_ty, [IntrSpeculatable]>;
}

//
// Reinterpreting data
//

defvar ConvertToFromSvboolTypes = [
  [llvm_nxv16i1_ty],
  [llvm_nxv8i1_ty],
  [llvm_nxv4i1_ty],
  [llvm_nxv2i1_ty],
  [llvm_nxv1i1_ty],
];
let TargetGuards = [Variant<"sve", "sme", ConvertToFromSvboolTypes>] in
  def int_aarch64_sve_convert_from_svbool : GuardedDefaultIntrinsic<
      [llvm_anyvector_ty], [llvm_nxv16i1_ty], [IntrNoMem, IntrSpeculatable]>;

let TargetGuards = [Variant<"sve", "sme", ConvertToFromSvboolTypes>] in
  def int_aarch64_sve_convert_to_svbool : GuardedDefaultIntrinsic<
      [llvm_nxv16i1_ty], [llvm_anyvector_ty], [IntrNoMem, IntrSpeculatable]>;

let TargetGuards = [Variant<"sve,(sve2p1|sme2)", "sve2p1|sme2", [[]]>] in
  def int_aarch64_sve_convert_from_svcount : GuardedDefaultIntrinsic<
      [llvm_nxv16i1_ty], [llvm_aarch64_svcount_ty], [IntrNoMem, IntrSpeculatable]>;

let TargetGuards = [Variant<"sve,(sve2p1|sme2)", "sve2p1|sme2", [[]]>] in
  def int_aarch64_sve_convert_to_svcount : GuardedDefaultIntrinsic<
      [llvm_aarch64_svcount_ty], [llvm_nxv16i1_ty], [IntrNoMem, IntrSpeculatable]>;

//
// Gather loads: scalar base + vector offsets
//

// 64 bit unscaled offsets
let TargetGuards = [NonStreamingVariant<"sve", DataPTR_OverloadPairs>] in
def int_aarch64_sve_ld1_gather : Guarded_AdvSIMD_GatherLoad_SV_64b_Offsets_Intrinsic;

// 64 bit scaled offsets
let TargetGuards = [NonStreamingVariant<"sve", DataPTR_OverloadPairs>] in
def int_aarch64_sve_ld1_gather_index : Guarded_AdvSIMD_GatherLoad_SV_64b_Offsets_Intrinsic;

// 32 bit unscaled offsets, sign (sxtw) or zero (zxtw) extended to 64 bits
let TargetGuards = [NonStreamingVariant<"sve", DataPTR_OverloadPairs>] in {
  def int_aarch64_sve_ld1_gather_sxtw : Guarded_AdvSIMD_GatherLoad_SV_32b_Offsets_Intrinsic;
  def int_aarch64_sve_ld1_gather_uxtw : Guarded_AdvSIMD_GatherLoad_SV_32b_Offsets_Intrinsic;
}

// 32 bit scaled offsets, sign (sxtw) or zero (zxtw) extended to 64 bits
let TargetGuards = [NonStreamingVariant<"sve", DataPTR_OverloadPairs>] in {
  def int_aarch64_sve_ld1_gather_sxtw_index : Guarded_AdvSIMD_GatherLoad_SV_32b_Offsets_Intrinsic;
  def int_aarch64_sve_ld1_gather_uxtw_index : Guarded_AdvSIMD_GatherLoad_SV_32b_Offsets_Intrinsic;
}

// 128-bit loads, scaled offsets (indices)
let TargetGuards = [NonStreamingVariant<"sve2p1", PackedDataPTR_OverloadPairs>] in
def int_aarch64_sve_ld1q_gather_index : Guarded_AdvSIMD_GatherLoadQ_SV_Intrinsic;

// 128-bit loads, unscaled offsets
let TargetGuards = [NonStreamingVariant<"sve2p1", DataPTR_OverloadPairs>] in
def int_aarch64_sve_ld1q_gather_vector_offset : Guarded_AdvSIMD_GatherLoadQ_SV_Intrinsic;

//
// Gather loads: vector base + scalar offset
//

let TargetGuards = [NonStreamingVariant<"sve", DataI32_DataI64_OverloadPairs>] in
def int_aarch64_sve_ld1_gather_scalar_offset : Guarded_AdvSIMD_GatherLoad_VS_Intrinsic;

// 128-bit loads, unscaled offsets
let TargetGuards = [NonStreamingVariant<"sve2p1", PackedDataI64_OverloadPairs>] in
def int_aarch64_sve_ld1q_gather_scalar_offset : Guarded_AdvSIMD_GatherLoadQ_VS_Intrinsic;

//
// First-faulting gather loads: scalar base + vector offsets
//

// 64 bit unscaled offsets
let TargetGuards = [NonStreamingVariant<"sve", DataPTR_OverloadPairs>] in
def int_aarch64_sve_ldff1_gather : Guarded_AdvSIMD_GatherLoad_SV_64b_Offsets_WriteFFR_Intrinsic;

// 64 bit scaled offsets
let TargetGuards = [NonStreamingVariant<"sve", DataPTR_OverloadPairs>] in
def int_aarch64_sve_ldff1_gather_index : Guarded_AdvSIMD_GatherLoad_SV_64b_Offsets_WriteFFR_Intrinsic;

// 32 bit unscaled offsets, sign (sxtw) or zero (uxtw) extended to 64 bits
let TargetGuards = [NonStreamingVariant<"sve", DataPTR_OverloadPairs>] in {
  def int_aarch64_sve_ldff1_gather_sxtw : Guarded_AdvSIMD_GatherLoad_SV_32b_Offsets_WriteFFR_Intrinsic;
  def int_aarch64_sve_ldff1_gather_uxtw : Guarded_AdvSIMD_GatherLoad_SV_32b_Offsets_WriteFFR_Intrinsic;
}

// 32 bit scaled offsets, sign (sxtw) or zero (uxtw) extended to 64 bits
let TargetGuards = [NonStreamingVariant<"sve", DataPTR_OverloadPairs>] in {
  def int_aarch64_sve_ldff1_gather_sxtw_index : Guarded_AdvSIMD_GatherLoad_SV_32b_Offsets_WriteFFR_Intrinsic;
  def int_aarch64_sve_ldff1_gather_uxtw_index : Guarded_AdvSIMD_GatherLoad_SV_32b_Offsets_WriteFFR_Intrinsic;
}

//
// First-faulting gather loads: vector base + scalar offset
//

let TargetGuards = [NonStreamingVariant<"sve", DataI32_DataI64_OverloadPairs>] in
def int_aarch64_sve_ldff1_gather_scalar_offset : Guarded_AdvSIMD_GatherLoad_VS_WriteFFR_Intrinsic;

//
// Non-temporal gather loads: scalar base + vector offsets
//

// 64 bit unscaled offsets
let TargetGuards = [NonStreamingVariant<"sve2", DataPTR_OverloadPairs>] in
def int_aarch64_sve_ldnt1_gather : Guarded_AdvSIMD_GatherLoad_SV_64b_Offsets_Intrinsic;

// 64 bit indices
let TargetGuards = [NonStreamingVariant<"sve2", DataPTR_OverloadPairs>] in
def int_aarch64_sve_ldnt1_gather_index : Guarded_AdvSIMD_GatherLoad_SV_64b_Offsets_Intrinsic;

// 32 bit unscaled offsets, zero (zxtw) extended to 64 bits
let TargetGuards = [NonStreamingVariant<"sve2", DataPTR_OverloadPairs>] in
def int_aarch64_sve_ldnt1_gather_uxtw : Guarded_AdvSIMD_GatherLoad_SV_32b_Offsets_Intrinsic;

//
// Non-temporal gather loads: vector base + scalar offset
//

let TargetGuards = [NonStreamingVariant<"sve2", DataI32_DataI64_OverloadPairs>] in
def int_aarch64_sve_ldnt1_gather_scalar_offset  : Guarded_AdvSIMD_GatherLoad_VS_Intrinsic;

//
// Scatter stores: scalar base + vector offsets
//

// 64 bit unscaled offsets
let TargetGuards = [NonStreamingVariant<"sve", DataPTR_OverloadPairs>] in
def int_aarch64_sve_st1_scatter : Guarded_AdvSIMD_ScatterStore_SV_64b_Offsets_Intrinsic;

// 64 bit scaled offsets
let TargetGuards = [NonStreamingVariant<"sve", DataPTR_OverloadPairs>] in
def int_aarch64_sve_st1_scatter_index
    : Guarded_AdvSIMD_ScatterStore_SV_64b_Offsets_Intrinsic;

// 32 bit unscaled offsets, sign (sxtw) or zero (zxtw) extended to 64 bits
let TargetGuards = [NonStreamingVariant<"sve", DataPTR_OverloadPairs>] in {
  def int_aarch64_sve_st1_scatter_sxtw
    : Guarded_AdvSIMD_ScatterStore_SV_32b_Offsets_Intrinsic;
  def int_aarch64_sve_st1_scatter_uxtw
    : Guarded_AdvSIMD_ScatterStore_SV_32b_Offsets_Intrinsic;
}

// 32 bit scaled offsets, sign (sxtw) or zero (zxtw) extended to 64 bits
let TargetGuards = [NonStreamingVariant<"sve", DataPTR_OverloadPairs>] in {
  def int_aarch64_sve_st1_scatter_sxtw_index
    : Guarded_AdvSIMD_ScatterStore_SV_32b_Offsets_Intrinsic;
  def int_aarch64_sve_st1_scatter_uxtw_index
    : Guarded_AdvSIMD_ScatterStore_SV_32b_Offsets_Intrinsic;
}

// 128-bit stores, scaled offsets (indices)
let TargetGuards = [NonStreamingVariant<"sve2p1", PackedDataPTR_OverloadPairs>] in
def int_aarch64_sve_st1q_scatter_index : Guarded_AdvSIMD_ScatterStoreQ_SV_Intrinsic;

// 128-bit stores, unscaled offsets
let TargetGuards = [NonStreamingVariant<"sve2p1", PackedDataPTR_OverloadPairs>] in
def int_aarch64_sve_st1q_scatter_vector_offset : Guarded_AdvSIMD_ScatterStoreQ_SV_Intrinsic;

//
// Scatter stores: vector base + scalar offset
//

let TargetGuards = [NonStreamingVariant<"sve", DataI32_DataI64_OverloadPairs>] in
def int_aarch64_sve_st1_scatter_scalar_offset : Guarded_AdvSIMD_ScatterStore_VS_Intrinsic;

// 128-bit stores, unscaled offsets
let TargetGuards = [NonStreamingVariant<"sve2p1", PackedDataI64_OverloadPairs>] in
def int_aarch64_sve_st1q_scatter_scalar_offset : Guarded_AdvSIMD_ScatterStoreQ_VS_Intrinsic;

//
// Non-temporal scatter stores: scalar base + vector offsets
//

// 64 bit unscaled offsets
let TargetGuards = [NonStreamingVariant<"sve2", DataPTR_OverloadPairs>] in
def int_aarch64_sve_stnt1_scatter : Guarded_AdvSIMD_ScatterStore_SV_64b_Offsets_Intrinsic;

// 64 bit indices
let TargetGuards = [NonStreamingVariant<"sve2", DataPTR_OverloadPairs>] in
def int_aarch64_sve_stnt1_scatter_index
    : Guarded_AdvSIMD_ScatterStore_SV_64b_Offsets_Intrinsic;

// 32 bit unscaled offsets, zero (zxtw) extended to 64 bits
let TargetGuards = [NonStreamingVariant<"sve2", DataPTR_OverloadPairs>] in
def int_aarch64_sve_stnt1_scatter_uxtw : Guarded_AdvSIMD_ScatterStore_SV_32b_Offsets_Intrinsic;

//
// Non-temporal scatter stores: vector base + scalar offset
//

let TargetGuards = [NonStreamingVariant<"sve2", DataI32_DataI64_OverloadPairs>] in
def int_aarch64_sve_stnt1_scatter_scalar_offset  : Guarded_AdvSIMD_ScatterStore_VS_Intrinsic;

//
// SVE2 - Uniform DSP operations
//

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I8_I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_saba          : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_shadd         : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_shsub         : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_shsub_u       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_shsubr        : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sli           : Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqabs         : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqadd         : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqdmulh       : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqneg         : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqrdmlah      : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqrdmlsh      : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqrdmulh      : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqrshl        : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqrshl_u      : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqshl         : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqshl_u       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqshlu        : Guarded_AdvSIMD_SVE_ShiftByImm_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqsub         : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqsub_u       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqsubr        : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_srhadd        : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sri           : Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_srshl         : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_srshl_u       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_srshr         : Guarded_AdvSIMD_SVE_ShiftByImm_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_srsra         : Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_ssra          : Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_suqadd        : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uaba          : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uhadd         : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uhsub         : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uhsub_u       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uhsubr        : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqadd         : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqrshl        : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqrshl_u      : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqshl         : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqshl_u       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqsub         : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqsub_u       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqsubr        : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_urhadd        : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_urshl         : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_urshl_u       : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_urshr         : Guarded_AdvSIMD_SVE_ShiftByImm_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_ursra         : Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_usqadd        : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_usra          : Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve,(sve2p3|sme2p3)", "sve2p3|sme2p3", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_sabal         : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uabal         : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_sqdmulh_lane  : Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqrdmlah_lane : Guarded_AdvSIMD_3VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqrdmlsh_lane : Guarded_AdvSIMD_3VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqrdmulh_lane : Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I32_Overloads>] in {
  def int_aarch64_sve_urecpe        : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_ursqrte       : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;
}

//
// SVE2 - Widening DSP operations
//

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_sabalb : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sabalt : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sabdlb : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sabdlt : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_saddlb : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_saddlt : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_saddwb : Guarded_SVE2_2VectorArg_Wide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_saddwt : Guarded_SVE2_2VectorArg_Wide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sshllb : Guarded_SVE2_1VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sshllt : Guarded_SVE2_1VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_ssublb : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_ssublt : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_ssubwb : Guarded_SVE2_2VectorArg_Wide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_ssubwt : Guarded_SVE2_2VectorArg_Wide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uabalb : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uabalt : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uabdlb : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uabdlt : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uaddlb : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uaddlt : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uaddwb : Guarded_SVE2_2VectorArg_Wide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uaddwt : Guarded_SVE2_2VectorArg_Wide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_ushllb : Guarded_SVE2_1VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_ushllt : Guarded_SVE2_1VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_usublb : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_usublt : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_usubwb : Guarded_SVE2_2VectorArg_Wide_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_usubwt : Guarded_SVE2_2VectorArg_Wide_Intrinsic<[IntrSpeculatable]>;
}

//
// SVE2 - Non-widening pairwise arithmetic
//

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I8_I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_addp    : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smaxp   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sminp   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umaxp   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uminp   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", F16_F32_F64_Overloads>] in {
  def int_aarch64_sve_faddp   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fmaxp   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fmaxnmp : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fminp   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_fminnmp : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
}

let TargetGuards = [Variant<"sve,(sve2p3|sme2p3)", "sve2p3|sme2p3", I8_I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_addqp   : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_addsubp : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_subp    : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
}

//
// SVE2 - Widening pairwise arithmetic
//

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_sadalp : Guarded_SVE2_2VectorArg_Pred_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uadalp : Guarded_SVE2_2VectorArg_Pred_Long_Intrinsic<[IntrSpeculatable]>;
}

//
// SVE2 - Uniform complex integer arithmetic
//

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I8_I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_cadd_x           : Guarded_AdvSIMD_SVE2_CADD_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqcadd_x         : Guarded_AdvSIMD_SVE2_CADD_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cmla_x           : Guarded_AdvSIMD_SVE2_CMLA_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqrdcmlah_x      : Guarded_AdvSIMD_SVE2_CMLA_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I16_I32_Overloads>] in {
  def int_aarch64_sve_cmla_lane_x      : Guarded_AdvSIMD_SVE_CMLA_LANE_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqrdcmlah_lane_x : Guarded_AdvSIMD_SVE_CMLA_LANE_Intrinsic<[IntrSpeculatable]>;
}

//
// SVE2 - Widening complex integer arithmetic
//

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_saddlbt   : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_ssublbt   : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_ssubltb   : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
}

//
// SVE2 - Widening complex integer dot product
//

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I32_I64_Overloads>] in {
  def int_aarch64_sve_cdot      : Guarded_AdvSIMD_SVE_DOT_Indexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_cdot_lane : Guarded_AdvSIMD_SVE_CDOT_LANE_Intrinsic<[IntrSpeculatable]>;
}

//
// SVE2 - Floating-point widening multiply-accumulate
//

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", F32_Overloads>] in {
  def int_aarch64_sve_fmlalb        : Guarded_SVE2_3VectorArg_Long_Intrinsic;
  def int_aarch64_sve_fmlalb_lane   : Guarded_SVE2_3VectorArgIndexed_Long_Intrinsic;
  def int_aarch64_sve_fmlalt        : Guarded_SVE2_3VectorArg_Long_Intrinsic;
  def int_aarch64_sve_fmlalt_lane   : Guarded_SVE2_3VectorArgIndexed_Long_Intrinsic;
  def int_aarch64_sve_fmlslb        : Guarded_SVE2_3VectorArg_Long_Intrinsic;
  def int_aarch64_sve_fmlslb_lane   : Guarded_SVE2_3VectorArgIndexed_Long_Intrinsic;
  def int_aarch64_sve_fmlslt        : Guarded_SVE2_3VectorArg_Long_Intrinsic;
  def int_aarch64_sve_fmlslt_lane   : Guarded_SVE2_3VectorArgIndexed_Long_Intrinsic;
}

//
// SVE2 - Multi-vector narrowing convert to floating point
//

defvar SveScvtfbTypes = [
  [llvm_nxv2f64_ty, llvm_nxv4i32_ty],
  [llvm_nxv4f32_ty, llvm_nxv8i16_ty],
  [llvm_nxv8f16_ty, llvm_nxv16i8_ty],
];

let TargetGuards = [Variant<"sve,(sve2p3|sme2p3)", "sve2p3|sme2p3", SveScvtfbTypes>] in {
  def int_aarch64_sve_scvtfb
    : GuardedDefaultIntrinsic<[llvm_anyfloat_ty], [llvm_anyint_ty], [IntrNoMem]>;
  def int_aarch64_sve_scvtft
    : GuardedDefaultIntrinsic<[llvm_anyfloat_ty], [llvm_anyint_ty], [IntrNoMem]>;
  def int_aarch64_sve_ucvtfb
    : GuardedDefaultIntrinsic<[llvm_anyfloat_ty], [llvm_anyint_ty], [IntrNoMem]>;
  def int_aarch64_sve_ucvtft
    : GuardedDefaultIntrinsic<[llvm_anyfloat_ty], [llvm_anyint_ty], [IntrNoMem]>;
}

//
// SVE2 - Floating-point integer binary logarithm
//

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", F16_F32_F64_Overloads>] in
def int_aarch64_sve_flogb : AdvSIMD_SVE_LOGB_Intrinsic;

//
// SVE2 - Vector histogram count
//

let TargetGuards = [NonStreamingVariant<"sve2", I32_I64_Overloads>] in
def int_aarch64_sve_histcnt : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic<[IntrSpeculatable]>;
let TargetGuards = [NonStreamingVariant<"sve2", I8_Overloads>] in
def int_aarch64_sve_histseg : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;

//
// SVE2 - Character match
//

let TargetGuards = [NonStreamingVariant<"sve2", I8_I16_Overloads>] in {
  def int_aarch64_sve_match   : Guarded_AdvSIMD_SVE_Compare_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_nmatch  : Guarded_AdvSIMD_SVE_Compare_Intrinsic<[IntrSpeculatable]>;
}

//
// SVE2 - Unary narrowing operations
//

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_sqxtnb  : Guarded_SVE2_1VectorArg_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqxtnt  : Guarded_SVE2_Merged1VectorArg_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqxtunb : Guarded_SVE2_1VectorArg_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqxtunt : Guarded_SVE2_Merged1VectorArg_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqxtnb  : Guarded_SVE2_1VectorArg_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqxtnt  : Guarded_SVE2_Merged1VectorArg_Narrowing_Intrinsic<[IntrSpeculatable]>;
}

//
// SVE2 - Binary narrowing DSP operations
//
let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_addhnb    : Guarded_SVE2_2VectorArg_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_addhnt    : Guarded_SVE2_Merged2VectorArg_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_raddhnb   : Guarded_SVE2_2VectorArg_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_raddhnt   : Guarded_SVE2_Merged2VectorArg_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_subhnb    : Guarded_SVE2_2VectorArg_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_subhnt    : Guarded_SVE2_Merged2VectorArg_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_rsubhnb   : Guarded_SVE2_2VectorArg_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_rsubhnt   : Guarded_SVE2_Merged2VectorArg_Narrowing_Intrinsic<[IntrSpeculatable]>;
}

// Narrowing shift right
let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_shrnb     : Guarded_SVE2_1VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_shrnt     : Guarded_SVE2_2VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_rshrnb    : Guarded_SVE2_1VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_rshrnt    : Guarded_SVE2_2VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
}

// Saturating shift right - signed input/output
let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_sqshrnb   : Guarded_SVE2_1VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqshrnt   : Guarded_SVE2_2VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqrshrnb  : Guarded_SVE2_1VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqrshrnt  : Guarded_SVE2_2VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
}

// Saturating shift right - unsigned input/output
let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_uqshrnb   : Guarded_SVE2_1VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqshrnt   : Guarded_SVE2_2VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqrshrnb  : Guarded_SVE2_1VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uqrshrnt  : Guarded_SVE2_2VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
}

// Saturating shift right - signed input, unsigned output
let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_sqshrunb  : Guarded_SVE2_1VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqshrunt  : Guarded_SVE2_2VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqrshrunb : Guarded_SVE2_1VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqrshrunt : Guarded_SVE2_2VectorArg_Imm_Narrowing_Intrinsic<[IntrSpeculatable]>;
}

// SVE2 MLA LANE.
let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I32_I64_Overloads>] in {
  def int_aarch64_sve_smlalb_lane   : Guarded_SVE2_3VectorArg_Indexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smlalt_lane   : Guarded_SVE2_3VectorArg_Indexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umlalb_lane   : Guarded_SVE2_3VectorArg_Indexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umlalt_lane   : Guarded_SVE2_3VectorArg_Indexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smlslb_lane   : Guarded_SVE2_3VectorArg_Indexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smlslt_lane   : Guarded_SVE2_3VectorArg_Indexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umlslb_lane   : Guarded_SVE2_3VectorArg_Indexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umlslt_lane   : Guarded_SVE2_3VectorArg_Indexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smullb_lane   : Guarded_SVE2_2VectorArgIndexed_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smullt_lane   : Guarded_SVE2_2VectorArgIndexed_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umullb_lane   : Guarded_SVE2_2VectorArgIndexed_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umullt_lane   : Guarded_SVE2_2VectorArgIndexed_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqdmlalb_lane : Guarded_SVE2_3VectorArg_Indexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqdmlalt_lane : Guarded_SVE2_3VectorArg_Indexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqdmlslb_lane : Guarded_SVE2_3VectorArg_Indexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqdmlslt_lane : Guarded_SVE2_3VectorArg_Indexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqdmullb_lane : Guarded_SVE2_2VectorArgIndexed_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqdmullt_lane : Guarded_SVE2_2VectorArgIndexed_Long_Intrinsic<[IntrSpeculatable]>;
}

// SVE2 MLA Unpredicated.
let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_smlalb      : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smlalt      : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umlalb      : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umlalt      : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smlslb      : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smlslt      : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umlslb      : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umlslt      : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smullb      : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smullt      : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umullb      : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_umullt      : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqdmlalb    : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqdmlalt    : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqdmlslb    : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqdmlslt    : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqdmullb    : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqdmullt    : Guarded_SVE2_2VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqdmlalbt   : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sqdmlslbt   : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
}

// SVE2 ADDSUB Long Unpredicated.
let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I32_I64_Overloads>] in {
  def int_aarch64_sve_adclb       : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_adclt       : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sbclb       : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sbclt       : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
}

//
// SVE2 - Polynomial arithmetic
//
let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I8_I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_eorbt       : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_eortb       : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
}
let TargetGuards = [
  Variant<"sve,(sve2|sme)", "sme", I8_Overloads>,
  Variant<"sve,sve-aes", "ssve-aes", I64_Overloads>,
  Variant<"sve,(sve2|sme)", "sme", I32_Overloads>,
] in {
  def int_aarch64_sve_pmullb_pair : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_pmullt_pair : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
}

//
// SVE2 bitwise ternary operations.
//
let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", I8_I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_eor3   : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_bcax   : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_bsl    : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_bsl1n  : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_bsl2n  : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_nbsl   : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_xar    : Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;
}

//
// SVE2 - Optional AES, SHA-3 and SM4
//

let TargetGuards = [Variant<"sve,sve-aes", "ssve-aes", EmptyOverload>] in {
  def int_aarch64_sve_aesd    : ClangBuiltin<"__builtin_sve_svaesd_u8">,
                              GuardedDefaultIntrinsic<[llvm_nxv16i8_ty],
                                        [llvm_nxv16i8_ty, llvm_nxv16i8_ty],
                                        [IntrNoMem, IntrSpeculatable]>;
  def int_aarch64_sve_aesimc  : ClangBuiltin<"__builtin_sve_svaesimc_u8">,
                              GuardedDefaultIntrinsic<[llvm_nxv16i8_ty],
                                        [llvm_nxv16i8_ty],
                                        [IntrNoMem, IntrSpeculatable]>;
  def int_aarch64_sve_aese    : ClangBuiltin<"__builtin_sve_svaese_u8">,
                              GuardedDefaultIntrinsic<[llvm_nxv16i8_ty],
                                        [llvm_nxv16i8_ty, llvm_nxv16i8_ty],
                                        [IntrNoMem, IntrSpeculatable]>;
  def int_aarch64_sve_aesmc   : ClangBuiltin<"__builtin_sve_svaesmc_u8">,
                              GuardedDefaultIntrinsic<[llvm_nxv16i8_ty],
                                        [llvm_nxv16i8_ty],
                                        [IntrNoMem, IntrSpeculatable]>;
}
let TargetGuards = [Variant<"sve,sve-sha3", "sme2p1,sve-sha3", EmptyOverload>] in
def int_aarch64_sve_rax1    : ClangBuiltin<"__builtin_sve_svrax1_u64">,
                              GuardedDefaultIntrinsic<[llvm_nxv2i64_ty],
                                        [llvm_nxv2i64_ty, llvm_nxv2i64_ty],
                                        [IntrNoMem, IntrSpeculatable]>;
let TargetGuards = [NonStreamingVariant<"sve,sve-sm4", EmptyOverload>] in {
  def int_aarch64_sve_sm4e    : ClangBuiltin<"__builtin_sve_svsm4e_u32">,
                              GuardedDefaultIntrinsic<[llvm_nxv4i32_ty],
                                        [llvm_nxv4i32_ty, llvm_nxv4i32_ty],
                                        [IntrNoMem, IntrSpeculatable]>;
  def int_aarch64_sve_sm4ekey : ClangBuiltin<"__builtin_sve_svsm4ekey_u32">,
                              GuardedDefaultIntrinsic<[llvm_nxv4i32_ty],
                                        [llvm_nxv4i32_ty, llvm_nxv4i32_ty],
                                        [IntrNoMem, IntrSpeculatable]>;
}
//
// SVE2 - Extended table lookup/permute
//

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", AllPackedData_Overloads>] in {
  def int_aarch64_sve_tbl2 : Guarded_AdvSIMD_SVE2_TBX_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_tbx  : Guarded_AdvSIMD_SVE2_TBX_Intrinsic<[IntrSpeculatable]>;
}

//
// SVE2 - Lookup Table
//

let TargetGuards = [Variant<"sve,(sve2|sme2),lut", "sme2,lut", I8_I16_BF16_F16_Overloads>] in {
  def int_aarch64_sve_luti2_lane : Guarded_SVE2_LUTI_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_luti4_lane : Guarded_SVE2_LUTI_Intrinsic<[IntrSpeculatable]>;
}
let TargetGuards = [NonStreamingVariant<"sve2p3", EmptyOverload>] in
  def int_aarch64_sve_luti6 : Guarded_SVE2_LUTI6_Intrinsic<[IntrSpeculatable]>;
let TargetGuards = [Variant<"sve,(sve2|sme2),lut", "sme2,lut", BF16_F16_I16_Overloads>] in
  def int_aarch64_sve_luti4_lane_x2 : Guarded_SVE2_LUTI_X2_Intrinsic<[IntrSpeculatable]>;
let TargetGuards = [Variant<"sve2p3", "sve2p3|sme2p3", BF16_F16_I16_Overloads>] in
  def int_aarch64_sve_luti6_lane_x2 : Guarded_SVE2_LUTI_X2_Intrinsic<[IntrSpeculatable]>;

//
// SVE2 - Optional bit permutation
//

let TargetGuards = [Variant<"sve,sve-bitperm", "ssve-bitperm", I8_I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_bdep_x : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_bext_x : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_bgrp_x : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
}

//
// SVE ACLE: 7.3. INT8 matrix multiply extensions
//
let TargetGuards = [NonStreamingVariant<"sve,i8mm", I32_Overloads>] in {
  def int_aarch64_sve_ummla : Guarded_SVE_MatMul_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_smmla : Guarded_SVE_MatMul_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_usmmla : Guarded_SVE_MatMul_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve,i8mm", "sme,i8mm", I32_Overloads>] in {
  def int_aarch64_sve_usdot : Guarded_AdvSIMD_SVE_DOT_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_usdot_lane : Guarded_AdvSIMD_SVE_DOT_Indexed_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_sudot_lane : Guarded_AdvSIMD_SVE_DOT_Indexed_Intrinsic<[IntrSpeculatable]>;
}

//
// SVE ACLE: 7.4/5. FP64/FP32 matrix multiply extensions
//

let TargetGuards = [
  NonStreamingVariant<"f64mm", F64F64_OverloadPair>,
  NonStreamingVariant<"f32mm", F32F32_OverloadPair>,
  NonStreamingVariant<"sve,bf16", F32BF16_OverloadPair>,
  NonStreamingVariant<"sve-f16f32mm", F32F16_OverloadPair>,
  NonStreamingVariant<"sve-b16mm", BF16BF16_OverloadPair>,
  NonStreamingVariant<"sve2p2,f16mm", F16F16_OverloadPair>,
] in
def int_aarch64_sve_fmmla
  : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                          [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<1>],
                          [IntrNoMem]>;

let TargetGuards = [
  NonStreamingVariant<"sve2,f8f32mm", F32_Overloads>,
  NonStreamingVariant<"sve2,f8f16mm", F16_Overloads>,
] in
def int_aarch64_sve_fp8_fmmla
  : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                          [LLVMMatchType<0>, llvm_nxv16i8_ty, llvm_nxv16i8_ty],
                          [IntrRead<[FPMR]>, IntrReadOnly]>;

//
// SVE ACLE: 7.2. BFloat16 extensions
//

let TargetGuards = [Variant<"sve,bf16", "sme,bf16", EmptyOverload>] in {
  def int_aarch64_sve_bfdot   : Guarded_SVE_4Vec_BF16;
  def int_aarch64_sve_bfmlalb : Guarded_SVE_4Vec_BF16;
  def int_aarch64_sve_bfmlalt : Guarded_SVE_4Vec_BF16;
  def int_aarch64_sve_bfdot_lane_v2   : Guarded_SVE_4Vec_BF16_Indexed;
  def int_aarch64_sve_bfmlalb_lane_v2 : Guarded_SVE_4Vec_BF16_Indexed;
  def int_aarch64_sve_bfmlalt_lane_v2 : Guarded_SVE_4Vec_BF16_Indexed;
}

//
// SVE2.1 - Contiguous loads to multiple consecutive vectors
//

  class Guarded_SVE2p1_Load_PN_X2_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                [llvm_aarch64_svcount_ty, llvm_anyptr_ty],
                [IntrReadMem, IntrArgMemOnly]>;

  class Guarded_SVE2p1_Load_PN_X4_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                             LLVMMatchType<0>, LLVMMatchType<0>],
                [llvm_aarch64_svcount_ty, llvm_anyptr_ty],
                [IntrReadMem, IntrArgMemOnly]>;

let TargetGuards = [Variant<"sve2p1", "sve2p1|sme2", PackedDataPTR_OverloadPairs>] in {
  def int_aarch64_sve_ld1_pn_x2 : Guarded_SVE2p1_Load_PN_X2_Intrinsic;
  def int_aarch64_sve_ld1_pn_x4 : Guarded_SVE2p1_Load_PN_X4_Intrinsic;
  def int_aarch64_sve_ldnt1_pn_x2 : Guarded_SVE2p1_Load_PN_X2_Intrinsic;
  def int_aarch64_sve_ldnt1_pn_x4 : Guarded_SVE2p1_Load_PN_X4_Intrinsic;
}

//
// SVE2.1 - Contiguous loads to quadword (single vector)
//

class Guarded_SVE2p1_Single_Load_Quadword
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                            [llvm_nxv1i1_ty, llvm_anyptr_ty],
                            [IntrReadMem, IntrArgMemOnly]>;
let TargetGuards = [NonStreamingVariant<"sve2p1", F32PTR_I32PTR_OverloadPair>] in
def int_aarch64_sve_ld1uwq : Guarded_SVE2p1_Single_Load_Quadword;
let TargetGuards = [NonStreamingVariant<"sve2p1", F64PTR_I64PTR_OverloadPair>] in
def int_aarch64_sve_ld1udq : Guarded_SVE2p1_Single_Load_Quadword;

//
// SVE2.1 - Contiguous store from quadword (single vector)
//

class Guarded_SVE2p1_Single_Store_Quadword
    : GuardedDefaultIntrinsic<[],
                            [llvm_anyvector_ty, llvm_nxv1i1_ty, llvm_anyptr_ty],
                            [IntrWriteMem, IntrArgMemOnly]>;
let TargetGuards = [NonStreamingVariant<"sve2p1", F32PTR_I32PTR_OverloadPair>] in
def int_aarch64_sve_st1wq : Guarded_SVE2p1_Single_Store_Quadword;

let TargetGuards = [NonStreamingVariant<"sve2p1", F64PTR_I64PTR_OverloadPair>] in
def int_aarch64_sve_st1dq : Guarded_SVE2p1_Single_Store_Quadword;

let TargetGuards = [
  Variant<"sve,(sve2p1|sme2p1)", "sve2p1|sme2p1", PackedDataPTR_OverloadPairs>,
] in {
  def int_aarch64_sve_ld2q_sret : Guarded_AdvSIMD_2Vec_PredLoad_Intrinsic;
  def int_aarch64_sve_ld3q_sret : Guarded_AdvSIMD_3Vec_PredLoad_Intrinsic;
  def int_aarch64_sve_ld4q_sret : Guarded_AdvSIMD_4Vec_PredLoad_Intrinsic;
  def int_aarch64_sve_st2q : Guarded_AdvSIMD_2Vec_PredStore_Intrinsic;
  def int_aarch64_sve_st3q : Guarded_AdvSIMD_3Vec_PredStore_Intrinsic;
  def int_aarch64_sve_st4q : Guarded_AdvSIMD_4Vec_PredStore_Intrinsic;
}

//
// SVE2.1 - Contiguous stores to multiple consecutive vectors
//

  class Guarded_SVE2p1_Store_PN_X2_Intrinsic
    : GuardedDefaultIntrinsic<[], [ llvm_anyvector_ty, LLVMMatchType<0>,
                                  llvm_aarch64_svcount_ty, llvm_anyptr_ty ],
                [IntrWriteMem, IntrArgMemOnly]>;

  class Guarded_SVE2p1_Store_PN_X4_Intrinsic
    : GuardedDefaultIntrinsic<[], [ llvm_anyvector_ty, LLVMMatchType<0>,
                                  LLVMMatchType<0>, LLVMMatchType<0>,
                                  llvm_aarch64_svcount_ty, llvm_anyptr_ty],
                [IntrWriteMem, IntrArgMemOnly]>;

let TargetGuards = [Variant<"sve2p1", "sve2p1|sme2", PackedDataPTR_OverloadPairs>] in {
  def int_aarch64_sve_st1_pn_x2 : Guarded_SVE2p1_Store_PN_X2_Intrinsic;
  def int_aarch64_sve_st1_pn_x4 : Guarded_SVE2p1_Store_PN_X4_Intrinsic;
  def int_aarch64_sve_stnt1_pn_x2 : Guarded_SVE2p1_Store_PN_X2_Intrinsic;
  def int_aarch64_sve_stnt1_pn_x4 : Guarded_SVE2p1_Store_PN_X4_Intrinsic;
}
}

//
// SVE2 - Contiguous conflict detection
//

let TargetGuards = [Variant<"sve,(sve2|sme)", "sme", PredicatePTR_OverloadPairs>] in {
  def int_aarch64_sve_whilerw_b : Guarded_SVE2_CONFLICT_DETECT_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_whilerw_h : Guarded_SVE2_CONFLICT_DETECT_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_whilerw_s : Guarded_SVE2_CONFLICT_DETECT_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_whilerw_d : Guarded_SVE2_CONFLICT_DETECT_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_whilewr_b : Guarded_SVE2_CONFLICT_DETECT_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_whilewr_h : Guarded_SVE2_CONFLICT_DETECT_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_whilewr_s : Guarded_SVE2_CONFLICT_DETECT_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_whilewr_d : Guarded_SVE2_CONFLICT_DETECT_Intrinsic<[IntrSpeculatable]>;
}

// Scalable Matrix Extension (SME) Intrinsics
let TargetPrefix = "aarch64" in {
  class Guarded_SME_Load_Intrinsic<LLVMType pred_ty>
    : GuardedDefaultIntrinsic<[],
        [pred_ty, llvm_anyptr_ty, llvm_i32_ty, llvm_i32_ty],
        [IntrRead<[ArgMem, ZA]>, IntrWrite<[ZA]>,
         ImmArg<ArgIndex<2>>]>;
  class Guarded_SME_Store_Intrinsic<LLVMType pred_ty>
    : GuardedDefaultIntrinsic<[],
        [pred_ty, llvm_anyptr_ty, llvm_i32_ty, llvm_i32_ty],
        [IntrRead<[ZA]>, IntrWrite<[ArgMem, ZA]>,
         ImmArg<ArgIndex<2>>]>;

  // Loads
  let TargetGuards = [StreamingVariant<"sme", PTR_Overloads>] in {
    def int_aarch64_sme_ld1b_horiz : Guarded_SME_Load_Intrinsic<llvm_nxv16i1_ty>;
    def int_aarch64_sme_ld1h_horiz : Guarded_SME_Load_Intrinsic<llvm_nxv8i1_ty>;
    def int_aarch64_sme_ld1w_horiz : Guarded_SME_Load_Intrinsic<llvm_nxv4i1_ty>;
    def int_aarch64_sme_ld1d_horiz : Guarded_SME_Load_Intrinsic<llvm_nxv2i1_ty>;
    def int_aarch64_sme_ld1q_horiz : Guarded_SME_Load_Intrinsic<llvm_nxv1i1_ty>;
    def int_aarch64_sme_ld1b_vert  : Guarded_SME_Load_Intrinsic<llvm_nxv16i1_ty>;
    def int_aarch64_sme_ld1h_vert  : Guarded_SME_Load_Intrinsic<llvm_nxv8i1_ty>;
    def int_aarch64_sme_ld1w_vert  : Guarded_SME_Load_Intrinsic<llvm_nxv4i1_ty>;
    def int_aarch64_sme_ld1d_vert  : Guarded_SME_Load_Intrinsic<llvm_nxv2i1_ty>;
    def int_aarch64_sme_ld1q_vert  : Guarded_SME_Load_Intrinsic<llvm_nxv1i1_ty>;
  }

  // Stores
  let TargetGuards = [StreamingVariant<"sme", PTR_Overloads>] in {
    def int_aarch64_sme_st1b_horiz : Guarded_SME_Store_Intrinsic<llvm_nxv16i1_ty>;
    def int_aarch64_sme_st1h_horiz : Guarded_SME_Store_Intrinsic<llvm_nxv8i1_ty>;
    def int_aarch64_sme_st1w_horiz : Guarded_SME_Store_Intrinsic<llvm_nxv4i1_ty>;
    def int_aarch64_sme_st1d_horiz : Guarded_SME_Store_Intrinsic<llvm_nxv2i1_ty>;
    def int_aarch64_sme_st1q_horiz : Guarded_SME_Store_Intrinsic<llvm_nxv1i1_ty>;
    def int_aarch64_sme_st1b_vert  : Guarded_SME_Store_Intrinsic<llvm_nxv16i1_ty>;
    def int_aarch64_sme_st1h_vert  : Guarded_SME_Store_Intrinsic<llvm_nxv8i1_ty>;
    def int_aarch64_sme_st1w_vert  : Guarded_SME_Store_Intrinsic<llvm_nxv4i1_ty>;
    def int_aarch64_sme_st1d_vert  : Guarded_SME_Store_Intrinsic<llvm_nxv2i1_ty>;
    def int_aarch64_sme_st1q_vert  : Guarded_SME_Store_Intrinsic<llvm_nxv1i1_ty>;
  }

  // Spill + fill
  let TargetGuards = [Variant<"sme", "sme", PTR_Overloads>] in {
    def int_aarch64_sme_ldr
    : GuardedDefaultIntrinsic<[], [llvm_i32_ty, llvm_anyptr_ty, llvm_i32_ty],
                            [IntrRead<[ArgMem, ZA]>, IntrWrite<[ZA]>]>;
    def int_aarch64_sme_str
    : GuardedDefaultIntrinsic<[], [llvm_i32_ty, llvm_anyptr_ty, llvm_i32_ty],
                            [IntrRead<[ZA]>, IntrWrite<[ArgMem, ZA]>]>;
  }

  class Guarded_SME_TileToVector_Intrinsic
      : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
          [LLVMMatchType<0>, LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
           llvm_i32_ty, llvm_i32_ty], [IntrRead<[ZA]>, IntrReadOnly,
                                        ImmArg<ArgIndex<2>>]>;
  class Guarded_SME_VectorToTile_Intrinsic
      : GuardedDefaultIntrinsic<[],
          [llvm_i32_ty, llvm_i32_ty, LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
           llvm_anyvector_ty], [IntrRead<[ZA]>, IntrWrite<[ZA]>,
                                ImmArg<ArgIndex<0>>]>;

  let TargetGuards = [StreamingVariant<"sme", AllPackedData_Overloads>] in {
    def int_aarch64_sme_read_horiz  : Guarded_SME_TileToVector_Intrinsic;
    def int_aarch64_sme_read_vert   : Guarded_SME_TileToVector_Intrinsic;
    def int_aarch64_sme_write_horiz : Guarded_SME_VectorToTile_Intrinsic;
    def int_aarch64_sme_write_vert  : Guarded_SME_VectorToTile_Intrinsic;

    def int_aarch64_sme_readq_horiz  : Guarded_SME_TileToVector_Intrinsic;
    def int_aarch64_sme_readq_vert   : Guarded_SME_TileToVector_Intrinsic;
    def int_aarch64_sme_writeq_horiz : Guarded_SME_VectorToTile_Intrinsic;
    def int_aarch64_sme_writeq_vert  : Guarded_SME_VectorToTile_Intrinsic;
  }

  class Guarded_SME_MOVAZ_TileToVector_X2_Intrinsic
      : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
          [llvm_i32_ty, llvm_i32_ty],
          [IntrWrite<[ZA]>, IntrRead<[ZA]>, ImmArg<ArgIndex<0>>]>;

  class Guarded_SME_MOVAZ_TileToVector_X4_Intrinsic
      : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
           LLVMMatchType<0>,LLVMMatchType<0>],
          [llvm_i32_ty, llvm_i32_ty],
          [IntrWrite<[ZA]>, IntrRead<[ZA]>, ImmArg<ArgIndex<0>>]>;

  let TargetGuards = [StreamingVariant<"sme2p1", AllPackedData_Overloads>] in {
    def int_aarch64_sme_readz_horiz_x2 : Guarded_SME_MOVAZ_TileToVector_X2_Intrinsic;
    def int_aarch64_sme_readz_vert_x2  : Guarded_SME_MOVAZ_TileToVector_X2_Intrinsic;

    def int_aarch64_sme_readz_horiz_x4 : Guarded_SME_MOVAZ_TileToVector_X4_Intrinsic;
    def int_aarch64_sme_readz_vert_x4  : Guarded_SME_MOVAZ_TileToVector_X4_Intrinsic;
  }

  class Guarded_SME_MOVAZ_TileToVector_Intrinsic
      : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
          [llvm_i32_ty, llvm_i32_ty],
          [IntrWrite<[ZA]>, IntrRead<[ZA]>, ImmArg<ArgIndex<0>>]>;

  let TargetGuards = [StreamingVariant<"sme2p1", AllPackedData_Overloads>] in {
    def int_aarch64_sme_readz_horiz : Guarded_SME_MOVAZ_TileToVector_Intrinsic;
    def int_aarch64_sme_readz_vert  : Guarded_SME_MOVAZ_TileToVector_Intrinsic;

    def int_aarch64_sme_readz_q_horiz : Guarded_SME_MOVAZ_TileToVector_Intrinsic;
    def int_aarch64_sme_readz_q_vert  : Guarded_SME_MOVAZ_TileToVector_Intrinsic;

    def int_aarch64_sme_readz_x2
      : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
          [llvm_i32_ty],
          [IntrWrite<[ZA]>, IntrRead<[ZA]>]>;
    def int_aarch64_sme_readz_x4
      : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
          [llvm_i32_ty],
          [IntrWrite<[ZA]>, IntrRead<[ZA]>]>;
  }

  let TargetGuards = [StreamingVariant<"sme-lutv2", AllPackedData_Overloads>] in {
    def int_aarch64_sme_write_lane_zt
       :  GuardedDefaultIntrinsic<[], [llvm_i32_ty, llvm_anyvector_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, IntrRead<[ZT0]>,
             IntrWrite<[ZT0]>]>;
    def int_aarch64_sme_write_zt
       :  GuardedDefaultIntrinsic<[], [llvm_i32_ty, llvm_anyvector_ty],
            [ImmArg<ArgIndex<0>>, IntrWrite<[ZT0]>, IntrWriteOnly]>;
  }

  let TargetGuards = [Variant<"sme", "sme", EmptyOverload>] in {
    def int_aarch64_sme_zero : GuardedDefaultIntrinsic<[], [llvm_i32_ty], [IntrWrite<[ZA]>, IntrWriteOnly, ImmArg<ArgIndex<0>>]>;
    def int_aarch64_sme_in_streaming_mode : GuardedDefaultIntrinsic<[llvm_i1_ty], [], [IntrNoMem]>, ClangBuiltin<"__builtin_arm_in_streaming_mode">;
  }

  class Guarded_SME_OuterProduct_Intrinsic
      : GuardedDefaultIntrinsic<[],
          [llvm_i32_ty,
           LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
           LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
           LLVMMatchType<0>,
           llvm_anyvector_ty], [IntrRead<[ZA]>, IntrWrite<[ZA]>,
                                ImmArg<ArgIndex<0>>]>;

  let TargetGuards = [
    StreamingVariant<"sme-f64f64", F64_Overloads>,
    StreamingVariant<"sme", F32_Overloads>,
    StreamingVariant<"sme-b16b16", BF16_Overloads>,
    StreamingVariant<"sme-f16f16", F16_Overloads>,
  ] in {
    def int_aarch64_sme_mopa : Guarded_SME_OuterProduct_Intrinsic;
    def int_aarch64_sme_mops : Guarded_SME_OuterProduct_Intrinsic;
  }

  let TargetGuards = [StreamingVariant<"sme", BF16_F16_Overloads>] in {
    def int_aarch64_sme_mopa_wide : Guarded_SME_OuterProduct_Intrinsic;
    def int_aarch64_sme_mops_wide : Guarded_SME_OuterProduct_Intrinsic;
  }

  let TargetGuards = [
    StreamingVariant<"sme", I8_Overloads>,
    StreamingVariant<"sme-i16i64", I16_Overloads>,
  ] in {
    def int_aarch64_sme_smopa_wide  : Guarded_SME_OuterProduct_Intrinsic;
    def int_aarch64_sme_smops_wide  : Guarded_SME_OuterProduct_Intrinsic;
    def int_aarch64_sme_umopa_wide  : Guarded_SME_OuterProduct_Intrinsic;
    def int_aarch64_sme_umops_wide  : Guarded_SME_OuterProduct_Intrinsic;
    def int_aarch64_sme_sumopa_wide : Guarded_SME_OuterProduct_Intrinsic;
    def int_aarch64_sme_sumops_wide : Guarded_SME_OuterProduct_Intrinsic;
    def int_aarch64_sme_usmopa_wide : Guarded_SME_OuterProduct_Intrinsic;
    def int_aarch64_sme_usmops_wide : Guarded_SME_OuterProduct_Intrinsic;
  }

  class Guarded_SME_OuterProduct_QuarterTile_Single_Single
      : GuardedDefaultIntrinsic<[],
          [llvm_i32_ty,
           llvm_anyvector_ty,
           LLVMMatchType<0>], [ImmArg<ArgIndex<0>>, IntrRead<[ZA]>,
                               IntrWrite<[ZA]>, IntrHasSideEffects]>;

 class Guarded_SME_FP8_OuterProduct_QuarterTile_Single_Single
      : GuardedDefaultIntrinsic<[],
          [llvm_i32_ty,
          llvm_anyvector_ty,
          LLVMMatchType<0>], [ImmArg<ArgIndex<0>>, IntrRead<[FPMR, ZA]>, IntrWrite<[ZA]>, IntrHasSideEffects]>;

  class Guarded_SME_OuterProduct_QuarterTile_Single_Multi
      : GuardedDefaultIntrinsic<[],
          [llvm_i32_ty,
          llvm_anyvector_ty,
          LLVMMatchType<0>,
          LLVMMatchType<0>], [ImmArg<ArgIndex<0>>, IntrRead<[ZA]>,
                              IntrWrite<[ZA]>, IntrHasSideEffects]>;

  class Guarded_SME_FP8_OuterProduct_QuarterTile_Single_Multi
      : GuardedDefaultIntrinsic<[],
          [llvm_i32_ty,
          llvm_anyvector_ty,
          LLVMMatchType<0>,
          LLVMMatchType<0>], [ImmArg<ArgIndex<0>>, IntrRead<[FPMR, ZA]>, IntrWrite<[ZA]>, IntrHasSideEffects]>;

  class Guarded_SME_OuterProduct_QuarterTile_Multi_Multi
      : GuardedDefaultIntrinsic<[],
          [llvm_i32_ty,
          llvm_anyvector_ty,
          LLVMMatchType<0>,
          LLVMMatchType<0>,
          LLVMMatchType<0>], [ImmArg<ArgIndex<0>>, IntrRead<[ZA]>,
                              IntrWrite<[ZA]>, IntrHasSideEffects]>;

  class Guarded_SME_FP8_OuterProduct_QuarterTile_Multi_Multi
      : GuardedDefaultIntrinsic<[],
          [llvm_i32_ty,
          llvm_anyvector_ty,
          LLVMMatchType<0>,
          LLVMMatchType<0>,
          LLVMMatchType<0>], [ImmArg<ArgIndex<0>>, IntrRead<[FPMR, ZA]>, IntrWrite<[ZA]>, IntrHasSideEffects]>;

  // 2-way and 4-way multi-vector signed/unsigned Quarter Tile Quarter Product A/S
  foreach op = ["smop4s", "umop4s", "smop4a", "umop4a"] in {
    let TargetGuards = [StreamingVariant<"sme2,sme-mop4", I8_I16_Overloads>] in {
      def int_aarch64_sme_ # op # _wide_1x1 : Guarded_SME_OuterProduct_QuarterTile_Single_Single;
      def int_aarch64_sme_ # op # _wide_1x2 : Guarded_SME_OuterProduct_QuarterTile_Single_Multi;
      def int_aarch64_sme_ # op # _wide_2x1 : Guarded_SME_OuterProduct_QuarterTile_Single_Multi;
      def int_aarch64_sme_ # op # _wide_2x2 : Guarded_SME_OuterProduct_QuarterTile_Multi_Multi;
    }
  }

  foreach op = ["sumop4s", "usmop4s", "sumop4a", "usmop4a"] in {
    let TargetGuards = [StreamingVariant<"sme2,sme-mop4", I8_Overloads>] in {
      def int_aarch64_sme_ # op # _wide_1x1 : Guarded_SME_OuterProduct_QuarterTile_Single_Single;
      def int_aarch64_sme_ # op # _wide_1x2 : Guarded_SME_OuterProduct_QuarterTile_Single_Multi;
      def int_aarch64_sme_ # op # _wide_2x1 : Guarded_SME_OuterProduct_QuarterTile_Single_Multi;
      def int_aarch64_sme_ # op # _wide_2x2 : Guarded_SME_OuterProduct_QuarterTile_Multi_Multi;
    }
  }

  foreach op = ["smop4s_za64", "umop4s_za64", "smop4a_za64", "umop4a_za64",
                "sumop4s_za64", "usmop4s_za64", "sumop4a_za64",
                "usmop4a_za64"] in {
    let TargetGuards = [StreamingVariant<"sme2,sme-mop4,sme-i16i64", I16_Overloads>] in {
      def int_aarch64_sme_ # op # _wide_1x1 : Guarded_SME_OuterProduct_QuarterTile_Single_Single;
      def int_aarch64_sme_ # op # _wide_1x2 : Guarded_SME_OuterProduct_QuarterTile_Single_Multi;
      def int_aarch64_sme_ # op # _wide_2x1 : Guarded_SME_OuterProduct_QuarterTile_Single_Multi;
      def int_aarch64_sme_ # op # _wide_2x2 : Guarded_SME_OuterProduct_QuarterTile_Multi_Multi;
    }
  }

  // 2-way and 4-way multi-vector floating point Quarter Tile Quarter Product A/S
  foreach op = ["mop4s", "mop4a"] in {
    let TargetGuards = [
      StreamingVariant<"sme2,sme-mop4,sme-f64f64", F64_Overloads>,
      StreamingVariant<"sme2,sme-mop4", F32_Overloads>,
      StreamingVariant<"sme2,sme-mop4,sme-b16b16", BF16_Overloads>,
      StreamingVariant<"sme2,sme-mop4,sme-f16f16", F16_Overloads>,
    ] in {
      def int_aarch64_sme_ # op # _1x1 : Guarded_SME_OuterProduct_QuarterTile_Single_Single;
      def int_aarch64_sme_ # op # _1x2 : Guarded_SME_OuterProduct_QuarterTile_Single_Multi;
      def int_aarch64_sme_ # op # _2x1 : Guarded_SME_OuterProduct_QuarterTile_Single_Multi;
      def int_aarch64_sme_ # op # _2x2 : Guarded_SME_OuterProduct_QuarterTile_Multi_Multi;
    }
  }

  foreach op = ["mop4s_wide", "mop4a_wide"] in {
    let TargetGuards = [StreamingVariant<"sme2,sme-mop4", BF16_F16_Overloads>] in {
      def int_aarch64_sme_ # op # _1x1 : Guarded_SME_OuterProduct_QuarterTile_Single_Single;
      def int_aarch64_sme_ # op # _1x2 : Guarded_SME_OuterProduct_QuarterTile_Single_Multi;
      def int_aarch64_sme_ # op # _2x1 : Guarded_SME_OuterProduct_QuarterTile_Single_Multi;
      def int_aarch64_sme_ # op # _2x2 : Guarded_SME_OuterProduct_QuarterTile_Multi_Multi;
    }
  }

  class Guarded_SME_OuterProduct_TMOP_Intrinsic
     : GuardedDefaultIntrinsic<[],
         [llvm_i32_ty,
          llvm_anyvector_ty,
          LLVMMatchType<0>,
          LLVMMatchType<0>,
          llvm_nxv16i8_ty,
          llvm_i32_ty],
         [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<5>>, IntrRead<[ZA]>,
          IntrWrite<[ZA]>]>;

  let TargetGuards = [
    StreamingVariant<"sme2,sme-tmop,sme-b16b16", BF16_Overloads>,
    StreamingVariant<"sme2,sme-tmop,sme-f16f16", F16_Overloads>,
  ] in
  def int_aarch64_sme_ftmopa_za16 : Guarded_SME_OuterProduct_TMOP_Intrinsic;
  let TargetGuards = [StreamingVariant<"sme2,sme-tmop", BF16_F16_F32_Overloads>] in
  def int_aarch64_sme_ftmopa_za32 : Guarded_SME_OuterProduct_TMOP_Intrinsic;
  let TargetGuards = [StreamingVariant<"sme2,sme-tmop", I8_I16_Overloads>] in {
    def int_aarch64_sme_stmopa_za32 : Guarded_SME_OuterProduct_TMOP_Intrinsic;
    def int_aarch64_sme_utmopa_za32 : Guarded_SME_OuterProduct_TMOP_Intrinsic;
  }
  let TargetGuards = [StreamingVariant<"sme2,sme-tmop", I8_Overloads>] in {
    def int_aarch64_sme_sutmopa_za32 : Guarded_SME_OuterProduct_TMOP_Intrinsic;
    def int_aarch64_sme_ustmopa_za32 : Guarded_SME_OuterProduct_TMOP_Intrinsic;
  }

  class Guarded_SME_FP8_OuterProduct_TMOP_Intrinsic
     : GuardedDefaultIntrinsic<[],
         [llvm_i32_ty,
          llvm_nxv16i8_ty,
          llvm_nxv16i8_ty,
          llvm_nxv16i8_ty,
          llvm_nxv16i8_ty,
          llvm_i32_ty],
         [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<5>>,
          IntrRead<[ZA, FPMR]>, IntrWrite<[ZA]>]>;

  let TargetGuards = [StreamingVariant<"sme2,sme-tmop,sme-f8f16", EmptyOverload>] in
  def int_aarch64_sme_fp8_ftmopa_za16 : Guarded_SME_FP8_OuterProduct_TMOP_Intrinsic;
  let TargetGuards = [StreamingVariant<"sme2,sme-tmop,sme-f8f32", EmptyOverload>] in
  def int_aarch64_sme_fp8_ftmopa_za32 : Guarded_SME_FP8_OuterProduct_TMOP_Intrinsic;

  // 16 and 32 bit multi-vector floating point 8 Quarter Tile Quarter Product
  let TargetGuards = [StreamingVariant<"sme2,sme-mop4,sme-f8f16", I8_Overloads>] in {
    def int_aarch64_sme_fp8_fmop4a_za16_1x1 : Guarded_SME_FP8_OuterProduct_QuarterTile_Single_Single;
    def int_aarch64_sme_fp8_fmop4a_za16_1x2 : Guarded_SME_FP8_OuterProduct_QuarterTile_Single_Multi;
    def int_aarch64_sme_fp8_fmop4a_za16_2x1 : Guarded_SME_FP8_OuterProduct_QuarterTile_Single_Multi;

    def int_aarch64_sme_fp8_fmop4a_za16_2x2 : Guarded_SME_FP8_OuterProduct_QuarterTile_Multi_Multi;
  }

  let TargetGuards = [StreamingVariant<"sme2,sme-mop4,sme-f8f32", I8_Overloads>] in {
    def int_aarch64_sme_fp8_fmop4a_za32_1x1 : Guarded_SME_FP8_OuterProduct_QuarterTile_Single_Single;
    def int_aarch64_sme_fp8_fmop4a_za32_1x2 : Guarded_SME_FP8_OuterProduct_QuarterTile_Single_Multi;
    def int_aarch64_sme_fp8_fmop4a_za32_2x1 : Guarded_SME_FP8_OuterProduct_QuarterTile_Single_Multi;

    def int_aarch64_sme_fp8_fmop4a_za32_2x2 : Guarded_SME_FP8_OuterProduct_QuarterTile_Multi_Multi;
  }

  class Guarded_SME_AddVectorToTile_Intrinsic
      : GuardedDefaultIntrinsic<[],
          [llvm_i32_ty,
           LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
           LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
           llvm_anyvector_ty], [IntrRead<[ZA]>, IntrWrite<[ZA]>,
                                ImmArg<ArgIndex<0>>]>;

  let TargetGuards = [
    StreamingVariant<"sme-i16i64", I64_Overloads>,
    StreamingVariant<"sme", I32_Overloads>,
  ] in {
    def int_aarch64_sme_addha : Guarded_SME_AddVectorToTile_Intrinsic;
    def int_aarch64_sme_addva : Guarded_SME_AddVectorToTile_Intrinsic;
  }

  //
  // Counting elements
  //

  let TargetGuards = [Variant<"sme", "sme", EmptyOverload>] in
  def int_aarch64_sme_cntsd
      : GuardedDefaultIntrinsic<[llvm_i64_ty], [], [IntrNoMem]>;

  // Clamp
  //

  let TargetGuards = [Variant<"sve,(sve2p1|sme)", "sme", I8_I16_I32_I64_Overloads>] in {
    def int_aarch64_sve_sclamp : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
    def int_aarch64_sve_uclamp : Guarded_AdvSIMD_3VectorArg_Intrinsic<[IntrSpeculatable]>;
  }
  let TargetGuards = [
    Variant<"sve,(sve2p1|sme2)", "sve2p1|sme2", F32_F64_Overloads>,
    Variant<"sve,sve-b16b16", "sme2,sve-b16b16", BF16_Overloads>,
    Variant<"sve,(sve2p1|sme2)", "sve2p1|sme2", F16_Overloads>,
  ] in
  def int_aarch64_sve_fclamp : Guarded_AdvSIMD_3VectorArg_Intrinsic;

  //
  // Reversal
  //

  let TargetGuards = [Variant<"sve,(sve2p1|sme)", "sme", AllPackedData_Overloads>] in
  def int_aarch64_sve_revd : Guarded_AdvSIMD_Merged1VectorArg_Intrinsic<[IntrSpeculatable]>;

  //
  // Predicate selection
  //

  defvar SvePSelTypes = [
    [llvm_nxv16i1_ty, llvm_nxv16i1_ty],
    [llvm_nxv16i1_ty, llvm_nxv8i1_ty],
    [llvm_nxv16i1_ty, llvm_nxv4i1_ty],
    [llvm_nxv16i1_ty, llvm_nxv2i1_ty],
  ];
  defvar Sve2p1PSelTypes = [
    [llvm_aarch64_svcount_ty, llvm_nxv16i1_ty],
    [llvm_aarch64_svcount_ty, llvm_nxv8i1_ty],
    [llvm_aarch64_svcount_ty, llvm_nxv4i1_ty],
    [llvm_aarch64_svcount_ty, llvm_nxv2i1_ty],
  ];
  let TargetGuards = [Variant<"sve,(sve2p1|sme)", "sve2p1|sme", SvePSelTypes>,
                      Variant<"sve,(sve2p1|sme2)", "sve2p1|sme2", Sve2p1PSelTypes>] in
  def int_aarch64_sve_psel
      : GuardedDefaultIntrinsic<[llvm_any_ty],
                                     [LLVMMatchType<0>, llvm_anyvector_ty, llvm_i32_ty],
                                     [IntrNoMem]>;

  //
  // Predicate-pair intrinsics
  //
  foreach cmp = ["ge", "gt", "hi", "hs", "le", "lo", "ls", "lt"] in {
    let TargetGuards = [Variant<"sve,(sve2p1|sme2)", "sve2p1|sme2", Predicate_Overloads>] in
    def int_aarch64_sve_while # cmp # _x2
        : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                                [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;
  }

  //
  // Predicate-as-counter intrinsics
  //

  let TargetGuards = [Variant<"sve2p1", "sve2p1|sme2", Predicate_Overloads>] in {
    def int_aarch64_sve_pext
      : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                              [llvm_aarch64_svcount_ty, llvm_i32_ty],
                              [IntrNoMem, ImmArg<ArgIndex<1>>]>;
    def int_aarch64_sve_pext_x2
      : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                              [llvm_aarch64_svcount_ty, llvm_i32_ty],
                              [IntrNoMem, ImmArg<ArgIndex<1>>]>;
  }

  let TargetGuards = [Variant<"sve2p1", "sve2p1|sme2", EmptyOverload>] in {
    def int_aarch64_sve_ptrue_c8
      : GuardedDefaultIntrinsic<[llvm_aarch64_svcount_ty], [], [IntrNoMem]>;
    def int_aarch64_sve_ptrue_c16
      : GuardedDefaultIntrinsic<[llvm_aarch64_svcount_ty], [], [IntrNoMem]>;
    def int_aarch64_sve_ptrue_c32
      : GuardedDefaultIntrinsic<[llvm_aarch64_svcount_ty], [], [IntrNoMem]>;
    def int_aarch64_sve_ptrue_c64
      : GuardedDefaultIntrinsic<[llvm_aarch64_svcount_ty], [], [IntrNoMem]>;

    def int_aarch64_sve_cntp_c8
      : GuardedDefaultIntrinsic<[llvm_i64_ty], [llvm_aarch64_svcount_ty, llvm_i32_ty],
                              [IntrNoMem, ImmArg<ArgIndex<1>>]>;
    def int_aarch64_sve_cntp_c16
      : GuardedDefaultIntrinsic<[llvm_i64_ty], [llvm_aarch64_svcount_ty, llvm_i32_ty],
                              [IntrNoMem, ImmArg<ArgIndex<1>>]>;
    def int_aarch64_sve_cntp_c32
      : GuardedDefaultIntrinsic<[llvm_i64_ty], [llvm_aarch64_svcount_ty, llvm_i32_ty],
                              [IntrNoMem, ImmArg<ArgIndex<1>>]>;
    def int_aarch64_sve_cntp_c64
      : GuardedDefaultIntrinsic<[llvm_i64_ty], [llvm_aarch64_svcount_ty, llvm_i32_ty],
                              [IntrNoMem, ImmArg<ArgIndex<1>>]>;
  }

  // While (predicate-as-counter) intrinsics
  foreach cmp = ["ge", "gt", "hi", "hs", "le", "lo", "ls", "lt"] in {
    foreach ty = ["c8", "c16", "c32", "c64"] in {
      let TargetGuards = [Variant<"sve2p1", "sve2p1|sme2", EmptyOverload>] in
      def int_aarch64_sve_while # cmp # _ # ty
          : GuardedDefaultIntrinsic<[llvm_aarch64_svcount_ty],
                                  [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty],
                                  [IntrNoMem, ImmArg<ArgIndex<2>>]>;
    }
  }

  //
  // SME2 Intrinsics
  //

  class Guarded_SME2_Matrix_ArrayVector_Single_Single_Intrinsic
    : GuardedDefaultIntrinsic<[],
                [llvm_i32_ty,
                 llvm_anyvector_ty, LLVMMatchType<0>],
                [IntrRead<[ZA]>, IntrWrite<[ZA]>]>;

  class Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Single_Intrinsic
    : GuardedDefaultIntrinsic<[],
                [llvm_i32_ty,
                 llvm_anyvector_ty, LLVMMatchType<0>,
                 LLVMMatchType<0>],
                [IntrRead<[ZA]>, IntrWrite<[ZA]>]>;

  class Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Single_Intrinsic
    : GuardedDefaultIntrinsic<[],
                [llvm_i32_ty,
                 llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>],
                [IntrRead<[ZA]>, IntrWrite<[ZA]>]>;

  class Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Multi_Intrinsic
    : GuardedDefaultIntrinsic<[],
                [llvm_i32_ty,
                 llvm_anyvector_ty, LLVMMatchType<0>,
                 LLVMMatchType<0>, LLVMMatchType<0>],
                [IntrRead<[ZA]>, IntrWrite<[ZA]>]>;

  class Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Multi_Intrinsic
    : GuardedDefaultIntrinsic<[],
                [llvm_i32_ty,
                 llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                [IntrRead<[ZA]>, IntrWrite<[ZA]>]>;

  class Guarded_SME2_Matrix_ArrayVector_Single_Index_Intrinsic
    : GuardedDefaultIntrinsic<[],
                [llvm_i32_ty,
                llvm_anyvector_ty,
                LLVMMatchType<0>, llvm_i32_ty],
                [IntrRead<[ZA]>, IntrWrite<[ZA]>,
                 ImmArg<ArgIndex<3>>]>;

  class Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic
    : GuardedDefaultIntrinsic<[],
                [llvm_i32_ty,
                 llvm_anyvector_ty, LLVMMatchType<0>,
                 LLVMMatchType<0>, llvm_i32_ty],
                [IntrRead<[ZA]>, IntrWrite<[ZA]>,
                 ImmArg<ArgIndex<4>>]>;

  class Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic
    : GuardedDefaultIntrinsic<[],
                [llvm_i32_ty,
                 llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>, llvm_i32_ty],
                [IntrRead<[ZA]>, IntrWrite<[ZA]>,
                 ImmArg<ArgIndex<6>>]>;

  class Guarded_SVE2_VG2_Multi_Imm_Intrinsic
    : GuardedDefaultIntrinsic<[LLVMSubdivide2VectorType<0>],
                [llvm_anyvector_ty, LLVMMatchType<0>,
                 llvm_i32_ty],
                [IntrNoMem, ImmArg<ArgIndex<2>>]>;

  class Guarded_SVE2_VG4_Multi_Imm_Intrinsic
    : GuardedDefaultIntrinsic<[LLVMSubdivide4VectorType<0>],
                [llvm_anyvector_ty, LLVMMatchType<0>,
                 LLVMMatchType<0>, LLVMMatchType<0>,
                 llvm_i32_ty],
                [IntrNoMem, ImmArg<ArgIndex<4>>]>;

  class Guarded_SME2_ZA_Write_VG2_Intrinsic
   : GuardedDefaultIntrinsic<[],
               [llvm_i32_ty,
                llvm_anyvector_ty, LLVMMatchType<0>],
               [IntrRead<[ZA]>, IntrWrite<[ZA]>]>;

  class Guarded_SME2_ZA_Write_VG4_Intrinsic
   : GuardedDefaultIntrinsic<[],
               [llvm_i32_ty,
                llvm_anyvector_ty, LLVMMatchType<0>,
                LLVMMatchType<0>,  LLVMMatchType<0>],
               [IntrRead<[ZA]>, IntrWrite<[ZA]>]>;

  class Guarded_SVE2_VG2_Multi_Single_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                [LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>],
                [IntrNoMem]>;

  class Guarded_SVE2_VG2_Multi_Single_Scale_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                [LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMVectorOfBitcastsToInt<0>],
                [IntrNoMem]>;

  class Guarded_SVE2_VG4_Multi_Single_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                             LLVMMatchType<0>,  LLVMMatchType<0>],
                            [LLVMMatchType<0>,  LLVMMatchType<0>,
                             LLVMMatchType<0>,  LLVMMatchType<0>,
                             LLVMMatchType<0>],
                            [IntrNoMem]>;

  class Guarded_SVE2_VG4_Multi_Single_Scale_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                             LLVMMatchType<0>,  LLVMMatchType<0>],
                            [LLVMMatchType<0>,  LLVMMatchType<0>,
                             LLVMMatchType<0>,  LLVMMatchType<0>,
                             LLVMVectorOfBitcastsToInt<0>],
                            [IntrNoMem]>;

  class Guarded_SVE2_VG2_Multi_Multi_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                [LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>, LLVMMatchType<0>],
                [IntrNoMem]>;

  class Guarded_SVE2_VG2_Multi_Multi_Scale_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                [LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMVectorOfBitcastsToInt<0>, LLVMVectorOfBitcastsToInt<0>],
                [IntrNoMem]>;

  class Guarded_SVE2_VG4_Multi_Multi_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                             LLVMMatchType<0>,  LLVMMatchType<0>],
                            [LLVMMatchType<0>,  LLVMMatchType<0>,
                             LLVMMatchType<0>,  LLVMMatchType<0>,
                             LLVMMatchType<0>, LLVMMatchType<0>,
                             LLVMMatchType<0>, LLVMMatchType<0>],
                            [IntrNoMem]>;

  class Guarded_SVE2_VG4_Multi_Multi_Scale_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                             LLVMMatchType<0>,  LLVMMatchType<0>],
                            [LLVMMatchType<0>,  LLVMMatchType<0>,
                             LLVMMatchType<0>,  LLVMMatchType<0>,
                             LLVMVectorOfBitcastsToInt<0>, LLVMVectorOfBitcastsToInt<0>,
                             LLVMVectorOfBitcastsToInt<0>, LLVMVectorOfBitcastsToInt<0>],
                            [IntrNoMem]>;

  class Guarded_SVE2_VG2_Sel_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                [llvm_aarch64_svcount_ty, LLVMMatchType<0>,
                 LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>], [IntrNoMem]>;

  class Guarded_SVE2_VG4_Sel_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                             LLVMMatchType<0>, LLVMMatchType<0>],
                [llvm_aarch64_svcount_ty, LLVMMatchType<0>,
                 LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>], [IntrNoMem]>;

  class Guarded_SVE2_CVT_VG2_SINGLE_Intrinsic
    : GuardedDefaultIntrinsic<[LLVMSubdivide2VectorType<0>],
                            [llvm_anyvector_ty, LLVMMatchType<0>],
                            [IntrNoMem]>;

  class Guarded_SVE2_CVT_VG2_SINGLE_BF16_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_nxv8bf16_ty],
                            [llvm_nxv4f32_ty, llvm_nxv4f32_ty],
                            [IntrNoMem]>;

  class Guarded_SVE2_CVT_WIDENING_VG2_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                            [LLVMSubdivide2VectorType<0>], [IntrNoMem]>;

  class Guarded_SVE2_CVT_VG4_SINGLE_Intrinsic
    : GuardedDefaultIntrinsic<[LLVMSubdivide4VectorType<0>],
                            [llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                            [IntrNoMem]>;

  class Guarded_SVE2_CVT_X2_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                            [llvm_anyvector_ty, LLVMMatchType<1>],
                            [IntrNoMem]>;

  class Guarded_SVE2_CVT_X4_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                            [llvm_anyvector_ty, LLVMMatchType<1>, LLVMMatchType<1>, LLVMMatchType<1>],
                            [IntrNoMem]>;

  class Guarded_SVE2_BFMLS_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_nxv4f32_ty],
                            [llvm_nxv4f32_ty, llvm_nxv8bf16_ty, llvm_nxv8bf16_ty],
                            [IntrNoMem]>;

  class Guarded_SVE2_BFMLS_Lane_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_nxv4f32_ty],
                            [llvm_nxv4f32_ty, llvm_nxv8bf16_ty, llvm_nxv8bf16_ty, llvm_i32_ty],
                            [IntrNoMem, ImmArg<ArgIndex<3>>]>;

  class Guarded_SME2_ZA_ArrayVector_Read_VG2_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                [llvm_i32_ty],
                [IntrRead<[ZA]>, IntrReadOnly]>;

  class Guarded_SME2_ZA_ArrayVector_Read_VG4_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                             LLVMMatchType<0>,  LLVMMatchType<0>],
                [llvm_i32_ty],
                [IntrRead<[ZA]>, IntrReadOnly]>;

  class Guarded_SME2_Matrix_TileVector_Read_VG2_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                [llvm_i32_ty, llvm_i32_ty],
                [IntrRead<[ZA]>, IntrReadOnly]>;

  class Guarded_SME2_Matrix_TileVector_Read_VG4_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                             LLVMMatchType<0>,  LLVMMatchType<0>],
                [llvm_i32_ty, llvm_i32_ty],
                [IntrRead<[ZA]>, IntrReadOnly]>;

  class Guarded_SME2_ZA_ArrayVector_Write_VG2_Intrinsic
   : GuardedDefaultIntrinsic<[],
               [llvm_i32_ty,
                llvm_anyvector_ty, LLVMMatchType<0>],
               [IntrWrite<[ZA]>, IntrWriteOnly]>;

  class Guarded_SME2_ZA_ArrayVector_Write_VG4_Intrinsic
   : GuardedDefaultIntrinsic<[],
               [llvm_i32_ty,
                llvm_anyvector_ty, LLVMMatchType<0>,
                LLVMMatchType<0>,  LLVMMatchType<0>],
               [IntrWrite<[ZA]>, IntrWriteOnly]>;

  class Guarded_SME2_Matrix_TileVector_Write_VG2_Intrinsic
   : GuardedDefaultIntrinsic<[],
               [llvm_i32_ty, llvm_i32_ty,
                llvm_anyvector_ty, LLVMMatchType<0>],
               [IntrWrite<[ZA]>, IntrWriteOnly, ImmArg<ArgIndex<0>>]>;

  class Guarded_SME2_Matrix_TileVector_Write_VG4_Intrinsic
   : GuardedDefaultIntrinsic<[],
               [llvm_i32_ty, llvm_i32_ty,
                llvm_anyvector_ty, LLVMMatchType<0>,
                LLVMMatchType<0>,  LLVMMatchType<0>],
               [IntrWrite<[ZA]>, IntrWriteOnly, ImmArg<ArgIndex<0>>]>;

  class Guarded_SVE2_VG2_Multi_Single_Single_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                [LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>, LLVMMatchType<0>],
                [IntrNoMem]>;

  class Guarded_SVE2_VG4_Multi_Single_Single_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                             LLVMMatchType<0>, LLVMMatchType<0>],
                [LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>, LLVMMatchType<0>],
                [IntrNoMem]>;

  class Guarded_SVE2_VG2_ZipUzp_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                [LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;

  class Guarded_SVE2_VG4_ZipUzp_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                             LLVMMatchType<0>, LLVMMatchType<0>],
                [LLVMMatchType<0>, LLVMMatchType<0>,
                 LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;

  class Guarded_SVE2_VG2_Unpk_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                [LLVMSubdivide2VectorType<0>], [IntrNoMem]>;

  class Guarded_SVE2_VG4_Unpk_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                             LLVMMatchType<0>, LLVMMatchType<0>],
                [LLVMSubdivide2VectorType<0>, LLVMSubdivide2VectorType<0>],
                [IntrNoMem]>;

  //
  // Multi-vector fused multiply-add/subtract
  //

  let TargetGuards = [
    StreamingVariant<"sme2,sme-f64f64", F64_Overloads>,
    StreamingVariant<"sme2", F32_Overloads>,
    StreamingVariant<"sme-b16b16", BF16_Overloads>,
    StreamingVariant<"sme-f16f16", F16_Overloads>,
  ] in {
    def int_aarch64_sme_fmla_single_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Single_Intrinsic;
    def int_aarch64_sme_fmls_single_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Single_Intrinsic;
    def int_aarch64_sme_fmla_single_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Single_Intrinsic;
    def int_aarch64_sme_fmls_single_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Single_Intrinsic;

    def int_aarch64_sme_fmla_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Multi_Intrinsic;
    def int_aarch64_sme_fmls_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Multi_Intrinsic;
    def int_aarch64_sme_fmla_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Multi_Intrinsic;
    def int_aarch64_sme_fmls_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Multi_Intrinsic;

    def int_aarch64_sme_fmla_lane_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic;
    def int_aarch64_sme_fmls_lane_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic;
    def int_aarch64_sme_fmla_lane_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
    def int_aarch64_sme_fmls_lane_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
  }

  //
  // Outer product and accumulate/subtract intrinsics
  //

  let TargetGuards = [StreamingVariant<"sme2", I16_Overloads>] in {
    def int_aarch64_sme_smopa_za32 : Guarded_SME_OuterProduct_Intrinsic;
    def int_aarch64_sme_umopa_za32 : Guarded_SME_OuterProduct_Intrinsic;
    def int_aarch64_sme_smops_za32 : Guarded_SME_OuterProduct_Intrinsic;
    def int_aarch64_sme_umops_za32 : Guarded_SME_OuterProduct_Intrinsic;
  }

  let TargetGuards = [StreamingVariant<"sme2", I32_Overloads>] in {
    def int_aarch64_sme_bmopa_za32 : Guarded_SME_OuterProduct_Intrinsic;
    def int_aarch64_sme_bmops_za32 : Guarded_SME_OuterProduct_Intrinsic;
  }

  //
  // Multi-vector rounding shift left intrinsics
  //

  let TargetGuards = [StreamingVariant<"sme2", I8_I16_I32_I64_Overloads>] in {
    def int_aarch64_sve_srshl_single_x2 : Guarded_SVE2_VG2_Multi_Single_Intrinsic;
    def int_aarch64_sve_urshl_single_x2 : Guarded_SVE2_VG2_Multi_Single_Intrinsic;
    def int_aarch64_sve_srshl_single_x4 : Guarded_SVE2_VG4_Multi_Single_Intrinsic;
    def int_aarch64_sve_urshl_single_x4 : Guarded_SVE2_VG4_Multi_Single_Intrinsic;

    def int_aarch64_sve_srshl_x2 : Guarded_SVE2_VG2_Multi_Multi_Intrinsic;
    def int_aarch64_sve_urshl_x2 : Guarded_SVE2_VG2_Multi_Multi_Intrinsic;
    def int_aarch64_sve_srshl_x4 : Guarded_SVE2_VG4_Multi_Multi_Intrinsic;
    def int_aarch64_sve_urshl_x4 : Guarded_SVE2_VG4_Multi_Multi_Intrinsic;
  }

  // Multi-vector saturating rounding shift right intrinsics

  let TargetGuards = [StreamingVariant<"sme2", I32_Overloads>] in {
    def int_aarch64_sve_sqrshr_x2 : Guarded_SVE2_VG2_Multi_Imm_Intrinsic;
    def int_aarch64_sve_uqrshr_x2 : Guarded_SVE2_VG2_Multi_Imm_Intrinsic;

    def int_aarch64_sve_sqrshru_x2 : Guarded_SVE2_VG2_Multi_Imm_Intrinsic;
  }

  let TargetGuards = [StreamingVariant<"sme2", I32_I64_Overloads>] in {
    def int_aarch64_sve_sqrshr_x4 : Guarded_SVE2_VG4_Multi_Imm_Intrinsic;
    def int_aarch64_sve_uqrshr_x4 : Guarded_SVE2_VG4_Multi_Imm_Intrinsic;

    def int_aarch64_sve_sqrshrn_x4 : Guarded_SVE2_VG4_Multi_Imm_Intrinsic;
    def int_aarch64_sve_uqrshrn_x4 : Guarded_SVE2_VG4_Multi_Imm_Intrinsic;

    def int_aarch64_sve_sqrshru_x4 : Guarded_SVE2_VG4_Multi_Imm_Intrinsic;
    def int_aarch64_sve_sqrshrun_x4 : Guarded_SVE2_VG4_Multi_Imm_Intrinsic;
  }

  let TargetGuards = [
    Variant<"sve,(sve2p1|sme2)", "sve2p1|sme2", I32_Overloads>,
    Variant<"sve,(sve2p3|sme2p3)", "sve2p3|sme2p3", I16_Overloads>,
  ] in {
    def int_aarch64_sve_sqrshrn_x2 : Guarded_SVE2_VG2_Multi_Imm_Intrinsic;
    def int_aarch64_sve_uqrshrn_x2 : Guarded_SVE2_VG2_Multi_Imm_Intrinsic;

    def int_aarch64_sve_sqrshrun_x2 : Guarded_SVE2_VG2_Multi_Imm_Intrinsic;
  }

  //
  // Multi-vector multiply-add/subtract long
  //

  foreach op = ["fmlal", "fmlsl"] in {
    let TargetGuards = [StreamingVariant<"sme2", BF16_F16_Overloads>] in {
      def int_aarch64_sme_ # op # _single_vg2x1 : Guarded_SME2_Matrix_ArrayVector_Single_Single_Intrinsic;
      def int_aarch64_sme_ # op # _single_vg2x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Single_Intrinsic;
      def int_aarch64_sme_ # op # _single_vg2x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Single_Intrinsic;
      def int_aarch64_sme_ # op # _vg2x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Multi_Intrinsic;
      def int_aarch64_sme_ # op # _vg2x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Multi_Intrinsic;
      def int_aarch64_sme_ # op # _lane_vg2x1 : Guarded_SME2_Matrix_ArrayVector_Single_Index_Intrinsic;
      def int_aarch64_sme_ # op # _lane_vg2x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic;
      def int_aarch64_sme_ # op # _lane_vg2x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
    }
  }

  foreach op = ["smlal", "smlsl", "umlal", "umlsl"] in {
    let TargetGuards = [StreamingVariant<"sme2", I16_Overloads>] in {
      def int_aarch64_sme_ # op # _single_vg2x1 : Guarded_SME2_Matrix_ArrayVector_Single_Single_Intrinsic;
      def int_aarch64_sme_ # op # _single_vg2x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Single_Intrinsic;
      def int_aarch64_sme_ # op # _single_vg2x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Single_Intrinsic;
      def int_aarch64_sme_ # op # _vg2x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Multi_Intrinsic;
      def int_aarch64_sme_ # op # _vg2x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Multi_Intrinsic;
      def int_aarch64_sme_ # op # _lane_vg2x1 : Guarded_SME2_Matrix_ArrayVector_Single_Index_Intrinsic;
      def int_aarch64_sme_ # op # _lane_vg2x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic;
      def int_aarch64_sme_ # op # _lane_vg2x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
    }
  }

  //
  // Multi-vector multiply-add long long
  //

  foreach op = ["smla", "smls", "umla", "umls"] in {
    let TargetGuards = [StreamingVariant<"sme2", I8_Overloads>] in {
      def int_aarch64_sme_ # op # _za32_single_vg4x1 : Guarded_SME2_Matrix_ArrayVector_Single_Single_Intrinsic;
      def int_aarch64_sme_ # op # _za32_single_vg4x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Single_Intrinsic;
      def int_aarch64_sme_ # op # _za32_single_vg4x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Single_Intrinsic;
      def int_aarch64_sme_ # op # _za32_vg4x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Multi_Intrinsic;
      def int_aarch64_sme_ # op # _za32_vg4x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Multi_Intrinsic;
      def int_aarch64_sme_ # op # _za32_lane_vg4x1 : Guarded_SME2_Matrix_ArrayVector_Single_Index_Intrinsic;
      def int_aarch64_sme_ # op # _za32_lane_vg4x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic;
      def int_aarch64_sme_ # op # _za32_lane_vg4x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
    }
  }

  foreach op = ["smla", "smls", "umla", "umls"] in {
    let TargetGuards = [StreamingVariant<"sme2,sme-i16i64", I16_Overloads>] in {
      def int_aarch64_sme_ # op # _za64_single_vg4x1 : Guarded_SME2_Matrix_ArrayVector_Single_Single_Intrinsic;
      def int_aarch64_sme_ # op # _za64_single_vg4x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Single_Intrinsic;
      def int_aarch64_sme_ # op # _za64_single_vg4x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Single_Intrinsic;
      def int_aarch64_sme_ # op # _za64_vg4x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Multi_Intrinsic;
      def int_aarch64_sme_ # op # _za64_vg4x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Multi_Intrinsic;
      def int_aarch64_sme_ # op # _za64_lane_vg4x1 : Guarded_SME2_Matrix_ArrayVector_Single_Index_Intrinsic;
      def int_aarch64_sme_ # op # _za64_lane_vg4x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic;
      def int_aarch64_sme_ # op # _za64_lane_vg4x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
    }
  }

  let TargetGuards = [StreamingVariant<"sme2", I8_Overloads>] in {
    def int_aarch64_sme_sumla_za32_single_vg4x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Single_Intrinsic;
    def int_aarch64_sme_sumla_za32_single_vg4x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Single_Intrinsic;

    def int_aarch64_sme_sumla_za32_lane_vg4x1 : Guarded_SME2_Matrix_ArrayVector_Single_Index_Intrinsic;
    def int_aarch64_sme_sumla_za32_lane_vg4x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic;
    def int_aarch64_sme_sumla_za32_lane_vg4x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;

    def int_aarch64_sme_usmla_za32_single_vg4x1 : Guarded_SME2_Matrix_ArrayVector_Single_Single_Intrinsic;
    def int_aarch64_sme_usmla_za32_single_vg4x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Single_Intrinsic;
    def int_aarch64_sme_usmla_za32_single_vg4x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Single_Intrinsic;

    def int_aarch64_sme_usmla_za32_vg4x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Multi_Intrinsic;
    def int_aarch64_sme_usmla_za32_vg4x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Multi_Intrinsic;

    def int_aarch64_sme_usmla_za32_lane_vg4x1 : Guarded_SME2_Matrix_ArrayVector_Single_Index_Intrinsic;
    def int_aarch64_sme_usmla_za32_lane_vg4x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic;
    def int_aarch64_sme_usmla_za32_lane_vg4x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
  }

  let TargetGuards = [Variant<"sve,(sve2p1|sme2)", "sve2p1|sme2", EmptyOverload>] in {
    def int_aarch64_sve_bfmlslb : Guarded_SVE2_BFMLS_Intrinsic;
    def int_aarch64_sve_bfmlslb_lane : Guarded_SVE2_BFMLS_Lane_Intrinsic;

    def int_aarch64_sve_bfmlslt : Guarded_SVE2_BFMLS_Intrinsic;
    def int_aarch64_sve_bfmlslt_lane : Guarded_SVE2_BFMLS_Lane_Intrinsic;
  }

  // Multi-vector zeroing

  foreach vg = ["vg1x2", "vg1x4", "vg2x1", "vg2x2", "vg2x4", "vg4x1", "vg4x2", "vg4x4"] in {
    let TargetGuards = [StreamingVariant<"sme2p1", EmptyOverload>] in
    def int_aarch64_sme_zero_za64_ # vg : GuardedDefaultIntrinsic<[], [llvm_i32_ty],  [IntrWrite<[ZA]>, IntrWriteOnly]>;
  }
  // Multi-vector signed saturating doubling multiply high
  let TargetGuards = [StreamingVariant<"sme2", I8_I16_I32_I64_Overloads>] in {
    def int_aarch64_sve_sqdmulh_single_vgx2 : Guarded_SVE2_VG2_Multi_Single_Intrinsic;
    def int_aarch64_sve_sqdmulh_single_vgx4 : Guarded_SVE2_VG4_Multi_Single_Intrinsic;

    def int_aarch64_sve_sqdmulh_vgx2 : Guarded_SVE2_VG2_Multi_Multi_Intrinsic;
    def int_aarch64_sve_sqdmulh_vgx4 : Guarded_SVE2_VG4_Multi_Multi_Intrinsic;
  }

  // Multi-vector floating-point round to integral value

  foreach inst = ["a", "m", "n", "p"] in {
    let TargetGuards = [StreamingVariant<"sme2", F32_Overloads>] in {
      def int_aarch64_sve_frint # inst # _x2 : Guarded_SVE2_VG2_ZipUzp_Intrinsic;
      def int_aarch64_sve_frint # inst # _x4 : Guarded_SVE2_VG4_ZipUzp_Intrinsic;
    }
  }

  //
  // Multi-vector min/max
  //

  foreach op = ["fmax", "fmin"] in {
    let TargetGuards = [
      StreamingVariant<"sme2", F32_F64_Overloads>,
      StreamingVariant<"sme2,sve-b16b16", BF16_Overloads>,
      StreamingVariant<"sme2", F16_Overloads>,
    ] in {
      def int_aarch64_sve_ # op # _single_x2 : Guarded_SVE2_VG2_Multi_Single_Intrinsic;
      def int_aarch64_sve_ # op # _single_x4 : Guarded_SVE2_VG4_Multi_Single_Intrinsic;
      def int_aarch64_sve_ # op # _x2 : Guarded_SVE2_VG2_Multi_Multi_Intrinsic;
      def int_aarch64_sve_ # op # _x4 : Guarded_SVE2_VG4_Multi_Multi_Intrinsic;
    }
  }

  foreach op = ["smax", "smin", "umax", "umin"] in {
    let TargetGuards = [StreamingVariant<"sme2", I8_I16_I32_I64_Overloads>] in {
      def int_aarch64_sve_ # op # _single_x2 : Guarded_SVE2_VG2_Multi_Single_Intrinsic;
      def int_aarch64_sve_ # op # _single_x4 : Guarded_SVE2_VG4_Multi_Single_Intrinsic;
      def int_aarch64_sve_ # op # _x2 : Guarded_SVE2_VG2_Multi_Multi_Intrinsic;
      def int_aarch64_sve_ # op # _x4 : Guarded_SVE2_VG4_Multi_Multi_Intrinsic;
    }
  }

  //
  // Multi-vector floating point min/max number, scale, and multiply
  //

  foreach op = ["fmaxnm", "fminnm"] in {
    let TargetGuards = [
      StreamingVariant<"sme2", F32_F64_Overloads>,
      StreamingVariant<"sme2,sve-b16b16", BF16_Overloads>,
      StreamingVariant<"sme2", F16_Overloads>,
    ] in {
      def int_aarch64_sve_ # op # _single_x2 : Guarded_SVE2_VG2_Multi_Single_Intrinsic;
      def int_aarch64_sve_ # op # _single_x4 : Guarded_SVE2_VG4_Multi_Single_Intrinsic;
      def int_aarch64_sve_ # op # _x2 : Guarded_SVE2_VG2_Multi_Multi_Intrinsic;
      def int_aarch64_sve_ # op # _x4 : Guarded_SVE2_VG4_Multi_Multi_Intrinsic;
    }
  }
  let TargetGuards = [
    StreamingVariant<"sme2p2", F32_F64_Overloads>,
    StreamingVariant<"sme2,sve-bfscale", BF16_Overloads>,
    StreamingVariant<"sme2p2", F16_Overloads>,
  ] in {
    def int_aarch64_sve_fmul_single_x2 : Guarded_SVE2_VG2_Multi_Single_Intrinsic;
    def int_aarch64_sve_fmul_single_x4 : Guarded_SVE2_VG4_Multi_Single_Intrinsic;

    def int_aarch64_sve_fmul_x2 : Guarded_SVE2_VG2_Multi_Multi_Intrinsic;
    def int_aarch64_sve_fmul_x4 : Guarded_SVE2_VG4_Multi_Multi_Intrinsic;
  }

  //
  // Multi-vector floating point scale
  //

  let TargetGuards = [StreamingVariant<"sme2,sve-bfscale", BF16_Overloads>] in {
    def int_aarch64_sve_fscale_single_x2 : Guarded_SVE2_VG2_Multi_Single_Scale_Intrinsic;
    def int_aarch64_sve_fscale_single_x4 : Guarded_SVE2_VG4_Multi_Single_Scale_Intrinsic;

    def int_aarch64_sve_fscale_x2 : Guarded_SVE2_VG2_Multi_Multi_Scale_Intrinsic;
    def int_aarch64_sve_fscale_x4 : Guarded_SVE2_VG4_Multi_Multi_Scale_Intrinsic;
  }

  //
  // Multi-vector floating point absolute min/max number
  //

  foreach instr = ["famax", "famin"] in {
    let TargetGuards = [StreamingVariant<"sme2,faminmax", F16_F32_F64_Overloads>] in {
      def int_aarch64_sme_ # instr # _x2 : Guarded_SVE2_VG2_Multi_Multi_Intrinsic;
      def int_aarch64_sme_ # instr # _x4 : Guarded_SVE2_VG4_Multi_Multi_Intrinsic;
    }
  }

  //
  // Multi-vector vertical dot-products
  //

  let TargetGuards = [StreamingVariant<"sme2", BF16_F16_Overloads>] in
  def int_aarch64_sme_fvdot_lane_za32_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic;

  let TargetGuards = [StreamingVariant<"sme2", I16_Overloads>] in {
    def int_aarch64_sme_svdot_lane_za32_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic;
    def int_aarch64_sme_uvdot_lane_za32_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic;
  }

  let TargetGuards = [StreamingVariant<"sme2", I8_Overloads>] in {
    def int_aarch64_sme_svdot_lane_za32_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
    def int_aarch64_sme_uvdot_lane_za32_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
    def int_aarch64_sme_suvdot_lane_za32_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
    def int_aarch64_sme_usvdot_lane_za32_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
  }

  let TargetGuards = [StreamingVariant<"sme2,sme-i16i64", I16_Overloads>] in {
    def int_aarch64_sme_svdot_lane_za64_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
    def int_aarch64_sme_uvdot_lane_za64_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
  }

  //
  //Multi-vector floating-point convert from half-precision to deinterleaved single-precision.
  //
  let TargetGuards = [StreamingVariant<"sme-f16f16", F32_Overloads>] in
  def int_aarch64_sve_fcvtl_widen_x2  : Guarded_SVE2_CVT_WIDENING_VG2_Intrinsic;

  //
  // Multi-vector floating-point CVT from single-precision to interleaved half-precision/BFloat16
  //
  let TargetGuards = [StreamingVariant<"sme2", F32_Overloads>] in
  def int_aarch64_sve_fcvtn_x2  : Guarded_SVE2_CVT_VG2_SINGLE_Intrinsic;
  let TargetGuards = [StreamingVariant<"sme2", EmptyOverload>] in
  def int_aarch64_sve_bfcvtn_x2 : Guarded_SVE2_CVT_VG2_SINGLE_BF16_Intrinsic;

  //
  // Multi-vector convert to/from floating-point.
  //

  let TargetGuards = [StreamingVariant<"sme2", F32_Overloads>] in
  def int_aarch64_sve_fcvt_x2  : Guarded_SVE2_CVT_VG2_SINGLE_Intrinsic;

  let TargetGuards = [StreamingVariant<"sme2", EmptyOverload>] in
  def int_aarch64_sve_bfcvt_x2 : Guarded_SVE2_CVT_VG2_SINGLE_BF16_Intrinsic;

  let TargetGuards = [StreamingVariant<"sme2", [[llvm_nxv4i32_ty, LLVMType<nxv4f32>]]>] in {
    def int_aarch64_sve_fcvtzs_x2 : Guarded_SVE2_CVT_X2_Intrinsic;
    def int_aarch64_sve_fcvtzu_x2 : Guarded_SVE2_CVT_X2_Intrinsic;
  }

  let TargetGuards = [StreamingVariant<"sme2", [[llvm_nxv4f32_ty, LLVMType<nxv4i32>]]>] in {
    def int_aarch64_sve_scvtf_x2  : Guarded_SVE2_CVT_X2_Intrinsic;
    def int_aarch64_sve_ucvtf_x2  : Guarded_SVE2_CVT_X2_Intrinsic;
  }

  let TargetGuards = [StreamingVariant<"sme2", [[llvm_nxv4i32_ty, LLVMType<nxv4f32>]]>] in {
    def int_aarch64_sve_fcvtzs_x4 : Guarded_SVE2_CVT_X4_Intrinsic;
    def int_aarch64_sve_fcvtzu_x4 : Guarded_SVE2_CVT_X4_Intrinsic;
  }

  let TargetGuards = [StreamingVariant<"sme2", [[llvm_nxv4f32_ty, LLVMType<nxv4i32>]]>] in {
    def int_aarch64_sve_scvtf_x4  : Guarded_SVE2_CVT_X4_Intrinsic;
    def int_aarch64_sve_ucvtf_x4  : Guarded_SVE2_CVT_X4_Intrinsic;
  }

  let TargetGuards = [StreamingVariant<"sme-f16f16", F32_Overloads>] in
  def int_aarch64_sve_fcvt_widen_x2 : Guarded_SVE2_CVT_WIDENING_VG2_Intrinsic;

  //
  // Multi-vector saturating extract narrow
  //

  let TargetGuards = [StreamingVariant<"sme2", I32_Overloads>] in {
    def int_aarch64_sve_sqcvt_x2  : Guarded_SVE2_CVT_VG2_SINGLE_Intrinsic;
    def int_aarch64_sve_uqcvt_x2  : Guarded_SVE2_CVT_VG2_SINGLE_Intrinsic;
    def int_aarch64_sve_sqcvtu_x2 : Guarded_SVE2_CVT_VG2_SINGLE_Intrinsic;
  }

  let TargetGuards = [StreamingVariant<"sme2", I32_I64_Overloads>] in {
    def int_aarch64_sve_sqcvt_x4  : Guarded_SVE2_CVT_VG4_SINGLE_Intrinsic;
    def int_aarch64_sve_uqcvt_x4  : Guarded_SVE2_CVT_VG4_SINGLE_Intrinsic;
    def int_aarch64_sve_sqcvtu_x4 : Guarded_SVE2_CVT_VG4_SINGLE_Intrinsic;
  }

  //
  // Multi-vector saturating extract narrow and interleave
  //
  let TargetGuards = [Variant<"sve,(sve2p1|sme2)", "sve2p1|sme2", I32_Overloads>] in {
    def int_aarch64_sve_sqcvtn_x2  : Guarded_SVE2_CVT_VG2_SINGLE_Intrinsic;
    def int_aarch64_sve_uqcvtn_x2  : Guarded_SVE2_CVT_VG2_SINGLE_Intrinsic;
    def int_aarch64_sve_sqcvtun_x2 : Guarded_SVE2_CVT_VG2_SINGLE_Intrinsic;
  }
  let TargetGuards = [StreamingVariant<"sme2", I32_I64_Overloads>] in {
    def int_aarch64_sve_sqcvtn_x4  : Guarded_SVE2_CVT_VG4_SINGLE_Intrinsic;
    def int_aarch64_sve_uqcvtn_x4  : Guarded_SVE2_CVT_VG4_SINGLE_Intrinsic;
    def int_aarch64_sve_sqcvtun_x4 : Guarded_SVE2_CVT_VG4_SINGLE_Intrinsic;
  }

  //
  // Multi-vector saturating shift right narrow and interleave
  //
  let TargetGuards = [
    Variant<"sve,(sve2p3|sme2p3)", "sve2p3|sme2p3", [[LLVMType<nxv4i32>], [LLVMType<nxv8i16>]]>,
  ] in {
    def int_aarch64_sve_sqshrn_x2 : Guarded_SVE2_VG2_Multi_Imm_Intrinsic;
    def int_aarch64_sve_uqshrn_x2 : Guarded_SVE2_VG2_Multi_Imm_Intrinsic;
    def int_aarch64_sve_sqshrun_x2 : Guarded_SVE2_VG2_Multi_Imm_Intrinsic;
  }

  //
  // Multi-Single add/sub
  //

  class Guarded_SME2_Add_Sub_Write_VG2_Multi_Single_Intrinsic
      : GuardedDefaultIntrinsic<[],
          [llvm_i32_ty,
          llvm_anyvector_ty, LLVMMatchType<0>,
          LLVMMatchType<0>],
          [IntrWrite<[ZA]>, IntrWriteOnly]>;

  class Guarded_SME2_Add_Sub_Write_VG4_Multi_Single_Intrinsic
      : GuardedDefaultIntrinsic<[],
          [llvm_i32_ty,
          llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>,
          LLVMMatchType<0>],
        [IntrWrite<[ZA]>, IntrWriteOnly]>;

  let TargetGuards = [
    StreamingVariant<"sme2,sme-i16i64", I64_Overloads>,
    StreamingVariant<"sme2", I32_Overloads>,
  ] in {
    def int_aarch64_sme_add_write_single_za_vg1x2 : Guarded_SME2_Add_Sub_Write_VG2_Multi_Single_Intrinsic;
    def int_aarch64_sme_sub_write_single_za_vg1x2 : Guarded_SME2_Add_Sub_Write_VG2_Multi_Single_Intrinsic;
    def int_aarch64_sme_add_write_single_za_vg1x4 : Guarded_SME2_Add_Sub_Write_VG4_Multi_Single_Intrinsic;
    def int_aarch64_sme_sub_write_single_za_vg1x4 : Guarded_SME2_Add_Sub_Write_VG4_Multi_Single_Intrinsic;
  }

  //
  // Multi-Multi add/sub
  //
  class Guarded_SME2_Add_Sub_Write_VG2_Multi_Multi_Intrinsic
      : GuardedDefaultIntrinsic<[],
          [llvm_i32_ty,
          llvm_anyvector_ty, LLVMMatchType<0>,
          LLVMMatchType<0>, LLVMMatchType<0>],
          [IntrWrite<[ZA]>, IntrWriteOnly]>;

  class Guarded_SME2_Add_Sub_Write_VG4_Multi_Multi_Intrinsic
      : GuardedDefaultIntrinsic<[],
          [llvm_i32_ty,
          llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>,
          LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>,
          LLVMMatchType<0>, LLVMMatchType<0>],
          [IntrWrite<[ZA]>, IntrWriteOnly]>;

  let TargetGuards = [
    StreamingVariant<"sme2,sme-i16i64", I64_Overloads>,
    StreamingVariant<"sme2", I32_Overloads>,
  ] in {
    def int_aarch64_sme_add_write_za_vg1x2 : Guarded_SME2_Add_Sub_Write_VG2_Multi_Multi_Intrinsic;
    def int_aarch64_sme_sub_write_za_vg1x2 : Guarded_SME2_Add_Sub_Write_VG2_Multi_Multi_Intrinsic;
    def int_aarch64_sme_add_write_za_vg1x4 : Guarded_SME2_Add_Sub_Write_VG4_Multi_Multi_Intrinsic;
    def int_aarch64_sme_sub_write_za_vg1x4 : Guarded_SME2_Add_Sub_Write_VG4_Multi_Multi_Intrinsic;
  }

  // Multi-vector clamps
  let TargetGuards = [StreamingVariant<"sme2", I8_I16_I32_I64_Overloads>] in {
    def int_aarch64_sve_sclamp_single_x2 : Guarded_SVE2_VG2_Multi_Single_Single_Intrinsic;
    def int_aarch64_sve_uclamp_single_x2 : Guarded_SVE2_VG2_Multi_Single_Single_Intrinsic;

    def int_aarch64_sve_sclamp_single_x4 : Guarded_SVE2_VG4_Multi_Single_Single_Intrinsic;
    def int_aarch64_sve_uclamp_single_x4 : Guarded_SVE2_VG4_Multi_Single_Single_Intrinsic;
  }

  let TargetGuards = [StreamingVariant<"sme2", F16_F32_F64_Overloads>] in {
    def int_aarch64_sve_fclamp_single_x2 : Guarded_SVE2_VG2_Multi_Single_Single_Intrinsic;
    def int_aarch64_sve_fclamp_single_x4 : Guarded_SVE2_VG4_Multi_Single_Single_Intrinsic;
  }

  let TargetGuards = [StreamingVariant<"sme2,sve-b16b16", BF16_Overloads>] in {
    def int_aarch64_sve_bfclamp_single_x2 : Guarded_SVE2_VG2_Multi_Single_Single_Intrinsic;
    def int_aarch64_sve_bfclamp_single_x4 : Guarded_SVE2_VG4_Multi_Single_Single_Intrinsic;
  }

  //
  // Multi-vector add/sub and accumulate into ZA
  //
  foreach op = ["add", "sub"] in {
    let TargetGuards = [
      StreamingVariant<"sme-b16b16", BF16_Overloads>,
      StreamingVariant<"sme-f16f16|sme-f8f16", F16_Overloads>,
    ] in {
      def int_aarch64_sme_ # op # _za16_vg1x2 : Guarded_SME2_ZA_Write_VG2_Intrinsic;
      def int_aarch64_sme_ # op # _za16_vg1x4 : Guarded_SME2_ZA_Write_VG4_Intrinsic;
    }
  }

  foreach op = ["add", "sub"] in {
    let TargetGuards = [StreamingVariant<"sme2", [[llvm_nxv4f32_ty], [llvm_nxv4i32_ty]]>] in {
      def int_aarch64_sme_ # op # _za32_vg1x2 : Guarded_SME2_ZA_Write_VG2_Intrinsic;
      def int_aarch64_sme_ # op # _za32_vg1x4 : Guarded_SME2_ZA_Write_VG4_Intrinsic;
    }
  }

  foreach op = ["add", "sub"] in {
    let TargetGuards = [
      StreamingVariant<"sme2,sme-f64f64", F64_Overloads>,
      StreamingVariant<"sme2,sme-i16i64", I64_Overloads>,
    ] in {
      def int_aarch64_sme_ # op # _za64_vg1x2 : Guarded_SME2_ZA_Write_VG2_Intrinsic;
      def int_aarch64_sme_ # op # _za64_vg1x4 : Guarded_SME2_ZA_Write_VG4_Intrinsic;
    }
  }

  //
  // Move multi-vectors to/from ZA
  //

  let TargetGuards = [StreamingVariant<"sme2", AllPackedData_Overloads>] in {
    def int_aarch64_sme_read_hor_vg2   : Guarded_SME2_Matrix_TileVector_Read_VG2_Intrinsic;
    def int_aarch64_sme_read_hor_vg4   : Guarded_SME2_Matrix_TileVector_Read_VG4_Intrinsic;

    def int_aarch64_sme_read_ver_vg2   : Guarded_SME2_Matrix_TileVector_Read_VG2_Intrinsic;
    def int_aarch64_sme_read_ver_vg4   : Guarded_SME2_Matrix_TileVector_Read_VG4_Intrinsic;

    def int_aarch64_sme_read_vg1x2 : Guarded_SME2_ZA_ArrayVector_Read_VG2_Intrinsic;
    def int_aarch64_sme_read_vg1x4 : Guarded_SME2_ZA_ArrayVector_Read_VG4_Intrinsic;

    def int_aarch64_sme_write_hor_vg2 : Guarded_SME2_Matrix_TileVector_Write_VG2_Intrinsic;
    def int_aarch64_sme_write_hor_vg4 : Guarded_SME2_Matrix_TileVector_Write_VG4_Intrinsic;

    def int_aarch64_sme_write_ver_vg2 : Guarded_SME2_Matrix_TileVector_Write_VG2_Intrinsic;
    def int_aarch64_sme_write_ver_vg4 : Guarded_SME2_Matrix_TileVector_Write_VG4_Intrinsic;

    def int_aarch64_sme_write_vg1x2 : Guarded_SME2_ZA_ArrayVector_Write_VG2_Intrinsic;
    def int_aarch64_sme_write_vg1x4 : Guarded_SME2_ZA_ArrayVector_Write_VG4_Intrinsic;
  }

  //
  // Multi-Single Vector add
  //
  let TargetGuards = [StreamingVariant<"sme2", I8_I16_I32_I64_Overloads>] in {
    def int_aarch64_sve_add_single_x2 : Guarded_SVE2_VG2_Multi_Single_Intrinsic;
    def int_aarch64_sve_add_single_x4 : Guarded_SVE2_VG4_Multi_Single_Intrinsic;
  }

  // 2-way and 4-way multi-vector signed/unsigned integer dot-product
  foreach op = ["sdot", "udot"] in {
    let TargetGuards = [StreamingVariant<"sme2", I8_I16_Overloads>] in {
      def int_aarch64_sme_ # op # _single_za32_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Single_Intrinsic;
      def int_aarch64_sme_ # op # _single_za32_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Single_Intrinsic;
      def int_aarch64_sme_ # op # _za32_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Multi_Intrinsic;
      def int_aarch64_sme_ # op # _za32_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Multi_Intrinsic;
      def int_aarch64_sme_ # op # _lane_za32_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic;
      def int_aarch64_sme_ # op # _lane_za32_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
    }
  }

  foreach op = ["sdot", "udot"] in {
    let TargetGuards = [StreamingVariant<"sme2,sme-i16i64", I16_Overloads>] in {
      def int_aarch64_sme_ # op # _single_za64_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Single_Intrinsic;
      def int_aarch64_sme_ # op # _single_za64_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Single_Intrinsic;
      def int_aarch64_sme_ # op # _za64_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Multi_Intrinsic;
      def int_aarch64_sme_ # op # _za64_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Multi_Intrinsic;
      def int_aarch64_sme_ # op # _lane_za64_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic;
      def int_aarch64_sme_ # op # _lane_za64_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
    }
  }

  foreach ty = ["su", "us"] in {
    let TargetGuards = [StreamingVariant<"sme2", I8_Overloads>] in {
      def int_aarch64_sme_ # ty # dot_single_za32_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Single_Intrinsic;
      def int_aarch64_sme_ # ty # dot_single_za32_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Single_Intrinsic;

      def int_aarch64_sme_ # ty # dot_lane_za32_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic;
      def int_aarch64_sme_ # ty # dot_lane_za32_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
    }
  }

  let TargetGuards = [StreamingVariant<"sme2", I8_Overloads>] in {
    def int_aarch64_sme_usdot_za32_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Multi_Intrinsic;
    def int_aarch64_sme_usdot_za32_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Multi_Intrinsic;
  }

  // Multi-vector half-precision or bfloat floating-point dot-product
  let TargetGuards = [StreamingVariant<"sme2", BF16_F16_Overloads>] in {
    def int_aarch64_sme_fdot_single_za32_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Single_Intrinsic;
    def int_aarch64_sme_fdot_single_za32_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Single_Intrinsic;

    def int_aarch64_sme_fdot_za32_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Multi_Intrinsic;
    def int_aarch64_sme_fdot_za32_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Multi_Intrinsic;

    def int_aarch64_sme_fdot_lane_za32_vg1x2 : Guarded_SME2_Matrix_ArrayVector_VG2_Multi_Index_Intrinsic;
    def int_aarch64_sme_fdot_lane_za32_vg1x4 : Guarded_SME2_Matrix_ArrayVector_VG4_Multi_Index_Intrinsic;
  }

  // Multi-vector zip and unzips
  let TargetGuards = [StreamingVariant<"sme2", AllPackedData_Overloads>] in {
    def int_aarch64_sve_zip_x2  : Guarded_SVE2_VG2_ZipUzp_Intrinsic;
    def int_aarch64_sve_zipq_x2 : Guarded_SVE2_VG2_ZipUzp_Intrinsic;
    def int_aarch64_sve_zip_x4  : Guarded_SVE2_VG4_ZipUzp_Intrinsic;
    def int_aarch64_sve_zipq_x4 : Guarded_SVE2_VG4_ZipUzp_Intrinsic;
    def int_aarch64_sve_uzp_x2  : Guarded_SVE2_VG2_ZipUzp_Intrinsic;
    def int_aarch64_sve_uzpq_x2 : Guarded_SVE2_VG2_ZipUzp_Intrinsic;
    def int_aarch64_sve_uzp_x4  : Guarded_SVE2_VG4_ZipUzp_Intrinsic;
    def int_aarch64_sve_uzpq_x4 : Guarded_SVE2_VG4_ZipUzp_Intrinsic;
  }

  // Vector dot-products (2-way)
  let TargetGuards = [
    Variant<"sve,(sve2p1|sme2)", "sve2p1|sme2", I32_Overloads>,
    Variant<"sve,(sve2p3|sme2p3)", "sve2p3|sme2p3", I16_Overloads>,
  ] in {
    def int_aarch64_sve_sdot_x2 : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;
    def int_aarch64_sve_udot_x2 : Guarded_SVE2_3VectorArg_Long_Intrinsic<[IntrSpeculatable]>;

    def int_aarch64_sve_sdot_lane_x2 : Guarded_SVE2_3VectorArgIndexed_Long_Intrinsic<[IntrSpeculatable]>;
    def int_aarch64_sve_udot_lane_x2 : Guarded_SVE2_3VectorArgIndexed_Long_Intrinsic<[IntrSpeculatable]>;
  }

  let TargetGuards = [Variant<"sve,(sve2p1|sme2)", "sve2p1|sme2", F32_Overloads>] in {
    def int_aarch64_sve_fdot_x2 : Guarded_SVE2_3VectorArg_Long_Intrinsic;
    def int_aarch64_sve_fdot_lane_x2 : Guarded_SVE2_3VectorArgIndexed_Long_Intrinsic;
  }

  //
  // Signed/unsigned multi-vector unpacks
  //
  let TargetGuards = [StreamingVariant<"sme2", I16_I32_I64_Overloads>] in {
    def int_aarch64_sve_sunpk_x2 : Guarded_SVE2_VG2_Unpk_Intrinsic;
    def int_aarch64_sve_uunpk_x2 : Guarded_SVE2_VG2_Unpk_Intrinsic;
    def int_aarch64_sve_sunpk_x4 : Guarded_SVE2_VG4_Unpk_Intrinsic;
    def int_aarch64_sve_uunpk_x4 : Guarded_SVE2_VG4_Unpk_Intrinsic;
  }

  // 2-way and 4-way vector selects
  let TargetGuards = [StreamingVariant<"sme2", AllPackedData_Overloads>] in {
    def int_aarch64_sve_sel_x2  : Guarded_SVE2_VG2_Sel_Intrinsic;
    def int_aarch64_sve_sel_x4  : Guarded_SVE2_VG4_Sel_Intrinsic;
  }

  let TargetGuards = [Variant<"sme2", "sme2", PTR_Overloads>] in {
    def int_aarch64_sme_ldr_zt
    : GuardedDefaultIntrinsic<[], [llvm_i32_ty, llvm_anyptr_ty],
                            [IntrRead<[ArgMem, ZT0]>, IntrWrite<[ZT0]>]>;
    def int_aarch64_sme_str_zt
    : GuardedDefaultIntrinsic<[], [llvm_i32_ty, llvm_anyptr_ty],
                            [IntrRead<[ZT0]>, IntrWrite<[ArgMem, ZT0]>]>;
  }

  //
  //  Zero ZT0
  //
  let TargetGuards = [Variant<"sme2", "sme2", EmptyOverload>] in
  def int_aarch64_sme_zero_zt : GuardedDefaultIntrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>, IntrWrite<[ZT0]>, IntrWriteOnly]>;

  defvar SmeLuti2LaneZtTypes = [
    [llvm_nxv16i8_ty],
    [llvm_nxv4f32_ty],
    [llvm_nxv4i32_ty],
    [llvm_nxv8bf16_ty],
    [llvm_nxv8f16_ty],
    [llvm_nxv8i16_ty],
  ];

  defvar SmeLuti4LaneZtX4Types = [
    [llvm_nxv4f32_ty],
    [llvm_nxv4i32_ty],
    [llvm_nxv8bf16_ty],
    [llvm_nxv8f16_ty],
    [llvm_nxv8i16_ty],
  ];

  //
  // Lookup table expand one register
  //
  let TargetGuards = [StreamingVariant<"sme2", SmeLuti2LaneZtTypes>] in {
    def int_aarch64_sme_luti2_lane_zt
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty], [llvm_i32_ty, llvm_nxv16i8_ty, llvm_i32_ty],
                            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, IntrRead<[ZT0]>, IntrReadOnly]>;
    def int_aarch64_sme_luti4_lane_zt
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty], [llvm_i32_ty, llvm_nxv16i8_ty, llvm_i32_ty],
                            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, IntrRead<[ZT0]>, IntrReadOnly]>;
  }

  let TargetGuards = [StreamingVariant<"sme2p3", EmptyOverload>] in
  def int_aarch64_sme_luti6_zt
    : GuardedDefaultIntrinsic<[llvm_nxv16i8_ty], [llvm_i32_ty, llvm_nxv16i8_ty],
                            [ImmArg<ArgIndex<0>>, IntrRead<[ZT0]>, IntrReadOnly]>;

  // Lookup table expand two registers
  //
  let TargetGuards = [StreamingVariant<"sme2", SmeLuti2LaneZtTypes>] in {
    def int_aarch64_sme_luti2_lane_zt_x2
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>], [llvm_i32_ty, llvm_nxv16i8_ty, llvm_i32_ty],
                            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, IntrRead<[ZT0]>, IntrReadOnly]>;
    def int_aarch64_sme_luti4_lane_zt_x2
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>], [llvm_i32_ty, llvm_nxv16i8_ty, llvm_i32_ty],
                            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, IntrRead<[ZT0]>, IntrReadOnly]>;
  }

  //
  // Lookup table expand four registers
  //
  let TargetGuards = [StreamingVariant<"sme2", SmeLuti2LaneZtTypes>] in
  def int_aarch64_sme_luti2_lane_zt_x4
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                            [llvm_i32_ty, llvm_nxv16i8_ty, llvm_i32_ty],
                            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, IntrRead<[ZT0]>, IntrReadOnly]>;
  let TargetGuards = [StreamingVariant<"sme2", SmeLuti4LaneZtX4Types>] in
  def int_aarch64_sme_luti4_lane_zt_x4
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                            [llvm_i32_ty, llvm_nxv16i8_ty, llvm_i32_ty],
                            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, IntrRead<[ZT0]>, IntrReadOnly]>;
  let TargetGuards = [StreamingVariant<"sme2p3", BF16_F16_I16_Overloads>] in {
    def int_aarch64_sme_luti6_lane_x4_x2
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                            [LLVMMatchType<0>, LLVMMatchType<0>, llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_i32_ty],
                            [ImmArg<ArgIndex<4>>, IntrNoMem, IntrSpeculatable]>;
    def int_aarch64_sme_luti6_lane_x4_x3
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                            [LLVMMatchType<0>, LLVMMatchType<0>, llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_i32_ty],
                            [ImmArg<ArgIndex<5>>, IntrNoMem, IntrSpeculatable]>;
  }

  let TargetGuards = [StreamingVariant<"sme-lutv2", I8_Overloads>] in
  def int_aarch64_sme_luti4_zt_x4
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                            [llvm_i32_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty],
                            [ImmArg<ArgIndex<0>>, IntrRead<[ZT0]>, IntrReadOnly]>;
  let TargetGuards = [StreamingVariant<"sme2p3", EmptyOverload>] in
  def int_aarch64_sme_luti6_zt_x4
    : GuardedDefaultIntrinsic<[llvm_nxv16i8_ty, llvm_nxv16i8_ty,
                             llvm_nxv16i8_ty, llvm_nxv16i8_ty],
                            [llvm_i32_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty],
                            [ImmArg<ArgIndex<0>>, IntrRead<[ZT0]>, IntrReadOnly]>;

  //
  // Register scaling
  //
  let TargetGuards = [StreamingVariant<"sme2,fp8", F16_F32_F64_Overloads>] in {
    def int_aarch64_sme_fp8_scale_single_x2
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                            [LLVMMatchType<0>, LLVMMatchType<0>, LLVMVectorOfBitcastsToInt<0>],
                            [IntrNoMem]>;
    def int_aarch64_sme_fp8_scale_single_x4
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                            [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>, LLVMVectorOfBitcastsToInt<0>],
                            [IntrNoMem]>;
    def int_aarch64_sme_fp8_scale_x2
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                            [LLVMMatchType<0>, LLVMMatchType<0>,
                             LLVMVectorOfBitcastsToInt<0>, LLVMVectorOfBitcastsToInt<0>],
                            [IntrNoMem]>;
    def int_aarch64_sme_fp8_scale_x4
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                            [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>,
                             LLVMVectorOfBitcastsToInt<0>, LLVMVectorOfBitcastsToInt<0>, LLVMVectorOfBitcastsToInt<0>, LLVMVectorOfBitcastsToInt<0>],
                            [IntrNoMem]>;
  }

  //
  // SVE2.3/SME2.3 - Multi-vector narrowing convert to floating point
  //
  defvar SveFcvtzsnX2Types = [
    [llvm_nxv16i8_ty, LLVMType<nxv8f16>],
    [llvm_nxv4i32_ty, LLVMType<nxv2f64>],
    [llvm_nxv8i16_ty, LLVMType<nxv4f32>],
  ];

  let TargetGuards = [Variant<"sve,(sve2p3|sme2p3)", "sve2p3|sme2p3", SveFcvtzsnX2Types>] in {
    def int_aarch64_sve_fcvtzsn_x2: Guarded_AdvSIMD_2Arg_FloatCompare_Intrinsic;
    def int_aarch64_sve_fcvtzun_x2: Guarded_AdvSIMD_2Arg_FloatCompare_Intrinsic;
  }
}

// SVE2.1 - ZIPQ1, ZIPQ2, UZPQ1, UZPQ2
//
let TargetGuards = [Variant<"sve,(sve2p1|sme2p1)", "sve2p1|sme2p1", AllPackedData_Overloads>] in {
  def int_aarch64_sve_zipq1 : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_zipq2 : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uzpq1 : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_uzpq2 : Guarded_AdvSIMD_2VectorArg_Intrinsic<[IntrSpeculatable]>;
}

// SVE2.1 - Programmable table lookup within each quadword vector segment
// (zeroing)/(merging)
//
let TargetGuards = [Variant<"sve,(sve2p1|sme2p1)", "sve2p1|sme2p1", AllPackedData_Overloads>] in {
  def int_aarch64_sve_tblq : Guarded_AdvSIMD_SVE_TBL_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_tbxq : Guarded_AdvSIMD_SVE2_TBX_Intrinsic<[IntrSpeculatable]>;
}

// SVE2.1 - Extract vector segment from each pair of quadword segments.
//
let TargetGuards = [Variant<"sve,(sve2p1|sme2p1)", "sve2p1|sme2p1", AllPackedData_Overloads>] in
def int_aarch64_sve_extq : Guarded_AdvSIMD_2VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;

//
// SVE2.1 - Move predicate to/from vector
//
let TargetGuards = [Variant<"sve,(sve2p1|sme2p1)", "sve2p1|sme2p1", I8_I16_I32_I64_Overloads>] in {
  def int_aarch64_sve_pmov_to_pred_lane : Guarded_SVE2_1VectorArgIndexed_Pred_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_pmov_to_pred_lane_zero : Guarded_SVE2_1VectorArg_Pred_Intrinsic<[IntrSpeculatable]>;
  def int_aarch64_sve_pmov_to_vector_lane_zeroing : Guarded_SVE2_Pred_1VectorArg_Intrinsic<[IntrSpeculatable]>;
}

let TargetGuards = [Variant<"sve,(sve2p1|sme2p1)", "sve2p1|sme2p1", I16_I32_I64_Overloads>] in
def int_aarch64_sve_pmov_to_vector_lane_merging : Guarded_SVE2_Pred_1VectorArgIndexed_Intrinsic<[IntrSpeculatable]>;

// SVE2/SME2 - Floating point absolute maximum and minimum

let TargetGuards = [
  Variant<"sve,(sve2|sme2),faminmax", "sme2,faminmax", F16_F32_F64_Overloads>,
] in {
  def int_aarch64_sve_famax   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_famax_u : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_famin   : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
  def int_aarch64_sve_famin_u : Guarded_AdvSIMD_Pred2VectorArg_Intrinsic;
}
// Neon absolute maximum and minimum
def int_aarch64_neon_famax :  AdvSIMD_2VectorArg_Intrinsic;
def int_aarch64_neon_famin :  AdvSIMD_2VectorArg_Intrinsic;

//
// FP8 Intrinsics
//
let TargetPrefix = "aarch64" in {

  // SVE Widening Conversions
  class Guarded_SVE2_FP8_Cvt
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                            [llvm_nxv16i8_ty],
                            [IntrRead<[FPMR]>, IntrReadOnly]>;

  let TargetGuards = [Variant<"sve,(sve2|sme2),fp8", "sme2,fp8", BF16_F16_Overloads>] in {
    def int_aarch64_sve_fp8_cvt1   : Guarded_SVE2_FP8_Cvt;
    def int_aarch64_sve_fp8_cvt2   : Guarded_SVE2_FP8_Cvt;
    def int_aarch64_sve_fp8_cvtlt1 : Guarded_SVE2_FP8_Cvt;
    def int_aarch64_sve_fp8_cvtlt2 : Guarded_SVE2_FP8_Cvt;
  }

  // SVE Narrowing Conversions
  class Guarded_SVE2_FP8_Narrow_Cvt
    : GuardedDefaultIntrinsic<[llvm_nxv16i8_ty],
                            [llvm_anyvector_ty, LLVMMatchType<0>],
                            [IntrRead<[FPMR]>, IntrReadOnly]>;
  let TargetGuards = [
    Variant<"sve,(sve2|sme2),fp8", "sme2,fp8", [[LLVMType<nxv8bf16>], [LLVMType<nxv8f16>]]>,
  ] in
  def int_aarch64_sve_fp8_cvtn  : Guarded_SVE2_FP8_Narrow_Cvt;

  let TargetGuards = [Variant<"sve,(sve2|sme2),fp8", "sme2,fp8", F32_Overloads>] in {
    def int_aarch64_sve_fp8_cvtnb : Guarded_SVE2_FP8_Narrow_Cvt;
    def int_aarch64_sve_fp8_cvtnt
    : GuardedDefaultIntrinsic<[llvm_nxv16i8_ty],
                            [llvm_nxv16i8_ty, llvm_anyvector_ty, LLVMMatchType<0>],
                            [IntrRead<[FPMR]>, IntrReadOnly]>;
  }

  // Dot product
  class Guarded_SVE2_FP8_FMLA_FDOT
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                            [LLVMMatchType<0>,
                             llvm_nxv16i8_ty, llvm_nxv16i8_ty],
                            [IntrRead<[FPMR]>, IntrReadOnly]>;
  class Guarded_SVE2_FP8_FMLA_FDOT_Lane
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                            [LLVMMatchType<0>,
                             llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_i32_ty],
                            [IntrRead<[FPMR]>, IntrReadOnly, ImmArg<ArgIndex<3>>]>;
  let TargetGuards = [
    Variant<"sve2,fp8dot4", "ssve-fp8dot4", F32_Overloads>,
    Variant<"sve2,fp8dot2", "ssve-fp8dot2", F16_Overloads>,
  ] in {
    def int_aarch64_sve_fp8_fdot      : Guarded_SVE2_FP8_FMLA_FDOT;
    def int_aarch64_sve_fp8_fdot_lane : Guarded_SVE2_FP8_FMLA_FDOT_Lane;
  }

  // Fused multiply-add
  let TargetGuards = [Variant<"sve2,fp8fma", "ssve-fp8fma", F16_Overloads>] in {
    def int_aarch64_sve_fp8_fmlalb      : Guarded_SVE2_FP8_FMLA_FDOT;
    def int_aarch64_sve_fp8_fmlalb_lane : Guarded_SVE2_FP8_FMLA_FDOT_Lane;

    def int_aarch64_sve_fp8_fmlalt      : Guarded_SVE2_FP8_FMLA_FDOT;
    def int_aarch64_sve_fp8_fmlalt_lane : Guarded_SVE2_FP8_FMLA_FDOT_Lane;
  }

  let TargetGuards = [Variant<"sve2,fp8fma", "ssve-fp8fma", F32_Overloads>] in {
    def int_aarch64_sve_fp8_fmlallbb      : Guarded_SVE2_FP8_FMLA_FDOT;
    def int_aarch64_sve_fp8_fmlallbb_lane : Guarded_SVE2_FP8_FMLA_FDOT_Lane;

    def int_aarch64_sve_fp8_fmlallbt      : Guarded_SVE2_FP8_FMLA_FDOT;
    def int_aarch64_sve_fp8_fmlallbt_lane : Guarded_SVE2_FP8_FMLA_FDOT_Lane;

    def int_aarch64_sve_fp8_fmlalltb      : Guarded_SVE2_FP8_FMLA_FDOT;
    def int_aarch64_sve_fp8_fmlalltb_lane : Guarded_SVE2_FP8_FMLA_FDOT_Lane;

    def int_aarch64_sve_fp8_fmlalltt      : Guarded_SVE2_FP8_FMLA_FDOT;
    def int_aarch64_sve_fp8_fmlalltt_lane : Guarded_SVE2_FP8_FMLA_FDOT_Lane;
  }

  class Guarded_SVE2_FP8_CVT_X2_Single_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                            [llvm_nxv16i8_ty],
                            [IntrRead<[FPMR]>, IntrReadOnly]>;

  class Guarded_SVE2_FP8_CVT_Single_X4_Intrinsic
    : GuardedDefaultIntrinsic<[llvm_nxv16i8_ty],
                            [llvm_nxv4f32_ty, llvm_nxv4f32_ty, llvm_nxv4f32_ty, llvm_nxv4f32_ty],
                            [IntrRead<[FPMR]>, IntrReadOnly]>;

  class Guarded_SME_FP8_OuterProduct_Intrinsic
      : GuardedDefaultIntrinsic<[],
          [llvm_i32_ty,
          llvm_nxv16i1_ty, llvm_nxv16i1_ty,
          llvm_nxv16i8_ty, llvm_nxv16i8_ty],
          [ImmArg<ArgIndex<0>>, IntrRead<[FPMR, ZA]>, IntrWrite<[ZA]>]>;

  class Guarded_SME_FP8_ZA_LANE_VGx1_Intrinsic
   : GuardedDefaultIntrinsic<[], [llvm_i32_ty,
                               llvm_nxv16i8_ty,
                               llvm_nxv16i8_ty,
                               llvm_i32_ty],
                          [IntrRead<[FPMR, ZA]>, IntrWrite<[ZA]>, ImmArg<ArgIndex<3>>]>;

  class Guarded_SME_FP8_ZA_LANE_VGx2_Intrinsic
    : GuardedDefaultIntrinsic<[], [llvm_i32_ty,
                                llvm_nxv16i8_ty, llvm_nxv16i8_ty,
                                llvm_nxv16i8_ty,
                                llvm_i32_ty],
                            [IntrRead<[FPMR, ZA]>, IntrWrite<[ZA]>, ImmArg<ArgIndex<4>>]>;

  class Guarded_SME_FP8_ZA_LANE_VGx4_Intrinsic
   : GuardedDefaultIntrinsic<[], [llvm_i32_ty,
                                llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty,
                                llvm_nxv16i8_ty,
                                llvm_i32_ty],
                            [IntrRead<[FPMR, ZA]>, IntrWrite<[ZA]>, ImmArg<ArgIndex<6>>]>;

  class Guarded_SME_FP8_ZA_VGx1_Intrinsic
    : GuardedDefaultIntrinsic<[], [llvm_i32_ty,
                                llvm_nxv16i8_ty,
                                llvm_nxv16i8_ty],
                            [IntrRead<[FPMR, ZA]>, IntrWrite<[ZA]>]>;

  class Guarded_SME_FP8_ZA_SINGLE_VGx2_Intrinsic
    : GuardedDefaultIntrinsic<[], [llvm_i32_ty,
                                llvm_nxv16i8_ty, llvm_nxv16i8_ty,
                                llvm_nxv16i8_ty],
                            [IntrRead<[FPMR, ZA]>, IntrWrite<[ZA]>]>;

  class Guarded_SME_FP8_ZA_SINGLE_VGx4_Intrinsic
    : GuardedDefaultIntrinsic<[], [llvm_i32_ty,
                                llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty,
                                llvm_nxv16i8_ty],
                              [IntrRead<[FPMR, ZA]>, IntrWrite<[ZA]>]>;

  class Guarded_SME_FP8_ZA_MULTI_VGx2_Intrinsic
    : GuardedDefaultIntrinsic<[], [llvm_i32_ty,
                                 llvm_nxv16i8_ty, llvm_nxv16i8_ty,
                                 llvm_nxv16i8_ty, llvm_nxv16i8_ty],
                            [IntrRead<[FPMR, ZA]>, IntrWrite<[ZA]>]>;

  class Guarded_SME_FP8_ZA_MULTI_VGx4_Intrinsic
    : GuardedDefaultIntrinsic<[], [llvm_i32_ty,
                                 llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty,
                                 llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty],
                            [IntrRead<[FPMR, ZA]>, IntrWrite<[ZA]>]>;
  //
  // CVT from FP8 to half-precision/BFloat16 multi-vector
  //
  let TargetGuards = [StreamingVariant<"sme2,fp8", BF16_F16_Overloads>] in {
    def int_aarch64_sve_fp8_cvt1_x2 : Guarded_SVE2_FP8_CVT_X2_Single_Intrinsic;
    def int_aarch64_sve_fp8_cvt2_x2 : Guarded_SVE2_FP8_CVT_X2_Single_Intrinsic;
  }

  //
  // CVT from FP8 to deinterleaved half-precision/BFloat16 multi-vector
  //
  let TargetGuards = [StreamingVariant<"sme2,fp8", BF16_F16_Overloads>] in {
    def int_aarch64_sve_fp8_cvtl1_x2 : Guarded_SVE2_FP8_CVT_X2_Single_Intrinsic;
    def int_aarch64_sve_fp8_cvtl2_x2 : Guarded_SVE2_FP8_CVT_X2_Single_Intrinsic;
  }

  //
  // CVT to FP8 from half-precision/BFloat16/single-precision multi-vector
  //
  let TargetGuards = [StreamingVariant<"sme2,fp8", [[LLVMType<nxv8bf16>], [LLVMType<nxv8f16>]]>] in
  def int_aarch64_sve_fp8_cvt_x2
    : GuardedDefaultIntrinsic<[llvm_nxv16i8_ty],
                            [llvm_anyvector_ty, LLVMMatchType<0>],
                            [IntrRead<[FPMR]>, IntrReadOnly]>;

  let TargetGuards = [StreamingVariant<"sme2,fp8", EmptyOverload>] in {
    def int_aarch64_sve_fp8_cvt_x4  : Guarded_SVE2_FP8_CVT_Single_X4_Intrinsic;
    def int_aarch64_sve_fp8_cvtn_x4 : Guarded_SVE2_FP8_CVT_Single_X4_Intrinsic;
  }

  // FP8 outer product
  let TargetGuards = [StreamingVariant<"sme-f8f16", EmptyOverload>] in
  def int_aarch64_sme_fp8_fmopa_za16 : Guarded_SME_FP8_OuterProduct_Intrinsic;
  let TargetGuards = [StreamingVariant<"sme-f8f32", EmptyOverload>] in
  def int_aarch64_sme_fp8_fmopa_za32 : Guarded_SME_FP8_OuterProduct_Intrinsic;

  //
  // ZA multiply-add
  //
  // Double-vector groups (F8F16)
  let TargetGuards = [StreamingVariant<"sme-f8f16", EmptyOverload>] in {
    def int_aarch64_sme_fp8_fmlal_lane_za16_vg2x1 : Guarded_SME_FP8_ZA_LANE_VGx1_Intrinsic;
    def int_aarch64_sme_fp8_fmlal_lane_za16_vg2x2 : Guarded_SME_FP8_ZA_LANE_VGx2_Intrinsic;
    def int_aarch64_sme_fp8_fmlal_lane_za16_vg2x4 : Guarded_SME_FP8_ZA_LANE_VGx4_Intrinsic;
  }
  // Single-Single
  let TargetGuards = [StreamingVariant<"sme-f8f16", EmptyOverload>] in
  def int_aarch64_sme_fp8_fmlal_za16_vg2x1 : Guarded_SME_FP8_ZA_VGx1_Intrinsic;
  // Multi-Single
  let TargetGuards = [StreamingVariant<"sme-f8f16", EmptyOverload>] in {
    def int_aarch64_sme_fp8_fmlal_single_za16_vg2x2 : Guarded_SME_FP8_ZA_SINGLE_VGx2_Intrinsic;
    def int_aarch64_sme_fp8_fmlal_single_za16_vg2x4 : Guarded_SME_FP8_ZA_SINGLE_VGx4_Intrinsic;
  }
  // Multi-Multi
  let TargetGuards = [StreamingVariant<"sme-f8f16", EmptyOverload>] in {
    def int_aarch64_sme_fp8_fmlal_multi_za16_vg2x2 : Guarded_SME_FP8_ZA_MULTI_VGx2_Intrinsic;
    def int_aarch64_sme_fp8_fmlal_multi_za16_vg2x4 : Guarded_SME_FP8_ZA_MULTI_VGx4_Intrinsic;
  }

  // Quad-vector groups (F8F32)
  let TargetGuards = [StreamingVariant<"sme-f8f32", EmptyOverload>] in {
    def int_aarch64_sme_fp8_fmlall_lane_za32_vg4x1 : Guarded_SME_FP8_ZA_LANE_VGx1_Intrinsic;
    def int_aarch64_sme_fp8_fmlall_lane_za32_vg4x2 : Guarded_SME_FP8_ZA_LANE_VGx2_Intrinsic;
    def int_aarch64_sme_fp8_fmlall_lane_za32_vg4x4 : Guarded_SME_FP8_ZA_LANE_VGx4_Intrinsic;
  }
  // Single-Single
  let TargetGuards = [StreamingVariant<"sme-f8f32", EmptyOverload>] in
  def int_aarch64_sme_fp8_fmlall_za32_vg4x1 : Guarded_SME_FP8_ZA_VGx1_Intrinsic;
  // Multi-Single
  let TargetGuards = [StreamingVariant<"sme-f8f32", EmptyOverload>] in {
    def int_aarch64_sme_fp8_fmlall_single_za32_vg4x2 : Guarded_SME_FP8_ZA_SINGLE_VGx2_Intrinsic;
    def int_aarch64_sme_fp8_fmlall_single_za32_vg4x4 : Guarded_SME_FP8_ZA_SINGLE_VGx4_Intrinsic;
  }
  // Multi-Multi
  let TargetGuards = [StreamingVariant<"sme-f8f32", EmptyOverload>] in {
    def int_aarch64_sme_fp8_fmlall_multi_za32_vg4x2 : Guarded_SME_FP8_ZA_MULTI_VGx2_Intrinsic;
    def int_aarch64_sme_fp8_fmlall_multi_za32_vg4x4 : Guarded_SME_FP8_ZA_MULTI_VGx4_Intrinsic;
  }

  //
  // FP8 FDOT intrinsics
  //
  // (indexed)
  let TargetGuards = [StreamingVariant<"sme-f8f16", EmptyOverload>] in {
    def int_aarch64_sme_fp8_fdot_lane_za16_vg1x2 : Guarded_SME_FP8_ZA_LANE_VGx2_Intrinsic;
    def int_aarch64_sme_fp8_fdot_lane_za16_vg1x4 : Guarded_SME_FP8_ZA_LANE_VGx4_Intrinsic;
  }

  let TargetGuards = [StreamingVariant<"sme-f8f32", EmptyOverload>] in {
    def int_aarch64_sme_fp8_fdot_lane_za32_vg1x2 : Guarded_SME_FP8_ZA_LANE_VGx2_Intrinsic;
    def int_aarch64_sme_fp8_fdot_lane_za32_vg1x4 : Guarded_SME_FP8_ZA_LANE_VGx4_Intrinsic;
  }
  // Single
  let TargetGuards = [StreamingVariant<"sme-f8f16", EmptyOverload>] in {
    def int_aarch64_sme_fp8_fdot_single_za16_vg1x2 : Guarded_SME_FP8_ZA_SINGLE_VGx2_Intrinsic;
    def int_aarch64_sme_fp8_fdot_single_za16_vg1x4 : Guarded_SME_FP8_ZA_SINGLE_VGx4_Intrinsic;
  }

  let TargetGuards = [StreamingVariant<"sme-f8f32", EmptyOverload>] in {
    def int_aarch64_sme_fp8_fdot_single_za32_vg1x2 : Guarded_SME_FP8_ZA_SINGLE_VGx2_Intrinsic;
    def int_aarch64_sme_fp8_fdot_single_za32_vg1x4 : Guarded_SME_FP8_ZA_SINGLE_VGx4_Intrinsic;
  }
  // Multi
  let TargetGuards = [StreamingVariant<"sme-f8f16", EmptyOverload>] in {
    def int_aarch64_sme_fp8_fdot_multi_za16_vg1x2 : Guarded_SME_FP8_ZA_MULTI_VGx2_Intrinsic;
    def int_aarch64_sme_fp8_fdot_multi_za16_vg1x4 : Guarded_SME_FP8_ZA_MULTI_VGx4_Intrinsic;
  }

  let TargetGuards = [StreamingVariant<"sme-f8f32", EmptyOverload>] in {
    def int_aarch64_sme_fp8_fdot_multi_za32_vg1x2 : Guarded_SME_FP8_ZA_MULTI_VGx2_Intrinsic;
    def int_aarch64_sme_fp8_fdot_multi_za32_vg1x4 : Guarded_SME_FP8_ZA_MULTI_VGx4_Intrinsic;
  }

  // FVDOT
  let TargetGuards = [StreamingVariant<"sme-f8f16", EmptyOverload>] in
  def int_aarch64_sme_fp8_fvdot_lane_za16_vg1x2  : Guarded_SME_FP8_ZA_LANE_VGx2_Intrinsic;
  let TargetGuards = [StreamingVariant<"sme-f8f32", EmptyOverload>] in {
    def int_aarch64_sme_fp8_fvdotb_lane_za32_vg1x4 : Guarded_SME_FP8_ZA_LANE_VGx2_Intrinsic;
    def int_aarch64_sme_fp8_fvdott_lane_za32_vg1x4 : Guarded_SME_FP8_ZA_LANE_VGx2_Intrinsic;
  }

  // AES2
  class Guarded_SVE2_Crypto_LANE_X2_Intrinsic
  : GuardedDefaultIntrinsic<[llvm_nxv16i8_ty, llvm_nxv16i8_ty],
      [llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_i32_ty],
      [ImmArg<ArgIndex<3>>, IntrNoMem]>;
  class Guarded_SVE2_Crypto_LANE_X4_Intrinsic
  : GuardedDefaultIntrinsic<[llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty],
      [llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_nxv16i8_ty,
       llvm_nxv16i8_ty, llvm_nxv16i8_ty, llvm_i32_ty],
      [ImmArg<ArgIndex<5>>, IntrNoMem]>;

  let TargetGuards = [Variant<"sve,sve-aes2", "sve-aes2,ssve-aes", EmptyOverload>] in {
    def int_aarch64_sve_aesd_lane_x2 : Guarded_SVE2_Crypto_LANE_X2_Intrinsic;
    def int_aarch64_sve_aesdimc_lane_x2 : Guarded_SVE2_Crypto_LANE_X2_Intrinsic;
    def int_aarch64_sve_aese_lane_x2 : Guarded_SVE2_Crypto_LANE_X2_Intrinsic;
    def int_aarch64_sve_aesemc_lane_x2 : Guarded_SVE2_Crypto_LANE_X2_Intrinsic;

    def int_aarch64_sve_aesd_lane_x4 : Guarded_SVE2_Crypto_LANE_X4_Intrinsic;
    def int_aarch64_sve_aesdimc_lane_x4 : Guarded_SVE2_Crypto_LANE_X4_Intrinsic;
    def int_aarch64_sve_aese_lane_x4 : Guarded_SVE2_Crypto_LANE_X4_Intrinsic;
    def int_aarch64_sve_aesemc_lane_x4 : Guarded_SVE2_Crypto_LANE_X4_Intrinsic;

    def int_aarch64_sve_pmull_pair_x2 : GuardedDefaultIntrinsic<[llvm_nxv2i64_ty, llvm_nxv2i64_ty],
      [llvm_nxv2i64_ty, llvm_nxv2i64_ty], [IntrNoMem]>;
    def int_aarch64_sve_pmlal_pair_x2 : GuardedDefaultIntrinsic<[llvm_nxv2i64_ty, llvm_nxv2i64_ty],
      [llvm_nxv2i64_ty, llvm_nxv2i64_ty, llvm_nxv2i64_ty, llvm_nxv2i64_ty], [IntrNoMem]>;
  }
}
