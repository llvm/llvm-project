//===- IntrinsicsAArch64.td - Defines AArch64 intrinsics -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the AArch64-specific intrinsics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Advanced SIMD (NEON)

let TargetPrefix = "aarch64" in {  // All intrinsics start with "llvm.aarch64.".

// Vector Absolute Compare (Floating Point)
def int_aarch64_neon_vacgeq :
  Intrinsic<[llvm_v2i64_ty], [llvm_v2f64_ty, llvm_v2f64_ty], [IntrNoMem]>;
def int_aarch64_neon_vacgtq :
  Intrinsic<[llvm_v2i64_ty], [llvm_v2f64_ty, llvm_v2f64_ty], [IntrNoMem]>;

// Vector saturating accumulate
def int_aarch64_neon_suqadd : Neon_2Arg_Intrinsic;
def int_aarch64_neon_usqadd : Neon_2Arg_Intrinsic;

// Vector Bitwise reverse
def int_aarch64_neon_rbit : Neon_1Arg_Intrinsic;

// Vector extract and narrow
def int_aarch64_neon_xtn : 
  Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;

// Vector floating-point convert
def int_aarch64_neon_frintn : Neon_1Arg_Intrinsic;
def int_aarch64_neon_fsqrt : Neon_1Arg_Intrinsic;
def int_aarch64_neon_fcvtxn :
  Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;
def int_aarch64_neon_fcvtns : 
  Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;
def int_aarch64_neon_fcvtnu :
  Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;
def int_aarch64_neon_fcvtps :
  Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;
def int_aarch64_neon_fcvtpu :
  Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;
def int_aarch64_neon_fcvtms :
  Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;
def int_aarch64_neon_fcvtmu :
  Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;
def int_aarch64_neon_fcvtas :
  Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;
def int_aarch64_neon_fcvtau :
  Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;
def int_aarch64_neon_fcvtzs :
  Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;
def int_aarch64_neon_fcvtzu :
  Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;

// Vector maxNum (Floating Point)
def int_aarch64_neon_vmaxnm : Neon_2Arg_Intrinsic;

// Vector minNum (Floating Point)
def int_aarch64_neon_vminnm : Neon_2Arg_Intrinsic;

// Vector Pairwise maxNum (Floating Point)
def int_aarch64_neon_vpmaxnm : Neon_2Arg_Intrinsic;

// Vector Pairwise minNum (Floating Point)
def int_aarch64_neon_vpminnm : Neon_2Arg_Intrinsic;

// Vector Multiply Extended and Scalar Multiply Extended (Floating Point)
def int_aarch64_neon_vmulx  :
  Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, LLVMMatchType<0>]>;

class Neon_N2V_Intrinsic
  : Intrinsic<[llvm_anyvector_ty], [LLVMMatchType<0>, llvm_i32_ty],
              [IntrNoMem]>;
class Neon_N3V_Intrinsic
  : Intrinsic<[llvm_anyvector_ty],
              [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty],
              [IntrNoMem]>;
class Neon_N2V_Narrow_Intrinsic
  : Intrinsic<[llvm_anyvector_ty],
              [LLVMExtendedElementVectorType<0>, llvm_i32_ty],
              [IntrNoMem]>;

// Vector rounding shift right by immediate (Signed)
def int_aarch64_neon_vsrshr : Neon_N2V_Intrinsic;
def int_aarch64_neon_vurshr : Neon_N2V_Intrinsic;
def int_aarch64_neon_vsqshlu : Neon_N2V_Intrinsic;

def int_aarch64_neon_vsri : Neon_N3V_Intrinsic;
def int_aarch64_neon_vsli : Neon_N3V_Intrinsic;

def int_aarch64_neon_vsqshrun : Neon_N2V_Narrow_Intrinsic;
def int_aarch64_neon_vrshrn : Neon_N2V_Narrow_Intrinsic;
def int_aarch64_neon_vsqrshrun : Neon_N2V_Narrow_Intrinsic;
def int_aarch64_neon_vsqshrn : Neon_N2V_Narrow_Intrinsic;
def int_aarch64_neon_vuqshrn : Neon_N2V_Narrow_Intrinsic;
def int_aarch64_neon_vsqrshrn : Neon_N2V_Narrow_Intrinsic;
def int_aarch64_neon_vuqrshrn : Neon_N2V_Narrow_Intrinsic;

// Vector across
class Neon_Across_Intrinsic
  : Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty], [IntrNoMem]>;

class Neon_2Arg_Across_Float_Intrinsic
  : Intrinsic<[llvm_anyvector_ty], [llvm_v4f32_ty], [IntrNoMem]>;

def int_aarch64_neon_saddlv : Neon_Across_Intrinsic;
def int_aarch64_neon_uaddlv : Neon_Across_Intrinsic;
def int_aarch64_neon_smaxv  : Neon_Across_Intrinsic;
def int_aarch64_neon_umaxv  : Neon_Across_Intrinsic;
def int_aarch64_neon_sminv  : Neon_Across_Intrinsic;
def int_aarch64_neon_uminv  : Neon_Across_Intrinsic;
def int_aarch64_neon_vaddv  : Neon_Across_Intrinsic;
def int_aarch64_neon_vmaxv  : Neon_Across_Intrinsic;
def int_aarch64_neon_vminv  : Neon_Across_Intrinsic;
def int_aarch64_neon_vmaxnmv : Neon_Across_Intrinsic;
def int_aarch64_neon_vminnmv : Neon_Across_Intrinsic;

// Vector Table Lookup.
def int_aarch64_neon_vtbl1 :
  Intrinsic<[llvm_anyvector_ty],
            [llvm_anyvector_ty, LLVMMatchType<0>], [IntrNoMem]>;

def int_aarch64_neon_vtbl2 :
  Intrinsic<[llvm_anyvector_ty],
            [llvm_anyvector_ty, LLVMMatchType<1>, LLVMMatchType<0>],
            [IntrNoMem]>;

def int_aarch64_neon_vtbl3 :
  Intrinsic<[llvm_anyvector_ty],
            [llvm_anyvector_ty, LLVMMatchType<1>, LLVMMatchType<1>,
            LLVMMatchType<0>], [IntrNoMem]>;

def int_aarch64_neon_vtbl4 :
  Intrinsic<[llvm_anyvector_ty],
            [llvm_anyvector_ty, LLVMMatchType<1>, LLVMMatchType<1>,
            LLVMMatchType<1>, LLVMMatchType<0>], [IntrNoMem]>;

// Vector Table Extension.
// Some elements of the destination vector may not be updated, so the original
// value of that vector is passed as the first argument.  The next 1-4
// arguments after that are the table.
def int_aarch64_neon_vtbx1 :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>],
            [IntrNoMem]>;

def int_aarch64_neon_vtbx2 :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<1>,
            LLVMMatchType<0>], [IntrNoMem]>;

def int_aarch64_neon_vtbx3 :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<1>,
            LLVMMatchType<1>, LLVMMatchType<0>], [IntrNoMem]>;

def int_aarch64_neon_vtbx4 :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>, llvm_anyvector_ty,  LLVMMatchType<1>,
            LLVMMatchType<1>,  LLVMMatchType<1>, LLVMMatchType<0>],
            [IntrNoMem]>;

// Vector Load/store
def int_aarch64_neon_vld1x2 : Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>],
                                        [llvm_ptr_ty, llvm_i32_ty],
                                        [IntrReadArgMem]>;
def int_aarch64_neon_vld1x3 : Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                                         LLVMMatchType<0>],
                                        [llvm_ptr_ty, llvm_i32_ty],
                                        [IntrReadArgMem]>;
def int_aarch64_neon_vld1x4 : Intrinsic<[llvm_anyvector_ty, LLVMMatchType<0>,
                                         LLVMMatchType<0>, LLVMMatchType<0>],
                                        [llvm_ptr_ty, llvm_i32_ty],
                                        [IntrReadArgMem]>;

def int_aarch64_neon_vst1x2 : Intrinsic<[],
                                        [llvm_ptr_ty, llvm_anyvector_ty,
                                         LLVMMatchType<0>, llvm_i32_ty],
                                        [IntrReadWriteArgMem]>;
def int_aarch64_neon_vst1x3 : Intrinsic<[],
                                        [llvm_ptr_ty, llvm_anyvector_ty,
                                         LLVMMatchType<0>, LLVMMatchType<0>,
                                         llvm_i32_ty], [IntrReadWriteArgMem]>;
def int_aarch64_neon_vst1x4 : Intrinsic<[],
                                        [llvm_ptr_ty, llvm_anyvector_ty,
                                         LLVMMatchType<0>, LLVMMatchType<0>,
                                         LLVMMatchType<0>, llvm_i32_ty],
                                        [IntrReadWriteArgMem]>;

// Scalar Add
def int_aarch64_neon_vaddds :
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;
def int_aarch64_neon_vadddu :
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;


// Scalar Sub
def int_aarch64_neon_vsubds :
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;
def int_aarch64_neon_vsubdu :
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;


// Scalar Shift
// Scalar Shift Left
def int_aarch64_neon_vshlds :
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;
def int_aarch64_neon_vshldu :
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

// Scalar Saturating Shift Left
def int_aarch64_neon_vqshls : Neon_2Arg_Intrinsic;
def int_aarch64_neon_vqshlu : Neon_2Arg_Intrinsic;

// Scalar Shift Rouding Left
def int_aarch64_neon_vrshlds :
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;
def int_aarch64_neon_vrshldu :
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

// Scalar Saturating Rounding Shift Left
def int_aarch64_neon_vqrshls : Neon_2Arg_Intrinsic;
def int_aarch64_neon_vqrshlu : Neon_2Arg_Intrinsic;

// Scalar Reduce Pairwise Add.
def int_aarch64_neon_vpadd :
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i64_ty],[IntrNoMem]>;
def int_aarch64_neon_vpfadd :
  Intrinsic<[llvm_v1f32_ty], [llvm_v2f32_ty], [IntrNoMem]>;
def int_aarch64_neon_vpfaddq :
  Intrinsic<[llvm_v1f64_ty], [llvm_v2f64_ty], [IntrNoMem]>;

// Scalar Reduce Pairwise Floating Point Max/Min.
def int_aarch64_neon_vpmax :
  Intrinsic<[llvm_v1f32_ty], [llvm_v2f32_ty], [IntrNoMem]>;
def int_aarch64_neon_vpmaxq :
  Intrinsic<[llvm_v1f64_ty], [llvm_v2f64_ty], [IntrNoMem]>;
def int_aarch64_neon_vpmin :
  Intrinsic<[llvm_v1f32_ty], [llvm_v2f32_ty], [IntrNoMem]>;
def int_aarch64_neon_vpminq :
  Intrinsic<[llvm_v1f64_ty], [llvm_v2f64_ty], [IntrNoMem]>;

// Scalar Reduce Pairwise Floating Point Maxnm/Minnm.
def int_aarch64_neon_vpfmaxnm :
  Intrinsic<[llvm_v1f32_ty], [llvm_v2f32_ty], [IntrNoMem]>;
def int_aarch64_neon_vpfmaxnmq :
  Intrinsic<[llvm_v1f64_ty], [llvm_v2f64_ty], [IntrNoMem]>;
def int_aarch64_neon_vpfminnm :
  Intrinsic<[llvm_v1f32_ty], [llvm_v2f32_ty], [IntrNoMem]>;
def int_aarch64_neon_vpfminnmq :
  Intrinsic<[llvm_v1f64_ty], [llvm_v2f64_ty], [IntrNoMem]>;

// Scalar Signed Integer Convert To Floating-point
def int_aarch64_neon_vcvtf32_s32 :
  Intrinsic<[llvm_float_ty], [llvm_v1i32_ty], [IntrNoMem]>;
def int_aarch64_neon_vcvtf64_s64 :
  Intrinsic<[llvm_double_ty], [llvm_v1i64_ty], [IntrNoMem]>;

// Scalar Unsigned Integer Convert To Floating-point
def int_aarch64_neon_vcvtf32_u32 :
  Intrinsic<[llvm_float_ty], [llvm_v1i32_ty], [IntrNoMem]>;
def int_aarch64_neon_vcvtf64_u64 :
  Intrinsic<[llvm_double_ty], [llvm_v1i64_ty], [IntrNoMem]>;

// Scalar Floating-point Reciprocal Exponent
def int_aarch64_neon_vrecpx : Neon_1Arg_Intrinsic;

class Neon_Cmp_Intrinsic
  : Intrinsic<[llvm_anyvector_ty], [llvm_anyvector_ty, llvm_anyvector_ty],
              [IntrNoMem]>;

// Scalar Compare Equal
def int_aarch64_neon_vceq : Neon_Cmp_Intrinsic;

// Scalar Compare Greater-Than or Equal
def int_aarch64_neon_vcge : Neon_Cmp_Intrinsic;
def int_aarch64_neon_vchs : Neon_Cmp_Intrinsic;

// Scalar Compare Less-Than or Equal
def int_aarch64_neon_vclez : Neon_Cmp_Intrinsic;

// Scalar Compare Less-Than
def int_aarch64_neon_vcltz : Neon_Cmp_Intrinsic;

// Scalar Compare Greater-Than
def int_aarch64_neon_vcgt : Neon_Cmp_Intrinsic;
def int_aarch64_neon_vchi : Neon_Cmp_Intrinsic;

// Scalar Compare Bitwise Test Bits
def int_aarch64_neon_vtstd : Neon_Cmp_Intrinsic;

// Scalar Floating-point Absolute Compare Greater Than Or Equal
def int_aarch64_neon_vcage : Neon_Cmp_Intrinsic;
 
// Scalar Floating-point Absolute Compare Greater Than
def int_aarch64_neon_vcagt : Neon_Cmp_Intrinsic;

// Scalar Signed Saturating Accumulated of Unsigned Value
def int_aarch64_neon_vuqadd : Neon_2Arg_Intrinsic;

// Scalar Unsigned Saturating Accumulated of Signed Value
def int_aarch64_neon_vsqadd : Neon_2Arg_Intrinsic;

// Scalar Absolute Value
def int_aarch64_neon_vabs :
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

// Scalar Absolute Difference
def int_aarch64_neon_vabd : Neon_2Arg_Intrinsic;

// Scalar Negate Value
def int_aarch64_neon_vneg :
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

// Signed Saturating Doubling Multiply-Add Long
def int_aarch64_neon_vqdmlal : Neon_3Arg_Long_Intrinsic;

// Signed Saturating Doubling Multiply-Subtract Long
def int_aarch64_neon_vqdmlsl : Neon_3Arg_Long_Intrinsic;

class Neon_2Arg_ShiftImm_Intrinsic
  : Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_i32_ty], [IntrNoMem]>;

class Neon_3Arg_ShiftImm_Intrinsic
  : Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_i32_ty],
              [IntrNoMem]>;

// Scalar Shift Right (Immediate)
def int_aarch64_neon_vshrds_n : Neon_2Arg_ShiftImm_Intrinsic;
def int_aarch64_neon_vshrdu_n : Neon_2Arg_ShiftImm_Intrinsic;

// Scalar Shift Right and Accumulate (Immediate)
def int_aarch64_neon_vsrads_n : Neon_3Arg_ShiftImm_Intrinsic;
def int_aarch64_neon_vsradu_n : Neon_3Arg_ShiftImm_Intrinsic;

// Scalar Rounding Shift Right and Accumulate (Immediate)
def int_aarch64_neon_vrsrads_n : Neon_3Arg_ShiftImm_Intrinsic;
def int_aarch64_neon_vrsradu_n : Neon_3Arg_ShiftImm_Intrinsic;

// Scalar Shift Left (Immediate)
def int_aarch64_neon_vshld_n : Neon_2Arg_ShiftImm_Intrinsic;

// Scalar Saturating Shift Left (Immediate)
def int_aarch64_neon_vqshls_n : Neon_N2V_Intrinsic;
def int_aarch64_neon_vqshlu_n : Neon_N2V_Intrinsic;

// Scalar Signed Saturating Shift Left Unsigned (Immediate)
def int_aarch64_neon_vqshlus_n : Neon_N2V_Intrinsic;

// Scalar Signed Fixed-point Convert To Floating-Point (Immediate)
def int_aarch64_neon_vcvtf32_n_s32 :
  Intrinsic<[llvm_float_ty], [llvm_v1i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_aarch64_neon_vcvtf64_n_s64 :
  Intrinsic<[llvm_double_ty], [llvm_v1i64_ty, llvm_i32_ty], [IntrNoMem]>;

// Scalar Unsigned Fixed-point Convert To Floating-Point (Immediate)
def int_aarch64_neon_vcvtf32_n_u32 :
  Intrinsic<[llvm_float_ty], [llvm_v1i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_aarch64_neon_vcvtf64_n_u64 :
  Intrinsic<[llvm_double_ty], [llvm_v1i64_ty, llvm_i32_ty], [IntrNoMem]>;

// Scalar Floating-point Convert To Signed Fixed-point (Immediate)
def int_aarch64_neon_vcvts_n_s32_f32 :
  Intrinsic<[llvm_v1i32_ty], [llvm_v1f32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_aarch64_neon_vcvtd_n_s64_f64 :
  Intrinsic<[llvm_v1i64_ty], [llvm_v1f64_ty, llvm_i32_ty], [IntrNoMem]>;

// Scalar Floating-point Convert To Unsigned Fixed-point (Immediate)
def int_aarch64_neon_vcvts_n_u32_f32 :
  Intrinsic<[llvm_v1i32_ty], [llvm_v1f32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_aarch64_neon_vcvtd_n_u64_f64 :
  Intrinsic<[llvm_v1i64_ty], [llvm_v1f64_ty, llvm_i32_ty], [IntrNoMem]>;

class Neon_SHA_Intrinsic
  : Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v1i32_ty, llvm_v4i32_ty],
              [IntrNoMem]>;

def int_aarch64_neon_sha1c : Neon_SHA_Intrinsic;
def int_aarch64_neon_sha1m : Neon_SHA_Intrinsic;
def int_aarch64_neon_sha1p : Neon_SHA_Intrinsic;
}
