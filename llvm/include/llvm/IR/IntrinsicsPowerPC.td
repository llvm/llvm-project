//===- IntrinsicsPowerPC.td - Defines PowerPC intrinsics ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the PowerPC-specific intrinsics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Definitions for all PowerPC intrinsics.
//

// Non-altivec intrinsics.
let TargetPrefix = "ppc" in {  // All intrinsics start with "llvm.ppc.".
  // dcba/dcbf/dcbi/dcbst/dcbt/dcbz/dcbzl(PPC970) instructions.
  def int_ppc_dcba  : Intrinsic<[], [llvm_ptr_ty], []>;
  def int_ppc_dcbf : GCCBuiltin<"__builtin_dcbf">,
                      Intrinsic<[], [llvm_ptr_ty], [IntrArgMemOnly]>;
  def int_ppc_dcbfps : Intrinsic<[], [llvm_ptr_ty], [IntrArgMemOnly]>;
  def int_ppc_dcbstps : Intrinsic<[], [llvm_ptr_ty], [IntrArgMemOnly]>;
  def int_ppc_dcbi  : Intrinsic<[], [llvm_ptr_ty], []>;
  def int_ppc_dcbt_with_hint: Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty],
    [IntrArgMemOnly, NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
  def int_ppc_dcbtst_with_hint: Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty],
    [IntrArgMemOnly, NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
  def int_ppc_dcbzl : Intrinsic<[], [llvm_ptr_ty], []>;

  // Get content from current FPSCR register
  def int_ppc_readflm : GCCBuiltin<"__builtin_readflm">,
                        Intrinsic<[llvm_double_ty], [], [IntrNoMem]>;
  // Set FPSCR register, and return previous content
  def int_ppc_setflm : GCCBuiltin<"__builtin_setflm">,
                       Intrinsic<[llvm_double_ty], [llvm_double_ty], []>;

  // Intrinsics for [double]word extended forms of divide instructions
  def int_ppc_divwe : GCCBuiltin<"__builtin_divwe">,
                      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                                [IntrNoMem]>;
  def int_ppc_divweu : GCCBuiltin<"__builtin_divweu">,
                       Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                                 [IntrNoMem]>;
  def int_ppc_divde : GCCBuiltin<"__builtin_divde">,
                      Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
                                [IntrNoMem]>;
  def int_ppc_divdeu : GCCBuiltin<"__builtin_divdeu">,
                       Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
                                 [IntrNoMem]>;

  // Generate a random number
  def int_ppc_darn : GCCBuiltin<"__builtin_darn">,
                     Intrinsic<[llvm_i64_ty], [], [IntrNoMem]>;
  def int_ppc_darnraw : GCCBuiltin<"__builtin_darn_raw">,
                     Intrinsic<[llvm_i64_ty], [], [IntrNoMem]>;
  def int_ppc_darn32 : GCCBuiltin<"__builtin_darn_32">,
                     Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

  // Bit permute doubleword
  def int_ppc_bpermd : GCCBuiltin<"__builtin_bpermd">,
                       Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
                                 [IntrNoMem]>;

  // Parallel Bits Deposit/Extract Doubleword Builtins.
  def int_ppc_pdepd
      : GCCBuiltin<"__builtin_pdepd">,
        Intrinsic <[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;
  def int_ppc_pextd
      : GCCBuiltin<"__builtin_pextd">,
        Intrinsic <[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;

  // Centrifuge Doubleword Builtin.
  def int_ppc_cfuged
      : GCCBuiltin<"__builtin_cfuged">,
        Intrinsic <[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;

  // Count Leading / Trailing Zeroes under bit Mask Builtins.
  def int_ppc_cntlzdm
      : GCCBuiltin<"__builtin_cntlzdm">,
        Intrinsic <[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;
  def int_ppc_cnttzdm
      : GCCBuiltin<"__builtin_cnttzdm">,
        Intrinsic <[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;

  def int_ppc_truncf128_round_to_odd
      : GCCBuiltin<"__builtin_truncf128_round_to_odd">,
        Intrinsic <[llvm_double_ty], [llvm_f128_ty], [IntrNoMem]>;
  def int_ppc_sqrtf128_round_to_odd
      : GCCBuiltin<"__builtin_sqrtf128_round_to_odd">,
        Intrinsic <[llvm_f128_ty], [llvm_f128_ty], [IntrNoMem]>;
  def int_ppc_addf128_round_to_odd
      : GCCBuiltin<"__builtin_addf128_round_to_odd">,
        Intrinsic <[llvm_f128_ty], [llvm_f128_ty,llvm_f128_ty], [IntrNoMem]>;
  def int_ppc_subf128_round_to_odd
      : GCCBuiltin<"__builtin_subf128_round_to_odd">,
        Intrinsic <[llvm_f128_ty], [llvm_f128_ty,llvm_f128_ty], [IntrNoMem]>;
  def int_ppc_mulf128_round_to_odd
      : GCCBuiltin<"__builtin_mulf128_round_to_odd">,
        Intrinsic <[llvm_f128_ty], [llvm_f128_ty,llvm_f128_ty], [IntrNoMem]>;
  def int_ppc_divf128_round_to_odd
      : GCCBuiltin<"__builtin_divf128_round_to_odd">,
        Intrinsic <[llvm_f128_ty], [llvm_f128_ty,llvm_f128_ty], [IntrNoMem]>;
  def int_ppc_fmaf128_round_to_odd
      : GCCBuiltin<"__builtin_fmaf128_round_to_odd">,
        Intrinsic <[llvm_f128_ty], [llvm_f128_ty,llvm_f128_ty,llvm_f128_ty], [IntrNoMem]>;
  def int_ppc_scalar_extract_expq
      : GCCBuiltin<"__builtin_vsx_scalar_extract_expq">,
        Intrinsic <[llvm_i64_ty], [llvm_f128_ty], [IntrNoMem]>;
  def int_ppc_scalar_insert_exp_qp
      : GCCBuiltin<"__builtin_vsx_scalar_insert_exp_qp">,
        Intrinsic <[llvm_f128_ty], [llvm_f128_ty, llvm_i64_ty], [IntrNoMem]>;

}


let TargetPrefix = "ppc" in {  // All PPC intrinsics start with "llvm.ppc.".
  /// PowerPC_Vec_Intrinsic - Base class for all altivec intrinsics.
  class PowerPC_Vec_Intrinsic<string GCCIntSuffix, list<LLVMType> ret_types,
                              list<LLVMType> param_types,
                              list<IntrinsicProperty> properties>
    : GCCBuiltin<!strconcat("__builtin_altivec_", GCCIntSuffix)>,
      Intrinsic<ret_types, param_types, properties>;

  /// PowerPC_VSX_Intrinsic - Base class for all VSX intrinsics.
  class PowerPC_VSX_Intrinsic<string GCCIntSuffix, list<LLVMType> ret_types,
                              list<LLVMType> param_types,
                              list<IntrinsicProperty> properties>
    : GCCBuiltin<!strconcat("__builtin_vsx_", GCCIntSuffix)>,
      Intrinsic<ret_types, param_types, properties>;
}

//===----------------------------------------------------------------------===//
// PowerPC MMA Intrinsic Multi Class Definitions.
//

multiclass PowerPC_MMA_ACC_Intrinsic<list<LLVMType> args> {
  def NAME: Intrinsic<[llvm_v512i1_ty], args, [IntrNoMem]>;
  def pp : Intrinsic<[llvm_v512i1_ty], !listconcat([llvm_v512i1_ty], args),
                     [IntrNoMem]>;
  def pn : Intrinsic<[llvm_v512i1_ty], !listconcat([llvm_v512i1_ty], args),
                     [IntrNoMem]>;
  def np : Intrinsic<[llvm_v512i1_ty], !listconcat([llvm_v512i1_ty], args),
                     [IntrNoMem]>;
  def nn : Intrinsic<[llvm_v512i1_ty], !listconcat([llvm_v512i1_ty], args),
                     [IntrNoMem]>;
}

multiclass PowerPC_MMA_ACC_PP_Intrinsic<list<LLVMType> args> {
  def NAME: Intrinsic<[llvm_v512i1_ty], args, [IntrNoMem]>;
  def pp : Intrinsic<[llvm_v512i1_ty], !listconcat([llvm_v512i1_ty], args),
                     [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// PowerPC Altivec Intrinsic Class Definitions.
//

/// PowerPC_Vec_FF_Intrinsic - A PowerPC intrinsic that takes one v4f32
/// vector and returns one.  These intrinsics have no side effects.
class PowerPC_Vec_FF_Intrinsic<string GCCIntSuffix>
  : PowerPC_Vec_Intrinsic<GCCIntSuffix,
                          [llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;

/// PowerPC_Vec_FFF_Intrinsic - A PowerPC intrinsic that takes two v4f32
/// vectors and returns one.  These intrinsics have no side effects.
class PowerPC_Vec_FFF_Intrinsic<string GCCIntSuffix>
  : PowerPC_Vec_Intrinsic<GCCIntSuffix,
                          [llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty],
                          [IntrNoMem]>;

/// PowerPC_Vec_BBB_Intrinsic - A PowerPC intrinsic that takes two v16i8
/// vectors and returns one.  These intrinsics have no side effects.
class PowerPC_Vec_BBB_Intrinsic<string GCCIntSuffix>
  : PowerPC_Vec_Intrinsic<GCCIntSuffix,
                          [llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                          [IntrNoMem]>;

/// PowerPC_Vec_HHH_Intrinsic - A PowerPC intrinsic that takes two v8i16
/// vectors and returns one.  These intrinsics have no side effects.
class PowerPC_Vec_HHH_Intrinsic<string GCCIntSuffix>
  : PowerPC_Vec_Intrinsic<GCCIntSuffix,
                          [llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                          [IntrNoMem]>;

/// PowerPC_Vec_WWW_Intrinsic - A PowerPC intrinsic that takes two v4i32
/// vectors and returns one.  These intrinsics have no side effects.
class PowerPC_Vec_WWW_Intrinsic<string GCCIntSuffix>
  : PowerPC_Vec_Intrinsic<GCCIntSuffix,
                          [llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                          [IntrNoMem]>;

/// PowerPC_Vec_DDD_Intrinsic - A PowerPC intrinsic that takes two v2i64
/// vectors and returns one.  These intrinsics have no side effects.
class PowerPC_Vec_DDD_Intrinsic<string GCCIntSuffix>
  : PowerPC_Vec_Intrinsic<GCCIntSuffix,
                          [llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                          [IntrNoMem]>;

/// PowerPC_Vec_QQQ_Intrinsic - A PowerPC intrinsic that takes two v1i128
/// vectors and returns one. These intrinsics have no side effects.
class PowerPC_Vec_QQQ_Intrinsic<string GCCIntSuffix>
  : PowerPC_Vec_Intrinsic<GCCIntSuffix,
                         [llvm_v1i128_ty], [llvm_v1i128_ty, llvm_v1i128_ty],
                         [IntrNoMem]>;

/// PowerPC_Vec_QDD_Intrinsic - A PowerPC intrinsic that takes two v2i64
/// vectors and returns one v1i128. These intrinsics have no side effects.
class PowerPC_Vec_QDD_Intrinsic<string GCCIntSuffix>
  : PowerPC_Vec_Intrinsic<GCCIntSuffix,
                          [llvm_v1i128_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                          [IntrNoMem]>;

//===----------------------------------------------------------------------===//
// PowerPC VSX Intrinsic Class Definitions.
//

/// PowerPC_VSX_Vec_DDD_Intrinsic - A PowerPC intrinsic that takes two v2f64
/// vectors and returns one.  These intrinsics have no side effects.
class PowerPC_VSX_Vec_DDD_Intrinsic<string GCCIntSuffix>
  : PowerPC_VSX_Intrinsic<GCCIntSuffix,
                          [llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty],
                          [IntrNoMem]>;

/// PowerPC_VSX_Vec_FFF_Intrinsic - A PowerPC intrinsic that takes two v4f32
/// vectors and returns one.  These intrinsics have no side effects.
class PowerPC_VSX_Vec_FFF_Intrinsic<string GCCIntSuffix>
  : PowerPC_VSX_Intrinsic<GCCIntSuffix,
                          [llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty],
                          [IntrNoMem]>;

/// PowerPC_VSX_Sca_DDD_Intrinsic - A PowerPC intrinsic that takes two f64
/// scalars and returns one.  These intrinsics have no side effects.
class PowerPC_VSX_Sca_DDD_Intrinsic<string GCCIntSuffix>
  : PowerPC_VSX_Intrinsic<GCCIntSuffix,
                          [llvm_double_ty], [llvm_double_ty, llvm_double_ty],
                          [IntrNoMem]>;

//===----------------------------------------------------------------------===//
// PowerPC Altivec Intrinsic Definitions.

let TargetPrefix = "ppc" in {  // All intrinsics start with "llvm.ppc.".
  // Data Stream Control.
  def int_ppc_altivec_dss : GCCBuiltin<"__builtin_altivec_dss">,
              Intrinsic<[], [llvm_i32_ty], []>;
  def int_ppc_altivec_dssall : GCCBuiltin<"__builtin_altivec_dssall">,
              Intrinsic<[], [], []>;
  def int_ppc_altivec_dst : GCCBuiltin<"__builtin_altivec_dst">,
              Intrinsic<[],
                        [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
                        []>;
  def int_ppc_altivec_dstt : GCCBuiltin<"__builtin_altivec_dstt">,
              Intrinsic<[],
                        [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
                        []>;
  def int_ppc_altivec_dstst : GCCBuiltin<"__builtin_altivec_dstst">,
              Intrinsic<[],
                        [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
                        []>;
  def int_ppc_altivec_dststt : GCCBuiltin<"__builtin_altivec_dststt">,
              Intrinsic<[],
                        [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
                        []>;

  // VSCR access.
  def int_ppc_altivec_mfvscr : GCCBuiltin<"__builtin_altivec_mfvscr">,
              Intrinsic<[llvm_v8i16_ty], [], [IntrNoMem, IntrHasSideEffects]>;
  def int_ppc_altivec_mtvscr : GCCBuiltin<"__builtin_altivec_mtvscr">,
              Intrinsic<[], [llvm_v4i32_ty], [IntrNoMem, IntrHasSideEffects]>;


  // Loads.  These don't map directly to GCC builtins because they represent the
  // source address with a single pointer.
  def int_ppc_altivec_lvx :
              Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], [IntrReadMem, IntrArgMemOnly]>;
  def int_ppc_altivec_lvxl :
              Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], [IntrReadMem, IntrArgMemOnly]>;
  def int_ppc_altivec_lvebx :
              Intrinsic<[llvm_v16i8_ty], [llvm_ptr_ty], [IntrReadMem, IntrArgMemOnly]>;
  def int_ppc_altivec_lvehx :
              Intrinsic<[llvm_v8i16_ty], [llvm_ptr_ty], [IntrReadMem, IntrArgMemOnly]>;
  def int_ppc_altivec_lvewx :
              Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], [IntrReadMem, IntrArgMemOnly]>;

  // Stores.  These don't map directly to GCC builtins because they represent the
  // source address with a single pointer.
  def int_ppc_altivec_stvx :
              Intrinsic<[], [llvm_v4i32_ty, llvm_ptr_ty],
                        [IntrWriteMem, IntrArgMemOnly]>;
  def int_ppc_altivec_stvxl :
              Intrinsic<[], [llvm_v4i32_ty, llvm_ptr_ty],
                        [IntrWriteMem, IntrArgMemOnly]>;
  def int_ppc_altivec_stvebx :
              Intrinsic<[], [llvm_v16i8_ty, llvm_ptr_ty],
                        [IntrWriteMem, IntrArgMemOnly]>;
  def int_ppc_altivec_stvehx :
              Intrinsic<[], [llvm_v8i16_ty, llvm_ptr_ty],
                        [IntrWriteMem, IntrArgMemOnly]>;
  def int_ppc_altivec_stvewx :
              Intrinsic<[], [llvm_v4i32_ty, llvm_ptr_ty],
                        [IntrWriteMem, IntrArgMemOnly]>;

  // Comparisons setting a vector.
  def int_ppc_altivec_vcmpbfp : GCCBuiltin<"__builtin_altivec_vcmpbfp">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpeqfp : GCCBuiltin<"__builtin_altivec_vcmpeqfp">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgefp : GCCBuiltin<"__builtin_altivec_vcmpgefp">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtfp : GCCBuiltin<"__builtin_altivec_vcmpgtfp">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;

  def int_ppc_altivec_vcmpequd : GCCBuiltin<"__builtin_altivec_vcmpequd">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtsd : GCCBuiltin<"__builtin_altivec_vcmpgtsd">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtud : GCCBuiltin<"__builtin_altivec_vcmpgtud">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;

  def int_ppc_altivec_vcmpequw : GCCBuiltin<"__builtin_altivec_vcmpequw">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtsw : GCCBuiltin<"__builtin_altivec_vcmpgtsw">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtuw : GCCBuiltin<"__builtin_altivec_vcmpgtuw">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpnew : GCCBuiltin<"__builtin_altivec_vcmpnew">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpnezw : GCCBuiltin<"__builtin_altivec_vcmpnezw">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;

  def int_ppc_altivec_vcmpequh : GCCBuiltin<"__builtin_altivec_vcmpequh">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtsh : GCCBuiltin<"__builtin_altivec_vcmpgtsh">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtuh : GCCBuiltin<"__builtin_altivec_vcmpgtuh">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpneh : GCCBuiltin<"__builtin_altivec_vcmpneh">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpnezh : GCCBuiltin<"__builtin_altivec_vcmpnezh">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                        [IntrNoMem]>;

  def int_ppc_altivec_vcmpequb : GCCBuiltin<"__builtin_altivec_vcmpequb">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtsb : GCCBuiltin<"__builtin_altivec_vcmpgtsb">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtub : GCCBuiltin<"__builtin_altivec_vcmpgtub">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpneb : GCCBuiltin<"__builtin_altivec_vcmpneb">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpnezb : GCCBuiltin<"__builtin_altivec_vcmpnezb">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                        [IntrNoMem]>;

  def int_ppc_altivec_vcmpequq : GCCBuiltin<"__builtin_altivec_vcmpequq">,
              Intrinsic<[llvm_v1i128_ty], [llvm_v1i128_ty, llvm_v1i128_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtsq : GCCBuiltin<"__builtin_altivec_vcmpgtsq">,
              Intrinsic<[llvm_v1i128_ty], [llvm_v1i128_ty, llvm_v1i128_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtuq : GCCBuiltin<"__builtin_altivec_vcmpgtuq">,
              Intrinsic<[llvm_v1i128_ty], [llvm_v1i128_ty, llvm_v1i128_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpequq_p : GCCBuiltin<"__builtin_altivec_vcmpequq_p">,
              Intrinsic<[llvm_i32_ty],
                        [llvm_i32_ty,llvm_v1i128_ty,llvm_v1i128_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtsq_p : GCCBuiltin<"__builtin_altivec_vcmpgtsq_p">,
              Intrinsic<[llvm_i32_ty],
                        [llvm_i32_ty,llvm_v1i128_ty,llvm_v1i128_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtuq_p : GCCBuiltin<"__builtin_altivec_vcmpgtuq_p">,
              Intrinsic<[llvm_i32_ty],
                        [llvm_i32_ty,llvm_v1i128_ty,llvm_v1i128_ty],
                        [IntrNoMem]>;

  // Predicate Comparisons.  The first operand specifies interpretation of CR6.
  def int_ppc_altivec_vcmpbfp_p : GCCBuiltin<"__builtin_altivec_vcmpbfp_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v4f32_ty,llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpeqfp_p : GCCBuiltin<"__builtin_altivec_vcmpeqfp_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v4f32_ty,llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgefp_p : GCCBuiltin<"__builtin_altivec_vcmpgefp_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v4f32_ty,llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtfp_p : GCCBuiltin<"__builtin_altivec_vcmpgtfp_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v4f32_ty,llvm_v4f32_ty],
                        [IntrNoMem]>;

  def int_ppc_altivec_vcmpequd_p : GCCBuiltin<"__builtin_altivec_vcmpequd_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v2i64_ty,llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtsd_p : GCCBuiltin<"__builtin_altivec_vcmpgtsd_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v2i64_ty,llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtud_p : GCCBuiltin<"__builtin_altivec_vcmpgtud_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v2i64_ty,llvm_v2i64_ty],
                        [IntrNoMem]>;

  def int_ppc_altivec_vcmpequw_p : GCCBuiltin<"__builtin_altivec_vcmpequw_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v4i32_ty,llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtsw_p : GCCBuiltin<"__builtin_altivec_vcmpgtsw_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v4i32_ty,llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtuw_p : GCCBuiltin<"__builtin_altivec_vcmpgtuw_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v4i32_ty,llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpnew_p : GCCBuiltin<"__builtin_altivec_vcmpnew_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v4i32_ty,llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpnezw_p : GCCBuiltin<"__builtin_altivec_vcmpnezw_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v4i32_ty,llvm_v4i32_ty],
                        [IntrNoMem]>;

  def int_ppc_altivec_vcmpequh_p : GCCBuiltin<"__builtin_altivec_vcmpequh_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v8i16_ty,llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtsh_p : GCCBuiltin<"__builtin_altivec_vcmpgtsh_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v8i16_ty,llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtuh_p : GCCBuiltin<"__builtin_altivec_vcmpgtuh_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v8i16_ty,llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpneh_p : GCCBuiltin<"__builtin_altivec_vcmpneh_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v8i16_ty,llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpnezh_p : GCCBuiltin<"__builtin_altivec_vcmpnezh_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v8i16_ty,llvm_v8i16_ty],
                        [IntrNoMem]>;

  def int_ppc_altivec_vcmpequb_p : GCCBuiltin<"__builtin_altivec_vcmpequb_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v16i8_ty,llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtsb_p : GCCBuiltin<"__builtin_altivec_vcmpgtsb_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v16i8_ty,llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtub_p : GCCBuiltin<"__builtin_altivec_vcmpgtub_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v16i8_ty,llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpneb_p : GCCBuiltin<"__builtin_altivec_vcmpneb_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v16i8_ty,llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpnezb_p : GCCBuiltin<"__builtin_altivec_vcmpnezb_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v16i8_ty,llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vclzlsbb : GCCBuiltin<"__builtin_altivec_vclzlsbb">,
              Intrinsic<[llvm_i32_ty],[llvm_v16i8_ty],[IntrNoMem]>;
  def int_ppc_altivec_vctzlsbb : GCCBuiltin<"__builtin_altivec_vctzlsbb">,
              Intrinsic<[llvm_i32_ty],[llvm_v16i8_ty],[IntrNoMem]>;
  def int_ppc_altivec_vprtybw : GCCBuiltin<"__builtin_altivec_vprtybw">,
              Intrinsic<[llvm_v4i32_ty],[llvm_v4i32_ty],[IntrNoMem]>;
  def int_ppc_altivec_vprtybd : GCCBuiltin<"__builtin_altivec_vprtybd">,
              Intrinsic<[llvm_v2i64_ty],[llvm_v2i64_ty],[IntrNoMem]>;
  def int_ppc_altivec_vprtybq : GCCBuiltin<"__builtin_altivec_vprtybq">,
              Intrinsic<[llvm_v1i128_ty],[llvm_v1i128_ty],[IntrNoMem]>;

  // P10 Vector Extract with Mask
  def int_ppc_altivec_vextractbm : GCCBuiltin<"__builtin_altivec_vextractbm">,
              Intrinsic<[llvm_i32_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_ppc_altivec_vextracthm : GCCBuiltin<"__builtin_altivec_vextracthm">,
              Intrinsic<[llvm_i32_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_ppc_altivec_vextractwm : GCCBuiltin<"__builtin_altivec_vextractwm">,
              Intrinsic<[llvm_i32_ty], [llvm_v4i32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vextractdm : GCCBuiltin<"__builtin_altivec_vextractdm">,
              Intrinsic<[llvm_i32_ty], [llvm_v2i64_ty], [IntrNoMem]>;
  def int_ppc_altivec_vextractqm : GCCBuiltin<"__builtin_altivec_vextractqm">,
              Intrinsic<[llvm_i32_ty], [llvm_v1i128_ty], [IntrNoMem]>;

  // P10 Vector Expand with Mask
  def int_ppc_altivec_vexpandbm : GCCBuiltin<"__builtin_altivec_vexpandbm">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_ppc_altivec_vexpandhm : GCCBuiltin<"__builtin_altivec_vexpandhm">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_ppc_altivec_vexpandwm : GCCBuiltin<"__builtin_altivec_vexpandwm">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vexpanddm : GCCBuiltin<"__builtin_altivec_vexpanddm">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty], [IntrNoMem]>;
  def int_ppc_altivec_vexpandqm : GCCBuiltin<"__builtin_altivec_vexpandqm">,
              Intrinsic<[llvm_v1i128_ty], [llvm_v1i128_ty], [IntrNoMem]>;

  // P10 Vector Count with Mask intrinsics.
  def int_ppc_altivec_vcntmbb : GCCBuiltin<"__builtin_altivec_vcntmbb">,
              Intrinsic<[llvm_i64_ty], [llvm_v16i8_ty, llvm_i32_ty],
                        [IntrNoMem, ImmArg<ArgIndex<1>>]>;
  def int_ppc_altivec_vcntmbh : GCCBuiltin<"__builtin_altivec_vcntmbh">,
              Intrinsic<[llvm_i64_ty], [llvm_v8i16_ty, llvm_i32_ty],
                        [IntrNoMem, ImmArg<ArgIndex<1>>]>;
  def int_ppc_altivec_vcntmbw : GCCBuiltin<"__builtin_altivec_vcntmbw">,
              Intrinsic<[llvm_i64_ty], [llvm_v4i32_ty, llvm_i32_ty],
                        [IntrNoMem, ImmArg<ArgIndex<1>>]>;
  def int_ppc_altivec_vcntmbd : GCCBuiltin<"__builtin_altivec_vcntmbd">,
              Intrinsic<[llvm_i64_ty], [llvm_v2i64_ty, llvm_i32_ty],
                        [IntrNoMem, ImmArg<ArgIndex<1>>]>;

  // P10 Move to VSR with Mask Intrinsics.
  def int_ppc_altivec_mtvsrbm : GCCBuiltin<"__builtin_altivec_mtvsrbm">,
              Intrinsic<[llvm_v16i8_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_ppc_altivec_mtvsrhm : GCCBuiltin<"__builtin_altivec_mtvsrhm">,
              Intrinsic<[llvm_v8i16_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_ppc_altivec_mtvsrwm : GCCBuiltin<"__builtin_altivec_mtvsrwm">,
              Intrinsic<[llvm_v4i32_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_ppc_altivec_mtvsrdm : GCCBuiltin<"__builtin_altivec_mtvsrdm">,
              Intrinsic<[llvm_v2i64_ty], [llvm_i64_ty], [IntrNoMem]>;
  def int_ppc_altivec_mtvsrqm : GCCBuiltin<"__builtin_altivec_mtvsrqm">,
              Intrinsic<[llvm_v1i128_ty], [llvm_i64_ty], [IntrNoMem]>;

  // P10 Vector Parallel Bits Deposit/Extract Doubleword Builtins.
  def int_ppc_altivec_vpdepd : GCCBuiltin<"__builtin_altivec_vpdepd">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vpextd : GCCBuiltin<"__builtin_altivec_vpextd">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;

  // P10 Vector String Isolate Intrinsics.
  def int_ppc_altivec_vstribr : GCCBuiltin<"__builtin_altivec_vstribr">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_ppc_altivec_vstribl : GCCBuiltin<"__builtin_altivec_vstribl">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_ppc_altivec_vstrihr : GCCBuiltin<"__builtin_altivec_vstrihr">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_ppc_altivec_vstrihl : GCCBuiltin<"__builtin_altivec_vstrihl">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  // Predicate Intrinsics: The first operand specifies interpretation of CR6.
  def int_ppc_altivec_vstribr_p : GCCBuiltin<"__builtin_altivec_vstribr_p">,
              Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_v16i8_ty], [IntrNoMem]>;
  def int_ppc_altivec_vstribl_p : GCCBuiltin<"__builtin_altivec_vstribl_p">,
              Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_v16i8_ty], [IntrNoMem]>;
  def int_ppc_altivec_vstrihr_p : GCCBuiltin<"__builtin_altivec_vstrihr_p">,
              Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_v8i16_ty], [IntrNoMem]>;
  def int_ppc_altivec_vstrihl_p : GCCBuiltin<"__builtin_altivec_vstrihl_p">,
              Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_v8i16_ty], [IntrNoMem]>;

  // P10 Vector Centrifuge Builtin.
  def int_ppc_altivec_vcfuged : GCCBuiltin<"__builtin_altivec_vcfuged">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;

  // P10 Vector Gather Every Nth Bit Builtin.
  def int_ppc_altivec_vgnb : GCCBuiltin<"__builtin_altivec_vgnb">,
              Intrinsic<[llvm_i64_ty], [llvm_v1i128_ty, llvm_i32_ty],
                        [IntrNoMem, ImmArg<ArgIndex<1>>]>;

   // P10 Vector Clear Bytes
   def int_ppc_altivec_vclrlb :  GCCBuiltin<"__builtin_altivec_vclrlb">,
               Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_i32_ty],
                         [IntrNoMem]>;
   def int_ppc_altivec_vclrrb :  GCCBuiltin<"__builtin_altivec_vclrrb">,
               Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_i32_ty],
                         [IntrNoMem]>;

  // P10 Vector Shift Double Bit Immediate.
  def int_ppc_altivec_vsldbi : GCCBuiltin<"__builtin_altivec_vsldbi">,
              Intrinsic<[llvm_v16i8_ty],
                        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty],
                        [IntrNoMem, ImmArg<ArgIndex<2>>]>;
  def int_ppc_altivec_vsrdbi : GCCBuiltin<"__builtin_altivec_vsrdbi">,
              Intrinsic<[llvm_v16i8_ty],
                        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty],
                        [IntrNoMem, ImmArg<ArgIndex<2>>]>;

  // P10 Vector Insert.
  def int_ppc_altivec_vinsblx : GCCBuiltin<"__builtin_altivec_vinsblx">,
              Intrinsic<[llvm_v16i8_ty],
                        [llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vinsbrx : GCCBuiltin<"__builtin_altivec_vinsbrx">,
              Intrinsic<[llvm_v16i8_ty],
                        [llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vinshlx : GCCBuiltin<"__builtin_altivec_vinshlx">,
              Intrinsic<[llvm_v8i16_ty],
                        [llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vinshrx : GCCBuiltin<"__builtin_altivec_vinshrx">,
              Intrinsic<[llvm_v8i16_ty],
                        [llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vinswlx : GCCBuiltin<"__builtin_altivec_vinswlx">,
              Intrinsic<[llvm_v4i32_ty],
                        [llvm_v4i32_ty, llvm_i32_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vinswrx : GCCBuiltin<"__builtin_altivec_vinswrx">,
              Intrinsic<[llvm_v4i32_ty],
                        [llvm_v4i32_ty, llvm_i32_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vinsdlx : GCCBuiltin<"__builtin_altivec_vinsdlx">,
              Intrinsic<[llvm_v2i64_ty],
                        [llvm_v2i64_ty, llvm_i64_ty, llvm_i64_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vinsdrx : GCCBuiltin<"__builtin_altivec_vinsdrx">,
              Intrinsic<[llvm_v2i64_ty],
                        [llvm_v2i64_ty, llvm_i64_ty, llvm_i64_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vinsbvlx : GCCBuiltin<"__builtin_altivec_vinsbvlx">,
              Intrinsic<[llvm_v16i8_ty],
                        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vinsbvrx : GCCBuiltin<"__builtin_altivec_vinsbvrx">,
              Intrinsic<[llvm_v16i8_ty],
                        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vinshvlx : GCCBuiltin<"__builtin_altivec_vinshvlx">,
              Intrinsic<[llvm_v8i16_ty],
                        [llvm_v8i16_ty, llvm_i32_ty, llvm_v8i16_ty],
                        [IntrNoMem]>;
 def int_ppc_altivec_vinshvrx : GCCBuiltin<"__builtin_altivec_vinshvrx">,
              Intrinsic<[llvm_v8i16_ty],
                        [llvm_v8i16_ty, llvm_i32_ty, llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vinswvlx : GCCBuiltin<"__builtin_altivec_vinswvlx">,
              Intrinsic<[llvm_v4i32_ty],
                        [llvm_v4i32_ty, llvm_i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vinswvrx : GCCBuiltin<"__builtin_altivec_vinswvrx">,
              Intrinsic<[llvm_v4i32_ty],
                        [llvm_v4i32_ty, llvm_i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  // P10 Vector Insert with immediate.
  def int_ppc_altivec_vinsw :
              Intrinsic<[llvm_v4i32_ty],
                        [llvm_v4i32_ty, llvm_i32_ty, llvm_i32_ty],
                        [IntrNoMem, ImmArg<ArgIndex<2>>]>;
  def int_ppc_altivec_vinsd :
              Intrinsic<[llvm_v2i64_ty],
                        [llvm_v2i64_ty, llvm_i64_ty, llvm_i32_ty],
                        [IntrNoMem, ImmArg<ArgIndex<2>>]>;
  // P10 Vector Extract.
  def int_ppc_altivec_vextdubvlx : GCCBuiltin<"__builtin_altivec_vextdubvlx">,
              Intrinsic<[llvm_v2i64_ty],
                        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vextdubvrx : GCCBuiltin<"__builtin_altivec_vextdubvrx">,
              Intrinsic<[llvm_v2i64_ty],
                        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vextduhvlx : GCCBuiltin<"__builtin_altivec_vextduhvlx">,
              Intrinsic<[llvm_v2i64_ty],
                        [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vextduhvrx : GCCBuiltin<"__builtin_altivec_vextduhvrx">,
              Intrinsic<[llvm_v2i64_ty],
                        [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vextduwvlx : GCCBuiltin<"__builtin_altivec_vextduwvlx">,
              Intrinsic<[llvm_v2i64_ty],
                        [llvm_v4i32_ty, llvm_v4i32_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vextduwvrx : GCCBuiltin<"__builtin_altivec_vextduwvrx">,
              Intrinsic<[llvm_v2i64_ty],
                        [llvm_v4i32_ty, llvm_v4i32_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vextddvlx : GCCBuiltin<"__builtin_altivec_vextddvlx">,
              Intrinsic<[llvm_v2i64_ty],
                        [llvm_v2i64_ty, llvm_v2i64_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vextddvrx : GCCBuiltin<"__builtin_altivec_vextddvrx">,
              Intrinsic<[llvm_v2i64_ty],
                        [llvm_v2i64_ty, llvm_v2i64_ty, llvm_i32_ty],
                        [IntrNoMem]>;
}

// Vector average.
def int_ppc_altivec_vavgsb : PowerPC_Vec_BBB_Intrinsic<"vavgsb">;
def int_ppc_altivec_vavgsh : PowerPC_Vec_HHH_Intrinsic<"vavgsh">;
def int_ppc_altivec_vavgsw : PowerPC_Vec_WWW_Intrinsic<"vavgsw">;
def int_ppc_altivec_vavgub : PowerPC_Vec_BBB_Intrinsic<"vavgub">;
def int_ppc_altivec_vavguh : PowerPC_Vec_HHH_Intrinsic<"vavguh">;
def int_ppc_altivec_vavguw : PowerPC_Vec_WWW_Intrinsic<"vavguw">;

// Vector maximum.
def int_ppc_altivec_vmaxfp : PowerPC_Vec_FFF_Intrinsic<"vmaxfp">;
def int_ppc_altivec_vmaxsb : PowerPC_Vec_BBB_Intrinsic<"vmaxsb">;
def int_ppc_altivec_vmaxsh : PowerPC_Vec_HHH_Intrinsic<"vmaxsh">;
def int_ppc_altivec_vmaxsw : PowerPC_Vec_WWW_Intrinsic<"vmaxsw">;
def int_ppc_altivec_vmaxsd : PowerPC_Vec_DDD_Intrinsic<"vmaxsd">;
def int_ppc_altivec_vmaxub : PowerPC_Vec_BBB_Intrinsic<"vmaxub">;
def int_ppc_altivec_vmaxuh : PowerPC_Vec_HHH_Intrinsic<"vmaxuh">;
def int_ppc_altivec_vmaxuw : PowerPC_Vec_WWW_Intrinsic<"vmaxuw">;
def int_ppc_altivec_vmaxud : PowerPC_Vec_DDD_Intrinsic<"vmaxud">;

// Vector minimum.
def int_ppc_altivec_vminfp : PowerPC_Vec_FFF_Intrinsic<"vminfp">;
def int_ppc_altivec_vminsb : PowerPC_Vec_BBB_Intrinsic<"vminsb">;
def int_ppc_altivec_vminsh : PowerPC_Vec_HHH_Intrinsic<"vminsh">;
def int_ppc_altivec_vminsw : PowerPC_Vec_WWW_Intrinsic<"vminsw">;
def int_ppc_altivec_vminsd : PowerPC_Vec_DDD_Intrinsic<"vminsd">;
def int_ppc_altivec_vminub : PowerPC_Vec_BBB_Intrinsic<"vminub">;
def int_ppc_altivec_vminuh : PowerPC_Vec_HHH_Intrinsic<"vminuh">;
def int_ppc_altivec_vminuw : PowerPC_Vec_WWW_Intrinsic<"vminuw">;
def int_ppc_altivec_vminud : PowerPC_Vec_DDD_Intrinsic<"vminud">;

// Saturating adds.
def int_ppc_altivec_vaddubs : PowerPC_Vec_BBB_Intrinsic<"vaddubs">;
def int_ppc_altivec_vaddsbs : PowerPC_Vec_BBB_Intrinsic<"vaddsbs">;
def int_ppc_altivec_vadduhs : PowerPC_Vec_HHH_Intrinsic<"vadduhs">;
def int_ppc_altivec_vaddshs : PowerPC_Vec_HHH_Intrinsic<"vaddshs">;
def int_ppc_altivec_vadduws : PowerPC_Vec_WWW_Intrinsic<"vadduws">;
def int_ppc_altivec_vaddsws : PowerPC_Vec_WWW_Intrinsic<"vaddsws">;
def int_ppc_altivec_vaddcuw : PowerPC_Vec_WWW_Intrinsic<"vaddcuw">;
def int_ppc_altivec_vaddcuq : PowerPC_Vec_QQQ_Intrinsic<"vaddcuq">;

// Saturating subs.
def int_ppc_altivec_vsububs : PowerPC_Vec_BBB_Intrinsic<"vsububs">;
def int_ppc_altivec_vsubsbs : PowerPC_Vec_BBB_Intrinsic<"vsubsbs">;
def int_ppc_altivec_vsubuhs : PowerPC_Vec_HHH_Intrinsic<"vsubuhs">;
def int_ppc_altivec_vsubshs : PowerPC_Vec_HHH_Intrinsic<"vsubshs">;
def int_ppc_altivec_vsubuws : PowerPC_Vec_WWW_Intrinsic<"vsubuws">;
def int_ppc_altivec_vsubsws : PowerPC_Vec_WWW_Intrinsic<"vsubsws">;
def int_ppc_altivec_vsubcuw : PowerPC_Vec_WWW_Intrinsic<"vsubcuw">;
def int_ppc_altivec_vsubcuq : PowerPC_Vec_QQQ_Intrinsic<"vsubcuq">;

let TargetPrefix = "ppc" in {  // All PPC intrinsics start with "llvm.ppc.".
  // Saturating multiply-adds.
  def int_ppc_altivec_vmhaddshs : GCCBuiltin<"__builtin_altivec_vmhaddshs">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty, llvm_v8i16_ty],
                         [IntrNoMem, IntrHasSideEffects]>;
  def int_ppc_altivec_vmhraddshs : GCCBuiltin<"__builtin_altivec_vmhraddshs">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty, llvm_v8i16_ty],
                         [IntrNoMem, IntrHasSideEffects]>;

  def int_ppc_altivec_vmaddfp : GCCBuiltin<"__builtin_altivec_vmaddfp">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_v4f32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vnmsubfp : GCCBuiltin<"__builtin_altivec_vnmsubfp">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_v4f32_ty], [IntrNoMem]>;

  // Vector Multiply Sum Instructions.
  def int_ppc_altivec_vmsummbm : GCCBuiltin<"__builtin_altivec_vmsummbm">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v16i8_ty, llvm_v16i8_ty,
                       llvm_v4i32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vmsumshm : GCCBuiltin<"__builtin_altivec_vmsumshm">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                       llvm_v4i32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vmsumshs : GCCBuiltin<"__builtin_altivec_vmsumshs">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                       llvm_v4i32_ty], [IntrNoMem, IntrHasSideEffects]>;
  def int_ppc_altivec_vmsumubm : GCCBuiltin<"__builtin_altivec_vmsumubm">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v16i8_ty, llvm_v16i8_ty,
                       llvm_v4i32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vmsumuhm : GCCBuiltin<"__builtin_altivec_vmsumuhm">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                       llvm_v4i32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vmsumudm : GCCBuiltin<"__builtin_altivec_vmsumudm">,
            Intrinsic<[llvm_v1i128_ty], [llvm_v2i64_ty, llvm_v2i64_ty,
                       llvm_v1i128_ty], [IntrNoMem]>;
  def int_ppc_altivec_vmsumuhs : GCCBuiltin<"__builtin_altivec_vmsumuhs">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                       llvm_v4i32_ty], [IntrNoMem, IntrHasSideEffects]>;
  def int_ppc_altivec_vmsumcud : GCCBuiltin<"__builtin_altivec_vmsumcud">,
            Intrinsic<[llvm_v1i128_ty],
                      [llvm_v2i64_ty, llvm_v2i64_ty, llvm_v1i128_ty], [IntrNoMem]>;

  // Vector Multiply Instructions.
  def int_ppc_altivec_vmulesb : GCCBuiltin<"__builtin_altivec_vmulesb">,
          Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmulesh : GCCBuiltin<"__builtin_altivec_vmulesh">,
          Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmulesw : GCCBuiltin<"__builtin_altivec_vmulesw">,
          Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmulesd : PowerPC_Vec_QDD_Intrinsic<"vmulesd">;
  def int_ppc_altivec_vmuleub : GCCBuiltin<"__builtin_altivec_vmuleub">,
          Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmuleuh : GCCBuiltin<"__builtin_altivec_vmuleuh">,
          Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmuleuw : GCCBuiltin<"__builtin_altivec_vmuleuw">,
          Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmuleud : PowerPC_Vec_QDD_Intrinsic<"vmuleud">;

  def int_ppc_altivec_vmulosb : GCCBuiltin<"__builtin_altivec_vmulosb">,
          Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmulosh : GCCBuiltin<"__builtin_altivec_vmulosh">,
          Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmulosw : GCCBuiltin<"__builtin_altivec_vmulosw">,
          Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmulosd : PowerPC_Vec_QDD_Intrinsic<"vmulosd">;
  def int_ppc_altivec_vmuloub : GCCBuiltin<"__builtin_altivec_vmuloub">,
          Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmulouh : GCCBuiltin<"__builtin_altivec_vmulouh">,
          Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmulouw : GCCBuiltin<"__builtin_altivec_vmulouw">,
          Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmuloud : PowerPC_Vec_QDD_Intrinsic<"vmuloud">;

  // Vector Sum Instructions.
  def int_ppc_altivec_vsumsws : GCCBuiltin<"__builtin_altivec_vsumsws">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                      [IntrNoMem, IntrHasSideEffects]>;
  def int_ppc_altivec_vsum2sws : GCCBuiltin<"__builtin_altivec_vsum2sws">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                      [IntrNoMem, IntrHasSideEffects]>;
  def int_ppc_altivec_vsum4sbs : GCCBuiltin<"__builtin_altivec_vsum4sbs">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v16i8_ty, llvm_v4i32_ty],
                      [IntrNoMem, IntrHasSideEffects]>;
  def int_ppc_altivec_vsum4shs : GCCBuiltin<"__builtin_altivec_vsum4shs">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v4i32_ty],
                      [IntrNoMem, IntrHasSideEffects]>;
  def int_ppc_altivec_vsum4ubs : GCCBuiltin<"__builtin_altivec_vsum4ubs">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v16i8_ty, llvm_v4i32_ty],
                      [IntrNoMem, IntrHasSideEffects]>;

  // Vector Sign Extension Instructions
  def int_ppc_altivec_vextsb2w : GCCBuiltin<"__builtin_altivec_vextsb2w">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_ppc_altivec_vextsb2d : GCCBuiltin<"__builtin_altivec_vextsb2d">,
            Intrinsic<[llvm_v2i64_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_ppc_altivec_vextsh2w : GCCBuiltin<"__builtin_altivec_vextsh2w">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_ppc_altivec_vextsh2d : GCCBuiltin<"__builtin_altivec_vextsh2d">,
            Intrinsic<[llvm_v2i64_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_ppc_altivec_vextsw2d : GCCBuiltin<"__builtin_altivec_vextsw2d">,
            Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vextsd2q : GCCBuiltin<"__builtin_altivec_vextsd2q">,
            Intrinsic<[llvm_v1i128_ty], [llvm_v2i64_ty], [IntrNoMem]>;

  // Other multiplies.
  def int_ppc_altivec_vmladduhm : GCCBuiltin<"__builtin_altivec_vmladduhm">,
            Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                       llvm_v8i16_ty], [IntrNoMem]>;

  // Packs.
  def int_ppc_altivec_vpkpx : GCCBuiltin<"__builtin_altivec_vpkpx">,
            Intrinsic<[llvm_v8i16_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                      [IntrNoMem]>;
  def int_ppc_altivec_vpkshss : GCCBuiltin<"__builtin_altivec_vpkshss">,
            Intrinsic<[llvm_v16i8_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                      [IntrNoMem, IntrHasSideEffects]>;
  def int_ppc_altivec_vpkshus : GCCBuiltin<"__builtin_altivec_vpkshus">,
            Intrinsic<[llvm_v16i8_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                      [IntrNoMem, IntrHasSideEffects]>;
  def int_ppc_altivec_vpkswss : GCCBuiltin<"__builtin_altivec_vpkswss">,
            Intrinsic<[llvm_v8i16_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                      [IntrNoMem, IntrHasSideEffects]>;
  def int_ppc_altivec_vpkswus : GCCBuiltin<"__builtin_altivec_vpkswus">,
            Intrinsic<[llvm_v8i16_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                      [IntrNoMem, IntrHasSideEffects]>;
  def int_ppc_altivec_vpksdss : GCCBuiltin<"__builtin_altivec_vpksdss">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                      [IntrNoMem, IntrHasSideEffects]>;
  def int_ppc_altivec_vpksdus : GCCBuiltin<"__builtin_altivec_vpksdus">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                      [IntrNoMem, IntrHasSideEffects]>;
  // vpkuhum is lowered to a shuffle.
  def int_ppc_altivec_vpkuhus : GCCBuiltin<"__builtin_altivec_vpkuhus">,
            Intrinsic<[llvm_v16i8_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                      [IntrNoMem, IntrHasSideEffects]>;
  // vpkuwum is lowered to a shuffle.
  def int_ppc_altivec_vpkuwus : GCCBuiltin<"__builtin_altivec_vpkuwus">,
            Intrinsic<[llvm_v8i16_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                      [IntrNoMem, IntrHasSideEffects]>;
  // vpkudum is lowered to a shuffle.
  def int_ppc_altivec_vpkudus : GCCBuiltin<"__builtin_altivec_vpkudus">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                      [IntrNoMem, IntrHasSideEffects]>;

  // Unpacks.
  def int_ppc_altivec_vupkhpx : GCCBuiltin<"__builtin_altivec_vupkhpx">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_ppc_altivec_vupkhsb : GCCBuiltin<"__builtin_altivec_vupkhsb">,
            Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_ppc_altivec_vupkhsh : GCCBuiltin<"__builtin_altivec_vupkhsh">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_ppc_altivec_vupkhsw : GCCBuiltin<"__builtin_altivec_vupkhsw">,
            Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vupklpx : GCCBuiltin<"__builtin_altivec_vupklpx">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_ppc_altivec_vupklsb : GCCBuiltin<"__builtin_altivec_vupklsb">,
            Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_ppc_altivec_vupklsh : GCCBuiltin<"__builtin_altivec_vupklsh">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_ppc_altivec_vupklsw : GCCBuiltin<"__builtin_altivec_vupklsw">,
            Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty], [IntrNoMem]>;


  // FP <-> integer conversion.
  def int_ppc_altivec_vcfsx : GCCBuiltin<"__builtin_altivec_vcfsx">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4i32_ty, llvm_i32_ty],
                        [IntrNoMem, ImmArg<ArgIndex<1>>]>;
  def int_ppc_altivec_vcfux : GCCBuiltin<"__builtin_altivec_vcfux">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4i32_ty, llvm_i32_ty],
                        [IntrNoMem, ImmArg<ArgIndex<1>>]>;
  def int_ppc_altivec_vctsxs : GCCBuiltin<"__builtin_altivec_vctsxs">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4f32_ty, llvm_i32_ty],
                        [IntrNoMem, ImmArg<ArgIndex<1>>]>;
  def int_ppc_altivec_vctuxs : GCCBuiltin<"__builtin_altivec_vctuxs">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4f32_ty, llvm_i32_ty],
                        [IntrNoMem, ImmArg<ArgIndex<1>>]>;

  def int_ppc_altivec_vrfim : GCCBuiltin<"__builtin_altivec_vrfim">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vrfin : GCCBuiltin<"__builtin_altivec_vrfin">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vrfip : GCCBuiltin<"__builtin_altivec_vrfip">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vrfiz : GCCBuiltin<"__builtin_altivec_vrfiz">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;

  // Add Extended Quadword
  def int_ppc_altivec_vaddeuqm : GCCBuiltin<"__builtin_altivec_vaddeuqm">,
              Intrinsic<[llvm_v1i128_ty],
                        [llvm_v1i128_ty, llvm_v1i128_ty, llvm_v1i128_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vaddecuq : GCCBuiltin<"__builtin_altivec_vaddecuq">,
              Intrinsic<[llvm_v1i128_ty],
                        [llvm_v1i128_ty, llvm_v1i128_ty, llvm_v1i128_ty],
                        [IntrNoMem]>;

  // Sub Extended Quadword
  def int_ppc_altivec_vsubeuqm : GCCBuiltin<"__builtin_altivec_vsubeuqm">,
              Intrinsic<[llvm_v1i128_ty],
                        [llvm_v1i128_ty, llvm_v1i128_ty, llvm_v1i128_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vsubecuq : GCCBuiltin<"__builtin_altivec_vsubecuq">,
              Intrinsic<[llvm_v1i128_ty],
                        [llvm_v1i128_ty, llvm_v1i128_ty, llvm_v1i128_ty],
                        [IntrNoMem]>;

  // P10 Vector Count Leading / Trailing Zeroes under bit Mask Builtins.
  def int_ppc_altivec_vclzdm : GCCBuiltin<"__builtin_altivec_vclzdm">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vctzdm : GCCBuiltin<"__builtin_altivec_vctzdm">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
}

def int_ppc_altivec_vsl   : PowerPC_Vec_WWW_Intrinsic<"vsl">;
def int_ppc_altivec_vslo  : PowerPC_Vec_WWW_Intrinsic<"vslo">;

def int_ppc_altivec_vslb  : PowerPC_Vec_BBB_Intrinsic<"vslb">;
def int_ppc_altivec_vslv  : PowerPC_Vec_BBB_Intrinsic<"vslv">;
def int_ppc_altivec_vsrv  : PowerPC_Vec_BBB_Intrinsic<"vsrv">;
def int_ppc_altivec_vslh  : PowerPC_Vec_HHH_Intrinsic<"vslh">;
def int_ppc_altivec_vslw  : PowerPC_Vec_WWW_Intrinsic<"vslw">;

// Right Shifts.
def int_ppc_altivec_vsr   : PowerPC_Vec_WWW_Intrinsic<"vsr">;
def int_ppc_altivec_vsro  : PowerPC_Vec_WWW_Intrinsic<"vsro">;

def int_ppc_altivec_vsrb  : PowerPC_Vec_BBB_Intrinsic<"vsrb">;
def int_ppc_altivec_vsrh  : PowerPC_Vec_HHH_Intrinsic<"vsrh">;
def int_ppc_altivec_vsrw  : PowerPC_Vec_WWW_Intrinsic<"vsrw">;
def int_ppc_altivec_vsrab : PowerPC_Vec_BBB_Intrinsic<"vsrab">;
def int_ppc_altivec_vsrah : PowerPC_Vec_HHH_Intrinsic<"vsrah">;
def int_ppc_altivec_vsraw : PowerPC_Vec_WWW_Intrinsic<"vsraw">;

// Rotates.
def int_ppc_altivec_vrlb  : PowerPC_Vec_BBB_Intrinsic<"vrlb">;
def int_ppc_altivec_vrlh  : PowerPC_Vec_HHH_Intrinsic<"vrlh">;
def int_ppc_altivec_vrlw  : PowerPC_Vec_WWW_Intrinsic<"vrlw">;
def int_ppc_altivec_vrld  : PowerPC_Vec_DDD_Intrinsic<"vrld">;

let TargetPrefix = "ppc" in {  // All PPC intrinsics start with "llvm.ppc.".
  // Miscellaneous.
  def int_ppc_altivec_lvsl :
              Intrinsic<[llvm_v16i8_ty], [llvm_ptr_ty], [IntrNoMem]>;
  def int_ppc_altivec_lvsr :
              Intrinsic<[llvm_v16i8_ty], [llvm_ptr_ty], [IntrNoMem]>;

  def int_ppc_altivec_vperm : GCCBuiltin<"__builtin_altivec_vperm_4si">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty, llvm_v16i8_ty], [IntrNoMem]>;
  def int_ppc_altivec_vsel : GCCBuiltin<"__builtin_altivec_vsel_4si">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vgbbd : GCCBuiltin<"__builtin_altivec_vgbbd">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_ppc_altivec_vbpermq : GCCBuiltin<"__builtin_altivec_vbpermq">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                        [IntrNoMem]>;
}

def int_ppc_altivec_vexptefp  : PowerPC_Vec_FF_Intrinsic<"vexptefp">;
def int_ppc_altivec_vlogefp   : PowerPC_Vec_FF_Intrinsic<"vlogefp">;
def int_ppc_altivec_vrefp     : PowerPC_Vec_FF_Intrinsic<"vrefp">;
def int_ppc_altivec_vrsqrtefp : PowerPC_Vec_FF_Intrinsic<"vrsqrtefp">;

// Power8 Intrinsics
// Crypto
let TargetPrefix = "ppc" in {  // All PPC intrinsics start with "llvm.ppc.".
  def int_ppc_altivec_crypto_vsbox :
              GCCBuiltin<"__builtin_altivec_crypto_vsbox">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty], [IntrNoMem]>;
  def int_ppc_altivec_crypto_vpermxor :
              GCCBuiltin<"__builtin_altivec_crypto_vpermxor">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty, llvm_v16i8_ty], [IntrNoMem]>;

def int_ppc_altivec_crypto_vshasigmad :
            GCCBuiltin<"__builtin_altivec_crypto_vshasigmad">,
            Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                       llvm_i32_ty, llvm_i32_ty], [IntrNoMem, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
def int_ppc_altivec_crypto_vshasigmaw :
            GCCBuiltin<"__builtin_altivec_crypto_vshasigmaw">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                       llvm_i32_ty, llvm_i32_ty], [IntrNoMem, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
}
def int_ppc_altivec_crypto_vcipher :
            PowerPC_Vec_DDD_Intrinsic<"crypto_vcipher">;
def int_ppc_altivec_crypto_vcipherlast :
            PowerPC_Vec_DDD_Intrinsic<"crypto_vcipherlast">;
def int_ppc_altivec_crypto_vncipher :
            PowerPC_Vec_DDD_Intrinsic<"crypto_vncipher">;
def int_ppc_altivec_crypto_vncipherlast :
            PowerPC_Vec_DDD_Intrinsic<"crypto_vncipherlast">;
def int_ppc_altivec_crypto_vpmsumb :
            PowerPC_Vec_BBB_Intrinsic<"crypto_vpmsumb">;
def int_ppc_altivec_crypto_vpmsumh :
            PowerPC_Vec_HHH_Intrinsic<"crypto_vpmsumh">;
def int_ppc_altivec_crypto_vpmsumw :
            PowerPC_Vec_WWW_Intrinsic<"crypto_vpmsumw">;
def int_ppc_altivec_crypto_vpmsumd :
            PowerPC_Vec_DDD_Intrinsic<"crypto_vpmsumd">;

// Absolute Difference intrinsics
def int_ppc_altivec_vabsdub : PowerPC_Vec_BBB_Intrinsic<"vabsdub">;
def int_ppc_altivec_vabsduh : PowerPC_Vec_HHH_Intrinsic<"vabsduh">;
def int_ppc_altivec_vabsduw : PowerPC_Vec_WWW_Intrinsic<"vabsduw">;

// Vector rotates
def int_ppc_altivec_vrlwnm :
      PowerPC_Vec_Intrinsic<"vrlwnm", [llvm_v4i32_ty],
                            [llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem]>;
def int_ppc_altivec_vrlwmi :
      PowerPC_Vec_Intrinsic<"vrlwmi", [llvm_v4i32_ty],
                            [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                            [IntrNoMem]>;
def int_ppc_altivec_vrldnm :
      PowerPC_Vec_Intrinsic<"vrldnm", [llvm_v2i64_ty],
                            [llvm_v2i64_ty, llvm_v2i64_ty], [IntrNoMem]>;
def int_ppc_altivec_vrldmi :
      PowerPC_Vec_Intrinsic<"vrldmi", [llvm_v2i64_ty],
                            [llvm_v2i64_ty, llvm_v2i64_ty, llvm_v2i64_ty],
                            [IntrNoMem]>;

def int_ppc_altivec_vrlqnm :
      PowerPC_Vec_Intrinsic<"vrlqnm", [llvm_v1i128_ty],
                           [llvm_v1i128_ty, llvm_v1i128_ty],
                            [IntrNoMem]>;
def int_ppc_altivec_vrlqmi :
      PowerPC_Vec_Intrinsic<"vrlqmi", [llvm_v1i128_ty],
                            [llvm_v1i128_ty, llvm_v1i128_ty, llvm_v1i128_ty],
                            [IntrNoMem]>;

// Vector Divide Extended Intrinsics.
def int_ppc_altivec_vdivesw : PowerPC_Vec_WWW_Intrinsic<"vdivesw">;
def int_ppc_altivec_vdiveuw : PowerPC_Vec_WWW_Intrinsic<"vdiveuw">;
def int_ppc_altivec_vdivesd : PowerPC_Vec_DDD_Intrinsic<"vdivesd">;
def int_ppc_altivec_vdiveud : PowerPC_Vec_DDD_Intrinsic<"vdiveud">;
def int_ppc_altivec_vdivesq : PowerPC_Vec_QQQ_Intrinsic<"vdivesq">;
def int_ppc_altivec_vdiveuq : PowerPC_Vec_QQQ_Intrinsic<"vdiveuq">;

// Vector Multiply High Intrinsics.
def int_ppc_altivec_vmulhsw : PowerPC_Vec_WWW_Intrinsic<"vmulhsw">;
def int_ppc_altivec_vmulhuw : PowerPC_Vec_WWW_Intrinsic<"vmulhuw">;
def int_ppc_altivec_vmulhsd : PowerPC_Vec_DDD_Intrinsic<"vmulhsd">;
def int_ppc_altivec_vmulhud : PowerPC_Vec_DDD_Intrinsic<"vmulhud">;

//===----------------------------------------------------------------------===//
// PowerPC VSX Intrinsic Definitions.

let TargetPrefix = "ppc" in {  // All intrinsics start with "llvm.ppc.".

// Vector load.
def int_ppc_vsx_lxvw4x :
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], [IntrReadMem, IntrArgMemOnly]>;
def int_ppc_vsx_lxvd2x :
      Intrinsic<[llvm_v2f64_ty], [llvm_ptr_ty], [IntrReadMem, IntrArgMemOnly]>;
def int_ppc_vsx_lxvw4x_be :
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], [IntrReadMem, IntrArgMemOnly]>;
def int_ppc_vsx_lxvd2x_be :
      Intrinsic<[llvm_v2f64_ty], [llvm_ptr_ty], [IntrReadMem, IntrArgMemOnly]>;
def int_ppc_vsx_lxvl :
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty, llvm_i64_ty], [IntrReadMem,
      IntrArgMemOnly]>;
def int_ppc_vsx_lxvll :
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty, llvm_i64_ty], [IntrReadMem,
      IntrArgMemOnly]>;
def int_ppc_vsx_lxvp :
      Intrinsic<[llvm_v256i1_ty], [llvm_ptr_ty], [IntrReadMem, IntrArgMemOnly]>;

// Vector store.
def int_ppc_vsx_stxvw4x : Intrinsic<[], [llvm_v4i32_ty, llvm_ptr_ty],
                                    [IntrWriteMem, IntrArgMemOnly]>;
def int_ppc_vsx_stxvd2x : Intrinsic<[], [llvm_v2f64_ty, llvm_ptr_ty],
                                    [IntrWriteMem, IntrArgMemOnly]>;
def int_ppc_vsx_stxvw4x_be : Intrinsic<[], [llvm_v4i32_ty, llvm_ptr_ty],
                                       [IntrWriteMem, IntrArgMemOnly]>;
def int_ppc_vsx_stxvd2x_be : Intrinsic<[], [llvm_v2f64_ty, llvm_ptr_ty],
                                       [IntrWriteMem, IntrArgMemOnly]>;
def int_ppc_vsx_stxvl :
      Intrinsic<[], [llvm_v4i32_ty, llvm_ptr_ty, llvm_i64_ty],
      [IntrWriteMem, IntrArgMemOnly]>;
def int_ppc_vsx_stxvll :
      Intrinsic<[], [llvm_v4i32_ty, llvm_ptr_ty, llvm_i64_ty],
      [IntrWriteMem, IntrArgMemOnly]>;
def int_ppc_vsx_stxvp :
      Intrinsic<[], [llvm_v256i1_ty, llvm_ptr_ty], [IntrWriteMem,
      IntrArgMemOnly]>;
// Vector and scalar maximum.
def int_ppc_vsx_xvmaxdp : PowerPC_VSX_Vec_DDD_Intrinsic<"xvmaxdp">;
def int_ppc_vsx_xvmaxsp : PowerPC_VSX_Vec_FFF_Intrinsic<"xvmaxsp">;
def int_ppc_vsx_xsmaxdp : PowerPC_VSX_Sca_DDD_Intrinsic<"xsmaxdp">;

// Vector and scalar minimum.
def int_ppc_vsx_xvmindp : PowerPC_VSX_Vec_DDD_Intrinsic<"xvmindp">;
def int_ppc_vsx_xvminsp : PowerPC_VSX_Vec_FFF_Intrinsic<"xvminsp">;
def int_ppc_vsx_xsmindp : PowerPC_VSX_Sca_DDD_Intrinsic<"xsmindp">;

// Vector divide.
def int_ppc_vsx_xvdivdp : PowerPC_VSX_Vec_DDD_Intrinsic<"xvdivdp">;
def int_ppc_vsx_xvdivsp : PowerPC_VSX_Vec_FFF_Intrinsic<"xvdivsp">;

// Vector round-to-infinity (ceil)
def int_ppc_vsx_xvrspip :
      Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvrdpip :
      Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty], [IntrNoMem]>;

// Vector reciprocal estimate
def int_ppc_vsx_xvresp : GCCBuiltin<"__builtin_vsx_xvresp">,
      Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvredp : GCCBuiltin<"__builtin_vsx_xvredp">,
      Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty], [IntrNoMem]>;

// Vector rsqrte
def int_ppc_vsx_xvrsqrtesp : GCCBuiltin<"__builtin_vsx_xvrsqrtesp">,
      Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvrsqrtedp : GCCBuiltin<"__builtin_vsx_xvrsqrtedp">,
      Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty], [IntrNoMem]>;

// Vector compare
def int_ppc_vsx_xvcmpeqdp :
      PowerPC_VSX_Intrinsic<"xvcmpeqdp", [llvm_v2i64_ty],
                            [llvm_v2f64_ty, llvm_v2f64_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcmpeqdp_p : GCCBuiltin<"__builtin_vsx_xvcmpeqdp_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v2f64_ty,llvm_v2f64_ty],
                        [IntrNoMem]>;
def int_ppc_vsx_xvcmpeqsp :
      PowerPC_VSX_Intrinsic<"xvcmpeqsp", [llvm_v4i32_ty],
                            [llvm_v4f32_ty, llvm_v4f32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcmpeqsp_p : GCCBuiltin<"__builtin_vsx_xvcmpeqsp_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v4f32_ty,llvm_v4f32_ty],
                        [IntrNoMem]>;
def int_ppc_vsx_xvcmpgedp :
      PowerPC_VSX_Intrinsic<"xvcmpgedp", [llvm_v2i64_ty],
                            [llvm_v2f64_ty, llvm_v2f64_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcmpgedp_p : GCCBuiltin<"__builtin_vsx_xvcmpgedp_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v2f64_ty,llvm_v2f64_ty],
                        [IntrNoMem]>;
def int_ppc_vsx_xvcmpgesp :
      PowerPC_VSX_Intrinsic<"xvcmpgesp", [llvm_v4i32_ty],
                            [llvm_v4f32_ty, llvm_v4f32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcmpgesp_p : GCCBuiltin<"__builtin_vsx_xvcmpgesp_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v4f32_ty,llvm_v4f32_ty],
                        [IntrNoMem]>;
def int_ppc_vsx_xvcmpgtdp :
      PowerPC_VSX_Intrinsic<"xvcmpgtdp", [llvm_v2i64_ty],
                            [llvm_v2f64_ty, llvm_v2f64_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcmpgtdp_p : GCCBuiltin<"__builtin_vsx_xvcmpgtdp_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v2f64_ty,llvm_v2f64_ty],
                        [IntrNoMem]>;
def int_ppc_vsx_xvcmpgtsp :
      PowerPC_VSX_Intrinsic<"xvcmpgtsp", [llvm_v4i32_ty],
                            [llvm_v4f32_ty, llvm_v4f32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcmpgtsp_p : GCCBuiltin<"__builtin_vsx_xvcmpgtsp_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v4f32_ty,llvm_v4f32_ty],
                        [IntrNoMem]>;
def int_ppc_vsx_xxleqv :
      PowerPC_VSX_Intrinsic<"xxleqv", [llvm_v4i32_ty],
                            [llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem]>;
def int_ppc_vsx_xviexpdp :
      PowerPC_VSX_Intrinsic<"xviexpdp",[llvm_v2f64_ty],
                            [llvm_v2i64_ty, llvm_v2i64_ty],[IntrNoMem]>;
def int_ppc_vsx_xviexpsp :
      PowerPC_VSX_Intrinsic<"xviexpsp",[llvm_v4f32_ty],
                            [llvm_v4i32_ty, llvm_v4i32_ty],[IntrNoMem]>;
def int_ppc_vsx_xvcvdpsxws :
      PowerPC_VSX_Intrinsic<"xvcvdpsxws", [llvm_v4i32_ty],
                            [llvm_v2f64_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcvdpuxws :
      PowerPC_VSX_Intrinsic<"xvcvdpuxws", [llvm_v4i32_ty],
                            [llvm_v2f64_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcvspsxds :
      PowerPC_VSX_Intrinsic<"xvcvspsxds", [llvm_v2i64_ty],
                            [llvm_v4f32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcvspuxds :
      PowerPC_VSX_Intrinsic<"xvcvspuxds", [llvm_v2i64_ty],
                            [llvm_v4f32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcvsxwdp :
      PowerPC_VSX_Intrinsic<"xvcvsxwdp", [llvm_v2f64_ty],
                            [llvm_v4i32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcvuxwdp :
      PowerPC_VSX_Intrinsic<"xvcvuxwdp", [llvm_v2f64_ty],
                            [llvm_v4i32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcvspdp :
      PowerPC_VSX_Intrinsic<"xvcvspdp", [llvm_v2f64_ty],
                            [llvm_v4f32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcvsxdsp :
      PowerPC_VSX_Intrinsic<"xvcvsxdsp", [llvm_v4f32_ty],
                            [llvm_v2i64_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcvuxdsp :
      PowerPC_VSX_Intrinsic<"xvcvuxdsp", [llvm_v4f32_ty],
                            [llvm_v2i64_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcvdpsp :
      PowerPC_VSX_Intrinsic<"xvcvdpsp", [llvm_v4f32_ty],
                            [llvm_v2f64_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcvsphp :
      PowerPC_VSX_Intrinsic<"xvcvsphp", [llvm_v4f32_ty],
                            [llvm_v4f32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvxexpdp :
      PowerPC_VSX_Intrinsic<"xvxexpdp", [llvm_v2i64_ty],
                            [llvm_v2f64_ty], [IntrNoMem]>;
def int_ppc_vsx_xvxexpsp :
      PowerPC_VSX_Intrinsic<"xvxexpsp", [llvm_v4i32_ty],
                            [llvm_v4f32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvxsigdp :
      PowerPC_VSX_Intrinsic<"xvxsigdp", [llvm_v2i64_ty],
                            [llvm_v2f64_ty], [IntrNoMem]>;
def int_ppc_vsx_xvxsigsp :
      PowerPC_VSX_Intrinsic<"xvxsigsp", [llvm_v4i32_ty],
                            [llvm_v4f32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvtstdcdp :
      PowerPC_VSX_Intrinsic<"xvtstdcdp", [llvm_v2i64_ty],
                            [llvm_v2f64_ty, llvm_i32_ty], [IntrNoMem, ImmArg<ArgIndex<1>>]>;
def int_ppc_vsx_xvtstdcsp :
      PowerPC_VSX_Intrinsic<"xvtstdcsp", [llvm_v4i32_ty],
                            [llvm_v4f32_ty,llvm_i32_ty], [IntrNoMem, ImmArg<ArgIndex<1>>]>;
def int_ppc_vsx_xvcvhpsp :
      PowerPC_VSX_Intrinsic<"xvcvhpsp", [llvm_v4f32_ty],
                            [llvm_v8i16_ty],[IntrNoMem]>;
def int_ppc_vsx_xvcvspbf16 :
      PowerPC_VSX_Intrinsic<"xvcvspbf16", [llvm_v16i8_ty],
                            [llvm_v16i8_ty], [IntrNoMem]>;
def int_ppc_vsx_xvcvbf16spn :
      PowerPC_VSX_Intrinsic<"xvcvbf16spn", [llvm_v16i8_ty],
                            [llvm_v16i8_ty], [IntrNoMem]>;
def int_ppc_vsx_xxextractuw :
      PowerPC_VSX_Intrinsic<"xxextractuw",[llvm_v2i64_ty],
                            [llvm_v2i64_ty,llvm_i32_ty], [IntrNoMem]>;
def int_ppc_vsx_xxinsertw :
      PowerPC_VSX_Intrinsic<"xxinsertw",[llvm_v4i32_ty],
                            [llvm_v4i32_ty,llvm_v2i64_ty,llvm_i32_ty],
                            [IntrNoMem]>;
def int_ppc_vsx_xvtlsbb :
      PowerPC_VSX_Intrinsic<"xvtlsbb", [llvm_i32_ty],
                            [llvm_v16i8_ty, llvm_i32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvtdivdp :
      PowerPC_VSX_Intrinsic<"xvtdivdp", [llvm_i32_ty],
                            [llvm_v2f64_ty, llvm_v2f64_ty], [IntrNoMem]>;
def int_ppc_vsx_xvtdivsp :
      PowerPC_VSX_Intrinsic<"xvtdivsp", [llvm_i32_ty],
                            [llvm_v4f32_ty, llvm_v4f32_ty], [IntrNoMem]>;
def int_ppc_vsx_xvtsqrtdp :
      PowerPC_VSX_Intrinsic<"xvtsqrtdp", [llvm_i32_ty], [llvm_v2f64_ty], [IntrNoMem]>;
def int_ppc_vsx_xvtsqrtsp :
      PowerPC_VSX_Intrinsic<"xvtsqrtsp", [llvm_i32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
def int_ppc_vsx_xxeval :
      PowerPC_VSX_Intrinsic<"xxeval", [llvm_v2i64_ty],
                           [llvm_v2i64_ty, llvm_v2i64_ty,
                            llvm_v2i64_ty, llvm_i32_ty],
                           [IntrNoMem, ImmArg<ArgIndex<3>>]>;
def int_ppc_vsx_xxgenpcvbm :
      PowerPC_VSX_Intrinsic<"xxgenpcvbm", [llvm_v16i8_ty],
                            [llvm_v16i8_ty, llvm_i32_ty], [IntrNoMem]>;
def int_ppc_vsx_xxgenpcvhm :
      PowerPC_VSX_Intrinsic<"xxgenpcvhm", [llvm_v8i16_ty],
                            [llvm_v8i16_ty, llvm_i32_ty], [IntrNoMem]>;
def int_ppc_vsx_xxgenpcvwm :
      PowerPC_VSX_Intrinsic<"xxgenpcvwm", [llvm_v4i32_ty],
                            [llvm_v4i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_ppc_vsx_xxgenpcvdm :
      PowerPC_VSX_Intrinsic<"xxgenpcvdm", [llvm_v2i64_ty],
                            [llvm_v2i64_ty, llvm_i32_ty], [IntrNoMem]>;

// P10 VSX Vector permute extended.
def int_ppc_vsx_xxpermx : 
      GCCBuiltin<"__builtin_vsx_xxpermx">,
      Intrinsic<[llvm_v16i8_ty],
                [llvm_v16i8_ty,llvm_v16i8_ty,llvm_v16i8_ty,llvm_i32_ty],
                [IntrNoMem, ImmArg<ArgIndex<3>>]>;
// P10 VSX Vector Blend Variable.
def  int_ppc_vsx_xxblendvb: GCCBuiltin<"__builtin_vsx_xxblendvb">,
       Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                 [IntrNoMem]>;
def  int_ppc_vsx_xxblendvh: GCCBuiltin<"__builtin_vsx_xxblendvh">,
       Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty,llvm_v8i16_ty],
                 [IntrNoMem]>;
def  int_ppc_vsx_xxblendvw: GCCBuiltin<"__builtin_vsx_xxblendvw">,
       Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                 [IntrNoMem]>;
def  int_ppc_vsx_xxblendvd: GCCBuiltin<"__builtin_vsx_xxblendvd">,
       Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty, llvm_v2i64_ty],
                 [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// PowerPC HTM Intrinsic Definitions.

let TargetPrefix = "ppc" in {  // All intrinsics start with "llvm.ppc.".

def int_ppc_tbegin : GCCBuiltin<"__builtin_tbegin">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_ppc_tend : GCCBuiltin<"__builtin_tend">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_ppc_tabort : GCCBuiltin<"__builtin_tabort">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty], []>;
def int_ppc_tabortwc : GCCBuiltin<"__builtin_tabortwc">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;
def int_ppc_tabortwci : GCCBuiltin<"__builtin_tabortwci">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;
def int_ppc_tabortdc : GCCBuiltin<"__builtin_tabortdc">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;
def int_ppc_tabortdci : GCCBuiltin<"__builtin_tabortdci">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_ppc_tcheck : GCCBuiltin<"__builtin_tcheck">,
      Intrinsic<[llvm_i32_ty], [], []>;
def int_ppc_treclaim : GCCBuiltin<"__builtin_treclaim">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty], []>;
def int_ppc_trechkpt : GCCBuiltin<"__builtin_trechkpt">,
      Intrinsic<[llvm_i32_ty], [], []>;
def int_ppc_tsr : GCCBuiltin<"__builtin_tsr">,
      Intrinsic<[llvm_i32_ty], [llvm_i32_ty], []>;

def int_ppc_get_texasr : GCCBuiltin<"__builtin_get_texasr">,
      Intrinsic<[llvm_i64_ty], [], []>;
def int_ppc_get_texasru : GCCBuiltin<"__builtin_get_texasru">,
      Intrinsic<[llvm_i64_ty], [], []>;
def int_ppc_get_tfhar : GCCBuiltin<"__builtin_get_tfhar">,
      Intrinsic<[llvm_i64_ty], [], []>;
def int_ppc_get_tfiar : GCCBuiltin<"__builtin_get_tfiar">,
      Intrinsic<[llvm_i64_ty], [], []>;

def int_ppc_set_texasr : GCCBuiltin<"__builtin_set_texasr">,
      Intrinsic<[], [llvm_i64_ty], []>;
def int_ppc_set_texasru : GCCBuiltin<"__builtin_set_texasru">,
      Intrinsic<[], [llvm_i64_ty], []>;
def int_ppc_set_tfhar : GCCBuiltin<"__builtin_set_tfhar">,
      Intrinsic<[], [llvm_i64_ty], []>;
def int_ppc_set_tfiar : GCCBuiltin<"__builtin_set_tfiar">,
      Intrinsic<[], [llvm_i64_ty], []>;

// Extended mnemonics
def int_ppc_tendall : GCCBuiltin<"__builtin_tendall">,
      Intrinsic<[llvm_i32_ty], [], []>;
def int_ppc_tresume : GCCBuiltin<"__builtin_tresume">,
      Intrinsic<[llvm_i32_ty], [], []>;
def int_ppc_tsuspend : GCCBuiltin<"__builtin_tsuspend">,
      Intrinsic<[llvm_i32_ty], [], []>;

def int_ppc_ttest : GCCBuiltin<"__builtin_ttest">,
      Intrinsic<[llvm_i64_ty], [], []>;

def int_ppc_cfence : Intrinsic<[], [llvm_anyint_ty], []>;

// PowerPC set FPSCR Intrinsic Definitions.
def int_ppc_setrnd : GCCBuiltin<"__builtin_setrnd">,
      Intrinsic<[llvm_double_ty], [llvm_i32_ty], []>;
}

let TargetPrefix = "ppc" in {
  def int_ppc_vsx_assemble_pair :
        Intrinsic<[llvm_v256i1_ty],
                  [llvm_v16i8_ty, llvm_v16i8_ty], [IntrNoMem]>;

  def int_ppc_vsx_disassemble_pair :
        Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty],
                  [llvm_v256i1_ty], [IntrNoMem]>;

  def int_ppc_mma_assemble_acc :
        Intrinsic<[llvm_v512i1_ty],
                  [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                  [IntrNoMem]>;

  def int_ppc_mma_disassemble_acc :
        Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                  [llvm_v512i1_ty], [IntrNoMem]>;

  def int_ppc_mma_xxmtacc :
        Intrinsic<[llvm_v512i1_ty], [llvm_v512i1_ty], [IntrNoMem]>;

  def int_ppc_mma_xxmfacc :
        Intrinsic<[llvm_v512i1_ty], [llvm_v512i1_ty], [IntrNoMem]>;

  def int_ppc_mma_xxsetaccz :
        Intrinsic<[llvm_v512i1_ty], [], [IntrNoMem]>;

  // MMA Reduced-Precision: Outer Product Intrinsic Definitions.
  defm int_ppc_mma_xvi4ger8 :
        PowerPC_MMA_ACC_PP_Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty]>;
  defm int_ppc_mma_pmxvi4ger8 :
        PowerPC_MMA_ACC_PP_Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty,
                                      llvm_i32_ty, llvm_i32_ty]>;

  defm int_ppc_mma_xvi8ger4 :
       PowerPC_MMA_ACC_PP_Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty]>;
  defm int_ppc_mma_pmxvi8ger4 :
       PowerPC_MMA_ACC_PP_Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty,
                                     llvm_i32_ty, llvm_i32_ty]>;

  defm int_ppc_mma_xvi16ger2s :
       PowerPC_MMA_ACC_PP_Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty]>;
  defm int_ppc_mma_pmxvi16ger2s :
       PowerPC_MMA_ACC_PP_Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty,
                                     llvm_i32_ty, llvm_i32_ty]>;

  defm int_ppc_mma_xvf16ger2 :
       PowerPC_MMA_ACC_Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty]>;
  defm int_ppc_mma_pmxvf16ger2 :
       PowerPC_MMA_ACC_Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty,
                                  llvm_i32_ty, llvm_i32_ty]>;
  defm int_ppc_mma_xvf32ger :
       PowerPC_MMA_ACC_Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty]>;
  defm int_ppc_mma_pmxvf32ger :
       PowerPC_MMA_ACC_Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty,
                                  llvm_i32_ty]>;
  defm int_ppc_mma_xvf64ger :
       PowerPC_MMA_ACC_Intrinsic<[llvm_v256i1_ty, llvm_v16i8_ty]>;
  defm int_ppc_mma_pmxvf64ger :
       PowerPC_MMA_ACC_Intrinsic<[llvm_v256i1_ty, llvm_v16i8_ty, llvm_i32_ty,
                                  llvm_i32_ty]>;

  // MMA Reduced-Precision: bfloat16 Outer Product Intrinsic Definitions.
  defm int_ppc_mma_xvbf16ger2 :
         PowerPC_MMA_ACC_Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty]>;
  defm int_ppc_mma_pmxvbf16ger2 :
         PowerPC_MMA_ACC_Intrinsic<
           [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty]>;

  // MMA Reduced-Precision: Missing Integer-based Outer Product Operations.
  defm int_ppc_mma_xvi16ger2 :
         PowerPC_MMA_ACC_PP_Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty]>;
  defm int_ppc_mma_pmxvi16ger2 :
         PowerPC_MMA_ACC_PP_Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty,
                                       llvm_i32_ty, llvm_i32_ty]>;
  def int_ppc_mma_xvi8ger4spp :
        Intrinsic<[llvm_v512i1_ty],
                  [llvm_v512i1_ty, llvm_v16i8_ty, llvm_v16i8_ty], [IntrNoMem]>;
  def int_ppc_mma_pmxvi8ger4spp :
        Intrinsic<[llvm_v512i1_ty],
                  [llvm_v512i1_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty,
                   llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
}

// XL Compat intrinsics.
let TargetPrefix = "ppc" in {
  def int_ppc_dcbfl : GCCBuiltin<"__builtin_ppc_dcbfl">,
                      Intrinsic<[], [llvm_ptr_ty], [IntrArgMemOnly]>;
  def int_ppc_dcbflp : GCCBuiltin<"__builtin_ppc_dcbflp">,
                       Intrinsic<[], [llvm_ptr_ty], [IntrArgMemOnly]>;
  def int_ppc_dcbst : GCCBuiltin<"__builtin_ppc_dcbst">,
                      Intrinsic<[], [llvm_ptr_ty], []>;
  def int_ppc_dcbt  : GCCBuiltin<"__builtin_ppc_dcbt">,
                      Intrinsic<[], [llvm_ptr_ty],
    [IntrArgMemOnly, NoCapture<ArgIndex<0>>]>;
  def int_ppc_dcbtst : GCCBuiltin<"__builtin_ppc_dcbtst">,
                       Intrinsic<[], [llvm_ptr_ty],
    [IntrArgMemOnly, NoCapture<ArgIndex<0>>]>;
  def int_ppc_dcbz  : GCCBuiltin<"__builtin_ppc_dcbz">,
                      Intrinsic<[], [llvm_ptr_ty], []>;
  def int_ppc_icbt : GCCBuiltin<"__builtin_ppc_icbt">,
                     Intrinsic<[], [llvm_ptr_ty], []>;
  
  // Population Count in each Byte.
  def int_ppc_popcntb : GCCBuiltin<"__builtin_ppc_popcntb">,
                        Intrinsic<[llvm_i64_ty], [llvm_i64_ty], [IntrNoMem]>;
  
  // sync instruction (i.e. sync 0, a.k.a hwsync)
  def int_ppc_sync : GCCBuiltin<"__builtin_ppc_sync">,
                     Intrinsic<[], [], []>;
  def int_ppc_iospace_sync : GCCBuiltin<"__builtin_ppc_iospace_sync">,
                             Intrinsic<[], [], []>;
  // isync instruction
  def int_ppc_isync : GCCBuiltin<"__builtin_ppc_isync">,
                      Intrinsic<[], [], []>;
  // lwsync is sync 1
  def int_ppc_lwsync : GCCBuiltin<"__builtin_ppc_lwsync">,
                       Intrinsic<[], [], []>;
  def int_ppc_iospace_lwsync : GCCBuiltin<"__builtin_ppc_iospace_lwsync">,
                               Intrinsic<[], [], []>;
  // eieio instruction
  def int_ppc_eieio : GCCBuiltin<"__builtin_ppc_eieio">,
                      Intrinsic<[],[],[]>;
  def int_ppc_iospace_eieio : GCCBuiltin<"__builtin_ppc_iospace_eieio">,
                              Intrinsic<[],[],[]>;
}

