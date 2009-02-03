//===- IntrinsicsPowerPC.td - Defines PowerPC intrinsics ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  def int_ppc_dcba  : Intrinsic<[llvm_void_ty], [llvm_ptr_ty], [IntrWriteMem]>;
  def int_ppc_dcbf  : Intrinsic<[llvm_void_ty], [llvm_ptr_ty], [IntrWriteMem]>;
  def int_ppc_dcbi  : Intrinsic<[llvm_void_ty], [llvm_ptr_ty], [IntrWriteMem]>;
  def int_ppc_dcbst : Intrinsic<[llvm_void_ty], [llvm_ptr_ty], [IntrWriteMem]>;
  def int_ppc_dcbt  : Intrinsic<[llvm_void_ty], [llvm_ptr_ty], [IntrWriteMem]>;
  def int_ppc_dcbtst: Intrinsic<[llvm_void_ty], [llvm_ptr_ty], [IntrWriteMem]>;
  def int_ppc_dcbz  : Intrinsic<[llvm_void_ty], [llvm_ptr_ty], [IntrWriteMem]>;
  def int_ppc_dcbzl : Intrinsic<[llvm_void_ty], [llvm_ptr_ty], [IntrWriteMem]>;

  // sync instruction
  def int_ppc_sync : Intrinsic<[llvm_void_ty], [], [IntrWriteMem]>;
}


let TargetPrefix = "ppc" in {  // All PPC intrinsics start with "llvm.ppc.".
  /// PowerPC_Vec_Intrinsic - Base class for all altivec intrinsics.
  class PowerPC_Vec_Intrinsic<string GCCIntSuffix, list<LLVMType> ret_types,
                              list<LLVMType> param_types,
                              list<IntrinsicProperty> properties>
    : GCCBuiltin<!strconcat("__builtin_altivec_", GCCIntSuffix)>,
      Intrinsic<ret_types, param_types, properties>;
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

/// PowerPC_Vec_BBB_Intrinsic - A PowerPC intrinsic that takes two v16f8
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


//===----------------------------------------------------------------------===//
// PowerPC Altivec Intrinsic Definitions.

let TargetPrefix = "ppc" in {  // All intrinsics start with "llvm.ppc.".
  // Data Stream Control.
  def int_ppc_altivec_dss : GCCBuiltin<"__builtin_altivec_dss">,
              Intrinsic<[llvm_void_ty], [llvm_i32_ty], [IntrWriteMem]>;
  def int_ppc_altivec_dssall : GCCBuiltin<"__builtin_altivec_dssall">,
              Intrinsic<[llvm_void_ty], [], [IntrWriteMem]>;
  def int_ppc_altivec_dst : GCCBuiltin<"__builtin_altivec_dst">,
              Intrinsic<[llvm_void_ty],
                        [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
                        [IntrWriteMem]>;
  def int_ppc_altivec_dstt : GCCBuiltin<"__builtin_altivec_dstt">,
              Intrinsic<[llvm_void_ty],
                        [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
                        [IntrWriteMem]>;
  def int_ppc_altivec_dstst : GCCBuiltin<"__builtin_altivec_dstst">,
              Intrinsic<[llvm_void_ty],
                        [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
                        [IntrWriteMem]>;
  def int_ppc_altivec_dststt : GCCBuiltin<"__builtin_altivec_dststt">,
              Intrinsic<[llvm_void_ty],
                        [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
                        [IntrWriteMem]>;

  // VSCR access.
  def int_ppc_altivec_mfvscr : GCCBuiltin<"__builtin_altivec_mfvscr">,
              Intrinsic<[llvm_v8i16_ty], [], [IntrReadMem]>;
  def int_ppc_altivec_mtvscr : GCCBuiltin<"__builtin_altivec_mtvscr">,
              Intrinsic<[llvm_void_ty], [llvm_v4i32_ty], [IntrWriteMem]>;


  // Loads.  These don't map directly to GCC builtins because they represent the
  // source address with a single pointer.
  def int_ppc_altivec_lvx :
              Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], [IntrReadMem]>;
  def int_ppc_altivec_lvxl :
              Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], [IntrReadMem]>;
  def int_ppc_altivec_lvebx :
              Intrinsic<[llvm_v16i8_ty], [llvm_ptr_ty], [IntrReadMem]>;
  def int_ppc_altivec_lvehx :
              Intrinsic<[llvm_v8i16_ty], [llvm_ptr_ty], [IntrReadMem]>;
  def int_ppc_altivec_lvewx :
              Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], [IntrReadMem]>;

  // Stores.  These don't map directly to GCC builtins because they represent the
  // source address with a single pointer.
  def int_ppc_altivec_stvx :
              Intrinsic<[llvm_void_ty], [llvm_v4i32_ty, llvm_ptr_ty],
                        [IntrWriteMem]>;
  def int_ppc_altivec_stvxl :
              Intrinsic<[llvm_void_ty], [llvm_v4i32_ty, llvm_ptr_ty],
                        [IntrWriteMem]>;
  def int_ppc_altivec_stvebx :
              Intrinsic<[llvm_void_ty], [llvm_v16i8_ty, llvm_ptr_ty],
                        [IntrWriteMem]>;
  def int_ppc_altivec_stvehx :
              Intrinsic<[llvm_void_ty], [llvm_v8i16_ty, llvm_ptr_ty],
                        [IntrWriteMem]>;
  def int_ppc_altivec_stvewx :
              Intrinsic<[llvm_void_ty], [llvm_v4i32_ty, llvm_ptr_ty],
                        [IntrWriteMem]>;

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
                        
  def int_ppc_altivec_vcmpequw : GCCBuiltin<"__builtin_altivec_vcmpequw">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtsw : GCCBuiltin<"__builtin_altivec_vcmpgtsw">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtuw : GCCBuiltin<"__builtin_altivec_vcmpgtuw">,
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

  def int_ppc_altivec_vcmpequb : GCCBuiltin<"__builtin_altivec_vcmpequb">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtsb : GCCBuiltin<"__builtin_altivec_vcmpgtsb">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtub : GCCBuiltin<"__builtin_altivec_vcmpgtub">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
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
                        
  def int_ppc_altivec_vcmpequw_p : GCCBuiltin<"__builtin_altivec_vcmpequw_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v4i32_ty,llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtsw_p : GCCBuiltin<"__builtin_altivec_vcmpgtsw_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v4i32_ty,llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtuw_p : GCCBuiltin<"__builtin_altivec_vcmpgtuw_p">,
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

  def int_ppc_altivec_vcmpequb_p : GCCBuiltin<"__builtin_altivec_vcmpequb_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v16i8_ty,llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtsb_p : GCCBuiltin<"__builtin_altivec_vcmpgtsb_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v16i8_ty,llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcmpgtub_p : GCCBuiltin<"__builtin_altivec_vcmpgtub_p">,
              Intrinsic<[llvm_i32_ty],[llvm_i32_ty,llvm_v16i8_ty,llvm_v16i8_ty],
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
def int_ppc_altivec_vmaxub : PowerPC_Vec_BBB_Intrinsic<"vmaxub">;
def int_ppc_altivec_vmaxuh : PowerPC_Vec_HHH_Intrinsic<"vmaxuh">;
def int_ppc_altivec_vmaxuw : PowerPC_Vec_WWW_Intrinsic<"vmaxuw">;

// Vector minimum.
def int_ppc_altivec_vminfp : PowerPC_Vec_FFF_Intrinsic<"vminfp">;
def int_ppc_altivec_vminsb : PowerPC_Vec_BBB_Intrinsic<"vminsb">;
def int_ppc_altivec_vminsh : PowerPC_Vec_HHH_Intrinsic<"vminsh">;
def int_ppc_altivec_vminsw : PowerPC_Vec_WWW_Intrinsic<"vminsw">;
def int_ppc_altivec_vminub : PowerPC_Vec_BBB_Intrinsic<"vminub">;
def int_ppc_altivec_vminuh : PowerPC_Vec_HHH_Intrinsic<"vminuh">;
def int_ppc_altivec_vminuw : PowerPC_Vec_WWW_Intrinsic<"vminuw">;

// Saturating adds.
def int_ppc_altivec_vaddubs : PowerPC_Vec_BBB_Intrinsic<"vaddubs">;
def int_ppc_altivec_vaddsbs : PowerPC_Vec_BBB_Intrinsic<"vaddsbs">;
def int_ppc_altivec_vadduhs : PowerPC_Vec_HHH_Intrinsic<"vadduhs">;
def int_ppc_altivec_vaddshs : PowerPC_Vec_HHH_Intrinsic<"vaddshs">;
def int_ppc_altivec_vadduws : PowerPC_Vec_WWW_Intrinsic<"vadduws">;
def int_ppc_altivec_vaddsws : PowerPC_Vec_WWW_Intrinsic<"vaddsws">;
def int_ppc_altivec_vaddcuw : PowerPC_Vec_WWW_Intrinsic<"vaddcuw">;

// Saturating subs.
def int_ppc_altivec_vsububs : PowerPC_Vec_BBB_Intrinsic<"vsububs">;
def int_ppc_altivec_vsubsbs : PowerPC_Vec_BBB_Intrinsic<"vsubsbs">;
def int_ppc_altivec_vsubuhs : PowerPC_Vec_HHH_Intrinsic<"vsubuhs">;
def int_ppc_altivec_vsubshs : PowerPC_Vec_HHH_Intrinsic<"vsubshs">;
def int_ppc_altivec_vsubuws : PowerPC_Vec_WWW_Intrinsic<"vsubuws">;
def int_ppc_altivec_vsubsws : PowerPC_Vec_WWW_Intrinsic<"vsubsws">;
def int_ppc_altivec_vsubcuw : PowerPC_Vec_WWW_Intrinsic<"vsubcuw">;

let TargetPrefix = "ppc" in {  // All PPC intrinsics start with "llvm.ppc.".
  // Saturating multiply-adds.
  def int_ppc_altivec_vmhaddshs : GCCBuiltin<"__builtin_altivec_vmhaddshs">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty, llvm_v8i16_ty], [IntrNoMem]>;
  def int_ppc_altivec_vmhraddshs : GCCBuiltin<"__builtin_altivec_vmhraddshs">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty, llvm_v8i16_ty], [IntrNoMem]>;

  def int_ppc_altivec_vmaddfp : GCCBuiltin<"__builtin_altivec_vmaddfp">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_v4f32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vnmsubfp : GCCBuiltin<"__builtin_altivec_vnmsubfp">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_v4f32_ty], [IntrNoMem]>;

  // Vector Multiply Sum Intructions.
  def int_ppc_altivec_vmsummbm : GCCBuiltin<"__builtin_altivec_vmsummbm">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v16i8_ty, llvm_v16i8_ty,
                       llvm_v4i32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vmsumshm : GCCBuiltin<"__builtin_altivec_vmsumshm">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                       llvm_v4i32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vmsumshs : GCCBuiltin<"__builtin_altivec_vmsumshs">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty, 
                       llvm_v4i32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vmsumubm : GCCBuiltin<"__builtin_altivec_vmsumubm">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v16i8_ty, llvm_v16i8_ty, 
                       llvm_v4i32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vmsumuhm : GCCBuiltin<"__builtin_altivec_vmsumuhm">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                       llvm_v4i32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vmsumuhs : GCCBuiltin<"__builtin_altivec_vmsumuhs">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                       llvm_v4i32_ty], [IntrNoMem]>;

  // Vector Multiply Intructions.
  def int_ppc_altivec_vmulesb : GCCBuiltin<"__builtin_altivec_vmulesb">,
          Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmulesh : GCCBuiltin<"__builtin_altivec_vmulesh">,
          Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmuleub : GCCBuiltin<"__builtin_altivec_vmuleub">,
          Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmuleuh : GCCBuiltin<"__builtin_altivec_vmuleuh">,
          Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                    [IntrNoMem]>;

  def int_ppc_altivec_vmulosb : GCCBuiltin<"__builtin_altivec_vmulosb">,
          Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmulosh : GCCBuiltin<"__builtin_altivec_vmulosh">,
          Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmuloub : GCCBuiltin<"__builtin_altivec_vmuloub">,
          Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                    [IntrNoMem]>;
  def int_ppc_altivec_vmulouh : GCCBuiltin<"__builtin_altivec_vmulouh">,
          Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                    [IntrNoMem]>;

  // Vector Sum Intructions.
  def int_ppc_altivec_vsumsws : GCCBuiltin<"__builtin_altivec_vsumsws">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                      [IntrNoMem]>;
  def int_ppc_altivec_vsum2sws : GCCBuiltin<"__builtin_altivec_vsum2sws">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                      [IntrNoMem]>;
  def int_ppc_altivec_vsum4sbs : GCCBuiltin<"__builtin_altivec_vsum4sbs">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v16i8_ty, llvm_v4i32_ty],
                      [IntrNoMem]>;
  def int_ppc_altivec_vsum4shs : GCCBuiltin<"__builtin_altivec_vsum4shs">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty, llvm_v4i32_ty],
                      [IntrNoMem]>;
  def int_ppc_altivec_vsum4ubs : GCCBuiltin<"__builtin_altivec_vsum4ubs">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v16i8_ty, llvm_v4i32_ty],
                      [IntrNoMem]>;

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
                      [IntrNoMem]>;
  def int_ppc_altivec_vpkshus : GCCBuiltin<"__builtin_altivec_vpkshus">,
            Intrinsic<[llvm_v16i8_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                      [IntrNoMem]>;
  def int_ppc_altivec_vpkswss : GCCBuiltin<"__builtin_altivec_vpkswss">,
            Intrinsic<[llvm_v16i8_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                      [IntrNoMem]>;
  def int_ppc_altivec_vpkswus : GCCBuiltin<"__builtin_altivec_vpkswus">,
            Intrinsic<[llvm_v8i16_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                      [IntrNoMem]>;
  // vpkuhum is lowered to a shuffle.
  def int_ppc_altivec_vpkuhus : GCCBuiltin<"__builtin_altivec_vpkuhus">,
            Intrinsic<[llvm_v16i8_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                      [IntrNoMem]>;
  // vpkuwum is lowered to a shuffle.
  def int_ppc_altivec_vpkuwus : GCCBuiltin<"__builtin_altivec_vpkuwus">,
            Intrinsic<[llvm_v8i16_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                      [IntrNoMem]>;

  // Unpacks.
  def int_ppc_altivec_vupkhpx : GCCBuiltin<"__builtin_altivec_vupkhpx">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_ppc_altivec_vupkhsb : GCCBuiltin<"__builtin_altivec_vupkhsb">,
            Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_ppc_altivec_vupkhsh : GCCBuiltin<"__builtin_altivec_vupkhsh">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_ppc_altivec_vupklpx : GCCBuiltin<"__builtin_altivec_vupklpx">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_ppc_altivec_vupklsb : GCCBuiltin<"__builtin_altivec_vupklsb">,
            Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_ppc_altivec_vupklsh : GCCBuiltin<"__builtin_altivec_vupklsh">,
            Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty], [IntrNoMem]>;


  // FP <-> integer conversion.
  def int_ppc_altivec_vcfsx : GCCBuiltin<"__builtin_altivec_vcfsx">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4i32_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vcfux : GCCBuiltin<"__builtin_altivec_vcfux">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4i32_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vctsxs : GCCBuiltin<"__builtin_altivec_vctsxs">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4f32_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_ppc_altivec_vctuxs : GCCBuiltin<"__builtin_altivec_vctuxs">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4f32_ty, llvm_i32_ty],
                        [IntrNoMem]>;

  def int_ppc_altivec_vrfim : GCCBuiltin<"__builtin_altivec_vrfim">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vrfin : GCCBuiltin<"__builtin_altivec_vrfin">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vrfip : GCCBuiltin<"__builtin_altivec_vrfip">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_ppc_altivec_vrfiz : GCCBuiltin<"__builtin_altivec_vrfiz">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
}

def int_ppc_altivec_vsl   : PowerPC_Vec_WWW_Intrinsic<"vsl">;
def int_ppc_altivec_vslo  : PowerPC_Vec_WWW_Intrinsic<"vslo">;

def int_ppc_altivec_vslb  : PowerPC_Vec_BBB_Intrinsic<"vslb">;
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
}

def int_ppc_altivec_vexptefp  : PowerPC_Vec_FF_Intrinsic<"vexptefp">;
def int_ppc_altivec_vlogefp   : PowerPC_Vec_FF_Intrinsic<"vlogefp">;
def int_ppc_altivec_vrefp     : PowerPC_Vec_FF_Intrinsic<"vrefp">;
def int_ppc_altivec_vrsqrtefp : PowerPC_Vec_FF_Intrinsic<"vrsqrtefp">;
