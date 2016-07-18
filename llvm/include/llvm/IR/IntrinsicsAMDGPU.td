//===- IntrinsicsAMDGPU.td - Defines AMDGPU intrinsics -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the R600-specific intrinsics.
//
//===----------------------------------------------------------------------===//

class AMDGPUReadPreloadRegisterIntrinsic
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

class AMDGPUReadPreloadRegisterIntrinsicNamed<string name>
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>, GCCBuiltin<name>;

let TargetPrefix = "r600" in {

multiclass AMDGPUReadPreloadRegisterIntrinsic_xyz {
  def _x : AMDGPUReadPreloadRegisterIntrinsic;
  def _y : AMDGPUReadPreloadRegisterIntrinsic;
  def _z : AMDGPUReadPreloadRegisterIntrinsic;
}

multiclass AMDGPUReadPreloadRegisterIntrinsic_xyz_named<string prefix> {
  def _x : AMDGPUReadPreloadRegisterIntrinsicNamed<!strconcat(prefix, "_x")>;
  def _y : AMDGPUReadPreloadRegisterIntrinsicNamed<!strconcat(prefix, "_y")>;
  def _z : AMDGPUReadPreloadRegisterIntrinsicNamed<!strconcat(prefix, "_z")>;
}

defm int_r600_read_global_size : AMDGPUReadPreloadRegisterIntrinsic_xyz_named
                                 <"__builtin_r600_read_global_size">;
defm int_r600_read_ngroups : AMDGPUReadPreloadRegisterIntrinsic_xyz_named
                             <"__builtin_r600_read_ngroups">;
defm int_r600_read_tgid : AMDGPUReadPreloadRegisterIntrinsic_xyz_named
                          <"__builtin_r600_read_tgid">;

defm int_r600_read_local_size : AMDGPUReadPreloadRegisterIntrinsic_xyz;
defm int_r600_read_tidig : AMDGPUReadPreloadRegisterIntrinsic_xyz;

def int_r600_read_workdim : AMDGPUReadPreloadRegisterIntrinsic;

def int_r600_group_barrier : GCCBuiltin<"__builtin_r600_group_barrier">,
  Intrinsic<[], [], [IntrConvergent]>;

// AS 7 is PARAM_I_ADDRESS, used for kernel arguments
def int_r600_implicitarg_ptr :
  GCCBuiltin<"__builtin_r600_implicitarg_ptr">,
  Intrinsic<[LLVMQualPointerType<llvm_i8_ty, 7>], [], [IntrNoMem]>;

def int_r600_rat_store_typed :
  // 1st parameter: Data
  // 2nd parameter: Index
  // 3rd parameter: Constant RAT ID
  Intrinsic<[], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_i32_ty], []>,
  GCCBuiltin<"__builtin_r600_rat_store_typed">;

def int_r600_recipsqrt_ieee :  Intrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]
>;

def int_r600_recipsqrt_clamped : Intrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]
>;

} // End TargetPrefix = "r600"

let TargetPrefix = "amdgcn" in {

defm int_amdgcn_workitem_id : AMDGPUReadPreloadRegisterIntrinsic_xyz;
defm int_amdgcn_workgroup_id : AMDGPUReadPreloadRegisterIntrinsic_xyz_named
                               <"__builtin_amdgcn_workgroup_id">;

def int_amdgcn_s_barrier : GCCBuiltin<"__builtin_amdgcn_s_barrier">,
  Intrinsic<[], [], [IntrConvergent]>;

def int_amdgcn_s_waitcnt : Intrinsic<[], [llvm_i32_ty], []>;

def int_amdgcn_div_scale : Intrinsic<
  // 1st parameter: Numerator
  // 2nd parameter: Denominator
  // 3rd parameter: Constant to select select between first and
  //                second. (0 = first, 1 = second).
  [llvm_anyfloat_ty, llvm_i1_ty],
  [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i1_ty],
  [IntrNoMem]
>;

def int_amdgcn_div_fmas : Intrinsic<[llvm_anyfloat_ty],
  [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>, llvm_i1_ty],
  [IntrNoMem]
>;

def int_amdgcn_div_fixup : Intrinsic<[llvm_anyfloat_ty],
  [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
  [IntrNoMem]
>;

def int_amdgcn_trig_preop : Intrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>, llvm_i32_ty], [IntrNoMem]
>;

def int_amdgcn_sin : Intrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]
>;

def int_amdgcn_cos : Intrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]
>;

def int_amdgcn_log_clamp : Intrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]
>;

def int_amdgcn_rcp : Intrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]
>;

def int_amdgcn_rsq :  Intrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]
>;

def int_amdgcn_rsq_legacy :  GCCBuiltin<"__builtin_amdgcn_rsq_legacy">,
  Intrinsic<
  [llvm_float_ty], [llvm_float_ty], [IntrNoMem]
>;

def int_amdgcn_rsq_clamp : Intrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;

def int_amdgcn_ldexp : Intrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>, llvm_i32_ty], [IntrNoMem]
>;

def int_amdgcn_frexp_mant : Intrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]
>;

def int_amdgcn_frexp_exp : Intrinsic<
  [llvm_i32_ty], [llvm_anyfloat_ty], [IntrNoMem]
>;

// v_fract is buggy on SI/CI. It mishandles infinities, may return 1.0
// and always uses rtz, so is not suitable for implementing the OpenCL
// fract function. It should be ok on VI.
def int_amdgcn_fract : Intrinsic<
  [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]
>;

def int_amdgcn_class : Intrinsic<
  [llvm_i1_ty], [llvm_anyfloat_ty, llvm_i32_ty], [IntrNoMem]
>;

def int_amdgcn_cubeid : GCCBuiltin<"__builtin_amdgcn_cubeid">,
  Intrinsic<[llvm_float_ty],
    [llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]
>;

def int_amdgcn_cubema : GCCBuiltin<"__builtin_amdgcn_cubema">,
  Intrinsic<[llvm_float_ty],
  [llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]
>;

def int_amdgcn_cubesc : GCCBuiltin<"__builtin_amdgcn_cubesc">,
  Intrinsic<[llvm_float_ty],
    [llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]
>;

def int_amdgcn_cubetc : GCCBuiltin<"__builtin_amdgcn_cubetc">,
  Intrinsic<[llvm_float_ty],
    [llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]
>;

// TODO: Do we want an ordering for these?
def int_amdgcn_atomic_inc : Intrinsic<[llvm_anyint_ty],
  [llvm_anyptr_ty, LLVMMatchType<0>],
  [IntrArgMemOnly, NoCapture<0>]
>;

def int_amdgcn_atomic_dec : Intrinsic<[llvm_anyint_ty],
  [llvm_anyptr_ty, LLVMMatchType<0>],
  [IntrArgMemOnly, NoCapture<0>]
>;

class AMDGPUImageLoad : Intrinsic <
  [llvm_v4f32_ty],    // vdata(VGPR)
  [llvm_anyint_ty,    // vaddr(VGPR)
   llvm_v8i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // dmask(imm)
   llvm_i1_ty,        // r128(imm)
   llvm_i1_ty,        // da(imm)
   llvm_i1_ty,        // glc(imm)
   llvm_i1_ty],       // slc(imm)
  [IntrReadMem]>;

def int_amdgcn_image_load : AMDGPUImageLoad;
def int_amdgcn_image_load_mip : AMDGPUImageLoad;

class AMDGPUImageStore : Intrinsic <
  [],
  [llvm_v4f32_ty,     // vdata(VGPR)
   llvm_anyint_ty,    // vaddr(VGPR)
   llvm_v8i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // dmask(imm)
   llvm_i1_ty,        // r128(imm)
   llvm_i1_ty,        // da(imm)
   llvm_i1_ty,        // glc(imm)
   llvm_i1_ty],       // slc(imm)
  []>;

def int_amdgcn_image_store : AMDGPUImageStore;
def int_amdgcn_image_store_mip : AMDGPUImageStore;

class AMDGPUImageAtomic : Intrinsic <
  [llvm_i32_ty],
  [llvm_i32_ty,       // vdata(VGPR)
   llvm_anyint_ty,    // vaddr(VGPR)
   llvm_v8i32_ty,     // rsrc(SGPR)
   llvm_i1_ty,        // r128(imm)
   llvm_i1_ty,        // da(imm)
   llvm_i1_ty],       // slc(imm)
  []>;

def int_amdgcn_image_atomic_swap : AMDGPUImageAtomic;
def int_amdgcn_image_atomic_add : AMDGPUImageAtomic;
def int_amdgcn_image_atomic_sub : AMDGPUImageAtomic;
def int_amdgcn_image_atomic_smin : AMDGPUImageAtomic;
def int_amdgcn_image_atomic_umin : AMDGPUImageAtomic;
def int_amdgcn_image_atomic_smax : AMDGPUImageAtomic;
def int_amdgcn_image_atomic_umax : AMDGPUImageAtomic;
def int_amdgcn_image_atomic_and : AMDGPUImageAtomic;
def int_amdgcn_image_atomic_or : AMDGPUImageAtomic;
def int_amdgcn_image_atomic_xor : AMDGPUImageAtomic;
def int_amdgcn_image_atomic_inc : AMDGPUImageAtomic;
def int_amdgcn_image_atomic_dec : AMDGPUImageAtomic;
def int_amdgcn_image_atomic_cmpswap : Intrinsic <
  [llvm_i32_ty],
  [llvm_i32_ty,       // src(VGPR)
   llvm_i32_ty,       // cmp(VGPR)
   llvm_anyint_ty,    // vaddr(VGPR)
   llvm_v8i32_ty,     // rsrc(SGPR)
   llvm_i1_ty,        // r128(imm)
   llvm_i1_ty,        // da(imm)
   llvm_i1_ty],       // slc(imm)
  []>;

class AMDGPUBufferLoad : Intrinsic <
  [llvm_anyfloat_ty],
  [llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(SGPR/VGPR/imm)
   llvm_i1_ty,        // glc(imm)
   llvm_i1_ty],       // slc(imm)
  [IntrReadMem]>;
def int_amdgcn_buffer_load_format : AMDGPUBufferLoad;
def int_amdgcn_buffer_load : AMDGPUBufferLoad;

class AMDGPUBufferStore : Intrinsic <
  [],
  [llvm_anyfloat_ty,  // vdata(VGPR) -- can currently only select f32, v2f32, v4f32
   llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(SGPR/VGPR/imm)
   llvm_i1_ty,        // glc(imm)
   llvm_i1_ty],       // slc(imm)
  [IntrWriteMem]>;
def int_amdgcn_buffer_store_format : AMDGPUBufferStore;
def int_amdgcn_buffer_store : AMDGPUBufferStore;

class AMDGPUBufferAtomic : Intrinsic <
  [llvm_i32_ty],
  [llvm_i32_ty,       // vdata(VGPR)
   llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(SGPR/VGPR/imm)
   llvm_i1_ty],       // slc(imm)
  []>;
def int_amdgcn_buffer_atomic_swap : AMDGPUBufferAtomic;
def int_amdgcn_buffer_atomic_add : AMDGPUBufferAtomic;
def int_amdgcn_buffer_atomic_sub : AMDGPUBufferAtomic;
def int_amdgcn_buffer_atomic_smin : AMDGPUBufferAtomic;
def int_amdgcn_buffer_atomic_umin : AMDGPUBufferAtomic;
def int_amdgcn_buffer_atomic_smax : AMDGPUBufferAtomic;
def int_amdgcn_buffer_atomic_umax : AMDGPUBufferAtomic;
def int_amdgcn_buffer_atomic_and : AMDGPUBufferAtomic;
def int_amdgcn_buffer_atomic_or : AMDGPUBufferAtomic;
def int_amdgcn_buffer_atomic_xor : AMDGPUBufferAtomic;
def int_amdgcn_buffer_atomic_cmpswap : Intrinsic<
  [llvm_i32_ty],
  [llvm_i32_ty,       // src(VGPR)
   llvm_i32_ty,       // cmp(VGPR)
   llvm_v4i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(SGPR/VGPR/imm)
   llvm_i1_ty],       // slc(imm)
  []>;

def int_amdgcn_read_workdim : AMDGPUReadPreloadRegisterIntrinsic;


def int_amdgcn_buffer_wbinvl1_sc :
  GCCBuiltin<"__builtin_amdgcn_buffer_wbinvl1_sc">,
  Intrinsic<[], [], []>;

def int_amdgcn_buffer_wbinvl1 :
  GCCBuiltin<"__builtin_amdgcn_buffer_wbinvl1">,
  Intrinsic<[], [], []>;

def int_amdgcn_s_dcache_inv :
  GCCBuiltin<"__builtin_amdgcn_s_dcache_inv">,
  Intrinsic<[], [], []>;

def int_amdgcn_s_memtime :
  GCCBuiltin<"__builtin_amdgcn_s_memtime">,
  Intrinsic<[llvm_i64_ty], [], []>;

def int_amdgcn_s_sleep :
  GCCBuiltin<"__builtin_amdgcn_s_sleep">,
  Intrinsic<[], [llvm_i32_ty], []> {
}

def int_amdgcn_s_getreg :
  GCCBuiltin<"__builtin_amdgcn_s_getreg">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrReadMem]>;

def int_amdgcn_groupstaticsize :
  GCCBuiltin<"__builtin_amdgcn_groupstaticsize">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_amdgcn_dispatch_ptr :
  GCCBuiltin<"__builtin_amdgcn_dispatch_ptr">,
  Intrinsic<[LLVMQualPointerType<llvm_i8_ty, 2>], [], [IntrNoMem]>;

def int_amdgcn_queue_ptr :
  GCCBuiltin<"__builtin_amdgcn_queue_ptr">,
  Intrinsic<[LLVMQualPointerType<llvm_i8_ty, 2>], [], [IntrNoMem]>;

def int_amdgcn_kernarg_segment_ptr :
  GCCBuiltin<"__builtin_amdgcn_kernarg_segment_ptr">,
  Intrinsic<[LLVMQualPointerType<llvm_i8_ty, 2>], [], [IntrNoMem]>;

def int_amdgcn_implicitarg_ptr :
  GCCBuiltin<"__builtin_amdgcn_implicitarg_ptr">,
  Intrinsic<[LLVMQualPointerType<llvm_i8_ty, 2>], [], [IntrNoMem]>;

// __builtin_amdgcn_interp_p1 <i>, <attr_chan>, <attr>, <m0>
def int_amdgcn_interp_p1 :
  GCCBuiltin<"__builtin_amdgcn_interp_p1">,
  Intrinsic<[llvm_float_ty],
            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem]>;  // This intrinsic reads from lds, but the memory
                           // values are constant, so it behaves like IntrNoMem.

// __builtin_amdgcn_interp_p2 <p1>, <j>, <attr_chan>, <attr>, <m0>
def int_amdgcn_interp_p2 :
  GCCBuiltin<"__builtin_amdgcn_interp_p2">,
  Intrinsic<[llvm_float_ty],
            [llvm_float_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem]>;  // See int_amdgcn_v_interp_p1 for why this is
                           // IntrNoMem.

// Pixel shaders only: whether the current pixel is live (i.e. not a helper
// invocation for derivative computation).
def int_amdgcn_ps_live : Intrinsic <
  [llvm_i1_ty],
  [],
  [IntrNoMem]>;

def int_amdgcn_mbcnt_lo :
  GCCBuiltin<"__builtin_amdgcn_mbcnt_lo">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

def int_amdgcn_mbcnt_hi :
  GCCBuiltin<"__builtin_amdgcn_mbcnt_hi">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

// llvm.amdgcn.ds.swizzle src offset
def int_amdgcn_ds_swizzle :
  GCCBuiltin<"__builtin_amdgcn_ds_swizzle">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem, IntrConvergent]>;

// llvm.amdgcn.lerp
def int_amdgcn_lerp :
  GCCBuiltin<"__builtin_amdgcn_lerp">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

//===----------------------------------------------------------------------===//
// CI+ Intrinsics
//===----------------------------------------------------------------------===//

def int_amdgcn_s_dcache_inv_vol :
  GCCBuiltin<"__builtin_amdgcn_s_dcache_inv_vol">,
  Intrinsic<[], [], []>;

def int_amdgcn_buffer_wbinvl1_vol :
  GCCBuiltin<"__builtin_amdgcn_buffer_wbinvl1_vol">,
  Intrinsic<[], [], []>;

//===----------------------------------------------------------------------===//
// VI Intrinsics
//===----------------------------------------------------------------------===//

// llvm.amdgcn.mov.dpp.i32 <src> <dpp_ctrl> <row_mask> <bank_mask> <bound_ctrl>
def int_amdgcn_mov_dpp :
  Intrinsic<[llvm_anyint_ty],
            [LLVMMatchType<0>, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
             llvm_i1_ty], [IntrNoMem, IntrConvergent]>;

def int_amdgcn_s_dcache_wb :
  GCCBuiltin<"__builtin_amdgcn_s_dcache_wb">,
  Intrinsic<[], [], []>;

def int_amdgcn_s_dcache_wb_vol :
  GCCBuiltin<"__builtin_amdgcn_s_dcache_wb_vol">,
  Intrinsic<[], [], []>;

def int_amdgcn_s_memrealtime :
  GCCBuiltin<"__builtin_amdgcn_s_memrealtime">,
  Intrinsic<[llvm_i64_ty], [], []>;

// llvm.amdgcn.ds.permute <index> <src>
def int_amdgcn_ds_permute :
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem, IntrConvergent]>;

// llvm.amdgcn.ds.bpermute <index> <src>
def int_amdgcn_ds_bpermute :
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem, IntrConvergent]>;

}
