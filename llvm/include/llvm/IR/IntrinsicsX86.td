//===- IntrinsicsX86.td - Defines X86 intrinsics -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the X86-specific intrinsics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Interrupt traps
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_int : Intrinsic<[], [llvm_i8_ty]>;
}

//===----------------------------------------------------------------------===//
// SEH intrinsics for Windows
let TargetPrefix = "x86" in {
  def int_x86_seh_lsda : Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem]>;

  // Marks the EH registration node created in LLVM IR prior to code generation.
  def int_x86_seh_ehregnode : Intrinsic<[], [llvm_ptr_ty], []>;

  // Marks the EH guard slot node created in LLVM IR prior to code generation.
  def int_x86_seh_ehguard : Intrinsic<[], [llvm_ptr_ty], []>;

  // Given a pointer to the end of an EH registration object, returns the true
  // parent frame address that can be used with llvm.localrecover.
  def int_x86_seh_recoverfp : Intrinsic<[llvm_ptr_ty],
                                        [llvm_ptr_ty, llvm_ptr_ty],
                                        [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// FLAGS.
let TargetPrefix = "x86" in {
  def int_x86_flags_read_u32 : GCCBuiltin<"__builtin_ia32_readeflags_u32">,
        Intrinsic<[llvm_i32_ty], [], []>;
  def int_x86_flags_read_u64 : GCCBuiltin<"__builtin_ia32_readeflags_u64">,
        Intrinsic<[llvm_i64_ty], [], []>;
  def int_x86_flags_write_u32 : GCCBuiltin<"__builtin_ia32_writeeflags_u32">,
        Intrinsic<[], [llvm_i32_ty], []>;
  def int_x86_flags_write_u64 : GCCBuiltin<"__builtin_ia32_writeeflags_u64">,
        Intrinsic<[], [llvm_i64_ty], []>;
}

//===----------------------------------------------------------------------===//
// Read Time Stamp Counter.
let TargetPrefix = "x86" in {
  def int_x86_rdtsc : GCCBuiltin<"__builtin_ia32_rdtsc">,
              Intrinsic<[llvm_i64_ty], [], []>;
  def int_x86_rdtscp : GCCBuiltin<"__builtin_ia32_rdtscp">,
              Intrinsic<[llvm_i64_ty], [llvm_ptr_ty], [IntrArgMemOnly]>;
}

// Read Performance-Monitoring Counter.
let TargetPrefix = "x86" in {
  def int_x86_rdpmc : GCCBuiltin<"__builtin_ia32_rdpmc">,
              Intrinsic<[llvm_i64_ty], [llvm_i32_ty], []>;
}

//===----------------------------------------------------------------------===//
// 3DNow!

let TargetPrefix = "x86" in {
  def int_x86_3dnow_pavgusb : GCCBuiltin<"__builtin_ia32_pavgusb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pf2id : GCCBuiltin<"__builtin_ia32_pf2id">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_3dnow_pfacc : GCCBuiltin<"__builtin_ia32_pfacc">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfadd : GCCBuiltin<"__builtin_ia32_pfadd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfcmpeq : GCCBuiltin<"__builtin_ia32_pfcmpeq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfcmpge : GCCBuiltin<"__builtin_ia32_pfcmpge">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfcmpgt : GCCBuiltin<"__builtin_ia32_pfcmpgt">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfmax : GCCBuiltin<"__builtin_ia32_pfmax">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfmin : GCCBuiltin<"__builtin_ia32_pfmin">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfmul : GCCBuiltin<"__builtin_ia32_pfmul">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfrcp : GCCBuiltin<"__builtin_ia32_pfrcp">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_3dnow_pfrcpit1 : GCCBuiltin<"__builtin_ia32_pfrcpit1">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfrcpit2 : GCCBuiltin<"__builtin_ia32_pfrcpit2">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfrsqrt : GCCBuiltin<"__builtin_ia32_pfrsqrt">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_3dnow_pfrsqit1 : GCCBuiltin<"__builtin_ia32_pfrsqit1">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfsub : GCCBuiltin<"__builtin_ia32_pfsub">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pfsubr : GCCBuiltin<"__builtin_ia32_pfsubr">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnow_pi2fd : GCCBuiltin<"__builtin_ia32_pi2fd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_3dnow_pmulhrw : GCCBuiltin<"__builtin_ia32_pmulhrw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// 3DNow! extensions

let TargetPrefix = "x86" in {
  def int_x86_3dnowa_pf2iw : GCCBuiltin<"__builtin_ia32_pf2iw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_3dnowa_pfnacc : GCCBuiltin<"__builtin_ia32_pfnacc">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnowa_pfpnacc : GCCBuiltin<"__builtin_ia32_pfpnacc">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_3dnowa_pi2fw : GCCBuiltin<"__builtin_ia32_pi2fw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_3dnowa_pswapd :
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// SSE1

// Arithmetic ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_sqrt_ss : GCCBuiltin<"__builtin_ia32_sqrtss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_sqrt_ps : GCCBuiltin<"__builtin_ia32_sqrtps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_rcp_ss : GCCBuiltin<"__builtin_ia32_rcpss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_rcp_ps : GCCBuiltin<"__builtin_ia32_rcpps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_rsqrt_ss : GCCBuiltin<"__builtin_ia32_rsqrtss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_rsqrt_ps : GCCBuiltin<"__builtin_ia32_rsqrtps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_sse_min_ss : GCCBuiltin<"__builtin_ia32_minss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_min_ps : GCCBuiltin<"__builtin_ia32_minps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_max_ss : GCCBuiltin<"__builtin_ia32_maxss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_max_ps : GCCBuiltin<"__builtin_ia32_maxps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
}

// Comparison ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmp_ss : GCCBuiltin<"__builtin_ia32_cmpss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_sse_cmp_ps :
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_sse_comieq_ss : GCCBuiltin<"__builtin_ia32_comieq">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_comilt_ss : GCCBuiltin<"__builtin_ia32_comilt">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_comile_ss : GCCBuiltin<"__builtin_ia32_comile">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_comigt_ss : GCCBuiltin<"__builtin_ia32_comigt">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_comige_ss : GCCBuiltin<"__builtin_ia32_comige">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_comineq_ss : GCCBuiltin<"__builtin_ia32_comineq">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomieq_ss : GCCBuiltin<"__builtin_ia32_ucomieq">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomilt_ss : GCCBuiltin<"__builtin_ia32_ucomilt">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomile_ss : GCCBuiltin<"__builtin_ia32_ucomile">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomigt_ss : GCCBuiltin<"__builtin_ia32_ucomigt">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomige_ss : GCCBuiltin<"__builtin_ia32_ucomige">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_ucomineq_ss : GCCBuiltin<"__builtin_ia32_ucomineq">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
}


// Conversion ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cvtss2si : GCCBuiltin<"__builtin_ia32_cvtss2si">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvtss2si64 : GCCBuiltin<"__builtin_ia32_cvtss2si64">,
              Intrinsic<[llvm_i64_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvttss2si : GCCBuiltin<"__builtin_ia32_cvttss2si">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvttss2si64 : GCCBuiltin<"__builtin_ia32_cvttss2si64">,
              Intrinsic<[llvm_i64_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvtsi2ss : // TODO: Remove this intrinsic.
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse_cvtsi642ss : // TODO: Remove this intrinsic.
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_i64_ty], [IntrNoMem]>;

  def int_x86_sse_cvtps2pi : GCCBuiltin<"__builtin_ia32_cvtps2pi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvttps2pi: GCCBuiltin<"__builtin_ia32_cvttps2pi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvtpi2ps : GCCBuiltin<"__builtin_ia32_cvtpi2ps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
}

// Cacheability support ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_sfence : GCCBuiltin<"__builtin_ia32_sfence">,
              Intrinsic<[], [], []>;
}

// Control register.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_stmxcsr :
              Intrinsic<[], [llvm_ptr_ty], []>;
  def int_x86_sse_ldmxcsr :
              Intrinsic<[], [llvm_ptr_ty], []>;
}

// Misc.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_movmsk_ps : GCCBuiltin<"__builtin_ia32_movmskps">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// SSE2

// FP arithmetic ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_sqrt_sd : GCCBuiltin<"__builtin_ia32_sqrtsd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_sse2_sqrt_pd : GCCBuiltin<"__builtin_ia32_sqrtpd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_sse2_min_sd : GCCBuiltin<"__builtin_ia32_minsd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_min_pd : GCCBuiltin<"__builtin_ia32_minpd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_max_sd : GCCBuiltin<"__builtin_ia32_maxsd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_max_pd : GCCBuiltin<"__builtin_ia32_maxpd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
}

// FP comparison ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_cmp_sd : GCCBuiltin<"__builtin_ia32_cmpsd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_sse2_cmp_pd :
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_sse2_comieq_sd : GCCBuiltin<"__builtin_ia32_comisdeq">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_comilt_sd : GCCBuiltin<"__builtin_ia32_comisdlt">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_comile_sd : GCCBuiltin<"__builtin_ia32_comisdle">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_comigt_sd : GCCBuiltin<"__builtin_ia32_comisdgt">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_comige_sd : GCCBuiltin<"__builtin_ia32_comisdge">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_comineq_sd : GCCBuiltin<"__builtin_ia32_comisdneq">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomieq_sd : GCCBuiltin<"__builtin_ia32_ucomisdeq">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomilt_sd : GCCBuiltin<"__builtin_ia32_ucomisdlt">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomile_sd : GCCBuiltin<"__builtin_ia32_ucomisdle">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomigt_sd : GCCBuiltin<"__builtin_ia32_ucomisdgt">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomige_sd : GCCBuiltin<"__builtin_ia32_ucomisdge">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_ucomineq_sd : GCCBuiltin<"__builtin_ia32_ucomisdneq">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
}

// Integer arithmetic ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_padds_b : GCCBuiltin<"__builtin_ia32_paddsb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_padds_w : GCCBuiltin<"__builtin_ia32_paddsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_paddus_b : GCCBuiltin<"__builtin_ia32_paddusb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_paddus_w : GCCBuiltin<"__builtin_ia32_paddusw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_psubs_b : GCCBuiltin<"__builtin_ia32_psubsb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_sse2_psubs_w : GCCBuiltin<"__builtin_ia32_psubsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_psubus_b : GCCBuiltin<"__builtin_ia32_psubusb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_sse2_psubus_w : GCCBuiltin<"__builtin_ia32_psubusw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_pmulhu_w : GCCBuiltin<"__builtin_ia32_pmulhuw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pmulh_w : GCCBuiltin<"__builtin_ia32_pmulhw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pmulu_dq : GCCBuiltin<"__builtin_ia32_pmuludq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pmadd_wd : GCCBuiltin<"__builtin_ia32_pmaddwd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pavg_b : GCCBuiltin<"__builtin_ia32_pavgb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_pavg_w : GCCBuiltin<"__builtin_ia32_pavgw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_sse2_psad_bw : GCCBuiltin<"__builtin_ia32_psadbw128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem, Commutative]>;
}

// Integer shift ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_psll_w : GCCBuiltin<"__builtin_ia32_psllw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_psll_d : GCCBuiltin<"__builtin_ia32_pslld128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psll_q : GCCBuiltin<"__builtin_ia32_psllq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_v2i64_ty], [IntrNoMem]>;
  def int_x86_sse2_psrl_w : GCCBuiltin<"__builtin_ia32_psrlw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_psrl_d : GCCBuiltin<"__builtin_ia32_psrld128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psrl_q : GCCBuiltin<"__builtin_ia32_psrlq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_v2i64_ty], [IntrNoMem]>;
  def int_x86_sse2_psra_w : GCCBuiltin<"__builtin_ia32_psraw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_psra_d : GCCBuiltin<"__builtin_ia32_psrad128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;

  def int_x86_sse2_pslli_w : GCCBuiltin<"__builtin_ia32_psllwi128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_pslli_d : GCCBuiltin<"__builtin_ia32_pslldi128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_pslli_q : GCCBuiltin<"__builtin_ia32_psllqi128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psrli_w : GCCBuiltin<"__builtin_ia32_psrlwi128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psrli_d : GCCBuiltin<"__builtin_ia32_psrldi128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psrli_q : GCCBuiltin<"__builtin_ia32_psrlqi128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psrai_w : GCCBuiltin<"__builtin_ia32_psrawi128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_psrai_d : GCCBuiltin<"__builtin_ia32_psradi128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
}

// Conversion ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_cvtdq2ps : GCCBuiltin<"__builtin_ia32_cvtdq2ps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtpd2dq : GCCBuiltin<"__builtin_ia32_cvtpd2dq">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvttpd2dq : GCCBuiltin<"__builtin_ia32_cvttpd2dq">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtpd2ps : GCCBuiltin<"__builtin_ia32_cvtpd2ps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtps2dq : GCCBuiltin<"__builtin_ia32_cvtps2dq">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse2_cvttps2dq : GCCBuiltin<"__builtin_ia32_cvttps2dq">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtsd2si : GCCBuiltin<"__builtin_ia32_cvtsd2si">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtsd2si64 : GCCBuiltin<"__builtin_ia32_cvtsd2si64">,
              Intrinsic<[llvm_i64_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvttsd2si : GCCBuiltin<"__builtin_ia32_cvttsd2si">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvttsd2si64 : GCCBuiltin<"__builtin_ia32_cvttsd2si64">,
              Intrinsic<[llvm_i64_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtsi2sd : // TODO: Remove this intrinsic.
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtsi642sd : // TODO: Remove this intrinsic.
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_i64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtsd2ss : GCCBuiltin<"__builtin_ia32_cvtsd2ss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_cvtss2sd : // TODO: Remove this intrinsic.
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse_cvtpd2pi : GCCBuiltin<"__builtin_ia32_cvtpd2pi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse_cvttpd2pi: GCCBuiltin<"__builtin_ia32_cvttpd2pi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse_cvtpi2pd : GCCBuiltin<"__builtin_ia32_cvtpi2pd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
}

// Misc.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_packsswb_128 : GCCBuiltin<"__builtin_ia32_packsswb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_packssdw_128 : GCCBuiltin<"__builtin_ia32_packssdw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sse2_packuswb_128 : GCCBuiltin<"__builtin_ia32_packuswb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_sse2_movmsk_pd : GCCBuiltin<"__builtin_ia32_movmskpd">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse2_pmovmskb_128 : GCCBuiltin<"__builtin_ia32_pmovmskb128">,
              Intrinsic<[llvm_i32_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_sse2_maskmov_dqu : GCCBuiltin<"__builtin_ia32_maskmovdqu">,
              Intrinsic<[], [llvm_v16i8_ty,
                         llvm_v16i8_ty, llvm_ptr_ty], []>;
  def int_x86_sse2_clflush : GCCBuiltin<"__builtin_ia32_clflush">,
              Intrinsic<[], [llvm_ptr_ty], []>;
  def int_x86_sse2_lfence : GCCBuiltin<"__builtin_ia32_lfence">,
              Intrinsic<[], [], []>;
  def int_x86_sse2_mfence : GCCBuiltin<"__builtin_ia32_mfence">,
              Intrinsic<[], [], []>;
  def int_x86_sse2_pause : GCCBuiltin<"__builtin_ia32_pause">,
              Intrinsic<[], [], []>;
}

//===----------------------------------------------------------------------===//
// SSE3

// Addition / subtraction ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse3_addsub_ps : GCCBuiltin<"__builtin_ia32_addsubps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse3_addsub_pd : GCCBuiltin<"__builtin_ia32_addsubpd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
}

// Horizontal ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse3_hadd_ps : GCCBuiltin<"__builtin_ia32_haddps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse3_hadd_pd : GCCBuiltin<"__builtin_ia32_haddpd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_sse3_hsub_ps : GCCBuiltin<"__builtin_ia32_hsubps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_sse3_hsub_pd : GCCBuiltin<"__builtin_ia32_hsubpd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty], [IntrNoMem]>;
}

// Specialized unaligned load.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse3_ldu_dq : GCCBuiltin<"__builtin_ia32_lddqu">,
              Intrinsic<[llvm_v16i8_ty], [llvm_ptr_ty], [IntrReadMem]>;
}

// Thread synchronization ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse3_monitor : GCCBuiltin<"__builtin_ia32_monitor">,
              Intrinsic<[], [llvm_ptr_ty,
                         llvm_i32_ty, llvm_i32_ty], []>;
  def int_x86_sse3_mwait : GCCBuiltin<"__builtin_ia32_mwait">,
              Intrinsic<[], [llvm_i32_ty,
                         llvm_i32_ty], []>;
}

//===----------------------------------------------------------------------===//
// SSSE3

// Horizontal arithmetic ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_ssse3_phadd_w         : GCCBuiltin<"__builtin_ia32_phaddw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_phadd_w_128     : GCCBuiltin<"__builtin_ia32_phaddw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;

  def int_x86_ssse3_phadd_d         : GCCBuiltin<"__builtin_ia32_phaddd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_phadd_d_128     : GCCBuiltin<"__builtin_ia32_phaddd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;

  def int_x86_ssse3_phadd_sw        : GCCBuiltin<"__builtin_ia32_phaddsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_phadd_sw_128    : GCCBuiltin<"__builtin_ia32_phaddsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;

  def int_x86_ssse3_phsub_w         : GCCBuiltin<"__builtin_ia32_phsubw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_phsub_w_128     : GCCBuiltin<"__builtin_ia32_phsubw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;

  def int_x86_ssse3_phsub_d         : GCCBuiltin<"__builtin_ia32_phsubd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_phsub_d_128     : GCCBuiltin<"__builtin_ia32_phsubd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;

  def int_x86_ssse3_phsub_sw        : GCCBuiltin<"__builtin_ia32_phsubsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_phsub_sw_128    : GCCBuiltin<"__builtin_ia32_phsubsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;

  def int_x86_ssse3_pmadd_ub_sw     : GCCBuiltin<"__builtin_ia32_pmaddubsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_pmadd_ub_sw_128 : GCCBuiltin<"__builtin_ia32_pmaddubsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem]>;
}

// Packed multiply high with round and scale
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_ssse3_pmul_hr_sw      : GCCBuiltin<"__builtin_ia32_pmulhrsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_ssse3_pmul_hr_sw_128  : GCCBuiltin<"__builtin_ia32_pmulhrsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem, Commutative]>;
}

// Shuffle ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_ssse3_pshuf_b         : GCCBuiltin<"__builtin_ia32_pshufb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_pshuf_b_128     : GCCBuiltin<"__builtin_ia32_pshufb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_sse_pshuf_w           : GCCBuiltin<"__builtin_ia32_pshufw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_i8_ty],
                         [IntrNoMem]>;
}

// Sign ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_ssse3_psign_b         : GCCBuiltin<"__builtin_ia32_psignb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_psign_b_128     : GCCBuiltin<"__builtin_ia32_psignb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                         llvm_v16i8_ty], [IntrNoMem]>;

  def int_x86_ssse3_psign_w         : GCCBuiltin<"__builtin_ia32_psignw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_psign_w_128     : GCCBuiltin<"__builtin_ia32_psignw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;

  def int_x86_ssse3_psign_d         : GCCBuiltin<"__builtin_ia32_psignd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_psign_d_128     : GCCBuiltin<"__builtin_ia32_psignd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
}

// Absolute value ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_ssse3_pabs_b     : GCCBuiltin<"__builtin_ia32_pabsb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_pabs_b_128 : GCCBuiltin<"__builtin_ia32_pabsb128">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty], [IntrNoMem]>;

  def int_x86_ssse3_pabs_w     : GCCBuiltin<"__builtin_ia32_pabsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_pabs_w_128 : GCCBuiltin<"__builtin_ia32_pabsw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty], [IntrNoMem]>;

  def int_x86_ssse3_pabs_d     : GCCBuiltin<"__builtin_ia32_pabsd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_ssse3_pabs_d_128 : GCCBuiltin<"__builtin_ia32_pabsd128">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// SSE4.1

// FP rounding ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_round_ss        : GCCBuiltin<"__builtin_ia32_roundss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse41_round_ps        : GCCBuiltin<"__builtin_ia32_roundps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse41_round_sd        : GCCBuiltin<"__builtin_ia32_roundsd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_sse41_round_pd        : GCCBuiltin<"__builtin_ia32_roundpd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
}

// Vector min element
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_phminposuw     : GCCBuiltin<"__builtin_ia32_phminposuw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty],
                        [IntrNoMem]>;
}

// Advanced Encryption Standard (AES) Instructions
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_aesni_aesimc          : GCCBuiltin<"__builtin_ia32_aesimc128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_aesni_aesenc          : GCCBuiltin<"__builtin_ia32_aesenc128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_aesni_aesenclast : GCCBuiltin<"__builtin_ia32_aesenclast128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_aesni_aesdec          : GCCBuiltin<"__builtin_ia32_aesdec128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_aesni_aesdeclast : GCCBuiltin<"__builtin_ia32_aesdeclast128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_aesni_aeskeygenassist :
              GCCBuiltin<"__builtin_ia32_aeskeygenassist128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_i8_ty],
                        [IntrNoMem]>;
}

// PCLMUL instruction
let TargetPrefix = "x86" in { // All intrinsics start with "llvm.x86.".
  def int_x86_pclmulqdq : GCCBuiltin<"__builtin_ia32_pclmulqdq128">,
          Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty, llvm_i8_ty],
                    [IntrNoMem]>;
}

// Vector pack
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_packusdw        : GCCBuiltin<"__builtin_ia32_packusdw128">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
}

// Vector multiply
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_pmuldq          : GCCBuiltin<"__builtin_ia32_pmuldq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem, Commutative]>;
}

// Vector insert
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_insertps       : GCCBuiltin<"__builtin_ia32_insertps128">,
          Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_i8_ty],
                    [IntrNoMem]>;
}

// Vector blend
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_pblendvb         : GCCBuiltin<"__builtin_ia32_pblendvb128">,
        Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty,llvm_v16i8_ty],
                  [IntrNoMem]>;
  def int_x86_sse41_blendvpd         : GCCBuiltin<"__builtin_ia32_blendvpd">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,llvm_v2f64_ty],
                  [IntrNoMem]>;
  def int_x86_sse41_blendvps         : GCCBuiltin<"__builtin_ia32_blendvps">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,llvm_v4f32_ty],
                  [IntrNoMem]>;
}

// Vector dot product
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_dppd            : GCCBuiltin<"__builtin_ia32_dppd">,
          Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty, llvm_i8_ty],
                    [IntrNoMem, Commutative]>;
  def int_x86_sse41_dpps            : GCCBuiltin<"__builtin_ia32_dpps">,
          Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_i8_ty],
                    [IntrNoMem, Commutative]>;
}

// Vector sum of absolute differences
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_mpsadbw         : GCCBuiltin<"__builtin_ia32_mpsadbw128">,
          Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty, llvm_v16i8_ty,llvm_i8_ty],
                    [IntrNoMem, Commutative]>;
}

// Cacheability support ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_movntdqa        : GCCBuiltin<"__builtin_ia32_movntdqa">,
          Intrinsic<[llvm_v2i64_ty], [llvm_ptr_ty], [IntrReadMem]>;
}

// Test instruction with bitwise comparison.
let TargetPrefix = "x86" in { // All intrinsics start with "llvm.x86.".
  def int_x86_sse41_ptestz          : GCCBuiltin<"__builtin_ia32_ptestz128">,
          Intrinsic<[llvm_i32_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                    [IntrNoMem]>;
  def int_x86_sse41_ptestc          : GCCBuiltin<"__builtin_ia32_ptestc128">,
          Intrinsic<[llvm_i32_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                    [IntrNoMem]>;
  def int_x86_sse41_ptestnzc        : GCCBuiltin<"__builtin_ia32_ptestnzc128">,
          Intrinsic<[llvm_i32_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                    [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// SSE4.2

// Miscellaneous
// CRC Instruction
let TargetPrefix = "x86" in { // All intrinsics start with "llvm.x86.".
  def int_x86_sse42_crc32_32_8       : GCCBuiltin<"__builtin_ia32_crc32qi">,
          Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_sse42_crc32_32_16      : GCCBuiltin<"__builtin_ia32_crc32hi">,
          Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i16_ty],
                    [IntrNoMem]>;
  def int_x86_sse42_crc32_32_32      : GCCBuiltin<"__builtin_ia32_crc32si">,
          Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                    [IntrNoMem]>;
  def int_x86_sse42_crc32_64_64      : GCCBuiltin<"__builtin_ia32_crc32di">,
          Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
                    [IntrNoMem]>;
}

// String/text processing ops.
let TargetPrefix = "x86" in { // All intrinsics start with "llvm.x86.".
  def int_x86_sse42_pcmpistrm128  : GCCBuiltin<"__builtin_ia32_pcmpistrm128">,
    Intrinsic<[llvm_v16i8_ty],
        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpistri128  : GCCBuiltin<"__builtin_ia32_pcmpistri128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpistria128 : GCCBuiltin<"__builtin_ia32_pcmpistria128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpistric128 : GCCBuiltin<"__builtin_ia32_pcmpistric128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpistrio128 : GCCBuiltin<"__builtin_ia32_pcmpistrio128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpistris128 : GCCBuiltin<"__builtin_ia32_pcmpistris128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpistriz128 : GCCBuiltin<"__builtin_ia32_pcmpistriz128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpestrm128  : GCCBuiltin<"__builtin_ia32_pcmpestrm128">,
    Intrinsic<[llvm_v16i8_ty],
        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty,
         llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpestri128  : GCCBuiltin<"__builtin_ia32_pcmpestri128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty,
         llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpestria128 : GCCBuiltin<"__builtin_ia32_pcmpestria128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty,
         llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpestric128 : GCCBuiltin<"__builtin_ia32_pcmpestric128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty,
         llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpestrio128 : GCCBuiltin<"__builtin_ia32_pcmpestrio128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty,
         llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpestris128 : GCCBuiltin<"__builtin_ia32_pcmpestris128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty,
         llvm_i8_ty],
        [IntrNoMem]>;
  def int_x86_sse42_pcmpestriz128 : GCCBuiltin<"__builtin_ia32_pcmpestriz128">,
    Intrinsic<[llvm_i32_ty],
        [llvm_v16i8_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty,
         llvm_i8_ty],
        [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// SSE4A

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse4a_extrqi : GCCBuiltin<"__builtin_ia32_extrqi">,
    Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_i8_ty, llvm_i8_ty],
              [IntrNoMem]>;
  def int_x86_sse4a_extrq  : GCCBuiltin<"__builtin_ia32_extrq">,
    Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v16i8_ty], [IntrNoMem]>;

  def int_x86_sse4a_insertqi : GCCBuiltin<"__builtin_ia32_insertqi">,
    Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty,
                                llvm_i8_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_sse4a_insertq  : GCCBuiltin<"__builtin_ia32_insertq">,
    Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// AVX

// Arithmetic ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_addsub_pd_256 : GCCBuiltin<"__builtin_ia32_addsubpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_addsub_ps_256 : GCCBuiltin<"__builtin_ia32_addsubps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;
  def int_x86_avx_max_pd_256 : GCCBuiltin<"__builtin_ia32_maxpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_max_ps_256 : GCCBuiltin<"__builtin_ia32_maxps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;
  def int_x86_avx_min_pd_256 : GCCBuiltin<"__builtin_ia32_minpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_min_ps_256 : GCCBuiltin<"__builtin_ia32_minps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;

  def int_x86_avx_sqrt_pd_256 : GCCBuiltin<"__builtin_ia32_sqrtpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_sqrt_ps_256 : GCCBuiltin<"__builtin_ia32_sqrtps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty], [IntrNoMem]>;

  def int_x86_avx_rsqrt_ps_256 : GCCBuiltin<"__builtin_ia32_rsqrtps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty], [IntrNoMem]>;

  def int_x86_avx_rcp_ps_256 : GCCBuiltin<"__builtin_ia32_rcpps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty], [IntrNoMem]>;

  def int_x86_avx_round_pd_256 : GCCBuiltin<"__builtin_ia32_roundpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx_round_ps_256 : GCCBuiltin<"__builtin_ia32_roundps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_i32_ty], [IntrNoMem]>;
}

// Horizontal ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_hadd_pd_256 : GCCBuiltin<"__builtin_ia32_haddpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_hsub_ps_256 : GCCBuiltin<"__builtin_ia32_hsubps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;
  def int_x86_avx_hsub_pd_256 : GCCBuiltin<"__builtin_ia32_hsubpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_hadd_ps_256 : GCCBuiltin<"__builtin_ia32_haddps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;
}

// Vector permutation
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_vpermilvar_pd : GCCBuiltin<"__builtin_ia32_vpermilvarpd">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                  llvm_v2i64_ty], [IntrNoMem]>;
  def int_x86_avx_vpermilvar_ps : GCCBuiltin<"__builtin_ia32_vpermilvarps">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                  llvm_v4i32_ty], [IntrNoMem]>;

  def int_x86_avx_vpermilvar_pd_256 :
        GCCBuiltin<"__builtin_ia32_vpermilvarpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty, llvm_v4i64_ty], [IntrNoMem]>;
  def int_x86_avx_vpermilvar_ps_256 :
        GCCBuiltin<"__builtin_ia32_vpermilvarps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty, llvm_v8i32_ty], [IntrNoMem]>;

  def int_x86_avx_vperm2f128_pd_256 :
        GCCBuiltin<"__builtin_ia32_vperm2f128_pd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx_vperm2f128_ps_256 :
        GCCBuiltin<"__builtin_ia32_vperm2f128_ps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx_vperm2f128_si_256 :
        GCCBuiltin<"__builtin_ia32_vperm2f128_si256">,
        Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                  llvm_v8i32_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_d_128 :
       GCCBuiltin<"__builtin_ia32_vpermi2vard128_mask">,
        Intrinsic<[llvm_v4i32_ty],
        [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty, llvm_i8_ty],
        [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_d_256 :
        GCCBuiltin<"__builtin_ia32_vpermi2vard256_mask">,
          Intrinsic<[llvm_v8i32_ty],
          [llvm_v8i32_ty, llvm_v8i32_ty, llvm_v8i32_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_d_512 :
        GCCBuiltin<"__builtin_ia32_vpermi2vard512_mask">,
          Intrinsic<[llvm_v16i32_ty],
          [llvm_v16i32_ty, llvm_v16i32_ty, llvm_v16i32_ty, llvm_i16_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_hi_128 :
        GCCBuiltin<"__builtin_ia32_vpermi2varhi128_mask">,
          Intrinsic<[llvm_v8i16_ty],
          [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_hi_256 :
        GCCBuiltin<"__builtin_ia32_vpermi2varhi256_mask">,
          Intrinsic<[llvm_v16i16_ty],
          [llvm_v16i16_ty, llvm_v16i16_ty, llvm_v16i16_ty, llvm_i16_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_hi_512 :
        GCCBuiltin<"__builtin_ia32_vpermi2varhi512_mask">,
          Intrinsic<[llvm_v32i16_ty],
          [llvm_v32i16_ty, llvm_v32i16_ty, llvm_v32i16_ty, llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_pd_128 :
        GCCBuiltin<"__builtin_ia32_vpermi2varpd128_mask">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2i64_ty, llvm_v2f64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_pd_256 :
        GCCBuiltin<"__builtin_ia32_vpermi2varpd256_mask">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_v4i64_ty, llvm_v4f64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_pd_512 :
        GCCBuiltin<"__builtin_ia32_vpermi2varpd512_mask">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty, llvm_v8i64_ty, llvm_v8f64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_ps_128 :
        GCCBuiltin<"__builtin_ia32_vpermi2varps128_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4i32_ty, llvm_v4f32_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_ps_256 :
        GCCBuiltin<"__builtin_ia32_vpermi2varps256_mask">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_v8i32_ty, llvm_v8f32_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_ps_512 :
        GCCBuiltin<"__builtin_ia32_vpermi2varps512_mask">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty, llvm_v16i32_ty, llvm_v16f32_ty, llvm_i16_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_q_128 :
        GCCBuiltin<"__builtin_ia32_vpermi2varq128_mask">,
          Intrinsic<[llvm_v2i64_ty],
          [llvm_v2i64_ty, llvm_v2i64_ty, llvm_v2i64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_q_256 :
        GCCBuiltin<"__builtin_ia32_vpermi2varq256_mask">,
          Intrinsic<[llvm_v4i64_ty],
          [llvm_v4i64_ty, llvm_v4i64_ty, llvm_v4i64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_q_512 :
        GCCBuiltin<"__builtin_ia32_vpermi2varq512_mask">,
          Intrinsic<[llvm_v8i64_ty],
          [llvm_v8i64_ty, llvm_v8i64_ty, llvm_v8i64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_d_512:
        GCCBuiltin<"__builtin_ia32_vpermt2vard512_mask">,
        Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty,
                  llvm_v16i32_ty, llvm_v16i32_ty, llvm_i16_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_q_512:
        GCCBuiltin<"__builtin_ia32_vpermt2varq512_mask">,
        Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty,
                  llvm_v8i64_ty, llvm_v8i64_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_ps_512:
        GCCBuiltin<"__builtin_ia32_vpermt2varps512_mask">,
        Intrinsic<[llvm_v16f32_ty], [llvm_v16i32_ty,
                  llvm_v16f32_ty, llvm_v16f32_ty, llvm_i16_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_pd_512:
        GCCBuiltin<"__builtin_ia32_vpermt2varpd512_mask">,
        Intrinsic<[llvm_v8f64_ty], [llvm_v8i64_ty,
                  llvm_v8f64_ty, llvm_v8f64_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_d_128 :
        GCCBuiltin<"__builtin_ia32_vpermt2vard128_mask">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_d_128 :
        GCCBuiltin<"__builtin_ia32_vpermt2vard128_maskz">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_d_256 :
        GCCBuiltin<"__builtin_ia32_vpermt2vard256_mask">,
          Intrinsic<[llvm_v8i32_ty],
          [llvm_v8i32_ty, llvm_v8i32_ty, llvm_v8i32_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_d_256 :
        GCCBuiltin<"__builtin_ia32_vpermt2vard256_maskz">,
          Intrinsic<[llvm_v8i32_ty],
          [llvm_v8i32_ty, llvm_v8i32_ty, llvm_v8i32_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_d_512 :
        GCCBuiltin<"__builtin_ia32_vpermt2vard512_maskz">,
          Intrinsic<[llvm_v16i32_ty],
          [llvm_v16i32_ty, llvm_v16i32_ty, llvm_v16i32_ty, llvm_i16_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_hi_128 :
        GCCBuiltin<"__builtin_ia32_vpermt2varhi128_mask">,
          Intrinsic<[llvm_v8i16_ty],
          [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_hi_128 :
        GCCBuiltin<"__builtin_ia32_vpermt2varhi128_maskz">,
          Intrinsic<[llvm_v8i16_ty],
          [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_hi_256 :
        GCCBuiltin<"__builtin_ia32_vpermt2varhi256_mask">,
          Intrinsic<[llvm_v16i16_ty],
          [llvm_v16i16_ty, llvm_v16i16_ty, llvm_v16i16_ty, llvm_i16_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_hi_256 :
        GCCBuiltin<"__builtin_ia32_vpermt2varhi256_maskz">,
          Intrinsic<[llvm_v16i16_ty],
          [llvm_v16i16_ty, llvm_v16i16_ty, llvm_v16i16_ty, llvm_i16_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_hi_512 :
        GCCBuiltin<"__builtin_ia32_vpermt2varhi512_mask">,
          Intrinsic<[llvm_v32i16_ty],
          [llvm_v32i16_ty, llvm_v32i16_ty, llvm_v32i16_ty, llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_hi_512 :
        GCCBuiltin<"__builtin_ia32_vpermt2varhi512_maskz">,
          Intrinsic<[llvm_v32i16_ty],
          [llvm_v32i16_ty, llvm_v32i16_ty, llvm_v32i16_ty, llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_pd_128 :
        GCCBuiltin<"__builtin_ia32_vpermt2varpd128_mask">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2i64_ty, llvm_v2f64_ty, llvm_v2f64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_pd_128 :
        GCCBuiltin<"__builtin_ia32_vpermt2varpd128_maskz">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2i64_ty, llvm_v2f64_ty, llvm_v2f64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_pd_256 :
        GCCBuiltin<"__builtin_ia32_vpermt2varpd256_mask">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4i64_ty, llvm_v4f64_ty, llvm_v4f64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_pd_256 :
        GCCBuiltin<"__builtin_ia32_vpermt2varpd256_maskz">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4i64_ty, llvm_v4f64_ty, llvm_v4f64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_pd_512 :
        GCCBuiltin<"__builtin_ia32_vpermt2varpd512_maskz">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8i64_ty, llvm_v8f64_ty, llvm_v8f64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_ps_128 :
        GCCBuiltin<"__builtin_ia32_vpermt2varps128_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4i32_ty, llvm_v4f32_ty, llvm_v4f32_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_ps_128 :
        GCCBuiltin<"__builtin_ia32_vpermt2varps128_maskz">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4i32_ty, llvm_v4f32_ty, llvm_v4f32_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_ps_256 :
        GCCBuiltin<"__builtin_ia32_vpermt2varps256_mask">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8i32_ty, llvm_v8f32_ty, llvm_v8f32_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_ps_256 :
        GCCBuiltin<"__builtin_ia32_vpermt2varps256_maskz">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8i32_ty, llvm_v8f32_ty, llvm_v8f32_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_ps_512 :
        GCCBuiltin<"__builtin_ia32_vpermt2varps512_maskz">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16i32_ty, llvm_v16f32_ty, llvm_v16f32_ty, llvm_i16_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_q_128 :
        GCCBuiltin<"__builtin_ia32_vpermt2varq128_mask">,
          Intrinsic<[llvm_v2i64_ty],
          [llvm_v2i64_ty, llvm_v2i64_ty, llvm_v2i64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_q_128 :
        GCCBuiltin<"__builtin_ia32_vpermt2varq128_maskz">,
          Intrinsic<[llvm_v2i64_ty],
          [llvm_v2i64_ty, llvm_v2i64_ty, llvm_v2i64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_q_256 :
        GCCBuiltin<"__builtin_ia32_vpermt2varq256_mask">,
          Intrinsic<[llvm_v4i64_ty],
          [llvm_v4i64_ty, llvm_v4i64_ty, llvm_v4i64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_q_256 :
        GCCBuiltin<"__builtin_ia32_vpermt2varq256_maskz">,
          Intrinsic<[llvm_v4i64_ty],
          [llvm_v4i64_ty, llvm_v4i64_ty, llvm_v4i64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_q_512 :
        GCCBuiltin<"__builtin_ia32_vpermt2varq512_maskz">,
          Intrinsic<[llvm_v8i64_ty],
          [llvm_v8i64_ty, llvm_v8i64_ty, llvm_v8i64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_qi_128 :
        GCCBuiltin<"__builtin_ia32_vpermi2varqi128_mask">,
          Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
          llvm_v16i8_ty, llvm_v16i8_ty, llvm_i16_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_qi_128 :
        GCCBuiltin<"__builtin_ia32_vpermt2varqi128_mask">,
          Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
          llvm_v16i8_ty, llvm_v16i8_ty, llvm_i16_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_qi_128 :
        GCCBuiltin<"__builtin_ia32_vpermt2varqi128_maskz">,
          Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
          llvm_v16i8_ty, llvm_v16i8_ty, llvm_i16_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_qi_256 :
        GCCBuiltin<"__builtin_ia32_vpermi2varqi256_mask">,
          Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
          llvm_v32i8_ty, llvm_v32i8_ty, llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_qi_256 :
        GCCBuiltin<"__builtin_ia32_vpermt2varqi256_mask">,
          Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
          llvm_v32i8_ty, llvm_v32i8_ty, llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_qi_256 :
        GCCBuiltin<"__builtin_ia32_vpermt2varqi256_maskz">,
          Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
          llvm_v32i8_ty, llvm_v32i8_ty, llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermi2var_qi_512 :
        GCCBuiltin<"__builtin_ia32_vpermi2varqi512_mask">,
          Intrinsic<[llvm_v64i8_ty], [llvm_v64i8_ty,
          llvm_v64i8_ty, llvm_v64i8_ty, llvm_i64_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vpermt2var_qi_512 :
        GCCBuiltin<"__builtin_ia32_vpermt2varqi512_mask">,
          Intrinsic<[llvm_v64i8_ty], [llvm_v64i8_ty,
          llvm_v64i8_ty, llvm_v64i8_ty, llvm_i64_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vpermt2var_qi_512 :
        GCCBuiltin<"__builtin_ia32_vpermt2varqi512_maskz">,
          Intrinsic<[llvm_v64i8_ty], [llvm_v64i8_ty,
          llvm_v64i8_ty, llvm_v64i8_ty, llvm_i64_ty],
          [IntrNoMem]>;

  def int_x86_avx512_vpermilvar_pd_512 :
        GCCBuiltin<"__builtin_ia32_vpermilvarpd512">,
          Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8i64_ty],
          [IntrNoMem]>;

  def int_x86_avx512_vpermilvar_ps_512 :
        GCCBuiltin<"__builtin_ia32_vpermilvarps512">,
          Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_pshuf_b_512 :
        GCCBuiltin<"__builtin_ia32_pshufb512">,
          Intrinsic<[llvm_v64i8_ty], [llvm_v64i8_ty, llvm_v64i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_shuf_f32x4_256 :
         GCCBuiltin<"__builtin_ia32_shuf_f32x4_256_mask">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_v8f32_ty, llvm_i32_ty, llvm_v8f32_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_shuf_f32x4 :
         GCCBuiltin<"__builtin_ia32_shuf_f32x4_mask">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty, llvm_v16f32_ty, llvm_i32_ty, llvm_v16f32_ty, llvm_i16_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_shuf_f64x2_256 :
         GCCBuiltin<"__builtin_ia32_shuf_f64x2_256_mask">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_v4f64_ty, llvm_i32_ty, llvm_v4f64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_shuf_f64x2 :
         GCCBuiltin<"__builtin_ia32_shuf_f64x2_mask">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty, llvm_v8f64_ty, llvm_i32_ty, llvm_v8f64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_shuf_i32x4_256 :
         GCCBuiltin<"__builtin_ia32_shuf_i32x4_256_mask">,
          Intrinsic<[llvm_v8i32_ty],
          [llvm_v8i32_ty, llvm_v8i32_ty, llvm_i32_ty, llvm_v8i32_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_shuf_i32x4 :
         GCCBuiltin<"__builtin_ia32_shuf_i32x4_mask">,
          Intrinsic<[llvm_v16i32_ty],
          [llvm_v16i32_ty, llvm_v16i32_ty, llvm_i32_ty, llvm_v16i32_ty, llvm_i16_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_shuf_i64x2_256 :
         GCCBuiltin<"__builtin_ia32_shuf_i64x2_256_mask">,
          Intrinsic<[llvm_v4i64_ty],
          [llvm_v4i64_ty, llvm_v4i64_ty, llvm_i32_ty, llvm_v4i64_ty, llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_shuf_i64x2 :
         GCCBuiltin<"__builtin_ia32_shuf_i64x2_mask">,
          Intrinsic<[llvm_v8i64_ty],
          [llvm_v8i64_ty, llvm_v8i64_ty, llvm_i32_ty, llvm_v8i64_ty, llvm_i8_ty],
          [IntrNoMem]>;
}

// Vector blend
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_blendv_pd_256 : GCCBuiltin<"__builtin_ia32_blendvpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty, llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_blendv_ps_256 : GCCBuiltin<"__builtin_ia32_blendvps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty, llvm_v8f32_ty], [IntrNoMem]>;
}

// Vector dot product
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_dp_ps_256 : GCCBuiltin<"__builtin_ia32_dpps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty, llvm_i8_ty], [IntrNoMem, Commutative]>;
}

// Vector compare
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_cmp_pd_256 :
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx_cmp_ps_256 :
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty, llvm_i8_ty], [IntrNoMem]>;
}

// Vector convert
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_cvtdq2_ps_256 : GCCBuiltin<"__builtin_ia32_cvtdq2ps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8i32_ty], [IntrNoMem]>;
  def int_x86_avx_cvt_pd2_ps_256 : GCCBuiltin<"__builtin_ia32_cvtpd2ps256">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_cvt_ps2dq_256 : GCCBuiltin<"__builtin_ia32_cvtps2dq256">,
        Intrinsic<[llvm_v8i32_ty], [llvm_v8f32_ty], [IntrNoMem]>;
  def int_x86_avx_cvtt_pd2dq_256 : GCCBuiltin<"__builtin_ia32_cvttpd2dq256">,
        Intrinsic<[llvm_v4i32_ty], [llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_cvt_pd2dq_256 : GCCBuiltin<"__builtin_ia32_cvtpd2dq256">,
        Intrinsic<[llvm_v4i32_ty], [llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_cvtt_ps2dq_256 : GCCBuiltin<"__builtin_ia32_cvttps2dq256">,
        Intrinsic<[llvm_v8i32_ty], [llvm_v8f32_ty], [IntrNoMem]>;
}

// Vector bit test
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_vtestz_pd : GCCBuiltin<"__builtin_ia32_vtestzpd">,
        Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                  llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_avx_vtestc_pd : GCCBuiltin<"__builtin_ia32_vtestcpd">,
        Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                  llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_avx_vtestnzc_pd : GCCBuiltin<"__builtin_ia32_vtestnzcpd">,
        Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                  llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_avx_vtestz_ps : GCCBuiltin<"__builtin_ia32_vtestzps">,
        Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                  llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_avx_vtestc_ps : GCCBuiltin<"__builtin_ia32_vtestcps">,
        Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                  llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_avx_vtestnzc_ps : GCCBuiltin<"__builtin_ia32_vtestnzcps">,
        Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                  llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_avx_vtestz_pd_256 : GCCBuiltin<"__builtin_ia32_vtestzpd256">,
        Intrinsic<[llvm_i32_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_vtestc_pd_256 : GCCBuiltin<"__builtin_ia32_vtestcpd256">,
        Intrinsic<[llvm_i32_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_vtestnzc_pd_256 : GCCBuiltin<"__builtin_ia32_vtestnzcpd256">,
        Intrinsic<[llvm_i32_ty], [llvm_v4f64_ty,
                  llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_vtestz_ps_256 : GCCBuiltin<"__builtin_ia32_vtestzps256">,
        Intrinsic<[llvm_i32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;
  def int_x86_avx_vtestc_ps_256 : GCCBuiltin<"__builtin_ia32_vtestcps256">,
        Intrinsic<[llvm_i32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;
  def int_x86_avx_vtestnzc_ps_256 : GCCBuiltin<"__builtin_ia32_vtestnzcps256">,
        Intrinsic<[llvm_i32_ty], [llvm_v8f32_ty,
                  llvm_v8f32_ty], [IntrNoMem]>;
  def int_x86_avx_ptestz_256 : GCCBuiltin<"__builtin_ia32_ptestz256">,
        Intrinsic<[llvm_i32_ty], [llvm_v4i64_ty,
                  llvm_v4i64_ty], [IntrNoMem]>;
  def int_x86_avx_ptestc_256 : GCCBuiltin<"__builtin_ia32_ptestc256">,
        Intrinsic<[llvm_i32_ty], [llvm_v4i64_ty,
                  llvm_v4i64_ty], [IntrNoMem]>;
  def int_x86_avx_ptestnzc_256 : GCCBuiltin<"__builtin_ia32_ptestnzc256">,
        Intrinsic<[llvm_i32_ty], [llvm_v4i64_ty,
                  llvm_v4i64_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestm_d_512 : GCCBuiltin<"__builtin_ia32_ptestmd512">,
        Intrinsic<[llvm_i16_ty], [llvm_v16i32_ty, llvm_v16i32_ty,
                  llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestm_q_512 : GCCBuiltin<"__builtin_ia32_ptestmq512">,
        Intrinsic<[llvm_i8_ty], [llvm_v8i64_ty, llvm_v8i64_ty,
                  llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_ptestm_b_128 : GCCBuiltin<"__builtin_ia32_ptestmb128">,
        Intrinsic<[llvm_i16_ty], [llvm_v16i8_ty,
                   llvm_v16i8_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestm_b_256 : GCCBuiltin<"__builtin_ia32_ptestmb256">,
        Intrinsic<[llvm_i32_ty], [llvm_v32i8_ty,
                   llvm_v32i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestm_b_512 : GCCBuiltin<"__builtin_ia32_ptestmb512">,
        Intrinsic<[llvm_i64_ty], [llvm_v64i8_ty,
                   llvm_v64i8_ty, llvm_i64_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestm_d_128 : GCCBuiltin<"__builtin_ia32_ptestmd128">,
        Intrinsic<[llvm_i8_ty], [llvm_v4i32_ty,
                   llvm_v4i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestm_d_256 : GCCBuiltin<"__builtin_ia32_ptestmd256">,
        Intrinsic<[llvm_i8_ty], [llvm_v8i32_ty,
                   llvm_v8i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestm_q_128 : GCCBuiltin<"__builtin_ia32_ptestmq128">,
        Intrinsic<[llvm_i8_ty], [llvm_v2i64_ty,
                   llvm_v2i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestm_q_256 : GCCBuiltin<"__builtin_ia32_ptestmq256">,
        Intrinsic<[llvm_i8_ty], [llvm_v4i64_ty,
                   llvm_v4i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestm_w_128 : GCCBuiltin<"__builtin_ia32_ptestmw128">,
        Intrinsic<[llvm_i8_ty], [llvm_v8i16_ty,
                   llvm_v8i16_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestm_w_256 : GCCBuiltin<"__builtin_ia32_ptestmw256">,
        Intrinsic<[llvm_i16_ty], [llvm_v16i16_ty,
                   llvm_v16i16_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestm_w_512 : GCCBuiltin<"__builtin_ia32_ptestmw512">,
        Intrinsic<[llvm_i32_ty], [llvm_v32i16_ty,
                   llvm_v32i16_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_ptestnm_b_128 : GCCBuiltin<"__builtin_ia32_ptestnmb128">,
        Intrinsic<[llvm_i16_ty], [llvm_v16i8_ty,
                   llvm_v16i8_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestnm_b_256 : GCCBuiltin<"__builtin_ia32_ptestnmb256">,
        Intrinsic<[llvm_i32_ty], [llvm_v32i8_ty,
                   llvm_v32i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestnm_b_512 : GCCBuiltin<"__builtin_ia32_ptestnmb512">,
        Intrinsic<[llvm_i64_ty], [llvm_v64i8_ty,
                   llvm_v64i8_ty, llvm_i64_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestnm_d_128 : GCCBuiltin<"__builtin_ia32_ptestnmd128">,
        Intrinsic<[llvm_i8_ty], [llvm_v4i32_ty,
                   llvm_v4i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestnm_d_256 : GCCBuiltin<"__builtin_ia32_ptestnmd256">,
        Intrinsic<[llvm_i8_ty], [llvm_v8i32_ty,
                   llvm_v8i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestnm_d_512 : GCCBuiltin<"__builtin_ia32_ptestnmd512">,
        Intrinsic<[llvm_i16_ty], [llvm_v16i32_ty,
                   llvm_v16i32_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestnm_q_128 : GCCBuiltin<"__builtin_ia32_ptestnmq128">,
        Intrinsic<[llvm_i8_ty], [llvm_v2i64_ty,
                   llvm_v2i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestnm_q_256 : GCCBuiltin<"__builtin_ia32_ptestnmq256">,
        Intrinsic<[llvm_i8_ty], [llvm_v4i64_ty,
                   llvm_v4i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestnm_q_512 : GCCBuiltin<"__builtin_ia32_ptestnmq512">,
        Intrinsic<[llvm_i8_ty], [llvm_v8i64_ty,
                   llvm_v8i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestnm_w_128 : GCCBuiltin<"__builtin_ia32_ptestnmw128">,
        Intrinsic<[llvm_i8_ty], [llvm_v8i16_ty,
                   llvm_v8i16_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestnm_w_256 : GCCBuiltin<"__builtin_ia32_ptestnmw256">,
        Intrinsic<[llvm_i16_ty], [llvm_v16i16_ty,
                   llvm_v16i16_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_ptestnm_w_512 : GCCBuiltin<"__builtin_ia32_ptestnmw512">,
        Intrinsic<[llvm_i32_ty], [llvm_v32i16_ty,
                   llvm_v32i16_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_fpclass_pd_128 :
         GCCBuiltin<"__builtin_ia32_fpclasspd128_mask">,
          Intrinsic<[llvm_i8_ty], [llvm_v2f64_ty, llvm_i32_ty, llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_fpclass_pd_256 :
         GCCBuiltin<"__builtin_ia32_fpclasspd256_mask">,
          Intrinsic<[llvm_i8_ty], [llvm_v4f64_ty, llvm_i32_ty, llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_fpclass_pd_512 :
         GCCBuiltin<"__builtin_ia32_fpclasspd512_mask">,
          Intrinsic<[llvm_i8_ty], [llvm_v8f64_ty, llvm_i32_ty, llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_fpclass_ps_128 :
         GCCBuiltin<"__builtin_ia32_fpclassps128_mask">,
          Intrinsic<[llvm_i8_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_fpclass_ps_256 :
         GCCBuiltin<"__builtin_ia32_fpclassps256_mask">,
          Intrinsic<[llvm_i8_ty], [llvm_v8f32_ty, llvm_i32_ty, llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_fpclass_ps_512 :
         GCCBuiltin<"__builtin_ia32_fpclassps512_mask">,
          Intrinsic<[llvm_i16_ty], [llvm_v16f32_ty, llvm_i32_ty, llvm_i16_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_fpclass_sd :
         GCCBuiltin<"__builtin_ia32_fpclasssd_mask">,
          Intrinsic<[llvm_i8_ty], [llvm_v2f64_ty, llvm_i32_ty, llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_fpclass_ss :
         GCCBuiltin<"__builtin_ia32_fpclassss_mask">,
          Intrinsic<[llvm_i8_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i8_ty],
          [IntrNoMem]>;
}

// Vector extract sign mask
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_movmsk_pd_256 : GCCBuiltin<"__builtin_ia32_movmskpd256">,
        Intrinsic<[llvm_i32_ty], [llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_avx_movmsk_ps_256 : GCCBuiltin<"__builtin_ia32_movmskps256">,
        Intrinsic<[llvm_i32_ty], [llvm_v8f32_ty], [IntrNoMem]>;
}

// Vector zero
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_vzeroall : GCCBuiltin<"__builtin_ia32_vzeroall">,
        Intrinsic<[], [], []>;
  def int_x86_avx_vzeroupper : GCCBuiltin<"__builtin_ia32_vzeroupper">,
        Intrinsic<[], [], []>;
}

// SIMD load ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_ldu_dq_256 : GCCBuiltin<"__builtin_ia32_lddqu256">,
        Intrinsic<[llvm_v32i8_ty], [llvm_ptr_ty], [IntrReadMem]>;
}

// Conditional load ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_maskload_pd : GCCBuiltin<"__builtin_ia32_maskloadpd">,
        Intrinsic<[llvm_v2f64_ty], [llvm_ptr_ty, llvm_v2i64_ty],
                  [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx_maskload_ps : GCCBuiltin<"__builtin_ia32_maskloadps">,
        Intrinsic<[llvm_v4f32_ty], [llvm_ptr_ty, llvm_v4i32_ty],
                  [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx_maskload_pd_256 : GCCBuiltin<"__builtin_ia32_maskloadpd256">,
        Intrinsic<[llvm_v4f64_ty], [llvm_ptr_ty, llvm_v4i64_ty],
                  [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx_maskload_ps_256 : GCCBuiltin<"__builtin_ia32_maskloadps256">,
        Intrinsic<[llvm_v8f32_ty], [llvm_ptr_ty, llvm_v8i32_ty],
                  [IntrReadMem, IntrArgMemOnly]>;
}

// Conditional store ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx_maskstore_pd : GCCBuiltin<"__builtin_ia32_maskstorepd">,
        Intrinsic<[], [llvm_ptr_ty,
                  llvm_v2i64_ty, llvm_v2f64_ty], [IntrArgMemOnly]>;
  def int_x86_avx_maskstore_ps : GCCBuiltin<"__builtin_ia32_maskstoreps">,
        Intrinsic<[], [llvm_ptr_ty,
                  llvm_v4i32_ty, llvm_v4f32_ty], [IntrArgMemOnly]>;
  def int_x86_avx_maskstore_pd_256 :
        GCCBuiltin<"__builtin_ia32_maskstorepd256">,
        Intrinsic<[], [llvm_ptr_ty,
                  llvm_v4i64_ty, llvm_v4f64_ty], [IntrArgMemOnly]>;
  def int_x86_avx_maskstore_ps_256 :
        GCCBuiltin<"__builtin_ia32_maskstoreps256">,
        Intrinsic<[], [llvm_ptr_ty,
                  llvm_v8i32_ty, llvm_v8f32_ty], [IntrArgMemOnly]>;

  def int_x86_avx512_mask_store_ss :
        GCCBuiltin<"__builtin_ia32_storess_mask">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v4f32_ty, llvm_i8_ty],
                  [IntrArgMemOnly]>;
}

//===----------------------------------------------------------------------===//
// AVX2

// Integer arithmetic ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_padds_b : GCCBuiltin<"__builtin_ia32_paddsb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_padds_w : GCCBuiltin<"__builtin_ia32_paddsw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_paddus_b : GCCBuiltin<"__builtin_ia32_paddusb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_paddus_w : GCCBuiltin<"__builtin_ia32_paddusw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_psubs_b : GCCBuiltin<"__builtin_ia32_psubsb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem]>;
  def int_x86_avx2_psubs_w : GCCBuiltin<"__builtin_ia32_psubsw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem]>;
  def int_x86_avx2_psubus_b : GCCBuiltin<"__builtin_ia32_psubusb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem]>;
  def int_x86_avx2_psubus_w : GCCBuiltin<"__builtin_ia32_psubusw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem]>;
  def int_x86_avx2_pmulhu_w : GCCBuiltin<"__builtin_ia32_pmulhuw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_pmulh_w : GCCBuiltin<"__builtin_ia32_pmulhw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_pmulu_dq : GCCBuiltin<"__builtin_ia32_pmuludq256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v8i32_ty,
                         llvm_v8i32_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_pmul_dq : GCCBuiltin<"__builtin_ia32_pmuldq256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v8i32_ty,
                         llvm_v8i32_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_pmadd_wd : GCCBuiltin<"__builtin_ia32_pmaddwd256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_pavg_b : GCCBuiltin<"__builtin_ia32_pavgb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_pavg_w : GCCBuiltin<"__builtin_ia32_pavgw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_psad_bw : GCCBuiltin<"__builtin_ia32_psadbw256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem, Commutative]>;
}

// Integer shift ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_psll_w : GCCBuiltin<"__builtin_ia32_psllw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_avx2_psll_d : GCCBuiltin<"__builtin_ia32_pslld256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psll_q : GCCBuiltin<"__builtin_ia32_psllq256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_v2i64_ty], [IntrNoMem]>;
  def int_x86_avx2_psrl_w : GCCBuiltin<"__builtin_ia32_psrlw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_avx2_psrl_d : GCCBuiltin<"__builtin_ia32_psrld256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psrl_q : GCCBuiltin<"__builtin_ia32_psrlq256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_v2i64_ty], [IntrNoMem]>;
  def int_x86_avx2_psra_w : GCCBuiltin<"__builtin_ia32_psraw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_avx2_psra_d : GCCBuiltin<"__builtin_ia32_psrad256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;

  def int_x86_avx2_pslli_w : GCCBuiltin<"__builtin_ia32_psllwi256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_pslli_d : GCCBuiltin<"__builtin_ia32_pslldi256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_pslli_q : GCCBuiltin<"__builtin_ia32_psllqi256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psrli_w : GCCBuiltin<"__builtin_ia32_psrlwi256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psrli_d : GCCBuiltin<"__builtin_ia32_psrldi256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psrli_q : GCCBuiltin<"__builtin_ia32_psrlqi256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psrai_w : GCCBuiltin<"__builtin_ia32_psrawi256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx2_psrai_d : GCCBuiltin<"__builtin_ia32_psradi256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_psra_q_128 : GCCBuiltin<"__builtin_ia32_psraq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_v2i64_ty], [IntrNoMem]>;
  def int_x86_avx512_psra_q_256 : GCCBuiltin<"__builtin_ia32_psraq256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_v2i64_ty], [IntrNoMem]>;

  def int_x86_avx512_psrai_q_128 : GCCBuiltin<"__builtin_ia32_psraqi128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_psrai_q_256 : GCCBuiltin<"__builtin_ia32_psraqi256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_psll_w_512 : GCCBuiltin<"__builtin_ia32_psllw512">,
              Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_avx512_psll_d_512 : GCCBuiltin<"__builtin_ia32_pslld512">,
              Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_avx512_psll_q_512 : GCCBuiltin<"__builtin_ia32_psllq512">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty,
                         llvm_v2i64_ty], [IntrNoMem]>;
  def int_x86_avx512_psrl_w_512 : GCCBuiltin<"__builtin_ia32_psrlw512">,
              Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_avx512_psrl_d_512 : GCCBuiltin<"__builtin_ia32_psrld512">,
              Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_avx512_psrl_q_512 : GCCBuiltin<"__builtin_ia32_psrlq512">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty,
                         llvm_v2i64_ty], [IntrNoMem]>;
  def int_x86_avx512_psra_w_512 : GCCBuiltin<"__builtin_ia32_psraw512">,
              Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty,
                         llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_avx512_psra_d_512 : GCCBuiltin<"__builtin_ia32_psrad512">,
              Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty,
                         llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_avx512_psra_q_512 : GCCBuiltin<"__builtin_ia32_psraq512">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty,
                         llvm_v2i64_ty], [IntrNoMem]>;

  def int_x86_avx512_pslli_w_512 : GCCBuiltin<"__builtin_ia32_psllwi512">,
              Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_pslli_d_512 : GCCBuiltin<"__builtin_ia32_pslldi512">,
              Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_pslli_q_512 : GCCBuiltin<"__builtin_ia32_psllqi512">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_psrli_w_512 : GCCBuiltin<"__builtin_ia32_psrlwi512">,
              Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_psrli_d_512 : GCCBuiltin<"__builtin_ia32_psrldi512">,
              Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_psrli_q_512 : GCCBuiltin<"__builtin_ia32_psrlqi512">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_psrai_w_512 : GCCBuiltin<"__builtin_ia32_psrawi512">,
              Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_psrai_d_512 : GCCBuiltin<"__builtin_ia32_psradi512">,
              Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_psrai_q_512 : GCCBuiltin<"__builtin_ia32_psraqi512">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty,
                         llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_pmultishift_qb_128:
        GCCBuiltin<"__builtin_ia32_vpmultishiftqb128_mask">,
        Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                   llvm_v16i8_ty, llvm_v16i8_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pmultishift_qb_256:
        GCCBuiltin<"__builtin_ia32_vpmultishiftqb256_mask">,
        Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                   llvm_v32i8_ty, llvm_v32i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pmultishift_qb_512:
        GCCBuiltin<"__builtin_ia32_vpmultishiftqb512_mask">,
        Intrinsic<[llvm_v64i8_ty], [llvm_v64i8_ty,
                   llvm_v64i8_ty, llvm_v64i8_ty, llvm_i64_ty], [IntrNoMem]>;
}

// Pack ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_packsswb : GCCBuiltin<"__builtin_ia32_packsswb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem]>;
  def int_x86_avx2_packssdw : GCCBuiltin<"__builtin_ia32_packssdw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v8i32_ty,
                         llvm_v8i32_ty], [IntrNoMem]>;
  def int_x86_avx2_packuswb : GCCBuiltin<"__builtin_ia32_packuswb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem]>;
  def int_x86_avx2_packusdw : GCCBuiltin<"__builtin_ia32_packusdw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v8i32_ty,
                         llvm_v8i32_ty], [IntrNoMem]>;
}

// Absolute value ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_pabs_b : GCCBuiltin<"__builtin_ia32_pabsb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty], [IntrNoMem]>;
  def int_x86_avx2_pabs_w : GCCBuiltin<"__builtin_ia32_pabsw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty], [IntrNoMem]>;
  def int_x86_avx2_pabs_d : GCCBuiltin<"__builtin_ia32_pabsd256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_pabs_b_128 :
       GCCBuiltin<"__builtin_ia32_pabsb128_mask">,
        Intrinsic<[llvm_v16i8_ty],
        [llvm_v16i8_ty, llvm_v16i8_ty,  llvm_i16_ty],
        [IntrNoMem]>;

  def int_x86_avx512_mask_pabs_b_256 :
        GCCBuiltin<"__builtin_ia32_pabsb256_mask">,
          Intrinsic<[llvm_v32i8_ty],
          [llvm_v32i8_ty, llvm_v32i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_pabs_b_512 :
        GCCBuiltin<"__builtin_ia32_pabsb512_mask">,
          Intrinsic<[llvm_v64i8_ty],
          [llvm_v64i8_ty, llvm_v64i8_ty,  llvm_i64_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_pabs_d_128 :
        GCCBuiltin<"__builtin_ia32_pabsd128_mask">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v4i32_ty, llvm_v4i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_pabs_d_256 :
        GCCBuiltin<"__builtin_ia32_pabsd256_mask">,
          Intrinsic<[llvm_v8i32_ty],
          [llvm_v8i32_ty, llvm_v8i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_pabs_d_512 :
        GCCBuiltin<"__builtin_ia32_pabsd512_mask">,
          Intrinsic<[llvm_v16i32_ty],
          [llvm_v16i32_ty, llvm_v16i32_ty,  llvm_i16_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_pabs_q_128 :
        GCCBuiltin<"__builtin_ia32_pabsq128_mask">,
          Intrinsic<[llvm_v2i64_ty],
          [llvm_v2i64_ty, llvm_v2i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_pabs_q_256 :
        GCCBuiltin<"__builtin_ia32_pabsq256_mask">,
          Intrinsic<[llvm_v4i64_ty],
          [llvm_v4i64_ty, llvm_v4i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_pabs_q_512 :
        GCCBuiltin<"__builtin_ia32_pabsq512_mask">,
          Intrinsic<[llvm_v8i64_ty],
          [llvm_v8i64_ty, llvm_v8i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_pabs_w_128 :
        GCCBuiltin<"__builtin_ia32_pabsw128_mask">,
          Intrinsic<[llvm_v8i16_ty],
          [llvm_v8i16_ty, llvm_v8i16_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_pabs_w_256 :
        GCCBuiltin<"__builtin_ia32_pabsw256_mask">,
          Intrinsic<[llvm_v16i16_ty],
          [llvm_v16i16_ty, llvm_v16i16_ty,  llvm_i16_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_pabs_w_512 :
        GCCBuiltin<"__builtin_ia32_pabsw512_mask">,
          Intrinsic<[llvm_v32i16_ty],
          [llvm_v32i16_ty, llvm_v32i16_ty,  llvm_i32_ty],
          [IntrNoMem]>;
}

// Horizontal arithmetic ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_phadd_w : GCCBuiltin<"__builtin_ia32_phaddw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem]>;
  def int_x86_avx2_phadd_d : GCCBuiltin<"__builtin_ia32_phaddd256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_v8i32_ty], [IntrNoMem]>;
  def int_x86_avx2_phadd_sw : GCCBuiltin<"__builtin_ia32_phaddsw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem]>;
  def int_x86_avx2_phsub_w : GCCBuiltin<"__builtin_ia32_phsubw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem]>;
  def int_x86_avx2_phsub_d : GCCBuiltin<"__builtin_ia32_phsubd256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_v8i32_ty], [IntrNoMem]>;
  def int_x86_avx2_phsub_sw : GCCBuiltin<"__builtin_ia32_phsubsw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem]>;
  def int_x86_avx2_pmadd_ub_sw : GCCBuiltin<"__builtin_ia32_pmaddubsw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem]>;
}

// Sign ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_psign_b : GCCBuiltin<"__builtin_ia32_psignb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem]>;
  def int_x86_avx2_psign_w : GCCBuiltin<"__builtin_ia32_psignw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem]>;
  def int_x86_avx2_psign_d : GCCBuiltin<"__builtin_ia32_psignd256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_v8i32_ty], [IntrNoMem]>;
}

// Packed multiply high with round and scale
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_pmul_hr_sw : GCCBuiltin<"__builtin_ia32_pmulhrsw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                         llvm_v16i16_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx512_mask_pmul_hr_sw_128 : GCCBuiltin<"__builtin_ia32_pmulhrsw128_mask">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                     llvm_v8i16_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pmul_hr_sw_256 : GCCBuiltin<"__builtin_ia32_pmulhrsw256_mask">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty, llvm_v16i16_ty,
                     llvm_v16i16_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pmul_hr_sw_512 : GCCBuiltin<"__builtin_ia32_pmulhrsw512_mask">,
              Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty, llvm_v32i16_ty,
                     llvm_v32i16_ty, llvm_i32_ty], [IntrNoMem]>;
}

// Vector blend
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_pblendvb : GCCBuiltin<"__builtin_ia32_pblendvb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty, llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem]>;
}

// Vector load with broadcast
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx512_mask_pbroadcast_b_gpr_128 :
          GCCBuiltin<"__builtin_ia32_pbroadcastb128_gpr_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_i8_ty, llvm_v16i8_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pbroadcast_b_gpr_256 :
          GCCBuiltin<"__builtin_ia32_pbroadcastb256_gpr_mask">,
          Intrinsic<[llvm_v32i8_ty],
                    [llvm_i8_ty, llvm_v32i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pbroadcast_b_gpr_512 :
          GCCBuiltin<"__builtin_ia32_pbroadcastb512_gpr_mask">,
          Intrinsic<[llvm_v64i8_ty],
                    [llvm_i8_ty, llvm_v64i8_ty, llvm_i64_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_pbroadcast_w_gpr_128 :
          GCCBuiltin<"__builtin_ia32_pbroadcastw128_gpr_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_i16_ty, llvm_v8i16_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pbroadcast_w_gpr_256 :
          GCCBuiltin<"__builtin_ia32_pbroadcastw256_gpr_mask">,
          Intrinsic<[llvm_v16i16_ty],
                    [llvm_i16_ty, llvm_v16i16_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pbroadcast_w_gpr_512 :
          GCCBuiltin<"__builtin_ia32_pbroadcastw512_gpr_mask">,
          Intrinsic<[llvm_v32i16_ty],
                    [llvm_i16_ty, llvm_v32i16_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_pbroadcast_d_gpr_128 :
          GCCBuiltin<"__builtin_ia32_pbroadcastd128_gpr_mask">,
          Intrinsic<[llvm_v4i32_ty],
                    [llvm_i32_ty, llvm_v4i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pbroadcast_d_gpr_256 :
          GCCBuiltin<"__builtin_ia32_pbroadcastd256_gpr_mask">,
          Intrinsic<[llvm_v8i32_ty],
                    [llvm_i32_ty, llvm_v8i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pbroadcast_d_gpr_512 :
          GCCBuiltin<"__builtin_ia32_pbroadcastd512_gpr_mask">,
          Intrinsic<[llvm_v16i32_ty],
                    [llvm_i32_ty, llvm_v16i32_ty, llvm_i16_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_pbroadcast_q_gpr_128 :
          GCCBuiltin<"__builtin_ia32_pbroadcastq128_gpr_mask">,
          Intrinsic<[llvm_v2i64_ty],
                    [llvm_i64_ty, llvm_v2i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pbroadcast_q_gpr_256 :
          GCCBuiltin<"__builtin_ia32_pbroadcastq256_gpr_mask">,
          Intrinsic<[llvm_v4i64_ty],
                    [llvm_i64_ty, llvm_v4i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pbroadcast_q_gpr_512 :
          GCCBuiltin<"__builtin_ia32_pbroadcastq512_gpr_mask">,
          Intrinsic<[llvm_v8i64_ty],
                    [llvm_i64_ty, llvm_v8i64_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_pbroadcast_q_mem_512 :
          GCCBuiltin<"__builtin_ia32_pbroadcastq512_mem_mask">,
          Intrinsic<[llvm_v8i64_ty],
                    [llvm_i64_ty, llvm_v8i64_ty, llvm_i8_ty], [IntrNoMem]>;
}

// Vector permutation
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_permd : GCCBuiltin<"__builtin_ia32_permvarsi256">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty, llvm_v8i32_ty],
                        [IntrNoMem]>;
  def int_x86_avx2_permps : GCCBuiltin<"__builtin_ia32_permvarsf256">,
              Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty, llvm_v8i32_ty],
                        [IntrNoMem]>;
  def int_x86_avx2_vperm2i128 : GCCBuiltin<"__builtin_ia32_permti256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_v4i64_ty, llvm_i8_ty], [IntrNoMem]>;
}

// Conditional load ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_maskload_d : GCCBuiltin<"__builtin_ia32_maskloadd">,
        Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty, llvm_v4i32_ty],
                  [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_maskload_q : GCCBuiltin<"__builtin_ia32_maskloadq">,
        Intrinsic<[llvm_v2i64_ty], [llvm_ptr_ty, llvm_v2i64_ty],
                  [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_maskload_d_256 : GCCBuiltin<"__builtin_ia32_maskloadd256">,
        Intrinsic<[llvm_v8i32_ty], [llvm_ptr_ty, llvm_v8i32_ty],
                  [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_maskload_q_256 : GCCBuiltin<"__builtin_ia32_maskloadq256">,
        Intrinsic<[llvm_v4i64_ty], [llvm_ptr_ty, llvm_v4i64_ty],
                  [IntrReadMem, IntrArgMemOnly]>;
}

// Conditional store ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_maskstore_d : GCCBuiltin<"__builtin_ia32_maskstored">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                  [IntrArgMemOnly]>;
  def int_x86_avx2_maskstore_q : GCCBuiltin<"__builtin_ia32_maskstoreq">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v2i64_ty, llvm_v2i64_ty],
                  [IntrArgMemOnly]>;
  def int_x86_avx2_maskstore_d_256 :
        GCCBuiltin<"__builtin_ia32_maskstored256">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v8i32_ty, llvm_v8i32_ty],
                  [IntrArgMemOnly]>;
  def int_x86_avx2_maskstore_q_256 :
        GCCBuiltin<"__builtin_ia32_maskstoreq256">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v4i64_ty, llvm_v4i64_ty],
                  [IntrArgMemOnly]>;
}

// Variable bit shift ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_psllv_d : GCCBuiltin<"__builtin_ia32_psllv4si">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_x86_avx2_psllv_d_256 : GCCBuiltin<"__builtin_ia32_psllv8si">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty, llvm_v8i32_ty],
                        [IntrNoMem]>;
  def int_x86_avx2_psllv_q : GCCBuiltin<"__builtin_ia32_psllv2di">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_avx2_psllv_q_256 : GCCBuiltin<"__builtin_ia32_psllv4di">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty, llvm_v4i64_ty],
                        [IntrNoMem]>;

  def int_x86_avx512_psllv_d_512 : GCCBuiltin<"__builtin_ia32_psllv16si">,
              Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty, llvm_v16i32_ty],
                        [IntrNoMem]>;
  def int_x86_avx512_psllv_q_512 : GCCBuiltin<"__builtin_ia32_psllv8di">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty, llvm_v8i64_ty],
                        [IntrNoMem]>;

  def int_x86_avx2_psrlv_d : GCCBuiltin<"__builtin_ia32_psrlv4si">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_x86_avx2_psrlv_d_256 : GCCBuiltin<"__builtin_ia32_psrlv8si">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty, llvm_v8i32_ty],
                        [IntrNoMem]>;
  def int_x86_avx2_psrlv_q : GCCBuiltin<"__builtin_ia32_psrlv2di">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_avx2_psrlv_q_256 : GCCBuiltin<"__builtin_ia32_psrlv4di">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty, llvm_v4i64_ty],
                        [IntrNoMem]>;

  def int_x86_avx512_psrlv_d_512 : GCCBuiltin<"__builtin_ia32_psrlv16si">,
              Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty, llvm_v16i32_ty],
                        [IntrNoMem]>;
  def int_x86_avx512_psrlv_q_512 : GCCBuiltin<"__builtin_ia32_psrlv8di">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty, llvm_v8i64_ty],
                        [IntrNoMem]>;

  def int_x86_avx2_psrav_d : GCCBuiltin<"__builtin_ia32_psrav4si">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_x86_avx2_psrav_d_256 : GCCBuiltin<"__builtin_ia32_psrav8si">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty, llvm_v8i32_ty],
                        [IntrNoMem]>;

  def int_x86_avx512_psrav_d_512 : GCCBuiltin<"__builtin_ia32_psrav16si">,
              Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty, llvm_v16i32_ty],
                        [IntrNoMem]>;
  def int_x86_avx512_psrav_q_128 : GCCBuiltin<"__builtin_ia32_psravq128">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_avx512_psrav_q_256 : GCCBuiltin<"__builtin_ia32_psravq256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty, llvm_v4i64_ty],
                        [IntrNoMem]>;
  def int_x86_avx512_psrav_q_512 : GCCBuiltin<"__builtin_ia32_psrav8di">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty, llvm_v8i64_ty],
                        [IntrNoMem]>;

  def int_x86_avx512_psllv_w_128 : GCCBuiltin<"__builtin_ia32_psllv8hi">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_x86_avx512_psllv_w_256 : GCCBuiltin<"__builtin_ia32_psllv16hi">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty, llvm_v16i16_ty],
                        [IntrNoMem]>;
  def int_x86_avx512_psllv_w_512 : GCCBuiltin<"__builtin_ia32_psllv32hi">,
              Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty, llvm_v32i16_ty],
                        [IntrNoMem]>;

  def int_x86_avx512_psrlv_w_128 : GCCBuiltin<"__builtin_ia32_psrlv8hi">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_x86_avx512_psrlv_w_256 : GCCBuiltin<"__builtin_ia32_psrlv16hi">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty, llvm_v16i16_ty],
                        [IntrNoMem]>;
  def int_x86_avx512_psrlv_w_512 : GCCBuiltin<"__builtin_ia32_psrlv32hi">,
              Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty, llvm_v32i16_ty],
                        [IntrNoMem]>;

  def int_x86_avx512_psrav_w_128 : GCCBuiltin<"__builtin_ia32_psrav8hi">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_x86_avx512_psrav_w_256 : GCCBuiltin<"__builtin_ia32_psrav16hi">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty, llvm_v16i16_ty],
                        [IntrNoMem]>;
  def int_x86_avx512_psrav_w_512 : GCCBuiltin<"__builtin_ia32_psrav32hi">,
              Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty, llvm_v32i16_ty],
                        [IntrNoMem]>;

  def int_x86_avx512_mask_prorv_d_128 : GCCBuiltin<"__builtin_ia32_prorvd128_mask">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty, llvm_v4i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_prorv_d_256 : GCCBuiltin<"__builtin_ia32_prorvd256_mask">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_v8i32_ty, llvm_v8i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_prorv_d_512 : GCCBuiltin<"__builtin_ia32_prorvd512_mask">,
              Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty,
                         llvm_v16i32_ty, llvm_v16i32_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_prorv_q_128 : GCCBuiltin<"__builtin_ia32_prorvq128_mask">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_v2i64_ty, llvm_v2i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_prorv_q_256 : GCCBuiltin<"__builtin_ia32_prorvq256_mask">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_v4i64_ty, llvm_v4i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_prorv_q_512 : GCCBuiltin<"__builtin_ia32_prorvq512_mask">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty,
                         llvm_v8i64_ty, llvm_v8i64_ty, llvm_i8_ty], [IntrNoMem]>;

   def int_x86_avx512_mask_prol_d_128 : GCCBuiltin<"__builtin_ia32_prold128_mask">,
              Intrinsic<[llvm_v4i32_ty] , [llvm_v4i32_ty,
                         llvm_i32_ty, llvm_v4i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_prol_d_256 : GCCBuiltin<"__builtin_ia32_prold256_mask">,
              Intrinsic<[llvm_v8i32_ty] , [llvm_v8i32_ty,
                         llvm_i32_ty, llvm_v8i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_prol_d_512 : GCCBuiltin<"__builtin_ia32_prold512_mask">,
              Intrinsic<[llvm_v16i32_ty] , [llvm_v16i32_ty,
                         llvm_i32_ty, llvm_v16i32_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_prol_q_128 : GCCBuiltin<"__builtin_ia32_prolq128_mask">,
              Intrinsic<[llvm_v2i64_ty] , [llvm_v2i64_ty,
                         llvm_i32_ty, llvm_v2i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_prol_q_256 : GCCBuiltin<"__builtin_ia32_prolq256_mask">,
              Intrinsic<[llvm_v4i64_ty] , [llvm_v4i64_ty,
                         llvm_i32_ty, llvm_v4i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_prol_q_512 : GCCBuiltin<"__builtin_ia32_prolq512_mask">,
              Intrinsic<[llvm_v8i64_ty] , [llvm_v8i64_ty,
                         llvm_i32_ty, llvm_v8i64_ty, llvm_i8_ty], [IntrNoMem]>;


  def int_x86_avx512_mask_prolv_d_128 : GCCBuiltin<"__builtin_ia32_prolvd128_mask">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_v4i32_ty, llvm_v4i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_prolv_d_256 : GCCBuiltin<"__builtin_ia32_prolvd256_mask">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_v8i32_ty, llvm_v8i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_prolv_d_512 : GCCBuiltin<"__builtin_ia32_prolvd512_mask">,
              Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty,
                         llvm_v16i32_ty, llvm_v16i32_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_prolv_q_128 : GCCBuiltin<"__builtin_ia32_prolvq128_mask">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_v2i64_ty, llvm_v2i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_prolv_q_256 : GCCBuiltin<"__builtin_ia32_prolvq256_mask">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_v4i64_ty, llvm_v4i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_prolv_q_512 : GCCBuiltin<"__builtin_ia32_prolvq512_mask">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty,
                         llvm_v8i64_ty, llvm_v8i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pror_d_128 : GCCBuiltin<"__builtin_ia32_prord128_mask">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty,
                         llvm_i32_ty, llvm_v4i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pror_d_256 : GCCBuiltin<"__builtin_ia32_prord256_mask">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                         llvm_i32_ty, llvm_v8i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pror_d_512 : GCCBuiltin<"__builtin_ia32_prord512_mask">,
              Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty,
                         llvm_i32_ty, llvm_v16i32_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pror_q_128 : GCCBuiltin<"__builtin_ia32_prorq128_mask">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty,
                         llvm_i32_ty, llvm_v2i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pror_q_256 : GCCBuiltin<"__builtin_ia32_prorq256_mask">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                         llvm_i32_ty, llvm_v4i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pror_q_512 : GCCBuiltin<"__builtin_ia32_prorq512_mask">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty,
                         llvm_i32_ty, llvm_v8i64_ty, llvm_i8_ty], [IntrNoMem]>;

}

// Gather ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_gather_d_pd : GCCBuiltin<"__builtin_ia32_gatherd_pd">,
      Intrinsic<[llvm_v2f64_ty],
        [llvm_v2f64_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_v2f64_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_gather_d_pd_256 : GCCBuiltin<"__builtin_ia32_gatherd_pd256">,
      Intrinsic<[llvm_v4f64_ty],
        [llvm_v4f64_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_v4f64_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_gather_q_pd : GCCBuiltin<"__builtin_ia32_gatherq_pd">,
      Intrinsic<[llvm_v2f64_ty],
        [llvm_v2f64_ty, llvm_ptr_ty, llvm_v2i64_ty, llvm_v2f64_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_gather_q_pd_256 : GCCBuiltin<"__builtin_ia32_gatherq_pd256">,
      Intrinsic<[llvm_v4f64_ty],
        [llvm_v4f64_ty, llvm_ptr_ty, llvm_v4i64_ty, llvm_v4f64_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_gather_d_ps : GCCBuiltin<"__builtin_ia32_gatherd_ps">,
      Intrinsic<[llvm_v4f32_ty],
        [llvm_v4f32_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_v4f32_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_gather_d_ps_256 : GCCBuiltin<"__builtin_ia32_gatherd_ps256">,
      Intrinsic<[llvm_v8f32_ty],
        [llvm_v8f32_ty, llvm_ptr_ty, llvm_v8i32_ty, llvm_v8f32_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_gather_q_ps : GCCBuiltin<"__builtin_ia32_gatherq_ps">,
      Intrinsic<[llvm_v4f32_ty],
        [llvm_v4f32_ty, llvm_ptr_ty, llvm_v2i64_ty, llvm_v4f32_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_gather_q_ps_256 : GCCBuiltin<"__builtin_ia32_gatherq_ps256">,
      Intrinsic<[llvm_v4f32_ty],
        [llvm_v4f32_ty, llvm_ptr_ty, llvm_v4i64_ty, llvm_v4f32_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx2_gather_d_q : GCCBuiltin<"__builtin_ia32_gatherd_q">,
      Intrinsic<[llvm_v2i64_ty],
        [llvm_v2i64_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_v2i64_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_gather_d_q_256 : GCCBuiltin<"__builtin_ia32_gatherd_q256">,
      Intrinsic<[llvm_v4i64_ty],
        [llvm_v4i64_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_v4i64_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_gather_q_q : GCCBuiltin<"__builtin_ia32_gatherq_q">,
      Intrinsic<[llvm_v2i64_ty],
        [llvm_v2i64_ty, llvm_ptr_ty, llvm_v2i64_ty, llvm_v2i64_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_gather_q_q_256 : GCCBuiltin<"__builtin_ia32_gatherq_q256">,
      Intrinsic<[llvm_v4i64_ty],
        [llvm_v4i64_ty, llvm_ptr_ty, llvm_v4i64_ty, llvm_v4i64_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_gather_d_d : GCCBuiltin<"__builtin_ia32_gatherd_d">,
      Intrinsic<[llvm_v4i32_ty],
        [llvm_v4i32_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_v4i32_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_gather_d_d_256 : GCCBuiltin<"__builtin_ia32_gatherd_d256">,
      Intrinsic<[llvm_v8i32_ty],
        [llvm_v8i32_ty, llvm_ptr_ty, llvm_v8i32_ty, llvm_v8i32_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_gather_q_d : GCCBuiltin<"__builtin_ia32_gatherq_d">,
      Intrinsic<[llvm_v4i32_ty],
        [llvm_v4i32_ty, llvm_ptr_ty, llvm_v2i64_ty, llvm_v4i32_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx2_gather_q_d_256 : GCCBuiltin<"__builtin_ia32_gatherq_d256">,
      Intrinsic<[llvm_v4i32_ty],
        [llvm_v4i32_ty, llvm_ptr_ty, llvm_v4i64_ty, llvm_v4i32_ty, llvm_i8_ty],
        [IntrReadMem, IntrArgMemOnly]>;
}

// Misc.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx2_pmovmskb : GCCBuiltin<"__builtin_ia32_pmovmskb256">,
              Intrinsic<[llvm_i32_ty], [llvm_v32i8_ty], [IntrNoMem]>;
  def int_x86_avx2_pshuf_b : GCCBuiltin<"__builtin_ia32_pshufb256">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                         llvm_v32i8_ty], [IntrNoMem]>;
  def int_x86_avx2_mpsadbw : GCCBuiltin<"__builtin_ia32_mpsadbw256">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v32i8_ty, llvm_v32i8_ty,
                         llvm_i8_ty], [IntrNoMem, Commutative]>;
  def int_x86_avx2_movntdqa : GCCBuiltin<"__builtin_ia32_movntdqa256">,
              Intrinsic<[llvm_v4i64_ty], [llvm_ptr_ty], [IntrReadMem]>;
}

//===----------------------------------------------------------------------===//
// FMA3 and FMA4

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_fma_vfmadd_ss : GCCBuiltin<"__builtin_ia32_vfmaddss">,
              Intrinsic<[llvm_v4f32_ty],
                        [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmadd_sd : GCCBuiltin<"__builtin_ia32_vfmaddsd">,
              Intrinsic<[llvm_v2f64_ty],
                        [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmadd_ps : GCCBuiltin<"__builtin_ia32_vfmaddps">,
              Intrinsic<[llvm_v4f32_ty],
                        [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmadd_pd : GCCBuiltin<"__builtin_ia32_vfmaddpd">,
              Intrinsic<[llvm_v2f64_ty],
                        [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmadd_ps_256 : GCCBuiltin<"__builtin_ia32_vfmaddps256">,
              Intrinsic<[llvm_v8f32_ty],
                        [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmadd_pd_256 : GCCBuiltin<"__builtin_ia32_vfmaddpd256">,
              Intrinsic<[llvm_v4f64_ty],
                        [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty],
                        [IntrNoMem]>;

  def int_x86_fma_vfmsub_ss : GCCBuiltin<"__builtin_ia32_vfmsubss">,
              Intrinsic<[llvm_v4f32_ty],
                        [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmsub_sd : GCCBuiltin<"__builtin_ia32_vfmsubsd">,
              Intrinsic<[llvm_v2f64_ty],
                        [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmsub_ps : GCCBuiltin<"__builtin_ia32_vfmsubps">,
              Intrinsic<[llvm_v4f32_ty],
                        [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmsub_pd : GCCBuiltin<"__builtin_ia32_vfmsubpd">,
              Intrinsic<[llvm_v2f64_ty],
                        [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmsub_ps_256 : GCCBuiltin<"__builtin_ia32_vfmsubps256">,
              Intrinsic<[llvm_v8f32_ty],
                        [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmsub_pd_256 : GCCBuiltin<"__builtin_ia32_vfmsubpd256">,
              Intrinsic<[llvm_v4f64_ty],
                        [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfnmadd_ss : GCCBuiltin<"__builtin_ia32_vfnmaddss">,
              Intrinsic<[llvm_v4f32_ty],
                        [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfnmadd_sd : GCCBuiltin<"__builtin_ia32_vfnmaddsd">,
              Intrinsic<[llvm_v2f64_ty],
                        [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfnmadd_ps : GCCBuiltin<"__builtin_ia32_vfnmaddps">,
              Intrinsic<[llvm_v4f32_ty],
                        [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfnmadd_pd : GCCBuiltin<"__builtin_ia32_vfnmaddpd">,
              Intrinsic<[llvm_v2f64_ty],
                        [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfnmadd_ps_256 : GCCBuiltin<"__builtin_ia32_vfnmaddps256">,
              Intrinsic<[llvm_v8f32_ty],
                        [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfnmadd_pd_256 : GCCBuiltin<"__builtin_ia32_vfnmaddpd256">,
              Intrinsic<[llvm_v4f64_ty],
                        [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfnmsub_ss : GCCBuiltin<"__builtin_ia32_vfnmsubss">,
              Intrinsic<[llvm_v4f32_ty],
                        [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfnmsub_sd : GCCBuiltin<"__builtin_ia32_vfnmsubsd">,
              Intrinsic<[llvm_v2f64_ty],
                        [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfnmsub_ps : GCCBuiltin<"__builtin_ia32_vfnmsubps">,
              Intrinsic<[llvm_v4f32_ty],
                        [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfnmsub_pd : GCCBuiltin<"__builtin_ia32_vfnmsubpd">,
              Intrinsic<[llvm_v2f64_ty],
                        [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfnmsub_ps_256 : GCCBuiltin<"__builtin_ia32_vfnmsubps256">,
              Intrinsic<[llvm_v8f32_ty],
                        [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfnmsub_pd_256 : GCCBuiltin<"__builtin_ia32_vfnmsubpd256">,
              Intrinsic<[llvm_v4f64_ty],
                        [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmaddsub_ps : GCCBuiltin<"__builtin_ia32_vfmaddsubps">,
              Intrinsic<[llvm_v4f32_ty],
                        [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmaddsub_pd : GCCBuiltin<"__builtin_ia32_vfmaddsubpd">,
              Intrinsic<[llvm_v2f64_ty],
                        [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmaddsub_ps_256 :
               GCCBuiltin<"__builtin_ia32_vfmaddsubps256">,
              Intrinsic<[llvm_v8f32_ty],
                        [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmaddsub_pd_256 :
              GCCBuiltin<"__builtin_ia32_vfmaddsubpd256">,
              Intrinsic<[llvm_v4f64_ty],
                        [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmsubadd_ps : GCCBuiltin<"__builtin_ia32_vfmsubaddps">,
              Intrinsic<[llvm_v4f32_ty],
                        [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmsubadd_pd : GCCBuiltin<"__builtin_ia32_vfmsubaddpd">,
              Intrinsic<[llvm_v2f64_ty],
                        [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmsubadd_ps_256 :
              GCCBuiltin<"__builtin_ia32_vfmsubaddps256">,
              Intrinsic<[llvm_v8f32_ty],
                        [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty],
                        [IntrNoMem]>;
  def int_x86_fma_vfmsubadd_pd_256 :
              GCCBuiltin<"__builtin_ia32_vfmsubaddpd256">,
              Intrinsic<[llvm_v4f64_ty],
                        [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty],
                        [IntrNoMem]>;

  def int_x86_avx512_mask_vfmadd_pd_128 :
         GCCBuiltin<"__builtin_ia32_vfmaddpd128_mask">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmadd_pd_128 :
         GCCBuiltin<"__builtin_ia32_vfmaddpd128_mask3">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vfmadd_pd_128 :
         GCCBuiltin<"__builtin_ia32_vfmaddpd128_maskz">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfmadd_pd_256 :
         GCCBuiltin<"__builtin_ia32_vfmaddpd256_mask">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmadd_pd_256 :
         GCCBuiltin<"__builtin_ia32_vfmaddpd256_mask3">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vfmadd_pd_256 :
         GCCBuiltin<"__builtin_ia32_vfmaddpd256_maskz">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfmadd_pd_512 :
         GCCBuiltin<"__builtin_ia32_vfmaddpd512_mask">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty, llvm_v8f64_ty, llvm_v8f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmadd_pd_512 :
         GCCBuiltin<"__builtin_ia32_vfmaddpd512_mask3">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty, llvm_v8f64_ty, llvm_v8f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_maskz_vfmadd_pd_512 :
         GCCBuiltin<"__builtin_ia32_vfmaddpd512_maskz">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty, llvm_v8f64_ty, llvm_v8f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_vfmadd_ps_128 :
         GCCBuiltin<"__builtin_ia32_vfmaddps128_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmadd_ps_128 :
         GCCBuiltin<"__builtin_ia32_vfmaddps128_mask3">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vfmadd_ps_128 :
         GCCBuiltin<"__builtin_ia32_vfmaddps128_maskz">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfmadd_ps_256 :
         GCCBuiltin<"__builtin_ia32_vfmaddps256_mask">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmadd_ps_256 :
         GCCBuiltin<"__builtin_ia32_vfmaddps256_mask3">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vfmadd_ps_256 :
         GCCBuiltin<"__builtin_ia32_vfmaddps256_maskz">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfmadd_ps_512 :
         GCCBuiltin<"__builtin_ia32_vfmaddps512_mask">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty, llvm_v16f32_ty, llvm_v16f32_ty,  llvm_i16_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmadd_ps_512 :
         GCCBuiltin<"__builtin_ia32_vfmaddps512_mask3">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty, llvm_v16f32_ty, llvm_v16f32_ty,  llvm_i16_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_maskz_vfmadd_ps_512 :
         GCCBuiltin<"__builtin_ia32_vfmaddps512_maskz">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty, llvm_v16f32_ty, llvm_v16f32_ty,  llvm_i16_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_vfmaddsub_pd_128 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubpd128_mask">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmaddsub_pd_128 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubpd128_mask3">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vfmaddsub_pd_128 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubpd128_maskz">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfmaddsub_pd_256 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubpd256_mask">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmaddsub_pd_256 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubpd256_mask3">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vfmaddsub_pd_256 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubpd256_maskz">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfmaddsub_pd_512 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubpd512_mask">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty, llvm_v8f64_ty, llvm_v8f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmaddsub_pd_512 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubpd512_mask3">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty, llvm_v8f64_ty, llvm_v8f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_maskz_vfmaddsub_pd_512 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubpd512_maskz">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty, llvm_v8f64_ty, llvm_v8f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_vfmaddsub_ps_128 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubps128_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmaddsub_ps_128 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubps128_mask3">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vfmaddsub_ps_128 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubps128_maskz">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfmaddsub_ps_256 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubps256_mask">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmaddsub_ps_256 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubps256_mask3">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_maskz_vfmaddsub_ps_256 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubps256_maskz">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfmaddsub_ps_512 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubps512_mask">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty, llvm_v16f32_ty, llvm_v16f32_ty,  llvm_i16_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmaddsub_ps_512 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubps512_mask3">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty, llvm_v16f32_ty, llvm_v16f32_ty,  llvm_i16_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_maskz_vfmaddsub_ps_512 :
         GCCBuiltin<"__builtin_ia32_vfmaddsubps512_maskz">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty, llvm_v16f32_ty, llvm_v16f32_ty,  llvm_i16_ty,
          llvm_i32_ty], [IntrNoMem]>;


  def int_x86_avx512_mask_vfmadd_sd :
         GCCBuiltin<"__builtin_ia32_vfmaddsd3_mask">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_vfmadd_ss :
         GCCBuiltin<"__builtin_ia32_vfmaddss3_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_maskz_vfmadd_sd :
         GCCBuiltin<"__builtin_ia32_vfmaddsd3_maskz">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_maskz_vfmadd_ss :
         GCCBuiltin<"__builtin_ia32_vfmaddss3_maskz">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmadd_sd :
         GCCBuiltin<"__builtin_ia32_vfmaddsd3_mask3">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmadd_ss :
         GCCBuiltin<"__builtin_ia32_vfmaddss3_mask3">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmsub_sd :
         GCCBuiltin<"__builtin_ia32_vfmsubsd3_mask3">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmsub_ss :
         GCCBuiltin<"__builtin_ia32_vfmsubss3_mask3">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmsub_pd_128 :
         GCCBuiltin<"__builtin_ia32_vfmsubpd128_mask3">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmsub_pd_256 :
         GCCBuiltin<"__builtin_ia32_vfmsubpd256_mask3">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmsub_pd_512 :
         GCCBuiltin<"__builtin_ia32_vfmsubpd512_mask3">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty, llvm_v8f64_ty, llvm_v8f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmsub_ps_128 :
         GCCBuiltin<"__builtin_ia32_vfmsubps128_mask3">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmsub_ps_256 :
         GCCBuiltin<"__builtin_ia32_vfmsubps256_mask3">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmsub_ps_512 :
         GCCBuiltin<"__builtin_ia32_vfmsubps512_mask3">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty, llvm_v16f32_ty, llvm_v16f32_ty,  llvm_i16_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmsubadd_pd_128 :
         GCCBuiltin<"__builtin_ia32_vfmsubaddpd128_mask3">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmsubadd_pd_256 :
         GCCBuiltin<"__builtin_ia32_vfmsubaddpd256_mask3">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmsubadd_pd_512 :
         GCCBuiltin<"__builtin_ia32_vfmsubaddpd512_mask3">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty, llvm_v8f64_ty, llvm_v8f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmsubadd_ps_128 :
         GCCBuiltin<"__builtin_ia32_vfmsubaddps128_mask3">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmsubadd_ps_256 :
         GCCBuiltin<"__builtin_ia32_vfmsubaddps256_mask3">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfmsubadd_ps_512 :
         GCCBuiltin<"__builtin_ia32_vfmsubaddps512_mask3">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty, llvm_v16f32_ty, llvm_v16f32_ty,  llvm_i16_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_vfnmadd_pd_128 :
         GCCBuiltin<"__builtin_ia32_vfnmaddpd128_mask">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfnmadd_pd_256 :
         GCCBuiltin<"__builtin_ia32_vfnmaddpd256_mask">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfnmadd_pd_512 :
         GCCBuiltin<"__builtin_ia32_vfnmaddpd512_mask">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty, llvm_v8f64_ty, llvm_v8f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_vfnmadd_ps_128 :
         GCCBuiltin<"__builtin_ia32_vfnmaddps128_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfnmadd_ps_256 :
         GCCBuiltin<"__builtin_ia32_vfnmaddps256_mask">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfnmadd_ps_512 :
         GCCBuiltin<"__builtin_ia32_vfnmaddps512_mask">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty, llvm_v16f32_ty, llvm_v16f32_ty,  llvm_i16_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfnmsub_sd :
         GCCBuiltin<"__builtin_ia32_vfnmsubsd3_mask3">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfnmsub_ss :
         GCCBuiltin<"__builtin_ia32_vfnmsubss3_mask3">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_vfnmsub_pd_128 :
         GCCBuiltin<"__builtin_ia32_vfnmsubpd128_mask">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfnmsub_pd_128 :
         GCCBuiltin<"__builtin_ia32_vfnmsubpd128_mask3">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfnmsub_pd_256 :
         GCCBuiltin<"__builtin_ia32_vfnmsubpd256_mask">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfnmsub_pd_256 :
         GCCBuiltin<"__builtin_ia32_vfnmsubpd256_mask3">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfnmsub_pd_512 :
         GCCBuiltin<"__builtin_ia32_vfnmsubpd512_mask">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty, llvm_v8f64_ty, llvm_v8f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfnmsub_pd_512 :
         GCCBuiltin<"__builtin_ia32_vfnmsubpd512_mask3">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty, llvm_v8f64_ty, llvm_v8f64_ty,  llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_vfnmsub_ps_128 :
         GCCBuiltin<"__builtin_ia32_vfnmsubps128_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfnmsub_ps_128 :
         GCCBuiltin<"__builtin_ia32_vfnmsubps128_mask3">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfnmsub_ps_256 :
         GCCBuiltin<"__builtin_ia32_vfnmsubps256_mask">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask3_vfnmsub_ps_256 :
         GCCBuiltin<"__builtin_ia32_vfnmsubps256_mask3">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_vfnmsub_ps_512 :
         GCCBuiltin<"__builtin_ia32_vfnmsubps512_mask">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty, llvm_v16f32_ty, llvm_v16f32_ty,  llvm_i16_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask3_vfnmsub_ps_512 :
         GCCBuiltin<"__builtin_ia32_vfnmsubps512_mask3">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty, llvm_v16f32_ty, llvm_v16f32_ty,  llvm_i16_ty,
          llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_vpmadd52h_uq_128 :
              GCCBuiltin<"__builtin_ia32_vpmadd52huq128_mask">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty,
                         llvm_v2i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_maskz_vpmadd52h_uq_128 :
              GCCBuiltin<"__builtin_ia32_vpmadd52huq128_maskz">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty,
                         llvm_v2i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_vpmadd52l_uq_128 :
              GCCBuiltin<"__builtin_ia32_vpmadd52luq128_mask">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty,
                         llvm_v2i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_maskz_vpmadd52l_uq_128 :
              GCCBuiltin<"__builtin_ia32_vpmadd52luq128_maskz">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty,
                         llvm_v2i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_vpmadd52h_uq_256 :
              GCCBuiltin<"__builtin_ia32_vpmadd52huq256_mask">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty, llvm_v4i64_ty,
                         llvm_v4i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_maskz_vpmadd52h_uq_256 :
              GCCBuiltin<"__builtin_ia32_vpmadd52huq256_maskz">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty, llvm_v4i64_ty,
                         llvm_v4i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_vpmadd52l_uq_256 :
              GCCBuiltin<"__builtin_ia32_vpmadd52luq256_mask">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty, llvm_v4i64_ty,
                         llvm_v4i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_maskz_vpmadd52l_uq_256 :
              GCCBuiltin<"__builtin_ia32_vpmadd52luq256_maskz">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty, llvm_v4i64_ty,
                         llvm_v4i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_vpmadd52h_uq_512 :
              GCCBuiltin<"__builtin_ia32_vpmadd52huq512_mask">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty, llvm_v8i64_ty,
                         llvm_v8i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_maskz_vpmadd52h_uq_512 :
              GCCBuiltin<"__builtin_ia32_vpmadd52huq512_maskz">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty, llvm_v8i64_ty,
                         llvm_v8i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_vpmadd52l_uq_512 :
              GCCBuiltin<"__builtin_ia32_vpmadd52luq512_mask">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty, llvm_v8i64_ty,
                         llvm_v8i64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_maskz_vpmadd52l_uq_512 :
              GCCBuiltin<"__builtin_ia32_vpmadd52luq512_maskz">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty, llvm_v8i64_ty,
                         llvm_v8i64_ty, llvm_i8_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// XOP

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_xop_vpermil2pd : GCCBuiltin<"__builtin_ia32_vpermil2pd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                                          llvm_v2i64_ty, llvm_i8_ty],
                        [IntrNoMem]>;

  def int_x86_xop_vpermil2pd_256 :
              GCCBuiltin<"__builtin_ia32_vpermil2pd256">,
              Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty, llvm_v4f64_ty,
                                          llvm_v4i64_ty, llvm_i8_ty],
                        [IntrNoMem]>;

  def int_x86_xop_vpermil2ps : GCCBuiltin<"__builtin_ia32_vpermil2ps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                                          llvm_v4i32_ty, llvm_i8_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpermil2ps_256 :
              GCCBuiltin<"__builtin_ia32_vpermil2ps256">,
              Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty, llvm_v8f32_ty,
                                          llvm_v8i32_ty, llvm_i8_ty],
                        [IntrNoMem]>;

  def int_x86_xop_vfrcz_pd : GCCBuiltin<"__builtin_ia32_vfrczpd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_xop_vfrcz_ps : GCCBuiltin<"__builtin_ia32_vfrczps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_xop_vfrcz_sd : GCCBuiltin<"__builtin_ia32_vfrczsd">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty], [IntrNoMem]>;
  def int_x86_xop_vfrcz_ss : GCCBuiltin<"__builtin_ia32_vfrczss">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_x86_xop_vfrcz_pd_256 : GCCBuiltin<"__builtin_ia32_vfrczpd256">,
              Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty], [IntrNoMem]>;
  def int_x86_xop_vfrcz_ps_256 : GCCBuiltin<"__builtin_ia32_vfrczps256">,
              Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty], [IntrNoMem]>;

  def int_x86_xop_vpcomb : GCCBuiltin<"__builtin_ia32_vpcomb">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty,
                         llvm_i8_ty], [IntrNoMem]>;
  def int_x86_xop_vpcomw : GCCBuiltin<"__builtin_ia32_vpcomw">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                         llvm_i8_ty], [IntrNoMem]>;
  def int_x86_xop_vpcomd : GCCBuiltin<"__builtin_ia32_vpcomd">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty,
                         llvm_i8_ty], [IntrNoMem]>;
  def int_x86_xop_vpcomq : GCCBuiltin<"__builtin_ia32_vpcomq">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty,
                         llvm_i8_ty], [IntrNoMem]>;
  def int_x86_xop_vpcomub : GCCBuiltin<"__builtin_ia32_vpcomub">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty,
                         llvm_i8_ty], [IntrNoMem]>;
  def int_x86_xop_vpcomuw : GCCBuiltin<"__builtin_ia32_vpcomuw">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                         llvm_i8_ty], [IntrNoMem]>;
  def int_x86_xop_vpcomud : GCCBuiltin<"__builtin_ia32_vpcomud">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty,
                         llvm_i8_ty], [IntrNoMem]>;
  def int_x86_xop_vpcomuq : GCCBuiltin<"__builtin_ia32_vpcomuq">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty,
                         llvm_i8_ty], [IntrNoMem]>;

  def int_x86_xop_vphaddbd :
              GCCBuiltin<"__builtin_ia32_vphaddbd">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_xop_vphaddbq :
              GCCBuiltin<"__builtin_ia32_vphaddbq">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_xop_vphaddbw :
              GCCBuiltin<"__builtin_ia32_vphaddbw">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_xop_vphadddq :
              GCCBuiltin<"__builtin_ia32_vphadddq">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_xop_vphaddubd :
              GCCBuiltin<"__builtin_ia32_vphaddubd">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_xop_vphaddubq :
              GCCBuiltin<"__builtin_ia32_vphaddubq">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_xop_vphaddubw :
              GCCBuiltin<"__builtin_ia32_vphaddubw">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_xop_vphaddudq :
              GCCBuiltin<"__builtin_ia32_vphaddudq">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_xop_vphadduwd :
              GCCBuiltin<"__builtin_ia32_vphadduwd">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_xop_vphadduwq :
              GCCBuiltin<"__builtin_ia32_vphadduwq">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_xop_vphaddwd :
              GCCBuiltin<"__builtin_ia32_vphaddwd">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_xop_vphaddwq :
              GCCBuiltin<"__builtin_ia32_vphaddwq">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_xop_vphsubbw :
              GCCBuiltin<"__builtin_ia32_vphsubbw">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_xop_vphsubdq :
              GCCBuiltin<"__builtin_ia32_vphsubdq">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_xop_vphsubwd :
              GCCBuiltin<"__builtin_ia32_vphsubwd">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_xop_vpmacsdd :
              GCCBuiltin<"__builtin_ia32_vpmacsdd">,
              Intrinsic<[llvm_v4i32_ty],
                        [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpmacsdqh :
              GCCBuiltin<"__builtin_ia32_vpmacsdqh">,
              Intrinsic<[llvm_v2i64_ty],
                        [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpmacsdql :
              GCCBuiltin<"__builtin_ia32_vpmacsdql">,
              Intrinsic<[llvm_v2i64_ty],
                        [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpmacssdd :
              GCCBuiltin<"__builtin_ia32_vpmacssdd">,
              Intrinsic<[llvm_v4i32_ty],
                        [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpmacssdqh :
              GCCBuiltin<"__builtin_ia32_vpmacssdqh">,
              Intrinsic<[llvm_v2i64_ty],
                        [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpmacssdql :
              GCCBuiltin<"__builtin_ia32_vpmacssdql">,
              Intrinsic<[llvm_v2i64_ty],
                        [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpmacsswd :
              GCCBuiltin<"__builtin_ia32_vpmacsswd">,
              Intrinsic<[llvm_v4i32_ty],
                        [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpmacssww :
              GCCBuiltin<"__builtin_ia32_vpmacssww">,
              Intrinsic<[llvm_v8i16_ty],
                        [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpmacswd :
              GCCBuiltin<"__builtin_ia32_vpmacswd">,
              Intrinsic<[llvm_v4i32_ty],
                        [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpmacsww :
              GCCBuiltin<"__builtin_ia32_vpmacsww">,
              Intrinsic<[llvm_v8i16_ty],
                        [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpmadcsswd :
              GCCBuiltin<"__builtin_ia32_vpmadcsswd">,
              Intrinsic<[llvm_v4i32_ty],
                        [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpmadcswd :
              GCCBuiltin<"__builtin_ia32_vpmadcswd">,
              Intrinsic<[llvm_v4i32_ty],
                        [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpperm :
              GCCBuiltin<"__builtin_ia32_vpperm">,
              Intrinsic<[llvm_v16i8_ty],
                        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                        [IntrNoMem]>;

  def int_x86_xop_vprotb : GCCBuiltin<"__builtin_ia32_vprotb">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vprotd : GCCBuiltin<"__builtin_ia32_vprotd">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vprotq : GCCBuiltin<"__builtin_ia32_vprotq">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vprotw : GCCBuiltin<"__builtin_ia32_vprotw">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vprotbi : GCCBuiltin<"__builtin_ia32_vprotbi">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_i8_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vprotdi : GCCBuiltin<"__builtin_ia32_vprotdi">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_i8_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vprotqi : GCCBuiltin<"__builtin_ia32_vprotqi">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_i8_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vprotwi : GCCBuiltin<"__builtin_ia32_vprotwi">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_i8_ty],
                        [IntrNoMem]>;

  def int_x86_xop_vpshab :
              GCCBuiltin<"__builtin_ia32_vpshab">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpshad :
              GCCBuiltin<"__builtin_ia32_vpshad">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpshaq :
              GCCBuiltin<"__builtin_ia32_vpshaq">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpshaw :
              GCCBuiltin<"__builtin_ia32_vpshaw">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpshlb :
              GCCBuiltin<"__builtin_ia32_vpshlb">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpshld :
              GCCBuiltin<"__builtin_ia32_vpshld">,
              Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpshlq :
              GCCBuiltin<"__builtin_ia32_vpshlq">,
              Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty],
                        [IntrNoMem]>;
  def int_x86_xop_vpshlw :
              GCCBuiltin<"__builtin_ia32_vpshlw">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty],
                        [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// MMX

// Empty MMX state op.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_mmx_emms  : GCCBuiltin<"__builtin_ia32_emms">,
              Intrinsic<[], [], []>;
  def int_x86_mmx_femms : GCCBuiltin<"__builtin_ia32_femms">,
              Intrinsic<[], [], []>;
}

// Integer arithmetic ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  // Addition
  def int_x86_mmx_padd_b : GCCBuiltin<"__builtin_ia32_paddb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_mmx_padd_w : GCCBuiltin<"__builtin_ia32_paddw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_mmx_padd_d : GCCBuiltin<"__builtin_ia32_paddd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_mmx_padd_q : GCCBuiltin<"__builtin_ia32_paddq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem, Commutative]>;

  def int_x86_mmx_padds_b : GCCBuiltin<"__builtin_ia32_paddsb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_padds_w : GCCBuiltin<"__builtin_ia32_paddsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;

  def int_x86_mmx_paddus_b : GCCBuiltin<"__builtin_ia32_paddusb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_paddus_w : GCCBuiltin<"__builtin_ia32_paddusw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;

  // Subtraction
  def int_x86_mmx_psub_b : GCCBuiltin<"__builtin_ia32_psubb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_psub_w : GCCBuiltin<"__builtin_ia32_psubw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_psub_d : GCCBuiltin<"__builtin_ia32_psubd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_psub_q : GCCBuiltin<"__builtin_ia32_psubq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;

  def int_x86_mmx_psubs_b : GCCBuiltin<"__builtin_ia32_psubsb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_psubs_w : GCCBuiltin<"__builtin_ia32_psubsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;

  def int_x86_mmx_psubus_b : GCCBuiltin<"__builtin_ia32_psubusb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_psubus_w : GCCBuiltin<"__builtin_ia32_psubusw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;

  // Multiplication
  def int_x86_mmx_pmulh_w : GCCBuiltin<"__builtin_ia32_pmulhw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pmull_w : GCCBuiltin<"__builtin_ia32_pmullw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pmulhu_w : GCCBuiltin<"__builtin_ia32_pmulhuw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pmulu_dq : GCCBuiltin<"__builtin_ia32_pmuludq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pmadd_wd : GCCBuiltin<"__builtin_ia32_pmaddwd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;

  // Bitwise operations
  def int_x86_mmx_pand : GCCBuiltin<"__builtin_ia32_pand">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_mmx_pandn : GCCBuiltin<"__builtin_ia32_pandn">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_por : GCCBuiltin<"__builtin_ia32_por">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem, Commutative]>;
  def int_x86_mmx_pxor : GCCBuiltin<"__builtin_ia32_pxor">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem, Commutative]>;

  // Averages
  def int_x86_mmx_pavg_b : GCCBuiltin<"__builtin_ia32_pavgb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pavg_w : GCCBuiltin<"__builtin_ia32_pavgw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;

  // Maximum
  def int_x86_mmx_pmaxu_b : GCCBuiltin<"__builtin_ia32_pmaxub">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pmaxs_w : GCCBuiltin<"__builtin_ia32_pmaxsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;

  // Minimum
  def int_x86_mmx_pminu_b : GCCBuiltin<"__builtin_ia32_pminub">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pmins_w : GCCBuiltin<"__builtin_ia32_pminsw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;

  // Packed sum of absolute differences
  def int_x86_mmx_psad_bw : GCCBuiltin<"__builtin_ia32_psadbw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
}

// Integer shift ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  // Shift left logical
  def int_x86_mmx_psll_w : GCCBuiltin<"__builtin_ia32_psllw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_psll_d : GCCBuiltin<"__builtin_ia32_pslld">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_psll_q : GCCBuiltin<"__builtin_ia32_psllq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;

  def int_x86_mmx_psrl_w : GCCBuiltin<"__builtin_ia32_psrlw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_psrl_d : GCCBuiltin<"__builtin_ia32_psrld">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_psrl_q : GCCBuiltin<"__builtin_ia32_psrlq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;

  def int_x86_mmx_psra_w : GCCBuiltin<"__builtin_ia32_psraw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_psra_d : GCCBuiltin<"__builtin_ia32_psrad">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;

  def int_x86_mmx_pslli_w : GCCBuiltin<"__builtin_ia32_psllwi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_mmx_pslli_d : GCCBuiltin<"__builtin_ia32_pslldi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_mmx_pslli_q : GCCBuiltin<"__builtin_ia32_psllqi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;

  def int_x86_mmx_psrli_w : GCCBuiltin<"__builtin_ia32_psrlwi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_mmx_psrli_d : GCCBuiltin<"__builtin_ia32_psrldi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_mmx_psrli_q : GCCBuiltin<"__builtin_ia32_psrlqi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;

  def int_x86_mmx_psrai_w : GCCBuiltin<"__builtin_ia32_psrawi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_mmx_psrai_d : GCCBuiltin<"__builtin_ia32_psradi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_i32_ty], [IntrNoMem]>;
}
// Permute
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx512_mask_permvar_df_256 : GCCBuiltin<"__builtin_ia32_permvardf256_mask">,
              Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty,
                        llvm_v4i64_ty, llvm_v4f64_ty, llvm_i8_ty],  [IntrNoMem]>;
  def int_x86_avx512_mask_permvar_df_512 : GCCBuiltin<"__builtin_ia32_permvardf512_mask">,
              Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty,
                        llvm_v8i64_ty, llvm_v8f64_ty, llvm_i8_ty],  [IntrNoMem]>;
  def int_x86_avx512_mask_permvar_di_256 : GCCBuiltin<"__builtin_ia32_permvardi256_mask">,
              Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty,
                        llvm_v4i64_ty, llvm_v4i64_ty, llvm_i8_ty],  [IntrNoMem]>;
  def int_x86_avx512_mask_permvar_di_512 : GCCBuiltin<"__builtin_ia32_permvardi512_mask">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty,
                        llvm_v8i64_ty, llvm_v8i64_ty, llvm_i8_ty],  [IntrNoMem]>;
  def int_x86_avx512_mask_permvar_hi_128 : GCCBuiltin<"__builtin_ia32_permvarhi128_mask">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty,
                        llvm_v8i16_ty, llvm_v8i16_ty, llvm_i8_ty],  [IntrNoMem]>;
  def int_x86_avx512_mask_permvar_hi_256 : GCCBuiltin<"__builtin_ia32_permvarhi256_mask">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty,
                        llvm_v16i16_ty, llvm_v16i16_ty, llvm_i16_ty],  [IntrNoMem]>;
  def int_x86_avx512_mask_permvar_hi_512 : GCCBuiltin<"__builtin_ia32_permvarhi512_mask">,
              Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty,
                        llvm_v32i16_ty, llvm_v32i16_ty, llvm_i32_ty],  [IntrNoMem]>;
  def int_x86_avx512_mask_permvar_qi_128 : GCCBuiltin<"__builtin_ia32_permvarqi128_mask">,
              Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty,
                        llvm_v16i8_ty, llvm_v16i8_ty, llvm_i16_ty],  [IntrNoMem]>;
  def int_x86_avx512_mask_permvar_qi_256 : GCCBuiltin<"__builtin_ia32_permvarqi256_mask">,
              Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty,
                        llvm_v32i8_ty, llvm_v32i8_ty, llvm_i32_ty],  [IntrNoMem]>;
  def int_x86_avx512_mask_permvar_qi_512 : GCCBuiltin<"__builtin_ia32_permvarqi512_mask">,
              Intrinsic<[llvm_v64i8_ty], [llvm_v64i8_ty,
                        llvm_v64i8_ty, llvm_v64i8_ty, llvm_i64_ty],  [IntrNoMem]>;
  def int_x86_avx512_mask_permvar_sf_256 : GCCBuiltin<"__builtin_ia32_permvarsf256_mask">,
              Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty,
                        llvm_v8i32_ty, llvm_v8f32_ty, llvm_i8_ty],  [IntrNoMem]>;
  def int_x86_avx512_mask_permvar_sf_512 : GCCBuiltin<"__builtin_ia32_permvarsf512_mask">,
              Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty,
                        llvm_v16i32_ty, llvm_v16f32_ty, llvm_i16_ty],  [IntrNoMem]>;
  def int_x86_avx512_mask_permvar_si_256 : GCCBuiltin<"__builtin_ia32_permvarsi256_mask">,
              Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty,
                        llvm_v8i32_ty, llvm_v8i32_ty, llvm_i8_ty],  [IntrNoMem]>;
  def int_x86_avx512_mask_permvar_si_512 : GCCBuiltin<"__builtin_ia32_permvarsi512_mask">,
              Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty,
                        llvm_v16i32_ty, llvm_v16i32_ty, llvm_i16_ty],  [IntrNoMem]>;
}
// Pack ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_mmx_packsswb : GCCBuiltin<"__builtin_ia32_packsswb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_packssdw : GCCBuiltin<"__builtin_ia32_packssdw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_packuswb : GCCBuiltin<"__builtin_ia32_packuswb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
}

// Unpacking ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_mmx_punpckhbw : GCCBuiltin<"__builtin_ia32_punpckhbw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_punpckhwd : GCCBuiltin<"__builtin_ia32_punpckhwd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_punpckhdq : GCCBuiltin<"__builtin_ia32_punpckhdq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_punpcklbw : GCCBuiltin<"__builtin_ia32_punpcklbw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_punpcklwd : GCCBuiltin<"__builtin_ia32_punpcklwd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
  def int_x86_mmx_punpckldq : GCCBuiltin<"__builtin_ia32_punpckldq">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty, llvm_x86mmx_ty],
                        [IntrNoMem]>;
}

// Integer comparison ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_mmx_pcmpeq_b : GCCBuiltin<"__builtin_ia32_pcmpeqb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pcmpeq_w : GCCBuiltin<"__builtin_ia32_pcmpeqw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;
  def int_x86_mmx_pcmpeq_d : GCCBuiltin<"__builtin_ia32_pcmpeqd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem, Commutative]>;

  def int_x86_mmx_pcmpgt_b : GCCBuiltin<"__builtin_ia32_pcmpgtb">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_pcmpgt_w : GCCBuiltin<"__builtin_ia32_pcmpgtw">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
  def int_x86_mmx_pcmpgt_d : GCCBuiltin<"__builtin_ia32_pcmpgtd">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                         llvm_x86mmx_ty], [IntrNoMem]>;
}

// Misc.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_mmx_maskmovq : GCCBuiltin<"__builtin_ia32_maskmovq">,
              Intrinsic<[], [llvm_x86mmx_ty, llvm_x86mmx_ty, llvm_ptr_ty], []>;

  def int_x86_mmx_pmovmskb : GCCBuiltin<"__builtin_ia32_pmovmskb">,
              Intrinsic<[llvm_i32_ty], [llvm_x86mmx_ty], [IntrNoMem]>;

  def int_x86_mmx_movnt_dq : GCCBuiltin<"__builtin_ia32_movntq">,
              Intrinsic<[], [llvm_ptrx86mmx_ty, llvm_x86mmx_ty], []>;

  def int_x86_mmx_palignr_b : GCCBuiltin<"__builtin_ia32_palignr">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                        llvm_x86mmx_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_mmx_pextr_w : GCCBuiltin<"__builtin_ia32_vec_ext_v4hi">,
              Intrinsic<[llvm_i32_ty], [llvm_x86mmx_ty, llvm_i32_ty],
                        [IntrNoMem]>;

  def int_x86_mmx_pinsr_w : GCCBuiltin<"__builtin_ia32_vec_set_v4hi">,
              Intrinsic<[llvm_x86mmx_ty], [llvm_x86mmx_ty,
                        llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// BMI

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_bmi_bextr_32 : GCCBuiltin<"__builtin_ia32_bextr_u32">,
              Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_bmi_bextr_64 : GCCBuiltin<"__builtin_ia32_bextr_u64">,
              Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;
  def int_x86_bmi_bzhi_32 : GCCBuiltin<"__builtin_ia32_bzhi_si">,
              Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_bmi_bzhi_64 : GCCBuiltin<"__builtin_ia32_bzhi_di">,
              Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;
  def int_x86_bmi_pdep_32 : GCCBuiltin<"__builtin_ia32_pdep_si">,
              Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_bmi_pdep_64 : GCCBuiltin<"__builtin_ia32_pdep_di">,
              Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;
  def int_x86_bmi_pext_32 : GCCBuiltin<"__builtin_ia32_pext_si">,
              Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_bmi_pext_64 : GCCBuiltin<"__builtin_ia32_pext_di">,
              Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// FS/GS Base

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_rdfsbase_32 : GCCBuiltin<"__builtin_ia32_rdfsbase32">,
              Intrinsic<[llvm_i32_ty], []>;
  def int_x86_rdgsbase_32 : GCCBuiltin<"__builtin_ia32_rdgsbase32">,
              Intrinsic<[llvm_i32_ty], []>;
  def int_x86_rdfsbase_64 : GCCBuiltin<"__builtin_ia32_rdfsbase64">,
              Intrinsic<[llvm_i64_ty], []>;
  def int_x86_rdgsbase_64 : GCCBuiltin<"__builtin_ia32_rdgsbase64">,
              Intrinsic<[llvm_i64_ty], []>;
  def int_x86_wrfsbase_32 : GCCBuiltin<"__builtin_ia32_wrfsbase32">,
              Intrinsic<[], [llvm_i32_ty]>;
  def int_x86_wrgsbase_32 : GCCBuiltin<"__builtin_ia32_wrgsbase32">,
              Intrinsic<[], [llvm_i32_ty]>;
  def int_x86_wrfsbase_64 : GCCBuiltin<"__builtin_ia32_wrfsbase64">,
              Intrinsic<[], [llvm_i64_ty]>;
  def int_x86_wrgsbase_64 : GCCBuiltin<"__builtin_ia32_wrgsbase64">,
              Intrinsic<[], [llvm_i64_ty]>;
}

//===----------------------------------------------------------------------===//
// FXSR
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_fxrstor : GCCBuiltin<"__builtin_ia32_fxrstor">,
              Intrinsic<[], [llvm_ptr_ty], []>;
  def int_x86_fxrstor64 : GCCBuiltin<"__builtin_ia32_fxrstor64">,
              Intrinsic<[], [llvm_ptr_ty], []>;
  def int_x86_fxsave : GCCBuiltin<"__builtin_ia32_fxsave">,
              Intrinsic<[], [llvm_ptr_ty], []>;
  def int_x86_fxsave64 : GCCBuiltin<"__builtin_ia32_fxsave64">,
              Intrinsic<[], [llvm_ptr_ty], []>;
}

//===----------------------------------------------------------------------===//
// XSAVE
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_xsave :
              Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_x86_xsave64 :
              Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_x86_xrstor :
              Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_x86_xrstor64 :
              Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_x86_xsaveopt :
              Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_x86_xsaveopt64 :
              Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_x86_xrstors :
              Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_x86_xrstors64 :
              Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_x86_xsavec :
              Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_x86_xsavec64 :
              Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_x86_xsaves :
              Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_x86_xsaves64 :
              Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], []>;
  def int_x86_xgetbv :
              Intrinsic<[llvm_i64_ty], [llvm_i32_ty], []>;
  def int_x86_xsetbv :
              Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;
}

//===----------------------------------------------------------------------===//
// CLFLUSHOPT
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_clflushopt : GCCBuiltin<"__builtin_ia32_clflushopt">,
              Intrinsic<[], [llvm_ptr_ty], []>;
}

//===----------------------------------------------------------------------===//
// Support protection key
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_rdpkru : GCCBuiltin <"__builtin_ia32_rdpkru">,
              Intrinsic<[llvm_i32_ty], [], []>;
  def int_x86_wrpkru : GCCBuiltin<"__builtin_ia32_wrpkru">,
              Intrinsic<[], [llvm_i32_ty], []>;
}
//===----------------------------------------------------------------------===//
// Half float conversion

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_vcvtph2ps_128 : GCCBuiltin<"__builtin_ia32_vcvtph2ps">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_vcvtph2ps_256 : GCCBuiltin<"__builtin_ia32_vcvtph2ps256">,
              Intrinsic<[llvm_v8f32_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_vcvtps2ph_128 : GCCBuiltin<"__builtin_ia32_vcvtps2ph">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v4f32_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_x86_vcvtps2ph_256 : GCCBuiltin<"__builtin_ia32_vcvtps2ph256">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8f32_ty, llvm_i32_ty],
                        [IntrNoMem]>;
  def int_x86_avx512_mask_vcvtph2ps_512 : GCCBuiltin<"__builtin_ia32_vcvtph2ps512_mask">,
              Intrinsic<[llvm_v16f32_ty], [llvm_v16i16_ty, llvm_v16f32_ty,
                                           llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_vcvtph2ps_256 : GCCBuiltin<"__builtin_ia32_vcvtph2ps256_mask">,
              Intrinsic<[llvm_v8f32_ty], [llvm_v8i16_ty, llvm_v8f32_ty,
                                           llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_vcvtph2ps_128 : GCCBuiltin<"__builtin_ia32_vcvtph2ps_mask">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v8i16_ty, llvm_v4f32_ty,
                                           llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_vcvtps2ph_512 : GCCBuiltin<"__builtin_ia32_vcvtps2ph512_mask">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16f32_ty, llvm_i32_ty,
                                           llvm_v16i16_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_vcvtps2ph_256 : GCCBuiltin<"__builtin_ia32_vcvtps2ph256_mask">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8f32_ty, llvm_i32_ty,
                                           llvm_v8i16_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_vcvtps2ph_128 : GCCBuiltin<"__builtin_ia32_vcvtps2ph_mask">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v4f32_ty, llvm_i32_ty,
                                           llvm_v8i16_ty, llvm_i8_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// TBM

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_tbm_bextri_u32 : GCCBuiltin<"__builtin_ia32_bextri_u32">,
        Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_tbm_bextri_u64 : GCCBuiltin<"__builtin_ia32_bextri_u64">,
        Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// RDRAND intrinsics - Return a random value and whether it is valid.
// RDSEED intrinsics - Return a NIST SP800-90B & C compliant random value and
// whether it is valid.

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  // These are declared side-effecting so they don't get eliminated by CSE or
  // LICM.
  def int_x86_rdrand_16 : Intrinsic<[llvm_i16_ty, llvm_i32_ty], [], []>;
  def int_x86_rdrand_32 : Intrinsic<[llvm_i32_ty, llvm_i32_ty], [], []>;
  def int_x86_rdrand_64 : Intrinsic<[llvm_i64_ty, llvm_i32_ty], [], []>;
  def int_x86_rdseed_16 : Intrinsic<[llvm_i16_ty, llvm_i32_ty], [], []>;
  def int_x86_rdseed_32 : Intrinsic<[llvm_i32_ty, llvm_i32_ty], [], []>;
  def int_x86_rdseed_64 : Intrinsic<[llvm_i64_ty, llvm_i32_ty], [], []>;
}

//===----------------------------------------------------------------------===//
// ADX

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_addcarryx_u32: GCCBuiltin<"__builtin_ia32_addcarryx_u32">,
        Intrinsic<[llvm_i8_ty], [llvm_i8_ty, llvm_i32_ty, llvm_i32_ty,
                                 llvm_ptr_ty], [IntrArgMemOnly]>;
  def int_x86_addcarryx_u64: GCCBuiltin<"__builtin_ia32_addcarryx_u64">,
        Intrinsic<[llvm_i8_ty], [llvm_i8_ty, llvm_i64_ty, llvm_i64_ty,
                                 llvm_ptr_ty], [IntrArgMemOnly]>;
  def int_x86_addcarry_u32: GCCBuiltin<"__builtin_ia32_addcarry_u32">,
        Intrinsic<[llvm_i8_ty], [llvm_i8_ty, llvm_i32_ty, llvm_i32_ty,
                                 llvm_ptr_ty], [IntrArgMemOnly]>;
  def int_x86_addcarry_u64: GCCBuiltin<"__builtin_ia32_addcarry_u64">,
        Intrinsic<[llvm_i8_ty], [llvm_i8_ty, llvm_i64_ty, llvm_i64_ty,
                                 llvm_ptr_ty], [IntrArgMemOnly]>;
  def int_x86_subborrow_u32: GCCBuiltin<"__builtin_ia32_subborrow_u32">,
        Intrinsic<[llvm_i8_ty], [llvm_i8_ty, llvm_i32_ty, llvm_i32_ty,
                                 llvm_ptr_ty], [IntrArgMemOnly]>;
  def int_x86_subborrow_u64: GCCBuiltin<"__builtin_ia32_subborrow_u64">,
        Intrinsic<[llvm_i8_ty], [llvm_i8_ty, llvm_i64_ty, llvm_i64_ty,
                                 llvm_ptr_ty], [IntrArgMemOnly]>;
}

//===----------------------------------------------------------------------===//
// RTM intrinsics. Transactional Memory support.

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_xbegin : GCCBuiltin<"__builtin_ia32_xbegin">,
              Intrinsic<[llvm_i32_ty], [], []>;
  def int_x86_xend : GCCBuiltin<"__builtin_ia32_xend">,
              Intrinsic<[], [], []>;
  def int_x86_xabort : GCCBuiltin<"__builtin_ia32_xabort">,
              Intrinsic<[], [llvm_i8_ty], []>;
  def int_x86_xtest : GCCBuiltin<"__builtin_ia32_xtest">,
              Intrinsic<[llvm_i32_ty], [], []>;
}

//===----------------------------------------------------------------------===//
// AVX512

// Mask ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  // Mask instructions
  // 16-bit mask
  def int_x86_avx512_kand_w : GCCBuiltin<"__builtin_ia32_kandhi">,
              Intrinsic<[llvm_i16_ty], [llvm_i16_ty, llvm_i16_ty],
                         [IntrNoMem]>;
  def int_x86_avx512_kandn_w : GCCBuiltin<"__builtin_ia32_kandnhi">,
              Intrinsic<[llvm_i16_ty], [llvm_i16_ty, llvm_i16_ty],
                         [IntrNoMem]>;
  def int_x86_avx512_knot_w : GCCBuiltin<"__builtin_ia32_knothi">,
              Intrinsic<[llvm_i16_ty], [llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_kor_w : GCCBuiltin<"__builtin_ia32_korhi">,
              Intrinsic<[llvm_i16_ty], [llvm_i16_ty, llvm_i16_ty],
                         [IntrNoMem]>;
  def int_x86_avx512_kxor_w : GCCBuiltin<"__builtin_ia32_kxorhi">,
              Intrinsic<[llvm_i16_ty], [llvm_i16_ty, llvm_i16_ty],
                         [IntrNoMem]>;
  def int_x86_avx512_kxnor_w : GCCBuiltin<"__builtin_ia32_kxnorhi">,
              Intrinsic<[llvm_i16_ty], [llvm_i16_ty, llvm_i16_ty],
                         [IntrNoMem]>;
  def int_x86_avx512_kunpck_bw : GCCBuiltin<"__builtin_ia32_kunpckhi">,
              Intrinsic<[llvm_i16_ty], [llvm_i16_ty, llvm_i16_ty],
                         [IntrNoMem]>;
  def int_x86_avx512_kunpck_wd : GCCBuiltin<"__builtin_ia32_kunpcksi">,
              Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                         [IntrNoMem]>;
  def int_x86_avx512_kunpck_dq : GCCBuiltin<"__builtin_ia32_kunpckdi">,
              Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
                         [IntrNoMem]>;
  def int_x86_avx512_kortestz_w : GCCBuiltin<"__builtin_ia32_kortestzhi">,
              Intrinsic<[llvm_i32_ty], [llvm_i16_ty, llvm_i16_ty],
                        [IntrNoMem]>;
  def int_x86_avx512_kortestc_w : GCCBuiltin<"__builtin_ia32_kortestchi">,
              Intrinsic<[llvm_i32_ty], [llvm_i16_ty, llvm_i16_ty],
                        [IntrNoMem]>;
}

// Conversion ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx512_cvttss2si : GCCBuiltin<"__builtin_ia32_vcvttss2si32">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvttss2si64 : GCCBuiltin<"__builtin_ia32_vcvttss2si64">,
              Intrinsic<[llvm_i64_ty], [llvm_v4f32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvttss2usi : GCCBuiltin<"__builtin_ia32_vcvttss2usi32">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvttss2usi64 : GCCBuiltin<"__builtin_ia32_vcvttss2usi64">,
              Intrinsic<[llvm_i64_ty], [llvm_v4f32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvtusi2ss : GCCBuiltin<"__builtin_ia32_cvtusi2ss32">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvtusi642ss : GCCBuiltin<"__builtin_ia32_cvtusi2ss64">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_i64_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvttsd2si : GCCBuiltin<"__builtin_ia32_vcvttsd2si32">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvttsd2si64 : GCCBuiltin<"__builtin_ia32_vcvttsd2si64">,
              Intrinsic<[llvm_i64_ty], [llvm_v2f64_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvttsd2usi : GCCBuiltin<"__builtin_ia32_vcvttsd2usi32">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvttsd2usi64 : GCCBuiltin<"__builtin_ia32_vcvttsd2usi64">,
              Intrinsic<[llvm_i64_ty], [llvm_v2f64_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvtusi2sd : GCCBuiltin<"__builtin_ia32_cvtusi2sd32">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvtusi642sd : GCCBuiltin<"__builtin_ia32_cvtusi2sd64">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_i64_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_vcvtss2usi32 : GCCBuiltin<"__builtin_ia32_vcvtss2usi32">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_vcvtss2usi64 : GCCBuiltin<"__builtin_ia32_vcvtss2usi64">,
              Intrinsic<[llvm_i64_ty], [llvm_v4f32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_vcvtss2si32 : GCCBuiltin<"__builtin_ia32_vcvtss2si32">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_vcvtss2si64 : GCCBuiltin<"__builtin_ia32_vcvtss2si64">,
              Intrinsic<[llvm_i64_ty], [llvm_v4f32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_vcvtsd2usi32 : GCCBuiltin<"__builtin_ia32_vcvtsd2usi32">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_vcvtsd2usi64 : GCCBuiltin<"__builtin_ia32_vcvtsd2usi64">,
              Intrinsic<[llvm_i64_ty], [llvm_v2f64_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_vcvtsd2si32 : GCCBuiltin<"__builtin_ia32_vcvtsd2si32">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_vcvtsd2si64 : GCCBuiltin<"__builtin_ia32_vcvtsd2si64">,
              Intrinsic<[llvm_i64_ty], [llvm_v2f64_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvtsi2ss32 : GCCBuiltin<"__builtin_ia32_cvtsi2ss32">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvtsi2ss64 : GCCBuiltin<"__builtin_ia32_cvtsi2ss64">,
              Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty,
                         llvm_i64_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvtsi2sd64 : GCCBuiltin<"__builtin_ia32_cvtsi2sd64">,
              Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty,
                         llvm_i64_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_cvtb2mask_128 : GCCBuiltin<"__builtin_ia32_cvtb2mask128">,
              Intrinsic<[llvm_i16_ty], [llvm_v16i8_ty], [IntrNoMem]>;
  def int_x86_avx512_cvtb2mask_256 : GCCBuiltin<"__builtin_ia32_cvtb2mask256">,
              Intrinsic<[llvm_i32_ty], [llvm_v32i8_ty], [IntrNoMem]>;
  def int_x86_avx512_cvtb2mask_512 : GCCBuiltin<"__builtin_ia32_cvtb2mask512">,
              Intrinsic<[llvm_i64_ty], [llvm_v64i8_ty], [IntrNoMem]>;

  def int_x86_avx512_cvtw2mask_128 : GCCBuiltin<"__builtin_ia32_cvtw2mask128">,
              Intrinsic<[llvm_i8_ty], [llvm_v8i16_ty], [IntrNoMem]>;
  def int_x86_avx512_cvtw2mask_256 : GCCBuiltin<"__builtin_ia32_cvtw2mask256">,
              Intrinsic<[llvm_i16_ty], [llvm_v16i16_ty], [IntrNoMem]>;
  def int_x86_avx512_cvtw2mask_512 : GCCBuiltin<"__builtin_ia32_cvtw2mask512">,
              Intrinsic<[llvm_i32_ty], [llvm_v32i16_ty], [IntrNoMem]>;

  def int_x86_avx512_cvtd2mask_128 : GCCBuiltin<"__builtin_ia32_cvtd2mask128">,
              Intrinsic<[llvm_i8_ty], [llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvtd2mask_256 : GCCBuiltin<"__builtin_ia32_cvtd2mask256">,
              Intrinsic<[llvm_i8_ty], [llvm_v8i32_ty], [IntrNoMem]>;
  def int_x86_avx512_cvtd2mask_512 : GCCBuiltin<"__builtin_ia32_cvtd2mask512">,
              Intrinsic<[llvm_i16_ty], [llvm_v16i32_ty], [IntrNoMem]>;

  def int_x86_avx512_cvtq2mask_128 : GCCBuiltin<"__builtin_ia32_cvtq2mask128">,
              Intrinsic<[llvm_i8_ty], [llvm_v2i64_ty], [IntrNoMem]>;
  def int_x86_avx512_cvtq2mask_256 : GCCBuiltin<"__builtin_ia32_cvtq2mask256">,
              Intrinsic<[llvm_i8_ty], [llvm_v4i64_ty], [IntrNoMem]>;
  def int_x86_avx512_cvtq2mask_512 : GCCBuiltin<"__builtin_ia32_cvtq2mask512">,
              Intrinsic<[llvm_i8_ty], [llvm_v8i64_ty], [IntrNoMem]>;

}

// Pack ops.
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx512_packsswb_512 : GCCBuiltin<"__builtin_ia32_packsswb512">,
              Intrinsic<[llvm_v64i8_ty], [llvm_v32i16_ty,llvm_v32i16_ty],
                        [IntrNoMem]>;
  def int_x86_avx512_packssdw_512 : GCCBuiltin<"__builtin_ia32_packssdw512">,
              Intrinsic<[llvm_v32i16_ty], [llvm_v16i32_ty, llvm_v16i32_ty],
                         [IntrNoMem]>;
  def int_x86_avx512_packuswb_512 : GCCBuiltin<"__builtin_ia32_packuswb512">,
              Intrinsic<[llvm_v64i8_ty], [llvm_v32i16_ty,llvm_v32i16_ty],
                         [IntrNoMem]>;
  def int_x86_avx512_packusdw_512 : GCCBuiltin<"__builtin_ia32_packusdw512">,
              Intrinsic<[llvm_v32i16_ty], [llvm_v16i32_ty, llvm_v16i32_ty],
                         [IntrNoMem]>;
}

// Vector convert
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx512_mask_cvtdq2ps_128 :
        GCCBuiltin<"__builtin_ia32_cvtdq2ps128_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4i32_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtdq2ps_256 :
        GCCBuiltin<"__builtin_ia32_cvtdq2ps256_mask">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8i32_ty, llvm_v8f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtdq2ps_512 :
        GCCBuiltin<"__builtin_ia32_cvtdq2ps512_mask">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16i32_ty, llvm_v16f32_ty,  llvm_i16_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtpd2dq_128 :
        GCCBuiltin<"__builtin_ia32_cvtpd2dq128_mask">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v2f64_ty, llvm_v4i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtpd2dq_256 :
        GCCBuiltin<"__builtin_ia32_cvtpd2dq256_mask">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v4f64_ty, llvm_v4i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtpd2dq_512 :
        GCCBuiltin<"__builtin_ia32_cvtpd2dq512_mask">,
          Intrinsic<[llvm_v8i32_ty],
          [llvm_v8f64_ty, llvm_v8i32_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtpd2ps_256 :
        GCCBuiltin<"__builtin_ia32_cvtpd2ps256_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f64_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtpd2ps_512 :
        GCCBuiltin<"__builtin_ia32_cvtpd2ps512_mask">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f64_ty, llvm_v8f32_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtsd2ss_round :
        GCCBuiltin<"__builtin_ia32_cvtsd2ss_round_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v2f64_ty, llvm_v4f32_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtss2sd_round :
        GCCBuiltin<"__builtin_ia32_cvtss2sd_round_mask">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v4f32_ty, llvm_v2f64_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtpd2ps :
        GCCBuiltin<"__builtin_ia32_cvtpd2ps_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v2f64_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtpd2qq_128 :
        GCCBuiltin<"__builtin_ia32_cvtpd2qq128_mask">,
          Intrinsic<[llvm_v2i64_ty],
          [llvm_v2f64_ty, llvm_v2i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtpd2qq_256 :
        GCCBuiltin<"__builtin_ia32_cvtpd2qq256_mask">,
          Intrinsic<[llvm_v4i64_ty],
          [llvm_v4f64_ty, llvm_v4i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtpd2qq_512 :
        GCCBuiltin<"__builtin_ia32_cvtpd2qq512_mask">,
          Intrinsic<[llvm_v8i64_ty],
          [llvm_v8f64_ty, llvm_v8i64_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtpd2udq_128 :
        GCCBuiltin<"__builtin_ia32_cvtpd2udq128_mask">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v2f64_ty, llvm_v4i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtpd2udq_256 :
        GCCBuiltin<"__builtin_ia32_cvtpd2udq256_mask">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v4f64_ty, llvm_v4i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtpd2udq_512 :
        GCCBuiltin<"__builtin_ia32_cvtpd2udq512_mask">,
          Intrinsic<[llvm_v8i32_ty],
          [llvm_v8f64_ty, llvm_v8i32_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtpd2uqq_128 :
        GCCBuiltin<"__builtin_ia32_cvtpd2uqq128_mask">,
          Intrinsic<[llvm_v2i64_ty],
          [llvm_v2f64_ty, llvm_v2i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtpd2uqq_256 :
        GCCBuiltin<"__builtin_ia32_cvtpd2uqq256_mask">,
          Intrinsic<[llvm_v4i64_ty],
          [llvm_v4f64_ty, llvm_v4i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtpd2uqq_512 :
        GCCBuiltin<"__builtin_ia32_cvtpd2uqq512_mask">,
          Intrinsic<[llvm_v8i64_ty],
          [llvm_v8f64_ty, llvm_v8i64_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtps2dq_128 :
        GCCBuiltin<"__builtin_ia32_cvtps2dq128_mask">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v4f32_ty, llvm_v4i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtps2dq_256 :
        GCCBuiltin<"__builtin_ia32_cvtps2dq256_mask">,
          Intrinsic<[llvm_v8i32_ty],
          [llvm_v8f32_ty, llvm_v8i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtps2dq_512 :
        GCCBuiltin<"__builtin_ia32_cvtps2dq512_mask">,
          Intrinsic<[llvm_v16i32_ty],
          [llvm_v16f32_ty, llvm_v16i32_ty,  llvm_i16_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtps2pd_128 :
        GCCBuiltin<"__builtin_ia32_cvtps2pd128_mask">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v4f32_ty, llvm_v2f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtps2pd_256 :
        GCCBuiltin<"__builtin_ia32_cvtps2pd256_mask">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f32_ty, llvm_v4f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtps2pd_512 :
        GCCBuiltin<"__builtin_ia32_cvtps2pd512_mask">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f32_ty, llvm_v8f64_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtps2qq_128 :
        GCCBuiltin<"__builtin_ia32_cvtps2qq128_mask">,
          Intrinsic<[llvm_v2i64_ty],
          [llvm_v4f32_ty, llvm_v2i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtps2qq_256 :
        GCCBuiltin<"__builtin_ia32_cvtps2qq256_mask">,
          Intrinsic<[llvm_v4i64_ty],
          [llvm_v4f32_ty, llvm_v4i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtps2qq_512 :
        GCCBuiltin<"__builtin_ia32_cvtps2qq512_mask">,
          Intrinsic<[llvm_v8i64_ty],
          [llvm_v8f32_ty, llvm_v8i64_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtps2udq_128 :
        GCCBuiltin<"__builtin_ia32_cvtps2udq128_mask">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v4f32_ty, llvm_v4i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtps2udq_256 :
        GCCBuiltin<"__builtin_ia32_cvtps2udq256_mask">,
          Intrinsic<[llvm_v8i32_ty],
          [llvm_v8f32_ty, llvm_v8i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtps2udq_512 :
        GCCBuiltin<"__builtin_ia32_cvtps2udq512_mask">,
          Intrinsic<[llvm_v16i32_ty],
          [llvm_v16f32_ty, llvm_v16i32_ty,  llvm_i16_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtps2uqq_128 :
        GCCBuiltin<"__builtin_ia32_cvtps2uqq128_mask">,
          Intrinsic<[llvm_v2i64_ty],
          [llvm_v4f32_ty, llvm_v2i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtps2uqq_256 :
        GCCBuiltin<"__builtin_ia32_cvtps2uqq256_mask">,
          Intrinsic<[llvm_v4i64_ty],
          [llvm_v4f32_ty, llvm_v4i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtps2uqq_512 :
        GCCBuiltin<"__builtin_ia32_cvtps2uqq512_mask">,
          Intrinsic<[llvm_v8i64_ty],
          [llvm_v8f32_ty, llvm_v8i64_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtqq2pd_128 :
        GCCBuiltin<"__builtin_ia32_cvtqq2pd128_mask">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2i64_ty, llvm_v2f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtqq2pd_256 :
        GCCBuiltin<"__builtin_ia32_cvtqq2pd256_mask">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4i64_ty, llvm_v4f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtqq2pd_512 :
        GCCBuiltin<"__builtin_ia32_cvtqq2pd512_mask">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8i64_ty, llvm_v8f64_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtqq2ps_128 :
        GCCBuiltin<"__builtin_ia32_cvtqq2ps128_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v2i64_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtqq2ps_256 :
        GCCBuiltin<"__builtin_ia32_cvtqq2ps256_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4i64_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtqq2ps_512 :
        GCCBuiltin<"__builtin_ia32_cvtqq2ps512_mask">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8i64_ty, llvm_v8f32_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttpd2dq_128 :
        GCCBuiltin<"__builtin_ia32_cvttpd2dq128_mask">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v2f64_ty, llvm_v4i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttpd2dq_256 :
        GCCBuiltin<"__builtin_ia32_cvttpd2dq256_mask">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v4f64_ty, llvm_v4i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttpd2dq_512 :
        GCCBuiltin<"__builtin_ia32_cvttpd2dq512_mask">,
          Intrinsic<[llvm_v8i32_ty],
          [llvm_v8f64_ty, llvm_v8i32_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttpd2qq_128 :
        GCCBuiltin<"__builtin_ia32_cvttpd2qq128_mask">,
          Intrinsic<[llvm_v2i64_ty],
          [llvm_v2f64_ty, llvm_v2i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttpd2qq_256 :
        GCCBuiltin<"__builtin_ia32_cvttpd2qq256_mask">,
          Intrinsic<[llvm_v4i64_ty],
          [llvm_v4f64_ty, llvm_v4i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttpd2qq_512 :
        GCCBuiltin<"__builtin_ia32_cvttpd2qq512_mask">,
          Intrinsic<[llvm_v8i64_ty],
          [llvm_v8f64_ty, llvm_v8i64_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttpd2udq_128 :
        GCCBuiltin<"__builtin_ia32_cvttpd2udq128_mask">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v2f64_ty, llvm_v4i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttpd2udq_256 :
        GCCBuiltin<"__builtin_ia32_cvttpd2udq256_mask">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v4f64_ty, llvm_v4i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttpd2udq_512 :
        GCCBuiltin<"__builtin_ia32_cvttpd2udq512_mask">,
          Intrinsic<[llvm_v8i32_ty],
          [llvm_v8f64_ty, llvm_v8i32_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttpd2uqq_128 :
        GCCBuiltin<"__builtin_ia32_cvttpd2uqq128_mask">,
          Intrinsic<[llvm_v2i64_ty],
          [llvm_v2f64_ty, llvm_v2i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttpd2uqq_256 :
        GCCBuiltin<"__builtin_ia32_cvttpd2uqq256_mask">,
          Intrinsic<[llvm_v4i64_ty],
          [llvm_v4f64_ty, llvm_v4i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttpd2uqq_512 :
        GCCBuiltin<"__builtin_ia32_cvttpd2uqq512_mask">,
          Intrinsic<[llvm_v8i64_ty],
          [llvm_v8f64_ty, llvm_v8i64_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttps2dq_128 :
        GCCBuiltin<"__builtin_ia32_cvttps2dq128_mask">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v4f32_ty, llvm_v4i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttps2dq_256 :
        GCCBuiltin<"__builtin_ia32_cvttps2dq256_mask">,
          Intrinsic<[llvm_v8i32_ty],
          [llvm_v8f32_ty, llvm_v8i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttps2dq_512 :
        GCCBuiltin<"__builtin_ia32_cvttps2dq512_mask">,
          Intrinsic<[llvm_v16i32_ty],
          [llvm_v16f32_ty, llvm_v16i32_ty,  llvm_i16_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttps2qq_128 :
        GCCBuiltin<"__builtin_ia32_cvttps2qq128_mask">,
          Intrinsic<[llvm_v2i64_ty],
          [llvm_v4f32_ty, llvm_v2i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttps2qq_256 :
        GCCBuiltin<"__builtin_ia32_cvttps2qq256_mask">,
          Intrinsic<[llvm_v4i64_ty],
          [llvm_v4f32_ty, llvm_v4i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttps2qq_512 :
        GCCBuiltin<"__builtin_ia32_cvttps2qq512_mask">,
          Intrinsic<[llvm_v8i64_ty],
          [llvm_v8f32_ty, llvm_v8i64_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttps2udq_128 :
        GCCBuiltin<"__builtin_ia32_cvttps2udq128_mask">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v4f32_ty, llvm_v4i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttps2udq_256 :
        GCCBuiltin<"__builtin_ia32_cvttps2udq256_mask">,
          Intrinsic<[llvm_v8i32_ty],
          [llvm_v8f32_ty, llvm_v8i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttps2udq_512 :
        GCCBuiltin<"__builtin_ia32_cvttps2udq512_mask">,
          Intrinsic<[llvm_v16i32_ty],
          [llvm_v16f32_ty, llvm_v16i32_ty,  llvm_i16_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttps2uqq_128 :
        GCCBuiltin<"__builtin_ia32_cvttps2uqq128_mask">,
          Intrinsic<[llvm_v2i64_ty],
          [llvm_v4f32_ty, llvm_v2i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttps2uqq_256 :
        GCCBuiltin<"__builtin_ia32_cvttps2uqq256_mask">,
          Intrinsic<[llvm_v4i64_ty],
          [llvm_v4f32_ty, llvm_v4i64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvttps2uqq_512 :
        GCCBuiltin<"__builtin_ia32_cvttps2uqq512_mask">,
          Intrinsic<[llvm_v8i64_ty],
          [llvm_v8f32_ty, llvm_v8i64_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtudq2ps_128 :
        GCCBuiltin<"__builtin_ia32_cvtudq2ps128_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4i32_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtudq2ps_256 :
        GCCBuiltin<"__builtin_ia32_cvtudq2ps256_mask">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8i32_ty, llvm_v8f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtudq2ps_512 :
        GCCBuiltin<"__builtin_ia32_cvtudq2ps512_mask">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16i32_ty, llvm_v16f32_ty,  llvm_i16_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtuqq2pd_128 :
        GCCBuiltin<"__builtin_ia32_cvtuqq2pd128_mask">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2i64_ty, llvm_v2f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtuqq2pd_256 :
        GCCBuiltin<"__builtin_ia32_cvtuqq2pd256_mask">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4i64_ty, llvm_v4f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtuqq2pd_512 :
        GCCBuiltin<"__builtin_ia32_cvtuqq2pd512_mask">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8i64_ty, llvm_v8f64_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtuqq2ps_128 :
        GCCBuiltin<"__builtin_ia32_cvtuqq2ps128_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v2i64_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtuqq2ps_256 :
        GCCBuiltin<"__builtin_ia32_cvtuqq2ps256_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4i64_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_cvtuqq2ps_512 :
        GCCBuiltin<"__builtin_ia32_cvtuqq2ps512_mask">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8i64_ty, llvm_v8f32_ty,  llvm_i8_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_rndscale_pd_128 : GCCBuiltin<"__builtin_ia32_rndscalepd_128_mask">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_i32_ty,
                                     llvm_v2f64_ty,  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_rndscale_pd_256 : GCCBuiltin<"__builtin_ia32_rndscalepd_256_mask">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty, llvm_i32_ty,
                                     llvm_v4f64_ty,  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_rndscale_pd_512 : GCCBuiltin<"__builtin_ia32_rndscalepd_mask">,
        Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_i32_ty, llvm_v8f64_ty,
                                     llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_rndscale_ps_128 : GCCBuiltin<"__builtin_ia32_rndscaleps_128_mask">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty,
                                     llvm_v4f32_ty,  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_rndscale_ps_256 : GCCBuiltin<"__builtin_ia32_rndscaleps_256_mask">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty, llvm_i32_ty,
                                     llvm_v8f32_ty,  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_rndscale_ps_512 : GCCBuiltin<"__builtin_ia32_rndscaleps_mask">,
        Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_i32_ty, llvm_v16f32_ty,
                                     llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_reduce_pd_128 : GCCBuiltin<"__builtin_ia32_reducepd128_mask">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_i32_ty,
                                     llvm_v2f64_ty,  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_reduce_pd_256 : GCCBuiltin<"__builtin_ia32_reducepd256_mask">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty, llvm_i32_ty,
                                     llvm_v4f64_ty,  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_reduce_pd_512 : GCCBuiltin<"__builtin_ia32_reducepd512_mask">,
        Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_i32_ty, llvm_v8f64_ty,
                                     llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_reduce_ps_128 : GCCBuiltin<"__builtin_ia32_reduceps128_mask">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty,
                                     llvm_v4f32_ty,  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_reduce_ps_256 : GCCBuiltin<"__builtin_ia32_reduceps256_mask">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty, llvm_i32_ty,
                                     llvm_v8f32_ty,  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_reduce_ps_512 : GCCBuiltin<"__builtin_ia32_reduceps512_mask">,
        Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_i32_ty, llvm_v16f32_ty,
                                     llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;
def int_x86_avx512_mask_range_pd_128 : GCCBuiltin<"__builtin_ia32_rangepd128_mask">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty, llvm_i32_ty,
                                    llvm_v2f64_ty,  llvm_i8_ty], [IntrNoMem]>;
def int_x86_avx512_mask_range_pd_256 : GCCBuiltin<"__builtin_ia32_rangepd256_mask">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty, llvm_v4f64_ty, llvm_i32_ty,
                                    llvm_v4f64_ty,  llvm_i8_ty], [IntrNoMem]>;
def int_x86_avx512_mask_range_pd_512 : GCCBuiltin<"__builtin_ia32_rangepd512_mask">,
        Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty, llvm_i32_ty,
                                    llvm_v8f64_ty,  llvm_i8_ty,  llvm_i32_ty], [IntrNoMem]>;
def int_x86_avx512_mask_range_ps_128 : GCCBuiltin<"__builtin_ia32_rangeps128_mask">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_i32_ty,
                                    llvm_v4f32_ty,  llvm_i8_ty], [IntrNoMem]>;
def int_x86_avx512_mask_range_ps_256 : GCCBuiltin<"__builtin_ia32_rangeps256_mask">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty, llvm_v8f32_ty, llvm_i32_ty,
                                    llvm_v8f32_ty,  llvm_i8_ty], [IntrNoMem]>;
def int_x86_avx512_mask_range_ps_512 : GCCBuiltin<"__builtin_ia32_rangeps512_mask">,
        Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty, llvm_i32_ty,
                                     llvm_v16f32_ty,  llvm_i16_ty,  llvm_i32_ty], [IntrNoMem]>;
}

// Vector load with broadcast
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_avx512_vbroadcast_ss_512 :
        GCCBuiltin<"__builtin_ia32_vbroadcastss512">,
        Intrinsic<[llvm_v16f32_ty], [llvm_ptr_ty], [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_vbroadcast_sd_512 :
        GCCBuiltin<"__builtin_ia32_vbroadcastsd512">,
        Intrinsic<[llvm_v8f64_ty], [llvm_ptr_ty], [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_mask_broadcastf32x2_256 :
          GCCBuiltin<"__builtin_ia32_broadcastf32x2_256_mask">,
          Intrinsic<[llvm_v8f32_ty],
                    [llvm_v4f32_ty, llvm_v8f32_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcastf32x2_512 :
          GCCBuiltin<"__builtin_ia32_broadcastf32x2_512_mask">,
          Intrinsic<[llvm_v16f32_ty],
                    [llvm_v4f32_ty, llvm_v16f32_ty, llvm_i16_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcasti32x2_128 :
          GCCBuiltin<"__builtin_ia32_broadcasti32x2_128_mask">,
          Intrinsic<[llvm_v4i32_ty],
                    [llvm_v4i32_ty, llvm_v4i32_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcasti32x2_256 :
          GCCBuiltin<"__builtin_ia32_broadcasti32x2_256_mask">,
          Intrinsic<[llvm_v8i32_ty],
                    [llvm_v4i32_ty, llvm_v8i32_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcasti32x2_512 :
          GCCBuiltin<"__builtin_ia32_broadcasti32x2_512_mask">,
          Intrinsic<[llvm_v16i32_ty],
                    [llvm_v4i32_ty, llvm_v16i32_ty, llvm_i16_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcastf32x4_256 :
            GCCBuiltin<"__builtin_ia32_broadcastf32x4_256_mask">,
            Intrinsic<[llvm_v8f32_ty],
                    [llvm_v4f32_ty, llvm_v8f32_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcastf32x4_512 :
          GCCBuiltin<"__builtin_ia32_broadcastf32x4_512">,
          Intrinsic<[llvm_v16f32_ty],
                    [llvm_v4f32_ty, llvm_v16f32_ty, llvm_i16_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcastf32x8_512 :
          GCCBuiltin<"__builtin_ia32_broadcastf32x8_512_mask">,
          Intrinsic<[llvm_v16f32_ty],
                    [llvm_v8f32_ty, llvm_v16f32_ty, llvm_i16_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcastf64x2_256 :
          GCCBuiltin<"__builtin_ia32_broadcastf64x2_256_mask">,
          Intrinsic<[llvm_v4f64_ty],
                    [llvm_v2f64_ty, llvm_v4f64_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcastf64x2_512 :
          GCCBuiltin<"__builtin_ia32_broadcastf64x2_512_mask">,
          Intrinsic<[llvm_v8f64_ty],
                    [llvm_v2f64_ty, llvm_v8f64_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcastf64x4_512 :
          GCCBuiltin<"__builtin_ia32_broadcastf64x4_512">,
          Intrinsic<[llvm_v8f64_ty],
                    [llvm_v4f64_ty, llvm_v8f64_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcasti32x4_256 :
          GCCBuiltin<"__builtin_ia32_broadcasti32x4_256_mask">,
          Intrinsic<[llvm_v8i32_ty],
                    [llvm_v4i32_ty, llvm_v8i32_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcasti32x4_512 :
          GCCBuiltin<"__builtin_ia32_broadcasti32x4_512">,
          Intrinsic<[llvm_v16i32_ty],
                    [llvm_v4i32_ty, llvm_v16i32_ty, llvm_i16_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcasti32x8_512 :
          GCCBuiltin<"__builtin_ia32_broadcasti32x8_512_mask">,
          Intrinsic<[llvm_v16i32_ty],
                    [llvm_v8i32_ty, llvm_v16i32_ty, llvm_i16_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcasti64x2_256 :
          GCCBuiltin<"__builtin_ia32_broadcasti64x2_256_mask">,
          Intrinsic<[llvm_v4i64_ty],
                    [llvm_v2i64_ty, llvm_v4i64_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcasti64x2_512 :
          GCCBuiltin<"__builtin_ia32_broadcasti64x2_512_mask">,
          Intrinsic<[llvm_v8i64_ty],
                    [llvm_v2i64_ty, llvm_v8i64_ty, llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_broadcasti64x4_512 :
          GCCBuiltin<"__builtin_ia32_broadcasti64x4_512">,
          Intrinsic<[llvm_v8i64_ty],
                    [llvm_v4i64_ty, llvm_v8i64_ty, llvm_i8_ty], [IntrNoMem]>;

   def int_x86_avx512_broadcastmw_512 :
          GCCBuiltin<"__builtin_ia32_broadcastmw512">,
          Intrinsic<[llvm_v16i32_ty], [llvm_i16_ty], [IntrNoMem]>;
   def int_x86_avx512_broadcastmw_256 :
          GCCBuiltin<"__builtin_ia32_broadcastmw256">,
          Intrinsic<[llvm_v8i32_ty], [llvm_i16_ty], [IntrNoMem]>;
   def int_x86_avx512_broadcastmw_128 :
          GCCBuiltin<"__builtin_ia32_broadcastmw128">,
          Intrinsic<[llvm_v4i32_ty], [llvm_i16_ty], [IntrNoMem]>;
   def int_x86_avx512_broadcastmb_512 :
          GCCBuiltin<"__builtin_ia32_broadcastmb512">,
          Intrinsic<[llvm_v8i64_ty], [llvm_i8_ty], [IntrNoMem]>;
   def int_x86_avx512_broadcastmb_256 :
          GCCBuiltin<"__builtin_ia32_broadcastmb256">,
          Intrinsic<[llvm_v4i64_ty], [llvm_i8_ty], [IntrNoMem]>;
   def int_x86_avx512_broadcastmb_128 :
          GCCBuiltin<"__builtin_ia32_broadcastmb128">,
          Intrinsic<[llvm_v2i64_ty], [llvm_i8_ty], [IntrNoMem]>;
}

// Arithmetic ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".

  def int_x86_avx512_mask_add_ps_512 : GCCBuiltin<"__builtin_ia32_addps512_mask">,
          Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                     llvm_v16f32_ty, llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_add_pd_512 : GCCBuiltin<"__builtin_ia32_addpd512_mask">,
          Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                     llvm_v8f64_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_sub_ps_512 : GCCBuiltin<"__builtin_ia32_subps512_mask">,
          Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                     llvm_v16f32_ty, llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_sub_pd_512 : GCCBuiltin<"__builtin_ia32_subpd512_mask">,
          Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                     llvm_v8f64_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_mul_ps_512 : GCCBuiltin<"__builtin_ia32_mulps512_mask">,
          Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                     llvm_v16f32_ty, llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_mul_pd_512 : GCCBuiltin<"__builtin_ia32_mulpd512_mask">,
          Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                     llvm_v8f64_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_div_ps_512 : GCCBuiltin<"__builtin_ia32_divps512_mask">,
          Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                     llvm_v16f32_ty, llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_div_pd_512 : GCCBuiltin<"__builtin_ia32_divpd512_mask">,
          Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                     llvm_v8f64_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_max_ps_512 : GCCBuiltin<"__builtin_ia32_maxps512_mask">,
          Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                     llvm_v16f32_ty, llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_max_pd_512 : GCCBuiltin<"__builtin_ia32_maxpd512_mask">,
          Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                     llvm_v8f64_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_min_ps_512 : GCCBuiltin<"__builtin_ia32_minps512_mask">,
          Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                     llvm_v16f32_ty, llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_min_pd_512 : GCCBuiltin<"__builtin_ia32_minpd512_mask">,
          Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                     llvm_v8f64_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_add_ss_round : GCCBuiltin<"__builtin_ia32_addss_round_mask">,
          Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                     llvm_v4f32_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_div_ss_round : GCCBuiltin<"__builtin_ia32_divss_round_mask">,
          Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                     llvm_v4f32_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_mul_ss_round : GCCBuiltin<"__builtin_ia32_mulss_round_mask">,
          Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                     llvm_v4f32_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_sub_ss_round : GCCBuiltin<"__builtin_ia32_subss_round_mask">,
          Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                     llvm_v4f32_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_max_ss_round : GCCBuiltin<"__builtin_ia32_maxss_round_mask">,
          Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                     llvm_v4f32_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_min_ss_round : GCCBuiltin<"__builtin_ia32_minss_round_mask">,
          Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                     llvm_v4f32_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_add_sd_round : GCCBuiltin<"__builtin_ia32_addsd_round_mask">,
          Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                     llvm_v2f64_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_div_sd_round : GCCBuiltin<"__builtin_ia32_divsd_round_mask">,
          Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                     llvm_v2f64_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_mul_sd_round : GCCBuiltin<"__builtin_ia32_mulsd_round_mask">,
          Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                     llvm_v2f64_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_sub_sd_round : GCCBuiltin<"__builtin_ia32_subsd_round_mask">,
          Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                     llvm_v2f64_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_max_sd_round : GCCBuiltin<"__builtin_ia32_maxsd_round_mask">,
          Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                     llvm_v2f64_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_min_sd_round : GCCBuiltin<"__builtin_ia32_minsd_round_mask">,
          Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                     llvm_v2f64_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_rndscale_ss : GCCBuiltin<"__builtin_ia32_rndscaless_round_mask">,
          Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,
                                     llvm_i8_ty, llvm_i32_ty, llvm_i32_ty],
                                     [IntrNoMem]>;
  def int_x86_avx512_mask_rndscale_sd : GCCBuiltin<"__builtin_ia32_rndscalesd_round_mask">,
          Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,
                                      llvm_i8_ty, llvm_i32_ty, llvm_i32_ty],
                                     [IntrNoMem]>;
  def int_x86_avx512_mask_range_ss : GCCBuiltin<"__builtin_ia32_rangess128_round_mask">,
          Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,
                                     llvm_i8_ty, llvm_i32_ty, llvm_i32_ty],
                                     [IntrNoMem]>;
  def int_x86_avx512_mask_range_sd : GCCBuiltin<"__builtin_ia32_rangesd128_round_mask">,
          Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,
                                      llvm_i8_ty, llvm_i32_ty, llvm_i32_ty],
                                     [IntrNoMem]>;
  def int_x86_avx512_mask_reduce_ss : GCCBuiltin<"__builtin_ia32_reducess_mask">,
          Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,
                                     llvm_i8_ty, llvm_i32_ty, llvm_i32_ty],
                                     [IntrNoMem]>;
  def int_x86_avx512_mask_reduce_sd : GCCBuiltin<"__builtin_ia32_reducesd_mask">,
          Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,
                                      llvm_i8_ty, llvm_i32_ty, llvm_i32_ty],
                                     [IntrNoMem]>;
  def int_x86_avx512_mask_scalef_sd : GCCBuiltin<"__builtin_ia32_scalefsd_round_mask">,
          Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                                      llvm_v2f64_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_scalef_ss : GCCBuiltin<"__builtin_ia32_scalefss_round_mask">,
          Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                                      llvm_v4f32_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_scalef_pd_128 : GCCBuiltin<"__builtin_ia32_scalefpd128_mask">,
          Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                    llvm_v2f64_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_scalef_pd_256 : GCCBuiltin<"__builtin_ia32_scalefpd256_mask">,
          Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty, llvm_v4f64_ty,
                    llvm_v4f64_ty, llvm_i8_ty],[IntrNoMem]>;
  def int_x86_avx512_mask_scalef_pd_512 : GCCBuiltin<"__builtin_ia32_scalefpd512_mask">,
          Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                    llvm_v8f64_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_scalef_ps_128 : GCCBuiltin<"__builtin_ia32_scalefps128_mask">,
          Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                    llvm_v4f32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_scalef_ps_256 : GCCBuiltin<"__builtin_ia32_scalefps256_mask">,
          Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty, llvm_v8f32_ty,
                    llvm_v8f32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_scalef_ps_512 : GCCBuiltin<"__builtin_ia32_scalefps512_mask">,
          Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                    llvm_v16f32_ty, llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_sqrt_ss : GCCBuiltin<"__builtin_ia32_sqrtss_round_mask">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,
                                    llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_sqrt_sd : GCCBuiltin<"__builtin_ia32_sqrtsd_round_mask">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,
                                    llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_sqrt_pd_128 : GCCBuiltin<"__builtin_ia32_sqrtpd128_mask">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                                    llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_sqrt_pd_256 : GCCBuiltin<"__builtin_ia32_sqrtpd256_mask">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty, llvm_v4f64_ty,
                                    llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_sqrt_pd_512 : GCCBuiltin<"__builtin_ia32_sqrtpd512_mask">,
        Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                                    llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_sqrt_ps_128 : GCCBuiltin<"__builtin_ia32_sqrtps128_mask">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                                     llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_sqrt_ps_256 : GCCBuiltin<"__builtin_ia32_sqrtps256_mask">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty, llvm_v8f32_ty,
                                     llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_sqrt_ps_512 : GCCBuiltin<"__builtin_ia32_sqrtps512_mask">,
        Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                                     llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_fixupimm_pd_128 :
         GCCBuiltin<"__builtin_ia32_fixupimmpd128_mask">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2i64_ty, llvm_i32_ty, llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_maskz_fixupimm_pd_128 :
         GCCBuiltin<"__builtin_ia32_fixupimmpd128_maskz">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2i64_ty, llvm_i32_ty, llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_fixupimm_pd_256 :
         GCCBuiltin<"__builtin_ia32_fixupimmpd256_mask">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4i64_ty, llvm_i32_ty, llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_maskz_fixupimm_pd_256 :
         GCCBuiltin<"__builtin_ia32_fixupimmpd256_maskz">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_v4f64_ty, llvm_v4i64_ty, llvm_i32_ty, llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_fixupimm_pd_512 :
         GCCBuiltin<"__builtin_ia32_fixupimmpd512_mask">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty, llvm_v8f64_ty, llvm_v8i64_ty, llvm_i32_ty, llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_maskz_fixupimm_pd_512 :
         GCCBuiltin<"__builtin_ia32_fixupimmpd512_maskz">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty, llvm_v8f64_ty, llvm_v8i64_ty, llvm_i32_ty, llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_fixupimm_ps_128 :
         GCCBuiltin<"__builtin_ia32_fixupimmps128_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4i32_ty, llvm_i32_ty, llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_maskz_fixupimm_ps_128 :
         GCCBuiltin<"__builtin_ia32_fixupimmps128_maskz">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4i32_ty, llvm_i32_ty, llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_fixupimm_ps_256 :
         GCCBuiltin<"__builtin_ia32_fixupimmps256_mask">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8i32_ty, llvm_i32_ty, llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_maskz_fixupimm_ps_256 :
         GCCBuiltin<"__builtin_ia32_fixupimmps256_maskz">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_v8f32_ty, llvm_v8i32_ty, llvm_i32_ty, llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_fixupimm_ps_512 :
         GCCBuiltin<"__builtin_ia32_fixupimmps512_mask">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty, llvm_v16f32_ty, llvm_v16i32_ty, llvm_i32_ty,
          llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_maskz_fixupimm_ps_512 :
         GCCBuiltin<"__builtin_ia32_fixupimmps512_maskz">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty, llvm_v16f32_ty, llvm_v16i32_ty, llvm_i32_ty,
          llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_fixupimm_sd :
         GCCBuiltin<"__builtin_ia32_fixupimmsd_mask">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2i64_ty, llvm_i32_ty, llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_maskz_fixupimm_sd :
         GCCBuiltin<"__builtin_ia32_fixupimmsd_maskz">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2i64_ty, llvm_i32_ty, llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_fixupimm_ss :
         GCCBuiltin<"__builtin_ia32_fixupimmss_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4i32_ty, llvm_i32_ty, llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_maskz_fixupimm_ss :
         GCCBuiltin<"__builtin_ia32_fixupimmss_maskz">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4i32_ty, llvm_i32_ty, llvm_i8_ty,
          llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_getexp_pd_128 : GCCBuiltin<"__builtin_ia32_getexppd128_mask">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                                    llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_getexp_pd_256 : GCCBuiltin<"__builtin_ia32_getexppd256_mask">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty, llvm_v4f64_ty,
                                    llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_getexp_pd_512 : GCCBuiltin<"__builtin_ia32_getexppd512_mask">,
        Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                                    llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_getexp_ps_128 : GCCBuiltin<"__builtin_ia32_getexpps128_mask">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                                     llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_getexp_ps_256 : GCCBuiltin<"__builtin_ia32_getexpps256_mask">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty, llvm_v8f32_ty,
                                     llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_getexp_ps_512 : GCCBuiltin<"__builtin_ia32_getexpps512_mask">,
        Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                                     llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_getexp_ss : GCCBuiltin<"__builtin_ia32_getexpss128_round_mask">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,
                                    llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_getexp_sd : GCCBuiltin<"__builtin_ia32_getexpsd128_round_mask">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,
                                    llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_getmant_pd_128 :
         GCCBuiltin<"__builtin_ia32_getmantpd128_mask">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty,llvm_i32_ty, llvm_v2f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_getmant_pd_256 :
         GCCBuiltin<"__builtin_ia32_getmantpd256_mask">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty,llvm_i32_ty, llvm_v4f64_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_getmant_pd_512 :
         GCCBuiltin<"__builtin_ia32_getmantpd512_mask">,
          Intrinsic<[llvm_v8f64_ty],
          [llvm_v8f64_ty,llvm_i32_ty, llvm_v8f64_ty,  llvm_i8_ty,llvm_i32_ty ],
          [IntrNoMem]>;

  def int_x86_avx512_mask_getmant_ps_128 :
         GCCBuiltin<"__builtin_ia32_getmantps128_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_i32_ty, llvm_v4f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_getmant_ps_256 :
         GCCBuiltin<"__builtin_ia32_getmantps256_mask">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_i32_ty, llvm_v8f32_ty,  llvm_i8_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_getmant_ps_512 :
         GCCBuiltin<"__builtin_ia32_getmantps512_mask">,
          Intrinsic<[llvm_v16f32_ty],
          [llvm_v16f32_ty,llvm_i32_ty, llvm_v16f32_ty,llvm_i16_ty,llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_getmant_ss :
         GCCBuiltin<"__builtin_ia32_getmantss_round_mask">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_v4f32_ty, llvm_i32_ty, llvm_v4f32_ty,
           llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_getmant_sd :
         GCCBuiltin<"__builtin_ia32_getmantsd_round_mask">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_v2f64_ty, llvm_i32_ty, llvm_v2f64_ty,
           llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_rsqrt14_ss : GCCBuiltin<"__builtin_ia32_rsqrt14ss_mask">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,
                                    llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_rsqrt14_sd : GCCBuiltin<"__builtin_ia32_rsqrt14sd_mask">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,
                                    llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_rsqrt14_pd_128 : GCCBuiltin<"__builtin_ia32_rsqrt14pd128_mask">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                                    llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_rsqrt14_pd_256 : GCCBuiltin<"__builtin_ia32_rsqrt14pd256_mask">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty, llvm_v4f64_ty,
                                    llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_rsqrt14_pd_512 : GCCBuiltin<"__builtin_ia32_rsqrt14pd512_mask">,
        Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                                    llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_rsqrt14_ps_128 : GCCBuiltin<"__builtin_ia32_rsqrt14ps128_mask">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                                     llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_rsqrt14_ps_256 : GCCBuiltin<"__builtin_ia32_rsqrt14ps256_mask">,
          Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty, llvm_v8f32_ty,
                                     llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_rsqrt14_ps_512 : GCCBuiltin<"__builtin_ia32_rsqrt14ps512_mask">,
        Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                                     llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_rcp14_ss : GCCBuiltin<"__builtin_ia32_rcp14ss_mask">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty,
                                    llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_rcp14_sd : GCCBuiltin<"__builtin_ia32_rcp14sd_mask">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty, llvm_v2f64_ty,
                                    llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_rcp14_pd_128 : GCCBuiltin<"__builtin_ia32_rcp14pd128_mask">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                                    llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_rcp14_pd_256 : GCCBuiltin<"__builtin_ia32_rcp14pd256_mask">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty, llvm_v4f64_ty,
                                    llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_rcp14_pd_512 : GCCBuiltin<"__builtin_ia32_rcp14pd512_mask">,
        Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                                    llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_rcp14_ps_128 : GCCBuiltin<"__builtin_ia32_rcp14ps128_mask">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                                     llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_rcp14_ps_256 : GCCBuiltin<"__builtin_ia32_rcp14ps256_mask">,
          Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty, llvm_v8f32_ty,
                                     llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_rcp14_ps_512 : GCCBuiltin<"__builtin_ia32_rcp14ps512_mask">,
        Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                                     llvm_i16_ty], [IntrNoMem]>;

  def int_x86_avx512_rcp28_ps : GCCBuiltin<"__builtin_ia32_rcp28ps_mask">,
            Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                                         llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_rcp28_pd : GCCBuiltin<"__builtin_ia32_rcp28pd_mask">,
            Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                                        llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_exp2_ps : GCCBuiltin<"__builtin_ia32_exp2ps_mask">,
            Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                                         llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_exp2_pd : GCCBuiltin<"__builtin_ia32_exp2pd_mask">,
            Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                                        llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_rcp28_ss : GCCBuiltin<"__builtin_ia32_rcp28ss_round_mask">,
            Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                                        llvm_v4f32_ty, llvm_i8_ty, llvm_i32_ty],
                      [IntrNoMem]>;
  def int_x86_avx512_rcp28_sd : GCCBuiltin<"__builtin_ia32_rcp28sd_round_mask">,
            Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                                        llvm_v2f64_ty, llvm_i8_ty, llvm_i32_ty],
                      [IntrNoMem]>;
  def int_x86_avx512_rsqrt28_ps : GCCBuiltin<"__builtin_ia32_rsqrt28ps_mask">,
            Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                                         llvm_i16_ty, llvm_i32_ty],
                      [IntrNoMem]>;
  def int_x86_avx512_rsqrt28_pd : GCCBuiltin<"__builtin_ia32_rsqrt28pd_mask">,
            Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                                        llvm_i8_ty, llvm_i32_ty],
                      [IntrNoMem]>;
  def int_x86_avx512_rsqrt28_ss : GCCBuiltin<"__builtin_ia32_rsqrt28ss_round_mask">,
            Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                                        llvm_v4f32_ty, llvm_i8_ty, llvm_i32_ty],
                      [IntrNoMem]>;
  def int_x86_avx512_rsqrt28_sd : GCCBuiltin<"__builtin_ia32_rsqrt28sd_round_mask">,
            Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                                        llvm_v2f64_ty, llvm_i8_ty, llvm_i32_ty],
                      [IntrNoMem]>;
  def int_x86_avx512_psad_bw_512 : GCCBuiltin<"__builtin_ia32_psadbw512">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v64i8_ty, llvm_v64i8_ty],
                        [IntrNoMem, Commutative]>;
}
// Integer arithmetic ops
let TargetPrefix = "x86" in {
  def int_x86_avx512_mask_padds_b_128 : GCCBuiltin<"__builtin_ia32_paddsb128_mask">,
          Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty,
                     llvm_v16i8_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_padds_b_256 : GCCBuiltin<"__builtin_ia32_paddsb256_mask">,
          Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty, llvm_v32i8_ty,
                     llvm_v32i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_padds_b_512 : GCCBuiltin<"__builtin_ia32_paddsb512_mask">,
          Intrinsic<[llvm_v64i8_ty], [llvm_v64i8_ty, llvm_v64i8_ty,
                     llvm_v64i8_ty, llvm_i64_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_padds_w_128 : GCCBuiltin<"__builtin_ia32_paddsw128_mask">,
          Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                     llvm_v8i16_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_padds_w_256 : GCCBuiltin<"__builtin_ia32_paddsw256_mask">,
          Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty, llvm_v16i16_ty,
                     llvm_v16i16_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_padds_w_512 : GCCBuiltin<"__builtin_ia32_paddsw512_mask">,
          Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty, llvm_v32i16_ty,
                     llvm_v32i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_paddus_b_128 : GCCBuiltin<"__builtin_ia32_paddusb128_mask">,
          Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty,
                     llvm_v16i8_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_paddus_b_256 : GCCBuiltin<"__builtin_ia32_paddusb256_mask">,
          Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty, llvm_v32i8_ty,
                     llvm_v32i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_paddus_b_512 : GCCBuiltin<"__builtin_ia32_paddusb512_mask">,
          Intrinsic<[llvm_v64i8_ty], [llvm_v64i8_ty, llvm_v64i8_ty,
                     llvm_v64i8_ty, llvm_i64_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_paddus_w_128 : GCCBuiltin<"__builtin_ia32_paddusw128_mask">,
          Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                     llvm_v8i16_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_paddus_w_256 : GCCBuiltin<"__builtin_ia32_paddusw256_mask">,
          Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty, llvm_v16i16_ty,
                     llvm_v16i16_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_paddus_w_512 : GCCBuiltin<"__builtin_ia32_paddusw512_mask">,
          Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty, llvm_v32i16_ty,
                     llvm_v32i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_psubs_b_128 : GCCBuiltin<"__builtin_ia32_psubsb128_mask">,
          Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty,
                     llvm_v16i8_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_psubs_b_256 : GCCBuiltin<"__builtin_ia32_psubsb256_mask">,
          Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty, llvm_v32i8_ty,
                     llvm_v32i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_psubs_b_512 : GCCBuiltin<"__builtin_ia32_psubsb512_mask">,
          Intrinsic<[llvm_v64i8_ty], [llvm_v64i8_ty, llvm_v64i8_ty,
                     llvm_v64i8_ty, llvm_i64_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_psubs_w_128 : GCCBuiltin<"__builtin_ia32_psubsw128_mask">,
          Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                     llvm_v8i16_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_psubs_w_256 : GCCBuiltin<"__builtin_ia32_psubsw256_mask">,
          Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty, llvm_v16i16_ty,
                     llvm_v16i16_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_psubs_w_512 : GCCBuiltin<"__builtin_ia32_psubsw512_mask">,
          Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty, llvm_v32i16_ty,
                     llvm_v32i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_psubus_b_128 : GCCBuiltin<"__builtin_ia32_psubusb128_mask">,
          Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty,
                     llvm_v16i8_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_psubus_b_256 : GCCBuiltin<"__builtin_ia32_psubusb256_mask">,
          Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty, llvm_v32i8_ty,
                     llvm_v32i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_psubus_b_512 : GCCBuiltin<"__builtin_ia32_psubusb512_mask">,
          Intrinsic<[llvm_v64i8_ty], [llvm_v64i8_ty, llvm_v64i8_ty,
                     llvm_v64i8_ty, llvm_i64_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_psubus_w_128 : GCCBuiltin<"__builtin_ia32_psubusw128_mask">,
          Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                     llvm_v8i16_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_psubus_w_256 : GCCBuiltin<"__builtin_ia32_psubusw256_mask">,
          Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty, llvm_v16i16_ty,
                     llvm_v16i16_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_psubus_w_512 : GCCBuiltin<"__builtin_ia32_psubusw512_mask">,
          Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty, llvm_v32i16_ty,
                     llvm_v32i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_pmulu_dq_512 : GCCBuiltin<"__builtin_ia32_pmuludq512">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v16i32_ty, llvm_v16i32_ty], [IntrNoMem]>;
  def int_x86_avx512_pmul_dq_512 : GCCBuiltin<"__builtin_ia32_pmuldq512">,
              Intrinsic<[llvm_v8i64_ty], [llvm_v16i32_ty, llvm_v16i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pmulhu_w_512 : GCCBuiltin<"__builtin_ia32_pmulhuw512_mask">,
              Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty, llvm_v32i16_ty,
                     llvm_v32i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pmulh_w_512 : GCCBuiltin<"__builtin_ia32_pmulhw512_mask">,
              Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty, llvm_v32i16_ty,
                     llvm_v32i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pmulhu_w_128 : GCCBuiltin<"__builtin_ia32_pmulhuw128_mask">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                     llvm_v8i16_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pmulhu_w_256 : GCCBuiltin<"__builtin_ia32_pmulhuw256_mask">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty, llvm_v16i16_ty,
                     llvm_v16i16_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pmulh_w_128 : GCCBuiltin<"__builtin_ia32_pmulhw128_mask">,
              Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                     llvm_v8i16_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pmulh_w_256 : GCCBuiltin<"__builtin_ia32_pmulhw256_mask">,
              Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty, llvm_v16i16_ty,
                     llvm_v16i16_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pavg_b_512 : GCCBuiltin<"__builtin_ia32_pavgb512_mask">,
          Intrinsic<[llvm_v64i8_ty], [llvm_v64i8_ty, llvm_v64i8_ty,
                    llvm_v64i8_ty, llvm_i64_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pavg_w_512 : GCCBuiltin<"__builtin_ia32_pavgw512_mask">,
          Intrinsic<[llvm_v32i16_ty], [llvm_v32i16_ty, llvm_v32i16_ty,
                    llvm_v32i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pavg_b_128 : GCCBuiltin<"__builtin_ia32_pavgb128_mask">,
          Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty, llvm_v16i8_ty,
                    llvm_v16i8_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pavg_b_256 : GCCBuiltin<"__builtin_ia32_pavgb256_mask">,
          Intrinsic<[llvm_v32i8_ty], [llvm_v32i8_ty, llvm_v32i8_ty,
                    llvm_v32i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pavg_w_128 : GCCBuiltin<"__builtin_ia32_pavgw128_mask">,
          Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty,
                    llvm_v8i16_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pavg_w_256 : GCCBuiltin<"__builtin_ia32_pavgw256_mask">,
          Intrinsic<[llvm_v16i16_ty], [llvm_v16i16_ty, llvm_v16i16_ty,
                    llvm_v16i16_ty, llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_pmaddw_d_128 :
         GCCBuiltin<"__builtin_ia32_pmaddwd128_mask">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v4i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_pmaddw_d_256 :
         GCCBuiltin<"__builtin_ia32_pmaddwd256_mask">,
          Intrinsic<[llvm_v8i32_ty],
          [llvm_v16i16_ty, llvm_v16i16_ty, llvm_v8i32_ty,  llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_pmaddw_d_512 :
         GCCBuiltin<"__builtin_ia32_pmaddwd512_mask">,
          Intrinsic<[llvm_v16i32_ty],
          [llvm_v32i16_ty, llvm_v32i16_ty, llvm_v16i32_ty,  llvm_i16_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_pmaddubs_w_128 :
         GCCBuiltin<"__builtin_ia32_pmaddubsw128_mask">,
          Intrinsic<[llvm_v8i16_ty],
          [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v8i16_ty,  llvm_i8_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_pmaddubs_w_256 :
         GCCBuiltin<"__builtin_ia32_pmaddubsw256_mask">,
          Intrinsic<[llvm_v16i16_ty],
          [llvm_v32i8_ty, llvm_v32i8_ty, llvm_v16i16_ty,  llvm_i16_ty],
          [IntrNoMem]>;
  def int_x86_avx512_mask_pmaddubs_w_512 :
         GCCBuiltin<"__builtin_ia32_pmaddubsw512_mask">,
          Intrinsic<[llvm_v32i16_ty],
          [llvm_v64i8_ty, llvm_v64i8_ty, llvm_v32i16_ty,  llvm_i32_ty],
          [IntrNoMem]>;

  def int_x86_avx512_mask_dbpsadbw_128 :
         GCCBuiltin<"__builtin_ia32_dbpsadbw128_mask">,
          Intrinsic<[llvm_v8i16_ty],
          [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_v8i16_ty,
           llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_dbpsadbw_256 :
         GCCBuiltin<"__builtin_ia32_dbpsadbw256_mask">,
          Intrinsic<[llvm_v16i16_ty],
          [llvm_v32i8_ty, llvm_v32i8_ty, llvm_i32_ty, llvm_v16i16_ty,
           llvm_i16_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_dbpsadbw_512 :
         GCCBuiltin<"__builtin_ia32_dbpsadbw512_mask">,
          Intrinsic<[llvm_v32i16_ty],
          [llvm_v64i8_ty, llvm_v64i8_ty, llvm_i32_ty, llvm_v32i16_ty,
           llvm_i32_ty], [IntrNoMem]>;
}

// Gather and Scatter ops
let TargetPrefix = "x86" in {
  def int_x86_avx512_gather_dpd_512  : GCCBuiltin<"__builtin_ia32_gathersiv8df">,
          Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_ptr_ty,
                     llvm_v8i32_ty, llvm_i8_ty, llvm_i32_ty],
                    [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_gather_dps_512  : GCCBuiltin<"__builtin_ia32_gathersiv16sf">,
          Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_ptr_ty,
                     llvm_v16i32_ty, llvm_i16_ty, llvm_i32_ty],
                    [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_gather_qpd_512  : GCCBuiltin<"__builtin_ia32_gatherdiv8df">,
          Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_ptr_ty,
                     llvm_v8i64_ty, llvm_i8_ty, llvm_i32_ty],
                    [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_gather_qps_512  : GCCBuiltin<"__builtin_ia32_gatherdiv16sf">,
          Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty, llvm_ptr_ty,
                     llvm_v8i64_ty, llvm_i8_ty, llvm_i32_ty],
                    [IntrReadMem, IntrArgMemOnly]>;


  def int_x86_avx512_gather_dpq_512  : GCCBuiltin<"__builtin_ia32_gathersiv8di">,
          Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty, llvm_ptr_ty,
                     llvm_v8i32_ty, llvm_i8_ty, llvm_i32_ty],
                    [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_gather_dpi_512  : GCCBuiltin<"__builtin_ia32_gathersiv16si">,
          Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty, llvm_ptr_ty,
                     llvm_v16i32_ty, llvm_i16_ty, llvm_i32_ty],
                    [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_gather_qpq_512  : GCCBuiltin<"__builtin_ia32_gatherdiv8di">,
          Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty, llvm_ptr_ty,
                     llvm_v8i64_ty, llvm_i8_ty, llvm_i32_ty],
                    [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_gather_qpi_512  : GCCBuiltin<"__builtin_ia32_gatherdiv16si">,
          Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty, llvm_ptr_ty,
                     llvm_v8i64_ty, llvm_i8_ty, llvm_i32_ty],
                    [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3div2_df :
        GCCBuiltin<"__builtin_ia32_gather3div2df">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_ptr_ty, llvm_v2i64_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3div2_di :
        GCCBuiltin<"__builtin_ia32_gather3div2di">,
          Intrinsic<[llvm_v2i64_ty],
          [llvm_v2i64_ty, llvm_ptr_ty, llvm_v2i64_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3div4_df :
        GCCBuiltin<"__builtin_ia32_gather3div4df">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_ptr_ty, llvm_v4i64_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3div4_di :
        GCCBuiltin<"__builtin_ia32_gather3div4di">,
          Intrinsic<[llvm_v4i64_ty],
          [llvm_v4i64_ty, llvm_ptr_ty, llvm_v4i64_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3div4_sf :
        GCCBuiltin<"__builtin_ia32_gather3div4sf">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_ptr_ty, llvm_v2i64_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3div4_si :
        GCCBuiltin<"__builtin_ia32_gather3div4si">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v4i32_ty, llvm_ptr_ty, llvm_v2i64_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3div8_sf :
        GCCBuiltin<"__builtin_ia32_gather3div8sf">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_ptr_ty, llvm_v4i64_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3div8_si :
        GCCBuiltin<"__builtin_ia32_gather3div8si">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v4i32_ty, llvm_ptr_ty, llvm_v4i64_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3siv2_df :
        GCCBuiltin<"__builtin_ia32_gather3siv2df">,
          Intrinsic<[llvm_v2f64_ty],
          [llvm_v2f64_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3siv2_di :
        GCCBuiltin<"__builtin_ia32_gather3siv2di">,
          Intrinsic<[llvm_v2i64_ty],
          [llvm_v2i64_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3siv4_df :
        GCCBuiltin<"__builtin_ia32_gather3siv4df">,
          Intrinsic<[llvm_v4f64_ty],
          [llvm_v4f64_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3siv4_di :
        GCCBuiltin<"__builtin_ia32_gather3siv4di">,
          Intrinsic<[llvm_v4i64_ty],
          [llvm_v4i64_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3siv4_sf :
        GCCBuiltin<"__builtin_ia32_gather3siv4sf">,
          Intrinsic<[llvm_v4f32_ty],
          [llvm_v4f32_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3siv4_si :
        GCCBuiltin<"__builtin_ia32_gather3siv4si">,
          Intrinsic<[llvm_v4i32_ty],
          [llvm_v4i32_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3siv8_sf :
        GCCBuiltin<"__builtin_ia32_gather3siv8sf">,
          Intrinsic<[llvm_v8f32_ty],
          [llvm_v8f32_ty, llvm_ptr_ty, llvm_v8i32_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_gather3siv8_si :
        GCCBuiltin<"__builtin_ia32_gather3siv8si">,
          Intrinsic<[llvm_v8i32_ty],
          [llvm_v8i32_ty, llvm_ptr_ty, llvm_v8i32_ty, llvm_i8_ty, llvm_i32_ty],
          [IntrReadMem, IntrArgMemOnly]>;

// scatter
  def int_x86_avx512_scatter_dpd_512  : GCCBuiltin<"__builtin_ia32_scattersiv8df">,
          Intrinsic<[], [llvm_ptr_ty, llvm_i8_ty,
                        llvm_v8i32_ty, llvm_v8f64_ty, llvm_i32_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_scatter_dps_512  : GCCBuiltin<"__builtin_ia32_scattersiv16sf">,
          Intrinsic<[], [llvm_ptr_ty, llvm_i16_ty,
                       llvm_v16i32_ty, llvm_v16f32_ty, llvm_i32_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_scatter_qpd_512  : GCCBuiltin<"__builtin_ia32_scatterdiv8df">,
          Intrinsic<[], [llvm_ptr_ty, llvm_i8_ty,
                     llvm_v8i64_ty, llvm_v8f64_ty, llvm_i32_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_scatter_qps_512  : GCCBuiltin<"__builtin_ia32_scatterdiv16sf">,
          Intrinsic<[], [llvm_ptr_ty, llvm_i8_ty,
                     llvm_v8i64_ty, llvm_v8f32_ty, llvm_i32_ty],
                    [IntrArgMemOnly]>;


  def int_x86_avx512_scatter_dpq_512  : GCCBuiltin<"__builtin_ia32_scattersiv8di">,
          Intrinsic<[], [llvm_ptr_ty, llvm_i8_ty,
                         llvm_v8i32_ty, llvm_v8i64_ty, llvm_i32_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_scatter_dpi_512  : GCCBuiltin<"__builtin_ia32_scattersiv16si">,
          Intrinsic<[], [llvm_ptr_ty, llvm_i16_ty,
                     llvm_v16i32_ty, llvm_v16i32_ty, llvm_i32_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_scatter_qpq_512  : GCCBuiltin<"__builtin_ia32_scatterdiv8di">,
          Intrinsic<[], [llvm_ptr_ty, llvm_i8_ty,llvm_v8i64_ty, llvm_v8i64_ty,
                         llvm_i32_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_scatter_qpi_512  : GCCBuiltin<"__builtin_ia32_scatterdiv16si">,
          Intrinsic<[], [llvm_ptr_ty, llvm_i8_ty, llvm_v8i64_ty, llvm_v8i32_ty,
                         llvm_i32_ty],
                    [IntrArgMemOnly]>;

  def int_x86_avx512_scatterdiv2_df :
       GCCBuiltin<"__builtin_ia32_scatterdiv2df">,
        Intrinsic<[],
        [llvm_ptr_ty, llvm_i8_ty, llvm_v2i64_ty, llvm_v2f64_ty, llvm_i32_ty],
        [IntrArgMemOnly]>;

  def int_x86_avx512_scatterdiv2_di :
        GCCBuiltin<"__builtin_ia32_scatterdiv2di">,
          Intrinsic<[],
          [llvm_ptr_ty, llvm_i8_ty, llvm_v2i64_ty, llvm_v2i64_ty, llvm_i32_ty],
          [IntrArgMemOnly]>;

  def int_x86_avx512_scatterdiv4_df :
        GCCBuiltin<"__builtin_ia32_scatterdiv4df">,
          Intrinsic<[],
          [llvm_ptr_ty, llvm_i8_ty, llvm_v4i64_ty, llvm_v4f64_ty, llvm_i32_ty],
          [IntrArgMemOnly]>;

  def int_x86_avx512_scatterdiv4_di :
        GCCBuiltin<"__builtin_ia32_scatterdiv4di">,
          Intrinsic<[],
          [llvm_ptr_ty, llvm_i8_ty, llvm_v4i64_ty, llvm_v4i64_ty, llvm_i32_ty],
          [IntrArgMemOnly]>;

  def int_x86_avx512_scatterdiv4_sf :
        GCCBuiltin<"__builtin_ia32_scatterdiv4sf">,
          Intrinsic<[],
          [llvm_ptr_ty, llvm_i8_ty, llvm_v2i64_ty, llvm_v4f32_ty, llvm_i32_ty],
          [IntrArgMemOnly]>;

  def int_x86_avx512_scatterdiv4_si :
        GCCBuiltin<"__builtin_ia32_scatterdiv4si">,
          Intrinsic<[],
          [llvm_ptr_ty, llvm_i8_ty, llvm_v2i64_ty, llvm_v4i32_ty, llvm_i32_ty],
          [IntrArgMemOnly]>;

  def int_x86_avx512_scatterdiv8_sf :
        GCCBuiltin<"__builtin_ia32_scatterdiv8sf">,
          Intrinsic<[],
          [llvm_ptr_ty, llvm_i8_ty, llvm_v4i64_ty, llvm_v4f32_ty, llvm_i32_ty],
          [IntrArgMemOnly]>;

  def int_x86_avx512_scatterdiv8_si :
        GCCBuiltin<"__builtin_ia32_scatterdiv8si">,
          Intrinsic<[],
          [llvm_ptr_ty, llvm_i8_ty, llvm_v4i64_ty, llvm_v4i32_ty, llvm_i32_ty],
          [IntrArgMemOnly]>;

  def int_x86_avx512_scattersiv2_df :
        GCCBuiltin<"__builtin_ia32_scattersiv2df">,
          Intrinsic<[],
          [llvm_ptr_ty, llvm_i8_ty, llvm_v4i32_ty, llvm_v2f64_ty, llvm_i32_ty],
          [IntrArgMemOnly]>;

  def int_x86_avx512_scattersiv2_di :
        GCCBuiltin<"__builtin_ia32_scattersiv2di">,
          Intrinsic<[],
          [llvm_ptr_ty, llvm_i8_ty, llvm_v4i32_ty, llvm_v2i64_ty, llvm_i32_ty],
          [IntrArgMemOnly]>;

  def int_x86_avx512_scattersiv4_df :
        GCCBuiltin<"__builtin_ia32_scattersiv4df">,
          Intrinsic<[],
          [llvm_ptr_ty, llvm_i8_ty, llvm_v4i32_ty, llvm_v4f64_ty, llvm_i32_ty],
          [IntrArgMemOnly]>;

  def int_x86_avx512_scattersiv4_di :
        GCCBuiltin<"__builtin_ia32_scattersiv4di">,
          Intrinsic<[],
          [llvm_ptr_ty, llvm_i8_ty, llvm_v4i32_ty, llvm_v4i64_ty, llvm_i32_ty],
          [IntrArgMemOnly]>;

  def int_x86_avx512_scattersiv4_sf :
        GCCBuiltin<"__builtin_ia32_scattersiv4sf">,
          Intrinsic<[],
          [llvm_ptr_ty, llvm_i8_ty, llvm_v4i32_ty, llvm_v4f32_ty, llvm_i32_ty],
          [IntrArgMemOnly]>;

  def int_x86_avx512_scattersiv4_si :
        GCCBuiltin<"__builtin_ia32_scattersiv4si">,
          Intrinsic<[],
          [llvm_ptr_ty, llvm_i8_ty, llvm_v4i32_ty, llvm_v4i32_ty, llvm_i32_ty],
          [IntrArgMemOnly]>;

  def int_x86_avx512_scattersiv8_sf :
        GCCBuiltin<"__builtin_ia32_scattersiv8sf">,
          Intrinsic<[],
          [llvm_ptr_ty, llvm_i8_ty, llvm_v8i32_ty, llvm_v8f32_ty, llvm_i32_ty],
          [IntrArgMemOnly]>;

  def int_x86_avx512_scattersiv8_si :
        GCCBuiltin<"__builtin_ia32_scattersiv8si">,
          Intrinsic<[],
          [llvm_ptr_ty, llvm_i8_ty, llvm_v8i32_ty, llvm_v8i32_ty, llvm_i32_ty],
          [IntrArgMemOnly]>;

  // gather prefetch
  def int_x86_avx512_gatherpf_dpd_512  : GCCBuiltin<"__builtin_ia32_gatherpfdpd">,
          Intrinsic<[], [llvm_i8_ty, llvm_v8i32_ty, llvm_ptr_ty,
                     llvm_i32_ty, llvm_i32_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_gatherpf_dps_512  : GCCBuiltin<"__builtin_ia32_gatherpfdps">,
          Intrinsic<[], [llvm_i16_ty, llvm_v16i32_ty, llvm_ptr_ty,
                     llvm_i32_ty, llvm_i32_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_gatherpf_qpd_512  : GCCBuiltin<"__builtin_ia32_gatherpfqpd">,
          Intrinsic<[], [llvm_i8_ty, llvm_v8i64_ty, llvm_ptr_ty,
                     llvm_i32_ty, llvm_i32_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_gatherpf_qps_512  : GCCBuiltin<"__builtin_ia32_gatherpfqps">,
          Intrinsic<[], [llvm_i8_ty, llvm_v8i64_ty, llvm_ptr_ty,
                     llvm_i32_ty, llvm_i32_ty], [IntrArgMemOnly]>;

  // scatter prefetch
  def int_x86_avx512_scatterpf_dpd_512  : GCCBuiltin<"__builtin_ia32_scatterpfdpd">,
          Intrinsic<[], [llvm_i8_ty, llvm_v8i32_ty, llvm_ptr_ty,
                     llvm_i32_ty, llvm_i32_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_scatterpf_dps_512  : GCCBuiltin<"__builtin_ia32_scatterpfdps">,
          Intrinsic<[], [llvm_i16_ty, llvm_v16i32_ty, llvm_ptr_ty,
                     llvm_i32_ty, llvm_i32_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_scatterpf_qpd_512  : GCCBuiltin<"__builtin_ia32_scatterpfqpd">,
          Intrinsic<[], [llvm_i8_ty, llvm_v8i64_ty, llvm_ptr_ty,
                     llvm_i32_ty, llvm_i32_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_scatterpf_qps_512  : GCCBuiltin<"__builtin_ia32_scatterpfqps">,
          Intrinsic<[], [llvm_i8_ty, llvm_v8i64_ty, llvm_ptr_ty,
                     llvm_i32_ty, llvm_i32_ty], [IntrArgMemOnly]>;
}

// AVX-512 conflict detection instruction
// Instructions that count the number of leading zero bits
let TargetPrefix = "x86" in {
  def int_x86_avx512_mask_conflict_d_128 :
          GCCBuiltin<"__builtin_ia32_vpconflictsi_128_mask">,
          Intrinsic<[llvm_v4i32_ty],
                    [llvm_v4i32_ty, llvm_v4i32_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_conflict_d_256 :
          GCCBuiltin<"__builtin_ia32_vpconflictsi_256_mask">,
          Intrinsic<[llvm_v8i32_ty],
                    [llvm_v8i32_ty, llvm_v8i32_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_conflict_d_512 :
          GCCBuiltin<"__builtin_ia32_vpconflictsi_512_mask">,
          Intrinsic<[llvm_v16i32_ty],
                    [llvm_v16i32_ty, llvm_v16i32_ty, llvm_i16_ty],
                    [IntrNoMem]>;

  def int_x86_avx512_mask_conflict_q_128 :
          GCCBuiltin<"__builtin_ia32_vpconflictdi_128_mask">,
          Intrinsic<[llvm_v2i64_ty],
                    [llvm_v2i64_ty, llvm_v2i64_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_conflict_q_256 :
          GCCBuiltin<"__builtin_ia32_vpconflictdi_256_mask">,
          Intrinsic<[llvm_v4i64_ty],
                    [llvm_v4i64_ty, llvm_v4i64_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_conflict_q_512 :
          GCCBuiltin<"__builtin_ia32_vpconflictdi_512_mask">,
          Intrinsic<[llvm_v8i64_ty],
                    [llvm_v8i64_ty, llvm_v8i64_ty, llvm_i8_ty],
                    [IntrNoMem]>;
}

// Compares
let TargetPrefix = "x86" in {
  // 512-bit
  def int_x86_avx512_vcomi_sd : GCCBuiltin<"__builtin_ia32_vcomisd">,
              Intrinsic<[llvm_i32_ty], [llvm_v2f64_ty,
                         llvm_v2f64_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_vcomi_ss : GCCBuiltin<"__builtin_ia32_vcomiss">,
              Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_cmp_b_512:
        Intrinsic<[llvm_i64_ty], [llvm_v64i8_ty, llvm_v64i8_ty, llvm_i32_ty,
                  llvm_i64_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_cmp_w_512:
        Intrinsic<[llvm_i32_ty], [llvm_v32i16_ty, llvm_v32i16_ty, llvm_i32_ty,
                  llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_cmp_d_512:
        Intrinsic<[llvm_i16_ty], [llvm_v16i32_ty, llvm_v16i32_ty, llvm_i32_ty,
                  llvm_i16_ty], [IntrNoMem ]>;
  def int_x86_avx512_mask_cmp_q_512:
        Intrinsic<[llvm_i8_ty], [llvm_v8i64_ty, llvm_v8i64_ty, llvm_i32_ty,
                  llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_ucmp_b_512:
        Intrinsic<[llvm_i64_ty], [llvm_v64i8_ty, llvm_v64i8_ty, llvm_i32_ty,
                  llvm_i64_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_ucmp_w_512:
        Intrinsic<[llvm_i32_ty], [llvm_v32i16_ty, llvm_v32i16_ty, llvm_i32_ty,
                  llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_ucmp_d_512:
        Intrinsic<[llvm_i16_ty], [llvm_v16i32_ty, llvm_v16i32_ty, llvm_i32_ty,
                  llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_ucmp_q_512:
        Intrinsic<[llvm_i8_ty], [llvm_v8i64_ty, llvm_v8i64_ty, llvm_i32_ty,
                  llvm_i8_ty], [IntrNoMem]>;

  // 256-bit
  def int_x86_avx512_mask_cmp_b_256:
        Intrinsic<[llvm_i32_ty], [llvm_v32i8_ty, llvm_v32i8_ty, llvm_i32_ty,
                  llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_cmp_w_256:
        Intrinsic<[llvm_i16_ty], [llvm_v16i16_ty, llvm_v16i16_ty, llvm_i32_ty,
                  llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_cmp_d_256:
        Intrinsic<[llvm_i8_ty], [llvm_v8i32_ty, llvm_v8i32_ty, llvm_i32_ty,
                  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_cmp_q_256:
        Intrinsic<[llvm_i8_ty], [llvm_v4i64_ty, llvm_v4i64_ty, llvm_i32_ty,
                  llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_ucmp_b_256:
        Intrinsic<[llvm_i32_ty], [llvm_v32i8_ty, llvm_v32i8_ty, llvm_i32_ty,
                  llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_ucmp_w_256:
        Intrinsic<[llvm_i16_ty], [llvm_v16i16_ty, llvm_v16i16_ty, llvm_i32_ty,
                  llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_ucmp_d_256:
        Intrinsic<[llvm_i8_ty], [llvm_v8i32_ty, llvm_v8i32_ty, llvm_i32_ty,
                  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_ucmp_q_256:
        Intrinsic<[llvm_i8_ty], [llvm_v4i64_ty, llvm_v4i64_ty, llvm_i32_ty,
                  llvm_i8_ty], [IntrNoMem]>;

  // 128-bit
  def int_x86_avx512_mask_cmp_b_128:
        Intrinsic<[llvm_i16_ty], [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty,
                  llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_cmp_w_128:
        Intrinsic<[llvm_i8_ty], [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty,
                  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_cmp_d_128:
        Intrinsic<[llvm_i8_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_i32_ty,
                  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_cmp_q_128:
        Intrinsic<[llvm_i8_ty], [llvm_v2i64_ty, llvm_v2i64_ty, llvm_i32_ty,
                  llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_ucmp_b_128:
        Intrinsic<[llvm_i16_ty], [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty,
                  llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_ucmp_w_128:
        Intrinsic<[llvm_i8_ty], [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty,
                  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_ucmp_d_128:
        Intrinsic<[llvm_i8_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_i32_ty,
                  llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_ucmp_q_128:
        Intrinsic<[llvm_i8_ty], [llvm_v2i64_ty, llvm_v2i64_ty, llvm_i32_ty,
                  llvm_i8_ty], [IntrNoMem]>;
}

// Compress, Expand
let TargetPrefix = "x86" in {
  def int_x86_avx512_mask_compress_ps_512 :
                             GCCBuiltin<"__builtin_ia32_compresssf512_mask">,
        Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                   llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_compress_pd_512 :
                             GCCBuiltin<"__builtin_ia32_compressdf512_mask">,
        Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_compress_ps_256 :
                             GCCBuiltin<"__builtin_ia32_compresssf256_mask">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty, llvm_v8f32_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_compress_pd_256 :
                             GCCBuiltin<"__builtin_ia32_compressdf256_mask">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty, llvm_v4f64_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_compress_ps_128 :
                             GCCBuiltin<"__builtin_ia32_compresssf128_mask">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_compress_pd_128 :
                             GCCBuiltin<"__builtin_ia32_compressdf128_mask">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                   llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_compress_store_ps_512 :
                            GCCBuiltin<"__builtin_ia32_compressstoresf512_mask">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v16f32_ty,
                   llvm_i16_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_mask_compress_store_pd_512 :
                            GCCBuiltin<"__builtin_ia32_compressstoredf512_mask">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v8f64_ty,
                   llvm_i8_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_mask_compress_store_ps_256 :
                            GCCBuiltin<"__builtin_ia32_compressstoresf256_mask">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v8f32_ty,
                   llvm_i8_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_mask_compress_store_pd_256 :
                            GCCBuiltin<"__builtin_ia32_compressstoredf256_mask">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v4f64_ty,
                   llvm_i8_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_mask_compress_store_ps_128 :
                            GCCBuiltin<"__builtin_ia32_compressstoresf128_mask">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v4f32_ty,
                   llvm_i8_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_mask_compress_store_pd_128 :
                            GCCBuiltin<"__builtin_ia32_compressstoredf128_mask">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v2f64_ty,
                   llvm_i8_ty], [IntrArgMemOnly]>;

  def int_x86_avx512_mask_compress_d_512 :
                             GCCBuiltin<"__builtin_ia32_compresssi512_mask">,
        Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty, llvm_v16i32_ty,
                   llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_compress_q_512 :
                             GCCBuiltin<"__builtin_ia32_compressdi512_mask">,
        Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty, llvm_v8i64_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_compress_d_256 :
                             GCCBuiltin<"__builtin_ia32_compresssi256_mask">,
        Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty, llvm_v8i32_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_compress_q_256 :
                             GCCBuiltin<"__builtin_ia32_compressdi256_mask">,
        Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty, llvm_v4i64_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_compress_d_128 :
                             GCCBuiltin<"__builtin_ia32_compresssi128_mask">,
        Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_compress_q_128 :
                             GCCBuiltin<"__builtin_ia32_compressdi128_mask">,
        Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty,
                   llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_compress_store_d_512 :
                            GCCBuiltin<"__builtin_ia32_compressstoresi512_mask">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v16i32_ty,
                   llvm_i16_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_mask_compress_store_q_512 :
                            GCCBuiltin<"__builtin_ia32_compressstoredi512_mask">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v8i64_ty,
                   llvm_i8_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_mask_compress_store_d_256 :
                            GCCBuiltin<"__builtin_ia32_compressstoresi256_mask">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v8i32_ty,
                   llvm_i8_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_mask_compress_store_q_256 :
                            GCCBuiltin<"__builtin_ia32_compressstoredi256_mask">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v4i64_ty,
                   llvm_i8_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_mask_compress_store_d_128 :
                            GCCBuiltin<"__builtin_ia32_compressstoresi128_mask">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v4i32_ty,
                   llvm_i8_ty], [IntrArgMemOnly]>;
  def int_x86_avx512_mask_compress_store_q_128 :
                            GCCBuiltin<"__builtin_ia32_compressstoredi128_mask">,
        Intrinsic<[], [llvm_ptr_ty, llvm_v2i64_ty,
                   llvm_i8_ty], [IntrArgMemOnly]>;

// expand
  def int_x86_avx512_mask_expand_ps_512 :
                             GCCBuiltin<"__builtin_ia32_expandsf512_mask">,
        Intrinsic<[llvm_v16f32_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                   llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_expand_pd_512 :
                             GCCBuiltin<"__builtin_ia32_expanddf512_mask">,
        Intrinsic<[llvm_v8f64_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_expand_ps_256 :
                             GCCBuiltin<"__builtin_ia32_expandsf256_mask">,
        Intrinsic<[llvm_v8f32_ty], [llvm_v8f32_ty, llvm_v8f32_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_expand_pd_256 :
                             GCCBuiltin<"__builtin_ia32_expanddf256_mask">,
        Intrinsic<[llvm_v4f64_ty], [llvm_v4f64_ty, llvm_v4f64_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_expand_ps_128 :
                             GCCBuiltin<"__builtin_ia32_expandsf128_mask">,
        Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_expand_pd_128 :
                             GCCBuiltin<"__builtin_ia32_expanddf128_mask">,
        Intrinsic<[llvm_v2f64_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                   llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_expand_load_ps_512 :
                            GCCBuiltin<"__builtin_ia32_expandloadsf512_mask">,
        Intrinsic<[llvm_v16f32_ty], [llvm_ptr_ty, llvm_v16f32_ty,
                   llvm_i16_ty], [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_mask_expand_load_pd_512 :
                            GCCBuiltin<"__builtin_ia32_expandloaddf512_mask">,
        Intrinsic<[llvm_v8f64_ty], [llvm_ptr_ty, llvm_v8f64_ty,
                   llvm_i8_ty], [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_mask_expand_load_ps_256 :
                            GCCBuiltin<"__builtin_ia32_expandloadsf256_mask">,
        Intrinsic<[llvm_v8f32_ty], [llvm_ptr_ty, llvm_v8f32_ty,
                   llvm_i8_ty], [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_mask_expand_load_pd_256 :
                            GCCBuiltin<"__builtin_ia32_expandloaddf256_mask">,
        Intrinsic<[llvm_v4f64_ty], [llvm_ptr_ty, llvm_v4f64_ty,
                   llvm_i8_ty], [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_mask_expand_load_ps_128 :
                            GCCBuiltin<"__builtin_ia32_expandloadsf128_mask">,
        Intrinsic<[llvm_v4f32_ty], [llvm_ptr_ty, llvm_v4f32_ty,
                   llvm_i8_ty], [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_mask_expand_load_pd_128 :
                            GCCBuiltin<"__builtin_ia32_expandloaddf128_mask">,
        Intrinsic<[llvm_v2f64_ty], [llvm_ptr_ty, llvm_v2f64_ty,
                   llvm_i8_ty], [IntrReadMem, IntrArgMemOnly]>;

  def int_x86_avx512_mask_expand_d_512 :
                             GCCBuiltin<"__builtin_ia32_expandsi512_mask">,
        Intrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty, llvm_v16i32_ty,
                   llvm_i16_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_expand_q_512 :
                             GCCBuiltin<"__builtin_ia32_expanddi512_mask">,
        Intrinsic<[llvm_v8i64_ty], [llvm_v8i64_ty, llvm_v8i64_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_expand_d_256 :
                             GCCBuiltin<"__builtin_ia32_expandsi256_mask">,
        Intrinsic<[llvm_v8i32_ty], [llvm_v8i32_ty, llvm_v8i32_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_expand_q_256 :
                             GCCBuiltin<"__builtin_ia32_expanddi256_mask">,
        Intrinsic<[llvm_v4i64_ty], [llvm_v4i64_ty, llvm_v4i64_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_expand_d_128 :
                             GCCBuiltin<"__builtin_ia32_expandsi128_mask">,
        Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty,
                   llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_expand_q_128 :
                             GCCBuiltin<"__builtin_ia32_expanddi128_mask">,
        Intrinsic<[llvm_v2i64_ty], [llvm_v2i64_ty, llvm_v2i64_ty,
                   llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_expand_load_d_512 :
                            GCCBuiltin<"__builtin_ia32_expandloadsi512_mask">,
        Intrinsic<[llvm_v16i32_ty], [llvm_ptr_ty, llvm_v16i32_ty,
                   llvm_i16_ty], [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_mask_expand_load_q_512 :
                            GCCBuiltin<"__builtin_ia32_expandloaddi512_mask">,
        Intrinsic<[llvm_v8i64_ty], [llvm_ptr_ty, llvm_v8i64_ty,
                   llvm_i8_ty], [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_mask_expand_load_d_256 :
                            GCCBuiltin<"__builtin_ia32_expandloadsi256_mask">,
        Intrinsic<[llvm_v8i32_ty], [llvm_ptr_ty, llvm_v8i32_ty,
                   llvm_i8_ty], [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_mask_expand_load_q_256 :
                            GCCBuiltin<"__builtin_ia32_expandloaddi256_mask">,
        Intrinsic<[llvm_v4i64_ty], [llvm_ptr_ty, llvm_v4i64_ty,
                   llvm_i8_ty], [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_mask_expand_load_d_128 :
                            GCCBuiltin<"__builtin_ia32_expandloadsi128_mask">,
        Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty, llvm_v4i32_ty,
                   llvm_i8_ty], [IntrReadMem, IntrArgMemOnly]>;
  def int_x86_avx512_mask_expand_load_q_128 :
                            GCCBuiltin<"__builtin_ia32_expandloaddi128_mask">,
        Intrinsic<[llvm_v2i64_ty], [llvm_ptr_ty, llvm_v2i64_ty,
                   llvm_i8_ty], [IntrReadMem, IntrArgMemOnly]>;

}

// truncate
let TargetPrefix = "x86" in {
  def int_x86_avx512_mask_pmov_qb_128 :
          GCCBuiltin<"__builtin_ia32_pmovqb128_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v2i64_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_qb_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovqb128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v2i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_qb_128 :
          GCCBuiltin<"__builtin_ia32_pmovsqb128_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v2i64_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_qb_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovsqb128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v2i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_qb_128 :
          GCCBuiltin<"__builtin_ia32_pmovusqb128_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v2i64_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_qb_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovusqb128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v2i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_qb_256 :
          GCCBuiltin<"__builtin_ia32_pmovqb256_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v4i64_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_qb_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovqb256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v4i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_qb_256 :
          GCCBuiltin<"__builtin_ia32_pmovsqb256_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v4i64_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_qb_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovsqb256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v4i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_qb_256 :
          GCCBuiltin<"__builtin_ia32_pmovusqb256_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v4i64_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_qb_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovusqb256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v4i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_qb_512 :
          GCCBuiltin<"__builtin_ia32_pmovqb512_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v8i64_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_qb_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovqb512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_qb_512 :
          GCCBuiltin<"__builtin_ia32_pmovsqb512_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v8i64_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_qb_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovsqb512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_qb_512 :
          GCCBuiltin<"__builtin_ia32_pmovusqb512_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v8i64_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_qb_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovusqb512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_qw_128 :
          GCCBuiltin<"__builtin_ia32_pmovqw128_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_v2i64_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_qw_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovqw128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v2i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_qw_128 :
          GCCBuiltin<"__builtin_ia32_pmovsqw128_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_v2i64_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_qw_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovsqw128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v2i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_qw_128 :
          GCCBuiltin<"__builtin_ia32_pmovusqw128_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_v2i64_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_qw_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovusqw128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v2i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_qw_256 :
          GCCBuiltin<"__builtin_ia32_pmovqw256_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_v4i64_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_qw_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovqw256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v4i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_qw_256 :
          GCCBuiltin<"__builtin_ia32_pmovsqw256_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_v4i64_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_qw_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovsqw256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v4i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_qw_256 :
          GCCBuiltin<"__builtin_ia32_pmovusqw256_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_v4i64_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_qw_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovusqw256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v4i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_qw_512 :
          GCCBuiltin<"__builtin_ia32_pmovqw512_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_v8i64_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_qw_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovqw512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_qw_512 :
          GCCBuiltin<"__builtin_ia32_pmovsqw512_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_v8i64_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_qw_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovsqw512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_qw_512 :
          GCCBuiltin<"__builtin_ia32_pmovusqw512_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_v8i64_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_qw_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovusqw512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_qd_128 :
          GCCBuiltin<"__builtin_ia32_pmovqd128_mask">,
          Intrinsic<[llvm_v4i32_ty],
                    [llvm_v2i64_ty, llvm_v4i32_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_qd_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovqd128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v2i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_qd_128 :
          GCCBuiltin<"__builtin_ia32_pmovsqd128_mask">,
          Intrinsic<[llvm_v4i32_ty],
                    [llvm_v2i64_ty, llvm_v4i32_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_qd_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovsqd128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v2i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_qd_128 :
          GCCBuiltin<"__builtin_ia32_pmovusqd128_mask">,
          Intrinsic<[llvm_v4i32_ty],
                    [llvm_v2i64_ty, llvm_v4i32_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_qd_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovusqd128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v2i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_qd_256 :
          GCCBuiltin<"__builtin_ia32_pmovqd256_mask">,
          Intrinsic<[llvm_v4i32_ty],
                    [llvm_v4i64_ty, llvm_v4i32_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_qd_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovqd256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v4i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_qd_256 :
          GCCBuiltin<"__builtin_ia32_pmovsqd256_mask">,
          Intrinsic<[llvm_v4i32_ty],
                    [llvm_v4i64_ty, llvm_v4i32_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_qd_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovsqd256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v4i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_qd_256 :
          GCCBuiltin<"__builtin_ia32_pmovusqd256_mask">,
          Intrinsic<[llvm_v4i32_ty],
                    [llvm_v4i64_ty, llvm_v4i32_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_qd_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovusqd256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v4i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_qd_512 :
          GCCBuiltin<"__builtin_ia32_pmovqd512_mask">,
          Intrinsic<[llvm_v8i32_ty],
                    [llvm_v8i64_ty, llvm_v8i32_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_qd_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovqd512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_qd_512 :
          GCCBuiltin<"__builtin_ia32_pmovsqd512_mask">,
          Intrinsic<[llvm_v8i32_ty],
                    [llvm_v8i64_ty, llvm_v8i32_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_qd_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovsqd512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_qd_512 :
          GCCBuiltin<"__builtin_ia32_pmovusqd512_mask">,
          Intrinsic<[llvm_v8i32_ty],
                    [llvm_v8i64_ty, llvm_v8i32_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_qd_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovusqd512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i64_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_db_128 :
          GCCBuiltin<"__builtin_ia32_pmovdb128_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v4i32_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_db_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovdb128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v4i32_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_db_128 :
          GCCBuiltin<"__builtin_ia32_pmovsdb128_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v4i32_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_db_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovsdb128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v4i32_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_db_128 :
          GCCBuiltin<"__builtin_ia32_pmovusdb128_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v4i32_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_db_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovusdb128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v4i32_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_db_256 :
          GCCBuiltin<"__builtin_ia32_pmovdb256_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v8i32_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_db_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovdb256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i32_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_db_256 :
          GCCBuiltin<"__builtin_ia32_pmovsdb256_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v8i32_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_db_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovsdb256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i32_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_db_256 :
          GCCBuiltin<"__builtin_ia32_pmovusdb256_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v8i32_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_db_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovusdb256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i32_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_db_512 :
          GCCBuiltin<"__builtin_ia32_pmovdb512_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v16i32_ty, llvm_v16i8_ty, llvm_i16_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_db_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovdb512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v16i32_ty, llvm_i16_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_db_512 :
          GCCBuiltin<"__builtin_ia32_pmovsdb512_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v16i32_ty, llvm_v16i8_ty, llvm_i16_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_db_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovsdb512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v16i32_ty, llvm_i16_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_db_512 :
          GCCBuiltin<"__builtin_ia32_pmovusdb512_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v16i32_ty, llvm_v16i8_ty, llvm_i16_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_db_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovusdb512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v16i32_ty, llvm_i16_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_dw_128 :
          GCCBuiltin<"__builtin_ia32_pmovdw128_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_v4i32_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_dw_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovdw128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v4i32_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_dw_128 :
          GCCBuiltin<"__builtin_ia32_pmovsdw128_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_v4i32_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_dw_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovsdw128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v4i32_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_dw_128 :
          GCCBuiltin<"__builtin_ia32_pmovusdw128_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_v4i32_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_dw_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovusdw128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v4i32_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_dw_256 :
          GCCBuiltin<"__builtin_ia32_pmovdw256_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_v8i32_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_dw_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovdw256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i32_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_dw_256 :
          GCCBuiltin<"__builtin_ia32_pmovsdw256_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_v8i32_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_dw_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovsdw256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i32_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_dw_256 :
          GCCBuiltin<"__builtin_ia32_pmovusdw256_mask">,
          Intrinsic<[llvm_v8i16_ty],
                    [llvm_v8i32_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_dw_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovusdw256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i32_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_dw_512 :
          GCCBuiltin<"__builtin_ia32_pmovdw512_mask">,
          Intrinsic<[llvm_v16i16_ty],
                    [llvm_v16i32_ty, llvm_v16i16_ty, llvm_i16_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_dw_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovdw512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v16i32_ty, llvm_i16_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_dw_512 :
          GCCBuiltin<"__builtin_ia32_pmovsdw512_mask">,
          Intrinsic<[llvm_v16i16_ty],
                    [llvm_v16i32_ty, llvm_v16i16_ty, llvm_i16_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_dw_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovsdw512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v16i32_ty, llvm_i16_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_dw_512 :
          GCCBuiltin<"__builtin_ia32_pmovusdw512_mask">,
          Intrinsic<[llvm_v16i16_ty],
                    [llvm_v16i32_ty, llvm_v16i16_ty, llvm_i16_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_dw_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovusdw512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v16i32_ty, llvm_i16_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_wb_128 :
          GCCBuiltin<"__builtin_ia32_pmovwb128_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v8i16_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_wb_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovwb128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_wb_128 :
          GCCBuiltin<"__builtin_ia32_pmovswb128_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v8i16_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_wb_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovswb128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_wb_128 :
          GCCBuiltin<"__builtin_ia32_pmovuswb128_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v8i16_ty, llvm_v16i8_ty, llvm_i8_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_wb_mem_128 :
          GCCBuiltin<"__builtin_ia32_pmovuswb128mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v8i16_ty, llvm_i8_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_wb_256 :
          GCCBuiltin<"__builtin_ia32_pmovwb256_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v16i16_ty, llvm_v16i8_ty, llvm_i16_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_wb_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovwb256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v16i16_ty, llvm_i16_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_wb_256 :
          GCCBuiltin<"__builtin_ia32_pmovswb256_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v16i16_ty, llvm_v16i8_ty, llvm_i16_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_wb_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovswb256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v16i16_ty, llvm_i16_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_wb_256 :
          GCCBuiltin<"__builtin_ia32_pmovuswb256_mask">,
          Intrinsic<[llvm_v16i8_ty],
                    [llvm_v16i16_ty, llvm_v16i8_ty, llvm_i16_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_wb_mem_256 :
          GCCBuiltin<"__builtin_ia32_pmovuswb256mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v16i16_ty, llvm_i16_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmov_wb_512 :
          GCCBuiltin<"__builtin_ia32_pmovwb512_mask">,
          Intrinsic<[llvm_v32i8_ty],
                    [llvm_v32i16_ty, llvm_v32i8_ty, llvm_i32_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmov_wb_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovwb512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v32i16_ty, llvm_i32_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovs_wb_512 :
          GCCBuiltin<"__builtin_ia32_pmovswb512_mask">,
          Intrinsic<[llvm_v32i8_ty],
                    [llvm_v32i16_ty, llvm_v32i8_ty, llvm_i32_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovs_wb_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovswb512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v32i16_ty, llvm_i32_ty],
                    [IntrArgMemOnly]>;
  def int_x86_avx512_mask_pmovus_wb_512 :
          GCCBuiltin<"__builtin_ia32_pmovuswb512_mask">,
          Intrinsic<[llvm_v32i8_ty],
                    [llvm_v32i16_ty, llvm_v32i8_ty, llvm_i32_ty],
                    [IntrNoMem]>;
  def int_x86_avx512_mask_pmovus_wb_mem_512 :
          GCCBuiltin<"__builtin_ia32_pmovuswb512mem_mask">,
          Intrinsic<[],
                    [llvm_ptr_ty, llvm_v32i16_ty, llvm_i32_ty],
                    [IntrArgMemOnly]>;
}

// Bitwise ternary logic
let TargetPrefix = "x86" in {
  def int_x86_avx512_mask_pternlog_d_128 :
          GCCBuiltin<"__builtin_ia32_pternlogd128_mask">,
          Intrinsic<[llvm_v4i32_ty],
                    [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty, llvm_i32_ty,
                     llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_maskz_pternlog_d_128 :
          GCCBuiltin<"__builtin_ia32_pternlogd128_maskz">,
          Intrinsic<[llvm_v4i32_ty],
                    [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty, llvm_i32_ty,
                     llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_pternlog_d_256 :
          GCCBuiltin<"__builtin_ia32_pternlogd256_mask">,
          Intrinsic<[llvm_v8i32_ty],
                    [llvm_v8i32_ty, llvm_v8i32_ty, llvm_v8i32_ty, llvm_i32_ty,
                     llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_maskz_pternlog_d_256 :
          GCCBuiltin<"__builtin_ia32_pternlogd256_maskz">,
          Intrinsic<[llvm_v8i32_ty],
                    [llvm_v8i32_ty, llvm_v8i32_ty, llvm_v8i32_ty, llvm_i32_ty,
                     llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_pternlog_d_512 :
          GCCBuiltin<"__builtin_ia32_pternlogd512_mask">,
          Intrinsic<[llvm_v16i32_ty],
                    [llvm_v16i32_ty, llvm_v16i32_ty, llvm_v16i32_ty, llvm_i32_ty,
                     llvm_i16_ty], [IntrNoMem]>;

  def int_x86_avx512_maskz_pternlog_d_512 :
          GCCBuiltin<"__builtin_ia32_pternlogd512_maskz">,
          Intrinsic<[llvm_v16i32_ty],
                    [llvm_v16i32_ty, llvm_v16i32_ty, llvm_v16i32_ty, llvm_i32_ty,
                     llvm_i16_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_pternlog_q_128 :
          GCCBuiltin<"__builtin_ia32_pternlogq128_mask">,
          Intrinsic<[llvm_v2i64_ty],
                    [llvm_v2i64_ty, llvm_v2i64_ty, llvm_v2i64_ty, llvm_i32_ty,
                     llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_maskz_pternlog_q_128 :
          GCCBuiltin<"__builtin_ia32_pternlogq128_maskz">,
          Intrinsic<[llvm_v2i64_ty],
                    [llvm_v2i64_ty, llvm_v2i64_ty, llvm_v2i64_ty, llvm_i32_ty,
                     llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_pternlog_q_256 :
          GCCBuiltin<"__builtin_ia32_pternlogq256_mask">,
          Intrinsic<[llvm_v4i64_ty],
                    [llvm_v4i64_ty, llvm_v4i64_ty, llvm_v4i64_ty, llvm_i32_ty,
                     llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_maskz_pternlog_q_256 :
          GCCBuiltin<"__builtin_ia32_pternlogq256_maskz">,
          Intrinsic<[llvm_v4i64_ty],
                    [llvm_v4i64_ty, llvm_v4i64_ty, llvm_v4i64_ty, llvm_i32_ty,
                     llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_mask_pternlog_q_512 :
          GCCBuiltin<"__builtin_ia32_pternlogq512_mask">,
          Intrinsic<[llvm_v8i64_ty],
                    [llvm_v8i64_ty, llvm_v8i64_ty, llvm_v8i64_ty, llvm_i32_ty,
                     llvm_i8_ty], [IntrNoMem]>;

  def int_x86_avx512_maskz_pternlog_q_512 :
          GCCBuiltin<"__builtin_ia32_pternlogq512_maskz">,
          Intrinsic<[llvm_v8i64_ty],
                    [llvm_v8i64_ty, llvm_v8i64_ty, llvm_v8i64_ty, llvm_i32_ty,
                     llvm_i8_ty], [IntrNoMem]>;
}

// Misc.
let TargetPrefix = "x86" in {
  def int_x86_avx512_mask_cmp_ps_512 :
        GCCBuiltin<"__builtin_ia32_cmpps512_mask">,
              Intrinsic<[llvm_i16_ty], [llvm_v16f32_ty, llvm_v16f32_ty,
                         llvm_i32_ty, llvm_i16_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_cmp_pd_512 :
        GCCBuiltin<"__builtin_ia32_cmppd512_mask">,
              Intrinsic<[llvm_i8_ty], [llvm_v8f64_ty, llvm_v8f64_ty,
                         llvm_i32_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_cmp_ps_256 :
        GCCBuiltin<"__builtin_ia32_cmpps256_mask">,
              Intrinsic<[llvm_i8_ty], [llvm_v8f32_ty, llvm_v8f32_ty,
                         llvm_i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_cmp_pd_256 :
        GCCBuiltin<"__builtin_ia32_cmppd256_mask">,
              Intrinsic<[llvm_i8_ty], [llvm_v4f64_ty, llvm_v4f64_ty,
                         llvm_i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_cmp_ps_128 :
        GCCBuiltin<"__builtin_ia32_cmpps128_mask">,
            Intrinsic<[llvm_i8_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                       llvm_i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_cmp_pd_128 :
        GCCBuiltin<"__builtin_ia32_cmppd128_mask">,
            Intrinsic<[llvm_i8_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                       llvm_i32_ty, llvm_i8_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_cmp_ss :
        GCCBuiltin<"__builtin_ia32_cmpss_mask">,
              Intrinsic<[llvm_i8_ty], [llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_i32_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_x86_avx512_mask_cmp_sd :
        GCCBuiltin<"__builtin_ia32_cmpsd_mask">,
              Intrinsic<[llvm_i8_ty], [llvm_v2f64_ty, llvm_v2f64_ty,
                         llvm_i32_ty, llvm_i8_ty, llvm_i32_ty], [IntrNoMem]>;

  def int_x86_avx512_movntdqa :
        GCCBuiltin<"__builtin_ia32_movntdqa512">,
            Intrinsic<[llvm_v8i64_ty], [llvm_ptr_ty], [IntrReadMem]>;
}

//===----------------------------------------------------------------------===//
// SHA intrinsics
let TargetPrefix = "x86" in {
  def int_x86_sha1rnds4 : GCCBuiltin<"__builtin_ia32_sha1rnds4">,
        Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_i8_ty],
                  [IntrNoMem]>;
  def int_x86_sha1nexte : GCCBuiltin<"__builtin_ia32_sha1nexte">,
      Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sha1msg1 : GCCBuiltin<"__builtin_ia32_sha1msg1">,
      Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sha1msg2 : GCCBuiltin<"__builtin_ia32_sha1msg2">,
      Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sha256rnds2 : GCCBuiltin<"__builtin_ia32_sha256rnds2">,
      Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                [IntrNoMem]>;
  def int_x86_sha256msg1 : GCCBuiltin<"__builtin_ia32_sha256msg1">,
      Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem]>;
  def int_x86_sha256msg2 : GCCBuiltin<"__builtin_ia32_sha256msg2">,
      Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem]>;
}

//===----------------------------------------------------------------------===//
// Thread synchronization ops with timer.
let TargetPrefix = "x86" in {
  def int_x86_monitorx
      : GCCBuiltin<"__builtin_ia32_monitorx">,
        Intrinsic<[], [ llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty ], []>;
  def int_x86_mwaitx
      : GCCBuiltin<"__builtin_ia32_mwaitx">,
        Intrinsic<[], [ llvm_i32_ty, llvm_i32_ty, llvm_i32_ty ], []>;
}

//===----------------------------------------------------------------------===//
// Cache-line zero
let TargetPrefix = "x86" in {
  def int_x86_clzero : GCCBuiltin<"__builtin_ia32_clzero">,
      Intrinsic<[], [llvm_ptr_ty], []>;
}
