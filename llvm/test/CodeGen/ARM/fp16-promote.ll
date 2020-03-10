; RUN: llc -asm-verbose=false < %s -mattr=+vfp3,+fp16 | FileCheck -allow-deprecated-dag-overlap %s -check-prefix=CHECK-FP16  --check-prefix=CHECK-VFP -check-prefix=CHECK-ALL
; RUN: llc -asm-verbose=false < %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix=CHECK-LIBCALL --check-prefix=CHECK-VFP -check-prefix=CHECK-ALL --check-prefix=CHECK-LIBCALL-VFP
; RUN: llc -asm-verbose=false < %s -mattr=-fpregs | FileCheck -allow-deprecated-dag-overlap %s --check-prefix=CHECK-LIBCALL -check-prefix=CHECK-NOVFP -check-prefix=CHECK-ALL

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32"
target triple = "armv7---eabihf"

; CHECK-ALL-LABEL: test_fadd:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-VFP: vadd.f32
; CHECK-NOVFP: bl __aeabi_fadd
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_fadd(half* %p, half* %q) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %r = fadd half %a, %b
  store half %r, half* %p
  ret void
}

; CHECK-ALL-LABEL: test_fsub:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-VFP: vsub.f32
; CHECK-NOVFP: bl __aeabi_fsub
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_fsub(half* %p, half* %q) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %r = fsub half %a, %b
  store half %r, half* %p
  ret void
}

; CHECK-ALL-LABEL: test_fmul:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-VFP: vmul.f32
; CHECK-NOVFP: bl __aeabi_fmul
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_fmul(half* %p, half* %q) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %r = fmul half %a, %b
  store half %r, half* %p
  ret void
}

; CHECK-ALL-LABEL: test_fdiv:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-VFP: vdiv.f32
; CHECK-NOVFP: bl __aeabi_fdiv
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_fdiv(half* %p, half* %q) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %r = fdiv half %a, %b
  store half %r, half* %p
  ret void
}

; CHECK-ALL-LABEL: test_frem:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl fmodf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_frem(half* %p, half* %q) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %r = frem half %a, %b
  store half %r, half* %p
  ret void
}

; CHECK-ALL-LABEL: test_load_store:
; CHECK-ALL-NEXT: .fnstart
; CHECK-ALL: ldrh {{r[0-9]+}}, [{{r[0-9]+}}]
; CHECK-ALL: strh {{r[0-9]+}}, [{{r[0-9]+}}]
define void @test_load_store(half* %p, half* %q) #0 {
  %a = load half, half* %p, align 2
  store half %a, half* %q
  ret void
}

; Testing only successfull compilation of function calls.  In ARM ABI, half
; args and returns are handled as f32.

declare half @test_callee(half %a, half %b) #0

; CHECK-ALL-LABEL: test_call:
; CHECK-ALL-NEXT: .fnstart
; CHECK-ALL-NEXT: .save {r11, lr}
; CHECK-ALL-NEXT: push {r11, lr}
; CHECK-ALL-NEXT: bl test_callee
; CHECK-ALL-NEXT: pop {r11, pc}
define half @test_call(half %a, half %b) #0 {
  %r = call half @test_callee(half %a, half %b)
  ret half %r
}

; CHECK-ALL-LABEL: test_call_flipped:
; CHECK-ALL-NEXT: .fnstart
; CHECK-ALL-NEXT: .save {r11, lr}
; CHECK-ALL-NEXT: push {r11, lr}
; CHECK-VFP-NEXT: vmov.f32 s2, s0
; CHECK-VFP-NEXT: vmov.f32 s0, s1
; CHECK-VFP-NEXT: vmov.f32 s1, s2
; CHECK-NOVFP-NEXT: mov r2, r0
; CHECK-NOVFP-NEXT: mov r0, r1
; CHECK-NOVFP-NEXT: mov r1, r2
; CHECK-ALL-NEXT: bl test_callee
; CHECK-ALL-NEXT: pop {r11, pc}
define half @test_call_flipped(half %a, half %b) #0 {
  %r = call half @test_callee(half %b, half %a)
  ret half %r
}

; CHECK-ALL-LABEL: test_tailcall_flipped:
; CHECK-ALL-NEXT: .fnstart
; CHECK-VFP-NEXT: vmov.f32 s2, s0
; CHECK-VFP-NEXT: vmov.f32 s0, s1
; CHECK-VFP-NEXT: vmov.f32 s1, s2
; CHECK-NOVFP-NEXT: mov r2, r0
; CHECK-NOVFP-NEXT: mov r0, r1
; CHECK-NOVFP-NEXT: mov r1, r2
; CHECK-ALL-NEXT: b test_callee
define half @test_tailcall_flipped(half %a, half %b) #0 {
  %r = tail call half @test_callee(half %b, half %a)
  ret half %r
}

; Optimizer picks %p or %q based on %c and only loads that value
; No conversion is needed
; CHECK-ALL-LABEL: test_select:
; CHECK-ALL: cmp {{r[0-9]+}}, #0
; CHECK-ALL: movne {{r[0-9]+}}, {{r[0-9]+}}
; CHECK-ALL: ldrh {{r[0-9]+}}, [{{r[0-9]+}}]
; CHECK-ALL: strh {{r[0-9]+}}, [{{r[0-9]+}}]
define void @test_select(half* %p, half* %q, i1 zeroext %c) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %r = select i1 %c, half %a, half %b
  store half %r, half* %p
  ret void
}

; Test only two variants of fcmp.  These get translated to f32 vcmp
; instructions anyway.
; CHECK-ALL-LABEL: test_fcmp_une:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-VFP: vcmp.f32
; CHECK-NOVFP: bl __aeabi_fcmpeq
; CHECK-VFP-NEXT: vmrs APSR_nzcv, fpscr
; CHECK-VFP-NEXT: movwne
; CHECK-NOVFP-NEXT: clz r0, r0
; CHECK-NOVFP-NEXT: lsr r0, r0, #5
define i1 @test_fcmp_une(half* %p, half* %q) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %r = fcmp une half %a, %b
  ret i1 %r
}

; CHECK-ALL-LABEL: test_fcmp_ueq:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-VFP: vcmp.f32
; CHECK-NOVFP: bl __aeabi_fcmpeq
; CHECK-FP16: vmrs APSR_nzcv, fpscr
; CHECK-LIBCALL: movw{{ne|eq}}
define i1 @test_fcmp_ueq(half* %p, half* %q) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %r = fcmp ueq half %a, %b
  ret i1 %r
}

; CHECK-ALL-LABEL: test_br_cc:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-VFP: vcmp.f32
; CHECK-NOVFP: bl __aeabi_fcmplt
; CHECK-FP16: vmrs APSR_nzcv, fpscr
; CHECK-VFP: strmi
; CHECK-VFP: strpl
; CHECK-NOVFP: strne
; CHECK-NOVFP: streq
define void @test_br_cc(half* %p, half* %q, i32* %p1, i32* %p2) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %c = fcmp uge half %a, %b
  br i1 %c, label %then, label %else
then:
  store i32 0, i32* %p1
  ret void
else:
  store i32 0, i32* %p2
  ret void
}

declare i1 @test_dummy(half* %p) #0
; CHECK-ALL-LABEL: test_phi:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: [[LOOP:.LBB[1-9_]+]]:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl      test_dummy
; CHECK-FP16: bne     [[LOOP]]
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-VFP: bl __aeabi_h2f
; CHECK-LIBCALL: [[LOOP:.LBB[1-9_]+]]:
; CHECK-LIBCALL-VFP: bl __aeabi_h2f
; CHECK-LIBCALL: bl test_dummy
; CHECK-LIBCALL: bne     [[LOOP]]
; CHECK-LIBCALL-VFP: bl __aeabi_f2h
define void @test_phi(half* %p) #0 {
entry:
  %a = load half, half* %p
  br label %loop
loop:
  %r = phi half [%a, %entry], [%b, %loop]
  %b = load half, half* %p
  %c = call i1 @test_dummy(half* %p)
  br i1 %c, label %loop, label %return
return:
  store half %r, half* %p
  ret void
}

; CHECK-ALL-LABEL: test_fptosi_i32:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-VFP: vcvt.s32.f32
; CHECK-NOVFP: bl __aeabi_f2iz
define i32 @test_fptosi_i32(half* %p) #0 {
  %a = load half, half* %p, align 2
  %r = fptosi half %a to i32
  ret i32 %r
}

; CHECK-ALL-LABEL: test_fptosi_i64:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-ALL: bl __aeabi_f2lz
define i64 @test_fptosi_i64(half* %p) #0 {
  %a = load half, half* %p, align 2
  %r = fptosi half %a to i64
  ret i64 %r
}

; CHECK-ALL-LABEL: test_fptoui_i32:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-VFP: vcvt.u32.f32
; CHECK-NOVFP: bl __aeabi_f2uiz
define i32 @test_fptoui_i32(half* %p) #0 {
  %a = load half, half* %p, align 2
  %r = fptoui half %a to i32
  ret i32 %r
}

; CHECK-ALL-LABEL: test_fptoui_i64:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-ALL: bl __aeabi_f2ulz
define i64 @test_fptoui_i64(half* %p) #0 {
  %a = load half, half* %p, align 2
  %r = fptoui half %a to i64
  ret i64 %r
}

; CHECK-ALL-LABEL: test_sitofp_i32:
; CHECK-VFP: vcvt.f32.s32
; CHECK-NOVFP: bl __aeabi_i2f
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_sitofp_i32(i32 %a, half* %p) #0 {
  %r = sitofp i32 %a to half
  store half %r, half* %p
  ret void
}

; CHECK-ALL-LABEL: test_uitofp_i32:
; CHECK-VFP: vcvt.f32.u32
; CHECK-NOVFP: bl __aeabi_ui2f
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_uitofp_i32(i32 %a, half* %p) #0 {
  %r = uitofp i32 %a to half
  store half %r, half* %p
  ret void
}

; CHECK-ALL-LABEL: test_sitofp_i64:
; CHECK-ALL: bl __aeabi_l2f
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_sitofp_i64(i64 %a, half* %p) #0 {
  %r = sitofp i64 %a to half
  store half %r, half* %p
  ret void
}

; CHECK-ALL-LABEL: test_uitofp_i64:
; CHECK-ALL: bl __aeabi_ul2f
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_uitofp_i64(i64 %a, half* %p) #0 {
  %r = uitofp i64 %a to half
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_fptrunc_float:
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_fptrunc_float:
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_fptrunc_float(float %f, half* %p) #0 {
  %a = fptrunc float %f to half
  store half %a, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_fptrunc_double:
; CHECK-FP16: bl __aeabi_d2h
; CHECK-LIBCALL-LABEL: test_fptrunc_double:
; CHECK-LIBCALL: bl __aeabi_d2h
define void @test_fptrunc_double(double %d, half* %p) #0 {
  %a = fptrunc double %d to half
  store half %a, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_fpextend_float:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL-LABEL: test_fpextend_float:
; CHECK-LIBCALL: bl __aeabi_h2f
define float @test_fpextend_float(half* %p) {
  %a = load half, half* %p, align 2
  %r = fpext half %a to float
  ret float %r
}

; CHECK-FP16-LABEL: test_fpextend_double:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL-LABEL: test_fpextend_double:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-VFP: vcvt.f64.f32
; CHECK-NOVFP: bl __aeabi_f2d
define double @test_fpextend_double(half* %p) {
  %a = load half, half* %p, align 2
  %r = fpext half %a to double
  ret double %r
}

; CHECK-ALL-LABEL: test_bitcast_halftoi16:
; CHECK-ALL-NEXT: .fnstart
; CHECK-ALL-NEXT: ldrh r0, [r0]
; CHECK-ALL-NEXT: bx lr
define i16 @test_bitcast_halftoi16(half* %p) #0 {
  %a = load half, half* %p, align 2
  %r = bitcast half %a to i16
  ret i16 %r
}

; CHECK-ALL-LABEL: test_bitcast_i16tohalf:
; CHECK-ALL-NEXT: .fnstart
; CHECK-ALL-NEXT: strh r0, [r1]
; CHECK-ALL-NEXT: bx lr
define void @test_bitcast_i16tohalf(i16 %a, half* %p) #0 {
  %r = bitcast i16 %a to half
  store half %r, half* %p
  ret void
}

declare half @llvm.sqrt.f16(half %a) #0
declare half @llvm.powi.f16(half %a, i32 %b) #0
declare half @llvm.sin.f16(half %a) #0
declare half @llvm.cos.f16(half %a) #0
declare half @llvm.pow.f16(half %a, half %b) #0
declare half @llvm.exp.f16(half %a) #0
declare half @llvm.exp2.f16(half %a) #0
declare half @llvm.log.f16(half %a) #0
declare half @llvm.log10.f16(half %a) #0
declare half @llvm.log2.f16(half %a) #0
declare half @llvm.fma.f16(half %a, half %b, half %c) #0
declare half @llvm.fabs.f16(half %a) #0
declare half @llvm.minnum.f16(half %a, half %b) #0
declare half @llvm.maxnum.f16(half %a, half %b) #0
declare half @llvm.copysign.f16(half %a, half %b) #0
declare half @llvm.floor.f16(half %a) #0
declare half @llvm.ceil.f16(half %a) #0
declare half @llvm.trunc.f16(half %a) #0
declare half @llvm.rint.f16(half %a) #0
declare half @llvm.nearbyint.f16(half %a) #0
declare half @llvm.round.f16(half %a) #0
declare half @llvm.fmuladd.f16(half %a, half %b, half %c) #0

; CHECK-ALL-LABEL: test_sqrt:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vsqrt.f32
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-VFP-LIBCALL: vsqrt.f32
; CHECK-NOVFP: bl sqrtf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_sqrt(half* %p) #0 {
  %a = load half, half* %p, align 2
  %r = call half @llvm.sqrt.f16(half %a)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_fpowi:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl __powisf2
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_fpowi:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __powisf2
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_fpowi(half* %p, i32 %b) #0 {
  %a = load half, half* %p, align 2
  %r = call half @llvm.powi.f16(half %a, i32 %b)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_sin:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl sinf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_sin:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl sinf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_sin(half* %p) #0 {
  %a = load half, half* %p, align 2
  %r = call half @llvm.sin.f16(half %a)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_cos:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl cosf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_cos:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl cosf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_cos(half* %p) #0 {
  %a = load half, half* %p, align 2
  %r = call half @llvm.cos.f16(half %a)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_pow:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl powf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_pow:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl powf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_pow(half* %p, half* %q) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %r = call half @llvm.pow.f16(half %a, half %b)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_cbrt:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl powf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_cbrt:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl powf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_cbrt(half* %p) #0 {
  %a = load half, half* %p, align 2
  %r = call half @llvm.pow.f16(half %a, half 0x3FD5540000000000)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_exp:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl expf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_exp:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl expf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_exp(half* %p) #0 {
  %a = load half, half* %p, align 2
  %r = call half @llvm.exp.f16(half %a)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_exp2:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl exp2f
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_exp2:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl exp2f
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_exp2(half* %p) #0 {
  %a = load half, half* %p, align 2
  %r = call half @llvm.exp2.f16(half %a)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_log:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl logf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_log:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl logf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_log(half* %p) #0 {
  %a = load half, half* %p, align 2
  %r = call half @llvm.log.f16(half %a)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_log10:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl log10f
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_log10:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl log10f
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_log10(half* %p) #0 {
  %a = load half, half* %p, align 2
  %r = call half @llvm.log10.f16(half %a)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_log2:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl log2f
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_log2:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl log2f
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_log2(half* %p) #0 {
  %a = load half, half* %p, align 2
  %r = call half @llvm.log2.f16(half %a)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_fma:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl fmaf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_fma:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl fmaf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_fma(half* %p, half* %q, half* %r) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %c = load half, half* %r, align 2
  %v = call half @llvm.fma.f16(half %a, half %b, half %c)
  store half %v, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_fabs:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vabs.f32
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_fabs:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bic
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_fabs(half* %p) {
  %a = load half, half* %p, align 2
  %r = call half @llvm.fabs.f16(half %a)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_minnum:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl fminf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_minnum:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl fminf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_minnum(half* %p, half* %q) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %r = call half @llvm.minnum.f16(half %a, half %b)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_maxnum:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl fmaxf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_maxnum:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl fmaxf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_maxnum(half* %p, half* %q) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %r = call half @llvm.maxnum.f16(half %a, half %b)
  store half %r, half* %p
  ret void
}

; CHECK-ALL-LABEL: test_minimum:
; CHECK-FP16: vmov.f32 s0, #1.000000e+00
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL-VFP: vmov.f32 s{{[0-9]+}}, #1.000000e+00
; CHECK-NOVFP: mov r{{[0-9]+}}, #1065353216
; CHECK-VFP: vcmp.f32
; CHECK-VFP: vmrs
; CHECK-VFP: vmovlt.f32
; CHECK-NOVFP: bl __aeabi_fcmpge
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_minimum(half* %p) #0 {
  %a = load half, half* %p, align 2
  %c = fcmp ult half %a, 1.0
  %r = select i1 %c, half %a, half 1.0
  store half %r, half* %p
  ret void
}

; CHECK-ALL-LABEL: test_maximum:
; CHECK-FP16: vmov.f32 s0, #1.000000e+00
; CHECK-FP16: vcvtb.f32.f16
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL-VFP: vmov.f32 s0, #1.000000e+00
; CHECK-NOVFP: mov r{{[0-9]+}}, #1065353216
; CHECK-VFP: vcmp.f32
; CHECK-VFP: vmrs
; CHECK-VFP: vmovhi.f32
; CHECK-NOVFP: bl __aeabi_fcmple
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_maximum(half* %p) #0 {
  %a = load half, half* %p, align 2
  %c = fcmp ugt half %a, 1.0
  %r = select i1 %c, half %a, half 1.0
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_copysign:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vbsl
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_copysign:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-VFP-LIBCALL: vbsl
; CHECK-NOVFP: and
; CHECK-NOVFP: bic
; CHECK-NOVFP: orr
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_copysign(half* %p, half* %q) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %r = call half @llvm.copysign.f16(half %a, half %b)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_floor:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl floorf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_floor:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl floorf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_floor(half* %p) {
  %a = load half, half* %p, align 2
  %r = call half @llvm.floor.f16(half %a)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_ceil:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl ceilf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_ceil:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl ceilf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_ceil(half* %p) {
  %a = load half, half* %p, align 2
  %r = call half @llvm.ceil.f16(half %a)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_trunc:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl truncf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_trunc:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl truncf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_trunc(half* %p) {
  %a = load half, half* %p, align 2
  %r = call half @llvm.trunc.f16(half %a)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_rint:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl rintf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_rint:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl rintf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_rint(half* %p) {
  %a = load half, half* %p, align 2
  %r = call half @llvm.rint.f16(half %a)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_nearbyint:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl nearbyintf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_nearbyint:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl nearbyintf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_nearbyint(half* %p) {
  %a = load half, half* %p, align 2
  %r = call half @llvm.nearbyint.f16(half %a)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_round:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: bl roundf
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_round:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl roundf
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_round(half* %p) {
  %a = load half, half* %p, align 2
  %r = call half @llvm.round.f16(half %a)
  store half %r, half* %p
  ret void
}

; CHECK-FP16-LABEL: test_fmuladd:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vcvtb.f32.f16
; CHECK-FP16: vmla.f32
; CHECK-FP16: vcvtb.f16.f32
; CHECK-LIBCALL-LABEL: test_fmuladd:
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-LIBCALL: bl __aeabi_h2f
; CHECK-VFP-LIBCALL: vmla.f32
; CHECK-NOVFP: bl __aeabi_fmul
; CHECK-LIBCALL: bl __aeabi_f2h
define void @test_fmuladd(half* %p, half* %q, half* %r) #0 {
  %a = load half, half* %p, align 2
  %b = load half, half* %q, align 2
  %c = load half, half* %r, align 2
  %v = call half @llvm.fmuladd.f16(half %a, half %b, half %c)
  store half %v, half* %p
  ret void
}

; f16 vectors are not legal in the backend.  Vector elements are not assigned
; to the register, but are stored in the stack instead.  Hence insertelement
; and extractelement have these extra loads and stores.

; CHECK-ALL-LABEL: test_insertelement:
; CHECK-ALL: sub sp, sp, #8

; CHECK-VFP:	and	
; CHECK-VFP:	mov	
; CHECK-VFP:	ldrd	
; CHECK-VFP:	orr	
; CHECK-VFP:	ldrh	
; CHECK-VFP:	stm	
; CHECK-VFP:	strh	
; CHECK-VFP:	ldm	
; CHECK-VFP:	stm	

; CHECK-NOVFP: ldrh
; CHECK-NOVFP: ldrh
; CHECK-NOVFP: ldrh
; CHECK-NOVFP: ldrh
; CHECK-NOVFP-DAG: strh
; CHECK-NOVFP-DAG: strh
; CHECK-NOVFP-DAG: mov
; CHECK-NOVFP-DAG: ldrh
; CHECK-NOVFP-DAG: orr
; CHECK-NOVFP-DAG: strh
; CHECK-NOVFP-DAG: strh
; CHECK-NOVFP-DAG: strh
; CHECK-NOVFP-DAG: ldrh
; CHECK-NOVFP-DAG: ldrh
; CHECK-NOVFP-DAG: ldrh
; CHECK-NOVFP-DAG: strh
; CHECK-NOVFP-DAG: strh
; CHECK-NOVFP-DAG: strh
; CHECK-NOVFP-DAG: strh

; CHECK-ALL: add sp, sp, #8
define void @test_insertelement(half* %p, <4 x half>* %q, i32 %i) #0 {
  %a = load half, half* %p, align 2
  %b = load <4 x half>, <4 x half>* %q, align 8
  %c = insertelement <4 x half> %b, half %a, i32 %i
  store <4 x half> %c, <4 x half>* %q
  ret void
}

; CHECK-ALL-LABEL: test_extractelement:
; CHECK-VFP: push {{{.*}}, lr}
; CHECK-VFP: sub sp, sp, #8
; CHECK-VFP: ldrd
; CHECK-VFP: mov
; CHECK-VFP: orr
; CHECK-VFP: ldrh
; CHECK-VFP: strh
; CHECK-VFP: add sp, sp, #8
; CHECK-VFP: pop {{{.*}}, pc}
; CHECK-NOVFP: ldrh
; CHECK-NOVFP: strh
; CHECK-NOVFP: ldrh
; CHECK-NOVFP: strh
; CHECK-NOVFP: ldrh
; CHECK-NOVFP: strh
; CHECK-NOVFP: ldrh
; CHECK-NOVFP: strh
; CHECK-NOVFP: ldrh
define void @test_extractelement(half* %p, <4 x half>* %q, i32 %i) #0 {
  %a = load <4 x half>, <4 x half>* %q, align 8
  %b = extractelement <4 x half> %a, i32 %i
  store half %b, half* %p
  ret void
}

; test struct operations

%struct.dummy = type { i32, half }

; CHECK-ALL-LABEL: test_insertvalue:
; CHECK-ALL-DAG: ldr
; CHECK-ALL-DAG: ldrh
; CHECK-ALL-DAG: strh
; CHECK-ALL-DAG: str
define void @test_insertvalue(%struct.dummy* %p, half* %q) {
  %a = load %struct.dummy, %struct.dummy* %p
  %b = load half, half* %q
  %c = insertvalue %struct.dummy %a, half %b, 1
  store %struct.dummy %c, %struct.dummy* %p
  ret void
}

; CHECK-ALL-LABEL: test_extractvalue:
; CHECK-ALL: .fnstart
; CHECK-ALL: ldrh
; CHECK-ALL: strh
define void @test_extractvalue(%struct.dummy* %p, half* %q) {
  %a = load %struct.dummy, %struct.dummy* %p
  %b = extractvalue %struct.dummy %a, 1
  store half %b, half* %q
  ret void
}

; CHECK-ALL-LABEL: test_struct_return:
; CHECK-FP16: vcvtb.f32.f16
; CHECK-VFP-LIBCALL: bl __aeabi_h2f
; CHECK-NOVFP-DAG: ldr
; CHECK-NOVFP-DAG: ldrh
define %struct.dummy @test_struct_return(%struct.dummy* %p) {
  %a = load %struct.dummy, %struct.dummy* %p
  ret %struct.dummy %a
}

; CHECK-ALL-LABEL: test_struct_arg:
; CHECK-ALL-NEXT: .fnstart
; CHECK-NOVFP-NEXT: mov r0, r1
; CHECK-ALL-NEXT: bx lr
define half @test_struct_arg(%struct.dummy %p) {
  %a = extractvalue %struct.dummy %p, 1
  ret half %a
}

; CHECK-LABEL: test_uitofp_i32_fadd:
; CHECK-VFP-DAG: vcvt.f32.u32
; CHECK-NOVFP-DAG: bl __aeabi_ui2f

; CHECK-FP16-DAG: vcvtb.f16.f32
; CHECK-FP16-DAG: vcvtb.f32.f16
; CHECK-LIBCALL-DAG: bl __aeabi_h2f
; CHECK-LIBCALL-DAG: bl __aeabi_h2f

; CHECK-VFP-DAG: vadd.f32
; CHECK-NOVFP-DAG: bl __aeabi_fadd

; CHECK-FP16-DAG: vcvtb.f16.f32
; CHECK-LIBCALL-DAG: bl __aeabi_f2h
define half @test_uitofp_i32_fadd(i32 %a, half %b) #0 {
  %c = uitofp i32 %a to half
  %r = fadd half %b, %c
  ret half %r
}

; CHECK-LABEL: test_sitofp_i32_fadd:
; CHECK-VFP-DAG: vcvt.f32.s32
; CHECK-NOVFP-DAG: bl __aeabi_i2f

; CHECK-FP16-DAG: vcvtb.f16.f32
; CHECK-FP16-DAG: vcvtb.f32.f16
; CHECK-LIBCALL-DAG: bl __aeabi_h2f
; CHECK-LIBCALL-DAG: bl __aeabi_h2f

; CHECK-VFP-DAG: vadd.f32
; CHECK-NOVFP-DAG: bl __aeabi_fadd

; CHECK-FP16-DAG: vcvtb.f16.f32
; CHECK-LIBCALL-DAG: bl __aeabi_f2h
define half @test_sitofp_i32_fadd(i32 %a, half %b) #0 {
  %c = sitofp i32 %a to half
  %r = fadd half %b, %c
  ret half %r
}

attributes #0 = { nounwind }
