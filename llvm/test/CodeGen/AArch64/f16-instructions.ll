; RUN: llc < %s -mtriple aarch64-unknown-unknown -aarch64-neon-syntax=apple -asm-verbose=false -disable-post-ra -disable-fp-elim | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: test_fadd:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fadd s0, s0, s1
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_fadd(half %a, half %b) #0 {
  %r = fadd half %a, %b
  ret half %r
}

; CHECK-LABEL: test_fsub:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fsub s0, s0, s1
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_fsub(half %a, half %b) #0 {
  %r = fsub half %a, %b
  ret half %r
}

; CHECK-LABEL: test_fmul:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fmul s0, s0, s1
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_fmul(half %a, half %b) #0 {
  %r = fmul half %a, %b
  ret half %r
}

; CHECK-LABEL: test_fdiv:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fdiv s0, s0, s1
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_fdiv(half %a, half %b) #0 {
  %r = fdiv half %a, %b
  ret half %r
}

; CHECK-LABEL: test_frem:
; CHECK-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-NEXT: mov  x29, sp
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: bl {{_?}}fmodf
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ldp x29, x30, [sp], #16
; CHECK-NEXT: ret
define half @test_frem(half %a, half %b) #0 {
  %r = frem half %a, %b
  ret half %r
}

; CHECK-LABEL: test_store:
; CHECK-NEXT: str  h0, [x0]
; CHECK-NEXT: ret
define void @test_store(half %a, half* %b) #0 {
  store half %a, half* %b
  ret void
}

; CHECK-LABEL: test_load:
; CHECK-NEXT: ldr  h0, [x0]
; CHECK-NEXT: ret
define half @test_load(half* %a) #0 {
  %r = load half, half* %a
  ret half %r
}


declare half @test_callee(half %a, half %b) #0

; CHECK-LABEL: test_call:
; CHECK-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-NEXT: mov  x29, sp
; CHECK-NEXT: bl {{_?}}test_callee
; CHECK-NEXT: ldp x29, x30, [sp], #16
; CHECK-NEXT: ret
define half @test_call(half %a, half %b) #0 {
  %r = call half @test_callee(half %a, half %b)
  ret half %r
}

; CHECK-LABEL: test_call_flipped:
; CHECK-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-NEXT: mov  x29, sp
; CHECK-NEXT: mov.16b  v2, v0
; CHECK-NEXT: mov.16b  v0, v1
; CHECK-NEXT: mov.16b  v1, v2
; CHECK-NEXT: bl {{_?}}test_callee
; CHECK-NEXT: ldp x29, x30, [sp], #16
; CHECK-NEXT: ret
define half @test_call_flipped(half %a, half %b) #0 {
  %r = call half @test_callee(half %b, half %a)
  ret half %r
}

; CHECK-LABEL: test_tailcall_flipped:
; CHECK-NEXT: mov.16b  v2, v0
; CHECK-NEXT: mov.16b  v0, v1
; CHECK-NEXT: mov.16b  v1, v2
; CHECK-NEXT: b {{_?}}test_callee
define half @test_tailcall_flipped(half %a, half %b) #0 {
  %r = tail call half @test_callee(half %b, half %a)
  ret half %r
}

; CHECK-LABEL: test_select:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: cmp  w0, #0
; CHECK-NEXT: fcsel s0, s0, s1, ne
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_select(half %a, half %b, i1 zeroext %c) #0 {
  %r = select i1 %c, half %a, half %b
  ret half %r
}

; CHECK-LABEL: test_select_cc:
; CHECK-DAG: fcvt s3, h3
; CHECK-DAG: fcvt s2, h2
; CHECK-DAG: fcvt s1, h1
; CHECK-DAG: fcvt s0, h0
; CHECK-DAG: fcmp s2, s3
; CHECK-DAG: cset [[CC:w[0-9]+]], ne
; CHECK-DAG: cmp [[CC]], #0
; CHECK-NEXT: fcsel s0, s0, s1, ne
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_select_cc(half %a, half %b, half %c, half %d) #0 {
  %cc = fcmp une half %c, %d
  %r = select i1 %cc, half %a, half %b
  ret half %r
}

; CHECK-LABEL: test_select_cc_f32_f16:
; CHECK-DAG:   fcvt s2, h2
; CHECK-DAG:   fcvt s3, h3
; CHECK-NEXT:  fcmp s2, s3
; CHECK-NEXT:  fcsel s0, s0, s1, ne
; CHECK-NEXT:  ret
define float @test_select_cc_f32_f16(float %a, float %b, half %c, half %d) #0 {
  %cc = fcmp une half %c, %d
  %r = select i1 %cc, float %a, float %b
  ret float %r
}

; CHECK-LABEL: test_select_cc_f16_f32:
; CHECK-DAG:  fcvt s0, h0
; CHECK-DAG:  fcvt s1, h1
; CHECK-DAG:  fcmp s2, s3
; CHECK-DAG:  cset w8, ne
; CHECK-NEXT: cmp w8, #0
; CHECK-NEXT: fcsel s0, s0, s1, ne
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_select_cc_f16_f32(half %a, half %b, float %c, float %d) #0 {
  %cc = fcmp une float %c, %d
  %r = select i1 %cc, half %a, half %b
  ret half %r
}

; CHECK-LABEL: test_fcmp_une:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcmp s0, s1
; CHECK-NEXT: cset  w0, ne
; CHECK-NEXT: ret
define i1 @test_fcmp_une(half %a, half %b) #0 {
  %r = fcmp une half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_ueq:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcmp s0, s1
; CHECK-NEXT: cset [[TRUE:w[0-9]+]], eq
; CHECK-NEXT: csinc w0, [[TRUE]], wzr, vc
; CHECK-NEXT: ret
define i1 @test_fcmp_ueq(half %a, half %b) #0 {
  %r = fcmp ueq half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_ugt:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcmp s0, s1
; CHECK-NEXT: cset  w0, hi
; CHECK-NEXT: ret
define i1 @test_fcmp_ugt(half %a, half %b) #0 {
  %r = fcmp ugt half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_uge:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcmp s0, s1
; CHECK-NEXT: cset  w0, pl
; CHECK-NEXT: ret
define i1 @test_fcmp_uge(half %a, half %b) #0 {
  %r = fcmp uge half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_ult:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcmp s0, s1
; CHECK-NEXT: cset  w0, lt
; CHECK-NEXT: ret
define i1 @test_fcmp_ult(half %a, half %b) #0 {
  %r = fcmp ult half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_ule:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcmp s0, s1
; CHECK-NEXT: cset  w0, le
; CHECK-NEXT: ret
define i1 @test_fcmp_ule(half %a, half %b) #0 {
  %r = fcmp ule half %a, %b
  ret i1 %r
}


; CHECK-LABEL: test_fcmp_uno:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcmp s0, s1
; CHECK-NEXT: cset  w0, vs
; CHECK-NEXT: ret
define i1 @test_fcmp_uno(half %a, half %b) #0 {
  %r = fcmp uno half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_one:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcmp s0, s1
; CHECK-NEXT: cset [[TRUE:w[0-9]+]], mi
; CHECK-NEXT: csinc w0, [[TRUE]], wzr, le
; CHECK-NEXT: ret
define i1 @test_fcmp_one(half %a, half %b) #0 {
  %r = fcmp one half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_oeq:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcmp s0, s1
; CHECK-NEXT: cset  w0, eq
; CHECK-NEXT: ret
define i1 @test_fcmp_oeq(half %a, half %b) #0 {
  %r = fcmp oeq half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_ogt:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcmp s0, s1
; CHECK-NEXT: cset  w0, gt
; CHECK-NEXT: ret
define i1 @test_fcmp_ogt(half %a, half %b) #0 {
  %r = fcmp ogt half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_oge:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcmp s0, s1
; CHECK-NEXT: cset  w0, ge
; CHECK-NEXT: ret
define i1 @test_fcmp_oge(half %a, half %b) #0 {
  %r = fcmp oge half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_olt:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcmp s0, s1
; CHECK-NEXT: cset  w0, mi
; CHECK-NEXT: ret
define i1 @test_fcmp_olt(half %a, half %b) #0 {
  %r = fcmp olt half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_ole:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcmp s0, s1
; CHECK-NEXT: cset  w0, ls
; CHECK-NEXT: ret
define i1 @test_fcmp_ole(half %a, half %b) #0 {
  %r = fcmp ole half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_fcmp_ord:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcmp s0, s1
; CHECK-NEXT: cset  w0, vc
; CHECK-NEXT: ret
define i1 @test_fcmp_ord(half %a, half %b) #0 {
  %r = fcmp ord half %a, %b
  ret i1 %r
}

; CHECK-LABEL: test_br_cc:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcmp s0, s1
; CHECK-NEXT: b.mi [[BRCC_ELSE:.?LBB[0-9_]+]]
; CHECK-NEXT: str  wzr, [x0]
; CHECK-NEXT: ret
; CHECK-NEXT: [[BRCC_ELSE]]:
; CHECK-NEXT: str  wzr, [x1]
; CHECK-NEXT: ret
define void @test_br_cc(half %a, half %b, i32* %p1, i32* %p2) #0 {
  %c = fcmp uge half %a, %b
  br i1 %c, label %then, label %else
then:
  store i32 0, i32* %p1
  ret void
else:
  store i32 0, i32* %p2
  ret void
}

; CHECK-LABEL: test_phi:
; CHECK: mov  x[[PTR:[0-9]+]], x0
; CHECK: ldr  h[[AB:[0-9]+]], [x0]
; CHECK: [[LOOP:LBB[0-9_]+]]:
; CHECK: mov.16b  v[[R:[0-9]+]], v[[AB]]
; CHECK: ldr  h[[AB]], [x[[PTR]]]
; CHECK: mov  x0, x[[PTR]]
; CHECK: bl {{_?}}test_dummy
; CHECK: mov.16b  v0, v[[R]]
; CHECK: ret
define half @test_phi(half* %p1) #0 {
entry:
  %a = load half, half* %p1
  br label %loop
loop:
  %r = phi half [%a, %entry], [%b, %loop]
  %b = load half, half* %p1
  %c = call i1 @test_dummy(half* %p1)
  br i1 %c, label %loop, label %return
return:
  ret half %r
}
declare i1 @test_dummy(half* %p1) #0

; CHECK-LABEL: test_fptosi_i32:
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcvtzs w0, s0
; CHECK-NEXT: ret
define i32 @test_fptosi_i32(half %a) #0 {
  %r = fptosi half %a to i32
  ret i32 %r
}

; CHECK-LABEL: test_fptosi_i64:
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcvtzs x0, s0
; CHECK-NEXT: ret
define i64 @test_fptosi_i64(half %a) #0 {
  %r = fptosi half %a to i64
  ret i64 %r
}

; CHECK-LABEL: test_fptoui_i32:
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcvtzu w0, s0
; CHECK-NEXT: ret
define i32 @test_fptoui_i32(half %a) #0 {
  %r = fptoui half %a to i32
  ret i32 %r
}

; CHECK-LABEL: test_fptoui_i64:
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcvtzu x0, s0
; CHECK-NEXT: ret
define i64 @test_fptoui_i64(half %a) #0 {
  %r = fptoui half %a to i64
  ret i64 %r
}

; CHECK-LABEL: test_uitofp_i32:
; CHECK-NEXT: ucvtf s0, w0
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_uitofp_i32(i32 %a) #0 {
  %r = uitofp i32 %a to half
  ret half %r
}

; CHECK-LABEL: test_uitofp_i64:
; CHECK-NEXT: ucvtf s0, x0
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_uitofp_i64(i64 %a) #0 {
  %r = uitofp i64 %a to half
  ret half %r
}

; CHECK-LABEL: test_sitofp_i32:
; CHECK-NEXT: scvtf s0, w0
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_sitofp_i32(i32 %a) #0 {
  %r = sitofp i32 %a to half
  ret half %r
}

; CHECK-LABEL: test_sitofp_i64:
; CHECK-NEXT: scvtf s0, x0
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_sitofp_i64(i64 %a) #0 {
  %r = sitofp i64 %a to half
  ret half %r
}

; CHECK-LABEL: test_uitofp_i32_fadd:
; CHECK-NEXT: ucvtf s1, w0
; CHECK-NEXT: fcvt h1, s1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fadd s0, s0, s1
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_uitofp_i32_fadd(i32 %a, half %b) #0 {
  %c = uitofp i32 %a to half
  %r = fadd half %b, %c
  ret half %r
}

; CHECK-LABEL: test_sitofp_i32_fadd:
; CHECK-NEXT: scvtf s1, w0
; CHECK-NEXT: fcvt h1, s1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fadd s0, s0, s1
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_sitofp_i32_fadd(i32 %a, half %b) #0 {
  %c = sitofp i32 %a to half
  %r = fadd half %b, %c
  ret half %r
}

; CHECK-LABEL: test_fptrunc_float:
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret

define half @test_fptrunc_float(float %a) #0 {
  %r = fptrunc float %a to half
  ret half %r
}

; CHECK-LABEL: test_fptrunc_double:
; CHECK-NEXT: fcvt h0, d0
; CHECK-NEXT: ret
define half @test_fptrunc_double(double %a) #0 {
  %r = fptrunc double %a to half
  ret half %r
}

; CHECK-LABEL: test_fpext_float:
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: ret
define float @test_fpext_float(half %a) #0 {
  %r = fpext half %a to float
  ret float %r
}

; CHECK-LABEL: test_fpext_double:
; CHECK-NEXT: fcvt d0, h0
; CHECK-NEXT: ret
define double @test_fpext_double(half %a) #0 {
  %r = fpext half %a to double
  ret double %r
}


; CHECK-LABEL: test_bitcast_halftoi16:
; CHECK-NEXT: fmov w0, s0
; CHECK-NEXT: ret
define i16 @test_bitcast_halftoi16(half %a) #0 {
  %r = bitcast half %a to i16
  ret i16 %r
}

; CHECK-LABEL: test_bitcast_i16tohalf:
; CHECK-NEXT: fmov s0, w0
; CHECK-NEXT: ret
define half @test_bitcast_i16tohalf(i16 %a) #0 {
  %r = bitcast i16 %a to half
  ret half %r
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

; CHECK-LABEL: test_sqrt:
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fsqrt s0, s0
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_sqrt(half %a) #0 {
  %r = call half @llvm.sqrt.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_powi:
; CHECK-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-NEXT: mov  x29, sp
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: bl {{_?}}__powisf2
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ldp x29, x30, [sp], #16
; CHECK-NEXT: ret
define half @test_powi(half %a, i32 %b) #0 {
  %r = call half @llvm.powi.f16(half %a, i32 %b)
  ret half %r
}

; CHECK-LABEL: test_sin:
; CHECK-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-NEXT: mov  x29, sp
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: bl {{_?}}sinf
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ldp x29, x30, [sp], #16
; CHECK-NEXT: ret
define half @test_sin(half %a) #0 {
  %r = call half @llvm.sin.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_cos:
; CHECK-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-NEXT: mov  x29, sp
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: bl {{_?}}cosf
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ldp x29, x30, [sp], #16
; CHECK-NEXT: ret
define half @test_cos(half %a) #0 {
  %r = call half @llvm.cos.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_pow:
; CHECK-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-NEXT: mov  x29, sp
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: bl {{_?}}powf
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ldp x29, x30, [sp], #16
; CHECK-NEXT: ret
define half @test_pow(half %a, half %b) #0 {
  %r = call half @llvm.pow.f16(half %a, half %b)
  ret half %r
}

; CHECK-LABEL: test_exp:
; CHECK-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-NEXT: mov  x29, sp
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: bl {{_?}}expf
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ldp x29, x30, [sp], #16
; CHECK-NEXT: ret
define half @test_exp(half %a) #0 {
  %r = call half @llvm.exp.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_exp2:
; CHECK-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-NEXT: mov  x29, sp
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: bl {{_?}}exp2f
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ldp x29, x30, [sp], #16
; CHECK-NEXT: ret
define half @test_exp2(half %a) #0 {
  %r = call half @llvm.exp2.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_log:
; CHECK-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-NEXT: mov  x29, sp
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: bl {{_?}}logf
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ldp x29, x30, [sp], #16
; CHECK-NEXT: ret
define half @test_log(half %a) #0 {
  %r = call half @llvm.log.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_log10:
; CHECK-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-NEXT: mov  x29, sp
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: bl {{_?}}log10f
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ldp x29, x30, [sp], #16
; CHECK-NEXT: ret
define half @test_log10(half %a) #0 {
  %r = call half @llvm.log10.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_log2:
; CHECK-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-NEXT: mov  x29, sp
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: bl {{_?}}log2f
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ldp x29, x30, [sp], #16
; CHECK-NEXT: ret
define half @test_log2(half %a) #0 {
  %r = call half @llvm.log2.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_fma:
; CHECK-NEXT: fcvt s2, h2
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fmadd s0, s0, s1, s2
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_fma(half %a, half %b, half %c) #0 {
  %r = call half @llvm.fma.f16(half %a, half %b, half %c)
  ret half %r
}

; CHECK-LABEL: test_fabs:
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fabs s0, s0
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_fabs(half %a) #0 {
  %r = call half @llvm.fabs.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_minnum:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fminnm s0, s0, s1
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_minnum(half %a, half %b) #0 {
  %r = call half @llvm.minnum.f16(half %a, half %b)
  ret half %r
}

; CHECK-LABEL: test_maxnum:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fmaxnm s0, s0, s1
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_maxnum(half %a, half %b) #0 {
  %r = call half @llvm.maxnum.f16(half %a, half %b)
  ret half %r
}

; CHECK-LABEL: test_copysign:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: movi.4s v2, #128, lsl #24
; CHECK-NEXT: bit.16b v0, v1, v2
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_copysign(half %a, half %b) #0 {
  %r = call half @llvm.copysign.f16(half %a, half %b)
  ret half %r
}

; CHECK-LABEL: test_copysign_f32:
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: movi.4s v2, #128, lsl #24
; CHECK-NEXT: bit.16b v0, v1, v2
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_copysign_f32(half %a, float %b) #0 {
  %tb = fptrunc float %b to half
  %r = call half @llvm.copysign.f16(half %a, half %tb)
  ret half %r
}

; CHECK-LABEL: test_copysign_f64:
; CHECK-NEXT: fcvt s1, d1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: movi.4s v2, #128, lsl #24
; CHECK-NEXT: bit.16b v0, v1, v2
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_copysign_f64(half %a, double %b) #0 {
  %tb = fptrunc double %b to half
  %r = call half @llvm.copysign.f16(half %a, half %tb)
  ret half %r
}

; Check that the FP promotion will use a truncating FP_ROUND, so we can fold
; away the (fpext (fp_round <result>)) here.

; CHECK-LABEL: test_copysign_extended:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: movi.4s v2, #128, lsl #24
; CHECK-NEXT: bit.16b v0, v1, v2
; CHECK-NEXT: ret
define float @test_copysign_extended(half %a, half %b) #0 {
  %r = call half @llvm.copysign.f16(half %a, half %b)
  %xr = fpext half %r to float
  ret float %xr
}

; CHECK-LABEL: test_floor:
; CHECK-NEXT: fcvt [[FLOAT32:s[0-9]+]], h0
; CHECK-NEXT: frintm [[INT32:s[0-9]+]], [[FLOAT32]]
; CHECK-NEXT: fcvt h0, [[INT32]]
; CHECK-NEXT: ret
define half @test_floor(half %a) #0 {
  %r = call half @llvm.floor.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_ceil:
; CHECK-NEXT: fcvt [[FLOAT32:s[0-9]+]], h0
; CHECK-NEXT: frintp [[INT32:s[0-9]+]], [[FLOAT32]]
; CHECK-NEXT: fcvt h0, [[INT32]]
; CHECK-NEXT: ret
define half @test_ceil(half %a) #0 {
  %r = call half @llvm.ceil.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_trunc:
; CHECK-NEXT: fcvt [[FLOAT32:s[0-9]+]], h0
; CHECK-NEXT: frintz [[INT32:s[0-9]+]], [[FLOAT32]]
; CHECK-NEXT: fcvt h0, [[INT32]]
; CHECK-NEXT: ret
define half @test_trunc(half %a) #0 {
  %r = call half @llvm.trunc.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_rint:
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: frintx s0, s0
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_rint(half %a) #0 {
  %r = call half @llvm.rint.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_nearbyint:
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: frinti s0, s0
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_nearbyint(half %a) #0 {
  %r = call half @llvm.nearbyint.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_round:
; CHECK-NEXT: fcvt [[FLOAT32:s[0-9]+]], h0
; CHECK-NEXT: frinta [[INT32:s[0-9]+]], [[FLOAT32]]
; CHECK-NEXT: fcvt h0, [[INT32]]
; CHECK-NEXT: ret
define half @test_round(half %a) #0 {
  %r = call half @llvm.round.f16(half %a)
  ret half %r
}

; CHECK-LABEL: test_fmuladd:
; CHECK-NEXT: fcvt s1, h1
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fmul s0, s0, s1
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: fcvt s0, h0
; CHECK-NEXT: fcvt s1, h2
; CHECK-NEXT: fadd s0, s0, s1
; CHECK-NEXT: fcvt h0, s0
; CHECK-NEXT: ret
define half @test_fmuladd(half %a, half %b, half %c) #0 {
  %r = call half @llvm.fmuladd.f16(half %a, half %b, half %c)
  ret half %r
}

attributes #0 = { nounwind }
