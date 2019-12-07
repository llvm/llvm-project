; RUN: llc < %s -mtriple aarch64-unknown-unknown -aarch64-neon-syntax=apple -asm-verbose=false -disable-post-ra -frame-pointer=non-leaf | FileCheck %s --check-prefix=CHECK-CVT --check-prefix=CHECK-COMMON
; RUN: llc < %s -mtriple aarch64-unknown-unknown -mattr=+fullfp16 -aarch64-neon-syntax=apple -asm-verbose=false -disable-post-ra -frame-pointer=non-leaf | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-FP16

; RUN: llc < %s -mtriple aarch64-unknown-unknown -aarch64-neon-syntax=apple \
; RUN: -asm-verbose=false -disable-post-ra -frame-pointer=non-leaf -global-isel \
; RUN: -global-isel-abort=2 -pass-remarks-missed=gisel-* 2>&1 | FileCheck %s \
; RUN: --check-prefixes=FALLBACK,GISEL-CVT,GISEL

; RUN: llc < %s -mtriple aarch64-unknown-unknown -mattr=+fullfp16 \
; RUN: -aarch64-neon-syntax=apple -asm-verbose=false -disable-post-ra \
; RUN: -frame-pointer=non-leaf -global-isel -global-isel-abort=2 \
; RUN: -pass-remarks-missed=gisel-* 2>&1 | FileCheck %s \
; RUN: --check-prefixes=FALLBACK-FP16,GISEL-FP16,GISEL

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; CHECK-CVT-LABEL: test_fadd:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fadd s0, s0, s1
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fadd:
; CHECK-FP16-NEXT:  fadd h0, h0, h1
; CHECK-FP16-NEXT:  ret

define half @test_fadd(half %a, half %b) #0 {
  %r = fadd half %a, %b
  ret half %r
}

; CHECK-CVT-LABEL: test_fsub:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fsub s0, s0, s1
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fsub:
; CHECK-FP16-NEXT: fsub h0, h0, h1
; CHECK-FP16-NEXT: ret

define half @test_fsub(half %a, half %b) #0 {
  %r = fsub half %a, %b
  ret half %r
}

; CHECK-CVT-LABEL: test_fmul:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fmul s0, s0, s1
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fmul:
; CHECK-FP16-NEXT: fmul h0, h0, h1
; CHECK-FP16-NEXT: ret

define half @test_fmul(half %a, half %b) #0 {
  %r = fmul half %a, %b
  ret half %r
}

; CHECK-CVT-LABEL: test_fdiv:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fdiv s0, s0, s1
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fdiv:
; CHECK-FP16-NEXT: fdiv	h0, h0, h1
; CHECK-FP16-NEXT: ret

define half @test_fdiv(half %a, half %b) #0 {
  %r = fdiv half %a, %b
  ret half %r
}

; CHECK-COMMON-LABEL: test_frem:
; CHECK-COMMON-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-COMMON-NEXT: mov  x29, sp
; CHECK-COMMON-NEXT: fcvt s0, h0
; CHECK-COMMON-NEXT: fcvt s1, h1
; CHECK-COMMON-NEXT: bl {{_?}}fmodf
; CHECK-COMMON-NEXT: fcvt h0, s0
; CHECK-COMMON-NEXT: ldp x29, x30, [sp], #16
; CHECK-COMMON-NEXT: ret
define half @test_frem(half %a, half %b) #0 {
  %r = frem half %a, %b
  ret half %r
}

; CHECK-COMMON-LABEL: test_store:
; CHECK-COMMON-NEXT: str  h0, [x0]
; CHECK-COMMON-NEXT: ret
define void @test_store(half %a, half* %b) #0 {
  store half %a, half* %b
  ret void
}

; CHECK-COMMON-LABEL: test_load:
; CHECK-COMMON-NEXT: ldr  h0, [x0]
; CHECK-COMMON-NEXT: ret
define half @test_load(half* %a) #0 {
  %r = load half, half* %a
  ret half %r
}

declare half @test_callee(half %a, half %b) #0

; CHECK-COMMON-LABEL: test_call:
; CHECK-COMMON-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-COMMON-NEXT: mov  x29, sp
; CHECK-COMMON-NEXT: bl {{_?}}test_callee
; CHECK-COMMON-NEXT: ldp x29, x30, [sp], #16
; CHECK-COMMON-NEXT: ret
define half @test_call(half %a, half %b) #0 {
  %r = call half @test_callee(half %a, half %b)
  ret half %r
}

; CHECK-COMMON-LABEL: test_call_flipped:
; CHECK-COMMON-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-COMMON-NEXT: mov  x29, sp
; CHECK-COMMON-NEXT: mov.16b  v2, v0
; CHECK-COMMON-NEXT: mov.16b  v0, v1
; CHECK-COMMON-NEXT: mov.16b  v1, v2
; CHECK-COMMON-NEXT: bl {{_?}}test_callee
; CHECK-COMMON-NEXT: ldp x29, x30, [sp], #16
; CHECK-COMMON-NEXT: ret
define half @test_call_flipped(half %a, half %b) #0 {
  %r = call half @test_callee(half %b, half %a)
  ret half %r
}

; CHECK-COMMON-LABEL: test_tailcall_flipped:
; CHECK-COMMON-NEXT: mov.16b  v2, v0
; CHECK-COMMON-NEXT: mov.16b  v0, v1
; CHECK-COMMON-NEXT: mov.16b  v1, v2
; CHECK-COMMON-NEXT: b {{_?}}test_callee
define half @test_tailcall_flipped(half %a, half %b) #0 {
  %r = tail call half @test_callee(half %b, half %a)
  ret half %r
}

; CHECK-CVT-LABEL: test_select:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: cmp  w0, #0
; CHECK-CVT-NEXT: fcsel s0, s0, s1, ne
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_select:
; CHECK-FP16-NEXT: cmp w0, #0
; CHECK-FP16-NEXT: fcsel h0, h0, h1, ne
; CHECK-FP16-NEXT: ret

define half @test_select(half %a, half %b, i1 zeroext %c) #0 {
  %r = select i1 %c, half %a, half %b
  ret half %r
}

; CHECK-CVT-LABEL: test_select_cc:
; CHECK-CVT-DAG: fcvt s3, h3
; CHECK-CVT-DAG: fcvt s2, h2
; CHECK-CVT-DAG: fcvt s1, h1
; CHECK-CVT-DAG: fcvt s0, h0
; CHECK-CVT-DAG: fcmp s2, s3
; CHECK-CVT-DAG: cset [[CC:w[0-9]+]], ne
; CHECK-CVT-DAG: cmp [[CC]], #0
; CHECK-CVT-NEXT: fcsel s0, s0, s1, ne
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_select_cc:
; CHECK-FP16-NEXT: fcmp h2, h3
; CHECK-FP16-NEXT: fcsel h0, h0, h1, ne
; CHECK-FP16-NEXT: ret

define half @test_select_cc(half %a, half %b, half %c, half %d) #0 {
  %cc = fcmp une half %c, %d
  %r = select i1 %cc, half %a, half %b
  ret half %r
}

; CHECK-CVT-LABEL: test_select_cc_f32_f16:
; CHECK-CVT-DAG:   fcvt s2, h2
; CHECK-CVT-DAG:   fcvt s3, h3
; CHECK-CVT-NEXT:  fcmp s2, s3
; CHECK-CVT-NEXT:  fcsel s0, s0, s1, ne
; CHECK-CVT-NEXT:  ret

; CHECK-FP16-LABEL: test_select_cc_f32_f16:
; CHECK-FP16-NEXT: fcmp	h2, h3
; CHECK-FP16-NEXT: fcsel	s0, s0, s1, ne
; CHECK-FP16-NEXT: ret

define float @test_select_cc_f32_f16(float %a, float %b, half %c, half %d) #0 {
  %cc = fcmp une half %c, %d
  %r = select i1 %cc, float %a, float %b
  ret float %r
}

; CHECK-CVT-LABEL: test_select_cc_f16_f32:
; CHECK-CVT-DAG:  fcvt s0, h0
; CHECK-CVT-DAG:  fcvt s1, h1
; CHECK-CVT-DAG:  fcmp s2, s3
; CHECK-CVT-DAG:  cset w8, ne
; CHECK-CVT-NEXT: cmp w8, #0
; CHECK-CVT-NEXT: fcsel s0, s0, s1, ne
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_select_cc_f16_f32:
; CHECK-FP16-NEXT: fcmp	s2, s3
; CHECK-FP16-NEXT: fcsel h0, h0, h1, ne
; CHECK-FP16-NEXT: ret

define half @test_select_cc_f16_f32(half %a, half %b, float %c, float %d) #0 {
  %cc = fcmp une float %c, %d
  %r = select i1 %cc, half %a, half %b
  ret half %r
}

; CHECK-CVT-LABEL: test_fcmp_une:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcmp s0, s1
; CHECK-CVT-NEXT: cset  w0, ne
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fcmp_une:
; CHECK-FP16-NEXT: fcmp	h0, h1
; CHECK-FP16-NEXT: cset w0, ne
; CHECK-FP16-NEXT: ret

define i1 @test_fcmp_une(half %a, half %b) #0 {
  %r = fcmp une half %a, %b
  ret i1 %r
}

; CHECK-CVT-LABEL: test_fcmp_ueq:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcmp s0, s1
; CHECK-CVT-NEXT: cset [[TRUE:w[0-9]+]], eq
; CHECK-CVT-NEXT: csinc w0, [[TRUE]], wzr, vc
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fcmp_ueq:
; CHECK-FP16-NEXT: fcmp	h0, h1
; CHECK-FP16-NEXT: cset [[TRUE:w[0-9]+]], eq
; CHECK-FP16-NEXT: csinc w0, [[TRUE]], wzr, vc
; CHECK-FP16-NEXT: ret

define i1 @test_fcmp_ueq(half %a, half %b) #0 {
  %r = fcmp ueq half %a, %b
  ret i1 %r
}

; CHECK-CVT-LABEL: test_fcmp_ugt:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcmp s0, s1
; CHECK-CVT-NEXT: cset  w0, hi
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fcmp_ugt:
; CHECK-FP16-NEXT: fcmp h0, h1
; CHECK-FP16-NEXT: cset  w0, hi
; CHECK-FP16-NEXT: ret

define i1 @test_fcmp_ugt(half %a, half %b) #0 {
  %r = fcmp ugt half %a, %b
  ret i1 %r
}

; CHECK-CVT-LABEL: test_fcmp_uge:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcmp s0, s1
; CHECK-CVT-NEXT: cset  w0, pl
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fcmp_uge:
; CHECK-FP16-NEXT: fcmp h0, h1
; CHECK-FP16-NEXT: cset  w0, pl
; CHECK-FP16-NEXT: ret

define i1 @test_fcmp_uge(half %a, half %b) #0 {
  %r = fcmp uge half %a, %b
  ret i1 %r
}

; CHECK-CVT-LABEL: test_fcmp_ult:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcmp s0, s1
; CHECK-CVT-NEXT: cset  w0, lt
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fcmp_ult:
; CHECK-FP16-NEXT: fcmp h0, h1
; CHECK-FP16-NEXT: cset  w0, lt
; CHECK-FP16-NEXT: ret

define i1 @test_fcmp_ult(half %a, half %b) #0 {
  %r = fcmp ult half %a, %b
  ret i1 %r
}

; CHECK-CVT-LABEL: test_fcmp_ule:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcmp s0, s1
; CHECK-CVT-NEXT: cset  w0, le
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fcmp_ule:
; CHECK-FP16-NEXT: fcmp h0, h1
; CHECK-FP16-NEXT: cset  w0, le
; CHECK-FP16-NEXT: ret

define i1 @test_fcmp_ule(half %a, half %b) #0 {
  %r = fcmp ule half %a, %b
  ret i1 %r
}

; CHECK-CVT-LABEL: test_fcmp_uno:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcmp s0, s1
; CHECK-CVT-NEXT: cset  w0, vs
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fcmp_uno:
; CHECK-FP16-NEXT: fcmp h0, h1
; CHECK-FP16-NEXT: cset  w0, vs
; CHECK-FP16-NEXT: ret

define i1 @test_fcmp_uno(half %a, half %b) #0 {
  %r = fcmp uno half %a, %b
  ret i1 %r
}

; CHECK-CVT-LABEL: test_fcmp_one:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcmp s0, s1
; CHECK-CVT-NEXT: cset [[TRUE:w[0-9]+]], mi
; CHECK-CVT-NEXT: csinc w0, [[TRUE]], wzr, le
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fcmp_one:
; CHECK-FP16-NEXT: fcmp h0, h1
; CHECK-FP16-NEXT: cset [[TRUE:w[0-9]+]], mi
; CHECK-FP16-NEXT: csinc w0, [[TRUE]], wzr, le
; CHECK-FP16-NEXT: ret

define i1 @test_fcmp_one(half %a, half %b) #0 {
  %r = fcmp one half %a, %b
  ret i1 %r
}

; CHECK-CVT-LABEL: test_fcmp_oeq:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcmp s0, s1
; CHECK-CVT-NEXT: cset  w0, eq
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fcmp_oeq:
; CHECK-FP16-NEXT: fcmp h0, h1
; CHECK-FP16-NEXT: cset  w0, eq
; CHECK-FP16-NEXT: ret

define i1 @test_fcmp_oeq(half %a, half %b) #0 {
  %r = fcmp oeq half %a, %b
  ret i1 %r
}

; CHECK-CVT-LABEL: test_fcmp_ogt:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcmp s0, s1
; CHECK-CVT-NEXT: cset  w0, gt
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fcmp_ogt:
; CHECK-FP16-NEXT: fcmp h0, h1
; CHECK-FP16-NEXT: cset  w0, gt
; CHECK-FP16-NEXT: ret

define i1 @test_fcmp_ogt(half %a, half %b) #0 {
  %r = fcmp ogt half %a, %b
  ret i1 %r
}

; CHECK-CVT-LABEL: test_fcmp_oge:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcmp s0, s1
; CHECK-CVT-NEXT: cset  w0, ge
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fcmp_oge:
; CHECK-FP16-NEXT: fcmp h0, h1
; CHECK-FP16-NEXT: cset  w0, ge
; CHECK-FP16-NEXT: ret

define i1 @test_fcmp_oge(half %a, half %b) #0 {
  %r = fcmp oge half %a, %b
  ret i1 %r
}

; CHECK-CVT-LABEL: test_fcmp_olt:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcmp s0, s1
; CHECK-CVT-NEXT: cset  w0, mi
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fcmp_olt:
; CHECK-FP16-NEXT: fcmp h0, h1
; CHECK-FP16-NEXT: cset  w0, mi
; CHECK-FP16-NEXT: ret

define i1 @test_fcmp_olt(half %a, half %b) #0 {
  %r = fcmp olt half %a, %b
  ret i1 %r
}

; CHECK-CVT-LABEL: test_fcmp_ole:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcmp s0, s1
; CHECK-CVT-NEXT: cset  w0, ls
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fcmp_ole:
; CHECK-FP16-NEXT: fcmp h0, h1
; CHECK-FP16-NEXT: cset  w0, ls
; CHECK-FP16-NEXT: ret

define i1 @test_fcmp_ole(half %a, half %b) #0 {
  %r = fcmp ole half %a, %b
  ret i1 %r
}

; CHECK-CVT-LABEL: test_fcmp_ord:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcmp s0, s1
; CHECK-CVT-NEXT: cset  w0, vc
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fcmp_ord:
; CHECK-FP16-NEXT: fcmp h0, h1
; CHECK-FP16-NEXT: cset  w0, vc
; CHECK-FP16-NEXT: ret

define i1 @test_fcmp_ord(half %a, half %b) #0 {
  %r = fcmp ord half %a, %b
  ret i1 %r
}

; CHECK-COMMON-LABEL: test_fccmp:
; CHECK-CVT:      fcvt  s0, h0
; CHECK-CVT-NEXT: fmov  s1, #8.00000000
; CHECK-CVT-NEXT: fmov  s2, #5.00000000
; CHECK-CVT-NEXT: fcmp  s0, s1
; CHECK-CVT-NEXT: cset  w8, gt
; CHECK-CVT-NEXT: fcmp  s0, s2
; CHECK-CVT-NEXT: cset  w9, mi
; CHECK-CVT-NEXT: tst   w8, w9
; CHECK-CVT-NEXT: fcsel s0, s0, s2, ne
; CHECK-CVT-NEXT: fcvt  h0, s0
; CHECK-CVT-NEXT: str   h0, [x0]
; CHECK-CVT-NEXT: ret
; CHECK-FP16:      fmov  h1, #5.00000000
; CHECK-FP16-NEXT: fcmp  h0, h1
; CHECK-FP16-NEXT: fmov  h2, #8.00000000
; CHECK-FP16-NEXT: fccmp h0, h2, #4, mi
; CHECK-FP16-NEXT: fcsel h0, h0, h1, gt
; CHECK-FP16-NEXT: str   h0, [x0]
; CHECK-FP16-NEXT: ret

define void @test_fccmp(half %in, half* %out) {
  %cmp1 = fcmp ogt half %in, 0xH4800
  %cmp2 = fcmp olt half %in, 0xH4500
  %cond = and i1 %cmp1, %cmp2
  %result = select i1 %cond, half %in, half 0xH4500
  store half %result, half* %out
  ret void
}

; CHECK-CVT-LABEL: test_br_cc:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcmp s0, s1
; CHECK-CVT-NEXT: b.mi [[BRCC_ELSE:.?LBB[0-9_]+]]
; CHECK-CVT-NEXT: str  wzr, [x0]
; CHECK-CVT-NEXT: ret
; CHECK-CVT-NEXT: [[BRCC_ELSE]]:
; CHECK-CVT-NEXT: str  wzr, [x1]
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_br_cc:
; CHECK-FP16-NEXT: fcmp h0, h1
; CHECK-FP16-NEXT: b.mi [[BRCC_ELSE:.?LBB[0-9_]+]]
; CHECK-FP16-NEXT: str  wzr, [x0]
; CHECK-FP16-NEXT: ret
; CHECK-FP16-NEXT: [[BRCC_ELSE]]:
; CHECK-FP16-NEXT: str  wzr, [x1]
; CHECK-FP16-NEXT: ret

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

; CHECK-COMMON-LABEL: test_phi:
; CHECK-COMMON: mov  x[[PTR:[0-9]+]], x0
; CHECK-COMMON: ldr  h[[AB:[0-9]+]], [x0]
; CHECK-COMMON: [[LOOP:LBB[0-9_]+]]:
; CHECK-COMMON: mov.16b  v[[R:[0-9]+]], v[[AB]]
; CHECK-COMMON: ldr  h[[AB]], [x[[PTR]]]
; CHECK-COMMON: mov  x0, x[[PTR]]
; CHECK-COMMON: bl {{_?}}test_dummy
; CHECK-COMMON: mov.16b  v0, v[[R]]
; CHECK-COMMON: ret
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

; CHECK-CVT-LABEL: test_fptosi_i32:
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcvtzs w0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fptosi_i32:
; CHECK-FP16-NEXT: fcvtzs w0, h0
; CHECK-FP16-NEXT: ret

define i32 @test_fptosi_i32(half %a) #0 {
  %r = fptosi half %a to i32
  ret i32 %r
}

; CHECK-CVT-LABEL: test_fptosi_i64:
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcvtzs x0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fptosi_i64:
; CHECK-FP16-NEXT: fcvtzs x0, h0
; CHECK-FP16-NEXT: ret

define i64 @test_fptosi_i64(half %a) #0 {
  %r = fptosi half %a to i64
  ret i64 %r
}

; CHECK-CVT-LABEL: test_fptoui_i32:
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcvtzu w0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fptoui_i32:
; CHECK-FP16-NEXT: fcvtzu w0, h0
; CHECK-FP16-NEXT: ret

define i32 @test_fptoui_i32(half %a) #0 {
  %r = fptoui half %a to i32
  ret i32 %r
}

; CHECK-CVT-LABEL: test_fptoui_i64:
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcvtzu x0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fptoui_i64:
; CHECK-FP16-NEXT: fcvtzu x0, h0
; CHECK-FP16-NEXT: ret

define i64 @test_fptoui_i64(half %a) #0 {
  %r = fptoui half %a to i64
  ret i64 %r
}

; CHECK-CVT-LABEL: test_uitofp_i32:
; CHECK-CVT-NEXT: ucvtf s0, w0
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_uitofp_i32:
; CHECK-FP16-NEXT: ucvtf h0, w0
; CHECK-FP16-NEXT: ret

define half @test_uitofp_i32(i32 %a) #0 {
  %r = uitofp i32 %a to half
  ret half %r
}

; CHECK-CVT-LABEL: test_uitofp_i64:
; CHECK-CVT-NEXT: ucvtf s0, x0
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_uitofp_i64:
; CHECK-FP16-NEXT: ucvtf h0, x0
; CHECK-FP16-NEXT: ret

define half @test_uitofp_i64(i64 %a) #0 {
  %r = uitofp i64 %a to half
  ret half %r
}

; CHECK-CVT-LABEL: test_sitofp_i32:
; CHECK-CVT-NEXT: scvtf s0, w0
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_sitofp_i32:
; CHECK-FP16-NEXT: scvtf h0, w0
; CHECK-FP16-NEXT: ret

define half @test_sitofp_i32(i32 %a) #0 {
  %r = sitofp i32 %a to half
  ret half %r
}

; CHECK-CVT-LABEL: test_sitofp_i64:
; CHECK-CVT-NEXT: scvtf s0, x0
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_sitofp_i64:
; CHECK-FP16-NEXT: scvtf h0, x0
; CHECK-FP16-NEXT: ret
define half @test_sitofp_i64(i64 %a) #0 {
  %r = sitofp i64 %a to half
  ret half %r
}

; CHECK-CVT-LABEL: test_uitofp_i32_fadd:
; CHECK-CVT-NEXT: ucvtf s1, w0
; CHECK-CVT-NEXT: fcvt h1, s1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fadd s0, s0, s1
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_uitofp_i32_fadd:
; CHECK-FP16-NEXT: ucvtf h1, w0
; CHECK-FP16-NEXT: fadd h0, h0, h1
; CHECK-FP16-NEXT: ret

define half @test_uitofp_i32_fadd(i32 %a, half %b) #0 {
  %c = uitofp i32 %a to half
  %r = fadd half %b, %c
  ret half %r
}

; CHECK-CVT-LABEL: test_sitofp_i32_fadd:
; CHECK-CVT-NEXT: scvtf s1, w0
; CHECK-CVT-NEXT: fcvt h1, s1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fadd s0, s0, s1
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_sitofp_i32_fadd:
; CHECK-FP16-NEXT: scvtf h1, w0
; CHECK-FP16-NEXT: fadd h0, h0, h1
; CHECK-FP16-NEXT: ret

define half @test_sitofp_i32_fadd(i32 %a, half %b) #0 {
  %c = sitofp i32 %a to half
  %r = fadd half %b, %c
  ret half %r
}

; CHECK-COMMON-LABEL: test_fptrunc_float:
; CHECK-COMMON-NEXT: fcvt h0, s0
; CHECK-COMMON-NEXT: ret

define half @test_fptrunc_float(float %a) #0 {
  %r = fptrunc float %a to half
  ret half %r
}

; CHECK-COMMON-LABEL: test_fptrunc_double:
; CHECK-COMMON-NEXT: fcvt h0, d0
; CHECK-COMMON-NEXT: ret
define half @test_fptrunc_double(double %a) #0 {
  %r = fptrunc double %a to half
  ret half %r
}

; CHECK-COMMON-LABEL: test_fpext_float:
; CHECK-COMMON-NEXT: fcvt s0, h0
; CHECK-COMMON-NEXT: ret
define float @test_fpext_float(half %a) #0 {
  %r = fpext half %a to float
  ret float %r
}

; CHECK-COMMON-LABEL: test_fpext_double:
; CHECK-COMMON-NEXT: fcvt d0, h0
; CHECK-COMMON-NEXT: ret
define double @test_fpext_double(half %a) #0 {
  %r = fpext half %a to double
  ret double %r
}


; CHECK-COMMON-LABEL: test_bitcast_halftoi16:
; CHECK-COMMON-NEXT: fmov w0, s0
; CHECK-COMMON-NEXT: ret
define i16 @test_bitcast_halftoi16(half %a) #0 {
  %r = bitcast half %a to i16
  ret i16 %r
}

; CHECK-COMMON-LABEL: test_bitcast_i16tohalf:
; CHECK-COMMON-NEXT: fmov s0, w0
; CHECK-COMMON-NEXT: ret
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
declare half @llvm.aarch64.neon.frecpe.f16(half %a) #0
declare half @llvm.aarch64.neon.frecpx.f16(half %a) #0
declare half @llvm.aarch64.neon.frsqrte.f16(half %a) #0

; FALLBACK-NOT: remark:{{.*}}test_sqrt
; FALLBACK-FP16-NOT: remark:{{.*}}test_sqrt

; CHECK-CVT-LABEL: test_sqrt:
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fsqrt s0, s0
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_sqrt:
; CHECK-FP16-NEXT: fsqrt h0, h0
; CHECK-FP16-NEXT: ret

; GISEL-CVT-LABEL: test_sqrt:
; GISEL-CVT-NEXT: fcvt s0, h0
; GISEL-CVT-NEXT: fsqrt s0, s0
; GISEL-CVT-NEXT: fcvt h0, s0
; GISEL-CVT-NEXT: ret

; GISEL-FP16-LABEL: test_sqrt:
; GISEL-FP16-NEXT: fsqrt h0, h0
; GISEL-FP16-NEXT: ret

define half @test_sqrt(half %a) #0 {
  %r = call half @llvm.sqrt.f16(half %a)
  ret half %r
}

; CHECK-COMMON-LABEL: test_powi:
; CHECK-COMMON-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-COMMON-NEXT: mov  x29, sp
; CHECK-COMMON-NEXT: fcvt s0, h0
; CHECK-COMMON-NEXT: bl {{_?}}__powisf2
; CHECK-COMMON-NEXT: fcvt h0, s0
; CHECK-COMMON-NEXT: ldp x29, x30, [sp], #16
; CHECK-COMMON-NEXT: ret
define half @test_powi(half %a, i32 %b) #0 {
  %r = call half @llvm.powi.f16(half %a, i32 %b)
  ret half %r
}

; FALLBACK-NOT: remark:{{.*}}test_sin
; FALLBACK-FP16-NOT: remark:{{.*}}test_sin

; CHECK-COMMON-LABEL: test_sin:
; CHECK-COMMON-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-COMMON-NEXT: mov  x29, sp
; CHECK-COMMON-NEXT: fcvt s0, h0
; CHECK-COMMON-NEXT: bl {{_?}}sinf
; CHECK-COMMON-NEXT: fcvt h0, s0
; CHECK-COMMON-NEXT: ldp x29, x30, [sp], #16
; CHECK-COMMON-NEXT: ret

; GISEL-LABEL: test_sin:
; GISEL-NEXT: stp x29, x30, [sp, #-16]!
; GISEL-NEXT: mov  x29, sp
; GISEL-NEXT: fcvt s0, h0
; GISEL-NEXT: bl {{_?}}sinf
; GISEL-NEXT: fcvt h0, s0
; GISEL-NEXT: ldp x29, x30, [sp], #16
; GISEL-NEXT: ret
define half @test_sin(half %a) #0 {
  %r = call half @llvm.sin.f16(half %a)
  ret half %r
}

; FALLBACK-NOT: remark:{{.*}}test_cos
; FALLBACK-FP16-NOT: remark:{{.*}}test_cos

; CHECK-COMMON-LABEL: test_cos:
; CHECK-COMMON-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-COMMON-NEXT: mov  x29, sp
; CHECK-COMMON-NEXT: fcvt s0, h0
; CHECK-COMMON-NEXT: bl {{_?}}cosf
; CHECK-COMMON-NEXT: fcvt h0, s0
; CHECK-COMMON-NEXT: ldp x29, x30, [sp], #16
; CHECK-COMMON-NEXT: ret

; GISEL-LABEL: test_cos:
; GISEL-NEXT: stp x29, x30, [sp, #-16]!
; GISEL-NEXT: mov  x29, sp
; GISEL-NEXT: fcvt s0, h0
; GISEL-NEXT: bl {{_?}}cosf
; GISEL-NEXT: fcvt h0, s0
; GISEL-NEXT: ldp x29, x30, [sp], #16
; GISEL-NEXT: ret
define half @test_cos(half %a) #0 {
  %r = call half @llvm.cos.f16(half %a)
  ret half %r
}

; CHECK-COMMON-LABEL: test_pow:
; CHECK-COMMON-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-COMMON-NEXT: mov  x29, sp
; CHECK-COMMON-NEXT: fcvt s0, h0
; CHECK-COMMON-NEXT: fcvt s1, h1
; CHECK-COMMON-NEXT: bl {{_?}}powf
; CHECK-COMMON-NEXT: fcvt h0, s0
; CHECK-COMMON-NEXT: ldp x29, x30, [sp], #16
; CHECK-COMMON-NEXT: ret
define half @test_pow(half %a, half %b) #0 {
  %r = call half @llvm.pow.f16(half %a, half %b)
  ret half %r
}

; FALLBACK-NOT: remark:{{.*}}test_exp
; FALLBACK-FP16-NOT: remark:{{.*}}test_exp

; CHECK-COMMON-LABEL: test_exp:
; CHECK-COMMON-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-COMMON-NEXT: mov  x29, sp
; CHECK-COMMON-NEXT: fcvt s0, h0
; CHECK-COMMON-NEXT: bl {{_?}}expf
; CHECK-COMMON-NEXT: fcvt h0, s0
; CHECK-COMMON-NEXT: ldp x29, x30, [sp], #16
; CHECK-COMMON-NEXT: ret

; GISEL-LABEL: test_exp:
; GISEL-NEXT: stp x29, x30, [sp, #-16]!
; GISEL-NEXT: mov  x29, sp
; GISEL-NEXT: fcvt s0, h0
; GISEL-NEXT: bl {{_?}}expf
; GISEL-NEXT: fcvt h0, s0
; GISEL-NEXT: ldp x29, x30, [sp], #16
; GISEL-NEXT: ret
define half @test_exp(half %a) #0 {
  %r = call half @llvm.exp.f16(half %a)
  ret half %r
}

; CHECK-COMMON-LABEL: test_exp2:
; CHECK-COMMON-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-COMMON-NEXT: mov  x29, sp
; CHECK-COMMON-NEXT: fcvt s0, h0
; CHECK-COMMON-NEXT: bl {{_?}}exp2f
; CHECK-COMMON-NEXT: fcvt h0, s0
; CHECK-COMMON-NEXT: ldp x29, x30, [sp], #16
; CHECK-COMMON-NEXT: ret

; GISEL-LABEL: test_exp2:
; GISEL-NEXT: stp x29, x30, [sp, #-16]!
; GISEL-NEXT: mov  x29, sp
; GISEL-NEXT: fcvt s0, h0
; GISEL-NEXT: bl {{_?}}exp2f
; GISEL-NEXT: fcvt h0, s0
; GISEL-NEXT: ldp x29, x30, [sp], #16
; GISEL-NEXT: ret
define half @test_exp2(half %a) #0 {
  %r = call half @llvm.exp2.f16(half %a)
  ret half %r
}

; FALLBACK-NOT: remark:{{.*}}test_log
; FALLBACK-FP16-NOT: remark:{{.*}}test_log

; CHECK-COMMON-LABEL: test_log:
; CHECK-COMMON-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-COMMON-NEXT: mov  x29, sp
; CHECK-COMMON-NEXT: fcvt s0, h0
; CHECK-COMMON-NEXT: bl {{_?}}logf
; CHECK-COMMON-NEXT: fcvt h0, s0
; CHECK-COMMON-NEXT: ldp x29, x30, [sp], #16
; CHECK-COMMON-NEXT: ret

; GISEL-LABEL: test_log:
; GISEL: stp x29, x30, [sp, #-16]!
; GISEL-NEXT: mov  x29, sp
; GISEL-NEXT: fcvt s0, h0
; GISEL-NEXT: bl {{_?}}logf
; GISEL-NEXT: fcvt h0, s0
; GISEL-NEXT: ldp x29, x30, [sp], #16
; GISEL-NEXT: ret

define half @test_log(half %a) #0 {
  %r = call half @llvm.log.f16(half %a)
  ret half %r
}

; FALLBACK-NOT: remark:{{.*}}test_log10
; FALLBACK-FP16-NOT: remark:{{.*}}test_log10

; CHECK-COMMON-LABEL: test_log10:
; CHECK-COMMON-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-COMMON-NEXT: mov  x29, sp
; CHECK-COMMON-NEXT: fcvt s0, h0
; CHECK-COMMON-NEXT: bl {{_?}}log10f
; CHECK-COMMON-NEXT: fcvt h0, s0
; CHECK-COMMON-NEXT: ldp x29, x30, [sp], #16
; CHECK-COMMON-NEXT: ret

; GISEL-LABEL: test_log10:
; GISEL-NEXT: stp x29, x30, [sp, #-16]!
; GISEL-NEXT: mov  x29, sp
; GISEL-NEXT: fcvt s0, h0
; GISEL-NEXT: bl {{_?}}log10f
; GISEL-NEXT: fcvt h0, s0
; GISEL-NEXT: ldp x29, x30, [sp], #16
; GISEL-NEXT: ret

define half @test_log10(half %a) #0 {
  %r = call half @llvm.log10.f16(half %a)
  ret half %r
}

; FALLBACK-NOT: remark:{{.*}}test_log2
; FALLBACK-FP16-NOT: remark:{{.*}}test_log2

; CHECK-COMMON-LABEL: test_log2:
; CHECK-COMMON-NEXT: stp x29, x30, [sp, #-16]!
; CHECK-COMMON-NEXT: mov  x29, sp
; CHECK-COMMON-NEXT: fcvt s0, h0
; CHECK-COMMON-NEXT: bl {{_?}}log2f
; CHECK-COMMON-NEXT: fcvt h0, s0
; CHECK-COMMON-NEXT: ldp x29, x30, [sp], #16
; CHECK-COMMON-NEXT: ret

; GISEL-LABEL: test_log2:
; GISEL-NEXT: stp x29, x30, [sp, #-16]!
; GISEL-NEXT: mov  x29, sp
; GISEL-NEXT: fcvt s0, h0
; GISEL-NEXT: bl {{_?}}log2f
; GISEL-NEXT: fcvt h0, s0
; GISEL-NEXT: ldp x29, x30, [sp], #16
; GISEL-NEXT: ret

define half @test_log2(half %a) #0 {
  %r = call half @llvm.log2.f16(half %a)
  ret half %r
}

; CHECK-CVT-LABEL: test_fma:
; CHECK-CVT-NEXT: fcvt s2, h2
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fmadd s0, s0, s1, s2
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fma:
; CHECK-FP16-NEXT: fmadd h0, h0, h1, h2
; CHECK-FP16-NEXT: ret

define half @test_fma(half %a, half %b, half %c) #0 {
  %r = call half @llvm.fma.f16(half %a, half %b, half %c)
  ret half %r
}

; CHECK-CVT-LABEL: test_fabs:
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fabs s0, s0
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fabs:
; CHECK-FP16-NEXT: fabs h0, h0
; CHECK-FP16-NEXT: ret

; FALLBACK-NOT: remark:{{.*}}test_fabs
; FALLBACK-FP16-NOT: remark:{{.*}}test_fabs

; GISEL-CVT-LABEL: test_fabs:
; GISEL-CVT-NEXT: fcvt s0, h0
; GISEL-CVT-NEXT: fabs s0, s0
; GISEL-CVT-NEXT: fcvt h0, s0
; GISEL-CVT-NEXT: ret

; GISEL-FP16-LABEL: test_fabs:
; GISEL-FP16-NEXT: fabs h0, h0
; GISEL-FP16-NEXT: ret

define half @test_fabs(half %a) #0 {
  %r = call half @llvm.fabs.f16(half %a)
  ret half %r
}

; CHECK-CVT-LABEL: test_minnum:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fminnm s0, s0, s1
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_minnum:
; CHECK-FP16-NEXT: fminnm h0, h0, h1
; CHECK-FP16-NEXT: ret

define half @test_minnum(half %a, half %b) #0 {
  %r = call half @llvm.minnum.f16(half %a, half %b)
  ret half %r
}

; CHECK-CVT-LABEL: test_maxnum:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fmaxnm s0, s0, s1
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_maxnum:
; CHECK-FP16-NEXT: fmaxnm h0, h0, h1
; CHECK-FP16-NEXT: ret

define half @test_maxnum(half %a, half %b) #0 {
  %r = call half @llvm.maxnum.f16(half %a, half %b)
  ret half %r
}

; CHECK-CVT-LABEL: test_copysign:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: movi.4s v2, #128, lsl #24
; CHECK-CVT-NEXT: bit.16b v0, v1, v2
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_copysign:
; CHECK-FP16-NEXT: movi.8h v2, #128, lsl #8
; CHECK-FP16-NEXT: bit.16b  v0, v1, v2
; CHECK-FP16-NEXT: ret

define half @test_copysign(half %a, half %b) #0 {
  %r = call half @llvm.copysign.f16(half %a, half %b)
  ret half %r
}

; CHECK-CVT-LABEL: test_copysign_f32:
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: movi.4s v2, #128, lsl #24
; CHECK-CVT-NEXT: bit.16b v0, v1, v2
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_copysign_f32:
; CHECK-FP16-NEXT: fcvt h1, s1
; CHECK-FP16-NEXT: movi.8h	v2, #128, lsl #8
; CHECK-FP16-NEXT: bit.16b v0, v1, v2
; CHECK-FP16-NEXT: ret

define half @test_copysign_f32(half %a, float %b) #0 {
  %tb = fptrunc float %b to half
  %r = call half @llvm.copysign.f16(half %a, half %tb)
  ret half %r
}

; CHECK-CVT-LABEL: test_copysign_f64:
; CHECK-CVT-NEXT: fcvt s1, d1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: movi.4s v2, #128, lsl #24
; CHECK-CVT-NEXT: bit.16b v0, v1, v2
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_copysign_f64:
; CHECK-FP16-NEXT: fcvt h1, d1
; CHECK-FP16-NEXT: movi.8h v2, #128, lsl #8
; CHECK-FP16-NEXT: bit.16b v0, v1, v2
; CHECK-FP16-NEXT: ret

define half @test_copysign_f64(half %a, double %b) #0 {
  %tb = fptrunc double %b to half
  %r = call half @llvm.copysign.f16(half %a, half %tb)
  ret half %r
}

; Check that the FP promotion will use a truncating FP_ROUND, so we can fold
; away the (fpext (fp_round <result>)) here.

; CHECK-CVT-LABEL: test_copysign_extended:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: movi.4s v2, #128, lsl #24
; CHECK-CVT-NEXT: bit.16b v0, v1, v2
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_copysign_extended:
; CHECK-FP16-NEXT: movi.8h v2, #128, lsl #8
; CHECK-FP16-NEXT: bit.16b v0, v1, v2
; CHECK-FP16-NEXT: fcvt s0, h0
; CHECK-FP16-NEXT: ret

define float @test_copysign_extended(half %a, half %b) #0 {
  %r = call half @llvm.copysign.f16(half %a, half %b)
  %xr = fpext half %r to float
  ret float %xr
}

; CHECK-CVT-LABEL: test_floor:
; CHECK-CVT-NEXT: fcvt [[FLOAT32:s[0-9]+]], h0
; CHECK-CVT-NEXT: frintm [[INT32:s[0-9]+]], [[FLOAT32]]
; CHECK-CVT-NEXT: fcvt h0, [[INT32]]
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_floor:
; CHECK-FP16-NEXT: frintm h0, h0
; CHECK-FP16-NEXT: ret

; FALLBACK-NOT: remark:{{.*}}test_floor
; FALLBACK-FP16-NOT: remark:{{.*}}test_floor

; GISEL-CVT-LABEL: test_floor:
; GISEL-CVT-NEXT: fcvt [[FLOAT32:s[0-9]+]], h0
; GISEL-CVT-NEXT: frintm [[INT32:s[0-9]+]], [[FLOAT32]]
; GISEL-CVT-NEXT: fcvt h0, [[INT32]]
; GISEL-CVT-NEXT: ret

; GISEL-FP16-LABEL: test_floor:
; GISEL-FP16-NEXT: frintm h0, h0
; GISEL-FP16-NEXT: ret

define half @test_floor(half %a) #0 {
  %r = call half @llvm.floor.f16(half %a)
  ret half %r
}

; CHECK-CVT-LABEL: test_ceil:
; CHECK-CVT-NEXT: fcvt [[FLOAT32:s[0-9]+]], h0
; CHECK-CVT-NEXT: frintp [[INT32:s[0-9]+]], [[FLOAT32]]
; CHECK-CVT-NEXT: fcvt h0, [[INT32]]
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_ceil:
; CHECK-FP16-NEXT: frintp h0, h0
; CHECK-FP16-NEXT: ret

; FALLBACK-NOT: remark:{{.*}}test_ceil
; FALLBACK-FP16-NOT: remark:{{.*}}test_ceil

; GISEL-CVT-LABEL: test_ceil:
; GISEL-CVT-NEXT: fcvt [[FLOAT32:s[0-9]+]], h0
; GISEL-CVT-NEXT: frintp [[INT32:s[0-9]+]], [[FLOAT32]]
; GISEL-CVT-NEXT: fcvt h0, [[INT32]]
; GISEL-CVT-NEXT: ret

; GISEL-FP16-LABEL: test_ceil:
; GISEL-FP16-NEXT: frintp h0, h0
; GISEL-FP16-NEXT: ret
define half @test_ceil(half %a) #0 {
  %r = call half @llvm.ceil.f16(half %a)
  ret half %r
}

; CHECK-CVT-LABEL: test_trunc:
; CHECK-CVT-NEXT: fcvt [[FLOAT32:s[0-9]+]], h0
; CHECK-CVT-NEXT: frintz [[INT32:s[0-9]+]], [[FLOAT32]]
; CHECK-CVT-NEXT: fcvt h0, [[INT32]]
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_trunc:
; CHECK-FP16-NEXT: frintz h0, h0
; CHECK-FP16-NEXT: ret

define half @test_trunc(half %a) #0 {
  %r = call half @llvm.trunc.f16(half %a)
  ret half %r
}

; CHECK-CVT-LABEL: test_rint:
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: frintx s0, s0
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_rint:
; CHECK-FP16-NEXT: frintx h0, h0
; CHECK-FP16-NEXT: ret

define half @test_rint(half %a) #0 {
  %r = call half @llvm.rint.f16(half %a)
  ret half %r
}

; CHECK-CVT-LABEL: test_nearbyint:
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: frinti s0, s0
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_nearbyint:
; CHECK-FP16-NEXT: frinti h0, h0
; CHECK-FP16-NEXT: ret

define half @test_nearbyint(half %a) #0 {
  %r = call half @llvm.nearbyint.f16(half %a)
  ret half %r
}

; CHECK-CVT-LABEL: test_round:
; CHECK-CVT-NEXT: fcvt [[FLOAT32:s[0-9]+]], h0
; CHECK-CVT-NEXT: frinta [[INT32:s[0-9]+]], [[FLOAT32]]
; CHECK-CVT-NEXT: fcvt h0, [[INT32]]
; CHECK-CVT-NEXT: ret

; GISEL-CVT-LABEL: test_round:
; GISEL-CVT-NEXT: fcvt [[FLOAT32:s[0-9]+]], h0
; GISEL-CVT-NEXT: frinta [[INT32:s[0-9]+]], [[FLOAT32]]
; GISEL-CVT-NEXT: fcvt h0, [[INT32]]
; GISEL-CVT-NEXT: ret


; CHECK-FP16-LABEL: test_round:
; CHECK-FP16-NEXT: frinta h0, h0
; CHECK-FP16-NEXT: ret

; GISEL-FP16-LABEL: test_round:
; GISEL-FP16-NEXT: frinta h0, h0
; GISEL-FP16-NEXT: ret

define half @test_round(half %a) #0 {
  %r = call half @llvm.round.f16(half %a)
  ret half %r
}

; CHECK-CVT-LABEL: test_fmuladd:
; CHECK-CVT-NEXT: fcvt s1, h1
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fmul s0, s0, s1
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: fcvt s0, h0
; CHECK-CVT-NEXT: fcvt s1, h2
; CHECK-CVT-NEXT: fadd s0, s0, s1
; CHECK-CVT-NEXT: fcvt h0, s0
; CHECK-CVT-NEXT: ret

; CHECK-FP16-LABEL: test_fmuladd:
; CHECK-FP16-NEXT: fmul h0, h0, h1
; CHECK-FP16-NEXT: fadd h0, h0, h2
; CHECK-FP16-NEXT: ret

define half @test_fmuladd(half %a, half %b, half %c) #0 {
  %r = call half @llvm.fmuladd.f16(half %a, half %b, half %c)
  ret half %r
}

; CHECK-FP16-LABEL: test_vrecpeh_f16:
; CHECK-FP16-NEXT: frecpe h0, h0
; CHECK-FP16-NEXT: ret

define half @test_vrecpeh_f16(half %a) #0 {
  %r = call half @llvm.aarch64.neon.frecpe.f16(half %a)
  ret half %r
}

; CHECK-FP16-LABEL: test_vrecpxh_f16:
; CHECK-FP16-NEXT: frecpx h0, h0
; CHECK-FP16-NEXT: ret

define half @test_vrecpxh_f16(half %a) #0 {
  %r = call half @llvm.aarch64.neon.frecpx.f16(half %a)
  ret half %r
}

; CHECK-FP16-LABEL: test_vrsqrteh_f16:
; CHECK-FP16-NEXT: frsqrte h0, h0
; CHECK-FP16-NEXT: ret

define half @test_vrsqrteh_f16(half %a) #0 {
  %r = call half @llvm.aarch64.neon.frsqrte.f16(half %a)
  ret half %r
}

attributes #0 = { nounwind }
