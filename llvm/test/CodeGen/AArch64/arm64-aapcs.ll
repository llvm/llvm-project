; RUN: llc -mtriple=arm64-linux-gnu -enable-misched=false -disable-post-ra < %s | FileCheck %s

@var = dso_local global i32 0, align 4

; CHECK-LABEL: @test_i128_align
define dso_local i128 @test_i128_align(i32, i128 %arg, i32 %after) {
  store i32 %after, i32* @var, align 4
; CHECK-DAG: str w4, [{{x[0-9]+}}, :lo12:var]

  ret i128 %arg
; CHECK-DAG: mov x0, x2
; CHECK-DAG: mov x1, x3
}

; CHECK-LABEL: @test_i64x2_align
define [2 x i64] @test_i64x2_align(i32, [2 x i64] %arg, i32 %after) {
  store i32 %after, i32* @var, align 4
; CHECK-DAG: str w3, [{{x[0-9]+}}, :lo12:var]

  ret [2 x i64] %arg
; CHECK-DAG: mov x0, x1
; CHECK: mov x1, x2
}

@var64 = dso_local global i64 0, align 8

; Check stack slots are 64-bit at all times.
define dso_local void @test_stack_slots([8 x i64], i1 %bool, i8 %char, i16 %short,
                                i32 %int, i64 %long) {
; CHECK-LABEL: test_stack_slots:
; CHECK-DAG: ldr w[[ext1:[0-9]+]], [sp, #24]
; CHECK-DAG: ldrh w[[ext2:[0-9]+]], [sp, #16]
; CHECK-DAG: ldrb w[[ext3:[0-9]+]], [sp, #8]
; CHECK-DAG: ldr x[[ext4:[0-9]+]], [sp, #32]
; CHECK-DAG: ldrb w[[ext5:[0-9]+]], [sp]

  %ext_bool = zext i1 %bool to i64
  store volatile i64 %ext_bool, i64* @var64, align 8
; CHECK: str x[[ext5]], [{{x[0-9]+}}, :lo12:var64]

  %ext_char = zext i8 %char to i64
  store volatile i64 %ext_char, i64* @var64, align 8
; CHECK: str x[[ext3]], [{{x[0-9]+}}, :lo12:var64]

  %ext_short = zext i16 %short to i64
  store volatile i64 %ext_short, i64* @var64, align 8
; CHECK: str x[[ext2]], [{{x[0-9]+}}, :lo12:var64]

  %ext_int = zext i32 %int to i64
  store volatile i64 %ext_int, i64* @var64, align 8
; CHECK: str x[[ext1]], [{{x[0-9]+}}, :lo12:var64]

  store volatile i64 %long, i64* @var64, align 8
; CHECK: str x[[ext4]], [{{x[0-9]+}}, :lo12:var64]

  ret void
}

; Make sure the callee does extensions (in the absence of zext/sext
; keyword on args) while we're here.

define dso_local void @test_extension(i1 %bool, i8 %char, i16 %short, i32 %int) {
  %ext_bool = zext i1 %bool to i64
  store volatile i64 %ext_bool, i64* @var64
; CHECK: and [[EXT:x[0-9]+]], x0, #0x1
; CHECK: str [[EXT]], [{{x[0-9]+}}, :lo12:var64]

  %ext_char = sext i8 %char to i64
  store volatile i64 %ext_char, i64* @var64
; CHECK: sxtb [[EXT:x[0-9]+]], w1
; CHECK: str [[EXT]], [{{x[0-9]+}}, :lo12:var64]

  %ext_short = zext i16 %short to i64
  store volatile i64 %ext_short, i64* @var64
; CHECK: and [[EXT:x[0-9]+]], x2, #0xffff
; CHECK: str [[EXT]], [{{x[0-9]+}}, :lo12:var64]

  %ext_int = zext i32 %int to i64
  store volatile i64 %ext_int, i64* @var64
; CHECK: mov w[[EXT:[0-9]+]], w3
; CHECK: str x[[EXT]], [{{x[0-9]+}}, :lo12:var64]

  ret void
}

declare void @variadic(i32 %a, ...)

  ; Under AAPCS variadic functions have the same calling convention as
  ; others. The extra arguments should go in registers rather than on the stack.
define dso_local void @test_variadic() {
  call void(i32, ...) @variadic(i32 0, i64 1, double 2.0)
; CHECK: mov w1, #1
; CHECK: fmov d0, #2.0
; CHECK: bl variadic
  ret void
}

; We weren't marking x7 as used after deciding that the i128 didn't fit into
; registers and putting the first half on the stack, so the *second* half went
; into x7. Yuck!
define dso_local i128 @test_i128_shadow([7 x i64] %x0_x6, i128 %sp) {
; CHECK-LABEL: test_i128_shadow:
; CHECK: ldp x0, x1, [sp]

  ret i128 %sp
}

; This test is to check if fp128 can be correctly handled on stack.
define fp128 @test_fp128([8 x float] %arg0, fp128 %arg1) {
; CHECK-LABEL: test_fp128:
; CHECK: ldr {{q[0-9]+}}, [sp]
  ret fp128 %arg1
}

; Check if VPR can be correctly pass by stack.
define dso_local <2 x double> @test_vreg_stack([8 x <2 x double>], <2 x double> %varg_stack) {
entry:
; CHECK-LABEL: test_vreg_stack:
; CHECK: ldr {{q[0-9]+}}, [sp]
  ret <2 x double> %varg_stack;
}

; Check that f16 can be passed and returned (ACLE 2.0 extension)
define half @test_half(float, half %arg) {
; CHECK-LABEL: test_half:
; CHECK: fmov s0, s1
  ret half %arg;
}

; Check that f16 constants are materialized correctly
define half @test_half_const() {
; CHECK-LABEL: test_half_const:
; CHECK: ldr h0, [x{{[0-9]+}}, :lo12:{{.*}}]
  ret half 0xH4248
}

; Check that v4f16 can be passed and returned in registers
define dso_local <4 x half> @test_v4_half_register(float, <4 x half> %arg) {
; CHECK-LABEL: test_v4_half_register:
; CHECK: fmov d0, d1
  ret <4 x half> %arg;
}

; Check that v8f16 can be passed and returned in registers
define dso_local <8 x half> @test_v8_half_register(float, <8 x half> %arg) {
; CHECK-LABEL: test_v8_half_register:
; CHECK: mov v0.16b, v1.16b
  ret <8 x half> %arg;
}

; Check that v4f16 can be passed and returned on the stack
define dso_local <4 x half> @test_v4_half_stack([8 x <2 x double>], <4 x half> %arg) {
; CHECK-LABEL: test_v4_half_stack:
; CHECK: ldr d0, [sp]
  ret <4 x half> %arg;
}

; Check that v8f16 can be passed and returned on the stack
define dso_local <8 x half> @test_v8_half_stack([8 x <2 x double>], <8 x half> %arg) {
; CHECK-LABEL: test_v8_half_stack:
; CHECK: ldr q0, [sp]
  ret <8 x half> %arg;
}
