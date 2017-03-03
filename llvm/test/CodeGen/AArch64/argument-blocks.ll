; RUN: llc -mtriple=aarch64-apple-ios7.0 -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-DARWINPCS
; RUN: llc -mtriple=aarch64-linux-gnu -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-AAPCS

declare void @callee(...)

define float @test_hfa_regs(float, [2 x float] %in) {
; CHECK-LABEL: test_hfa_regs:
; CHECK: fadd s0, s1, s2

  %lhs = extractvalue [2 x float] %in, 0
  %rhs = extractvalue [2 x float] %in, 1
  %sum = fadd float %lhs, %rhs
  ret float %sum
}

; Check that the array gets allocated to a contiguous block on the stack (rather
; than the default of 2 8-byte slots).
define float @test_hfa_block([7 x float], [2 x float] %in) {
; CHECK-LABEL: test_hfa_block:
; CHECK: ldp [[LHS:s[0-9]+]], [[RHS:s[0-9]+]], [sp]
; CHECK: fadd s0, [[LHS]], [[RHS]]

  %lhs = extractvalue [2 x float] %in, 0
  %rhs = extractvalue [2 x float] %in, 1
  %sum = fadd float %lhs, %rhs
  ret float %sum
}

; Check that an HFA prevents backfilling of VFP registers (i.e. %rhs must go on
; the stack rather than in s7).
define float @test_hfa_block_consume([7 x float], [2 x float] %in, float %rhs) {
; CHECK-LABEL: test_hfa_block_consume:
; CHECK-DAG: ldr [[LHS:s[0-9]+]], [sp]
; CHECK-DAG: ldr [[RHS:s[0-9]+]], [sp, #8]
; CHECK: fadd s0, [[LHS]], [[RHS]]

  %lhs = extractvalue [2 x float] %in, 0
  %sum = fadd float %lhs, %rhs
  ret float %sum
}

define float @test_hfa_stackalign([8 x float], [1 x float], [2 x float] %in) {
; CHECK-LABEL: test_hfa_stackalign:
; CHECK-AAPCS: ldp [[LHS:s[0-9]+]], [[RHS:s[0-9]+]], [sp, #8]
; CHECK-DARWINPCS: ldp [[LHS:s[0-9]+]], [[RHS:s[0-9]+]], [sp, #4]
; CHECK: fadd s0, [[LHS]], [[RHS]]
  %lhs = extractvalue [2 x float] %in, 0
  %rhs = extractvalue [2 x float] %in, 1
  %sum = fadd float %lhs, %rhs
  ret float %sum
}

; An HFA that ends up on the stack should not have any effect on where
; integer-based arguments go.
define i64 @test_hfa_ignores_gprs([7 x float], [2 x float] %in, i64, i64 %res) {
; CHECK-LABEL: test_hfa_ignores_gprs:
; CHECK: mov x0, x1
  ret i64 %res
}

; [2 x float] should not be promoted to double by the Darwin varargs handling,
; but should go in an 8-byte aligned slot.
define void @test_varargs_stackalign() {
; CHECK-LABEL: test_varargs_stackalign:
; CHECK-DARWINPCS: stp {{w[0-9]+}}, {{w[0-9]+}}, [sp, #16]

  call void(...) @callee([3 x float] undef, [2 x float] [float 1.0, float 2.0])
  ret void
}

define i64 @test_smallstruct_block([7 x i64], [2 x i64] %in) {
; CHECK-LABEL: test_smallstruct_block:
; CHECK: ldp [[LHS:x[0-9]+]], [[RHS:x[0-9]+]], [sp]
; CHECK: add x0, [[LHS]], [[RHS]]
  %lhs = extractvalue [2 x i64] %in, 0
  %rhs = extractvalue [2 x i64] %in, 1
  %sum = add i64 %lhs, %rhs
  ret i64 %sum
}

; Check that a small struct prevents backfilling of registers (i.e. %rhs
; must go on the stack rather than in x7).
define i64 @test_smallstruct_block_consume([7 x i64], [2 x i64] %in, i64 %rhs) {
; CHECK-LABEL: test_smallstruct_block_consume:
; CHECK-DAG: ldr [[LHS:x[0-9]+]], [sp]
; CHECK-DAG: ldr [[RHS:x[0-9]+]], [sp, #16]
; CHECK: add x0, [[LHS]], [[RHS]]

  %lhs = extractvalue [2 x i64] %in, 0
  %sum = add i64 %lhs, %rhs
  ret i64 %sum
}

define <1 x i64> @test_v1i64_blocked([7 x double], [2 x <1 x i64>] %in) {
; CHECK-LABEL: test_v1i64_blocked:
; CHECK: ldr d0, [sp]
  %val = extractvalue [2 x <1 x i64>] %in, 0
  ret <1 x i64> %val
}

define <1 x double> @test_v1f64_blocked([7 x double], [2 x <1 x double>] %in) {
; CHECK-LABEL: test_v1f64_blocked:
; CHECK: ldr d0, [sp]
  %val = extractvalue [2 x <1 x double>] %in, 0
  ret <1 x double> %val
}

define <2 x i32> @test_v2i32_blocked([7 x double], [2 x <2 x i32>] %in) {
; CHECK-LABEL: test_v2i32_blocked:
; CHECK: ldr d0, [sp]
  %val = extractvalue [2 x <2 x i32>] %in, 0
  ret <2 x i32> %val
}

define <2 x float> @test_v2f32_blocked([7 x double], [2 x <2 x float>] %in) {
; CHECK-LABEL: test_v2f32_blocked:
; CHECK: ldr d0, [sp]
  %val = extractvalue [2 x <2 x float>] %in, 0
  ret <2 x float> %val
}

define <4 x i16> @test_v4i16_blocked([7 x double], [2 x <4 x i16>] %in) {
; CHECK-LABEL: test_v4i16_blocked:
; CHECK: ldr d0, [sp]
  %val = extractvalue [2 x <4 x i16>] %in, 0
  ret <4 x i16> %val
}

define <4 x half> @test_v4f16_blocked([7 x double], [2 x <4 x half>] %in) {
; CHECK-LABEL: test_v4f16_blocked:
; CHECK: ldr d0, [sp]
  %val = extractvalue [2 x <4 x half>] %in, 0
  ret <4 x half> %val
}

define <8 x i8> @test_v8i8_blocked([7 x double], [2 x <8 x i8>] %in) {
; CHECK-LABEL: test_v8i8_blocked:
; CHECK: ldr d0, [sp]
  %val = extractvalue [2 x <8 x i8>] %in, 0
  ret <8 x i8> %val
}

define <2 x i64> @test_v2i64_blocked([7 x double], [2 x <2 x i64>] %in) {
; CHECK-LABEL: test_v2i64_blocked:
; CHECK: ldr q0, [sp]
  %val = extractvalue [2 x <2 x i64>] %in, 0
  ret <2 x i64> %val
}

define <2 x double> @test_v2f64_blocked([7 x double], [2 x <2 x double>] %in) {
; CHECK-LABEL: test_v2f64_blocked:
; CHECK: ldr q0, [sp]
  %val = extractvalue [2 x <2 x double>] %in, 0
  ret <2 x double> %val
}

define <4 x i32> @test_v4i32_blocked([7 x double], [2 x <4 x i32>] %in) {
; CHECK-LABEL: test_v4i32_blocked:
; CHECK: ldr q0, [sp]
  %val = extractvalue [2 x <4 x i32>] %in, 0
  ret <4 x i32> %val
}

define <4 x float> @test_v4f32_blocked([7 x double], [2 x <4 x float>] %in) {
; CHECK-LABEL: test_v4f32_blocked:
; CHECK: ldr q0, [sp]
  %val = extractvalue [2 x <4 x float>] %in, 0
  ret <4 x float> %val
}

define <8 x i16> @test_v8i16_blocked([7 x double], [2 x <8 x i16>] %in) {
; CHECK-LABEL: test_v8i16_blocked:
; CHECK: ldr q0, [sp]
  %val = extractvalue [2 x <8 x i16>] %in, 0
  ret <8 x i16> %val
}

define <8 x half> @test_v8f16_blocked([7 x double], [2 x <8 x half>] %in) {
; CHECK-LABEL: test_v8f16_blocked:
; CHECK: ldr q0, [sp]
  %val = extractvalue [2 x <8 x half>] %in, 0
  ret <8 x half> %val
}

define <16 x i8> @test_v16i8_blocked([7 x double], [2 x <16 x i8>] %in) {
; CHECK-LABEL: test_v16i8_blocked:
; CHECK: ldr q0, [sp]
  %val = extractvalue [2 x <16 x i8>] %in, 0
  ret <16 x i8> %val
}

define half @test_f16_blocked([7 x double], [2 x half] %in) {
; CHECK-LABEL: test_f16_blocked:
; CHECK: ldr h0, [sp]
  %val = extractvalue [2 x half] %in, 0
  ret half %val
}
