; RUN: llc -mtriple=thumbv7k-apple-watchos2.0 -o - %s | FileCheck %s

%struct = type { i8, i64, i8, double, i8, <2 x float>, i8, <4 x float> }

define i32 @test_i64_align() "frame-pointer"="all" {
; CHECK-LABEL: test_i64_align:
; CHECL: movs r0, #8
  ret i32 ptrtoint(i64* getelementptr(%struct, %struct* null, i32 0, i32 1) to i32)
}

define i32 @test_f64_align() "frame-pointer"="all" {
; CHECK-LABEL: test_f64_align:
; CHECL: movs r0, #24
  ret i32 ptrtoint(double* getelementptr(%struct, %struct* null, i32 0, i32 3) to i32)
}

define i32 @test_v2f32_align() "frame-pointer"="all" {
; CHECK-LABEL: test_v2f32_align:
; CHECL: movs r0, #40
  ret i32 ptrtoint(<2 x float>* getelementptr(%struct, %struct* null, i32 0, i32 5) to i32)
}

define i32 @test_v4f32_align() "frame-pointer"="all" {
; CHECK-LABEL: test_v4f32_align:
; CHECL: movs r0, #64
  ret i32 ptrtoint(<4 x float>* getelementptr(%struct, %struct* null, i32 0, i32 7) to i32)
}

; Key point here is than an extra register has to be saved so that the DPRs end
; up in an aligned location (as prologue/epilogue inserter had calculated).
define void @test_dpr_unwind_align() "frame-pointer"="all" {
; CHECK-LABEL: test_dpr_unwind_align:
; CHECK: push {r5, r6, r7, lr}
; CHECK-NOT: sub sp
; CHECK: vpush {d8, d9}
; CHECK: .cfi_offset d9, -24
; CHECK: .cfi_offset d8, -32
; [...]
; CHECK: bl _test_i64_align
; CHECK-NOT: add sp,
; CHECK: vpop {d8, d9}
; CHECK-NOT: add sp,
; CHECK: pop {r5, r6, r7, pc}

  call void asm sideeffect "", "~{r6},~{d8},~{d9}"()

  ; Whatever
  call i32 @test_i64_align()
  ret void
}

; This time, there's no viable way to tack CS-registers onto the list: a real SP
; adjustment needs to be performed to put d8 and d9 where they should be.
define void @test_dpr_unwind_align_manually() "frame-pointer"="all" {
; CHECK-LABEL: test_dpr_unwind_align_manually:
; CHECK: push {r4, r5, r6, r7, lr}
; CHECK-NOT: sub sp
; CHECK: push.w {r8, r11}
; CHECK: sub sp, #4
; CHECK: vpush {d8, d9}
; CHECK: .cfi_offset d9, -40
; CHECK: .cfi_offset d8, -48
; [...]
; CHECK: bl _test_i64_align
; CHECK-NOT: add sp,
; CHECK: vpop {d8, d9}
; CHECK: add sp, #4
; CHECK: pop.w {r8, r11}
; CHECK: pop {r4, r5, r6, r7, pc}

  call void asm sideeffect "", "~{r4},~{r5},~{r6},~{r7},~{r8},~{d8},~{d9}"()

  ; Whatever
  call i32 @test_i64_align()
  ret void
}

; If there's only a CS1 area, the sub should be in the right place:
define void @test_dpr_unwind_align_just_cs1() "frame-pointer"="all" {
; CHECK-LABEL: test_dpr_unwind_align_just_cs1:
; CHECK: push {r4, r5, r6, r7, lr}
; CHECK: sub sp, #4
; CHECK: vpush {d8, d9}
; CHECK: .cfi_offset d9, -32
; CHECK: .cfi_offset d8, -40
; CHECK: sub sp, #8
; [...]
; CHECK: bl _test_i64_align
; CHECK: add sp, #8
; CHECK: vpop {d8, d9}
; CHECK: add sp, #4
; CHECK: pop {r4, r5, r6, r7, pc}

  call void asm sideeffect "", "~{r4},~{r5},~{r6},~{r7},~{d8},~{d9}"()

  ; Whatever
  call i32 @test_i64_align()
  ret void
}

; If there are no DPRs, we shouldn't try to align the stack in stages anyway
define void @test_dpr_unwind_align_no_dprs() "frame-pointer"="all" {
; CHECK-LABEL: test_dpr_unwind_align_no_dprs:
; CHECK: push {r4, r5, r6, r7, lr}
; CHECK: sub sp, #12
; [...]
; CHECK: bl _test_i64_align
; CHECK: add sp, #12
; CHECK: pop {r4, r5, r6, r7, pc}

  call void asm sideeffect "", "~{r4},~{r5},~{r6},~{r7}"()

  ; Whatever
  call i32 @test_i64_align()
  ret void
}

; 128-bit vectors should use 128-bit (i.e. correctly aligned) slots on
; the stack.
define <4 x float> @test_v128_stack_pass([8 x double], float, <4 x float> %in) "frame-pointer"="all" {
; CHECK-LABEL: test_v128_stack_pass:
; CHECK: add r[[ADDR:[0-9]+]], sp, #16
; CHECK: vld1.64 {d0, d1}, [r[[ADDR]]:128]

  ret <4 x float> %in
}

declare void @varargs(i32, ...)

; When varargs are enabled, we go down a different route. Still want 128-bit
; alignment though.
define void @test_v128_stack_pass_varargs(<4 x float> %in) "frame-pointer"="all" {
; CHECK-LABEL: test_v128_stack_pass_varargs:
; CHECK: add r[[ADDR:[0-9]+]], sp, #16
; CHECK: vst1.64 {d0, d1}, [r[[ADDR]]:128]

  call void(i32, ...) @varargs(i32 undef, [3 x i32] undef, float undef, <4 x float> %in)
  ret void
}

; To be compatible with AAPCS's va_start model (store r0-r3 at incoming SP, give
; a single pointer), 64-bit quantities must be pass
define i64 @test_64bit_gpr_align(i32, i64 %r2_r3, i32 %sp) "frame-pointer"="all" {
; CHECK-LABEL: test_64bit_gpr_align:
; CHECK: ldr [[RHS:r[0-9]+]], [sp]
; CHECK: adds r0, [[RHS]], r2
; CHECK: adc r1, r3, #0

  %ext = zext i32 %sp to i64
  %sum = add i64 %ext, %r2_r3
  ret i64 %sum
}
