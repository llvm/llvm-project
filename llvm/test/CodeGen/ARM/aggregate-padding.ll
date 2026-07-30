; RUN: llc -mtriple=armv7-linux-gnueabihf %s -o - | FileCheck %s

; [2 x i64] should be contiguous when split (e.g. we shouldn't try to align all
; i32 components to 64 bits). Also makes sure i64 based types are properly
; aligned on the stack.
define i64 @test_i64_contiguous_on_stack([8 x double], float, i32 %in, [2 x i64] %arg) nounwind {
; CHECK-LABEL: test_i64_contiguous_on_stack:
; CHECK-DAG: ldr [[LO0:r[0-9]+]], [sp, #8]
; CHECK-DAG: ldr [[HI0:r[0-9]+]], [sp, #12]
; CHECK-DAG: ldr [[LO1:r[0-9]+]], [sp, #16]
; CHECK-DAG: ldr [[HI1:r[0-9]+]], [sp, #20]
; CHECK: adds r0, [[LO0]], [[LO1]]
; CHECK: adc r1, [[HI0]], [[HI1]]

  %val1 = extractvalue [2 x i64] %arg, 0
  %val2 = extractvalue [2 x i64] %arg, 1
  %sum = add i64 %val1, %val2
  ret i64 %sum
}

; [2 x i64] should try to use looks for 4 regs, not 8 (which might happen if the
; i64 -> i32, i32 split wasn't handled correctly).
define i64 @test_2xi64_uses_4_regs([8 x double], float, [2 x i64] %arg) nounwind {
; CHECK-LABEL: test_2xi64_uses_4_regs:
; CHECK-DAG: mov r0, r2
; CHECK-DAG: mov r1, r3

  %val = extractvalue [2 x i64] %arg, 1
  ret i64 %val
}

; An aggregate should be able to split between registers and stack if there is
; nothing else on the stack.
define i32 @test_aggregates_split([8 x double], i32, [4 x i32] %arg) nounwind {
; CHECK-LABEL: test_aggregates_split:
; CHECK: ldr [[VAL3:r[0-9]+]], [sp]
; CHECK: add r0, r1, [[VAL3]]

  %val0 = extractvalue [4 x i32] %arg, 0
  %val3 = extractvalue [4 x i32] %arg, 3
  %sum = add i32 %val0, %val3
  ret i32 %sum
}

; If an aggregate has to be moved entirely onto the stack, nothing should be
; able to use r0-r3 any more. Also checks that [2 x i64] properly aligned when
; it uses regs.
define i32 @test_no_int_backfilling([8 x double], float, i32, [2 x i64], i32 %arg) nounwind {
; CHECK-LABEL: test_no_int_backfilling:
; CHECK: ldr r0, [sp, #24]
  ret i32 %arg
}

; Even if the argument was successfully allocated as reg block, there should be
; no backfillig to r1.
define i32 @test_no_int_backfilling_regsonly(i32, [1 x i64], i32 %arg) {
; CHECK-LABEL: test_no_int_backfilling_regsonly:
; CHECK: ldr r0, [sp]
  ret i32 %arg
}

; If an aggregate has to be moved entirely onto the stack, nothing should be
; able to use r0-r3 any more.
define float @test_no_float_backfilling([7 x double], [4 x i32], i32, [4 x double], float %arg) nounwind {
; CHECK-LABEL: test_no_float_backfilling:
; CHECK: vldr s0, [sp, #40]
  ret float %arg
}

; They're a bit pointless, but types like [N x i8] should work as well.
define i8 @test_i8_in_regs(i32, [3 x i8] %arg) {
; CHECK-LABEL: test_i8_in_regs:
; CHECK: add r0, r1, r3
  %val0 = extractvalue [3 x i8] %arg, 0
  %val2 = extractvalue [3 x i8] %arg, 2
  %sum = add i8 %val0, %val2
  ret i8 %sum
}

define i16 @test_i16_split(i32, i32, [3 x i16] %arg) {
; CHECK-LABEL: test_i16_split:
; CHECK: ldrh [[VAL2:r[0-9]+]], [sp]
; CHECK: add r0, r2, [[VAL2]]
  %val0 = extractvalue [3 x i16] %arg, 0
  %val2 = extractvalue [3 x i16] %arg, 2
  %sum = add i16 %val0, %val2
  ret i16 %sum
}

; Beware: on the stack each i16 still gets a 32-bit slot, the array is not
; packed.
define i16 @test_i16_forced_stack([8 x double], double, i32, i32, [3 x i16] %arg) {
; CHECK-LABEL: test_i16_forced_stack:
; CHECK-DAG: ldrh [[VAL0:r[0-9]+]], [sp, #8]
; CHECK-DAG: ldrh [[VAL2:r[0-9]+]], [sp, #16]
; CHECK: add r0, [[VAL0]], [[VAL2]]
  %val0 = extractvalue [3 x i16] %arg, 0
  %val2 = extractvalue [3 x i16] %arg, 2
  %sum = add i16 %val0, %val2
  ret i16 %sum
}

; [2 x <4 x i32>] should be aligned only on a 64-bit boundary and contiguous.
; None of the two <4 x i32> elements should introduce any padding to 128 bits.
define i32 @test_4xi32_64bit_aligned_and_contiguous([8 x double], float, [2 x <4 x i32>] %arg) nounwind {
; CHECK-LABEL: test_4xi32_64bit_aligned_and_contiguous:
; CHECK-DAG: ldr [[VAL0_0:r[0-9]+]], [sp, #8]
; CHECK-DAG: ldr [[VAL1_0:r[0-9]+]], [sp, #24]
; CHECK: add r0, [[VAL0_0]], [[VAL1_0]]

  %val0 = extractvalue [2 x <4 x i32>] %arg, 0
  %val0_0 = extractelement <4 x i32> %val0, i32 0
  %val1 = extractvalue [2 x <4 x i32>] %arg, 1
  %val1_0 = extractelement <4 x i32> %val1, i32 0
  %sum = add i32 %val0_0, %val1_0
  ret i32 %sum
}
