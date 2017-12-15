; RUN: llc -mtriple=armv7-linux-gnueabihf %s -o - | FileCheck %s --check-prefix=CHECK-ARM --check-prefix=CHECK-COMMON
; RUN: llc -mtriple=armv7eb-linux-gnueabihf %s -o - | FileCheck %s --check-prefix=CHECK-BE
; RUN: llc -mtriple=thumbv7-linux-gnueabihf %s -o - | FileCheck %s --check-prefix=CHECK-THUMB --check-prefix=CHECK-COMMON
; RUN: llc -mtriple=thumbv7m %s -o - | FileCheck %s --check-prefix=CHECK-THUMB --check-prefix=CHECK-COMMON
; RUN: llc -mtriple=thumbv7m -mattr=+strict-align %s -o - | FileCheck %s --check-prefix=CHECK-ALIGN --check-prefix=CHECK-COMMON
; RUN: llc -mtriple=thumbv6m %s -o - | FileCheck %s --check-prefix=CHECK-V6M

@array = weak global [4 x i32] zeroinitializer

define i32 @test_lshr_and1(i32 %x) {
entry:
;CHECK-LABLE: test_lshr_and1:
;CHECK-COMMON:      movw r1, :lower16:array
;CHECK-COMMON-NEXT: and  r0, r0, #12
;CHECK-COMMON-NEXT: movt r1, :upper16:array
;CHECK-COMMON-NEXT: ldr  r0, [r1, r0]
;CHECK-COMMON-NEXT: bx   lr
  %tmp2 = lshr i32 %x, 2
  %tmp3 = and i32 %tmp2, 3
  %tmp4 = getelementptr [4 x i32], [4 x i32]* @array, i32 0, i32 %tmp3
  %tmp5 = load i32, i32* %tmp4, align 4
  ret i32 %tmp5
}
define i32 @test_lshr_and2(i32 %x) {
entry:
;CHECK-LABEL: test_lshr_and2:
;CHECK-COMMON:  ubfx r0, r0, #1, #15
;CHECK-ARM:     add  r0, r0, r0
;CHECK-THUMB:   add  r0, r0
;CHECK-COMMON:  bx   lr
  %a = and i32 %x, 65534
  %b = lshr i32 %a, 1
  %c = and i32 %x, 65535
  %d = lshr i32 %c, 1
  %e = add i32 %b, %d
  ret i32 %e
}

; CHECK-LABEL: test_lshr_load1
; CHECK-BE:         ldrb r0, [r0]
; CHECK-COMMON:     ldrb r0, [r0, #1]
; CHECK-COMMON-NEXT: bx
define arm_aapcscc i32 @test_lshr_load1(i16* %a) {
entry:
  %0 = load i16, i16* %a, align 2
  %conv1 = zext i16 %0 to i32
  %1 = lshr i32 %conv1, 8
  ret i32 %1
}

; CHECK-LABEL: test_lshr_load1_sext
; CHECK-ARM:        ldrsh r0, [r0]
; CHECK-ARM-NEXT:   lsr r0, r0, #8
; CHECK-THUMB:      ldrsh.w r0, [r0]
; CHECK-THUMB-NEXT: lsrs r0, r0, #8
; CHECK-COMMON:     bx
define arm_aapcscc i32 @test_lshr_load1_sext(i16* %a) {
entry:
  %0 = load i16, i16* %a, align 2
  %conv1 = sext i16 %0 to i32
  %1 = lshr i32 %conv1, 8
  ret i32 %1
}

; CHECK-LABEL: test_lshr_load1_fail
; CHECK-COMMON: ldrh r0, [r0]
; CHECK-ARM:    lsr r0, r0, #9
; CHECK-THUMB:  lsrs r0, r0, #9
; CHECK-COMMON: bx
define arm_aapcscc i32 @test_lshr_load1_fail(i16* %a) {
entry:
  %0 = load i16, i16* %a, align 2
  %conv1 = zext i16 %0 to i32
  %1 = lshr i32 %conv1, 9
  ret i32 %1
}

; CHECK-LABEL: test_lshr_load32
; CHECK-COMMON: ldr r0, [r0]
; CHECK-ARM:    lsr r0, r0, #8
; CHECK-THUMB:  lsrs r0, r0, #8
; CHECK-COMMON: bx
define arm_aapcscc i32 @test_lshr_load32(i32* %a) {
entry:
  %0 = load i32, i32* %a, align 4
  %1 = lshr i32 %0, 8
  ret i32 %1
}

; CHECK-LABEL: test_lshr_load32_2
; CHECK-BE:         ldrh r0, [r0]
; CHECK-COMMON:     ldrh r0, [r0, #2]
; CHECK-COMMON-NEXT: bx
define arm_aapcscc i32 @test_lshr_load32_2(i32* %a) {
entry:
  %0 = load i32, i32* %a, align 4
  %1 = lshr i32 %0, 16
  ret i32 %1
}

; CHECK-LABEL: test_lshr_load32_1
; CHECK-BE:         ldrb r0, [r0]
; CHECK-COMMON:     ldrb r0, [r0, #3]
; CHECK-COMMON-NEXT: bx
define arm_aapcscc i32 @test_lshr_load32_1(i32* %a) {
entry:
  %0 = load i32, i32* %a, align 4
  %1 = lshr i32 %0, 24
  ret i32 %1
}

; CHECK-LABEL: test_lshr_load32_fail
; CHECK-BE:     ldr r0, [r0]
; CHECK-BE-NEXT: lsr r0, r0, #15
; CHECK-COMMON: ldr r0, [r0]
; CHECK-ARM:    lsr r0, r0, #15
; CHECK-THUMB:  lsrs r0, r0, #15
; CHECK-COMMON: bx
define arm_aapcscc i32 @test_lshr_load32_fail(i32* %a) {
entry:
  %0 = load i32, i32* %a, align 4
  %1 = lshr i32 %0, 15
  ret i32 %1
}

; CHECK-LABEL: test_lshr_load64_4_unaligned
; CHECK-BE:         ldr [[HIGH:r[0-9]+]], [r0]
; CHECK-BE-NEXT:    ldrh [[LOW:r[0-9]+]], [r0, #4]
; CHECK-BE-NEXT:    orr r0, [[LOW]], [[HIGH]], lsl #16
; CHECK-V6M:        ldrh [[LOW:r[0-9]+]], [r0, #2]
; CHECK-V6M:        ldr [[HIGH:r[0-9]+]], [r0, #4]
; CHECK-V6M-NEXT:   lsls [[HIGH]], [[HIGH]], #16
; CHECK-V6M-NEXT:   orrs r0, r1
; CHECK-ALIGN:      ldr [[HIGH:r[0-9]+]], [r0, #4]
; CHECK-ALIGN-NEXT: ldrh [[LOW:r[0-9]+]], [r0, #2]
; CHECK-ALIGN-NEXT: orr.w r0, [[LOW]], [[HIGH]], lsl #16
; CHECK-ARM:        ldr r0, [r0, #2]
; CHECK-THUMB:      ldr.w r0, [r0, #2]
; CHECK-COMMON:     bx
define arm_aapcscc i32 @test_lshr_load64_4_unaligned(i64* %a) {
entry:
  %0 = load i64, i64* %a, align 8
  %1 = lshr i64 %0, 16
  %conv = trunc i64 %1 to i32
  ret i32 %conv
}

; CHECK-LABEL: test_lshr_load64_1_lsb
; CHECK-BE:         ldr r1, [r0]
; CHECK-BE-NEXT:    ldrb r0, [r0, #4]
; CHECK-BE-NEXT:    orr r0, r0, r1, lsl #8
; CHECK-ARM:        ldr r0, [r0, #3]
; CHECK-THUMB:      ldr.w r0, [r0, #3]
; CHECK-ALIGN:      ldr [[HIGH:r[0-9]+]], [r0, #4]
; CHECK-ALIGN-NEXT: ldrb [[LOW:r[0-9]+]], [r0, #3]
; CHECK-ALIGN-NEXT: orr.w r0, [[LOW]], [[HIGH]], lsl #8
; CHECK-COMMON: bx
define arm_aapcscc i32 @test_lshr_load64_1_lsb(i64* %a) {
entry:
  %0 = load i64, i64* %a, align 8
  %1 = lshr i64 %0, 24
  %conv = trunc i64 %1 to i32
  ret i32 %conv
}

; CHECK-LABEL: test_lshr_load64_1_msb
; CHECK-BE:         ldrb r0, [r0]
; CHECK-BE-NEXT:    bx
; CHECK-COMMON:     ldrb r0, [r0, #7]
; CHECK-COMMON-NEXT: bx
define arm_aapcscc i32 @test_lshr_load64_1_msb(i64* %a) {
entry:
  %0 = load i64, i64* %a, align 8
  %1 = lshr i64 %0, 56
  %conv = trunc i64 %1 to i32
  ret i32 %conv
}

; CHECK-LABEL: test_lshr_load64_4
; CHECK-BE:         ldr r0, [r0]
; CHECK-BE-NEXT:    bx
; CHECK-COMMON:     ldr r0, [r0, #4]
; CHECK-COMMON-NEXT: bx
define arm_aapcscc i32 @test_lshr_load64_4(i64* %a) {
entry:
  %0 = load i64, i64* %a, align 8
  %1 = lshr i64 %0, 32
  %conv = trunc i64 %1 to i32
  ret i32 %conv
}

; CHECK-LABEL: test_lshr_load64_2
; CHECK-BE:         ldrh r0, [r0]
; CHECK-BE-NEXT:    bx
; CHECK-COMMON:     ldrh r0, [r0, #6]
; CHECK-COMMON-NEXT:bx
define arm_aapcscc i32 @test_lshr_load64_2(i64* %a) {
entry:
  %0 = load i64, i64* %a, align 8
  %1 = lshr i64 %0, 48
  %conv = trunc i64 %1 to i32
  ret i32 %conv
}

; CHECK-LABEL: test_lshr_load4_fail
; CHECK-COMMON:     ldrd r0, r1, [r0]
; CHECK-ARM:        lsr r0, r0, #8
; CHECK-ARM-NEXT:   orr r0, r0, r1, lsl #24
; CHECK-THUMB:      lsrs r0, r0, #8
; CHECK-THUMB-NEXT: orr.w r0, r0, r1, lsl #24
; CHECK-COMMON:     bx
define arm_aapcscc i32 @test_lshr_load4_fail(i64* %a) {
entry:
  %0 = load i64, i64* %a, align 8
  %1 = lshr i64 %0, 8
  %conv = trunc i64 %1 to i32
  ret i32 %conv
}

; CHECK-LABEL: test_shift8_mask8
; CHECK-BE:         ldr r1, [r0]
; CHECK-COMMON:     ldr r1, [r0]
; CHECK-COMMON:     ubfx r1, r1, #8, #8
; CHECK-COMMON:     str r1, [r0]
define arm_aapcscc void @test_shift8_mask8(i32* nocapture %p) {
entry:
  %0 = load i32, i32* %p, align 4
  %shl = lshr i32 %0, 8
  %and = and i32 %shl, 255
  store i32 %and, i32* %p, align 4
  ret void
}

; CHECK-LABEL: test_shift8_mask16
; CHECK-BE:         ldr r1, [r0]
; CHECK-COMMON:     ldr r1, [r0]
; CHECK-COMMON:     ubfx r1, r1, #8, #16
; CHECK-COMMON:     str r1, [r0]
define arm_aapcscc void @test_shift8_mask16(i32* nocapture %p) {
entry:
  %0 = load i32, i32* %p, align 4
  %shl = lshr i32 %0, 8
  %and = and i32 %shl, 65535
  store i32 %and, i32* %p, align 4
  ret void
}

; CHECK-LABEL: test_shift8_mask16
; CHECK-BE:         ldrb r0, [r0]
; CHECK-COMMON:     ldrb r0, [r0, #1]
; CHECK-COMMON:     str r0, [r1]
define arm_aapcscc void @test_sext_shift8_mask8(i16* %p, i32* %q) {
entry:
  %0 = load i16, i16* %p, align 4
  %1 = sext i16 %0 to i32
  %shl = lshr i32 %1, 8
  %and = and i32 %shl, 255
  store i32 %and, i32* %q, align 4
  ret void
}

; CHECK-LABEL: test_shift8_mask16
; CHECK-ARM:        ldrsh r0, [r0]
; CHECK-BE:         ldrsh r0, [r0]
; CHECK-THUMB:      ldrsh.w r0, [r0]
; CHECK-COMMON:     ubfx r0, r0, #8, #16
; CHECK-COMMON:     str r0, [r1]
define arm_aapcscc void @test_sext_shift8_mask16(i16* %p, i32* %q) {
entry:
  %0 = load i16, i16* %p, align 4
  %1 = sext i16 %0 to i32
  %shl = lshr i32 %1, 8
  %and = and i32 %shl, 65535
  store i32 %and, i32* %q, align 4
  ret void
}
