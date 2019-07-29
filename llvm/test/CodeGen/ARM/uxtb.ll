; RUN: llc -mtriple armv6-apple-darwin -filetype asm -o - %s | FileCheck %s

define i32 @test1(i32 %x) {
; CHECK-LABEL: test1:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    uxtb16 r0, r0
; CHECK-NEXT:    bx lr
  %tmp1 = and i32 %x, 16711935
  ret i32 %tmp1
}

define i32 @test2(i32 %x) {
; CHECK-LABEL: test2:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    uxtb16 r0, r0, ror #8
; CHECK-NEXT:    bx lr
  %tmp1 = lshr i32 %x, 8
  %tmp2 = and i32 %tmp1, 16711935
  ret i32 %tmp2
}

define i32 @test3(i32 %x) {
; CHECK-LABEL: test3:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    uxtb16 r0, r0, ror #8
; CHECK-NEXT:    bx lr
  %tmp1 = lshr i32 %x, 8
  %tmp2 = and i32 %tmp1, 16711935
  ret i32 %tmp2
}

define i32 @test4(i32 %x) {
; CHECK-LABEL: test4:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    uxtb16 r0, r0, ror #8
; CHECK-NEXT:    bx lr
  %tmp1 = lshr i32 %x, 8
  %tmp6 = and i32 %tmp1, 16711935
  ret i32 %tmp6
}

define i32 @test5(i32 %x) {
; CHECK-LABEL: test5:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    uxtb16 r0, r0, ror #8
; CHECK-NEXT:    bx lr
  %tmp1 = lshr i32 %x, 8
  %tmp2 = and i32 %tmp1, 16711935
  ret i32 %tmp2
}

define i32 @test6(i32 %x) {
; CHECK-LABEL: test6:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    uxtb16 r0, r0, ror #16
; CHECK-NEXT:    bx lr
  %tmp1 = lshr i32 %x, 16
  %tmp2 = and i32 %tmp1, 255
  %tmp4 = shl i32 %x, 16
  %tmp5 = and i32 %tmp4, 16711680
  %tmp6 = or i32 %tmp2, %tmp5
  ret i32 %tmp6
}

define i32 @test7(i32 %x) {
; CHECK-LABEL: test7:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    uxtb16 r0, r0, ror #16
; CHECK-NEXT:    bx lr
  %tmp1 = lshr i32 %x, 16
  %tmp2 = and i32 %tmp1, 255
  %tmp4 = shl i32 %x, 16
  %tmp5 = and i32 %tmp4, 16711680
  %tmp6 = or i32 %tmp2, %tmp5
  ret i32 %tmp6
}

define i32 @test8(i32 %x) {
; CHECK-LABEL: test8:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    uxtb16 r0, r0, ror #24
; CHECK-NEXT:    bx lr
  %tmp1 = shl i32 %x, 8
  %tmp2 = and i32 %tmp1, 16711680
  %tmp5 = lshr i32 %x, 24
  %tmp6 = or i32 %tmp2, %tmp5
  ret i32 %tmp6
}

define i32 @test9(i32 %x) {
; CHECK-LABEL: test9:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    uxtb16 r0, r0, ror #24
; CHECK-NEXT:    bx lr
  %tmp1 = lshr i32 %x, 24
  %tmp4 = shl i32 %x, 8
  %tmp5 = and i32 %tmp4, 16711680
  %tmp6 = or i32 %tmp5, %tmp1
  ret i32 %tmp6
}

define i32 @test10(i32 %p0) {
; CHECK-LABEL: test10:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    mov r1, #248
; CHECK-NEXT:    orr r1, r1, #16252928
; CHECK-NEXT:    and r0, r1, r0, lsr #7
; CHECK-NEXT:    lsr r1, r0, #5
; CHECK-NEXT:    uxtb16 r1, r1
; CHECK-NEXT:    orr r0, r1, r0
; CHECK-NEXT:    bx lr
  %tmp1 = lshr i32 %p0, 7
  %tmp2 = and i32 %tmp1, 16253176
  %tmp4 = lshr i32 %tmp2, 5
  %tmp5 = and i32 %tmp4, 458759
  %tmp7 = or i32 %tmp5, %tmp2
  ret i32 %tmp7
}

