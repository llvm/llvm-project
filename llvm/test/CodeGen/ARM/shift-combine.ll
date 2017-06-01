; RUN: llc -mtriple=armv7-linux-gnueabihf %s -o - | FileCheck %s

@array = weak global [4 x i32] zeroinitializer

define i32 @test_lshr_and1(i32 %x) {
entry:
;CHECK-LABLE: test_lshr_and1:
;CHECK:         movw r1, :lower16:array
;CHECK-NEXT:    and  r0, r0, #12
;CHECK-NEXT:    movt r1, :upper16:array
;CHECK-NEXT:    ldr  r0, [r1, r0]
;CHECK-NEXT:    bx   lr
  %tmp2 = lshr i32 %x, 2
  %tmp3 = and i32 %tmp2, 3
  %tmp4 = getelementptr [4 x i32], [4 x i32]* @array, i32 0, i32 %tmp3
  %tmp5 = load i32, i32* %tmp4, align 4
  ret i32 %tmp5
}
define i32 @test_lshr_and2(i32 %x) {
entry:
;CHECK-LABLE: test_lshr_and2:
;CHECK:         ubfx r0, r0, #1, #15
;CHECK-NEXT:    add  r0, r0, r0
;CHECK-NEXT:    bx   lr
  %a = and i32 %x, 65534
  %b = lshr i32 %a, 1
  %c = and i32 %x, 65535
  %d = lshr i32 %c, 1
  %e = add i32 %b, %d
  ret i32 %e
}
