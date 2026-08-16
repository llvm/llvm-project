; RUN: llc -mtriple=armv7-linux-gnueabihf %s -o - | FileCheck %s

define i32 @test_shl_add_ult(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test_shl_add_ult:
; CHECK: cmp r1, r2
; CHECK-NEXT: sbc r1, r1, r1
; CHECK-NEXT: rsb r0, r1, r0, lsl #1
; CHECK-NEXT: bx lr
  %cmp = icmp ult i32 %b, %c
  %zext = zext i1 %cmp to i32
  %shl = shl i32 %a, 1
  %add = add i32 %shl, %zext
  ret i32 %add
}

define i32 @test_shl_add_ugt(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test_shl_add_ugt:
; CHECK: cmp r2, r1
; CHECK-NEXT: sbc r1, r2, r2
; CHECK-NEXT: rsb r0, r1, r0, lsl #1
; CHECK-NEXT: bx lr
  %cmp = icmp ugt i32 %b, %c
  %zext = zext i1 %cmp to i32
  %shl = shl i32 %a, 1
  %add = add i32 %shl, %zext
  ret i32 %add
}

define i32 @test_shl2_add_ult(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: test_shl2_add_ult:
; CHECK: cmp r1, r2
; CHECK-NEXT: sbc r1, r1, r1
; CHECK-NEXT: rsb r0, r1, r0, lsl #2
; CHECK-NEXT: bx lr
  %cmp = icmp ult i32 %b, %c
  %zext = zext i1 %cmp to i32
  %shl = shl i32 %a, 2
  %add = add i32 %shl, %zext
  ret i32 %add
}
