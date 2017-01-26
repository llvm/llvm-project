; RUN: llc -mtriple=aarch64 -verify-machineinstrs < %s | FileCheck %s

declare void @foo()

; Check that the inverted or doesn't inhibit the splitting of the
; complex conditional into three branch instructions.
; CHECK-LABEL: test_and_not
; CHECK:       cbz w0, [[L:\.LBB[0-9_]+]]
; CHECK:       cmp w1, #2
; CHECK:       b.lo [[L]]
; CHECK:       cmp w2, #2
; CHECK:       b.hi [[L]]
define void @test_and_not(i32 %a, i32 %b, i32 %c) {
bb1:
  %cmp1 = icmp ult i32 %a, 1
  %cmp2 = icmp ult i32 %b, 2
  %cmp3 = icmp ult i32 %c, 3
  %or = or i1 %cmp1, %cmp2
  %not.or = xor i1 %or, -1
  %and = and i1 %not.or, %cmp3
  br i1 %and, label %bb2, label %bb3

bb2:
  ret void

bb3:
  call void @foo()
  ret void
}



