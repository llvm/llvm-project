; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -stop-after=finalize-isel | FileCheck %s

%struct.R = type { i64, i64, i64, i64, i64, i64, i64, i64, i64, i32 }

declare x86_regcallcc void @callee_arg(i64, i64, i64, i64, i64, i64, i64, i64, i64, i32)
declare x86_regcallcc %struct.R @callee_ret()

; In test_arg_subreg, %a9 is passed in R14D. R14 must be cleared from the call
; preserved mask (CustomRegMask).
define void @test_arg_subreg(i32 %arg) nounwind {
; CHECK-LABEL: name: test_arg_subreg
; CHECK: CALL64m {{.*}} CustomRegMask(
; CHECK-NOT: $r14,
; CHECK-SAME: ), {{.*}} implicit $r14d
entry:
  call x86_regcallcc void @callee_arg(i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 %arg)
  ret void
}

; In test_ret_subreg, the 10th return value is returned in R14D. R14 must be
; cleared from the call preserved mask (CustomRegMask).
define i32 @test_ret_subreg() nounwind {
; CHECK-LABEL: name: test_ret_subreg
; CHECK: CALL64m {{.*}} CustomRegMask(
; CHECK-NOT: $r14,
; CHECK-SAME: ), {{.*}} implicit-def $r14d
entry:
  %ret = call x86_regcallcc %struct.R @callee_ret()
  %val = extractvalue %struct.R %ret, 9
  ret i32 %val
}
