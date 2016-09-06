; RUN: llc -mtriple=x86_64-linux-gnu %s -o - -verify-machineinstrs | FileCheck %s

define i32 @tail_merge_unreachable(i32 %i) {
entry:
  br i1 undef, label %sw, label %end
sw:
  switch i32 %i, label %end [
    i32 99,  label %sw.bb
    i32 98,  label %sw.bb
    i32 101, label %sw.bb
    i32 97,  label %sw.bb2
    i32 96,  label %sw.bb2
    i32 100, label %sw.bb2
  ]
sw.bb:
  unreachable
sw.bb2:
  unreachable
end:
  %p = phi i32 [ 1, %sw ], [ 0, %entry ]
  ret i32 %p

; CHECK-LABEL: tail_merge_unreachable:
; Range Check
; CHECK: addl $-96
; CHECK: cmpl $5
; CHECK: jbe [[JUMP_TABLE_BLOCK:[.][A-Za-z0-9_]+]]
; CHECK: retq
; CHECK: [[JUMP_TABLE_BLOCK]]:
; CHECK: btl
; CHECK: jae [[UNREACHABLE_BLOCK:[.][A-Za-z0-9_]+]]
; CHECK [[UNREACHABLE_BLOCK]]:
; CHECK: .Lfunc_end0
}
