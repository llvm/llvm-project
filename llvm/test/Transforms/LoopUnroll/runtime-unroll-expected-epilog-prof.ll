; RUN: opt -passes=loop-unroll -unroll-runtime -unroll-count=4 -S %s \
; RUN:   | FileCheck %s
;
; Remainder guards are new edges. Don't copy llvm.expect onto them.

; CHECK-NOT: Function Attrs: approxprofile
; CHECK-LABEL: define void @expected_epilog(
; CHECK: br i1 %lcmp.mod, label %loop.epil.preheader, label %exit, !prof [[UNKNOWN:![0-9]+]]
; CHECK: [[UNKNOWN]] = !{!"unknown", !"loop-unroll"}

define void @expected_epilog(ptr %p, i64 %n) !prof !1 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %ptr = getelementptr inbounds i32, ptr %p, i64 %iv
  %v = load i32, ptr %ptr
  %add = add i32 %v, 1
  store i32 %add, ptr %ptr
  %iv.next = add i64 %iv, 1
  %cmp.loop = icmp eq i64 %iv.next, %n
  br i1 %cmp.loop, label %exit, label %loop, !prof !0

exit:
  ret void
}

!0 = !{!"branch_weights", !"expected", i32 1, i32 3}
!1 = !{!"function_entry_count", i64 10}
