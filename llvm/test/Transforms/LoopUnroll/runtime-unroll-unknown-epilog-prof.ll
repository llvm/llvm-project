; RUN: opt -passes=loop-unroll -unroll-runtime -unroll-count=4 -S %s \
; RUN:   | FileCheck %s
;
; An unknown latch stays unknown on the guard; 0-0 counts do not.

; CHECK-LABEL: define void @unknown_epilog(ptr %p, i64 %n) {
; CHECK-NOT: Function Attrs:
; CHECK: br i1 {{.*}}, label %loop.epil.preheader, label %entry.new, !prof [[UNKNOWN:![0-9]+]]
; CHECK: br i1 %lcmp.mod, label %loop.epil.preheader, label %exit, !prof [[UNKNOWN]]
;
; CHECK: Function Attrs: approxprofile
; CHECK-LABEL: define void @zero_zero_epilog(
; CHECK: br i1 %lcmp.mod, label %loop.epil.preheader, label %exit, !prof [[ZERO_ZERO:![0-9]+]]
; CHECK: attributes #[[ATTR:[0-9]+]] = { approxprofile }
; CHECK: [[UNKNOWN]] = !{!"unknown", !"test"}
; CHECK: [[ZERO_ZERO]] = !{!"branch_weights", i32 1, i32 3}

define void @unknown_epilog(ptr %p, i64 %n) {
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

define void @zero_zero_epilog(ptr %p, i64 %n) !prof !1 {
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
  br i1 %cmp.loop, label %exit, label %loop, !prof !2

exit:
  ret void
}

!0 = !{!"unknown", !"test"}
!1 = !{!"function_entry_count", i64 10}
!2 = !{!"branch_weights", i32 0, i32 0}
