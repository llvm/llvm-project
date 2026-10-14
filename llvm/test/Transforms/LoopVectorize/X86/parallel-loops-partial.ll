; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; An unannotated store to a separate object must not cause runtime checks
; between annotated accesses.
; CHECK-LABEL: @parallel_with_unannotated_alloca(
; CHECK-NOT:     vector.memcheck:
; CHECK:         vector.body:
; CHECK:           load <4 x i32>
; CHECK:           store <4 x i32>
; CHECK-NOT:     vector.memcheck:
define void @parallel_with_unannotated_alloca(ptr %a, ptr %b, i64 %n) {
entry:
  %scratch = alloca [4 x i32]
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %v = load i32, ptr %gep.a, align 4, !llvm.access.group !1
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %v, ptr %gep.b, align 4, !llvm.access.group !1
  %idx = and i64 %iv, 3
  %gep.scratch = getelementptr inbounds [4 x i32], ptr %scratch, i64 0, i64 %idx
  store i32 %v, ptr %gep.scratch, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body, !llvm.loop !0

exit:
  ret void
}

; Same, but the load from %a is not annotated: a run-time check between %a and
; %b is still required.
;
; CHECK-LABEL: @unannotated_load(
; CHECK:         vector.memcheck:
; CHECK:         vector.body:
define void @unannotated_load(ptr %a, ptr %b, i64 %n) {
entry:
  %scratch = alloca [4 x i32]
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %v = load i32, ptr %gep.a, align 4
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %v, ptr %gep.b, align 4, !llvm.access.group !1
  %idx = and i64 %iv, 3
  %gep.scratch = getelementptr inbounds [4 x i32], ptr %scratch, i64 0, i64 %idx
  store i32 %v, ptr %gep.scratch, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %for.body, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !{!"llvm.loop.parallel_accesses", !1}}
!1 = distinct !{}
