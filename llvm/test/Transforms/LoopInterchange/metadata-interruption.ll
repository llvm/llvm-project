; RUN: opt < %s -passes=loop-interchange -loop-interchange-only-when-forced=1 -pass-remarks-output=%t -disable-output
; RUN: FileCheck -input-file %t %s

; Test that the loop-interchange stops processing for some reason even though
; some loops have metadata specyfing interchange enable.

@a = dso_local local_unnamed_addr global [128 x [128 x [128 x i32]]] zeroinitializer, align 4

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Innermost
; CHECK-NEXT: Function:        enable_innermost
; CHECK-NEXT: Args:
; CHECK-NEXT:  - String:       The metadata is invalid with an innermost loop.
define void @enable_innermost() {
entry:
  br label %for.i.header

for.i.header:
  %iv.i = phi i64 [ 0, %entry ], [ %iv.i.next, %for.i.cleanup ]
  br label %for.j.header

for.j.header:
  %iv.j = phi i64 [ 0, %for.i.header ], [ %iv.j.next, %for.j.cleanup ]
  br label %for.body

for.body:
  %iv.k = phi i64 [ 0, %for.j.header ], [ %iv.k.next, %for.body ]
  %ptr = getelementptr inbounds nuw [128 x [128 x [128 x i32]]], ptr @a, i64 %iv.k, i64 %iv.j, i64 %iv.i
  %val = load i32, ptr %ptr, align 4
  %inc = add nuw nsw i32 %val, 1
  store i32 %inc, ptr %ptr, align 4
  %iv.k.next = add nuw nsw i64 %iv.k, 1
  %exitcond.k = icmp eq i64 %iv.k.next, 128
  br i1 %exitcond.k, label %for.j.cleanup, label %for.body, !llvm.loop !0

for.j.cleanup:
  %iv.j.next = add nuw nsw i64 %iv.j, 1
  %exitcond.j = icmp eq i64 %iv.j.next, 128
  br i1 %exitcond.j, label %for.i.cleanup, label %for.j.header

for.i.cleanup:
  %iv.i.next = add nuw nsw i64 %iv.i, 1
  %exitcond.i = icmp eq i64 %iv.i.next, 128
  br i1 %exitcond.i, label %exit, label %for.i.header

exit:
  ret void
}

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            AmbiguousOrder
; CHECK-NEXT: Function:        ambiguous_order
; CHECK-NEXT: Args:
; CHECK-NEXT:  - String:       The loops to be interchanged are overlapping.
define void @ambiguous_order() {
entry:
  br label %for.i.header

for.i.header:
  %iv.i = phi i64 [ 0, %entry ], [ %iv.i.next, %for.i.cleanup ]
  br label %for.j.header

for.j.header:
  %iv.j = phi i64 [ 0, %for.i.header ], [ %iv.j.next, %for.j.cleanup ]
  br label %for.body

for.body:
  %iv.k = phi i64 [ 0, %for.j.header ], [ %iv.k.next, %for.body ]
  %ptr = getelementptr inbounds nuw [128 x [128 x [128 x i32]]], ptr @a, i64 %iv.k, i64 %iv.j, i64 %iv.i
  %val = load i32, ptr %ptr, align 4
  %inc = add nuw nsw i32 %val, 1
  store i32 %inc, ptr %ptr, align 4
  %iv.k.next = add nuw nsw i64 %iv.k, 1
  %exitcond.k = icmp eq i64 %iv.k.next, 128
  br i1 %exitcond.k, label %for.j.cleanup, label %for.body

for.j.cleanup:
  %iv.j.next = add nuw nsw i64 %iv.j, 1
  %exitcond.j = icmp eq i64 %iv.j.next, 128
  br i1 %exitcond.j, label %for.i.cleanup, label %for.j.header, !llvm.loop !0

for.i.cleanup:
  %iv.i.next = add nuw nsw i64 %iv.i, 1
  %exitcond.i = icmp eq i64 %iv.i.next, 128
  br i1 %exitcond.i, label %exit, label %for.i.header, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.interchange.enable", i1 true}
