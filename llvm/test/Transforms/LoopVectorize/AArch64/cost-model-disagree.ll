; REQUIRES: asserts
; RUN: opt --passes=loop-vectorize -mtriple=aarch64 -S -debug < %s 2>&1 | FileCheck %s

; Legacy and VPlan cost models disagree about whether one of the compares is
; scalar or vector.
define i32 @scalar_vector_compare(i32 %arg) {
; CHECK-LABEL: LV: Checking a loop in 'scalar_vector_compare'
; CHECK: Cost of 1 for VF 2: EMIT vp<%7> = icmp ule vp<%6>, vp<%3>
; CHECK: Cost of 1 for VF 2: EMIT branch-on-count vp<%index.next>, vp<%2>
; CHECK: Cost of 2 for VF 4: EMIT vp<%7> = icmp ule vp<%6>, vp<%3>
; CHECK: Cost of 1 for VF 4: EMIT branch-on-count vp<%index.next>, vp<%2>
; CHECK: Cost of 4 for VF 8: EMIT vp<%7> = icmp ule vp<%6>, vp<%3>
; CHECK: Cost of 1 for VF 8: EMIT branch-on-count vp<%index.next>, vp<%2>
; CHECK: Cost of 8 for VF 16: EMIT vp<%7> = icmp ule vp<%6>, vp<%3>
; CHECK: Cost of 1 for VF 16: EMIT branch-on-count vp<%index.next>, vp<%2>
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %cmp1 = icmp ugt i64 %iv, 755
; CHECK: LV: Found an estimated cost of 1 for VF 2 For instruction:   %cmp2 = icmp eq i32 %dec, 0
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %cmp1 = icmp ugt i64 %iv, 755
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %cmp2 = icmp eq i32 %dec, 0
; CHECK: LV: Found an estimated cost of 1 for VF 8 For instruction:   %cmp1 = icmp ugt i64 %iv, 755
; CHECK: LV: Found an estimated cost of 1 for VF 8 For instruction:   %cmp2 = icmp eq i32 %dec, 0
; CHECK: LV: Found an estimated cost of 1 for VF 16 For instruction:   %cmp1 = icmp ugt i64 %iv, 755
; CHECK: LV: Found an estimated cost of 1 for VF 16 For instruction:   %cmp2 = icmp eq i32 %dec, 0
; CHECK: LV: Legacy and VPlan disagree about number of compares. Legacy Scalar=2 Vector=0, VPlan Scalar=1 Vector=1
entry:
  %and = and i32 %arg, 3
  %add = add i32 %and, 3
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %ret = phi i32 [ %add, %entry ], [ %dec, %loop ]
  %iv.next = add i64 %iv, 1
  %arrayidx = getelementptr i8, ptr null, i64 %iv.next
  store i8 0, ptr %arrayidx, align 1
  %dec = add i32 %ret, -1
  %cmp1 = icmp ugt i64 %iv, 755
  %cmp2 = icmp eq i32 %dec, 0
  %or.cond = or i1 %cmp1, %cmp2
  br i1 %or.cond, label %exit, label %loop

exit:
  ret i32 0
}

; The VPlan has an extra vector compare due to the loop being predicated.
define void @extra_vector_compare(ptr %dst) {
; CHECK-LABEL: LV: Checking a loop in 'extra_vector_compare'
; CHECK: Cost of 1 for VF 2: EMIT vp<%5> = icmp ule ir<%iv>, vp<%3>
; CHECK: Cost of 1 for VF 2: EMIT branch-on-count vp<%index.next>, vp<%2>
; CHECK: Cost of 2 for VF 4: EMIT vp<%5> = icmp ule ir<%iv>, vp<%3>
; CHECK: Cost of 1 for VF 4: EMIT branch-on-count vp<%index.next>, vp<%2>
; CHECK: Cost of 4 for VF 8: EMIT vp<%5> = icmp ule ir<%iv>, vp<%3>
; CHECK: Cost of 1 for VF 8: EMIT branch-on-count vp<%index.next>, vp<%2>
; CHECK: Cost of 8 for VF 16: EMIT vp<%5> = icmp ule ir<%iv>, vp<%3>
; CHECK: Cost of 1 for VF 16: EMIT branch-on-count vp<%index.next>, vp<%2>
; CHECK: Found an estimated cost of 1 for VF 2 For instruction:   %ec = icmp eq i64 %iv.next, 100
; CHECK: Found an estimated cost of 1 for VF 4 For instruction:   %ec = icmp eq i64 %iv.next, 100
; CHECK: Found an estimated cost of 1 for VF 8 For instruction:   %ec = icmp eq i64 %iv.next, 100
; CHECK: Found an estimated cost of 1 for VF 16 For instruction:   %ec = icmp eq i64 %iv.next, 100
; CHECK: LV: Legacy and VPlan disagree about number of compares. Legacy Scalar=1 Vector=0, VPlan Scalar=1 Vector=1
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i8, ptr %dst, i64 %iv
  store i8 0, ptr %gep, align 1
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 100
  br i1 %ec, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}
