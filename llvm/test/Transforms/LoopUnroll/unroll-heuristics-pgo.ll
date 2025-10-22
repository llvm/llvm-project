; RUN: opt < %s -S -passes=loop-unroll -unroll-runtime -unroll-threshold=40 -unroll-max-percent-threshold-boost=100 | FileCheck %s

@known_constant = internal unnamed_addr constant [9 x i32] [i32 0, i32 -1, i32 0, i32 -1, i32 5, i32 -1, i32 0, i32 -1, i32 0], align 16

; CHECK-LABEL: @bar_prof
; CHECK: loop:
; CHECK:   %mul = mul
; CHECK:   %mul.1 = mul
; CHECK:   %mul.2 = mul
; CHECK:   %mul.3 = mul
; CHECK:   br i1 %niter.ncmp.7, label %loop.end.unr-lcssa, label %loop, !prof [[PROF0:![0-9]+]]
; CHECK: loop.epil:
; CHECK:   br i1 %epil.iter.cmp, label %loop.epil, label %loop.end.epilog-lcssa, !prof [[PROF1:![0-9]+]], !llvm.loop {{![0-9]+}}
define i32 @bar_prof(ptr noalias nocapture readonly %src, i64 %c) !prof !1 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %r  = phi i32 [ 0, %entry ], [ %add, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %src, i64 %iv
  %src_element = load i32, ptr %arrayidx, align 4
  %array_const_idx = getelementptr inbounds [9 x i32], ptr @known_constant, i64 0, i64 %iv
  %const_array_element = load i32, ptr %array_const_idx, align 4
  %mul = mul nsw i32 %src_element, %const_array_element
  %add = add nsw i32 %mul, %r
  %inc = add nuw nsw i64 %iv, 1
  %exitcond86.i = icmp eq i64 %inc, %c
  br i1 %exitcond86.i, label %loop.end, label %loop, !prof !2

loop.end:
  %r.lcssa = phi i32 [ %r, %loop ]
  ret i32 %r.lcssa
}

; CHECK-LABEL: @bar_prof_flat
; CHECK-NOT: loop.epil
define i32 @bar_prof_flat(ptr noalias nocapture readonly %src, i64 %c) !prof !1 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %r  = phi i32 [ 0, %entry ], [ %add, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %src, i64 %iv
  %src_element = load i32, ptr %arrayidx, align 4
  %array_const_idx = getelementptr inbounds [9 x i32], ptr @known_constant, i64 0, i64 %iv
  %const_array_element = load i32, ptr %array_const_idx, align 4
  %mul = mul nsw i32 %src_element, %const_array_element
  %add = add nsw i32 %mul, %r
  %inc = add nuw nsw i64 %iv, 1
  %exitcond86.i = icmp eq i64 %inc, %c
  br i1 %exitcond86.i, label %loop, label %loop.end, !prof !2

loop.end:
  %r.lcssa = phi i32 [ %r, %loop ]
  ret i32 %r.lcssa
}

!1 = !{!"function_entry_count", i64 1}
!2 = !{!"branch_weights", i32 1, i32 1000}

; CHECK: [[PROF0]] = !{!"branch_weights", i32 1, i32 124}
; CHECK: [[PROF1]] = !{!"branch_weights", i32 3, i32 1}
