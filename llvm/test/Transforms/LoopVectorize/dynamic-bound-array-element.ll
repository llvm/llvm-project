; RUN: opt < %s -passes=loop-vectorize -enable-vectorize-loads-as-bound -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s
;
; Negative test: Loop bound loaded from an array element A[2], where A is also
; written inside the loop. The load and store use different indices, so
; pointer identity does not flag a conflict but then LoopAccessAnalysis 
; finds an unresolvable (same-base) dependence between the A[2] read
; and the A[i] write and refuses to vectorize
;
;   void foo(int *A, int *B, int *C, int *Len) {
;     for (int i = 0; i < A[2]; i++)
;       A[i] = B[i] + C[i];
;   }
;

define dso_local void @foo(ptr noundef captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef readonly captures(none) %C, ptr noundef readnone captures(none) %Len) #0 {
; CHECK-LABEL: @foo(
; CHECK-NOT:   vector.body
entry:
  %arrayidx = getelementptr inbounds nuw i8, ptr %A, i64 8
  %0 = load i32, ptr %arrayidx, align 4, !tbaa !5
  %cmp11 = icmp sgt i32 %0, 0
  br i1 %cmp11, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx1 = getelementptr inbounds nuw i32, ptr %B, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx1, align 4, !tbaa !5
  %arrayidx3 = getelementptr inbounds nuw i32, ptr %C, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx3, align 4, !tbaa !5
  %add = add nsw i32 %2, %1
  %arrayidx5 = getelementptr inbounds nuw i32, ptr %A, i64 %indvars.iv
  store i32 %add, ptr %arrayidx5, align 4, !tbaa !5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %3 = load i32, ptr %arrayidx, align 4, !tbaa !5
  %4 = sext i32 %3 to i64
  %cmp = icmp slt i64 %indvars.iv.next, %4
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !9
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable }

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 21.1.8"}
!5 = !{!6, !6, i64 0, i64 4}
!6 = !{!7, i64 4, !"int"}
!7 = !{!8, i64 1, !"omnipotent char"}
!8 = !{!"Simple C++ TBAA"}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.unroll.disable"}
