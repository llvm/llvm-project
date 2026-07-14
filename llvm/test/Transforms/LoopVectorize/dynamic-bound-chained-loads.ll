; RUN: opt < %s -passes=loop-vectorize -enable-vectorize-loads-as-bound -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s
;
; Negative test: the upper bound is **PtrLen (an indirect load).
; LoopAccessAnalysis cannot compute SCEV bounds for the indirect 
; load (load i32, ptr %ptr) because %ptr itself is loaded inside the loop. 
; The loop will be rejected.
;
; C source:
;   void foo(int *A, int *B, int *C, int **PtrLen) {
;     for (int i = 0; i < **PtrLen; i++)
;       A[i] = B[i] + C[i];
;   }

define dso_local void @foo(ptr noundef writeonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef readonly captures(none) %C, ptr noundef readonly captures(none) %PtrLen) #0 {
; CHECK-LABEL: @foo(
; CHECK-NOT:   vector.body
; CHECK-NOT:   .bound.pre
entry:
  %ptr0 = load ptr, ptr %PtrLen, align 8, !tbaa !5
  %val0 = load i32, ptr %ptr0, align 4, !tbaa !10
  %cmp9 = icmp sgt i32 %val0, 0
  br i1 %cmp9, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i32, ptr %B, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx, align 4, !tbaa !10
  %arrayidx2 = getelementptr inbounds nuw i32, ptr %C, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx2, align 4, !tbaa !10
  %add = add nsw i32 %2, %1
  %arrayidx4 = getelementptr inbounds nuw i32, ptr %A, i64 %indvars.iv
  store i32 %add, ptr %arrayidx4, align 4, !tbaa !10
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %ptr = load ptr, ptr %PtrLen, align 8, !tbaa !5
  %val = load i32, ptr %ptr, align 4, !tbaa !10
  %3 = sext i32 %val to i64
  %cmp = icmp slt i64 %indvars.iv.next, %3
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !12
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable }

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 21.1.8"}
!5 = !{!6, !6, i64 0, i64 8}
!6 = !{!7, i64 8, !"p1 int"}
!7 = !{!8, i64 8, !"any pointer"}
!8 = !{!9, i64 1, !"omnipotent char"}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !11, i64 0, i64 4}
!11 = !{!8, i64 4, !"int"}
!12 = distinct !{!12, !13, !14}
!13 = !{!"llvm.loop.mustprogress"}
!14 = !{!"llvm.loop.unroll.disable"}
