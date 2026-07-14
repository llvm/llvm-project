; RUN: opt < %s -passes=loop-vectorize -enable-vectorize-loads-as-bound -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s
;
; The bound pointer %Len is also written to inside the loop body with a computed value, 
; so the bound changes each iteration and cannot be hoisted.  The ModifiedPtrs check detects that the same
; pointer is used as both a store target and the bound load's address.
;
;   void foo(int *A, int *B, int *C, int *Len) {
;     for (int i = 0; i < *Len; i++) {
;       A[i] = B[i] + C[i];
;       *Len = B[i] + C[i];   // overwrite bound with computed value
;     }
;   }

define dso_local void @foo(ptr noundef %A, ptr noundef readonly %B, ptr noundef readonly %C, ptr noundef %Len) #0 {
; CHECK-LABEL: @foo(
; CHECK-NOT:   vector.body
; CHECK-NOT:   .bound.pre
entry:
  %len0 = load i32, ptr %Len, align 4
  %cmp0 = icmp sgt i32 %len0, 0
  br i1 %cmp0, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i32, ptr %B, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds nuw i32, ptr %C, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds nuw i32, ptr %A, i64 %indvars.iv
  store i32 %add, ptr %arrayidx4, align 4
  store i32 %add, ptr %Len, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %len = load i32, ptr %Len, align 4
  %ext = sext i32 %len to i64
  %cmp = icmp slt i64 %indvars.iv.next, %ext
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable }

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.unroll.disable"}
