; RUN: opt < %s -passes=loop-vectorize -enable-vectorize-loads-as-bound -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s
;
; Negative test: the exit condition is an fcmp (not icmp). 
; SCEV cannot compute the trip count of a float-bounded loop, so the
; pass enters the dynamic-bound path.  It then rejects because the
; branch condition is an FCmpInst, not an ICmpInst.
;
;   void foo(float *A, float *B, float *Limit) {
;     for (float x = 0.0f; x < *Limit; x += 1.0f)
;       A[(int)x] = B[(int)x];
;   }

define dso_local void @foo(ptr noundef %A, ptr noundef readonly %B, ptr noundef readonly %Limit) #0 {
; CHECK-LABEL: @foo(
; CHECK-NOT:   vector.body
; CHECK-NOT:   .bound.pre
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %fiv = phi float [ 0.0, %entry ], [ %fiv.next, %for.body ]
  %gep.b = getelementptr inbounds i32, ptr %B, i64 %iv
  %b = load i32, ptr %gep.b, align 4
  %gep.a = getelementptr inbounds i32, ptr %A, i64 %iv
  store i32 %b, ptr %gep.a, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %fiv.next = fadd float %fiv, 1.0
  %lim = load float, ptr %Limit, align 4
  %cmp = fcmp olt float %fiv.next, %lim
  br i1 %cmp, label %for.body, label %for.exit, !llvm.loop !0

for.exit:
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable }

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.unroll.disable"}
