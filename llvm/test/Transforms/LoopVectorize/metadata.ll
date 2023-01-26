; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define i32 @test1(ptr nocapture %a, ptr nocapture readonly %b) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4, !tbaa !0
  %conv = fptosi float %0 to i32
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %conv, ptr %arrayidx2, align 4, !tbaa !4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1600
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 0
}

; CHECK-LABEL: @test1
; CHECK: load <4 x float>, ptr %{{.*}}, align 4, !tbaa ![[TFLT:[0-9]+]]
; CHECK: store <4 x i32> %{{.*}}, ptr %{{.*}}, align 4, !tbaa ![[TINT:[0-9]+]]
; CHECK: ret i32 0

; CHECK-DAG: ![[TFLT]] = !{![[TFLT1:[0-9]+]]
; CHECK-DAG: ![[TFLT1]] = !{!"float"

; CHECK-DAG: ![[TINT]] = !{![[TINT1:[0-9]+]]
; CHECK-DAG: ![[TINT1]] = !{!"int"

attributes #0 = { nounwind uwtable }

!0 = !{!1, !1, i64 0}
!1 = !{!"float", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !2, i64 0}

