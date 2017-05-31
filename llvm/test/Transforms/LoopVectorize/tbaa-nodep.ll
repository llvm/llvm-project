; RUN: opt < %s  -tbaa -basicaa -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine -simplifycfg -S | FileCheck %s
; RUN: opt < %s  -basicaa -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine -simplifycfg -S | FileCheck %s --check-prefix=CHECK-NOTBAA
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @test1(i32* nocapture %a, float* nocapture readonly %b) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %b, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4, !tbaa !0
  %conv = fptosi float %0 to i32
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  store i32 %conv, i32* %arrayidx2, align 4, !tbaa !4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1600
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 0

; TBAA partitions the accesses in this loop, so it can be vectorized without
; runtime checks.

; CHECK-LABEL: @test1
; CHECK: entry:
; CHECK-NEXT: br label %vector.body
; CHECK: vector.body:

; CHECK: load <4 x float>, <4 x float>* %{{.*}}, align 4, !tbaa
; CHECK: store <4 x i32> %{{.*}}, <4 x i32>* %{{.*}}, align 4, !tbaa

; CHECK: ret i32 0

; CHECK-NOTBAA-LABEL: @test1
; CHECK-NOTBAA: icmp uge i32*

; CHECK-NOTBAA: load <4 x float>, <4 x float>* %{{.*}}, align 4, !tbaa
; CHECK-NOTBAA: store <4 x i32> %{{.*}}, <4 x i32>* %{{.*}}, align 4, !tbaa

; CHECK-NOTBAA: ret i32 0
}

; Function Attrs: nounwind uwtable
define i32 @test2(i32* nocapture readonly %a, float* nocapture readonly %b, float* nocapture %c) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %b, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4, !tbaa !0
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx2, align 4, !tbaa !4
  %conv = sitofp i32 %1 to float
  %mul = fmul float %0, %conv
  %arrayidx4 = getelementptr inbounds float, float* %c, i64 %indvars.iv
  store float %mul, float* %arrayidx4, align 4, !tbaa !0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1600
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 0

; This test is like the first, except here there is still one runtime check
; required. Without TBAA, however, two checks are required.

; CHECK-LABEL: @test2
; CHECK: icmp uge float*
; CHECK: icmp uge float*
; CHECK-NOT: icmp uge i32*

; CHECK: load <4 x float>, <4 x float>* %{{.*}}, align 4, !tbaa
; CHECK: store <4 x float> %{{.*}}, <4 x float>* %{{.*}}, align 4, !tbaa

; CHECK: ret i32 0

; CHECK-NOTBAA-LABEL: @test2
; CHECK-NOTBAA: icmp uge float*
; CHECK-NOTBAA: icmp uge float*
; CHECK-NOTBAA-DAG: icmp uge float*
; CHECK-NOTBAA-DAG: icmp uge i32*

; CHECK-NOTBAA: load <4 x float>, <4 x float>* %{{.*}}, align 4, !tbaa
; CHECK-NOTBAA: store <4 x float> %{{.*}}, <4 x float>* %{{.*}}, align 4, !tbaa

; CHECK-NOTBAA: ret i32 0
}

attributes #0 = { nounwind uwtable }

!0 = !{!1, !1, i64 0}
!1 = !{!"float", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !2, i64 0}

