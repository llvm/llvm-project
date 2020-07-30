; RUN: opt < %s -basic-aa -scoped-noalias -aa-eval -evaluate-aa-metadata -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=basic-aa,scoped-noalias-aa -passes=aa-eval -evaluate-aa-metadata -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo1(float* nocapture %a, float* nocapture readonly %c) #0 {
entry:
; CHECK-LABEL: Function: foo1
  %0 = load float, float* %c, align 4, !alias.scope !2
  %arrayidx.i = getelementptr inbounds float, float* %a, i64 5
  store float %0, float* %arrayidx.i, align 4, !noalias !2
  %1 = load float, float* %c, align 4
  %arrayidx = getelementptr inbounds float, float* %a, i64 7
  store float %1, float* %arrayidx, align 4
  ret void

; CHECK: NoAlias:   %0 = load float, float* %c, align 4, !alias.scope !0 <->   store float %0, float* %arrayidx.i, align 4, !noalias !0
; CHECK: MayAlias:   %0 = load float, float* %c, align 4, !alias.scope !0 <->   store float %1, float* %arrayidx, align 4
; CHECK: MayAlias:   %1 = load float, float* %c, align 4 <->   store float %0, float* %arrayidx.i, align 4, !noalias !0
; CHECK: MayAlias:   %1 = load float, float* %c, align 4 <->   store float %1, float* %arrayidx, align 4
; CHECK: NoAlias:   store float %1, float* %arrayidx, align 4 <->   store float %0, float* %arrayidx.i, align 4, !noalias !0
}

attributes #0 = { nounwind uwtable }

!0 = !{!0, !"some domain"}
!1 = !{!1, !0, !"some scope"}
!2 = !{!1}
