; RUN: opt < %s -aa-pipeline=basic-aa,scoped-noalias-aa -passes=aa-eval -evaluate-aa-metadata -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

define void @foo1(ptr nocapture %a, ptr nocapture readonly %c) {
entry:
; CHECK-LABEL: Function: foo1
  %0 = load float, ptr %c, align 4, !alias.scope !3
  %arrayidx.i = getelementptr inbounds float, ptr %a, i64 5
  store float %0, ptr %arrayidx.i, align 4, !alias.scope !4

; CHECK: NoAlias:   %0 = load float, ptr %c, align 4, !alias.scope !0 <->   store float %0, ptr %arrayidx.i, align 4, !alias.scope !3
  ret void
}

define void @foo2(ptr nocapture %a, ptr nocapture readonly %c) {
entry:
; CHECK-LABEL: Function: foo2
  %0 = load float, ptr %c, align 4, !alias.scope !6
  %arrayidx.i = getelementptr inbounds float, ptr %a, i64 5
  store float %0, ptr %arrayidx.i, align 4, !alias.scope !3
  %arrayidx.i2 = getelementptr inbounds float, ptr %a, i64 15
  store float %0, ptr %arrayidx.i2, align 4, !alias.scope !5

; CHECK: MayAlias:   %0 = load float, ptr %c, align 4, !alias.scope !0 <->   store float %0, ptr %arrayidx.i, align 4, !alias.scope !4
; CHECK: NoAlias:   %0 = load float, ptr %c, align 4, !alias.scope !0 <->   store float %0, ptr %arrayidx.i2, align 4, !alias.scope !5
; CHECK: NoAlias:   store float %0, ptr %arrayidx.i2, align 4, !alias.scope !5 <->   store float %0, ptr %arrayidx.i, align 4, !alias.scope !4
  ret void
}

define void @foo3(ptr nocapture %a, ptr nocapture readonly %c) {
entry:
; CHECK-LABEL: Function: foo3
  %0 = load float, ptr %c, align 4, !alias.scope !9
  %arrayidx.i = getelementptr inbounds float, ptr %a, i64 5
  store float %0, ptr %arrayidx.i, align 4, !alias.scope !10

; CHECK: MayAlias:   %0 = load float, ptr %c, align 4, !alias.scope !0 <->   store float %0, ptr %arrayidx.i, align 4, !alias.scope !3
  ret void
}

define void @foo4(ptr nocapture %a, ptr nocapture readonly %c) {
entry:
; CHECK-LABEL: Function: foo4
  %0 = load float, ptr %c, align 4, !alias.scope !3
  %arrayidx.i = getelementptr inbounds float, ptr %a, i64 5
  store float %0, ptr %arrayidx.i, align 4, !alias.scope !13

; CHECK: MayAlias:   %0 = load float, ptr %c, align 4, !alias.scope !0 <->   store float %0, ptr %arrayidx.i, align 4, !alias.scope !3
  ret void
}

!0 = !{!0, i1 true, !"disjoint domain"}
!1 = !{!1, !0, !"scope 1"}
!2 = !{!2, !0, !"scope 2"}
!12 = !{!12, !0, !"scope 3"}

!3 = !{!1}
!4 = !{!2}
!5 = !{!12}
!6 = !{!1, !2}

!7 = !{!7, i1 false, !"plain domain"}
!8 = !{!8, !7, !"scope 1"}
!11 = !{!11, !7, !"scope 2"}

!9 = !{!8}
!10 = !{!11}

!14 = !{!14, i1 true, !"another disjoint domain"}
!15 = !{!15, !14, !"scope 1"}

!13 = !{!15}
