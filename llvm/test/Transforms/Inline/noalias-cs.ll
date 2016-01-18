; RUN: opt -inline -enable-noalias-to-md-conversion -S < %s | FileCheck %s
target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @foo2(float* nocapture %a, float* nocapture %b, float* nocapture readonly %c) #0 {
entry:
  %0 = load float, float* %c, align 4, !noalias !3
  %arrayidx.i = getelementptr inbounds float, float* %a, i64 5
  store float %0, float* %arrayidx.i, align 4, !alias.scope !7, !noalias !8
  %arrayidx1.i = getelementptr inbounds float, float* %b, i64 8
  store float %0, float* %arrayidx1.i, align 4, !alias.scope !8, !noalias !7
  %1 = load float, float* %c, align 4
  %arrayidx = getelementptr inbounds float, float* %a, i64 7
  store float %1, float* %arrayidx, align 4
  ret void
}

define void @foo(float* nocapture %a, float* nocapture %b, float* nocapture readonly %c) #0 {
entry:
  call void @foo2(float* %a, float* %b, float* %c), !noalias !0
  call void @foo2(float* %b, float* %b, float* %a), !alias.scope !0
  ret void
}

; CHECK: define void @foo(float* nocapture %a, float* nocapture %b, float* nocapture readonly %c) #0 {
; CHECK: entry:
; CHECK:   %0 = load float, float* %c, align 4, !noalias !6
; CHECK:   %arrayidx.i.i = getelementptr inbounds float, float* %a, i64 5
; CHECK:   store float %0, float* %arrayidx.i.i, align 4, !alias.scope !12, !noalias !13
; CHECK:   %arrayidx1.i.i = getelementptr inbounds float, float* %b, i64 8
; CHECK:   store float %0, float* %arrayidx1.i.i, align 4, !alias.scope !14, !noalias !15
; CHECK:   %1 = load float, float* %c, align 4, !noalias !16
; CHECK:   %arrayidx.i = getelementptr inbounds float, float* %a, i64 7
; CHECK:   store float %1, float* %arrayidx.i, align 4, !noalias !16
; CHECK:   %2 = load float, float* %a, align 4, !alias.scope !16, !noalias !17
; CHECK:   %arrayidx.i.i.1 = getelementptr inbounds float, float* %b, i64 5
; CHECK:   store float %2, float* %arrayidx.i.i.1, align 4, !alias.scope !21, !noalias !22
; CHECK:   %arrayidx1.i.i.2 = getelementptr inbounds float, float* %b, i64 8
; CHECK:   store float %2, float* %arrayidx1.i.i.2, align 4, !alias.scope !23, !noalias !24
; CHECK:   %3 = load float, float* %a, align 4, !alias.scope !16
; CHECK:   %arrayidx.i.3 = getelementptr inbounds float, float* %b, i64 7
; CHECK:   store float %3, float* %arrayidx.i.3, align 4, !alias.scope !16
; CHECK:   ret void
; CHECK: }

attributes #0 = { nounwind uwtable }

!0 = !{!1}
!1 = distinct !{!1, !2, !"hello: %a"}
!2 = distinct !{!2, !"hello"}
!3 = !{!4, !6}
!4 = distinct !{!4, !5, !"hello2: %a"}
!5 = distinct !{!5, !"hello2"}
!6 = distinct !{!6, !5, !"hello2: %b"}
!7 = !{!4}
!8 = !{!6}

; CHECK: !0 = !{!1, !3}
; CHECK: !1 = distinct !{!1, !2, !"hello2: %a"}
; CHECK: !2 = distinct !{!2, !"hello2"}
; CHECK: !3 = distinct !{!3, !2, !"hello2: %b"}
; CHECK: !4 = !{!1}
; CHECK: !5 = !{!3}
; CHECK: !6 = !{!7, !9, !10}
; CHECK: !7 = distinct !{!7, !8, !"hello2: %a"}
; CHECK: !8 = distinct !{!8, !"hello2"}
; CHECK: !9 = distinct !{!9, !8, !"hello2: %b"}
; CHECK: !10 = distinct !{!10, !11, !"hello: %a"}
; CHECK: !11 = distinct !{!11, !"hello"}
; CHECK: !12 = !{!7}
; CHECK: !13 = !{!9, !10}
; CHECK: !14 = !{!9}
; CHECK: !15 = !{!7, !10}
; CHECK: !16 = !{!10}
; CHECK: !17 = !{!18, !20}
; CHECK: !18 = distinct !{!18, !19, !"hello2: %a"}
; CHECK: !19 = distinct !{!19, !"hello2"}
; CHECK: !20 = distinct !{!20, !19, !"hello2: %b"}
; CHECK: !21 = !{!18, !10}
; CHECK: !22 = !{!20}
; CHECK: !23 = !{!20, !10}
; CHECK: !24 = !{!18}

