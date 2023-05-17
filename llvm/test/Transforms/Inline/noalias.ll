; RUN: opt -passes=inline -enable-noalias-to-md-conversion -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @hello(ptr noalias nocapture %a, ptr nocapture readonly %c) #0 {
entry:
  %0 = load float, ptr %c, align 4
  %arrayidx = getelementptr inbounds float, ptr %a, i64 5
  store float %0, ptr %arrayidx, align 4
  ret void
}

define void @foo(ptr nocapture %a, ptr nocapture readonly %c) #0 {
entry:
  tail call void @hello(ptr %a, ptr %c)
  %0 = load float, ptr %c, align 4
  %arrayidx = getelementptr inbounds float, ptr %a, i64 7
  store float %0, ptr %arrayidx, align 4
  ret void
}

; CHECK-LABEL: define void @foo(ptr nocapture %a, ptr nocapture readonly %c) #0 {
; CHECK: entry:
; CHECK:   call void @llvm.experimental.noalias.scope.decl
; CHECK:   [[TMP0:%.+]] = load float, ptr %c, align 4, !noalias !0
; CHECK:   %arrayidx.i = getelementptr inbounds float, ptr %a, i64 5
; CHECK:   store float [[TMP0]], ptr %arrayidx.i, align 4, !alias.scope !0
; CHECK:   [[TMP1:%.+]] = load float, ptr %c, align 4
; CHECK:   %arrayidx = getelementptr inbounds float, ptr %a, i64 7
; CHECK:   store float [[TMP1]], ptr %arrayidx, align 4
; CHECK:   ret void
; CHECK: }

define void @hello2(ptr noalias nocapture %a, ptr noalias nocapture %b, ptr nocapture readonly %c) #0 {
entry:
  %0 = load float, ptr %c, align 4
  %arrayidx = getelementptr inbounds float, ptr %a, i64 5
  store float %0, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %b, i64 8
  store float %0, ptr %arrayidx1, align 4
  ret void
}

define void @foo2(ptr nocapture %a, ptr nocapture %b, ptr nocapture readonly %c) #0 {
entry:
  tail call void @hello2(ptr %a, ptr %b, ptr %c)
  %0 = load float, ptr %c, align 4
  %arrayidx = getelementptr inbounds float, ptr %a, i64 7
  store float %0, ptr %arrayidx, align 4
  ret void
}

; CHECK-LABEL: define void @foo2(ptr nocapture %a, ptr nocapture %b, ptr nocapture readonly %c) #0 {
; CHECK: entry:
; CHECK:   call void @llvm.experimental.noalias.scope.decl(metadata !3)
; CHECK:   call void @llvm.experimental.noalias.scope.decl(metadata !6)
; CHECK:   [[TMP0:%.+]] = load float, ptr %c, align 4, !noalias !8
; CHECK:   %arrayidx.i = getelementptr inbounds float, ptr %a, i64 5
; CHECK:   store float [[TMP0]], ptr %arrayidx.i, align 4, !alias.scope !3, !noalias !6
; CHECK:   %arrayidx1.i = getelementptr inbounds float, ptr %b, i64 8
; CHECK:   store float [[TMP0]], ptr %arrayidx1.i, align 4, !alias.scope !6, !noalias !3
; CHECK:   [[TMP1:%.+]] = load float, ptr %c, align 4
; CHECK:   %arrayidx = getelementptr inbounds float, ptr %a, i64 7
; CHECK:   store float [[TMP1]], ptr %arrayidx, align 4
; CHECK:   ret void
; CHECK: }

attributes #0 = { nounwind uwtable }

; CHECK: !0 = !{!1}
; CHECK: !1 = distinct !{!1, !2, !"hello: %a"}
; CHECK: !2 = distinct !{!2, !"hello"}
; CHECK: !3 = !{!4}
; CHECK: !4 = distinct !{!4, !5, !"hello2: %a"}
; CHECK: !5 = distinct !{!5, !"hello2"}
; CHECK: !6 = !{!7}
; CHECK: !7 = distinct !{!7, !5, !"hello2: %b"}
; CHECK: !8 = !{!4, !7}
