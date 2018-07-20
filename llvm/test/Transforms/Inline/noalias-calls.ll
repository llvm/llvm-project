; RUN: opt -basicaa -inline -enable-noalias-to-md-conversion -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) #0
declare void @hey() #0

define void @hello(i8* noalias nocapture %a, i8* noalias nocapture readonly %c, i8* nocapture %b) #1 {
entry:
  %l = alloca i8, i32 512, align 1
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %a, i8* align 16 %b, i64 16, i1 0)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %b, i8* align 16 %c, i64 16, i1 0)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %a, i8* align 16 %c, i64 16, i1 0)
  call void @hey()
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %l, i8* align 16 %c, i64 16, i1 0)
  ret void
}

define void @foo(i8* nocapture %a, i8* nocapture readonly %c, i8* nocapture %b) #2 {
entry:
  tail call void @hello(i8* %a, i8* %c, i8* %b)
  ret void
}

; CHECK: define void @foo(i8* nocapture %a, i8* nocapture readonly %c, i8* nocapture %b) #2 {
; CHECK: entry:
; CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %a, i8* align 16 %b, i64 16, i1 false) #1, !noalias !0
; CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %b, i8* align 16 %c, i64 16, i1 false) #1, !noalias !3
; CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %a, i8* align 16 %c, i64 16, i1 false) #1, !alias.scope !5
; CHECK:   call void @hey() #1, !noalias !5
; CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %{{.*}}, i8* align 16 %c, i64 16, i1 false) #1, !noalias !3
; CHECK:   ret void
; CHECK: }

attributes #0 = { nounwind argmemonly }
attributes #1 = { nounwind }
attributes #2 = { nounwind uwtable }

; CHECK: !0 = !{!1}
; CHECK: !1 = distinct !{!1, !2, !"hello: %c"}
; CHECK: !2 = distinct !{!2, !"hello"}
; CHECK: !3 = !{!4}
; CHECK: !4 = distinct !{!4, !2, !"hello: %a"}
; CHECK: !5 = !{!4, !1}

