; RUN: llc < %s -march=x86
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@c0 = common global i8 0, align 1

define void @func() nounwind uwtable {
entry:
  %0 = load i8* @c0, align 1, !tbaa !0
  %tobool = icmp ne i8 %0, 0
  %conv = zext i1 %tobool to i8
  %storemerge = shl nuw nsw i8 %conv, %conv
  store i8 %storemerge, i8* @c0, align 1
  ret void
}

!0 = metadata !{metadata !"omnipotent char", metadata !1}
!1 = metadata !{metadata !"Simple C/C++ TBAA"}
