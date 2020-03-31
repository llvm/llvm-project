; RUN: llc -O0 -mtriple=aarch64-unknown-unknown -stop-after=irtranslator -o - %s | FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @snork() {
bb:
  %tmp1 = getelementptr i16, i16* null, i64 0
  %tmp5 = getelementptr i16, i16* null, i64 2
  %tmp6 = load i16, i16* %tmp1, align 2, !tbaa !0
  store i16 %tmp6, i16* %tmp5, align 2, !tbaa !0
  ; CHECK: [[LOAD:%[0-9]+]]:_(s16) = G_LOAD %{{[0-9]+}}(p0) :: (load 2 from %ir.tmp1, !tbaa !0)
  ; CHECK: G_STORE [[LOAD]](s16), %{{[0-9]+}}(p0) :: (store 2 into %ir.tmp5, !tbaa !0)
  ret void
}

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
