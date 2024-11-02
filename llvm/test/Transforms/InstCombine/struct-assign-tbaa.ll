; RUN: opt -passes=instcombine -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind

; Verify that instcombine preserves TBAA tags when converting a memcpy into
; a scalar load and store.

%struct.test1 = type { float }

; CHECK: @test
; CHECK: %[[LOAD:.*]] = load i32, ptr %{{.*}}, align 4, !tbaa !0
; CHECK: store i32 %[[LOAD:.*]], ptr %{{.*}}, align 4, !tbaa !0
; CHECK: ret
define void @test1(ptr nocapture %a, ptr nocapture %b) {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a, ptr align 4 %b, i64 4, i1 false), !tbaa.struct !3
  ret void
}

%struct.test2 = type { ptr }

define ptr @test2() {
; CHECK-LABEL: @test2(
; CHECK-NOT: memcpy
; CHECK: ret
  %tmp = alloca %struct.test2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %tmp, ptr align 8 undef, i64 8, i1 false), !tbaa.struct !4
  %tmp3 = load ptr, ptr %tmp
  ret ptr %tmp
}

; CHECK: !0 = !{!1, !1, i64 0}
; CHECK: !1 = !{!"float", !2}

!0 = !{!"Simple C/C++ TBAA"}
!1 = !{!"omnipotent char", !0}
!2 = !{!5, !5, i64 0}
!3 = !{i64 0, i64 4, !2}
!4 = !{i64 0, i64 8, null}
!5 = !{!"float", !0}
