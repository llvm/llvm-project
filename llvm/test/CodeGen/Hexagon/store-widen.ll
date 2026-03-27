; RUN: llc -mtriple=hexagon < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define void @foo(ptr nocapture %a) nounwind {
entry:
; There should be a memw, not memh.
; CHECK: memw
  ; Cheated on the alignment, just to trigger the widening.
  store i16 0, ptr %a, align 4, !tbaa !0
  %arrayidx1 = getelementptr inbounds i16, ptr %a, i32 1
  store i16 0, ptr %arrayidx1, align 2, !tbaa !0
  ret void
}

!0 = !{!"short", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
