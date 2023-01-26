; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; CHECK-NOT: ##var
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

@var = external global i32

define i32 @foo() nounwind readonly {
entry:
  %0 = load i32, ptr @var, align 4, !tbaa !0
  ret i32 %0
}

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
