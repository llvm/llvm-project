; RUN: llc -O2 < %s | FileCheck %s
; We should generate a MUX instruction for one of the selects.
; CHECK: mux
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

%struct.struct_t = type { i32, i32, i32 }

define void @foo(ptr nocapture %p, i32 %x, i32 %y, i32 %z) nounwind {
entry:
  %cmp = icmp slt i32 %x, 4660
  %add = add nsw i32 %x, 1
  %add.y = select i1 %cmp, i32 %add, i32 %y
  %x.add.y = select i1 %cmp, i32 %x, i32 %y
  %. = zext i1 %cmp to i32
  %b.0 = add nsw i32 %x.add.y, %z
  store i32 %add.y, ptr %p, align 4, !tbaa !0
  %b4 = getelementptr inbounds %struct.struct_t, ptr %p, i32 0, i32 1
  store i32 %b.0, ptr %b4, align 4, !tbaa !0
  %c5 = getelementptr inbounds %struct.struct_t, ptr %p, i32 0, i32 2
  store i32 %., ptr %c5, align 4, !tbaa !0
  ret void
}

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
