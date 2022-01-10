; RUN: llc -march=hexagon --combiner-store-merging=false < %s | FileCheck %s
; CHECK-NOT: memh
; Check that store widening does not merge the two stores.

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

%struct.type_t = type { i8, i8, [2 x i8] }

define zeroext i8 @foo(%struct.type_t* nocapture %p) nounwind {
entry:
  %a = getelementptr inbounds %struct.type_t, %struct.type_t* %p, i32 0, i32 0
  store i8 0, i8* %a, align 2, !tbaa !0
  %b = getelementptr inbounds %struct.type_t, %struct.type_t* %p, i32 0, i32 1
  %0 = load i8, i8* %b, align 1, !tbaa !0
  store i8 0, i8* %b, align 1, !tbaa !0
  ret i8 %0
}

!0 = !{!"omnipotent char", !1}
!1 = !{!"Simple C/C++ TBAA"}
