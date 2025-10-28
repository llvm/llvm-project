; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: v{{[0-9]*}}.w = vadd

target triple = "hexagon"

@g0 = common global <16 x i32> zeroinitializer, align 64
@g1 = common global <16 x i32> zeroinitializer, align 64
@g2 = common global <16 x i32> zeroinitializer, align 64

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  %v0 = load <16 x i32>, ptr @g0, align 32, !tbaa !0
  %v1 = load <16 x i32>, ptr @g1, align 32, !tbaa !0
  %v2 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v0, <16 x i32> %v1)
  store <16 x i32> %v2, ptr @g2, align 64, !tbaa !0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
