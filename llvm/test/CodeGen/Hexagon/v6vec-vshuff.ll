; RUN: llc -march=hexagon -disable-hexagon-shuffle=1 -O2 < %s | FileCheck %s
; Generate vshuff with 3rd param as an Rt8.
; v1:0=vshuff(v0,v1,r7)
; CHECK: vshuff(v{{[0-9]+}},v{{[0-9]+}},r{{[0-7]}})

target triple = "hexagon"

@g0 = common global [15 x <16 x i32>] zeroinitializer, align 64
@g1 = common global <32 x i32> zeroinitializer, align 128

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = load <16 x i32>, ptr @g0, align 64, !tbaa !0
  %v1 = load <16 x i32>, ptr getelementptr inbounds ([15 x <16 x i32>], ptr @g0, i32 0, i32 1), align 64, !tbaa !0
  %v2 = tail call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> %v0, <16 x i32> %v1, i32 -2147483648)
  store <32 x i32> %v2, ptr @g1, align 128, !tbaa !0
  %v3 = tail call i32 @f1(i32 1024, ptr @g1) #0
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32>, <16 x i32>, i32) #1

declare i32 @f1(...) #0

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
