; RUN: llc -march=hexagon -O3 < %s | FileCheck %s

; Test that unaligned load is enabled for 128B
; CHECK-NOT: r{{[0-9]+}} = memw

; Function Attrs: nounwind
define void @f0(i8* noalias nocapture readonly %a0, i16* nocapture %a1) #0 {
b0:
  %v0 = bitcast i8* %a0 to <32 x i32>*
  %v1 = load <32 x i32>, <32 x i32>* %v0, align 4, !tbaa !0
  %v2 = tail call <32 x i32> @llvm.hexagon.V6.vrmpyub.128B(<32 x i32> %v1, i32 16843009)
  %v3 = bitcast i16* %a1 to <32 x i32>*
  store <32 x i32> %v2, <32 x i32>* %v3, align 128, !tbaa !0
  ret void
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vrmpyub.128B(<32 x i32>, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
