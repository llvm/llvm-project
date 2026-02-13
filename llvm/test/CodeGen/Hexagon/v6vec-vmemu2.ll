; RUN: llc -mtriple=hexagon -disable-hexagon-shuffle=0 -O2 < %s | FileCheck %s

; Generate vmemu (unaligned).
; CHECK: vmemu
; CHECK: vmemu
; CHECK: vmemu
; CHECK-NOT: vmem

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(ptr nocapture readonly %a0, ptr nocapture readonly %a1, ptr nocapture %a2) #0 {
b0:
  %v1 = load <16 x i32>, ptr %a0, align 4, !tbaa !0
  %v3 = load <16 x i32>, ptr %a1, align 4, !tbaa !0
  %v4 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v1, <16 x i32> %v3)
  store <16 x i32> %v4, ptr %a2, align 4, !tbaa !0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
