; RUN: llc -march=hexagon -O3 < %s | FileCheck %s
; CHECK: vsplat

target triple = "hexagon"

@g0 = common global [15 x <16 x i32>] zeroinitializer, align 64
@g1 = common global [15 x <32 x i32>] zeroinitializer, align 128

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  %v0 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  store <16 x i32> %v0, ptr @g0, align 64, !tbaa !0
  %v1 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 2)
  store <16 x i32> %v1, ptr getelementptr inbounds ([15 x <16 x i32>], ptr @g0, i32 0, i32 1), align 64, !tbaa !0
  %v2 = tail call <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32> %v0, <16 x i32> %v1)
  store <32 x i32> %v2, ptr @g1, align 128, !tbaa !0
  store <32 x i32> %v2, ptr getelementptr inbounds ([15 x <32 x i32>], ptr @g1, i32 0, i32 1), align 128, !tbaa !0
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 3)
  store <16 x i32> %v3, ptr getelementptr inbounds ([15 x <16 x i32>], ptr @g0, i32 0, i32 2), align 64, !tbaa !0
  %v4 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 4)
  store <16 x i32> %v4, ptr getelementptr inbounds ([15 x <16 x i32>], ptr @g0, i32 0, i32 3), align 64, !tbaa !0
  %v5 = tail call <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32> %v3, <16 x i32> %v4)
  store <32 x i32> %v5, ptr getelementptr inbounds ([15 x <32 x i32>], ptr @g1, i32 0, i32 2), align 128, !tbaa !0
  store <32 x i32> %v5, ptr getelementptr inbounds ([15 x <32 x i32>], ptr @g1, i32 0, i32 3), align 128, !tbaa !0
  %v6 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 5)
  store <16 x i32> %v6, ptr getelementptr inbounds ([15 x <16 x i32>], ptr @g0, i32 0, i32 4), align 64, !tbaa !0
  %v7 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 6)
  store <16 x i32> %v7, ptr getelementptr inbounds ([15 x <16 x i32>], ptr @g0, i32 0, i32 5), align 64, !tbaa !0
  %v8 = tail call <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32> %v6, <16 x i32> %v7)
  store <32 x i32> %v8, ptr getelementptr inbounds ([15 x <32 x i32>], ptr @g1, i32 0, i32 4), align 128, !tbaa !0
  store <32 x i32> %v8, ptr getelementptr inbounds ([15 x <32 x i32>], ptr @g1, i32 0, i32 5), align 128, !tbaa !0
  %v9 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 7)
  store <16 x i32> %v9, ptr getelementptr inbounds ([15 x <16 x i32>], ptr @g0, i32 0, i32 6), align 64, !tbaa !0
  %v10 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 8)
  store <16 x i32> %v10, ptr getelementptr inbounds ([15 x <16 x i32>], ptr @g0, i32 0, i32 7), align 64, !tbaa !0
  %v11 = tail call <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32> %v9, <16 x i32> %v10)
  store <32 x i32> %v11, ptr getelementptr inbounds ([15 x <32 x i32>], ptr @g1, i32 0, i32 6), align 128, !tbaa !0
  store <32 x i32> %v11, ptr getelementptr inbounds ([15 x <32 x i32>], ptr @g1, i32 0, i32 7), align 128, !tbaa !0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
