; RUN: llc -march=hexagon -O3 < %s | FileCheck %s
; CHECK-NOT: vsplat
; CHECK: call f2
; CHECK: v{{[0-9]+}} = vsplat
; CHECK: v{{[0-9]+}} = vsplat
; CHECK: v{{[0-9]+}} = vsplat
; CHECK: v{{[0-9]+}} = vsplat

target triple = "hexagon"

@g0 = common global [2 x <32 x i32>] zeroinitializer, align 128
@g1 = common global <32 x i32> zeroinitializer, align 128
@g2 = common global [2 x <16 x i32>] zeroinitializer, align 64

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  tail call void @f1() #2
  %v0 = tail call i32 @f2(i8 zeroext 0) #2
  %v1 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1) #2
  store <16 x i32> %v1, ptr @g2, align 64, !tbaa !0
  %v2 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 2) #2
  store <16 x i32> %v2, ptr getelementptr inbounds ([2 x <16 x i32>], ptr @g2, i32 0, i32 1), align 64, !tbaa !0
  %v3 = tail call <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32> %v1, <16 x i32> %v2) #2
  store <32 x i32> %v3, ptr @g0, align 128, !tbaa !0
  store <32 x i32> %v3, ptr getelementptr inbounds ([2 x <32 x i32>], ptr @g0, i32 0, i32 1), align 128, !tbaa !0
  %v4 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v3, <32 x i32> %v3, i32 -2147483648)
  store <32 x i32> %v4, ptr @g1, align 128, !tbaa !0
  ret i32 0
}

declare void @f1() #0

declare i32 @f2(i8 zeroext) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind
define void @f3() #0 {
b0:
  %v0 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  store <16 x i32> %v0, ptr @g2, align 64, !tbaa !0
  %v1 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 2)
  store <16 x i32> %v1, ptr getelementptr inbounds ([2 x <16 x i32>], ptr @g2, i32 0, i32 1), align 64, !tbaa !0
  %v2 = tail call <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32> %v0, <16 x i32> %v1)
  store <32 x i32> %v2, ptr @g0, align 128, !tbaa !0
  store <32 x i32> %v2, ptr getelementptr inbounds ([2 x <32 x i32>], ptr @g0, i32 0, i32 1), align 128, !tbaa !0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
