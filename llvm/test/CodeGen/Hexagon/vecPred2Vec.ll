; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; CHECK: q{{[0-3]}} = vand(v{{[0-9]+}},r{{[0-9]+}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]+}},r{{[0-9]+}})
; CHECK: q{{[0-3]}} = and(q{{[0-3]}},q{{[0-3]}})

target triple = "hexagon"

@g0 = common global <16 x i32> zeroinitializer, align 64

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v0, i32 -1)
  %v2 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 2)
  %v3 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v2, i32 -1)
  %v4 = tail call <64 x i1> @llvm.hexagon.V6.pred.and(<64 x i1> %v1, <64 x i1> %v3)
  %v5 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v4, i32 -1)
  store <16 x i32> %v5, <16 x i32>* @g0, align 64, !tbaa !0
  ret i32 0
}

declare <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1>, i32)
declare <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32)
declare <64 x i1> @llvm.hexagon.V6.pred.and(<64 x i1>, <64 x i1>) #1
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
