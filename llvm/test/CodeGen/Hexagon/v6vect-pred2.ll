; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; CHECK-DAG: v{{[0-9]+}} = vsplat(r{{[0-9]+}})
; CHECK-DAG: v{{[0-9]+}} = vsplat(r{{[0-9]+}})
; CHECK-DAG: q{{[0-3]}} = vand(v{{[0-9]+}},r{{[0-9]+}})
; CHECK: v{{[0-9]+}} = vmux(q{{[0-3]}},v{{[0-9]+}},v{{[0-9]+}})

target triple = "hexagon"

@g0 = common global <16 x i32> zeroinitializer, align 64
@g1 = common global <16 x i32> zeroinitializer, align 64
@g2 = common global <16 x i32> zeroinitializer, align 64
@g3 = common global <16 x i32> zeroinitializer, align 64

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 11)
  store <16 x i32> %v0, ptr @g1, align 64, !tbaa !0
  %v1 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 12)
  store <16 x i32> %v1, ptr @g2, align 64, !tbaa !0
  %v2 = load <16 x i32>, ptr @g0, align 64, !tbaa !0
  %v3 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v2, i32 -1)
  %v4 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> %v3, <16 x i32> %v0, <16 x i32> %v1)
  store <16 x i32> %v4, ptr @g3, align 64, !tbaa !0
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
