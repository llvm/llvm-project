; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; CHECK: allocframe(r29,#{{[1-9][0-9]*}}):raw
; CHECK: r29 = and(r29,#-64)

target triple = "hexagon"

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = alloca <16 x i32>, align 64
  %v1 = bitcast <16 x i32>* %v0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* %v1) #3
  %v2 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vsubh.rt(<16 x i32> %v2, i32 -1)
  store <16 x i32> %v3, <16 x i32>* %v0, align 64, !tbaa !0
  call void @f1(i32 64, i8* %v1) #3
  call void @llvm.lifetime.end.p0i8(i64 64, i8* %v1) #3
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubh.rt(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

declare void @f1(i32, i8*) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
