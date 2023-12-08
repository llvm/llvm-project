; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; CHECK: vmem(r{{[0-9]*}}+#1) =

target triple = "hexagon"

; Function Attrs: nounwind
define i32 @f0(ptr %a0, ptr %a1) #0 {
b0:
  %v0 = call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  store <16 x i32> %v0, ptr %a0, align 64
  %v1 = load <16 x i32>, ptr %a0, align 64
  %v2 = call <32 x i32> @llvm.hexagon.V6.vunpackh(<16 x i32> %v1)
  store <32 x i32> %v2, ptr %a1, align 64
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vunpackh(<16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
