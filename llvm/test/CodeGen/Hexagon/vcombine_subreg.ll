; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

target triple = "hexagon"

@g0 = common global <16 x i32> zeroinitializer, align 64
@g1 = common global <32 x i32> zeroinitializer, align 128
@g2 = common global <32 x i32> zeroinitializer, align 128

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  %v0 = load <16 x i32>, ptr @g0, align 64
  %v1 = load <32 x i32>, ptr @g1, align 128
  %v2 = call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v1)
  %v3 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v0, <16 x i32> %v2)
  store <32 x i32> %v3, ptr @g2, align 128
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
