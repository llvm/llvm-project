; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
;   generate vmems for W_equals_W (vassignp)
; CHECK: vmem
; CHECK: vmem
; CHECK: vmem
; CHECK: vmem

target triple = "hexagon"

@g0 = common global [15 x <32 x i32>] zeroinitializer, align 64
@g1 = common global <32 x i32> zeroinitializer, align 64

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  store i32 0, i32* %v0
  store i32 0, i32* %v1, align 4
  %v2 = load <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @g0, i32 0, i32 0), align 64
  %v3 = call <32 x i32> @llvm.hexagon.V6.vassignp(<32 x i32> %v2)
  store <32 x i32> %v3, <32 x i32>* @g1, align 64
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vassignp(<32 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
