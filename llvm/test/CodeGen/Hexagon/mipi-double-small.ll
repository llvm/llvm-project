; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br label %b1

b1:                                               ; preds = %b0
  br i1 undef, label %b2, label %b3

b2:                                               ; preds = %b1
  %v0 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> undef)
  store <32 x i32> %v0, <32 x i32>* undef, align 128
  unreachable

b3:                                               ; preds = %b1
  ret void
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b" }
attributes #1 = { nounwind readnone }
