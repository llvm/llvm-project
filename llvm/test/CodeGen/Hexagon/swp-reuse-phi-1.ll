; RUN: llc  -march=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v0 = phi i32 [ 0, %b1 ], [ %v9, %b2 ]
  %v1 = phi <16 x i32> [ undef, %b1 ], [ %v2, %b2 ]
  %v2 = phi <16 x i32> [ undef, %b1 ], [ %v4, %b2 ]
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v2, <16 x i32> %v1, i32 62)
  %v4 = tail call <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32> undef, <16 x i32> zeroinitializer)
  %v5 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v4, <16 x i32> %v2, i32 2)
  %v6 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> undef, <16 x i32> %v3)
  %v7 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v6, <16 x i32> %v5)
  %v8 = tail call <16 x i32> @llvm.hexagon.V6.vabsh(<16 x i32> %v7)
  store <16 x i32> %v8, ptr undef, align 64
  %v9 = add nsw i32 %v0, 1
  %v10 = icmp slt i32 %v9, undef
  br i1 %v10, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vabsh(<16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
