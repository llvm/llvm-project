; RUN: llc -march=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; Test that causes an assert when the phi reuse code does not set
; PhiOp2 correctly for use in the next stage. This occurs when the
; number of stages is two or more.

; Function Attrs: nounwind
define void @f0(i16* noalias nocapture %a0) #0 {
b0:
  br i1 undef, label %b1, label %b3

b1:                                               ; preds = %b0
  %v0 = bitcast i16* %a0 to <16 x i32>*
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ 0, %b1 ], [ %v15, %b2 ]
  %v2 = phi <16 x i32>* [ %v0, %b1 ], [ %v14, %b2 ]
  %v3 = phi <16 x i32>* [ undef, %b1 ], [ %v6, %b2 ]
  %v4 = phi <16 x i32> [ undef, %b1 ], [ %v10, %b2 ]
  %v5 = phi <16 x i32> [ undef, %b1 ], [ %v4, %b2 ]
  %v6 = getelementptr inbounds <16 x i32>, <16 x i32>* %v3, i32 1
  %v7 = load <16 x i32>, <16 x i32>* %v3, align 64
  %v8 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> undef, <16 x i32> %v7)
  %v9 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v4, <16 x i32> %v5, i32 62)
  %v10 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v8, <16 x i32> undef)
  %v11 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v10, <16 x i32> %v4, i32 2)
  %v12 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffh(<16 x i32> %v9, <16 x i32> %v11)
  %v13 = getelementptr inbounds <16 x i32>, <16 x i32>* %v2, i32 1
  store <16 x i32> %v12, <16 x i32>* %v2, align 64
  %v14 = getelementptr inbounds <16 x i32>, <16 x i32>* %v2, i32 2
  store <16 x i32> zeroinitializer, <16 x i32>* %v13, align 64
  %v15 = add nsw i32 %v1, 1
  %v16 = icmp slt i32 %v15, undef
  br i1 %v16, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vabsdiffh(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
