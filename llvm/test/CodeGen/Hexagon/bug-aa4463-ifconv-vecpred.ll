; RUN: llc -mtriple=hexagon -O2 < %s
; REQUIRES: asserts

define inreg <16 x i32> @f0(i32 %a0, ptr nocapture %a1) #0 {
b0:
  %v0 = tail call <64 x i1> @llvm.hexagon.V6.pred.scalar2(i32 %a0)
  %v1 = tail call <64 x i1> @llvm.hexagon.V6.pred.not(<64 x i1> %v0)
  %v2 = icmp ult i32 %a0, 48
  br i1 %v2, label %b1, label %b2

b1:                                               ; preds = %b0
  %v3 = add nuw nsw i32 %a0, 16
  %v4 = tail call <64 x i1> @llvm.hexagon.V6.pred.scalar2(i32 %v3)
  %v5 = tail call <64 x i1> @llvm.hexagon.V6.pred.and(<64 x i1> %v4, <64 x i1> %v1)
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v6 = phi <64 x i1> [ %v5, %b1 ], [ %v1, %b0 ]
  %v7 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v6, i32 -1)
  %v8 = getelementptr inbounds <16 x i32>, ptr %a1, i32 1
  %v9 = load <16 x i32>, ptr %v8, align 64
  %v10 = getelementptr inbounds <16 x i32>, ptr %a1, i32 2
  %v11 = load <16 x i32>, ptr %v10, align 64
  %v12 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> %v6, <16 x i32> %v9, <16 x i32> %v11)
  store <16 x i32> %v12, ptr %a1, align 64
  ret <16 x i32> %v7
}

declare <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1>, i32) #1
declare <64 x i1> @llvm.hexagon.V6.pred.not(<64 x i1>) #1
declare <64 x i1> @llvm.hexagon.V6.pred.scalar2(i32) #1
declare <64 x i1> @llvm.hexagon.V6.pred.and(<64 x i1>, <64 x i1>) #1
declare <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1>, <16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
