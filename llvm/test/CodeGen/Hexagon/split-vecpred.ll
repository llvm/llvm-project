; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Test that the splitVecPredRegs pass in the Hexagon Peephole pass does not
; move a vector predicate definition illegally, which ends up causing an assert
; later. The assert occurs because there is a use of a register that does not
; have a correct definition.

define void @f0() local_unnamed_addr #0 {
b0:
  br label %b1

b1:                                               ; preds = %b0
  br i1 undef, label %b2, label %b3

b2:                                               ; preds = %b1
  unreachable

b3:                                               ; preds = %b1
  br label %b4

b4:                                               ; preds = %b3
  br label %b5

b5:                                               ; preds = %b4
  br i1 undef, label %b13, label %b6

b6:                                               ; preds = %b5
  br label %b7

b7:                                               ; preds = %b6
  br label %b8

b8:                                               ; preds = %b7
  %v0 = tail call <512 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> undef, i32 -1)
  br i1 undef, label %b9, label %b11

b9:                                               ; preds = %b8
  br label %b12

b10:                                              ; preds = %b12
  br label %b11

b11:                                              ; preds = %b10, %b8
  %v1 = phi <512 x i1> [ %v0, %b8 ], [ undef, %b10 ]
  %v2 = tail call <512 x i1> @llvm.hexagon.V6.pred.and(<512 x i1> %v1, <512 x i1> undef)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vaddbq(<512 x i1> %v2, <16 x i32> undef, <16 x i32> undef)
  %v4 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> undef, <16 x i32> %v3, i32 undef)
  %v5 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v4, <16 x i32> undef, i32 undef)
  %v6 = tail call <16 x i32> @llvm.hexagon.V6.vand(<16 x i32> %v5, <16 x i32> undef)
  %v7 = tail call <16 x i32> @llvm.hexagon.V6.vor(<16 x i32> %v6, <16 x i32> undef)
  %v8 = tail call <16 x i32> @llvm.hexagon.V6.vsatwh(<16 x i32> %v7, <16 x i32> undef)
  %v9 = tail call <32 x i32> @llvm.hexagon.V6.vshufoeb(<16 x i32> undef, <16 x i32> %v8)
  %v10 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v9)
  %v11 = tail call <16 x i32> @llvm.hexagon.V6.vor(<16 x i32> %v10, <16 x i32> undef)
  %v12 = tail call <512 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %v11, i32 -1)
  %v13 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<512 x i1> %v12, i32 undef)
  tail call void @llvm.hexagon.V6.vmaskedstoreq(<512 x i1> undef, i8* undef, <16 x i32> %v13)
  unreachable

b12:                                              ; preds = %b12, %b9
  %v14 = phi i32 [ %v15, %b12 ], [ 0, %b9 ]
  %v15 = add nuw nsw i32 %v14, 1
  %v16 = icmp slt i32 %v15, undef
  br i1 %v16, label %b12, label %b10

b13:                                              ; preds = %b5
  ret void
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.pred.and(<512 x i1>, <512 x i1>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vandqrt(<512 x i1>, i32) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vmaskedstoreq(<512 x i1>, i8*, <16 x i32>) #2

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddbq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vor(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vand(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsatwh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vshufoeb(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
attributes #2 = { argmemonly nounwind }
