; RUN: llc -mtriple=hexagon -verify-machineinstrs < %s
; REQUIRES: asserts

; Test that we constrain the new register operands for instructions
; to be the same as the register class of the original instruction.
; In this case, the register class of a valign scalar operand changed
; from IntRegsLow8 to IntRegs, which is incorrect.

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b6

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b4, %b1
  %v0 = phi <16 x i32> [ undef, %b1 ], [ %v17, %b4 ]
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v1 = phi i32 [ 0, %b2 ], [ %v19, %b3 ]
  %v2 = phi i32 [ undef, %b2 ], [ %v18, %b3 ]
  %v3 = phi <16 x i32> [ %v0, %b2 ], [ %v17, %b3 ]
  %v4 = tail call i32 @llvm.hexagon.A2.combine.ll(i32 0, i32 0)
  %v5 = tail call <16 x i32> @llvm.hexagon.V6.vlalignb(<16 x i32> undef, <16 x i32> undef, i32 %v2)
  %v6 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffuh(<16 x i32> %v5, <16 x i32> undef)
  %v7 = tail call <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32> %v6, <16 x i32> %v6)
  %v8 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v7)
  %v9 = tail call <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32> %v8, i32 17)
  %v10 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v9, i32 151587081)
  %v11 = tail call <16 x i32> @llvm.hexagon.V6.vsatwh(<16 x i32> %v10, <16 x i32> undef)
  %v12 = tail call <16 x i32> @llvm.hexagon.V6.vsubuhsat(<16 x i32> undef, <16 x i32> %v11)
  %v13 = tail call <16 x i32> @llvm.hexagon.V6.vmaxh(<16 x i32> %v12, <16 x i32> undef)
  %v14 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwh(<16 x i32> %v13, i32 %v4)
  %v15 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwh(<16 x i32> undef, i32 %v4)
  %v16 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v14, <16 x i32> %v15)
  %v17 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v3, <16 x i32> %v16)
  %v18 = add nsw i32 %v2, -2
  %v19 = add nsw i32 %v1, 1
  %v20 = icmp eq i32 %v19, 2
  br i1 %v20, label %b4, label %b3

b4:                                               ; preds = %b3
  br i1 undef, label %b5, label %b2

b5:                                               ; preds = %b4
  unreachable

b6:                                               ; preds = %b0
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.combine.ll(i32, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlalignb(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vabsdiffuh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsatwh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubuhsat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmaxh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyiwh(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
attributes #1 = { nounwind readnone }
