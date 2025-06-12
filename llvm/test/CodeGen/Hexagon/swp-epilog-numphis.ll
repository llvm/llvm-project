; XFAIL: *
; Needs some fixed in the pipeliner.
; RUN: llc -mtriple=hexagon < %s -pipeliner-experimental-cg=true | FileCheck %s

; CHECK: endloop0
; CHECK: vmem
; CHECK: vmem([[REG:r([0-9]+)]]+#1) =
; CHECK: vmem([[REG]]+#0) =

define void @f0(i32 %a0) local_unnamed_addr #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ %v33, %b1 ], [ %a0, %b0 ]
  %v1 = phi ptr [ %v32, %b1 ], [ undef, %b0 ]
  %v2 = phi ptr [ %v23, %b1 ], [ undef, %b0 ]
  %v3 = phi ptr [ %v10, %b1 ], [ undef, %b0 ]
  %v4 = phi ptr [ %v8, %b1 ], [ null, %b0 ]
  %v5 = phi <32 x i32> [ %v12, %b1 ], [ undef, %b0 ]
  %v6 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v5)
  %v7 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v6, <16 x i32> undef, i32 6)
  %v8 = getelementptr inbounds <16 x i32>, ptr %v4, i32 1
  %v9 = load <16 x i32>, ptr %v4, align 64
  %v10 = getelementptr inbounds <16 x i32>, ptr %v3, i32 1
  %v11 = load <16 x i32>, ptr %v3, align 64
  %v12 = tail call <32 x i32> @llvm.hexagon.V6.vsububh(<16 x i32> %v11, <16 x i32> %v9)
  %v13 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v12)
  %v14 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v13, <16 x i32> undef)
  %v15 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v14, <16 x i32> undef, i32 4)
  %v16 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v14, <16 x i32> %v15)
  %v17 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v14, <16 x i32> undef, i32 4)
  %v18 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v16, <16 x i32> undef, i32 2)
  %v19 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> undef, <16 x i32> %v17)
  %v20 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v18, <16 x i32> %v19)
  %v21 = getelementptr inbounds <16 x i32>, ptr %v2, i32 1
  %v22 = load <16 x i32>, ptr %v2, align 64
  %v23 = getelementptr inbounds <16 x i32>, ptr %v2, i32 2
  %v24 = load <16 x i32>, ptr %v21, align 64
  %v25 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v22, <16 x i32> %v7)
  %v26 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v24, <16 x i32> undef)
  %v27 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v25, <16 x i32> %v20)
  %v28 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v26, <16 x i32> %v20)
  store <16 x i32> %v27, ptr %v2, align 64
  store <16 x i32> %v28, ptr %v21, align 64
  %v29 = tail call <16 x i32> @llvm.hexagon.V6.vmpyhsrs(<16 x i32> %v27, i32 17760527)
  %v30 = tail call <16 x i32> @llvm.hexagon.V6.vmpyhsrs(<16 x i32> %v28, i32 17760527)
  %v31 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %v30, <16 x i32> %v29)
  %v32 = getelementptr inbounds <16 x i32>, ptr %v1, i32 1
  store <16 x i32> %v31, ptr %v1, align 64
  %v33 = add nsw i32 %v0, -64
  %v34 = icmp sgt i32 %v0, 192
  br i1 %v34, label %b1, label %b2

b2:                                               ; preds = %b1
  unreachable
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vsububh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyhsrs(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv65" "target-features"="+hvxv65,+hvx-length64b" }
attributes #1 = { nounwind readnone }
