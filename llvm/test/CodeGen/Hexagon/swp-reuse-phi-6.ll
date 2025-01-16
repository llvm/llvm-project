; RUN: llc -mtriple=hexagon < %s -pipeliner-experimental-cg=true | FileCheck %s

; Test that the pipeliner generates correct code when attempting to reuse
; an existing phi. This test case contains a phi that references another
; phi (the value from the previous iteration), and when there is a use that
; is schedule in a later iteration. When this occurs, the pipeliner was
; using a value from the wrong iteration.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: vlalign([[VREG1:v([0-9]+)]],[[VREG2:v([0-9]+)]],#2)
; CHECK: [[VREG2]]:{{[0-9]+}} = vcombine([[VREG1]],v{{[0-9]+}})
; CHECK: }{{[ \t]*}}:endloop0

; Function Attrs: nounwind
define void @f0(i32 %a0, i32 %a1, ptr %a2, ptr %a3) #0 {
b0:
  %v0 = shl nsw i32 %a0, 1
  %v1 = sub i32 0, %v0
  %v2 = sub i32 0, %a0
  %v3 = getelementptr inbounds i8, ptr %a2, i32 %v1
  %v4 = getelementptr inbounds i8, ptr %a2, i32 %v2
  %v5 = getelementptr inbounds i8, ptr %a2, i32 %a0
  %v6 = getelementptr inbounds i8, ptr %a2, i32 %v0
  %v7 = getelementptr inbounds i8, ptr %v6, i32 64
  %v9 = getelementptr inbounds i8, ptr %v5, i32 64
  %v11 = getelementptr inbounds i8, ptr %a2, i32 64
  %v13 = getelementptr inbounds i8, ptr %v4, i32 64
  %v15 = getelementptr inbounds i8, ptr %v3, i32 64
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v17 = phi ptr [ %v59, %b1 ], [ %a3, %b0 ]
  %v18 = phi ptr [ %v34, %b1 ], [ %v7, %b0 ]
  %v19 = phi ptr [ %v32, %b1 ], [ %v9, %b0 ]
  %v20 = phi ptr [ %v30, %b1 ], [ %v11, %b0 ]
  %v21 = phi ptr [ %v28, %b1 ], [ %v13, %b0 ]
  %v22 = phi ptr [ %v26, %b1 ], [ %v15, %b0 ]
  %v23 = phi <32 x i32> [ %v39, %b1 ], [ undef, %b0 ]
  %v24 = phi <32 x i32> [ %v23, %b1 ], [ undef, %b0 ]
  %v25 = phi i32 [ %v60, %b1 ], [ %a1, %b0 ]
  %v26 = getelementptr inbounds <16 x i32>, ptr %v22, i32 1
  %v27 = load <16 x i32>, ptr %v22, align 64
  %v28 = getelementptr inbounds <16 x i32>, ptr %v21, i32 1
  %v29 = load <16 x i32>, ptr %v21, align 64
  %v30 = getelementptr inbounds <16 x i32>, ptr %v20, i32 1
  %v31 = load <16 x i32>, ptr %v20, align 64
  %v32 = getelementptr inbounds <16 x i32>, ptr %v19, i32 1
  %v33 = load <16 x i32>, ptr %v19, align 64
  %v34 = getelementptr inbounds <16 x i32>, ptr %v18, i32 1
  %v35 = load <16 x i32>, ptr %v18, align 64
  %v36 = tail call <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32> %v27, <16 x i32> %v35) #2
  %v37 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %v36, <16 x i32> %v31, i32 101058054) #2
  %v38 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v33, <16 x i32> %v29) #2
  %v39 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v37, <32 x i32> %v38, i32 67372036) #2
  %v40 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v23) #2
  %v41 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v24) #2
  %v42 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v40, <16 x i32> %v41, i32 2) #2
  %v43 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v23) #2
  %v44 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v24) #2
  %v45 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v43, <16 x i32> %v44, i32 2) #2
  %v46 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v39) #2
  %v47 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v46, <16 x i32> %v40, i32 2) #2
  %v48 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v39) #2
  %v49 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v48, <16 x i32> %v43, i32 2) #2
  %v50 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v45, <16 x i32> %v43) #2
  %v51 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v40, <16 x i32> %v47) #2
  %v52 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v42, <16 x i32> %v47) #2
  %v53 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %v52, <16 x i32> %v40, i32 101058054) #2
  %v54 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %v53, <16 x i32> %v50, i32 67372036) #2
  %v55 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v45, <16 x i32> %v49) #2
  %v56 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %v55, <16 x i32> %v43, i32 101058054) #2
  %v57 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %v56, <16 x i32> %v51, i32 67372036) #2
  %v58 = tail call <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32> %v57, <16 x i32> %v54) #2
  %v59 = getelementptr inbounds <16 x i32>, ptr %v17, i32 1
  store <16 x i32> %v58, ptr %v17, align 64
  %v60 = add nsw i32 %v25, -64
  %v61 = icmp sgt i32 %v25, 128
  br i1 %v61, label %b1, label %b2

b2:                                               ; preds = %b1
  ret void
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #1

declare <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32>, <32 x i32>, i32) #1
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1
declare <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32>, <16 x i32>, i32) #1
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #1
declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #1
declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32>, <16 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32>, <16 x i32>, i32) #1
declare <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32>, <16 x i32>, i32) #1
declare <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv65" "target-features"="+hvxv65,+hvx-length64b" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
