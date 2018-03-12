; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Test that the register scavenger does not fail because it can't find
; a spill slot. This occurs the offset for a spilled object is too large
; and requires another register to compute the location on the stack.

; Function Attrs: nounwind
define void @f0(i8* nocapture readonly %a0, i32 %a1, i32 %a2, i8* nocapture readonly %a3, i8* nocapture readonly %a4, i8* nocapture %a5) #0 {
b0:
  %v0 = tail call <16 x i32> @llvm.hexagon.V6.vshuffb(<16 x i32> zeroinitializer)
  br i1 undef, label %b1, label %b5

b1:                                               ; preds = %b0
  %v1 = getelementptr inbounds i8, i8* %a3, i32 31
  br label %b2

b2:                                               ; preds = %b4, %b1
  %v2 = phi <16 x i32>* [ undef, %b1 ], [ %v102, %b4 ]
  %v3 = phi i32 [ %a2, %b1 ], [ undef, %b4 ]
  %v4 = tail call <32 x i32> @llvm.hexagon.V6.vmpyh(<16 x i32> undef, i32 undef)
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v5 = phi <32 x i32> [ %v4, %b2 ], [ %v72, %b3 ]
  %v6 = phi <32 x i32> [ zeroinitializer, %b2 ], [ %v71, %b3 ]
  %v7 = phi i32 [ -4, %b2 ], [ %v73, %b3 ]
  %v8 = load <16 x i32>, <16 x i32>* undef, align 64
  %v9 = mul nsw i32 %v7, 9
  %v10 = tail call <16 x i32> @llvm.hexagon.V6.vlalignb(<16 x i32> %v8, <16 x i32> undef, i32 4)
  %v11 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> undef, <16 x i32> %v8, i32 4)
  %v12 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v10, <16 x i32> undef)
  %v13 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v11, <16 x i32> undef)
  %v14 = tail call <16 x i32> @llvm.hexagon.V6.vlutvvb(<16 x i32> %v12, <16 x i32> zeroinitializer, i32 0)
  %v15 = tail call <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32> %v14, <16 x i32> %v12, <16 x i32> zeroinitializer, i32 1)
  %v16 = tail call <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32> %v15, <16 x i32> %v12, <16 x i32> undef, i32 2)
  %v17 = tail call <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32> %v16, <16 x i32> %v12, <16 x i32> undef, i32 3)
  %v18 = tail call <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32> %v17, <16 x i32> %v12, <16 x i32> %v0, i32 4)
  %v19 = tail call <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32> %v18, <16 x i32> %v12, <16 x i32> %v0, i32 5)
  %v20 = tail call <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32> %v19, <16 x i32> %v12, <16 x i32> undef, i32 6)
  %v21 = tail call <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32> %v20, <16 x i32> %v12, <16 x i32> undef, i32 7)
  %v22 = tail call <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32> undef, <16 x i32> %v13, <16 x i32> %v0, i32 4)
  %v23 = tail call <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32> %v22, <16 x i32> %v13, <16 x i32> %v0, i32 5)
  %v24 = tail call <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32> %v23, <16 x i32> %v13, <16 x i32> undef, i32 6)
  %v25 = tail call <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32> %v24, <16 x i32> %v13, <16 x i32> undef, i32 7)
  %v26 = add nsw i32 %v9, 36
  %v27 = getelementptr inbounds i8, i8* %a3, i32 %v26
  %v28 = load i8, i8* %v27, align 1
  %v29 = zext i8 %v28 to i32
  %v30 = tail call i32 @llvm.hexagon.S2.vsplatrb(i32 %v29)
  %v31 = tail call <32 x i32> @llvm.hexagon.V6.vmpyub(<16 x i32> %v21, i32 %v30)
  %v32 = tail call <32 x i32> @llvm.hexagon.V6.vmpyub(<16 x i32> %v25, i32 %v30)
  %v33 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v31)
  %v34 = tail call <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32> undef, <16 x i32> %v33)
  %v35 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v32)
  %v36 = tail call <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32> %v35, <16 x i32> undef)
  %v37 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v36, <16 x i32> %v34)
  %v38 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v6, <32 x i32> %v37, i32 16843009)
  %v39 = tail call <32 x i32> @llvm.hexagon.V6.vmpyubv(<16 x i32> %v10, <16 x i32> %v34)
  %v40 = tail call <32 x i32> @llvm.hexagon.V6.vmpyubv(<16 x i32> %v11, <16 x i32> %v36)
  %v41 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v39)
  %v42 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v40)
  %v43 = tail call <32 x i32> @llvm.hexagon.V6.vadduhw(<16 x i32> %v41, <16 x i32> %v42)
  %v44 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v5, <32 x i32> %v43)
  %v45 = add nsw i32 %v9, 37
  %v46 = getelementptr inbounds i8, i8* %a3, i32 %v45
  %v47 = load i8, i8* %v46, align 1
  %v48 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v38, <32 x i32> undef, i32 16843009)
  %v49 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v44, <32 x i32> undef)
  %v50 = tail call <16 x i32> @llvm.hexagon.V6.vlalignb(<16 x i32> %v8, <16 x i32> undef, i32 2)
  %v51 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v48, <32 x i32> undef, i32 16843009)
  %v52 = tail call <32 x i32> @llvm.hexagon.V6.vmpyubv(<16 x i32> %v50, <16 x i32> undef)
  %v53 = tail call <32 x i32> @llvm.hexagon.V6.vmpyubv(<16 x i32> undef, <16 x i32> zeroinitializer)
  %v54 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v52)
  %v55 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v53)
  %v56 = tail call <32 x i32> @llvm.hexagon.V6.vadduhw(<16 x i32> %v54, <16 x i32> %v55)
  %v57 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v49, <32 x i32> %v56)
  %v58 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> undef, <16 x i32> %v8, i32 1)
  %v59 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v58, <16 x i32> undef)
  %v60 = tail call <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32> undef, <16 x i32> %v59, <16 x i32> undef, i32 7)
  %v61 = load i8, i8* undef, align 1
  %v62 = zext i8 %v61 to i32
  %v63 = tail call i32 @llvm.hexagon.S2.vsplatrb(i32 %v62)
  %v64 = tail call <32 x i32> @llvm.hexagon.V6.vmpyub(<16 x i32> undef, i32 %v63)
  %v65 = tail call <32 x i32> @llvm.hexagon.V6.vmpyub(<16 x i32> %v60, i32 %v63)
  %v66 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v64)
  %v67 = tail call <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32> undef, <16 x i32> %v66)
  %v68 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v65)
  %v69 = tail call <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32> undef, <16 x i32> %v68)
  %v70 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v69, <16 x i32> %v67)
  %v71 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v51, <32 x i32> %v70, i32 16843009)
  %v72 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v57, <32 x i32> undef)
  %v73 = add nsw i32 %v7, 1
  %v74 = icmp eq i32 %v73, 5
  br i1 %v74, label %b4, label %b3

b4:                                               ; preds = %b3
  %v75 = phi <32 x i32> [ %v72, %b3 ]
  %v76 = phi <32 x i32> [ %v71, %b3 ]
  %v77 = load i8, i8* %v1, align 1
  %v78 = zext i8 %v77 to i32
  %v79 = tail call i32 @llvm.hexagon.S2.vsplatrb(i32 %v78)
  %v80 = tail call <32 x i32> @llvm.hexagon.V6.vmpyub(<16 x i32> undef, i32 %v79)
  %v81 = tail call <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32> undef, <16 x i32> undef)
  %v82 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v80)
  %v83 = tail call <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32> undef, <16 x i32> %v82)
  %v84 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v83, <16 x i32> %v81)
  %v85 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> undef, <32 x i32> %v84, i32 16843009)
  %v86 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v85)
  %v87 = tail call <32 x i32> @llvm.hexagon.V6.vmpyh(<16 x i32> %v86, i32 8388736)
  %v88 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v87)
  %v89 = tail call <16 x i32> @llvm.hexagon.V6.vasrw(<16 x i32> %v88, i32 1)
  %v90 = tail call <16 x i32> @llvm.hexagon.V6.vasrw(<16 x i32> %v89, i32 1)
  %v91 = tail call <16 x i32> @llvm.hexagon.V6.vasrw(<16 x i32> %v90, i32 1)
  %v92 = tail call <16 x i32> @llvm.hexagon.V6.vasrw(<16 x i32> %v91, i32 1)
  %v93 = tail call <16 x i32> @llvm.hexagon.V6.vasrw(<16 x i32> %v92, i32 1)
  %v94 = tail call <16 x i32> @llvm.hexagon.V6.vsubwnq(<512 x i1> undef, <16 x i32> undef, <16 x i32> %v93)
  %v95 = tail call <16 x i32> @llvm.hexagon.V6.vsubwnq(<512 x i1> undef, <16 x i32> %v94, <16 x i32> undef)
  %v96 = tail call <16 x i32> @llvm.hexagon.V6.vasrw(<16 x i32> undef, i32 1)
  %v97 = tail call <512 x i1> @llvm.hexagon.V6.vgtw(<16 x i32> %v96, <16 x i32> %v95)
  %v98 = tail call <16 x i32> @llvm.hexagon.V6.vaddwnq(<512 x i1> %v97, <16 x i32> undef, <16 x i32> undef)
  %v99 = tail call <16 x i32> @llvm.hexagon.V6.vaddwnq(<512 x i1> undef, <16 x i32> undef, <16 x i32> undef)
  %v100 = tail call <16 x i32> @llvm.hexagon.V6.vshufeh(<16 x i32> %v99, <16 x i32> %v98)
  %v101 = tail call <16 x i32> @llvm.hexagon.V6.vshuffeb(<16 x i32> %v100, <16 x i32> undef)
  %v102 = getelementptr inbounds <16 x i32>, <16 x i32>* %v2, i32 1
  store <16 x i32> %v101, <16 x i32>* %v2, align 64
  %v103 = icmp sgt i32 %v3, 64
  br i1 %v103, label %b2, label %b5

b5:                                               ; preds = %b4, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshuffb(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyh(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlalignb(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlutvvb(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32>, <16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.vsplatrb(i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyub(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyubv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vadduhw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddwnq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubwnq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrw(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshuffeb(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshufeh(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv62" "target-features"="+hvxv62,+hvx-length64b" }
attributes #1 = { nounwind readnone }
