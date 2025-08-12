; RUN: llc -mtriple=hexagon -O2 < %s -pipeliner-experimental-cg=true -verify-machineinstrs | FileCheck %s

; We do not pipeline sigma yet, but the non-pipelined version
; with good scheduling is pretty fast. The compiler generates
; 18 packets, and the assembly version is 16.

; CHECK:  loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK-COUNT-17: }
; CHECK: }{{[ \t]*}}:endloop

@g0 = external constant [10 x i16], align 128

declare i32 @llvm.hexagon.S2.vsplatrb(i32) #0
declare <16 x i32> @llvm.hexagon.V6.vshuffh(<16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #0
declare <16 x i32> @llvm.hexagon.V6.vd0() #0
declare <32 x i32> @llvm.hexagon.V6.vsububh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32>, <32 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vlutvwh(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyhvsrs(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32>, <16 x i32>) #0

define void @f0(ptr nocapture readonly %a0, i32 %a1, i32 %a2, i32 %a3, i8 zeroext %a4, ptr nocapture %a5) #1 {
b0:
  %v0 = add nsw i32 %a3, -1
  %v1 = icmp sgt i32 %v0, 1
  br i1 %v1, label %b1, label %b8

b1:                                               ; preds = %b0
  %v2 = mul i32 %a1, 2
  %v3 = load <16 x i32>, ptr @g0, align 128
  %v4 = tail call <16 x i32> @llvm.hexagon.V6.vshuffh(<16 x i32> %v3) #2
  %v5 = zext i8 %a4 to i32
  %v6 = tail call i32 @llvm.hexagon.S2.vsplatrb(i32 %v5) #2
  %v7 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 %v6) #2
  %v8 = tail call <16 x i32> @llvm.hexagon.V6.vd0() #2
  %v9 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 16843009) #2
  %v10 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 33686018) #2
  %v11 = icmp sgt i32 %a2, 64
  %v12 = add i32 %a1, 64
  %v13 = add i32 %v12, %a1
  %v14 = icmp sgt i32 %a2, 0
  %v15 = add i32 %a3, -2
  %v17 = load <16 x i32>, ptr %a0, align 64
  br label %b2

b2:                                               ; preds = %b7, %b1
  %v18 = phi <16 x i32> [ %v17, %b1 ], [ %v28, %b7 ]
  %v19 = phi ptr [ %a0, %b1 ], [ %v23, %b7 ]
  %v20 = phi ptr [ %a5, %b1 ], [ %v22, %b7 ]
  %v21 = phi i32 [ 1, %b1 ], [ %v118, %b7 ]
  %v22 = getelementptr inbounds i8, ptr %v20, i32 %a1
  %v23 = getelementptr inbounds i8, ptr %v19, i32 %a1
  %v25 = getelementptr inbounds i8, ptr %v19, i32 %v2
  %v28 = load <16 x i32>, ptr %v23, align 64
  %v29 = load <16 x i32>, ptr %v25, align 64
  br i1 %v11, label %b3, label %b4

b3:                                               ; preds = %b2
  %v30 = getelementptr inbounds i8, ptr %v19, i32 64
  %v31 = getelementptr inbounds i8, ptr %v19, i32 %v12
  %v33 = getelementptr inbounds i8, ptr %v19, i32 %v13
  br label %b5

b4:                                               ; preds = %b2
  br i1 %v14, label %b5, label %b7

b5:                                               ; preds = %b4, %b3
  %v35 = phi ptr [ %v25, %b4 ], [ %v33, %b3 ]
  %v36 = phi ptr [ %v23, %b4 ], [ %v31, %b3 ]
  %v37 = phi ptr [ %v19, %b4 ], [ %v30, %b3 ]
  br label %b6

b6:                                               ; preds = %b6, %b5
  %v39 = phi ptr [ %v108, %b6 ], [ %v22, %b5 ]
  %v40 = phi ptr [ %v115, %b6 ], [ %v35, %b5 ]
  %v41 = phi ptr [ %v114, %b6 ], [ %v36, %b5 ]
  %v42 = phi ptr [ %v113, %b6 ], [ %v37, %b5 ]
  %v43 = phi i32 [ %v116, %b6 ], [ %a2, %b5 ]
  %v44 = phi <16 x i32> [ %v45, %b6 ], [ %v8, %b5 ]
  %v45 = phi <16 x i32> [ %v50, %b6 ], [ %v18, %b5 ]
  %v46 = phi <16 x i32> [ %v47, %b6 ], [ %v8, %b5 ]
  %v47 = phi <16 x i32> [ %v51, %b6 ], [ %v28, %b5 ]
  %v48 = phi <16 x i32> [ %v49, %b6 ], [ %v8, %b5 ]
  %v49 = phi <16 x i32> [ %v52, %b6 ], [ %v29, %b5 ]
  %v50 = load <16 x i32>, ptr %v42, align 64
  %v51 = load <16 x i32>, ptr %v41, align 64
  %v52 = load <16 x i32>, ptr %v40, align 64
  %v53 = tail call <32 x i32> @llvm.hexagon.V6.vsububh(<16 x i32> %v8, <16 x i32> %v47) #2
  %v54 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v45, <16 x i32> %v47) #2
  %v55 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v49, <16 x i32> %v47) #2
  %v56 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v54, <16 x i32> %v7) #2
  %v57 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v55, <16 x i32> %v7) #2
  %v58 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> %v56, <16 x i32> %v9, <16 x i32> %v10) #2
  %v59 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v57, <16 x i32> %v58, <16 x i32> %v9) #2
  %v60 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> %v56, <16 x i32> %v8, <16 x i32> %v45) #2
  %v61 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> %v57, <16 x i32> %v8, <16 x i32> %v49) #2
  %v62 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v61, <16 x i32> %v60) #2
  %v63 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v53, <32 x i32> %v62, i32 -1) #2
  %v64 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v45, <16 x i32> %v44, i32 1) #2
  %v65 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v49, <16 x i32> %v48, i32 1) #2
  %v66 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v64, <16 x i32> %v47) #2
  %v67 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v65, <16 x i32> %v47) #2
  %v68 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v66, <16 x i32> %v7) #2
  %v69 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v67, <16 x i32> %v7) #2
  %v70 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v68, <16 x i32> %v59, <16 x i32> %v9) #2
  %v71 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v69, <16 x i32> %v70, <16 x i32> %v9) #2
  %v72 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> %v68, <16 x i32> %v8, <16 x i32> %v64) #2
  %v73 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> %v69, <16 x i32> %v8, <16 x i32> %v65) #2
  %v74 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v73, <16 x i32> %v72) #2
  %v75 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v63, <32 x i32> %v74, i32 -1) #2
  %v76 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v50, <16 x i32> %v45, i32 1) #2
  %v77 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v52, <16 x i32> %v49, i32 1) #2
  %v78 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v76, <16 x i32> %v47) #2
  %v79 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v77, <16 x i32> %v47) #2
  %v80 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v78, <16 x i32> %v7) #2
  %v81 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v79, <16 x i32> %v7) #2
  %v82 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v80, <16 x i32> %v71, <16 x i32> %v9) #2
  %v83 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v81, <16 x i32> %v82, <16 x i32> %v9) #2
  %v84 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> %v80, <16 x i32> %v8, <16 x i32> %v76) #2
  %v85 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> %v81, <16 x i32> %v8, <16 x i32> %v77) #2
  %v86 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v85, <16 x i32> %v84) #2
  %v87 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v75, <32 x i32> %v86, i32 -1) #2
  %v88 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v47, <16 x i32> %v46, i32 1) #2
  %v89 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v51, <16 x i32> %v47, i32 1) #2
  %v90 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v88, <16 x i32> %v47) #2
  %v91 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v89, <16 x i32> %v47) #2
  %v92 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v90, <16 x i32> %v7) #2
  %v93 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v91, <16 x i32> %v7) #2
  %v94 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v92, <16 x i32> %v83, <16 x i32> %v9) #2
  %v95 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v93, <16 x i32> %v94, <16 x i32> %v9) #2
  %v96 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> %v92, <16 x i32> %v8, <16 x i32> %v88) #2
  %v97 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> %v93, <16 x i32> %v8, <16 x i32> %v89) #2
  %v98 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v97, <16 x i32> %v96) #2
  %v99 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v87, <32 x i32> %v98, i32 -1) #2
  %v100 = tail call <32 x i32> @llvm.hexagon.V6.vlutvwh(<16 x i32> %v95, <16 x i32> %v4, i32 0) #2
  %v101 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v99) #2
  %v102 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v100) #2
  %v103 = tail call <16 x i32> @llvm.hexagon.V6.vmpyhvsrs(<16 x i32> %v101, <16 x i32> %v102) #2
  %v104 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v99) #2
  %v105 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v100) #2
  %v106 = tail call <16 x i32> @llvm.hexagon.V6.vmpyhvsrs(<16 x i32> %v104, <16 x i32> %v105) #2
  %v107 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %v106, <16 x i32> %v103) #2
  %v108 = getelementptr inbounds <16 x i32>, ptr %v39, i32 1
  store <16 x i32> %v107, ptr %v39, align 64
  %v109 = icmp sgt i32 %v43, 128
  %v110 = getelementptr inbounds <16 x i32>, ptr %v42, i32 1
  %v111 = getelementptr inbounds <16 x i32>, ptr %v41, i32 1
  %v112 = getelementptr inbounds <16 x i32>, ptr %v40, i32 1
  %v113 = select i1 %v109, ptr %v110, ptr %v42
  %v114 = select i1 %v109, ptr %v111, ptr %v41
  %v115 = select i1 %v109, ptr %v112, ptr %v40
  %v116 = add nsw i32 %v43, -64
  %v117 = icmp sgt i32 %v43, 64
  br i1 %v117, label %b6, label %b7

b7:                                               ; preds = %b6, %b4
  %v118 = add nuw nsw i32 %v21, 1
  %v119 = icmp eq i32 %v21, %v15
  br i1 %v119, label %b8, label %b2

b8:                                               ; preds = %b7, %b0
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv62" "target-features"="+hvx,+hvx-length64b" }
attributes #2 = { nounwind }
