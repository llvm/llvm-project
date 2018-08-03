; RUN: llc -march=hexagon -O2 -mcpu=hexagonv60 -hexagon-initial-cfg-cleanup=0 --stats -o - 2>&1  < %s | FileCheck %s
; This was aborting while processing SUnits.
; REQUIRES: asserts

; CHECK: vmem

; CHECK-NOT: Number of node order issues found
; CHECK: Number of loops software pipelined
; CHECK-NOT: Number of node order issues found

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown--elf"

declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32>, <16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vdealvdd(<16 x i32>, <16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vshufeh(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32>, <16 x i32>, i32) #0

define void @f0() #1 {
b0:
  %v0 = load i16*, i16** undef, align 4
  %v1 = load i32*, i32** undef, align 4
  br label %b1

b1:                                               ; preds = %b3, %b0
  %v2 = phi i32 [ 0, %b0 ], [ %v129, %b3 ]
  %v3 = mul nuw nsw i32 %v2, 768
  %v4 = add nuw nsw i32 %v3, 32
  %v5 = add nuw nsw i32 %v3, 64
  %v6 = add nuw nsw i32 %v3, 96
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v7 = phi i32* [ %v1, %b1 ], [ %v127, %b2 ]
  %v8 = phi i16* [ %v0, %b1 ], [ %v128, %b2 ]
  %v9 = phi i32 [ 0, %b1 ], [ %v125, %b2 ]
  %v10 = mul nuw nsw i32 %v9, 32
  %v11 = bitcast i32* %v7 to <16 x i32>*
  %v12 = load <16 x i32>, <16 x i32>* %v11, align 64, !tbaa !1
  %v13 = add nuw nsw i32 %v10, 16
  %v14 = getelementptr inbounds i32, i32* %v1, i32 %v13
  %v15 = bitcast i32* %v14 to <16 x i32>*
  %v16 = load <16 x i32>, <16 x i32>* %v15, align 64, !tbaa !1
  %v17 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v16, <16 x i32> %v12)
  %v18 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v17) #2
  %v19 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v17) #2
  %v20 = tail call <32 x i32> @llvm.hexagon.V6.vdealvdd(<16 x i32> %v19, <16 x i32> %v18, i32 -4) #2
  %v21 = bitcast i16* %v8 to <16 x i32>*
  %v22 = load <16 x i32>, <16 x i32>* %v21, align 64, !tbaa !4
  %v23 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v20) #2
  %v24 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v20) #2
  %v25 = tail call <16 x i32> @llvm.hexagon.V6.vshufeh(<16 x i32> %v24, <16 x i32> %v23) #2
  %v26 = tail call <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32> %v24, <16 x i32> %v23) #2
  %v27 = tail call <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32> %v25, <16 x i32> %v22) #2
  %v28 = tail call <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32> %v26, <16 x i32> %v22) #2
  %v29 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v27) #2
  %v30 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v28) #2
  %v31 = tail call <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32> %v29, <16 x i32> %v30, i32 16) #2
  %v32 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v28) #2
  %v33 = tail call <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32> undef, <16 x i32> %v32, i32 16) #2
  %v34 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v33, <16 x i32> %v31) #2
  %v35 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v34) #2
  %v36 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v34) #2
  %v37 = tail call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> %v36, <16 x i32> %v35, i32 -4) #2
  %v38 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v37)
  %v39 = add nuw nsw i32 %v10, %v3
  %v40 = getelementptr inbounds i32, i32* undef, i32 %v39
  %v41 = bitcast i32* %v40 to <16 x i32>*
  store <16 x i32> %v38, <16 x i32>* %v41, align 64, !tbaa !6
  %v42 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v37)
  store <16 x i32> %v42, <16 x i32>* undef, align 64, !tbaa !6
  %v43 = getelementptr i32, i32* %v7, i32 32
  %v44 = getelementptr i16, i16* %v8, i32 32
  %v45 = bitcast i32* %v43 to <16 x i32>*
  %v46 = load <16 x i32>, <16 x i32>* %v45, align 64, !tbaa !1
  %v47 = add nuw nsw i32 %v10, 48
  %v48 = getelementptr inbounds i32, i32* %v1, i32 %v47
  %v49 = bitcast i32* %v48 to <16 x i32>*
  %v50 = load <16 x i32>, <16 x i32>* %v49, align 64, !tbaa !1
  %v51 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v50, <16 x i32> %v46)
  %v52 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v51) #2
  %v53 = tail call <32 x i32> @llvm.hexagon.V6.vdealvdd(<16 x i32> undef, <16 x i32> %v52, i32 -4) #2
  %v54 = bitcast i16* %v44 to <16 x i32>*
  %v55 = load <16 x i32>, <16 x i32>* %v54, align 64, !tbaa !4
  %v56 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v53) #2
  %v57 = tail call <16 x i32> @llvm.hexagon.V6.vshufeh(<16 x i32> undef, <16 x i32> %v56) #2
  %v58 = tail call <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32> undef, <16 x i32> %v56) #2
  %v59 = tail call <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32> %v57, <16 x i32> %v55) #2
  %v60 = tail call <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32> %v58, <16 x i32> %v55) #2
  %v61 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v59) #2
  %v62 = tail call <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32> %v61, <16 x i32> undef, i32 16) #2
  %v63 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v59) #2
  %v64 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v60) #2
  %v65 = tail call <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32> %v63, <16 x i32> %v64, i32 16) #2
  %v66 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v65, <16 x i32> %v62) #2
  %v67 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v66) #2
  %v68 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v66) #2
  %v69 = tail call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> %v68, <16 x i32> %v67, i32 -4) #2
  %v70 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v69)
  %v71 = add nuw nsw i32 %v4, %v10
  %v72 = getelementptr inbounds i32, i32* undef, i32 %v71
  %v73 = bitcast i32* %v72 to <16 x i32>*
  store <16 x i32> %v70, <16 x i32>* %v73, align 64, !tbaa !6
  %v74 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v69)
  %v75 = add nuw nsw i32 %v71, 16
  %v76 = getelementptr inbounds i32, i32* undef, i32 %v75
  %v77 = bitcast i32* %v76 to <16 x i32>*
  store <16 x i32> %v74, <16 x i32>* %v77, align 64, !tbaa !6
  %v78 = getelementptr i32, i32* %v7, i32 64
  %v79 = getelementptr i16, i16* %v8, i32 64
  %v80 = bitcast i32* %v78 to <16 x i32>*
  %v81 = load <16 x i32>, <16 x i32>* %v80, align 64, !tbaa !1
  %v82 = add nuw nsw i32 %v10, 80
  %v83 = getelementptr inbounds i32, i32* %v1, i32 %v82
  %v84 = bitcast i32* %v83 to <16 x i32>*
  %v85 = load <16 x i32>, <16 x i32>* %v84, align 64, !tbaa !1
  %v86 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v85, <16 x i32> %v81)
  %v87 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v86) #2
  %v88 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v86) #2
  %v89 = tail call <32 x i32> @llvm.hexagon.V6.vdealvdd(<16 x i32> %v88, <16 x i32> %v87, i32 -4) #2
  %v90 = bitcast i16* %v79 to <16 x i32>*
  %v91 = load <16 x i32>, <16 x i32>* %v90, align 64, !tbaa !4
  %v92 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v89) #2
  %v93 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v89) #2
  %v94 = tail call <16 x i32> @llvm.hexagon.V6.vshufeh(<16 x i32> %v93, <16 x i32> %v92) #2
  %v95 = tail call <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32> %v93, <16 x i32> %v92) #2
  %v96 = tail call <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32> %v94, <16 x i32> %v91) #2
  %v97 = tail call <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32> %v95, <16 x i32> %v91) #2
  %v98 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v97) #2
  %v99 = tail call <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32> undef, <16 x i32> %v98, i32 16) #2
  %v100 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v96) #2
  %v101 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v97) #2
  %v102 = tail call <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32> %v100, <16 x i32> %v101, i32 16) #2
  %v103 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v102, <16 x i32> %v99) #2
  %v104 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v103) #2
  %v105 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v103) #2
  %v106 = tail call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> %v105, <16 x i32> %v104, i32 -4) #2
  %v107 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v106)
  %v108 = add nuw nsw i32 %v5, %v10
  %v109 = getelementptr inbounds i32, i32* undef, i32 %v108
  %v110 = bitcast i32* %v109 to <16 x i32>*
  store <16 x i32> %v107, <16 x i32>* %v110, align 64, !tbaa !6
  %v111 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v106)
  %v112 = add nuw nsw i32 %v108, 16
  %v113 = getelementptr inbounds i32, i32* undef, i32 %v112
  %v114 = bitcast i32* %v113 to <16 x i32>*
  store <16 x i32> %v111, <16 x i32>* %v114, align 64, !tbaa !6
  %v115 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> undef) #2
  %v116 = tail call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> undef, <16 x i32> %v115, i32 -4) #2
  %v117 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v116)
  %v118 = add nuw nsw i32 %v6, %v10
  %v119 = getelementptr inbounds i32, i32* undef, i32 %v118
  %v120 = bitcast i32* %v119 to <16 x i32>*
  store <16 x i32> %v117, <16 x i32>* %v120, align 64, !tbaa !6
  %v121 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v116)
  %v122 = add nuw nsw i32 %v118, 16
  %v123 = getelementptr inbounds i32, i32* undef, i32 %v122
  %v124 = bitcast i32* %v123 to <16 x i32>*
  store <16 x i32> %v121, <16 x i32>* %v124, align 64, !tbaa !6
  %v125 = add nuw nsw i32 %v9, 4
  %v126 = icmp eq i32 %v125, 24
  %v127 = getelementptr i32, i32* %v7, i32 128
  %v128 = getelementptr i16, i16* %v8, i32 128
  br i1 %v126, label %b3, label %b2

b3:                                               ; preds = %b2
  %v129 = add nuw nsw i32 %v2, 1
  br label %b1
}

attributes #0 = { nounwind readnone }
attributes #1 = { "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"halide_mattrs", !"+hvx"}
!1 = !{!2, !2, i64 0}
!2 = !{!"in_u32", !3}
!3 = !{!"Halide buffer"}
!4 = !{!5, !5, i64 0}
!5 = !{!"in_u16", !3}
!6 = !{!7, !7, i64 0}
!7 = !{!"op_vmpy_v__uh_v__uh__1", !3}
