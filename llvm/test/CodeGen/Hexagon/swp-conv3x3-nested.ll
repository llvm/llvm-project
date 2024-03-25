; RUN: llc -march=hexagon < %s -pipeliner-experimental-cg=true | FileCheck %s

; This version of the conv3x3 test has both loops. This test checks that the
; inner loop has 14 packets.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: }
; CHECK: }
; CHECK: }
; CHECK: }
; CHECK: }
; CHECK: }
; CHECK: }
; CHECK: }
; CHECK: }
; CHECK: }
; CHECK: }
; CHECK: }
; CHECK: }
; CHECK-NOT: }
; CHECK: }{{[ \t]*}}:endloop0

declare <16 x i32> @llvm.hexagon.V6.vd0() #0
declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32>, i32, i32) #0
declare <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32>, <32 x i32>, i32, i32) #0
declare <16 x i32> @llvm.hexagon.V6.vasrwhsat(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32>, <16 x i32>, i32) #0

define void @f0(ptr noalias nocapture readonly %a0, i32 %a1, i32 %a2, i32 %a3, ptr noalias nocapture readonly %a4, i32 %a5, ptr noalias nocapture %a6) local_unnamed_addr #1 {
b0:
  %v0 = add nsw i32 %a3, -1
  %v1 = icmp sgt i32 %a3, 2
  br i1 %v1, label %b1, label %b6

b1:                                               ; preds = %b0
  %v2 = getelementptr inbounds i8, ptr %a6, i32 %a1
  %v3 = getelementptr inbounds i8, ptr %a0, i32 %a1
  %v5 = load i32, ptr %a4, align 4, !tbaa !1, !alias.scope !5, !noalias !8
  %v6 = getelementptr inbounds i8, ptr %a4, i32 4
  %v8 = load i32, ptr %v6, align 4, !tbaa !1, !alias.scope !5, !noalias !8
  %v9 = getelementptr inbounds i8, ptr %a4, i32 8
  %v11 = load i32, ptr %v9, align 4, !tbaa !1, !alias.scope !5, !noalias !8
  %v12 = sub i32 0, %a1
  %v13 = shl nsw i32 %a1, 1
  %v14 = tail call <16 x i32> @llvm.hexagon.V6.vd0() #2
  %v15 = icmp sgt i32 %a2, 0
  br label %b2

b2:                                               ; preds = %b5, %b1
  %v16 = phi ptr [ %v2, %b1 ], [ %v102, %b5 ]
  %v17 = phi ptr [ %v3, %b1 ], [ %v21, %b5 ]
  %v18 = phi i32 [ 1, %b1 ], [ %v103, %b5 ]
  %v19 = getelementptr inbounds i8, ptr %v17, i32 %v12
  %v20 = getelementptr inbounds i8, ptr %v17, i32 %a1
  %v21 = getelementptr inbounds i8, ptr %v17, i32 %v13
  br i1 %v15, label %b3, label %b5

b3:                                               ; preds = %b2
  %v23 = load <16 x i32>, ptr %v21, align 64, !tbaa !11, !alias.scope !12, !noalias !13
  %v24 = getelementptr inbounds i8, ptr %v21, i32 64
  %v27 = load <16 x i32>, ptr %v20, align 64, !tbaa !11, !alias.scope !12, !noalias !13
  %v28 = getelementptr inbounds i8, ptr %v20, i32 64
  %v31 = load <16 x i32>, ptr %v17, align 64, !tbaa !11, !alias.scope !12, !noalias !13
  %v32 = getelementptr inbounds i8, ptr %v17, i32 64
  %v35 = load <16 x i32>, ptr %v19, align 64, !tbaa !11, !alias.scope !12, !noalias !13
  %v36 = getelementptr inbounds i8, ptr %v19, i32 64
  %v38 = getelementptr inbounds i8, ptr %v16, i32 %a1
  br label %b4

b4:                                               ; preds = %b4, %b3
  %v41 = phi ptr [ %v38, %b3 ], [ %v99, %b4 ]
  %v42 = phi ptr [ %v16, %b3 ], [ %v84, %b4 ]
  %v43 = phi ptr [ %v24, %b3 ], [ %v60, %b4 ]
  %v44 = phi ptr [ %v28, %b3 ], [ %v58, %b4 ]
  %v45 = phi ptr [ %v32, %b3 ], [ %v56, %b4 ]
  %v46 = phi ptr [ %v36, %b3 ], [ %v54, %b4 ]
  %v47 = phi i32 [ %a2, %b3 ], [ %v100, %b4 ]
  %v48 = phi <16 x i32> [ %v35, %b3 ], [ %v55, %b4 ]
  %v49 = phi <16 x i32> [ %v31, %b3 ], [ %v57, %b4 ]
  %v50 = phi <16 x i32> [ %v27, %b3 ], [ %v59, %b4 ]
  %v51 = phi <16 x i32> [ %v23, %b3 ], [ %v61, %b4 ]
  %v52 = phi <16 x i32> [ %v14, %b3 ], [ %v82, %b4 ]
  %v53 = phi <16 x i32> [ %v14, %b3 ], [ %v97, %b4 ]
  %v54 = getelementptr inbounds <16 x i32>, ptr %v46, i32 1
  %v55 = load <16 x i32>, ptr %v46, align 64, !tbaa !11, !alias.scope !12, !noalias !13
  %v56 = getelementptr inbounds <16 x i32>, ptr %v45, i32 1
  %v57 = load <16 x i32>, ptr %v45, align 64, !tbaa !11, !alias.scope !12, !noalias !13
  %v58 = getelementptr inbounds <16 x i32>, ptr %v44, i32 1
  %v59 = load <16 x i32>, ptr %v44, align 64, !tbaa !11, !alias.scope !12, !noalias !13
  %v60 = getelementptr inbounds <16 x i32>, ptr %v43, i32 1
  %v61 = load <16 x i32>, ptr %v43, align 64, !tbaa !11, !alias.scope !12, !noalias !13
  %v62 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v55, <16 x i32> %v48, i32 4) #2
  %v63 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v57, <16 x i32> %v49, i32 4) #2
  %v64 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v59, <16 x i32> %v50, i32 4) #2
  %v65 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v61, <16 x i32> %v51, i32 4) #2
  %v66 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v62, <16 x i32> %v48) #2
  %v67 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v63, <16 x i32> %v49) #2
  %v68 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v64, <16 x i32> %v50) #2
  %v69 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v65, <16 x i32> %v51) #2
  %v70 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32> %v66, i32 %v5, i32 0) #2
  %v71 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32> %v66, i32 %v5, i32 1) #2
  %v72 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v70, <32 x i32> %v67, i32 %v8, i32 0) #2
  %v73 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v71, <32 x i32> %v67, i32 %v8, i32 1) #2
  %v74 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v72, <32 x i32> %v68, i32 %v11, i32 0) #2
  %v75 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v73, <32 x i32> %v68, i32 %v11, i32 1) #2
  %v76 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v75) #2
  %v77 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v75) #2
  %v78 = tail call <16 x i32> @llvm.hexagon.V6.vasrwhsat(<16 x i32> %v76, <16 x i32> %v77, i32 %a5) #2
  %v79 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v74) #2
  %v80 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v74) #2
  %v81 = tail call <16 x i32> @llvm.hexagon.V6.vasrwhsat(<16 x i32> %v79, <16 x i32> %v80, i32 %a5) #2
  %v82 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %v78, <16 x i32> %v81) #2
  %v83 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v82, <16 x i32> %v52, i32 1) #2
  %v84 = getelementptr inbounds <16 x i32>, ptr %v42, i32 1
  store <16 x i32> %v83, ptr %v42, align 64, !tbaa !11, !alias.scope !14, !noalias !15
  %v85 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32> %v67, i32 %v5, i32 0) #2
  %v86 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32> %v67, i32 %v5, i32 1) #2
  %v87 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v85, <32 x i32> %v68, i32 %v8, i32 0) #2
  %v88 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v86, <32 x i32> %v68, i32 %v8, i32 1) #2
  %v89 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v87, <32 x i32> %v69, i32 %v11, i32 0) #2
  %v90 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v88, <32 x i32> %v69, i32 %v11, i32 1) #2
  %v91 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v90) #2
  %v92 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v90) #2
  %v93 = tail call <16 x i32> @llvm.hexagon.V6.vasrwhsat(<16 x i32> %v91, <16 x i32> %v92, i32 %a5) #2
  %v94 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v89) #2
  %v95 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v89) #2
  %v96 = tail call <16 x i32> @llvm.hexagon.V6.vasrwhsat(<16 x i32> %v94, <16 x i32> %v95, i32 %a5) #2
  %v97 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %v93, <16 x i32> %v96) #2
  %v98 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v97, <16 x i32> %v53, i32 1) #2
  %v99 = getelementptr inbounds <16 x i32>, ptr %v41, i32 1
  store <16 x i32> %v98, ptr %v41, align 64, !tbaa !11, !alias.scope !14, !noalias !15
  %v100 = add nsw i32 %v47, -64
  %v101 = icmp sgt i32 %v47, 64
  br i1 %v101, label %b4, label %b5

b5:                                               ; preds = %b4, %b2
  %v102 = getelementptr inbounds i8, ptr %v16, i32 %v13
  %v103 = add nuw nsw i32 %v18, 2
  %v104 = icmp slt i32 %v103, %v0
  br i1 %v104, label %b2, label %b6

b6:                                               ; preds = %b5, %b0
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv62" "target-features"="+hvx-length64b,+hvxv62" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6}
!6 = distinct !{!6, !7, !"x: %a"}
!7 = distinct !{!7, !"x"}
!8 = !{!9, !10}
!9 = distinct !{!9, !7, !"x: %b"}
!10 = distinct !{!10, !7, !"x: %c"}
!11 = !{!3, !3, i64 0}
!12 = !{!9}
!13 = !{!6, !10}
!14 = !{!10}
!15 = !{!9, !6}
