; RUN: llc -march=hexagon -O3 < %s | FileCheck %s
; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; RUN: llc -march=hexagon -O1 < %s | FileCheck %s
; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
;
; CHECK-NOT: v{{[0-9]*}}.cur
;
; CHECK: {
; CHECK: v{{[0-9]+}}.h = vasr(v{{[0-9]+}}.w,v{{[0-9]+}}.w,r{{[0-7]+}})

; CHECK: }
; CHECK: {
; CHECK: v{{[0-9]+}}.h = vasr(v{{[0-9]+}}.w,v{{[0-9]+}}.w,r{{[0-7]+}})
; CHECK: }
; CHECK-NOT: vand
; CHECK: v{{[0-9]+}} = v{{[0-9]+}}

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(ptr nocapture readonly %a0, ptr nocapture readonly %a1, i32 %a2, ptr nocapture %a3, i32 %a4) #0 {
b0:
  %v1 = load i32, ptr %a1, align 4, !tbaa !0
  %v2 = getelementptr inbounds i8, ptr %a1, i32 4
  %v4 = load i32, ptr %v2, align 4, !tbaa !0
  %v5 = getelementptr inbounds i8, ptr %a1, i32 8
  %v7 = load i32, ptr %v5, align 4, !tbaa !0
  %v8 = mul i32 %a4, 2
  %v9 = add i32 %v8, %a4
  %v10 = icmp sgt i32 %a4, 0
  br i1 %v10, label %b1, label %b4

b1:                                               ; preds = %b0
  %v11 = getelementptr inbounds i8, ptr %a0, i32 %v9
  %v12 = getelementptr inbounds i8, ptr %a0, i32 %v8
  %v13 = getelementptr inbounds i8, ptr %a0, i32 %a4
  %v14 = add i32 %v9, 64
  %v16 = add i32 %v8, 64
  %v18 = add i32 %a4, 64
  %v21 = getelementptr inbounds i8, ptr %a0, i32 %v14
  %v22 = load <16 x i32>, ptr %v11, align 64, !tbaa !4
  %v23 = getelementptr inbounds i8, ptr %a0, i32 %v16
  %v24 = load <16 x i32>, ptr %v12, align 64, !tbaa !4
  %v25 = getelementptr inbounds i8, ptr %a0, i32 %v18
  %v26 = load <16 x i32>, ptr %v13, align 64, !tbaa !4
  %v27 = load <16 x i32>, ptr %a0, align 64, !tbaa !4
  %v28 = getelementptr inbounds i8, ptr %a3, i32 %a4
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v29 = phi ptr [ %a0, %b1 ], [ %v40, %b2 ]
  %v30 = phi ptr [ %a3, %b1 ], [ %v74, %b2 ]
  %v31 = phi ptr [ %v25, %b1 ], [ %v45, %b2 ]
  %v32 = phi ptr [ %v23, %b1 ], [ %v48, %b2 ]
  %v33 = phi ptr [ %v21, %b1 ], [ %v51, %b2 ]
  %v34 = phi ptr [ %v28, %b1 ], [ %v89, %b2 ]
  %v35 = phi i32 [ 0, %b1 ], [ %v90, %b2 ]
  %v36 = phi <16 x i32> [ %v27, %b1 ], [ %v42, %b2 ]
  %v37 = phi <16 x i32> [ %v26, %b1 ], [ %v44, %b2 ]
  %v38 = phi <16 x i32> [ %v24, %b1 ], [ %v47, %b2 ]
  %v39 = phi <16 x i32> [ %v22, %b1 ], [ %v50, %b2 ]
  %v40 = getelementptr inbounds i8, ptr %v29, i32 64
  %v42 = load <16 x i32>, ptr %v40, align 64, !tbaa !4
  %v44 = load <16 x i32>, ptr %v31, align 64, !tbaa !4
  %v45 = getelementptr inbounds i8, ptr %v31, i32 64
  %v47 = load <16 x i32>, ptr %v32, align 64, !tbaa !4
  %v48 = getelementptr inbounds i8, ptr %v32, i32 64
  %v50 = load <16 x i32>, ptr %v33, align 64, !tbaa !4
  %v51 = getelementptr inbounds i8, ptr %v33, i32 64
  %v52 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v42, <16 x i32> %v36, i32 4)
  %v53 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v44, <16 x i32> %v37, i32 4)
  %v54 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v47, <16 x i32> %v38, i32 4)
  %v55 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v50, <16 x i32> %v39, i32 4)
  %v56 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v52, <16 x i32> %v36)
  %v57 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v53, <16 x i32> %v37)
  %v58 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v54, <16 x i32> %v38)
  %v59 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v55, <16 x i32> %v39)
  %v60 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32> %v56, i32 %v1, i32 0)
  %v61 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32> %v56, i32 %v1, i32 1)
  %v62 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v60, <32 x i32> %v57, i32 %v4, i32 0)
  %v63 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v61, <32 x i32> %v57, i32 %v4, i32 1)
  %v64 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v62, <32 x i32> %v58, i32 %v7, i32 0)
  %v65 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v63, <32 x i32> %v58, i32 %v7, i32 1)
  %v66 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v65)
  %v67 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v65)
  %v68 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %v66, <16 x i32> %v67, i32 %a2)
  %v69 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v64)
  %v70 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v64)
  %v71 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %v69, <16 x i32> %v70, i32 %a2)
  %v72 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %v68, <16 x i32> %v71)
  store <16 x i32> %v72, ptr %v30, align 64, !tbaa !4
  %v74 = getelementptr inbounds i8, ptr %v30, i32 64
  %v75 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32> %v57, i32 %v1, i32 0)
  %v76 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32> %v57, i32 %v1, i32 1)
  %v77 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v75, <32 x i32> %v58, i32 %v4, i32 0)
  %v78 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v76, <32 x i32> %v58, i32 %v4, i32 1)
  %v79 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v77, <32 x i32> %v59, i32 %v7, i32 0)
  %v80 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v78, <32 x i32> %v59, i32 %v7, i32 1)
  %v81 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v80)
  %v82 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v80)
  %v83 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %v81, <16 x i32> %v82, i32 %a2)
  %v84 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v79)
  %v85 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v79)
  %v86 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %v84, <16 x i32> %v85, i32 %a2)
  %v87 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %v83, <16 x i32> %v86)
  store <16 x i32> %v87, ptr %v34, align 64, !tbaa !4
  %v89 = getelementptr inbounds i8, ptr %v34, i32 64
  %v90 = add nsw i32 %v35, 64
  %v91 = icmp slt i32 %v90, %a4
  br i1 %v91, label %b2, label %b3

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b3, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32>, i32, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32>, <32 x i32>, i32, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!2, !2, i64 0}
