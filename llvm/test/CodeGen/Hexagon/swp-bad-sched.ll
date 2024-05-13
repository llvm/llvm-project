; REQUIRES: asserts
; RUN: llc -march=hexagon -enable-pipeliner -enable-aa-sched-mi < %s -pipeliner-experimental-cg=true | FileCheck %s

; CHECK: loop0(
; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: or
; CHECK: or
; CHECK: }
; CHECK: {
; CHECK: }
; CHECK: {
; CHECK: memw
; CHECK-NEXT: }{{[ \t]*}}:endloop0

; Function Attrs: nounwind
define void @f0(ptr nocapture %a0, i32 %a1, ptr nocapture %a2) #0 {
b0:
  %v0 = icmp sgt i32 %a1, 0
  br i1 %v0, label %b1, label %b9

b1:                                               ; preds = %b0
  %v1 = icmp ugt i32 %a1, 3
  %v2 = add i32 %a1, -3
  br i1 %v1, label %b2, label %b5

b2:                                               ; preds = %b1
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v3 = phi i32 [ %v48, %b3 ], [ 0, %b2 ]
  %v4 = phi i32 [ %v46, %b3 ], [ 0, %b2 ]
  %v5 = phi i32 [ %v49, %b3 ], [ 0, %b2 ]
  %v6 = getelementptr inbounds [576 x i32], ptr %a0, i32 0, i32 %v5
  %v7 = load i32, ptr %v6, align 4, !tbaa !0
  %v8 = getelementptr inbounds [576 x i32], ptr %a0, i32 1, i32 %v5
  %v9 = load i32, ptr %v8, align 4, !tbaa !0
  %v10 = add nsw i32 %v9, %v7
  store i32 %v10, ptr %v6, align 4, !tbaa !0
  %v11 = sub nsw i32 %v7, %v9
  store i32 %v11, ptr %v8, align 4, !tbaa !0
  %v12 = tail call i32 @llvm.hexagon.A2.abs(i32 %v10)
  %v13 = or i32 %v12, %v4
  %v14 = tail call i32 @llvm.hexagon.A2.abs(i32 %v11)
  %v15 = or i32 %v14, %v3
  %v16 = add nsw i32 %v5, 1
  %v17 = getelementptr inbounds [576 x i32], ptr %a0, i32 0, i32 %v16
  %v18 = load i32, ptr %v17, align 4, !tbaa !0
  %v19 = getelementptr inbounds [576 x i32], ptr %a0, i32 1, i32 %v16
  %v20 = load i32, ptr %v19, align 4, !tbaa !0
  %v21 = add nsw i32 %v20, %v18
  store i32 %v21, ptr %v17, align 4, !tbaa !0
  %v22 = sub nsw i32 %v18, %v20
  store i32 %v22, ptr %v19, align 4, !tbaa !0
  %v23 = tail call i32 @llvm.hexagon.A2.abs(i32 %v21)
  %v24 = or i32 %v23, %v13
  %v25 = tail call i32 @llvm.hexagon.A2.abs(i32 %v22)
  %v26 = or i32 %v25, %v15
  %v27 = add nsw i32 %v5, 2
  %v28 = getelementptr inbounds [576 x i32], ptr %a0, i32 0, i32 %v27
  %v29 = load i32, ptr %v28, align 4, !tbaa !0
  %v30 = getelementptr inbounds [576 x i32], ptr %a0, i32 1, i32 %v27
  %v31 = load i32, ptr %v30, align 4, !tbaa !0
  %v32 = add nsw i32 %v31, %v29
  store i32 %v32, ptr %v28, align 4, !tbaa !0
  %v33 = sub nsw i32 %v29, %v31
  store i32 %v33, ptr %v30, align 4, !tbaa !0
  %v34 = tail call i32 @llvm.hexagon.A2.abs(i32 %v32)
  %v35 = or i32 %v34, %v24
  %v36 = tail call i32 @llvm.hexagon.A2.abs(i32 %v33)
  %v37 = or i32 %v36, %v26
  %v38 = add nsw i32 %v5, 3
  %v39 = getelementptr inbounds [576 x i32], ptr %a0, i32 0, i32 %v38
  %v40 = load i32, ptr %v39, align 4, !tbaa !0
  %v41 = getelementptr inbounds [576 x i32], ptr %a0, i32 1, i32 %v38
  %v42 = load i32, ptr %v41, align 4, !tbaa !0
  %v43 = add nsw i32 %v42, %v40
  store i32 %v43, ptr %v39, align 4, !tbaa !0
  %v44 = sub nsw i32 %v40, %v42
  store i32 %v44, ptr %v41, align 4, !tbaa !0
  %v45 = tail call i32 @llvm.hexagon.A2.abs(i32 %v43)
  %v46 = or i32 %v45, %v35
  %v47 = tail call i32 @llvm.hexagon.A2.abs(i32 %v44)
  %v48 = or i32 %v47, %v37
  %v49 = add nsw i32 %v5, 4
  %v50 = icmp slt i32 %v49, %v2
  br i1 %v50, label %b3, label %b4

b4:                                               ; preds = %b3
  br label %b5

b5:                                               ; preds = %b4, %b1
  %v51 = phi i32 [ 0, %b1 ], [ %v49, %b4 ]
  %v52 = phi i32 [ 0, %b1 ], [ %v48, %b4 ]
  %v53 = phi i32 [ 0, %b1 ], [ %v46, %b4 ]
  %v54 = icmp eq i32 %v51, %a1
  br i1 %v54, label %b9, label %b6

b6:                                               ; preds = %b5
  br label %b7

b7:                                               ; preds = %b7, %b6
  %v55 = phi i32 [ %v67, %b7 ], [ %v52, %b6 ]
  %v56 = phi i32 [ %v65, %b7 ], [ %v53, %b6 ]
  %v57 = phi i32 [ %v68, %b7 ], [ %v51, %b6 ]
  %v58 = getelementptr inbounds [576 x i32], ptr %a0, i32 0, i32 %v57
  %v59 = load i32, ptr %v58, align 4, !tbaa !0
  %v60 = getelementptr inbounds [576 x i32], ptr %a0, i32 1, i32 %v57
  %v61 = load i32, ptr %v60, align 4, !tbaa !0
  %v62 = add nsw i32 %v61, %v59
  store i32 %v62, ptr %v58, align 4, !tbaa !0
  %v63 = sub nsw i32 %v59, %v61
  store i32 %v63, ptr %v60, align 4, !tbaa !0
  %v64 = tail call i32 @llvm.hexagon.A2.abs(i32 %v62)
  %v65 = or i32 %v64, %v56
  %v66 = tail call i32 @llvm.hexagon.A2.abs(i32 %v63)
  %v67 = or i32 %v66, %v55
  %v68 = add nsw i32 %v57, 1
  %v69 = icmp eq i32 %v68, %a1
  br i1 %v69, label %b8, label %b7

b8:                                               ; preds = %b7
  br label %b9

b9:                                               ; preds = %b8, %b5, %b0
  %v70 = phi i32 [ 0, %b0 ], [ %v52, %b5 ], [ %v67, %b8 ]
  %v71 = phi i32 [ 0, %b0 ], [ %v53, %b5 ], [ %v65, %b8 ]
  %v72 = load i32, ptr %a2, align 4, !tbaa !0
  %v73 = or i32 %v72, %v71
  store i32 %v73, ptr %a2, align 4, !tbaa !0
  %v74 = getelementptr inbounds i32, ptr %a2, i32 1
  %v75 = load i32, ptr %v74, align 4, !tbaa !0
  %v76 = or i32 %v75, %v70
  store i32 %v76, ptr %v74, align 4, !tbaa !0
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.abs(i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
