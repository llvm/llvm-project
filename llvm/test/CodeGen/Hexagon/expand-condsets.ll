; RUN: llc -march=hexagon -O3 < %s | FileCheck %s
; Check if all or's in the loop were predicated.
; CHECK: if{{.*}} = or
; CHECK: if{{.*}} = or
; CHECK: if{{.*}} = or
; CHECK: if{{.*}} = or
; CHECK: endloop

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(i32 %a0, ptr nocapture %a1, ptr nocapture %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6) #0 {
b0:
  %v0 = icmp ugt i32 %a0, 32
  %v1 = lshr i32 %a0, 6
  %v2 = select i1 %v0, i32 %v1, i32 1
  %v3 = icmp eq i32 %v2, 0
  br i1 %v3, label %b9, label %b1

b1:                                               ; preds = %b0
  %v4 = lshr i32 %a0, 2
  %v5 = getelementptr inbounds i32, ptr %a1, i32 %v4
  br label %b2

b2:                                               ; preds = %b7, %b1
  %v6 = phi ptr [ %v5, %b1 ], [ %v9, %b7 ]
  %v7 = phi ptr [ %a1, %b1 ], [ %v49, %b7 ]
  %v8 = phi i32 [ 0, %b1 ], [ %v55, %b7 ]
  %v9 = getelementptr i32, ptr %v6, i32 64
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v10 = phi i32 [ 2, %b2 ], [ %v46, %b3 ]
  %v11 = phi i32 [ 1, %b2 ], [ %v45, %b3 ]
  %v12 = phi ptr [ %v6, %b2 ], [ %v23, %b3 ]
  %v13 = phi ptr [ %v7, %b2 ], [ %v19, %b3 ]
  %v14 = phi i32 [ 0, %b2 ], [ %v47, %b3 ]
  %v15 = phi i32 [ 0, %b2 ], [ %v41, %b3 ]
  %v16 = phi i32 [ 0, %b2 ], [ %v44, %b3 ]
  %v17 = getelementptr inbounds i32, ptr %v13, i32 1
  %v18 = load i32, ptr %v13, align 4, !tbaa !0
  %v19 = getelementptr inbounds i32, ptr %v13, i32 2
  %v20 = load i32, ptr %v17, align 4, !tbaa !0
  %v21 = getelementptr inbounds i32, ptr %v12, i32 1
  %v22 = load i32, ptr %v12, align 4, !tbaa !0
  %v23 = getelementptr inbounds i32, ptr %v12, i32 2
  %v24 = load i32, ptr %v21, align 4, !tbaa !0
  %v25 = tail call i32 @llvm.hexagon.A2.add(i32 %v22, i32 %a4)
  %v26 = tail call i32 @llvm.hexagon.A2.sub(i32 %v25, i32 %a3)
  %v27 = tail call i32 @llvm.hexagon.A2.add(i32 %v24, i32 %a4)
  %v28 = tail call i32 @llvm.hexagon.A2.sub(i32 %v27, i32 %a3)
  %v29 = tail call i32 @llvm.hexagon.A2.sub(i32 %v18, i32 %a5)
  %v30 = tail call i32 @llvm.hexagon.A2.add(i32 %v29, i32 %a6)
  %v31 = tail call i32 @llvm.hexagon.A2.sub(i32 %v20, i32 %a5)
  %v32 = tail call i32 @llvm.hexagon.A2.add(i32 %v31, i32 %a6)
  %v33 = icmp ugt i32 %v26, %v18
  %v34 = select i1 %v33, i32 0, i32 %v11
  %v35 = or i32 %v34, %v15
  %v36 = icmp ult i32 %v30, %v22
  %v37 = select i1 %v36, i32 %v11, i32 0
  %v38 = or i32 %v37, %v16
  %v39 = icmp ugt i32 %v28, %v20
  %v40 = select i1 %v39, i32 0, i32 %v10
  %v41 = or i32 %v35, %v40
  %v42 = icmp ult i32 %v32, %v24
  %v43 = select i1 %v42, i32 %v10, i32 0
  %v44 = or i32 %v38, %v43
  %v45 = shl i32 %v11, 2
  %v46 = shl i32 %v10, 2
  %v47 = add i32 %v14, 1
  %v48 = icmp eq i32 %v47, 32
  br i1 %v48, label %b4, label %b3

b4:                                               ; preds = %b3
  %v49 = getelementptr i32, ptr %v7, i32 64
  br i1 %v0, label %b5, label %b6

b5:                                               ; preds = %b4
  %v50 = getelementptr inbounds i32, ptr %a2, i32 %v8
  store i32 %v41, ptr %v50, align 4, !tbaa !0
  %v51 = add i32 %v8, %v2
  %v52 = getelementptr inbounds i32, ptr %a2, i32 %v51
  store i32 %v44, ptr %v52, align 4, !tbaa !0
  br label %b7

b6:                                               ; preds = %b4
  %v53 = or i32 %v41, %v44
  %v54 = getelementptr inbounds i32, ptr %a2, i32 %v8
  store i32 %v53, ptr %v54, align 4, !tbaa !0
  br label %b7

b7:                                               ; preds = %b6, %b5
  %v55 = add i32 %v8, 1
  %v56 = icmp eq i32 %v55, %v2
  br i1 %v56, label %b8, label %b2

b8:                                               ; preds = %b7
  br label %b9

b9:                                               ; preds = %b8, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.sub(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.add(i32, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
