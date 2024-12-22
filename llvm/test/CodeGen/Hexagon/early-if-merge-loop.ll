; RUN: llc -march=hexagon < %s | FileCheck %s
; Make sure that the loop in the end has only one basic block.

; CHECK-LABEL: fred
; CHECK: %b2
; Rely on the comments, make sure the one for the loop header is present.
; CHECK: %loop
; CHECK: %should_merge
; CHECK: %exit

target triple = "hexagon"

define i32 @fred(i32 %a0, ptr nocapture readonly %a1) #0 {
b2:
  %v4 = getelementptr inbounds i32, ptr %a1, i32 1
  %v5 = zext i32 %a0 to i64
  br label %loop

loop:                                             ; preds = %should_merge, %b2
  %v7 = phi i32 [ 0, %b2 ], [ %v49, %should_merge ]
  %v8 = phi i32 [ 0, %b2 ], [ %v42, %should_merge ]
  %v9 = phi ptr [ %v4, %b2 ], [ %v53, %should_merge ]
  %v10 = phi i32 [ 0, %b2 ], [ %v30, %should_merge ]
  %v11 = phi ptr [ %a1, %b2 ], [ %v51, %should_merge ]
  %v12 = phi i32 [ 0, %b2 ], [ %v23, %should_merge ]
  %v13 = phi i32 [ 2, %b2 ], [ %v54, %should_merge ]
  %v14 = load i32, ptr %v11, align 4, !tbaa !0
  %v15 = load i32, ptr %v9, align 4, !tbaa !0
  %v16 = icmp ult i32 %v13, 30
  %v17 = zext i32 %v12 to i64
  %v18 = shl nuw i64 %v17, 32
  %v19 = zext i32 %v14 to i64
  %v20 = or i64 %v18, %v19
  %v21 = tail call i64 @llvm.hexagon.A2.addp(i64 %v20, i64 %v5)
  %v22 = lshr i64 %v21, 32
  %v23 = trunc i64 %v22 to i32
  %v24 = zext i32 %v10 to i64
  %v25 = shl nuw i64 %v24, 32
  %v26 = zext i32 %v15 to i64
  %v27 = or i64 %v25, %v26
  %v28 = tail call i64 @llvm.hexagon.A2.addp(i64 %v27, i64 %v5)
  %v29 = lshr i64 %v28, 32
  %v30 = trunc i64 %v29 to i32
  %v31 = getelementptr inbounds i32, ptr %a1, i32 %v13
  %v32 = load i32, ptr %v31, align 4, !tbaa !0
  %v33 = or i32 %v13, 1
  %v34 = getelementptr inbounds i32, ptr %a1, i32 %v33
  %v35 = load i32, ptr %v34, align 4, !tbaa !0
  %v36 = zext i32 %v8 to i64
  %v37 = shl nuw i64 %v36, 32
  %v38 = zext i32 %v32 to i64
  %v39 = or i64 %v37, %v38
  %v40 = tail call i64 @llvm.hexagon.A2.subp(i64 %v39, i64 %v5)
  %v41 = lshr i64 %v40, 32
  %v42 = trunc i64 %v41 to i32
  %v43 = zext i32 %v7 to i64
  %v44 = shl nuw i64 %v43, 32
  %v45 = zext i32 %v35 to i64
  %v46 = or i64 %v44, %v45
  %v47 = tail call i64 @llvm.hexagon.A2.subp(i64 %v46, i64 %v5)
  %v48 = lshr i64 %v47, 32
  %v49 = trunc i64 %v48 to i32
  br i1 %v16, label %should_merge, label %exit

should_merge:                                     ; preds = %loop
  %v50 = add nuw nsw i32 %v13, 2
  %v51 = getelementptr inbounds i32, ptr %a1, i32 %v50
  %v52 = add nuw nsw i32 %v13, 3
  %v53 = getelementptr inbounds i32, ptr %a1, i32 %v52
  %v54 = add nuw nsw i32 %v13, 4
  br label %loop

exit:                                             ; preds = %loop
  %v57 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v42, i32 %v23)
  %v58 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v49, i32 %v30)
  %v59 = tail call i64 @llvm.hexagon.A2.addp(i64 %v57, i64 %v58)
  %v60 = lshr i64 %v59, 32
  %v61 = trunc i64 %v60 to i32
  ret i32 %v61
}

declare i64 @llvm.hexagon.A2.addp(i64, i64) #1
declare i64 @llvm.hexagon.A2.subp(i64, i64) #1
declare i64 @llvm.hexagon.A2.combinew(i32, i32) #1

attributes #0 = { nounwind readonly "target-cpu"="hexagonv60" "target-features"="-hvx,-long-calls" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"long", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
