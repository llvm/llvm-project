; RUN: llc -mtriple=hexagon -O3 < %s
; REQUIRES: asserts

; Test that we generate the correct names for Phis when there is
; a Phi that references a Phi that references another Phi. For example,
;  v6 = phi(v1, v9)
;  v7 = phi(v0, v6)
;  v8 = phi(v2, v7)

; Function Attrs: nounwind
define void @f0(ptr noalias nocapture readonly %a0, i32 %a1, i32 %a2, ptr noalias nocapture %a3, i32 %a4) #0 {
b0:
  %v0 = add i32 %a1, -1
  %v2 = getelementptr inbounds i8, ptr %a0, i32 undef
  br i1 undef, label %b1, label %b4

b1:                                               ; preds = %b1, %b0
  br i1 undef, label %b1, label %b2

b2:                                               ; preds = %b1
  %v4 = getelementptr inbounds i8, ptr %a0, i32 undef
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v5 = phi ptr [ %v10, %b3 ], [ %a3, %b2 ]
  %v6 = phi ptr [ %v25, %b3 ], [ %v4, %b2 ]
  %v7 = phi ptr [ %v6, %b3 ], [ %v2, %b2 ]
  %v8 = phi ptr [ %v7, %b3 ], [ %a0, %b2 ]
  %v9 = phi i32 [ %v26, %b3 ], [ 1, %b2 ]
  %v10 = getelementptr inbounds i8, ptr %v5, i32 %a4
  %v11 = getelementptr inbounds i8, ptr %v8, i32 -1
  %v12 = load i8, ptr %v11, align 1, !tbaa !0
  %v13 = zext i8 %v12 to i32
  %v14 = add nuw nsw i32 %v13, 0
  %v15 = add nuw nsw i32 %v14, 0
  %v16 = add nuw nsw i32 %v15, 0
  %v17 = load i8, ptr %v6, align 1, !tbaa !0
  %v18 = zext i8 %v17 to i32
  %v19 = add nuw nsw i32 %v16, %v18
  %v20 = add nuw nsw i32 %v19, 0
  %v21 = mul nsw i32 %v20, 7282
  %v22 = add nsw i32 %v21, 32768
  %v23 = lshr i32 %v22, 16
  %v24 = trunc i32 %v23 to i8
  store i8 %v24, ptr %v10, align 1, !tbaa !0
  %v25 = getelementptr inbounds i8, ptr %v6, i32 %a2
  %v26 = add i32 %v9, 1
  %v27 = icmp eq i32 %v26, %v0
  br i1 %v27, label %b4, label %b3

b4:                                               ; preds = %b3, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
