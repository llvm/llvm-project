; RUN: llc -march=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; Function Attrs: nounwind
define void @f0(ptr nocapture %a0) #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v0 = phi i64 [ %v9, %b1 ], [ 0, %b0 ]
  %v1 = phi i32 [ %v10, %b1 ], [ 0, %b0 ]
  %v2 = getelementptr inbounds i32, ptr %a0, i32 %v1
  %v3 = load i32, ptr %v2, align 4, !tbaa !0
  %v4 = zext i32 %v3 to i64
  %v5 = load i32, ptr undef, align 4, !tbaa !0
  %v6 = zext i32 %v5 to i64
  %v7 = shl nuw i64 %v6, 32
  %v8 = or i64 %v7, %v4
  %v9 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 %v0, i64 %v8, i64 %v8)
  %v10 = add nsw i32 %v1, 4
  %v11 = icmp slt i32 %v10, undef
  br i1 %v11, label %b1, label %b2

b2:                                               ; preds = %b1, %b0
  %v12 = phi i64 [ 0, %b0 ], [ %v9, %b1 ]
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.vdmacs.s0(i64, i64, i64) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
