; RUN: llc -mtriple=hexagon -pipeliner-ignore-recmii -pipeliner-max-stages=2 -enable-pipeliner < %s -pipeliner-experimental-cg=true | FileCheck %s

; This is a loop we pipeline to three packets, though we could do bettter.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: {
; CHECK: }
; CHECK: {
; CHECK: }
; CHECK: {
; CHECK: }{{[ \t]*}}:endloop0

; Function Attrs: nounwind
define void @f0(ptr nocapture %a0, i16 signext %a1) #0 {
b0:
  %v0 = sext i16 %a1 to i32
  %v1 = add i32 %v0, -1
  %v2 = icmp sgt i32 %v1, 0
  br i1 %v2, label %b1, label %b4

b1:                                               ; preds = %b0
  %v3 = getelementptr i32, ptr %a0, i32 %v1
  %v4 = load i32, ptr %v3, align 4
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v5 = phi i32 [ %v16, %b2 ], [ %v1, %b1 ]
  %v6 = phi i32 [ %v5, %b2 ], [ %v0, %b1 ]
  %v7 = phi i32 [ %v10, %b2 ], [ %v4, %b1 ]
  %v8 = add nsw i32 %v6, -2
  %v9 = getelementptr inbounds i32, ptr %a0, i32 %v8
  %v10 = load i32, ptr %v9, align 4, !tbaa !0
  %v11 = tail call i64 @llvm.hexagon.M2.dpmpyss.s0(i32 %v10, i32 7946)
  %v12 = tail call i64 @llvm.hexagon.S2.asl.r.p(i64 %v11, i32 -13)
  %v13 = getelementptr inbounds i32, ptr %a0, i32 %v5
  %v14 = tail call i32 @llvm.hexagon.A2.sat(i64 %v12)
  %v15 = tail call i32 @llvm.hexagon.A2.subsat(i32 %v7, i32 %v14)
  store i32 %v15, ptr %v13, align 4, !tbaa !0
  %v16 = add nsw i32 %v5, -1
  %v17 = icmp sgt i32 %v16, 0
  br i1 %v17, label %b2, label %b3

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b3, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.asl.r.p(i64, i32) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.dpmpyss.s0(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.subsat(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.sat(i64) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
