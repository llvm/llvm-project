; RUN: llc -march=hexagon < %s -pipeliner-experimental-cg=true | FileCheck %s

; Test that the pipeliner cause an assert and correctly pipelines the
; loop.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: [[REG0:r([0-9]+)]] = sath([[REG1:r([0-9]+)]])
; CHECK: memh(r{{[0-9]+}}++#2) = [[REG0]].new
; CHECK: [[REG1]] =
; CHECK: endloop0

define void @f0(i16* nocapture %a0, float* nocapture readonly %a1, float %a2, i32 %a3) {
b0:
  %v0 = icmp sgt i32 %a3, 0
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v1 = phi i32 [ %v11, %b1 ], [ 0, %b0 ]
  %v2 = phi i16* [ %v10, %b1 ], [ %a0, %b0 ]
  %v3 = phi float* [ %v4, %b1 ], [ %a1, %b0 ]
  %v4 = getelementptr inbounds float, float* %v3, i32 1
  %v5 = load float, float* %v3, align 4, !tbaa !0
  %v6 = fmul float %v5, %a2
  %v7 = tail call i32 @llvm.hexagon.F2.conv.sf2w(float %v6)
  %v8 = tail call i32 @llvm.hexagon.A2.sath(i32 %v7)
  %v9 = trunc i32 %v8 to i16
  %v10 = getelementptr inbounds i16, i16* %v2, i32 1
  store i16 %v9, i16* %v2, align 2, !tbaa !4
  %v11 = add nuw nsw i32 %v1, 1
  %v12 = icmp eq i32 %v11, %a3
  br i1 %v12, label %b2, label %b1

b2:                                               ; preds = %b1, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.sath(i32) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.F2.conv.sf2w(float) #0

attributes #0 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"float", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"short", !2, i64 0}
