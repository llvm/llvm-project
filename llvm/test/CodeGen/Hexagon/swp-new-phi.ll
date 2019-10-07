; RUN: llc -march=hexagon -enable-pipeliner -pipeliner-max-stages=2 < %s -pipeliner-experimental-cg=true | FileCheck %s

; Test that the generatePhi code doesn't rename a a Phi instruction that's defined
; in the same block.  The bug causes a Phi to incorrectly depend on another Phi.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: memh([[REG0:(r[0-9]+)]]++#2:circ
; CHECK: = mem{{u?}}h([[REG0]]+#0)
; CHECK: endloop0

; Function Attrs: argmemonly nounwind
declare i8* @llvm.hexagon.circ.sthhi(i8*, i32, i32, i32) #1

; Function Attrs: nounwind optsize
define signext i16 @f0(i16* %a0, i16* %a1, i16 signext %a2, i16 signext %a3) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i16* [ %v10, %b1 ], [ %a1, %b0 ]
  %v1 = phi i32 [ %v13, %b1 ], [ 1, %b0 ]
  %v2 = phi i16 [ %v12, %b1 ], [ 0, %b0 ]
  %v3 = bitcast i16* %v0 to i8*
  %v4 = add nsw i32 %v1, 10
  %v5 = getelementptr inbounds i16, i16* %a0, i32 %v4
  %v6 = load i16, i16* %v5, align 2, !tbaa !0
  %v7 = sext i16 %v6 to i32
  %v8 = add nsw i32 %v7, 40000
  %v9 = tail call i8* @llvm.hexagon.circ.sthhi(i8* %v3, i32 %v8, i32 117441022, i32 2)
  %v10 = bitcast i8* %v9 to i16*
  %v11 = load i16, i16* %v10, align 2, !tbaa !0
  %v12 = add i16 %v11, %v2
  %v13 = add i32 %v1, 1
  %v14 = icmp eq i32 %v13, 1000
  br i1 %v14, label %b2, label %b1

b2:                                               ; preds = %b1
  br label %b3

b3:                                               ; preds = %b2
  ret i16 %v12
}

attributes #0 = { nounwind optsize "target-cpu"="hexagonv55" }
attributes #1 = { argmemonly nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
