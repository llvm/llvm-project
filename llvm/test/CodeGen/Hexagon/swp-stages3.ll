; RUN: llc -march=hexagon -enable-pipeliner -pipeliner-max-stages=2 < %s
; REQUIRES: asserts

; Test that the compiler doesn't seg fault due to incorrect names in epilog.

; Function Attrs: nounwind
define void @f0(i16* nocapture %a0, i16* nocapture %a1, i16 signext %a2) #0 {
b0:
  %v0 = icmp sgt i16 %a2, 0
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  %v1 = sext i16 %a2 to i32
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v2 = phi i16 [ %v16, %b2 ], [ undef, %b1 ]
  %v3 = phi i32 [ %v17, %b2 ], [ 0, %b1 ]
  %v4 = phi i16* [ undef, %b2 ], [ %a0, %b1 ]
  %v5 = phi i16* [ %v6, %b2 ], [ %a1, %b1 ]
  %v6 = getelementptr inbounds i16, i16* %v5, i32 1
  %v7 = load i16, i16* %v5, align 2, !tbaa !0
  %v8 = sext i16 %v7 to i32
  %v9 = tail call i32 @llvm.hexagon.A2.aslh(i32 %v8)
  %v10 = tail call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v9, i32 undef)
  %v11 = sext i16 %v2 to i32
  %v12 = tail call i32 @llvm.hexagon.M2.mpy.nac.sat.ll.s1(i32 %v10, i32 %v11, i32 undef)
  %v13 = tail call i32 @llvm.hexagon.S2.asl.r.r.sat(i32 %v12, i32 undef)
  %v14 = tail call i32 @llvm.hexagon.A2.addsat(i32 %v13, i32 32768)
  %v15 = tail call i32 @llvm.hexagon.A2.asrh(i32 %v14)
  %v16 = trunc i32 %v15 to i16
  store i16 %v16, i16* %v4, align 2, !tbaa !0
  %v17 = add i32 %v3, 1
  %v18 = icmp eq i32 %v17, %v1
  br i1 %v18, label %b3, label %b2

b3:                                               ; preds = %b2, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.aslh(i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.asr.r.r.sat(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.nac.sat.ll.s1(i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.asl.r.r.sat(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.asrh(i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.addsat(i32, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
