; RUN: llc -mtriple=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; Test that the pipeliner doesn't ICE due to incorrect PHI
; generation code that attemps to reuse an exsting PHI.
; Similar to the other swp-epillog-reuse test, but from a
; different test case.

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.asl.r.r.sat(i32, i32) #0

; Function Attrs: nounwind
define void @f0() #1 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  br i1 undef, label %b2, label %b1

b2:                                               ; preds = %b2, %b1
  br i1 undef, label %b3, label %b2

b3:                                               ; preds = %b3, %b2
  %v0 = phi i16 [ %v10, %b3 ], [ undef, %b2 ]
  %v1 = phi i16 [ %v0, %b3 ], [ undef, %b2 ]
  %v2 = phi i32 [ %v26, %b3 ], [ undef, %b2 ]
  %v3 = phi ptr [ undef, %b3 ], [ undef, %b2 ]
  %v4 = phi ptr [ %v5, %b3 ], [ undef, %b2 ]
  %v5 = getelementptr inbounds i16, ptr %v4, i32 1
  %v6 = sext i16 %v1 to i32
  %v7 = call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 0, i32 %v6, i32 undef)
  %v8 = getelementptr inbounds i16, ptr %v4, i32 2
  %v9 = call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 %v7, i32 undef, i32 undef)
  %v10 = load i16, ptr %v8, align 2, !tbaa !0
  %v11 = call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 %v9, i32 undef, i32 undef)
  %v12 = call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v11, i32 undef)
  %v13 = getelementptr [166 x i32], ptr null, i32 0, i32 undef
  %v14 = load i32, ptr %v13, align 4
  %v15 = call i64 @llvm.hexagon.M2.dpmpyss.s0(i32 %v14, i32 undef)
  %v16 = call i64 @llvm.hexagon.S2.asr.i.p(i64 %v15, i32 15)
  %v17 = call i32 @llvm.hexagon.A2.sat(i64 %v16)
  %v18 = call i32 @llvm.hexagon.A2.subsat(i32 %v12, i32 %v17)
  %v19 = getelementptr [166 x i32], ptr null, i32 0, i32 undef
  %v20 = load i32, ptr %v19, align 4
  %v21 = call i64 @llvm.hexagon.M2.dpmpyss.s0(i32 %v20, i32 0)
  %v22 = call i64 @llvm.hexagon.S2.asr.i.p(i64 %v21, i32 15)
  %v23 = call i32 @llvm.hexagon.A2.sat(i64 %v22)
  %v24 = call i32 @llvm.hexagon.A2.subsat(i32 %v18, i32 %v23)
  %v25 = call i32 @llvm.hexagon.S2.asl.r.r.sat(i32 %v24, i32 undef)
  store i32 %v25, ptr %v3, align 4, !tbaa !4
  %v26 = add i32 %v2, 1
  %v27 = icmp eq i32 %v26, 164
  br i1 %v27, label %b4, label %b3

b4:                                               ; preds = %b3
  call void @llvm.trap()
  unreachable
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32, i32, i32) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.asr.r.r.sat(i32, i32) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.sat(i64) #0

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.asr.i.p(i64, i32) #0

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.dpmpyss.s0(i32, i32) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.subsat(i32, i32) #0

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #2

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv55" }
attributes #2 = { noreturn nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"long", !2, i64 0}
