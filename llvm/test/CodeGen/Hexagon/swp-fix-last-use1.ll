; RUN: llc -march=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b4

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v0 = phi i32 [ undef, %b1 ], [ %v13, %b2 ]
  %v1 = phi i32 [ 0, %b1 ], [ %v20, %b2 ]
  %v2 = zext i32 %v0 to i64
  %v3 = or i64 0, %v2
  %v4 = tail call i64 @llvm.hexagon.S2.lsl.r.vh(i64 %v3, i32 4)
  %v5 = or i64 %v4, -9223231297218904064
  %v6 = lshr i64 %v5, 32
  %v7 = trunc i64 %v6 to i32
  %v8 = tail call i64 @llvm.hexagon.S2.vzxthw(i32 %v7)
  %v9 = lshr i64 %v8, 32
  %v10 = trunc i64 %v9 to i32
  %v11 = tail call i32 @llvm.hexagon.S2.lsr.r.r(i32 %v10, i32 undef)
  %v12 = load i64, ptr undef, align 8, !tbaa !0
  %v13 = trunc i64 %v12 to i32
  %v14 = lshr i64 %v12, 32
  %v15 = trunc i64 %v14 to i32
  %v16 = zext i32 %v11 to i64
  %v17 = shl nuw i64 %v16, 32
  %v18 = or i64 %v17, 0
  %v19 = tail call i32 @llvm.hexagon.S2.vsatwuh(i64 %v18)
  %v20 = add nsw i32 %v1, 1
  %v21 = icmp eq i32 %v20, undef
  br i1 %v21, label %b3, label %b2

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b3, %b0
  %v22 = phi i32 [ %v19, %b3 ], [ undef, %b0 ]
  %v23 = phi i32 [ %v15, %b3 ], [ undef, %b0 ]
  %v24 = zext i32 %v22 to i64
  %v25 = shl nuw i64 %v24, 32
  %v26 = or i64 %v25, 0
  store i64 %v26, ptr undef, align 8, !tbaa !0
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.lsl.r.vh(i64, i32) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.vzxthw(i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.lsr.r.r(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.vsatwuh(i64) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"long long", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
