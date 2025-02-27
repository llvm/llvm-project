; RUN: llc -mtriple=hexagon < %s
; REQUIRES: asserts

target triple = "hexagon"

%s.0 = type { %s.1 }
%s.1 = type { i32, ptr }

@g0 = external unnamed_addr constant [6 x [2 x i32]], align 8
@g1 = external constant %s.0, align 4

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  unreachable

b2:                                               ; preds = %b0
  br i1 undef, label %b3, label %b4

b3:                                               ; preds = %b2
  switch i32 undef, label %b4 [
    i32 10, label %b5
  ]

b4:                                               ; preds = %b3, %b2
  unreachable

b5:                                               ; preds = %b3
  br i1 undef, label %b7, label %b6

b6:                                               ; preds = %b5
  switch i32 undef, label %b40 [
    i32 10, label %b38
    i32 5, label %b8
  ]

b7:                                               ; preds = %b5
  unreachable

b8:                                               ; preds = %b6
  br i1 undef, label %b9, label %b37

b9:                                               ; preds = %b8
  br i1 undef, label %b10, label %b37

b10:                                              ; preds = %b9
  br i1 undef, label %b12, label %b11

b11:                                              ; preds = %b10
  unreachable

b12:                                              ; preds = %b10
  br i1 undef, label %b13, label %b17

b13:                                              ; preds = %b12
  br i1 undef, label %b14, label %b15

b14:                                              ; preds = %b13
  unreachable

b15:                                              ; preds = %b13
  br i1 undef, label %b16, label %b18

b16:                                              ; preds = %b15
  unreachable

b17:                                              ; preds = %b12
  unreachable

b18:                                              ; preds = %b15
  br i1 undef, label %b19, label %b20

b19:                                              ; preds = %b18
  br label %b21

b20:                                              ; preds = %b18
  unreachable

b21:                                              ; preds = %b35, %b19
  %v0 = phi i32 [ 0, %b19 ], [ %v43, %b35 ]
  %v1 = phi i32 [ 0, %b19 ], [ %v44, %b35 ]
  %v2 = phi i16 [ undef, %b19 ], [ %v42, %b35 ]
  %v3 = trunc i32 %v1 to i10
  %v4 = lshr i10 366, %v3
  %v5 = and i10 %v4, 1
  %v6 = icmp eq i10 %v5, 0
  br i1 %v6, label %b35, label %b22

b22:                                              ; preds = %b21
  %v7 = load i32, ptr undef, align 4, !tbaa !0
  %v8 = load i32, ptr undef, align 4, !tbaa !4
  %v9 = load i32, ptr undef, align 4, !tbaa !4
  %v10 = icmp ne i32 %v8, 0
  %v11 = and i1 %v10, undef
  %v12 = and i1 undef, %v11
  br i1 %v12, label %b23, label %b24

b23:                                              ; preds = %b22
  %v13 = mul nsw i32 %v9, %v9
  %v14 = sdiv i32 %v13, undef
  %v15 = trunc i32 %v14 to i16
  br label %b24

b24:                                              ; preds = %b23, %b22
  %v16 = phi i16 [ %v15, %b23 ], [ 0, %b22 ]
  %v17 = icmp ugt i16 %v16, undef
  %v18 = zext i1 %v17 to i32
  %v19 = add nsw i32 %v18, %v0
  %v20 = load i8, ptr undef, align 4, !tbaa !6
  %v21 = zext i8 %v20 to i32
  %v22 = sub nsw i32 6, %v21
  %v23 = add nsw i32 %v22, -1
  br i1 false, label %b39, label %b25, !prof !19

b25:                                              ; preds = %b24
  %v24 = getelementptr inbounds [6 x [2 x i32]], ptr @g0, i32 0, i32 %v21, i32 0
  %v25 = load i32, ptr %v24, align 8, !tbaa !0
  %v26 = icmp eq i32 undef, %v25
  br i1 %v26, label %b26, label %b27

b26:                                              ; preds = %b25
  br i1 undef, label %b32, label %b27

b27:                                              ; preds = %b26, %b25
  %v27 = getelementptr inbounds [6 x [2 x i32]], ptr @g0, i32 0, i32 %v23, i32 0
  %v28 = load i32, ptr %v27, align 8, !tbaa !0
  %v29 = icmp eq i32 undef, %v28
  br i1 %v29, label %b28, label %b29

b28:                                              ; preds = %b27
  br i1 undef, label %b32, label %b29

b29:                                              ; preds = %b28, %b27
  %v30 = load i32, ptr undef, align 4, !tbaa !4
  %v31 = load i32, ptr undef, align 4, !tbaa !4
  %v32 = icmp ne i32 %v30, 0
  %v33 = and i1 %v32, undef
  %v34 = and i1 undef, %v33
  br i1 %v34, label %b30, label %b31

b30:                                              ; preds = %b29
  %v35 = mul nsw i32 %v31, %v31
  %v36 = sdiv i32 %v35, 0
  %v37 = trunc i32 %v36 to i16
  br label %b31

b31:                                              ; preds = %b30, %b29
  %v38 = phi i16 [ %v37, %b30 ], [ 0, %b29 ]
  br label %b32

b32:                                              ; preds = %b31, %b28, %b26
  %v39 = phi i16 [ %v38, %b31 ], [ %v2, %b28 ], [ %v2, %b26 ]
  br i1 undef, label %b33, label %b34

b33:                                              ; preds = %b32
  call void (ptr, i32, ...) @f1(ptr nonnull @g1, i32 6, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 %v7) #0
  br label %b34

b34:                                              ; preds = %b33, %b32
  %v40 = icmp slt i32 %v19, 0
  %v41 = and i1 %v40, undef
  br i1 %v41, label %b35, label %b36

b35:                                              ; preds = %b34, %b21
  %v42 = phi i16 [ %v2, %b21 ], [ %v39, %b34 ]
  %v43 = phi i32 [ %v0, %b21 ], [ %v19, %b34 ]
  %v44 = add nuw nsw i32 %v1, 1
  br label %b21

b36:                                              ; preds = %b34
  unreachable

b37:                                              ; preds = %b9, %b8
  unreachable

b38:                                              ; preds = %b6
  unreachable

b39:                                              ; preds = %b24
  unreachable

b40:                                              ; preds = %b6
  ret void
}

; Function Attrs: nounwind
declare void @f1(ptr, i32, ...) #0

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"long", !2, i64 0}
!6 = !{!7, !2, i64 136}
!7 = !{!"x", !8, i64 0, !9, i64 8, !11, i64 52, !14, i64 88, !2, i64 116, !2, i64 117, !18, i64 118, !15, i64 128, !15, i64 132, !2, i64 136, !2, i64 140, !2, i64 180, !12, i64 220}
!8 = !{!"", !2, i64 0, !2, i64 1, !2, i64 2, !2, i64 3, !2, i64 4, !2, i64 5}
!9 = !{!"", !2, i64 0, !5, i64 4, !5, i64 8, !5, i64 12, !5, i64 16, !5, i64 20, !5, i64 24, !10, i64 28, !2, i64 32, !2, i64 33, !10, i64 36, !5, i64 40}
!10 = !{!"any pointer", !2, i64 0}
!11 = !{!"", !5, i64 0, !2, i64 4, !12, i64 20, !2, i64 32}
!12 = !{!"", !13, i64 0, !13, i64 2, !13, i64 4, !13, i64 6, !13, i64 8, !13, i64 10}
!13 = !{!"short", !2, i64 0}
!14 = !{!"", !15, i64 0, !13, i64 2, !13, i64 4, !16, i64 8}
!15 = !{!"", !2, i64 0}
!16 = !{!"", !1, i64 0, !2, i64 4, !2, i64 5, !17, i64 8}
!17 = !{!"", !2, i64 0, !5, i64 4, !5, i64 8}
!18 = !{!"", !2, i64 0, !2, i64 1, !2, i64 2, !2, i64 3, !8, i64 4}
!19 = !{!"branch_weights", i32 4, i32 64}
