; RUN: llc -march=hexagon -disable-block-placement=0 -O2 < %s | FileCheck %s
; CHECK: [[Reg:r[0-9]+]] = {{lsr\(r[0-9]+,#16\)|extractu\(r[0-9]+,#16,#16\)}}
; CHECK-NOT: [[Reg]] = #0
; CHECK: align
; CHECK-NEXT: LBB

target triple = "hexagon"

@g0 = common global [4 x i16] zeroinitializer, align 8

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = alloca i16, align 2
  call void @f1(i16* getelementptr inbounds ([4 x i16], [4 x i16]* @g0, i32 0, i32 0), i16* %v0) #0
  %v1 = load i16, i16* %v0, align 2, !tbaa !0
  %v2 = icmp slt i16 %v1, -15
  br i1 %v2, label %b1, label %b4

b1:                                               ; preds = %b0
  %v3 = load i32, i32* bitcast ([4 x i16]* @g0 to i32*), align 8
  %v4 = trunc i32 %v3 to i16
  %v5 = lshr i32 %v3, 16
  %v6 = trunc i32 %v5 to i16
  %v7 = load i32, i32* bitcast (i16* getelementptr inbounds ([4 x i16], [4 x i16]* @g0, i32 0, i32 2) to i32*), align 4
  %v8 = trunc i32 %v7 to i16
  %v9 = lshr i32 %v7, 16
  %v10 = trunc i32 %v9 to i16
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v11 = phi i16 [ %v1, %b1 ], [ %v20, %b2 ]
  %v12 = phi i16 [ %v10, %b1 ], [ 0, %b2 ]
  %v13 = phi i16 [ %v8, %b1 ], [ %v12, %b2 ]
  %v14 = phi i16 [ %v6, %b1 ], [ %v13, %b2 ]
  %v15 = phi i16 [ %v4, %b1 ], [ %v14, %b2 ]
  %v16 = phi i16 [ 0, %b1 ], [ %v19, %b2 ]
  %v17 = icmp ne i16 %v16, 0
  %v18 = zext i1 %v17 to i16
  %v19 = or i16 %v15, %v18
  %v20 = add i16 %v11, 16
  %v21 = icmp slt i16 %v20, -15
  br i1 %v21, label %b2, label %b3

b3:                                               ; preds = %b2
  store i16 %v14, i16* getelementptr inbounds ([4 x i16], [4 x i16]* @g0, i32 0, i32 0), align 8, !tbaa !0
  store i16 %v13, i16* getelementptr inbounds ([4 x i16], [4 x i16]* @g0, i32 0, i32 1), align 2, !tbaa !0
  store i16 %v12, i16* getelementptr inbounds ([4 x i16], [4 x i16]* @g0, i32 0, i32 2), align 4, !tbaa !0
  store i16 0, i16* getelementptr inbounds ([4 x i16], [4 x i16]* @g0, i32 0, i32 3), align 2, !tbaa !0
  store i16 %v20, i16* %v0, align 2, !tbaa !0
  br label %b4

b4:                                               ; preds = %b3, %b0
  %v22 = phi i16 [ %v19, %b3 ], [ 0, %b0 ]
  call void @f2(i16* getelementptr inbounds ([4 x i16], [4 x i16]* @g0, i32 0, i32 0), i16 signext %v22) #0
  ret i32 0
}

declare void @f1(i16*, i16*) #0

declare void @f2(i16*, i16 signext) #0

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
