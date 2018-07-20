; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK-NOT: r{{[0-9]+}}:{{[0-9]+}} = insert(r{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})

@g0 = common global [512 x i16] zeroinitializer, align 8
@g1 = common global [512 x i8] zeroinitializer, align 8
@g2 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 8

; Function Attrs: nounwind
declare i32 @f0(i8* nocapture, ...) #0

; Function Attrs: nounwind
define i32 @f1() #0 {
b0:
  br label %b4

b1:                                               ; preds = %b3, %b1
  %v0 = phi i32 [ 0, %b3 ], [ %v5, %b1 ]
  %v1 = getelementptr [512 x i8], [512 x i8]* @g1, i32 0, i32 %v0
  %v2 = load i8, i8* %v1, align 1, !tbaa !0
  %v3 = zext i8 %v2 to i32
  %v4 = tail call i32 (i8*, ...) @f0(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g2, i32 0, i32 0), i32 %v3) #0
  %v5 = add nsw i32 %v0, 1
  %v6 = icmp eq i32 %v5, 512
  br i1 %v6, label %b2, label %b1

b2:                                               ; preds = %b1
  ret i32 0

b3:                                               ; preds = %b4
  tail call void @f2(i16* getelementptr inbounds ([512 x i16], [512 x i16]* @g0, i32 0, i32 0), i8* getelementptr inbounds ([512 x i8], [512 x i8]* @g1, i32 0, i32 0)) #0
  br label %b1

b4:                                               ; preds = %b4, %b0
  %v7 = phi i64 [ 0, %b0 ], [ %v10, %b4 ]
  %v8 = phi <2 x i32> [ <i32 0, i32 1>, %b0 ], [ %v12, %b4 ]
  %v9 = phi <2 x i32> [ <i32 2, i32 3>, %b0 ], [ %v13, %b4 ]
  %v10 = add nsw i64 %v7, 4
  %v11 = trunc i64 %v7 to i32
  %v12 = add <2 x i32> %v8, <i32 4, i32 4>
  %v13 = add <2 x i32> %v9, <i32 4, i32 4>
  %v14 = mul <2 x i32> %v8, <i32 7, i32 7>
  %v15 = mul <2 x i32> %v9, <i32 7, i32 7>
  %v16 = add <2 x i32> %v14, <i32 -840, i32 -840>
  %v17 = add <2 x i32> %v15, <i32 -840, i32 -840>
  %v18 = trunc <2 x i32> %v16 to <2 x i16>
  %v19 = trunc <2 x i32> %v17 to <2 x i16>
  %v20 = shufflevector <2 x i16> %v18, <2 x i16> %v19, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v21 = getelementptr [512 x i16], [512 x i16]* @g0, i32 0, i32 %v11
  %v22 = bitcast i16* %v21 to <4 x i16>*
  store <4 x i16> %v20, <4 x i16>* %v22, align 8
  %v23 = icmp slt i64 %v10, 512
  br i1 %v23, label %b4, label %b3
}

declare void @f2(i16*, i8*)

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
