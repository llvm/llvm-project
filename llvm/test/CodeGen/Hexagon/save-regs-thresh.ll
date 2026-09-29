; RUN: llc -mtriple=hexagon -O2 -spill-func-threshold=4 < %s | FileCheck %s --check-prefix=NOSAVE
; RUN: llc -mtriple=hexagon -O2 -spill-func-threshold=2 < %s | FileCheck %s --check-prefix=SAVE
; NOSAVE-NOT: call __save_r16_
; SAVE: call __save_r16_

target triple = "hexagon"

%s.0 = type { %s.1, [50 x %s.2], i8, i32 }
%s.1 = type { i8, i8, i8, i8, i8, i8, i8, i8, [2 x i8], [2 x i8], [4 x i8] }
%s.2 = type { %s.3, [16 x i8] }
%s.3 = type { %s.4, %s.5 }
%s.4 = type { i8, i8, [2 x i8], [4 x i8] }
%s.5 = type { i16, i16 }

@g0 = private unnamed_addr constant [21 x i8] c"....................\00", align 1
@g1 = internal unnamed_addr global [1 x ptr] zeroinitializer, align 4

; Function Attrs: nounwind
define void @f0(i8 zeroext %a0, ptr nocapture %a1) #0 {
b0:
  %v0 = tail call ptr @f1(i8 zeroext %a0, i32 1424, ptr @g0, i32 118) #0
  %v2 = zext i8 %a0 to i32
  %v3 = getelementptr inbounds [1 x ptr], ptr @g1, i32 0, i32 %v2
  store ptr %v0, ptr %v3, align 4, !tbaa !0
  store ptr %v0, ptr %a1, align 4, !tbaa !0
  ret void
}

declare ptr @f1(i8 zeroext, i32, ptr, i32)

; Function Attrs: nounwind
define void @f2(i8 zeroext %a0) #0 {
b0:
  %v0 = zext i8 %a0 to i32
  %v1 = getelementptr inbounds [1 x ptr], ptr @g1, i32 0, i32 %v0
  %v2 = load ptr, ptr %v1, align 4, !tbaa !0
  tail call void @f3(i8 zeroext %a0, ptr %v2, ptr @g0, i32 142) #0
  store ptr null, ptr %v1, align 4, !tbaa !0
  ret void
}

declare void @f3(i8 zeroext, ptr, ptr, i32)

; Function Attrs: nounwind
define void @f4(i8 zeroext %a0, i8 zeroext %a1, i8 zeroext %a2, i8 zeroext %a3, i8 zeroext %a4) #0 {
b0:
  %v0 = alloca [7 x i32], align 4
  %v1 = zext i8 %a0 to i32
  %v2 = getelementptr inbounds [1 x ptr], ptr @g1, i32 0, i32 %v1
  %v3 = load ptr, ptr %v2, align 4, !tbaa !0
  %v4 = getelementptr inbounds %s.0, ptr %v3, i32 0, i32 3
  %v5 = load i32, ptr %v4, align 4, !tbaa !4
  %v6 = and i32 %v5, 8
  %v7 = icmp eq i32 %v6, 0
  br i1 %v7, label %b2, label %b1

b1:                                               ; preds = %b0
  %v10 = call i32 @f5() #0
  %v11 = getelementptr [7 x i32], ptr %v0, i32 0, i32 1
  store i32 %v10, ptr %v11, align 4
  %v12 = call zeroext i16 @f6(i8 zeroext %a0) #0
  %v13 = zext i16 %v12 to i32
  %v14 = shl nuw i32 %v13, 16
  %v15 = or i32 %v14, 260
  store i32 %v15, ptr %v0, align 4
  %v16 = zext i8 %a1 to i32
  %v17 = getelementptr [7 x i32], ptr %v0, i32 0, i32 2
  %v18 = zext i8 %a2 to i32
  %v19 = shl nuw nsw i32 %v18, 12
  %v20 = zext i8 %a3 to i32
  %v21 = shl nuw nsw i32 %v20, 16
  %v22 = and i32 %v21, 458752
  %v23 = and i32 %v19, 61440
  %v24 = zext i8 %a4 to i32
  %v25 = shl nuw nsw i32 %v24, 19
  %v26 = and i32 %v25, 3670016
  %v27 = or i32 %v23, %v16
  %v28 = or i32 %v27, %v22
  %v29 = or i32 %v28, %v26
  %v30 = call zeroext i8 @f7(i8 zeroext %a0, i8 zeroext %a1) #0
  %v31 = zext i8 %v30 to i32
  %v32 = shl nuw nsw i32 %v31, 8
  %v33 = and i32 %v32, 3840
  %v34 = or i32 %v33, %v29
  store i32 %v34, ptr %v17, align 4
  %v35 = call i32 @f8(i32 %v1, ptr %v0) #0
  br label %b2

b2:                                               ; preds = %b1, %b0
  ret void
}

declare i32 @f5()

declare zeroext i16 @f6(i8 zeroext)

declare zeroext i8 @f7(i8 zeroext, i8 zeroext)

declare i32 @f8(...)

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"long", !2}
