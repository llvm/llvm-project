; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: cmpb.gtu

target triple = "hexagon"

%s.0 = type { ptr, ptr, ptr, ptr }
%s.1 = type { ptr, ptr }
%s.2 = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%s.3 = type { %s.4, [2 x %s.5], i8, %s.7, %s.19, i8, %s.8, i16, [6 x %s.14], %s.17, %s.18, %s.19 }
%s.4 = type { i16, i8, ptr }
%s.5 = type { i16, i8, i8, i8, i8, i8, i8, i8, i16, i8, i8, i8, i16, %s.6 }
%s.6 = type { i8, i16, i16, i8, i8 }
%s.7 = type { i8, i8, i8, i8, i64, i64 }
%s.8 = type { i16, %s.9, i32, ptr, i8, i8 }
%s.9 = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
%s.10 = type { i8, [14 x i8], [14 x ptr] }
%s.11 = type { i8, i8, i8, i8, i8, i16, i8, i8, i8, i8, i8, i8, i8, i8, %s.12, %s.13 }
%s.12 = type { i16, i8, i8, i8, i16, i8, i8 }
%s.13 = type { i8, i8, i8, i8, i8, i8 }
%s.14 = type { i16, %s.15 }
%s.15 = type { i16, i8, i16, i16, i16, i16, [1 x %s.16], i8, i8, i8, i32 }
%s.16 = type { i8, i16, i16, i8 }
%s.17 = type { i16, i16, i16, i8, i8 }
%s.18 = type { i8, i8, i32 }
%s.19 = type { i8, i8, i8, i8 }
%s.22 = type { %s.23, %s.24 }
%s.23 = type { i8, i8, i8, i8, i8, i8, i8, ptr, ptr, ptr, i8 }
%s.24 = type { i16, i16, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i16, i8, i8 }
%s.25 = type { %s.26 }
%s.26 = type { i8, i8, i32 }

@g0 = external global %s.0
@g1 = external global %s.1
@g2 = external global %s.2
@g3 = external global %s.0
@g4 = external global %s.0
@g5 = external global %s.0
@g6 = external global %s.0
@g7 = external global %s.0
@g8 = external global %s.1
@g9 = external global %s.2
@g10 = external global %s.0
@g11 = external global %s.0
@g12 = external global %s.1
@g13 = external global %s.2
@g14 = external global %s.0
@g15 = external global %s.1
@g16 = external global %s.2
@g17 = external global %s.0
@g18 = external global %s.2
@g19 = common global [6 x %s.22] zeroinitializer, align 8
@g20 = common global %s.25 zeroinitializer, align 4

declare void @f0()

declare void @f1()

declare void @f2()

declare void @f3()

declare void @f4()

declare void @f5(i8 zeroext)

declare void @f6()

; Function Attrs: nounwind
define void @f7() #0 {
b0:
  %v0 = load i8, ptr getelementptr inbounds (%s.25, ptr @g20, i32 0, i32 0, i32 1), align 1, !tbaa !0
  %v1 = icmp eq i8 %v0, 1
  br label %b1

b1:                                               ; preds = %b5, %b0
  %v2 = phi i32 [ 0, %b0 ], [ %v14, %b5 ]
  %v3 = getelementptr inbounds [6 x %s.22], ptr @g19, i32 0, i32 %v2, i32 1, i32 4
  %v4 = load i8, ptr %v3, align 2, !tbaa !0
  %v5 = icmp eq i8 %v4, 1
  br i1 %v5, label %b2, label %b5

b2:                                               ; preds = %b1
  br i1 %v1, label %b3, label %b4

b3:                                               ; preds = %b2
  %v6 = getelementptr inbounds [6 x %s.22], ptr @g19, i32 0, i32 %v2, i32 1, i32 6
  %v7 = load i8, ptr %v6, align 4, !tbaa !0
  %v8 = add i8 %v7, -2
  %v9 = icmp ult i8 %v8, 44
  br i1 %v9, label %b5, label %b4

b4:                                               ; preds = %b3, %b2
  %v10 = shl i32 1, %v2
  %v11 = load i32, ptr getelementptr inbounds (%s.25, ptr @g20, i32 0, i32 0, i32 2), align 4, !tbaa !3
  %v12 = or i32 %v11, %v10
  store i32 %v12, ptr getelementptr inbounds (%s.25, ptr @g20, i32 0, i32 0, i32 2), align 4, !tbaa !3
  %v13 = getelementptr inbounds [6 x %s.22], ptr @g19, i32 0, i32 %v2, i32 1, i32 13
  store i8 1, ptr %v13, align 4, !tbaa !0
  br label %b5

b5:                                               ; preds = %b4, %b3, %b1
  %v14 = add i32 %v2, 1
  %v15 = trunc i32 %v14 to i8
  %v16 = icmp eq i8 %v15, 6
  br i1 %v16, label %b6, label %b1

b6:                                               ; preds = %b5
  ret void
}

declare void @f8(i8 zeroext)

declare void @f9()

declare void @f10()

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!4, !4, i64 0}
!4 = !{!"long", !1}
