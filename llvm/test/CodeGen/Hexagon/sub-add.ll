; RUN: llc -mtriple=hexagon -enable-timing-class-latency=true < %s | FileCheck -check-prefix=CHECK-ONE %s
; REQUIRES: asserts
; Check there is no assert when enabling enable-timing-class-latency
; CHECK-ONE: f0:

; RUN: llc -mtriple=hexagon < %s | FileCheck -check-prefix=CHECK %s
; CHECK: add(r{{[0-9]*}},sub(#1,r{{[0-9]*}})
; CHECK: call f1

target triple = "hexagon"

%s.0 = type { i16, i16, i16, ptr, i16, i16, i16, i16, i16, i16, i32, i32, i16, ptr, i16, ptr, i16, ptr, ptr, ptr, i16, ptr, ptr, i16, ptr, i16, ptr, ptr, %s.5, %s.4, %s.5, %s.5, i32, i32, i32, %s.6, i32, i32, i16, %s.7, %s.7 }
%s.1 = type { i16, i16, i16, i16 }
%s.2 = type { i16, i16 }
%s.3 = type { i16, i16, i16, i16 }
%s.4 = type { i32, i32, ptr }
%s.5 = type { i32, i32, ptr }
%s.6 = type { i16, i16, i16, i16 }
%s.7 = type { i16, i16, i16, i16, i16, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }

; Function Attrs: nounwind
define i32 @f0(ptr %a0) #0 {
b0:
  %v0 = alloca i16, align 2
  %v1 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 12
  %v2 = load i16, ptr %v1, align 2, !tbaa !0
  %v3 = icmp sgt i16 %v2, 0
  br i1 %v3, label %b1, label %b9

b1:                                               ; preds = %b0
  %v4 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 17
  %v5 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 29
  br label %b2

b2:                                               ; preds = %b7, %b1
  %v6 = phi i16 [ %v2, %b1 ], [ %v23, %b7 ]
  %v7 = phi i32 [ 0, %b1 ], [ %v25, %b7 ]
  %v8 = phi i16 [ 1, %b1 ], [ %v26, %b7 ]
  %v9 = load ptr, ptr %v4, align 4, !tbaa !4
  %v10 = getelementptr inbounds %s.3, ptr %v9, i32 %v7, i32 0
  %v11 = load i16, ptr %v10, align 2, !tbaa !0
  %v12 = getelementptr inbounds %s.3, ptr %v9, i32 %v7, i32 1
  %v13 = load i16, ptr %v12, align 2, !tbaa !0
  %v14 = icmp sgt i16 %v11, %v13
  br i1 %v14, label %b6, label %b3

b3:                                               ; preds = %b2
  %v15 = sext i16 %v11 to i32
  %v16 = sext i16 %v13 to i32
  %v17 = add i32 %v16, 1
  br label %b4

b4:                                               ; preds = %b4, %b3
  %v18 = phi i32 [ %v15, %b3 ], [ %v20, %b4 ]
  %v19 = call i32 @f1(ptr %v5, i32 %v7, i32 undef, ptr %v0) #0
  %v20 = add i32 %v18, 1
  %v21 = icmp eq i32 %v20, %v17
  br i1 %v21, label %b5, label %b4

b5:                                               ; preds = %b4
  %v22 = load i16, ptr %v1, align 2, !tbaa !0
  br label %b6

b6:                                               ; preds = %b5, %b2
  %v23 = phi i16 [ %v22, %b5 ], [ %v6, %b2 ]
  %v24 = icmp slt i16 %v8, %v23
  br i1 %v24, label %b7, label %b8

b7:                                               ; preds = %b6
  %v25 = sext i16 %v8 to i32
  %v26 = add i16 %v8, 1
  br label %b2

b8:                                               ; preds = %b6
  br label %b9

b9:                                               ; preds = %b8, %b0
  ret i32 0
}

declare i32 @f1(...)

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"any pointer", !2}
