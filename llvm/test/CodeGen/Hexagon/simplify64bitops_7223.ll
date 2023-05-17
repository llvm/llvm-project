; RUN: llc -march=hexagon -enable-pipeliner=false < %s | FileCheck %s
; RUN: llc -march=hexagon -enable-pipeliner < %s
; REQUIRES: asserts
; CHECK-NOT: and(
; CHECK-NOT: or(
; CHECK-NOT: combine(0
; CHECK: add
; CHECK: add(
; CHECK-NEXT: memuh(
; CHECK-NEXT: endloop

%s.22 = type { i64 }

@g0 = common global i32 0, align 4

; Function Attrs: nounwind
define i64 @f0(ptr nocapture %a0, i32 %a1) #0 {
b0:
  %v1 = load i16, ptr %a0, align 2, !tbaa !0
  %v2 = zext i16 %v1 to i64
  %v3 = icmp sgt i32 %a1, 0
  br i1 %v3, label %b1, label %b4

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v4 = phi ptr [ %v8, %b2 ], [ %a0, %b1 ]
  %v5 = phi i32 [ %v10, %b2 ], [ undef, %b1 ]
  %v6 = phi i32 [ %v15, %b2 ], [ 0, %b1 ]
  %v7 = phi i64 [ %v14, %b2 ], [ %v2, %b1 ]
  %v8 = getelementptr inbounds i16, ptr %v4, i32 1
  %v9 = trunc i64 %v7 to i32
  %v10 = add i32 %v5, %v9
  %v11 = load i16, ptr %v8, align 2, !tbaa !0
  %v12 = zext i16 %v11 to i64
  %v13 = and i64 %v7, -4294967296
  %v14 = or i64 %v12, %v13
  %v15 = add nsw i32 %v6, 1
  %v16 = icmp eq i32 %v15, %a1
  br i1 %v16, label %b3, label %b2

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b3, %b0
  %v17 = phi i32 [ undef, %b0 ], [ %v10, %b3 ]
  %v18 = phi i64 [ %v2, %b0 ], [ %v14, %b3 ]
  store volatile i32 %v17, ptr @g0, align 4, !tbaa !4
  ret i64 %v18
}

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"long", !2}
