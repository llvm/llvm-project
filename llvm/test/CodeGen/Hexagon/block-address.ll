; RUN: llc -mtriple=hexagon < %s
; RUN: llc -mtriple=hexagon -hexagon-small-data-threshold=0 < %s
; REQUIRES: asserts

@g0 = external global ptr

; Function Attrs: nounwind
define i32 @f0(i32 %a0, i32 %a1) #0 {
b0:
  %v0 = load ptr, ptr @g0, align 4, !tbaa !0
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v1 = phi ptr [ %v0, %b0 ], [ %v5, %b1 ]
  %v2 = phi i32 [ %a0, %b0 ], [ %v3, %b1 ]
  %v3 = add nsw i32 %v2, 10
  %v4 = tail call i32 @f1(ptr %v1, ptr blockaddress(@f0, %b1), ptr blockaddress(@f0, %b2)) #0
  %v5 = load ptr, ptr @g0, align 4, !tbaa !0
  indirectbr ptr %v5, [label %b1, label %b2]

b2:                                               ; preds = %b1
  %v6 = add nsw i32 %v2, 19
  %v7 = add i32 %v2, 69
  %v8 = add i32 %v7, %v3
  %v9 = mul nsw i32 %v8, %v6
  ret i32 %v9
}

declare i32 @f1(ptr, ptr, ptr)

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
