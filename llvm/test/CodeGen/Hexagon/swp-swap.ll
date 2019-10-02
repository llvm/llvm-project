; RUN: llc -march=hexagon -enable-pipeliner -stats -o /dev/null < %s 2>&1 -pipeliner-experimental-cg=true | FileCheck %s --check-prefix=STATS
; REQUIRES: asserts

; Test that we don't pipeline, incorrectly, the swap operation.

; STATS-NOT: 1 pipeliner   - Number of loops software pipelined

@g0 = common global i32* null, align 4

; Function Attrs: nounwind
define void @f0(i32 %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp sgt i32 %a2, 0
  br i1 %v0, label %b1, label %b4

b1:                                               ; preds = %b0
  %v1 = load i32*, i32** @g0, align 4, !tbaa !0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v2 = phi i32 [ %a0, %b1 ], [ %v9, %b2 ]
  %v3 = phi i32 [ %a2, %b1 ], [ %v11, %b2 ]
  %v4 = phi i32 [ %a1, %b1 ], [ %v10, %b2 ]
  %v5 = getelementptr inbounds i32, i32* %v1, i32 %v2
  %v6 = load i32, i32* %v5, align 4, !tbaa !4
  %v7 = getelementptr inbounds i32, i32* %v1, i32 %v4
  %v8 = load i32, i32* %v7, align 4, !tbaa !4
  store i32 %v8, i32* %v5, align 4, !tbaa !4
  store i32 %v6, i32* %v7, align 4, !tbaa !4
  %v9 = add nsw i32 %v2, 1
  %v10 = add nsw i32 %v4, 1
  %v11 = add nsw i32 %v3, -1
  %v12 = icmp sgt i32 %v11, 0
  br i1 %v12, label %b2, label %b3

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b3, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !2, i64 0}
