; RUN: llc -enable-aa-sched-mi -march=hexagon -mcpu=hexagonv5 -rdf-opt=0 \
; RUN:      < %s | FileCheck %s

; CHECK: {
; CHECK: = memd([[REG0:(r[0-9]+)]]++#8)
; CHECK-NOT: memw([[REG0]]+#0) =
; CHECK: }


define void @f0(ptr %a0) #0 {
b0:
  store i32 -1, ptr %a0, align 8, !tbaa !0
  br label %b4

b1:                                               ; preds = %b3
  unreachable

b2:                                               ; preds = %b3
  ret void

b3:                                               ; preds = %b4
  %v0 = extractelement <2 x i32> %v6, i32 1
  br i1 undef, label %b2, label %b1

b4:                                               ; preds = %b4, %b0
  %v1 = phi <2 x i32> [ %v6, %b4 ], [ zeroinitializer, %b0 ]
  %v2 = phi ptr [ %v9, %b4 ], [ %a0, %b0 ]
  %v3 = phi i32 [ %v7, %b4 ], [ 0, %b0 ]
  %v5 = load <2 x i32>, ptr %v2, align 8
  %v6 = add <2 x i32> %v5, %v1
  %v7 = add nsw i32 %v3, 2
  %v8 = icmp slt i32 %v3, 4
  %v9 = getelementptr i32, ptr %v2, i32 2
  br i1 %v8, label %b4, label %b3
}

attributes #0 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
