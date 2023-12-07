; RUN: llc -march=hexagon -enable-pipeliner=true -stats -o /dev/null < %s \
; RUN:      2>&1 -pipeliner-experimental-cg=true | FileCheck %s --check-prefix=STATS
; REQUIRES: asserts

; Test that we do not schedule chained references too far apart,
; which enables the loop to be pipelined. In this test, the loop should
; not be pipelined when the chained references are constrained correctly.

; STATS-NOT: 1 pipeliner   - Number of loops software pipelined

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br label %b2

b1:                                               ; preds = %b6
  br label %b7

b2:                                               ; preds = %b6, %b0
  br label %b3

b3:                                               ; preds = %b5, %b2
  br i1 undef, label %b4, label %b5

b4:                                               ; preds = %b3
  br label %b5

b5:                                               ; preds = %b4, %b3
  br i1 undef, label %b3, label %b6

b6:                                               ; preds = %b5
  br i1 undef, label %b1, label %b2

b7:                                               ; preds = %b7, %b1
  %v0 = phi i32 [ 0, %b1 ], [ %v4, %b7 ]
  %v1 = load i16, ptr undef, align 8, !tbaa !0
  %v2 = icmp sgt i16 %v1, undef
  %v3 = select i1 %v2, i16 4, i16 undef
  store i16 %v3, ptr undef, align 2, !tbaa !0
  store i16 -32768, ptr undef, align 2, !tbaa !0
  %v4 = add i32 %v0, 1
  %v5 = icmp eq i32 %v4, 5
  br i1 %v5, label %b8, label %b7

b8:                                               ; preds = %b7
  br i1 undef, label %b9, label %b10

b9:                                               ; preds = %b8
  br label %b10

b10:                                              ; preds = %b9, %b8
  br i1 undef, label %b11, label %b12

b11:                                              ; preds = %b10
  unreachable

b12:                                              ; preds = %b10
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
