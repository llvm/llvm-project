; RUN: llc -march=hexagon -enable-pipeliner -pipeliner-max-stages=2 \
; RUN:     -pipeliner-ignore-recmii -disable-hexagon-nv-schedule -stats -o /dev/null\
; RUN:     -enable-aa-sched-mi < %s 2>&1 | FileCheck %s --check-prefix=STATS
; REQUIRES: asserts
;
; Test that we generate the correct phis in the last epilog block when
; allowing multiple stages.
;
; STATS: 1 pipeliner        - Number of loops software pipelined

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br i1 undef, label %b6, label %b1

b1:                                               ; preds = %b0
  br i1 undef, label %b6, label %b2

b2:                                               ; preds = %b1
  br label %b4

b3:                                               ; preds = %b4, %b3
  %v0 = add nsw i32 0, 57344
  %v1 = trunc i32 %v0 to i16
  store i16 %v1, i16* null, align 2, !tbaa !0
  %v2 = getelementptr inbounds i8, i8* null, i32 undef
  %v3 = load i8, i8* %v2, align 1, !tbaa !4
  %v4 = zext i8 %v3 to i32
  %v5 = shl nuw nsw i32 %v4, 6
  %v6 = add nsw i32 %v5, 57344
  %v7 = trunc i32 %v6 to i16
  store i16 %v7, i16* undef, align 2, !tbaa !0
  br i1 undef, label %b5, label %b3

b4:                                               ; preds = %b5, %b2
  %v8 = phi i32 [ 0, %b2 ], [ %v9, %b5 ]
  br label %b3

b5:                                               ; preds = %b3
  %v9 = add i32 %v8, 1
  %v10 = icmp eq i32 %v9, undef
  br i1 %v10, label %b6, label %b4

b6:                                               ; preds = %b5, %b1, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!2, !2, i64 0}
