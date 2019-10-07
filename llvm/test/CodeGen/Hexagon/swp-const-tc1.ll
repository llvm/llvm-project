; RUN: llc -march=hexagon -enable-pipeliner -enable-pipeliner-opt-size \
; RUN:     -verify-machineinstrs -hexagon-initial-cfg-cleanup=0 \
; RUN:     -enable-aa-sched-mi=false -hexagon-expand-condsets=0 \
; RUN:     < %s -pipeliner-experimental-cg=true | FileCheck %s

; Disable expand-condsets because it will assert on undefined registers.

; Test that we change the CFG correctly for pipelined loops where the trip
; count is a compile-time constant, and the trip count is the same as the
; number of prolog blocks (i.e., stages).

; CHECK: memb(r{{[0-9]+}}+#0) =
; CHECK: memb(r{{[0-9]+}}+#0) =

; Function Attrs: nounwind optsize
define void @f0(i1 %x) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b5, %b0
  %v0 = load i16, i16* undef, align 2, !tbaa !0
  %v1 = sext i16 %v0 to i32
  %v2 = load i16, i16* undef, align 2, !tbaa !0
  %v3 = sext i16 %v2 to i32
  %v4 = and i32 %v1, 7
  %v5 = and i32 %v3, 7
  br label %b2

b2:                                               ; preds = %b4, %b1
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v6 = phi i32 [ 0, %b2 ], [ %v22, %b3 ]
  %v7 = add i32 %v6, undef
  %v8 = icmp slt i32 undef, %v7
  %v9 = add nsw i32 %v7, 1
  %v10 = select i1 %x, i32 1, i32 %v9
  %v11 = add i32 %v10, 0
  %v12 = getelementptr inbounds i8, i8* null, i32 %v11
  %v13 = load i8, i8* %v12, align 1, !tbaa !4
  %v14 = zext i8 %v13 to i32
  %v15 = mul i32 %v14, %v4
  %v16 = add i32 %v15, 0
  %v17 = mul i32 %v16, %v5
  %v18 = add i32 %v17, 32
  %v19 = add i32 %v18, 0
  %v20 = lshr i32 %v19, 6
  %v21 = trunc i32 %v20 to i8
  store i8 %v21, i8* undef, align 1, !tbaa !4
  %v22 = add i32 %v6, 1
  %v23 = icmp eq i32 %v22, 2
  br i1 %v23, label %b4, label %b3

b4:                                               ; preds = %b3
  br i1 undef, label %b5, label %b2

b5:                                               ; preds = %b4
  br i1 undef, label %b1, label %b6

b6:                                               ; preds = %b5
  ret void
}

attributes #0 = { nounwind optsize "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!2, !2, i64 0}
