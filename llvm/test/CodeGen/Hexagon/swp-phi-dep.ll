; RUN: llc -march=hexagon -hexagon-initial-cfg-cleanup=0 -enable-pipeliner -pipeliner-max-stages=2 < %s -pipeliner-experimental-cg=true | FileCheck %s

; Check that the pipelined code uses the proper address in the
; prolog and the kernel. The bug occurs when the address computation
; computes the same value twice.

; CHECK: = addasl(r{{[0-9]+}},[[REG0:(r[0-9]+)]],#1)
; CHECK-NOT: = addasl(r{{[0-9]+}},[[REG0]],#1)

; Function Attrs: nounwind
define void @f0(i32 %a0, ptr nocapture %a1) #0 {
b0:
  br i1 undef, label %b2, label %b1

b1:                                               ; preds = %b0
  unreachable

b2:                                               ; preds = %b0
  br label %b3

b3:                                               ; preds = %b4, %b2
  br i1 undef, label %b4, label %b5

b4:                                               ; preds = %b3
  br label %b3

b5:                                               ; preds = %b3
  br i1 undef, label %b6, label %b7

b6:                                               ; preds = %b5
  unreachable

b7:                                               ; preds = %b5
  br i1 undef, label %b8, label %b12

b8:                                               ; preds = %b7
  br i1 undef, label %b9, label %b11

b9:                                               ; preds = %b9, %b8
  br i1 undef, label %b9, label %b10

b10:                                              ; preds = %b9
  br i1 undef, label %b12, label %b11

b11:                                              ; preds = %b11, %b10, %b8
  %v0 = phi i32 [ %v6, %b11 ], [ undef, %b8 ], [ undef, %b10 ]
  %v1 = phi i32 [ %v0, %b11 ], [ %a0, %b8 ], [ undef, %b10 ]
  %v2 = add nsw i32 %v1, -2
  %v3 = getelementptr inbounds i16, ptr %a1, i32 %v2
  %v4 = load i16, ptr %v3, align 2, !tbaa !0
  %v5 = getelementptr inbounds i16, ptr %a1, i32 %v0
  store i16 %v4, ptr %v5, align 2, !tbaa !0
  %v6 = add nsw i32 %v0, -1
  %v7 = icmp sgt i32 %v6, 0
  br i1 %v7, label %b11, label %b12

b12:                                              ; preds = %b11, %b10, %b7
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
