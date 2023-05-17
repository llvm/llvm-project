; RUN: llc -march=hexagon -O3 -verify-machineinstrs -disable-hexagon-peephole < %s
; REQUIRES: asserts

; This test checks if tied operands are consistent.
target triple = "hexagon-unknown--elf"

; Function Attrs: nounwind
define void @f0(ptr nocapture %a0) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b5, %b0
  %v0 = phi ptr [ %a0, %b0 ], [ %v5, %b5 ]
  %v1 = phi i16 [ undef, %b0 ], [ %v10, %b5 ]
  br i1 undef, label %b2, label %b3

b2:                                               ; preds = %b1
  %v2 = getelementptr inbounds i16, ptr %v0, i32 1
  %v3 = load i16, ptr %v0, align 2, !tbaa !0
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v4 = phi i16 [ %v3, %b2 ], [ %v1, %b1 ]
  %v5 = phi ptr [ %v2, %b2 ], [ %v0, %b1 ]
  %v6 = lshr i16 %v4, 4
  %v7 = zext i16 %v6 to i32
  %v8 = and i32 %v7, 15
  %v9 = icmp ult i32 %v8, 9
  br i1 %v9, label %b4, label %b5

b4:                                               ; preds = %b3
  call void @llvm.trap()
  unreachable

b5:                                               ; preds = %b3
  %v10 = lshr i16 %v4, 8
  br label %b1
}

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { noreturn nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
