; RUN: llc -mtriple=hexagon -mno-pairing -mno-compound -hexagon-initial-cfg-cleanup=0 < %s -pipeliner-experimental-cg=true | FileCheck %s
; XFAIL: *

; Test that we generate the correct phi names in the epilog when the pipeliner
; schedules a phi and it's loop definition in different stages, e.g., a phi is
; scheduled in stage 2, but the loop definition in scheduled in stage 0). The
; code in generateExistingPhis was generating the wrong name for the last
; epilog bock.

; CHECK: endloop0
; CHECK: sub([[REG:r([0-9]+)]],r{{[0-9]+}}):sat
; CHECK-NOT: sub([[REG]],r{{[0-9]+}}):sat

define void @f0() {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  br i1 undef, label %b2, label %b1

b2:                                               ; preds = %b1
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v0 = phi i32 [ %v8, %b3 ], [ 7, %b2 ]
  %v1 = phi i32 [ %v6, %b3 ], [ undef, %b2 ]
  %v2 = phi i32 [ %v1, %b3 ], [ undef, %b2 ]
  %v3 = getelementptr inbounds [9 x i32], ptr undef, i32 0, i32 %v0
  %v4 = add nsw i32 %v0, -2
  %v5 = getelementptr inbounds [9 x i32], ptr undef, i32 0, i32 %v4
  %v6 = load i32, ptr %v5, align 4
  %v7 = tail call i32 @llvm.hexagon.A2.subsat(i32 %v2, i32 %v6)
  store i32 %v7, ptr %v3, align 4
  %v8 = add i32 %v0, -1
  %v9 = icmp sgt i32 %v8, 1
  br i1 %v9, label %b3, label %b4

b4:                                               ; preds = %b3
  unreachable
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.subsat(i32, i32) #0

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" }
