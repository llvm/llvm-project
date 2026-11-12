; RUN: llc -mtriple=hexagon -enable-pipeliner < %s -pipeliner-experimental-cg=true | FileCheck %s

; A test that the Phi rewrite logic is correct.

; CHECK: [[REG0:(r[0-9]+)]] = #0
; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: memh([[REG0]]+#0) = #0

define void @f0(i32 %a0) #0 {
b0:
  %v0 = add i32 %a0, -4
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v1 = phi ptr [ %v4, %b1 ], [ null, %b0 ]
  %v2 = phi i32 [ %v5, %b1 ], [ 0, %b0 ]
  %v3 = getelementptr inbounds i16, ptr %v1, i32 1
  store i16 0, ptr %v1, align 2
  %v4 = getelementptr inbounds i16, ptr %v1, i32 2
  store i16 0, ptr %v3, align 2
  %v5 = add nsw i32 %v2, 8
  %v6 = icmp slt i32 %v5, %v0
  br i1 %v6, label %b1, label %b2

b2:                                               ; preds = %b1
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
