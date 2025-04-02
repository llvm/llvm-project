; RUN: llc -mtriple=hexagon -hexagon-expand-condsets=0 -hexagon-initial-cfg-cleanup=0 < %s | FileCheck %s
;
; Expand-condsets eliminates the "mux" instruction, which is what this
; testcase is checking.

; Test that we don't generate a new value compare if the operands are
; the same register.

; CHECK-NOT: cmp.eq([[REG0:(r[0-9]+)]].new,[[REG0]])
; CHECK: cmp.eq([[REG1:(r[0-9]+)]],[[REG1]])

%s.0 = type { i16, i8, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, [2 x i32], ptr, ptr, ptr, %s.1, ptr, [8 x i8], i8 }
%s.1 = type { i32, i16, i16 }

@g0 = external global %s.0
@g1 = external unnamed_addr constant [23 x i8], align 8

; Function Attrs: nounwind
declare void @f0(ptr nocapture, ptr nocapture readonly, ...) #0

define void @f1() #1 {
b0:
  %v0 = load ptr, ptr undef, align 4
  %v1 = load i32, ptr undef, align 4
  br i1 undef, label %b4, label %b1

b1:                                               ; preds = %b0
  %v2 = icmp slt i32 %v1, 0
  %v3 = lshr i32 %v1, 5
  %v4 = add i32 %v3, -134217728
  %v5 = select i1 %v2, i32 %v4, i32 %v3
  %v6 = getelementptr inbounds i32, ptr %v0, i32 %v5
  %v7 = icmp ult ptr %v6, %v0
  %v8 = select i1 %v7, i32 0, i32 1
  br i1 undef, label %b2, label %b4

b2:                                               ; preds = %b1
  %v9 = icmp slt i32 %v1, 0
  %v10 = lshr i32 %v1, 5
  %v11 = add i32 %v10, -134217728
  %v12 = select i1 %v9, i32 %v11, i32 %v10
  %v13 = getelementptr inbounds i32, ptr %v0, i32 %v12
  %v14 = icmp ult ptr %v13, %v0
  %v15 = select i1 %v14, i32 0, i32 1
  %v16 = icmp eq i32 %v8, %v15
  br i1 %v16, label %b4, label %b3

b3:                                               ; preds = %b2
  call void (ptr, ptr, ...) @f0(ptr @g0, ptr @g1, i32 %v8, i32 %v15) #0
  unreachable

b4:                                               ; preds = %b2, %b1, %b0
  br i1 undef, label %b6, label %b5

b5:                                               ; preds = %b4
  unreachable

b6:                                               ; preds = %b4
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
attributes #1 = { "target-cpu"="hexagonv5" }
