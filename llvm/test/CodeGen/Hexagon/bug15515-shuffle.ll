; RUN: opt -march=hexagon -O2 -S < %s
; REQUIRES: asserts
;
; -fvectorize-loops infinite compile/memory
; test checks that the compile completes successfully

target triple = "hexagon"

@g0 = global i8 -1, align 1
@g1 = common global [15 x i8] zeroinitializer, align 8
@g2 = common global [15 x ptr] zeroinitializer, align 8

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  %v0 = alloca i32, align 4
  store i32 0, ptr %v0, align 4
  store i32 0, ptr %v0, align 4
  br label %b1

b1:                                               ; preds = %b3, %b0
  %v1 = load i32, ptr %v0, align 4
  %v2 = icmp slt i32 %v1, 15
  br i1 %v2, label %b2, label %b4

b2:                                               ; preds = %b1
  %v3 = load i32, ptr %v0, align 4
  %v4 = getelementptr inbounds [15 x i8], ptr @g1, i32 0, i32 %v3
  store i8 0, ptr %v4, align 1
  %v5 = load i32, ptr %v0, align 4
  %v6 = getelementptr inbounds [15 x ptr], ptr @g2, i32 0, i32 %v5
  store ptr @g0, ptr %v6, align 4
  br label %b3

b3:                                               ; preds = %b2
  %v7 = load i32, ptr %v0, align 4
  %v8 = add nsw i32 %v7, 1
  store i32 %v8, ptr %v0, align 4
  br label %b1

b4:                                               ; preds = %b1
  ret void
}

attributes #0 = { nounwind }
