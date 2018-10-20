; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that we generate new value jump.

; CHECK: if (cmp.eq(r{{[0-9]+}}.new,#0)) jump{{.}}

@g0 = global i32 0, align 4
@g1 = global i32 10, align 4

define i32 @f0(i32 %a0) #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  %v2 = load i32, i32* @g0, align 4
  store i32 %v2, i32* %v0, align 4
  call void @f2(i32 1, i32 2)
  %v3 = load i32, i32* @g1, align 4
  %v4 = icmp ne i32 %v3, 0
  br i1 %v4, label %b1, label %b2

b1:                                               ; preds = %b0
  call void @f3(i32 1, i32 2)
  br label %b3

b2:                                               ; preds = %b0
  call void @f1(i32 10, i32 20)
  br label %b3

b3:                                               ; preds = %b2, %b1
  ret i32 0
}

declare void @f1(i32, i32) #0
declare void @f2(i32, i32) #0
declare void @f3(i32, i32) #0

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
