; RUN: llc -mtriple=hexagon -hexagon-small-data-threshold=0 < %s
; REQUIRES: asserts

; Check for successful compilation.

; Function Attrs: nounwind
declare void @f0(i32) #0

; Function Attrs: nounwind
define i32 @f1(i32 %a0) #0 {
b0:
  switch i32 %a0, label %b1 [
    i32 1, label %b2
    i32 2, label %b3
    i32 3, label %b4
    i32 4, label %b5
    i32 5, label %b6
  ]

b1:                                               ; preds = %b0
  ret i32 0

b2:                                               ; preds = %b0
  call void @f0(i32 4)
  ret i32 4

b3:                                               ; preds = %b0
  call void @f0(i32 2)
  call void @f0(i32 42)
  ret i32 42

b4:                                               ; preds = %b0
  call void @f0(i32 -1)
  ret i32 -1

b5:                                               ; preds = %b0
  call void @f0(i32 123)
  ret i32 123

b6:                                               ; preds = %b0
  call void @f0(i32 88)
  ret i32 4
}

attributes #0 = { nounwind }
