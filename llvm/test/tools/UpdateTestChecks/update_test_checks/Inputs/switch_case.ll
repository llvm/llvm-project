; RUN: opt < %s -S | FileCheck %s

; Test whether the UTC format the switch-cases correctly, which requires TWO extra spaces.

define i8 @testi8(i8 %x) {
  switch i8 %x, label %default [
    i8 0, label %case1
    i8 1, label %case2
    i8 2, label %case3
    i8 3, label %case3
  ]
default:
  ret i8 0
case1:
  ret i8 1
case2:
  ret i8 2
case3:
  ret i8 3
}

define i32 @testi32(i32 %x) {
  switch i32 %x, label %default [
    i32 0, label %case1
    i32 1, label %case2
    i32 2, label %case3
    i32 3, label %case3
  ]
default:
  ret i32 0
case1:
  ret i32 1
case2:
  ret i32 2
case3:
  ret i32 3
}

define i128 @testi128(i128 %x) {
  switch i128 %x, label %default [
    i128 0, label %case1
    i128 1, label %case2
    i128 2, label %case3
    i128 3, label %case3
  ]
default:
  ret i128 0
case1:
  ret i128 1
case2:
  ret i128 2
case3:
  ret i128 3
}
