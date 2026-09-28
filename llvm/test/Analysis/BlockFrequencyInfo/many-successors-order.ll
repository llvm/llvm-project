; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s

; Blocks with more than 128 successor edges combine duplicate edges via a map.
; Make sure that distributing the mass does not depend on the iteration order.

; CHECK-LABEL: block-frequency-info: f
; CHECK-NEXT:  - entry: float = 1.0, int = [[ENTRY:[0-9]+]]
; CHECK-NEXT:  - default: float = 0.0077519, int = {{[0-9]+}}
; CHECK-NEXT:  - b0: float = 0.49612, int = 8937376004318074
; CHECK-NEXT:  - b1: float = 0.49612, int = 8937376004964352
; CHECK-NEXT:  - exit: float = 1.0, int = [[ENTRY]]
define void @f(i32 %x) {
entry:
  switch i32 %x, label %default [
    i32 0, label %b0
    i32 1, label %b1
    i32 2, label %b0
    i32 3, label %b1
    i32 4, label %b0
    i32 5, label %b1
    i32 6, label %b0
    i32 7, label %b1
    i32 8, label %b0
    i32 9, label %b1
    i32 10, label %b0
    i32 11, label %b1
    i32 12, label %b0
    i32 13, label %b1
    i32 14, label %b0
    i32 15, label %b1
    i32 16, label %b0
    i32 17, label %b1
    i32 18, label %b0
    i32 19, label %b1
    i32 20, label %b0
    i32 21, label %b1
    i32 22, label %b0
    i32 23, label %b1
    i32 24, label %b0
    i32 25, label %b1
    i32 26, label %b0
    i32 27, label %b1
    i32 28, label %b0
    i32 29, label %b1
    i32 30, label %b0
    i32 31, label %b1
    i32 32, label %b0
    i32 33, label %b1
    i32 34, label %b0
    i32 35, label %b1
    i32 36, label %b0
    i32 37, label %b1
    i32 38, label %b0
    i32 39, label %b1
    i32 40, label %b0
    i32 41, label %b1
    i32 42, label %b0
    i32 43, label %b1
    i32 44, label %b0
    i32 45, label %b1
    i32 46, label %b0
    i32 47, label %b1
    i32 48, label %b0
    i32 49, label %b1
    i32 50, label %b0
    i32 51, label %b1
    i32 52, label %b0
    i32 53, label %b1
    i32 54, label %b0
    i32 55, label %b1
    i32 56, label %b0
    i32 57, label %b1
    i32 58, label %b0
    i32 59, label %b1
    i32 60, label %b0
    i32 61, label %b1
    i32 62, label %b0
    i32 63, label %b1
    i32 64, label %b0
    i32 65, label %b1
    i32 66, label %b0
    i32 67, label %b1
    i32 68, label %b0
    i32 69, label %b1
    i32 70, label %b0
    i32 71, label %b1
    i32 72, label %b0
    i32 73, label %b1
    i32 74, label %b0
    i32 75, label %b1
    i32 76, label %b0
    i32 77, label %b1
    i32 78, label %b0
    i32 79, label %b1
    i32 80, label %b0
    i32 81, label %b1
    i32 82, label %b0
    i32 83, label %b1
    i32 84, label %b0
    i32 85, label %b1
    i32 86, label %b0
    i32 87, label %b1
    i32 88, label %b0
    i32 89, label %b1
    i32 90, label %b0
    i32 91, label %b1
    i32 92, label %b0
    i32 93, label %b1
    i32 94, label %b0
    i32 95, label %b1
    i32 96, label %b0
    i32 97, label %b1
    i32 98, label %b0
    i32 99, label %b1
    i32 100, label %b0
    i32 101, label %b1
    i32 102, label %b0
    i32 103, label %b1
    i32 104, label %b0
    i32 105, label %b1
    i32 106, label %b0
    i32 107, label %b1
    i32 108, label %b0
    i32 109, label %b1
    i32 110, label %b0
    i32 111, label %b1
    i32 112, label %b0
    i32 113, label %b1
    i32 114, label %b0
    i32 115, label %b1
    i32 116, label %b0
    i32 117, label %b1
    i32 118, label %b0
    i32 119, label %b1
    i32 120, label %b0
    i32 121, label %b1
    i32 122, label %b0
    i32 123, label %b1
    i32 124, label %b0
    i32 125, label %b1
    i32 126, label %b0
    i32 127, label %b1
  ]

default:
  br label %exit

b0:
  br label %exit

b1:
  br label %exit

exit:
  ret void
}
