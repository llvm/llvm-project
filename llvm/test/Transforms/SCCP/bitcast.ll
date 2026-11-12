; RUN: opt < %s -passes=ipsccp -S | FileCheck %s

define i128 @vector_to_int_cast() {
  %A = bitcast <4 x i32> <i32 1073741824, i32 1073741824, i32 1073741824, i32 1073741824> to i128
  ret i128 %A
}

; CHECK: define i128 @vector_to_int_cast(
; CHECK-NEXT:  ret i128 85070591750041656499021422275829170176
