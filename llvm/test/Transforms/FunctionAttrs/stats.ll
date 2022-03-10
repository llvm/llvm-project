; RUN: opt -passes=function-attrs -stats -disable-output %s 2>&1 | FileCheck %s

; REQUIRES: asserts

@g = global i32 20

define i32 @test_only_read_arg(i32* %ptr) {
entry:
  %l = load i32, i32* %ptr
  ret i32 %l
}

define void @test_write_global() {
entry:
  store i32 0, i32* @g
  ret void
}

; CHECK:      1 function-attrs - Number of arguments marked nocapture
; CHECK-NEXT: 1 function-attrs - Number of functions marked as nofree
; CHECK-NEXT: 2 function-attrs - Number of functions marked as norecurse
; CHECK-NEXT: 2 function-attrs - Number of functions marked as nosync
; CHECK-NEXT: 2 function-attrs - Number of functions marked as nounwind
; CHECK-NEXT: 1 function-attrs - Number of functions marked readonly
; CHECK-NEXT: 1 function-attrs - Number of arguments marked readonly
; CHECK-NEXT: 2 function-attrs - Number of functions marked as willreturn
; CHECK-NEXT: 1 function-attrs - Number of functions marked writeonly
