; RUN: opt < %s -inline -S | FileCheck %s

define available_externally i32 @test_function() {
; CHECK-NOT: @test_function
entry:
  ret i32 4
}


define i32 @result() {
; CHECK-LABEL: define i32 @result()
entry:
  %A = call i32 @test_function()
; CHECK-NOT: call
; CHECK-NOT: @test_function

  %B = add i32 %A, 1
  ret i32 %B
; CHECK: add i32
; CHECK-NEXT: ret i32
}

; CHECK-NOT: @test_function
