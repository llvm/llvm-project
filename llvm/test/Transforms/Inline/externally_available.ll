; RUN: opt < %s -inline -constprop -S | FileCheck %s

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
; CHECK: ret i32 5
}

; CHECK-NOT: @test_function
