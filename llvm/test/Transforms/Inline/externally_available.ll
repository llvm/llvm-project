; RUN: opt < %s -inline -constprop -S > %t
; RUN: not grep test_function %t
; RUN: grep "ret i32 5" %t


; test_function should not be emitted to the .s file.
define available_externally i32 @test_function() {
  ret i32 4
}


define i32 @result() {
  %A = call i32 @test_function()
  %B = add i32 %A, 1
  ret i32 %B
}
